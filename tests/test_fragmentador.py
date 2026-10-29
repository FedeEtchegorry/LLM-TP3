from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.modulo_a.fragmentador import (
    EstrategiaDeChunking,
    Fragmentador,
    informe_de_secciones,
)
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración

pytestmark = pytest.mark.nivel1

@pytest.fixture
def cfg_caracteres() -> Configuración:
    return Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.tamano_objetivo": "120",
            "fragmentacion.solapamiento": "0",
        }
    )

def _registro(aid="cma:1", **campos) -> RegistroCanónico:
    textos = tuple(
        CampoDeTexto(section, texto, f"https://ficha/{aid}", "CC0")
        for section, texto in campos.items()
        if texto is not None
    )
    return RegistroCanónico(
        artwork_id=aid,
        museum=aid.split(":")[0],
        type="Painting",
        title="t",
        artist="a",
        date_display="1800",
        year=1800,
        medium="oil",
        style=None,
        culture_raw=("France",),
        source_url=f"https://ficha/{aid}",
        imagen=Imagen("https://x.jpg", "sha256:ab"),
        media_license="CC0",
        campos_de_texto=textos,
        procedencia={},
    )

def test_un_campo_corto_es_un_solo_fragmento_y_el_splitter_no_entra(cfg_caracteres):
    frag = Fragmentador(cfg_caracteres)
    fragmentos = frag.fragmentar(_registro(description="Corot painted this view."))
    assert len(fragmentos) == 1
    assert fragmentos[0].texto == "Corot painted this view."
    assert frag.conteos_por_seccion()["description"]["n_campos_subdivididos"] == 0

def test_cada_campo_lleva_su_section_y_su_chunk_index_propio(cfg_caracteres):
    frag = Fragmentador(cfg_caracteres)
    fragmentos = frag.fragmentar(
        _registro(description="d" * 300, provenance="Sold in 1923.")
    )
    ids = [f.fragment_id for f in fragmentos]
    assert ids[-1] == "cma:1::provenance::0"
    assert [f.chunk_index for f in fragmentos if f.section == "description"] == [0, 1, 2]

def test_un_campo_mas_largo_que_el_objetivo_se_subdivide_y_se_cuenta(cfg_caracteres):
    frag = Fragmentador(cfg_caracteres)
    fragmentos = frag.fragmentar(_registro(description="palabra " * 60))
    assert len(fragmentos) > 1
    conteo = frag.conteos_por_seccion()["description"]
    assert conteo["n_campos_subdivididos"] == 1
    assert conteo["n_fragmentos"] == len(fragmentos)

def test_todos_los_trozos_son_subcadenas_literales_del_campo(cfg_caracteres):
    texto = "Primera oración. Segunda oración.\n\nOtro párrafo con más texto.\n" * 6
    fragmentos = Fragmentador(cfg_caracteres).fragmentar(_registro(description=texto))
    assert len(fragmentos) > 1
    for f in fragmentos:
        assert f.texto in texto

def test_ningun_trozo_supera_el_objetivo(cfg_caracteres):
    frag = Fragmentador(cfg_caracteres)
    fragmentos = frag.fragmentar(_registro(description="palabra " * 200))
    assert all(frag.medidor(f.texto) <= 120 for f in fragmentos)

def test_un_campo_sin_separadores_se_corta_igual(cfg_caracteres):
    fragmentos = Fragmentador(cfg_caracteres).fragmentar(_registro(description="x" * 400))
    assert len(fragmentos) >= 4

def test_una_obra_sin_los_campos_configurados_sale_con_cero_fragmentos(cfg_caracteres):
    fragmentos = Fragmentador(cfg_caracteres).fragmentar(
        _registro(exhibitions="una lista de exposiciones")
    )
    assert fragmentos == []

def test_solo_entran_los_campos_configurados(cfg_caracteres):
    frag = Fragmentador(cfg_caracteres)
    fragmentos = frag.fragmentar(
        _registro(description="d", exhibitions="e", citations="c", provenance="p")
    )
    assert sorted({f.section for f in fragmentos}) == ["description", "provenance"]

def test_agregar_una_candidata_de_p2_es_configuracion_y_no_codigo():
    cfg = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, citations]",
        }
    )
    fragmentos = Fragmentador(cfg).fragmentar(_registro(description="d", citations="c"))
    assert sorted({f.section for f in fragmentos}) == ["citations", "description"]

def test_la_estrategia_declara_el_tamano_objetivo_que_ningun_documento_tenia():
    estrategia = EstrategiaDeChunking.desde_configuración(Configuración.cargar())
    assert estrategia.tamano_objetivo == 500
    assert estrategia.unidad == "tokens"
    assert "tamano_objetivo" in estrategia.a_json()

def test_el_tamano_objetivo_cambia_el_variante_id():
    from tp3_rag.nucleo.identidad import variante_id

    def con(tamano: int) -> str:
        cfg = Configuración.cargar(
            overrides={"fragmentacion.tamano_objetivo": str(tamano)}
        )
        return variante_id(
            catalogo_id="c",
            split_id="s",
            criterio_de_inclusion={"description_no_vacia": True},
            checkpoint_de_clip="openai/clip-vit-base-patch32",
            version_del_preprocesado="1",
            estrategia_de_chunking=EstrategiaDeChunking.desde_configuración(cfg).a_json(),
            campos_de_knowledge=["description"],
            modelo_de_embedding_de_texto="text-embedding-3-small",
            criterio_de_culture="crudo",
        )

    assert con(500) != con(2000)

def test_una_estrategia_desconocida_falla_y_nombra_el_brazo_ciego():
    with pytest.raises(ErrorDeConfiguración, match="ciego"):
        Configuración.cargar(overrides={"fragmentacion.estrategia": "ciego"})

def test_un_solapamiento_mayor_que_el_objetivo_falla_antes_de_correr():
    with pytest.raises(ErrorDeConfiguración, match="solapamiento"):
        Configuración.cargar(
            overrides={
                "fragmentacion.tamano_objetivo": "100",
                "fragmentacion.solapamiento": "100",
            }
        )

def test_una_jerarquia_que_no_termina_en_vacio_falla():
    with pytest.raises(ErrorDeConfiguración, match="último recurso"):
        Configuración.cargar(overrides={"fragmentacion.separadores": "[a, b]"})

def test_una_unidad_desconocida_falla():
    with pytest.raises(ErrorDeConfiguración, match="unidad"):
        Configuración.cargar(overrides={"fragmentacion.unidad": "palabras"})

def test_una_section_que_ningun_museo_produce_falla_antes_de_correr():
    with pytest.raises(ErrorDeConfiguración, match="ningún museo configurado produce"):
        Configuración.cargar(
            overrides={"fragmentacion.campos_de_knowledge": "[description, descripcion]"}
        )

def test_campos_de_knowledge_vacio_falla_nombrando_la_registracion():
    with pytest.raises(ErrorDeConfiguración, match="rechazaría la variante"):
        Fragmentador(
            Configuración.cargar(
                overrides={
                    "fragmentacion.unidad": "caracteres",
                    "fragmentacion.campos_de_knowledge": "[]",
                }
            )
        )

def test_el_informe_mide_las_secciones_que_la_variante_no_indexa(cfg_caracteres):
    registros = [
        _registro("aic:1", description="d", citations="c" * 50),
        _registro("cma:2", description="d", exhibitions="e", provenance="p"),
    ]
    informe = informe_de_secciones(cfg_caracteres, registros, medidor=len)
    assert sorted(informe["candidatas_de_p2"]) == ["citations", "exhibitions"]
    assert informe["total"]["citations"]["n_obras_con_el_campo"] == 1
    assert informe["por_museo"]["aic"]["secciones"]["citations"]["n_fragmentos"] == 1
    assert informe["secciones_configuradas"] == [
        "description",
        "did_you_know",
        "provenance",
    ]
    assert "citations" not in informe["secciones_configuradas"]
