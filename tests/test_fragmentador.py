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

def test_una_estrategia_desconocida_falla_y_nombra_los_dos_brazos():
    with pytest.raises(ErrorDeConfiguración, match="por_campo_semantico"):
        Configuración.cargar(overrides={"fragmentacion.estrategia": "a_ojo"})

def test_el_brazo_ciego_existe_y_la_configuracion_lo_acepta():
    cfg = Configuración.cargar(overrides={"fragmentacion.estrategia": "ciego"})
    estrategia = EstrategiaDeChunking.desde_configuración(cfg)
    assert estrategia.es_ciega
    assert estrategia.orden_de_concatenacion == (
        "description",
        "did_you_know",
        "provenance",
    )

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

def test_campos_de_knowledge_vacio_falla_nombrando_el_invariante_3():
    with pytest.raises(ErrorDeConfiguración, match="invariante 3"):
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

@pytest.fixture
def cfg_ciego() -> Configuración:
    return Configuración.cargar(
        overrides={
            "fragmentacion.estrategia": "ciego",
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.tamano_objetivo": "120",
            "fragmentacion.solapamiento": "0",
        }
    )

def _registro_mixto(aid="aic:1"):
    campos = (
        CampoDeTexto("description", "Una vista del estanque.", f"https://ficha/{aid}", "CC-BY-4.0"),
        CampoDeTexto("provenance", "Vendido en 1923 a X.", f"https://ficha/{aid}", "CC0"),
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
        campos_de_texto=campos,
        procedencia={},
    )

def test_ciego_concatena_y_FUSIONA_en_vez_de_subdividir(cfg_ciego):
    registro = _registro(description="corto", provenance="también corto")
    ciego = Fragmentador(cfg_ciego).fragmentar(registro)
    por_campo = Fragmentador(
        Configuración.cargar(
            overrides={
                "fragmentacion.unidad": "caracteres",
                "fragmentacion.tamano_objetivo": "120",
                "fragmentacion.solapamiento": "0",
            }
        )
    ).fragmentar(registro)
    assert len(ciego) == 1
    assert len(por_campo) == 2
    assert ciego[0].campos_abarcados == ("description", "provenance")

def test_ciego_fr2_el_trozo_es_subcadena_de_la_concatenacion_declarada(cfg_ciego):
    registro = _registro(description="a" * 200, provenance="b" * 200)
    frag = Fragmentador(cfg_ciego)
    concatenacion = "\n\n".join(("a" * 200, "b" * 200))
    fragmentos = frag.fragmentar(registro)
    assert len(fragmentos) > 1
    for f in fragmentos:
        assert f.texto in concatenacion

def test_ciego_un_chunk_que_cruza_el_borde_declara_los_dos_campos_y_las_dos_licencias(cfg_ciego):
    fragmentos = Fragmentador(cfg_ciego).fragmentar(_registro_mixto())
    assert len(fragmentos) == 1
    f = fragmentos[0]
    assert f.campos_abarcados == ("description", "provenance")
    assert f.licencias_abarcadas == ("CC-BY-4.0", "CC0")
    assert f.mezcla_licencias
    assert f.license == "CC-BY-4.0"
    assert f.section == "description"

def test_ciego_el_conteo_de_mezcla_lo_emite_el_pipeline(cfg_ciego):
    frag = Fragmentador(cfg_ciego)
    frag.fragmentar_muchos([_registro_mixto("aic:1"), _registro("cma:2", description="d")])
    informe = frag.conteos_de_concatenacion()
    assert informe["n_obras"] == 2
    assert informe["n_fragmentos_que_mezclan_licencias"] == 1
    assert informe["combinaciones_de_licencias"] == {"CC-BY-4.0 + CC0": 1, "CC0": 1}
    ejemplo = informe["ejemplo_de_mezcla"]
    assert ejemplo["artwork_id"] == "aic:1"
    assert [p["license"] for p in ejemplo["partes"]] == ["CC-BY-4.0", "CC0"]

def test_el_brazo_por_campo_no_puede_mezclar_y_el_informe_esta_AUSENTE():
    cfg = Configuración.cargar(overrides={"fragmentacion.unidad": "caracteres"})
    frag = Fragmentador(cfg)
    fragmentos = frag.fragmentar(_registro_mixto())
    assert frag.conteos_de_concatenacion() == {}
    assert not any(f.mezcla_licencias for f in fragmentos)

def test_ciego_la_concatenacion_sigue_el_orden_declarado_y_no_el_alfabetico(cfg_ciego):
    from tp3_rag.nucleo.identidad import variante_id

    def id_con(campos: str) -> tuple[str, str]:
        cfg = Configuración.cargar(
            overrides={
                "fragmentacion.estrategia": "ciego",
                "fragmentacion.unidad": "caracteres",
                "fragmentacion.campos_de_knowledge": campos,
            }
        )
        frag = Fragmentador(cfg, medidor=len)
        texto = frag.fragmentar(_registro("cma:1", description="DD", provenance="PP"))[0].texto
        vid = variante_id(
            catalogo_id="c",
            split_id="s",
            criterio_de_inclusion={"description_no_vacia": True},
            checkpoint_de_clip="k",
            version_del_preprocesado="1",
            estrategia_de_chunking=frag.estrategia.a_json(),
            campos_de_knowledge=frag.campos_de_knowledge,
            modelo_de_embedding_de_texto="m",
            criterio_de_culture="crudo",
        )
        return texto, vid

    directo, id_directo = id_con("[description, provenance]")
    invertido, id_invertido = id_con("[provenance, description]")
    assert directo == "DD\n\nPP" and invertido == "PP\n\nDD"
    assert id_directo != id_invertido

def test_ciego_una_obra_sin_ningun_campo_declarado_sale_con_cero(cfg_ciego):
    assert Fragmentador(cfg_ciego).fragmentar(_registro("cma:9", exhibitions="e")) == []

def test_ciego_ningun_trozo_supera_el_objetivo(cfg_ciego):
    registro = _registro(description="x " * 300, provenance="y " * 300)
    for f in Fragmentador(cfg_ciego).fragmentar(registro):
        assert len(f.texto) <= 120

def test_el_informe_de_p2_tambien_corre_en_el_brazo_ciego(cfg_ciego):
    registros = [
        _registro("aic:1", description="d", citations="c" * 50),
        _registro("cma:2", description="d", exhibitions="e", provenance="p"),
    ]
    informe = informe_de_secciones(cfg_ciego, registros, medidor=len)
    assert sorted(informe["candidatas_de_p2"]) == ["citations", "exhibitions"]
    assert informe["total"]["citations"]["n_obras_con_el_campo"] == 1
