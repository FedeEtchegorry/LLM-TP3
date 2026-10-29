from __future__ import annotations

import json
from pathlib import Path

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.dominio.consultas import CATEGORÍAS, SetDeConsultas
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.modulo_c.set_de_consultas import (
    AutorDelSetDeConsultas,
    Declaración,
    verificar_hash_contra_el_catalogo,
)
from tp3_rag.modulo_c.vocabulario_de_culture import VocabularioDeCulture
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import (
    CulturaPedidaFueraDelCorpus,
    ErrorDeArtefacto,
    ErrorDeConfiguración,
    ErrorDeConstrucción,
)
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

pytestmark = pytest.mark.nivel1

BYTES_DE_IMAGEN = b"\xff\xd8\xff\xe0JFIF-de-prueba"

def _registro(aid: str, *, culture=("France",), con_imagen=True):
    return RegistroCanónico(
        artwork_id=aid,
        museum=aid.split(":")[0],
        type="Painting",
        title=f"obra {aid}",
        artist="alguien",
        date_display="c. 1890",
        year=1890,
        medium="Oil on canvas",
        style=None,
        culture_raw=tuple(culture) if culture else None,
        source_url=f"https://x/{aid}",
        imagen=Imagen(f"https://x/{aid}.jpg", "sha256:ab") if con_imagen else None,
        media_license="CC0",
        campos_de_texto=(CampoDeTexto("description", "prosa", f"https://x/{aid}", "CC0"),),
        procedencia={},
    )

@pytest.fixture
def vocabulario():
    registros = [
        _registro("cma:1", culture=("Mughal India",)),
        _registro("cma:2", culture=("Mughal India",)),
        _registro("cma:3", culture=("Eastern India",)),
        _registro("cma:4", culture=("India", "Mughal India")),
        _registro("aic:5", culture=("France",)),
        _registro("aic:6", culture=None),
    ]
    return VocabularioDeCulture(registros, criterio="crudo", poblacion="corpus de prueba")

def test_el_n_es_la_union_de_obras_y_no_la_suma_de_las_etiquetas(vocabulario):
    unión = vocabulario.union(["Mughal India", "Eastern India", "India"])
    assert unión.n_por_etiqueta == {"Mughal India": 3, "Eastern India": 1, "India": 1}
    assert unión.suma_de_etiquetas == 5
    assert unión.n == 4
    assert unión.solapamiento == 1

def test_una_etiqueta_sola_publica_un_n_que_deja_afuera_a_sus_sinonimos(vocabulario):
    assert vocabulario.union(["India"]).n == 1
    assert vocabulario.union(["India", "Mughal India", "Eastern India"]).n == 4

def test_una_etiqueta_que_no_existe_es_error_duro_y_no_un_n_igual_a_cero(vocabulario):
    with pytest.raises(CulturaPedidaFueraDelCorpus) as e:
        vocabulario.union(["Japón"])
    assert "n = 0" in str(e.value)

def test_el_error_sugiere_la_etiqueta_real(vocabulario):
    assert "France" in vocabulario.sugerencias("Francia")
    assert "Mughal India" in vocabulario.sugerencias("India del Mughal")

def test_la_poblacion_viaja_con_el_numero(vocabulario):
    unión = vocabulario.union(["France"])
    assert unión.poblacion == "corpus de prueba"
    assert unión.n_poblacion == 6

def test_las_obras_sin_culture_se_cuentan(vocabulario):
    assert vocabulario.n_sin_culture == 1
    assert vocabulario.distribucion()["n_obras_sin_culture"] == 1

def test_los_grupos_propuestos_dicen_que_etiquetas_no_estan_en_este_corpus(vocabulario):
    india = next(g for g in vocabulario.grupos_propuestos() if g["clave"] == "india")
    assert india["n"] == 4
    assert "Northern India" in india["ausentes"]

def test_la_identidad_lleva_los_dos_ingredientes_en_el_nombre():
    ident = identidad.identidad_del_set_de_consultas(
        split_id="5b6cba32126fed61", autoria="revision-1"
    )
    assert ident == "5b6cba32126fed61-revision-1"
    assert identidad.partir_identidad_del_set(ident) == ("5b6cba32126fed61", "revision-1")

def test_la_autoria_no_puede_tener_forma_de_hash():
    with pytest.raises(ErrorDeConfiguración, match="hash_corto"):
        identidad.etiqueta_declarada("a1b2c3d4e5f60718", campo="autoría")

def test_el_criterio_de_culture_no_entra_en_la_identidad():
    import inspect

    parámetros = inspect.signature(identidad.identidad_del_set_de_consultas).parameters
    assert "criterio_de_culture" not in parámetros
    assert "variante_id" not in parámetros

@pytest.fixture
def mundo(cfg, tmp_path):
    almacén = AlmacenamientoEnArchivos(tmp_path)
    registros = [
        _registro("cma:1", culture=("Japan",)),
        _registro("cma:2", culture=("Japan",)),
        _registro("aic:9", culture=("France",)),
    ]
    for registro in registros:
        destino = almacén.ruta(
            Path(cfg["rutas.imagenes"]) / f"{registro.artwork_id.replace(':', '_')}.jpg"
        )
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(BYTES_DE_IMAGEN + registro.artwork_id.encode())

    class _Catálogo:
        catalogo_id = "cat1"

    catálogo = _Catálogo()
    catálogo.registros = tuple(
        RegistroCanónico(
            **{
                **r.__dict__,
                "imagen": Imagen(
                    r.imagen.image_url,
                    identidad.hash_de_bytes(BYTES_DE_IMAGEN + r.artwork_id.encode()),
                ),
            }
        )
        for r in registros
    )
    manifiesto = ManifiestoDeSplit(
        {
            "split_id": "5b6cba32126fed61",
            "corpus_indexado": {"ids": ["cma:1", "cma:2"]},
            "held_out": {"ids": ["aic:9"]},
            "conteos": {
                "elegibles_con_texto_e_imagen": 3,
                "corpus_indexado": 2,
                "held_out": 1,
            },
        }
    )
    return almacén, catálogo, manifiesto

def _declaración(tmp_path, *, autoria="revision-1", **cambios) -> Declaración:
    base = {
        "split_id": "5b6cba32126fed61",
        "autoria": autoria,
        "estado": "plantilla",
        "criterio_de_culture": "crudo",
        "revision": {"n_imagenes_revisadas": 3},
        "imagenes": [
            {
                "artwork_id": "aic:9",
                "atributos": ["paisaje", "paleta_fria"],
                "es_ejemplo": True,
                "consultas": [
                    {
                        "consulta_id": "c01-visual",
                        "categoria": "visual_cromatico",
                        "pedido": "versiones más cálidas",
                        "es_ejemplo": True,
                    },
                    {
                        "consulta_id": "c01-cultural",
                        "categoria": "conceptual_cultural",
                        "pedido": "el mismo tema pero en la cultura japonesa",
                        "es_ejemplo": True,
                        "cultura_pedida": {"nombre": "Japón", "etiquetas": ["Japan"]},
                    },
                    {
                        "consulta_id": "c01-ambiguo",
                        "categoria": "ambiguo",
                        "pedido": "con más vida",
                        "es_ejemplo": True,
                    },
                ],
            }
        ],
    }
    base.update(cambios)
    return Declaración(base, ruta=tmp_path / "declaracion.toml")

def test_autorar_congela_el_set_y_versiona_los_bytes(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    autor = AutorDelSetDeConsultas(cfg, almacén=almacén)
    resultado = autor.autorar(
        _declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto
    )

    assert not resultado.ya_estaba_congelado
    assert resultado.set_de_consultas.identidad == "5b6cba32126fed61-revision-1"
    versionada = resultado.rutas_de_imagenes[0]
    assert versionada.exists()
    assert versionada.read_bytes() == BYTES_DE_IMAGEN + b"aic:9"
    assert resultado.set_de_consultas.imagenes()[0].imagen_hash.startswith("sha256:")

def test_el_n_de_la_cultura_pedida_lo_computa_el_autor(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    resultado = AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
        _declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto
    )
    cultural = next(
        c for c in resultado.set_de_consultas.consultas() if c.cultura_pedida
    )
    assert cultural.cultura_pedida.n == 2
    assert cultural.cultura_pedida.n_poblacion == 2
    assert "corpus indexado del split" in cultural.cultura_pedida.poblacion

def test_la_poblacion_es_el_corpus_y_no_el_universo(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    vocabulario = VocabularioDeCulture.del_corpus_del_split(
        catálogo, manifiesto, criterio="crudo"
    )
    assert vocabulario.n_poblacion == 2 == len(manifiesto.ids_corpus)
    assert not vocabulario.existe("France")

def test_autorar_dos_veces_lee_y_no_reescribe(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    autor = AutorDelSetDeConsultas(cfg, almacén=almacén)
    primero = autor.autorar(_declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto)
    antes = primero.ruta.read_bytes()

    segundo = autor.autorar(_declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto)
    assert segundo.ya_estaba_congelado
    assert primero.ruta.read_bytes() == antes

def test_autorar_otro_set_del_mismo_split_exige_decirlo(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    autor = AutorDelSetDeConsultas(cfg, almacén=almacén)
    autor.autorar(_declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto)

    otra = _declaración(tmp_path, autoria="revision-2")
    with pytest.raises(ErrorDeConstrucción, match="derivar-otro"):
        autor.autorar(otra, catálogo=catálogo, manifiesto=manifiesto)

    autor.autorar(otra, catálogo=catálogo, manifiesto=manifiesto, derivar_otro=True)
    assert len(SetDeConsultas.del_split(almacén, autor.raíz, "5b6cba32126fed61")) == 2

def test_una_imagen_que_no_es_del_held_out_se_rechaza(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    declaración = _declaración(tmp_path)
    declaración.contenido["imagenes"][0]["artwork_id"] = "cma:1"
    with pytest.raises(ErrorDeConstrucción, match="held-out"):
        AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
            declaración, catálogo=catálogo, manifiesto=manifiesto
        )

def test_una_declaracion_que_ancla_a_otro_split_se_rechaza(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    declaración = _declaración(tmp_path)
    declaración.contenido["split_id"] = "0000000000000000"
    with pytest.raises(ErrorDeConstrucción, match="ancla"):
        AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
            declaración, catálogo=catálogo, manifiesto=manifiesto
        )

def test_el_n_no_se_declara_a_mano(tmp_path):
    declaración = _declaración(tmp_path)
    declaración.contenido["imagenes"][0]["consultas"][1]["cultura_pedida"]["n"] = 999
    with pytest.raises(ErrorDeConfiguración, match="no se declara"):
        declaración.validar_forma()

def test_un_consulta_id_faltante_no_tiene_default(tmp_path):
    declaración = _declaración(tmp_path)
    del declaración.contenido["imagenes"][0]["consultas"][0]["consulta_id"]
    with pytest.raises(ErrorDeConfiguración, match="consulta_id"):
        declaración.validar_forma()

def _contenido_válido() -> dict:
    consultas = [
        {"consulta_id": "c01-visual", "categoria": "visual_cromatico", "pedido": "p1"},
        {
            "consulta_id": "c01-cultural",
            "categoria": "conceptual_cultural",
            "pedido": "p2",
            "cultura_pedida": {
                "nombre": "Japón",
                "etiquetas": ["Japan"],
                "n": 341,
                "n_por_etiqueta": {"Japan": 341},
                "poblacion": "corpus",
                "n_poblacion": 2605,
                "criterio_de_culture": "crudo",
            },
        },
        {"consulta_id": "c01-ambiguo", "categoria": "ambiguo", "pedido": "p3"},
    ]
    return {
        "identidad": "5b6cba32126fed61-revision-1",
        "split_id": "5b6cba32126fed61",
        "autoria": "revision-1",
        "estado": "plantilla",
        "criterio_de_culture": "crudo",
        "revision": {"n_imagenes_revisadas": 30},
        "imagenes": [
            {
                "artwork_id": "aic:9",
                "imagen_hash": "sha256:ab",
                "archivo": "x.jpg",
                "atributos": ["paisaje"],
                "es_ejemplo": True,
                "consultas": [{**c, "es_ejemplo": True} for c in consultas],
            }
        ],
        "conteos": {
            "n_imagenes": 1,
            "n_consultas": 3,
            "n_pedidos_culturales": 1,
            "por_categoria": {"visual_cromatico": 1, "conceptual_cultural": 1, "ambiguo": 1},
        },
    }

def test_el_contenido_de_referencia_valida():
    SetDeConsultas(_contenido_válido()).validar()

def test_un_consulta_id_con_forma_de_hash_se_rechaza():
    contenido = _contenido_válido()
    contenido["imagenes"][0]["consultas"][0]["consulta_id"] = "a1b2c3d4e5f60718"
    with pytest.raises(ErrorDeArtefacto, match="se declaran, no se derivan"):
        SetDeConsultas(contenido).validar()

def test_una_cultura_pedida_con_n_cero_se_rechaza():
    contenido = _contenido_válido()
    cp = contenido["imagenes"][0]["consultas"][1]["cultura_pedida"]
    cp["n"] = 0
    cp["n_por_etiqueta"] = {"Japan": 0}
    with pytest.raises(ErrorDeArtefacto, match="≥ 1"):
        SetDeConsultas(contenido).validar()

def test_un_n_que_no_es_una_union_de_su_desglose_se_rechaza():
    contenido = _contenido_válido()
    cp = contenido["imagenes"][0]["consultas"][1]["cultura_pedida"]
    cp["etiquetas"] = ["Japan", "Korea"]
    cp["n_por_etiqueta"] = {"Japan": 341, "Korea": 56}
    cp["n"] = 500
    with pytest.raises(ErrorDeArtefacto, match="unión de obras"):
        SetDeConsultas(contenido).validar()

def test_un_pedido_cultural_sin_cultura_pedida_se_rechaza():
    contenido = _contenido_válido()
    del contenido["imagenes"][0]["consultas"][1]["cultura_pedida"]
    contenido["conteos"]["n_pedidos_culturales"] = 0
    with pytest.raises(ErrorDeArtefacto, match="cobertura cultural"):
        SetDeConsultas(contenido).validar()

def test_una_cultura_pedida_en_un_pedido_cromatico_se_rechaza():
    contenido = _contenido_válido()
    cultural = contenido["imagenes"][0]["consultas"][1]
    contenido["imagenes"][0]["consultas"][0]["cultura_pedida"] = cultural["cultura_pedida"]
    contenido["conteos"]["n_pedidos_culturales"] = 2
    with pytest.raises(ErrorDeArtefacto, match="Solo la categoría"):
        SetDeConsultas(contenido).validar()

def test_una_imagen_sin_las_tres_categorias_se_rechaza():
    contenido = _contenido_válido()
    contenido["imagenes"][0]["consultas"][0]["categoria"] = "ambiguo"
    contenido["conteos"]["por_categoria"] = {
        "visual_cromatico": 0, "conceptual_cultural": 1, "ambiguo": 2
    }
    with pytest.raises(ErrorDeArtefacto, match="una de cada"):
        SetDeConsultas(contenido).validar()

def test_una_imagen_sin_atributos_se_rechaza():
    contenido = _contenido_válido()
    contenido["imagenes"][0]["atributos"] = []
    with pytest.raises(ErrorDeArtefacto, match="atributos"):
        SetDeConsultas(contenido).validar()

def test_un_set_completo_tiene_que_tener_diez_por_tres():
    contenido = _contenido_válido()
    contenido["estado"] = "completo"
    for consulta in contenido["imagenes"][0]["consultas"]:
        consulta["es_ejemplo"] = False
    contenido["imagenes"][0]["es_ejemplo"] = False
    with pytest.raises(ErrorDeArtefacto, match="10 imágenes × 3 pedidos"):
        SetDeConsultas(contenido).validar()

def test_un_set_completo_de_diez_por_tres_valida():
    plantilla = _contenido_válido()
    cultura = plantilla["imagenes"][0]["consultas"][1]["cultura_pedida"]
    imagenes = []
    for n in range(1, 11):
        imagenes.append(
            {
                "artwork_id": f"cma:{n}",
                "imagen_hash": "sha256:ab",
                "archivo": f"{n}.jpg",
                "atributos": ["paisaje"],
                "consultas": [
                    {
                        "consulta_id": f"c{n:02d}-visual",
                        "categoria": "visual_cromatico",
                        "pedido": "versiones más cálidas",
                    },
                    {
                        "consulta_id": f"c{n:02d}-cultural",
                        "categoria": "conceptual_cultural",
                        "pedido": "el mismo tema pero en la cultura japonesa",
                        "cultura_pedida": dict(cultura),
                    },
                    {
                        "consulta_id": f"c{n:02d}-ambiguo",
                        "categoria": "ambiguo",
                        "pedido": "con más vida",
                    },
                ],
            }
        )
    contenido = {
        **plantilla,
        "estado": "completo",
        "imagenes": imagenes,
        "revision": {"n_imagenes_revisadas": 30},
        "conteos": {
            "n_imagenes": 10,
            "n_consultas": 30,
            "n_pedidos_culturales": 10,
            "por_categoria": {c: 10 for c in CATEGORÍAS},
        },
    }
    set_ = SetDeConsultas(contenido)
    set_.validar()
    assert set_.es_usable_para_medir
    assert len(set_.consultas()) == 30
    assert len(set_.culturas_pedidas()) == 10

def test_una_plantilla_tiene_todo_marcado_como_ejemplo():
    contenido = _contenido_válido()
    contenido["imagenes"][0]["consultas"][0]["es_ejemplo"] = False
    with pytest.raises(ErrorDeArtefacto, match="es_ejemplo"):
        SetDeConsultas(contenido).validar()

def test_revisadas_tiene_que_ser_al_menos_las_elegidas():
    contenido = _contenido_válido()
    contenido["revision"]["n_imagenes_revisadas"] = 0
    with pytest.raises(ErrorDeArtefacto, match="cuántas imágenes se miraron"):
        SetDeConsultas(contenido).validar()

def test_un_conteo_que_no_cierra_contra_su_contenido_se_rechaza():
    contenido = _contenido_válido()
    contenido["conteos"]["n_consultas"] = 30
    with pytest.raises(ErrorDeArtefacto, match="conteos.n_consultas"):
        SetDeConsultas(contenido).validar()

def test_la_identidad_tiene_que_coincidir_con_sus_ingredientes():
    contenido = _contenido_válido()
    contenido["autoria"] = "otra"
    with pytest.raises(ErrorDeArtefacto, match="identidad"):
        SetDeConsultas(contenido).validar()

def test_un_byte_distinto_se_detecta_al_leer(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    autor = AutorDelSetDeConsultas(cfg, almacén=almacén)
    resultado = autor.autorar(
        _declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto
    )
    resultado.rutas_de_imagenes[0].write_bytes(b"otra-imagen")

    with pytest.raises(ErrorDeArtefacto, match="no son los declarados"):
        SetDeConsultas.leer(almacén, autor.raíz, resultado.set_de_consultas.identidad)

def test_un_jpeg_reemplazado_por_el_museo_se_detecta_contra_el_catalogo(
    cfg, mundo, tmp_path
):
    almacén, catálogo, manifiesto = mundo
    resultado = AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
        _declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto
    )
    catálogo.registros = tuple(
        RegistroCanónico(
            **{**r.__dict__, "imagen": Imagen(r.imagen.image_url, "sha256:otro")}
        )
        if r.artwork_id == "aic:9"
        else r
        for r in catálogo.registros
    )
    with pytest.raises(ErrorDeArtefacto, match="reemplazó el JPEG"):
        verificar_hash_contra_el_catalogo(resultado.set_de_consultas, catálogo)

def test_el_set_expone_el_consulta_id_declarado_para_el_ejecutor(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    set_ = AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
        _declaración(tmp_path), catálogo=catálogo, manifiesto=manifiesto
    ).set_de_consultas

    consulta = set_.consulta("c01-cultural")
    assert consulta.artwork_id == "aic:9"
    assert consulta.imagen_hash == set_.imagenes()[0].imagen_hash
    assert consulta.archivo_de_imagen.endswith("aic_9.jpg")
    assert set_.por_categoria() == {c: 1 for c in CATEGORÍAS}
    with pytest.raises(ErrorDeArtefacto, match="no hay ninguna consulta"):
        set_.consulta("no-existe")

def test_el_artefacto_es_un_superconjunto_de_la_declaracion(cfg, mundo, tmp_path):
    almacén, catálogo, manifiesto = mundo
    declaración = _declaración(tmp_path)
    set_ = AutorDelSetDeConsultas(cfg, almacén=almacén).autorar(
        declaración, catálogo=catálogo, manifiesto=manifiesto
    ).set_de_consultas

    assert "declaracion" not in set_.contenido
    imagen_declarada = declaración.contenido["imagenes"][0]
    imagen = set_.imagenes()[0]
    assert imagen.atributos == tuple(imagen_declarada["atributos"])
    assert [c.pedido for c in imagen.consultas] == [
        c["pedido"] for c in imagen_declarada["consultas"]
    ]
    assert set_.criterio_de_culture == declaración.contenido["criterio_de_culture"]
    assert set_.n_imagenes_revisadas == declaración.contenido["revision"][
        "n_imagenes_revisadas"
    ]
    json.dumps(set_.a_json(), ensure_ascii=False)

def test_una_clave_que_el_artefacto_no_guardaria_se_rechaza(tmp_path):
    declaración = _declaración(tmp_path)
    declaración.contenido["imagenes"][0]["paleta"] = "fría"
    with pytest.raises(ErrorDeConfiguración, match="no se guardan"):
        declaración.validar_forma()

@pytest.fixture
def held_out_variado():
    registros = []
    for i, (museo, year, cultura, medium) in enumerate(
        [
            ("aic", 1890, "France", "Oil on canvas"),
            ("cma", 1650, "Italy", "oil on canvas"),
            ("cma", 1750, "Mughal India", "gum tempera, ink, and gold on paper"),
            ("cma", 1300, "Japan", "hanging scroll; ink on paper"),
            ("aic", 1500, "Germany", "Tempera on panel"),
            ("cma", 1820, "China", "Album leaf; ink and color on silk"),
        ]
        * 10
    ):
        registros.append(_registro(f"{museo}:{i}", culture=(cultura,)))
        registros[-1] = RegistroCanónico(
            **{**registros[-1].__dict__, "year": year, "medium": medium}
        )
    cola = _registro("aic:999", culture=("Niagara Falls",))
    registros.append(RegistroCanónico(**{**cola.__dict__, "artist": None}))
    return registros

def test_la_lista_corta_es_deterministica(cfg, held_out_variado):
    from tp3_rag.modulo_c.candidatas import PropuestaDeCandidatas

    a = PropuestaDeCandidatas(cfg).proponer(held_out_variado, cupo=20)
    b = PropuestaDeCandidatas(cfg).proponer(list(reversed(held_out_variado)), cupo=20)
    assert [c.artwork_id for c in a.candidatas] == [c.artwork_id for c in b.candidatas]

def test_la_lista_corta_cubre_los_valores_con_masa_de_cada_eje(cfg, held_out_variado):
    from tp3_rag.modulo_c.candidatas import PropuestaDeCandidatas

    propuesta = PropuestaDeCandidatas(cfg).proponer(held_out_variado, cupo=20)
    for eje in ("periodo", "museo", "tiene_artista"):
        datos = propuesta.cobertura[eje]
        assert datos["valores_cubiertos"] == datos["valores_del_held_out"], eje

def test_la_lista_corta_nombra_lo_que_dejo_afuera(cfg, held_out_variado):
    from tp3_rag.modulo_c.candidatas import PropuestaDeCandidatas

    propuesta = PropuestaDeCandidatas(cfg).proponer(held_out_variado, cupo=6)
    sin_cubrir = propuesta.afuera["cultura"]
    assert sin_cubrir["n_valores_sin_cubrir"] > 0
    assert sum(sin_cubrir["valores"].values()) == sin_cubrir["n_obras_en_esos_valores"]

def test_se_puede_rescatar_una_obra_de_la_cola(cfg, held_out_variado):
    from tp3_rag.modulo_c.candidatas import PropuestaDeCandidatas

    propuesta = PropuestaDeCandidatas(cfg).proponer(
        held_out_variado, cupo=6, incluir=["aic:999"]
    )
    assert "aic:999" in {c.artwork_id for c in propuesta.candidatas}

def test_no_se_puede_pedir_una_obra_que_no_es_del_held_out(cfg, held_out_variado):
    from tp3_rag.modulo_c.candidatas import PropuestaDeCandidatas

    with pytest.raises(ErrorDeConfiguración, match="held-out"):
        PropuestaDeCandidatas(cfg).proponer(held_out_variado, incluir=["cma:999999"])

def test_la_familia_de_tecnica_pliega_mayusculas(cfg):
    from tp3_rag.modulo_c.candidatas import familia_de_tecnica

    assert familia_de_tecnica("Oil on canvas") == familia_de_tecnica("oil on fabric")
    assert familia_de_tecnica("gum tempera, ink, and gold on paper") == "temple_sobre_papel"
    assert familia_de_tecnica(None) == "sin_medium"
