from __future__ import annotations

import pytest

from tp3_rag.nucleo.identidad import (
    catalogo_id,
    clave_de_cache_http,
    hash_de_bytes,
    hash_de_registros,
)

pytestmark = pytest.mark.nivel1

def _registro(**cambios) -> dict:
    base = {
        "artwork_id": "cma:124078",
        "museum": "cma",
        "type": "Painting",
        "title": "Pond at Ville-d'Avray",
        "year": 1865,
        "source_url": "https://clevelandart.org/art/1944.80",
        "media_license": "CC0",
        "campos_de_texto": [
            {
                "section": "description",
                "texto": "Corot painted this view of a pond…",
                "source_url": "https://clevelandart.org/art/1944.80",
                "license": "CC0",
            }
        ],
        "procedencia": {"year": ["cma", "sortable_date"]},
    }
    base.update(cambios)
    return base

def _id(registros, **kw) -> str:
    return catalogo_id(
        museos=kw.get("museos", ["aic", "cma"]),
        tipos=kw.get("tipos", ["Painting"]),
        version_del_normalizador=kw.get("version", "1"),
        registros=registros,
    )

def test_una_description_editada_produce_otro_catalogo_id():
    original = _registro()
    editado = _registro(
        campos_de_texto=[{**original["campos_de_texto"][0], "texto": "Corot painted this pond."}]
    )
    assert _id([original]) != _id([editado])

def test_una_imagen_reemplazada_produce_otro_catalogo_id():
    a = _registro(imagen={"image_url": "u", "hash_bytes": "sha256:aaa"})
    b = _registro(imagen={"image_url": "u", "hash_bytes": "sha256:bbb"})
    assert _id([a]) != _id([b])

def test_una_obra_nueva_produce_otro_catalogo_id():
    assert _id([_registro()]) != _id([_registro(), _registro(artwork_id="cma:999")])

def test_la_licencia_de_un_campo_entra_en_la_identidad():
    a = _registro()
    b = _registro(campos_de_texto=[{**a["campos_de_texto"][0], "license": "CC-BY-4.0"}])
    assert _id([a]) != _id([b])

def test_los_museos_los_tipos_y_la_version_entran():
    r = [_registro()]
    assert _id(r) != _id(r, museos=["cma"])
    assert _id(r) != _id(r, tipos=["Painting", "Print"])
    assert _id(r) != _id(r, version="2")

def test_la_fecha_de_ingesta_no_entra_en_el_catalogo_id():
    import inspect

    assert "fecha" not in inspect.signature(catalogo_id).parameters

def test_dos_corridas_del_mismo_universo_dan_el_mismo_id():
    assert _id([_registro(), _registro(artwork_id="aic:1")]) == _id(
        [_registro(), _registro(artwork_id="aic:1")]
    )

def test_el_orden_de_las_paginas_no_es_el_contenido():
    a, b = _registro(artwork_id="aic:1"), _registro(artwork_id="cma:2")
    assert hash_de_registros([a, b]) == hash_de_registros([b, a])

def test_el_criterio_de_inclusion_no_entra():
    import inspect

    params = set(inspect.signature(catalogo_id).parameters)
    assert not params & {"criterio", "criterio_de_inclusion", "split_id", "variante_id"}

def test_la_clave_de_cache_depende_de_la_url():
    assert clave_de_cache_http("https://a/1", ["User-Agent"]) != clave_de_cache_http(
        "https://a/2", ["User-Agent"]
    )

def test_la_clave_de_cache_depende_de_que_headers_exige_el_dominio():
    assert clave_de_cache_http("https://a/1", []) != clave_de_cache_http(
        "https://a/1", ["AIC-User-Agent"]
    )

def test_la_clave_de_cache_no_depende_del_orden_de_los_headers():
    assert clave_de_cache_http("https://a/1", ["B", "A"]) == clave_de_cache_http(
        "https://a/1", ["A", "B"]
    )

def test_la_clave_de_cache_no_lleva_el_valor_del_contacto():
    import inspect

    firma = inspect.signature(clave_de_cache_http)
    assert "nombres_de_headers" in firma.parameters
    assert clave_de_cache_http("https://a/1", ["AIC-User-Agent"]) == clave_de_cache_http(
        "https://a/1", ["AIC-User-Agent"]
    )

def test_el_hash_de_bytes_lleva_el_algoritmo_adelante():
    assert hash_de_bytes(b"x").startswith("sha256:")
    assert hash_de_bytes(b"x") != hash_de_bytes(b"y")

INGREDIENTES = dict(
    catalogo_id="e10c4e4f9cce0e74",
    split_id="5b6cba32126fed61",
    criterio_de_inclusion={"description_no_vacia": True, "imagen_real": True},
    checkpoint_de_clip="openai/clip-vit-base-patch32",
    version_del_preprocesado="1",
    estrategia_de_chunking={"nombre": "por_campo_semantico", "tamano_objetivo": 500},
    campos_de_knowledge=["description", "did_you_know", "provenance"],
    modelo_de_embedding_de_texto="text-embedding-3-small",
    criterio_de_culture="primer_componente_por_coma",
)

def _variante(**cambios) -> str:
    from tp3_rag.nucleo.identidad import variante_id

    return variante_id(**{**INGREDIENTES, **cambios})

@pytest.mark.parametrize(
    "ingrediente, otro_valor",
    [
        ("catalogo_id", "0000000000000000"),
        ("split_id", "0000000000000000"),
        ("criterio_de_inclusion", {"description_no_vacia": True, "imagen_real": False}),
        ("checkpoint_de_clip", "openai/clip-vit-large-patch14"),
        ("version_del_preprocesado", "2"),
        ("estrategia_de_chunking", {"nombre": "por_campo_semantico", "tamano_objetivo": 2000}),
        ("campos_de_knowledge", ["description"]),
        ("modelo_de_embedding_de_texto", "text-embedding-3-large"),
        ("criterio_de_culture", "crudo"),
    ],
)
def test_los_nueve_ingredientes_mueven_el_variante_id(ingrediente, otro_valor):
    assert _variante(**{ingrediente: otro_valor}) != _variante()

def test_el_orden_de_los_campos_de_knowledge_no_mueve_el_id():
    assert _variante(campos_de_knowledge=["provenance", "description", "did_you_know"]) == (
        _variante()
    )

def test_el_orden_del_criterio_de_inclusion_tampoco():
    assert _variante(
        criterio_de_inclusion={"imagen_real": True, "description_no_vacia": True}
    ) == _variante()

def test_la_firma_no_acepta_el_ordinal_del_embebido():
    import inspect

    from tp3_rag.nucleo.identidad import variante_id

    parámetros = set(inspect.signature(variante_id).parameters)
    assert "ordinal" not in parámetros
    assert not [p for p in parámetros if "ordinal" in p]

def test_alfa_y_n_no_entran_porque_son_de_la_corrida():
    import inspect

    from tp3_rag.nucleo.identidad import variante_id

    parámetros = set(inspect.signature(variante_id).parameters)
    assert not {"alfa", "alpha", "n", "fusion", "prompt_id"} & parámetros

def test_las_tres_identidades_de_clase_3_son_legibles_y_no_hashes():
    from tp3_rag.nucleo import identidad

    assert identidad.rubrica_id("piloto") == "piloto"
    assert identidad.identidad_del_pool(
        split_id="0123456789abcdef", designacion="piloto"
    ) == "0123456789abcdef-piloto"
    assert identidad.identidad_del_aporte(
        split_id="0123456789abcdef", rubrica_id="piloto", anotador="ana"
    ) == "0123456789abcdef-piloto-ana"

def test_el_rubrica_id_no_puede_tener_forma_de_hash():
    from tp3_rag.nucleo import identidad
    from tp3_rag.nucleo.errores import ErrorDeConfiguración

    with pytest.raises(ErrorDeConfiguración, match="hash_corto"):
        identidad.rubrica_id("a1b2c3d4e5f60718")

def test_el_pool_id_se_parte_de_vuelta_sin_ambiguedad():
    from tp3_rag.nucleo import identidad

    assert identidad.partir_identidad_del_pool("0123456789abcdef-definitivo-2") == (
        "0123456789abcdef",
        "definitivo-2",
    )

def test_la_identidad_del_aporte_no_se_puede_partir_y_por_eso_no_hay_inversa():
    from tp3_rag.nucleo import identidad

    assert not hasattr(identidad, "partir_identidad_del_aporte")
    assert identidad.identidad_del_aporte(
        split_id="0123456789abcdef", rubrica_id="piloto-x", anotador="y"
    ) == identidad.identidad_del_aporte(
        split_id="0123456789abcdef", rubrica_id="piloto", anotador="x-y"
    )

def test_el_anotador_se_valida_como_etiqueta_declarada():
    from tp3_rag.nucleo import identidad
    from tp3_rag.nucleo.errores import ErrorDeConfiguración

    for inválido in ("../ana", "Ana Pérez", "ana/beto", ""):
        with pytest.raises(ErrorDeConfiguración, match="anotador"):
            identidad.identidad_del_aporte(
                split_id="0123456789abcdef", rubrica_id="piloto", anotador=inválido
            )
