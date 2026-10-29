from __future__ import annotations

import pytest

from tp3_rag.dominio.variante import RegistroDeÍndices
from tp3_rag.modulo_c import validadores as val
from tp3_rag.nucleo import identidad

pytestmark = pytest.mark.nivel1

RECETA_BASE = {
    "catalogo_id": "e10c4e4f9cce0e74",
    "split_id": "5b6cba32126fed61",
    "criterio_de_inclusion": {"description_no_vacia": True},
    "checkpoint_de_clip": "openai/clip-vit-base-patch32",
    "version_del_preprocesado": "1",
    "estrategia_de_chunking": {"nombre": "por_campo_semantico", "tamano_objetivo": 500},
    "campos_de_knowledge": ["description", "did_you_know", "provenance"],
    "modelo_de_embedding_de_texto": "openai/text-embedding-3-small",
    "criterio_de_culture": "primer_componente_por_coma",
}

def _id_de(receta: dict) -> str:
    return identidad.variante_id(
        catalogo_id=receta["catalogo_id"],
        split_id=receta["split_id"],
        criterio_de_inclusion=receta["criterio_de_inclusion"],
        checkpoint_de_clip=receta["checkpoint_de_clip"],
        version_del_preprocesado=receta["version_del_preprocesado"],
        estrategia_de_chunking=receta["estrategia_de_chunking"],
        campos_de_knowledge=receta["campos_de_knowledge"],
        modelo_de_embedding_de_texto=receta["modelo_de_embedding_de_texto"],
        criterio_de_culture=receta["criterio_de_culture"],
    )

def _registro(
    *,
    campos=None,
    variante_id=None,
    fecha="2026-10-24T02:34:06+00:00",
    rol_de_imagen=None,
    ordinal=2,
    n_embebidas=5751,
    raiz=None,
    receta_extra=None,
) -> RegistroDeÍndices:
    receta = dict(RECETA_BASE)
    if campos is not None:
        receta["campos_de_knowledge"] = list(campos)
    receta.update(receta_extra or {})
    vid = variante_id or _id_de(receta)
    libro = {"embedding_texto": {"n_llamadas": 49, "n_items": 6232, "tokens_in": 622_314}}
    if rol_de_imagen is not None:
        libro["embedding_imagen"] = rol_de_imagen
    return RegistroDeÍndices(
        {
            "variante_id": vid,
            "receta": receta,
            "conteos": {"n_indexadas": 2605, "n_fragmentos": len(receta["campos_de_knowledge"]) * 2000},
            "colecciones": {"raiz": raiz or f"data/chroma/{vid}"},
            "libro_mayor": libro,
            "procedencia": {
                "fecha_de_construccion": fecha,
                "embebido": {
                    "ordinal": ordinal,
                    "n_embebidas": n_embebidas,
                    "n_ordinales_disponibles": 2,
                },
            },
            "invariantes_de_9_6": {},
        }
    )

CONGELADA = dict(campos=["description", "did_you_know", "provenance"])
AMPLIADA = dict(
    campos=["description", "did_you_know", "provenance", "exhibitions", "citations"],
    fecha="2026-10-27T19:02:27+00:00",
)

def test_r2_pasa_con_dos_variantes_que_difieren_en_un_eje_declarado():
    r = val.r2_variantes_coexistiendo([_registro(**CONGELADA), _registro(**AMPLIADA)])
    assert r.pasa and (r.n, r.de) == (4, 4)
    assert r.detalle["c_id_re_derivado_de_la_receta_sola"] == "2/2"
    assert r.detalle["ejes_de_la_receta_que_difieren"] == ["campos_de_knowledge"]

def test_r2_no_es_verificable_con_una_sola_variante():
    r = val.r2_variantes_coexistiendo([_registro(**CONGELADA)])
    assert not r.pasa
    assert r.detalle["a_coexisten"].startswith("1 variantes")

def test_r2_atrapa_la_colision_de_6_7_3_con_su_causa_declarada():
    vid = _id_de(dict(RECETA_BASE))
    r = val.r2_variantes_coexistiendo(
        [
            _registro(variante_id=vid, **CONGELADA),
            _registro(variante_id=vid, **AMPLIADA),
        ]
    )
    assert not r.pasa
    assert any("falta un ingrediente" in f.get("causa_declarada", "") for f in r.fallas)

def test_r2_atrapa_un_id_que_no_sale_de_su_propia_receta():
    r = val.r2_variantes_coexistiendo(
        [_registro(**CONGELADA), _registro(variante_id="deadbeefdeadbeef", **AMPLIADA)]
    )
    assert not r.pasa
    assert r.detalle["c_id_re_derivado_de_la_receta_sola"] == "1/2"
    assert any(f.get("variante_id") == "deadbeefdeadbeef" for f in r.fallas)

def test_r2_atrapa_dos_variantes_compartiendo_el_directorio_de_colecciones():
    r = val.r2_variantes_coexistiendo(
        [
            _registro(raiz="data/chroma/fijo", **CONGELADA),
            _registro(raiz="data/chroma/fijo", **AMPLIADA),
        ]
    )
    assert not r.pasa
    assert r.detalle["d_un_directorio_por_variante"].startswith("1 raíces")

def test_r3_pasa_con_cero_llamadas_y_el_mismo_ordinal_citado():
    r = val.r3_cache_de_imagen_reusado([_registro(**CONGELADA), _registro(**AMPLIADA)])
    assert r.pasa
    assert r.detalle["llamadas_nuevas_al_PuertoEmbeddingImagen"] == 0
    assert r.detalle["a_rol_ausente_del_libro_mayor"] == "1/1"

def test_r3_no_es_verificable_con_una_sola_variante():
    r = val.r3_cache_de_imagen_reusado([_registro(**CONGELADA)])
    assert not r.pasa
    assert "necesita dos" in r.poblacion

def test_r3_atrapa_que_la_segunda_variante_haya_llamado_a_clip():
    r = val.r3_cache_de_imagen_reusado(
        [
            _registro(**CONGELADA),
            _registro(
                rol_de_imagen={"n_llamadas": 180, "n_items": 5751, "segundos": 400.0},
                **AMPLIADA,
            ),
        ]
    )
    assert not r.pasa
    assert r.detalle["llamadas_nuevas_al_PuertoEmbeddingImagen"] == 180
    assert any(f.get("llamadas_a_embedding_imagen") == 180 for f in r.fallas)

def test_r3_atrapa_un_ordinal_de_embebido_nuevo_aunque_el_libro_este_limpio():
    r = val.r3_cache_de_imagen_reusado(
        [
            _registro(**CONGELADA),
            _registro(ordinal=3, n_embebidas=5800, **AMPLIADA),
        ]
    )
    assert not r.pasa
    assert any("un solo ordinal" in str(f.get("esperado", "")) for f in r.fallas)

def test_r3_ordena_por_fecha_de_construccion_y_no_por_el_nombre():
    vieja = _registro(fecha="2026-10-01T00:00:00+00:00", **CONGELADA)
    nueva = _registro(
        campos=["description", "did_you_know", "provenance", "exhibitions", "citations"],
        fecha="2026-10-27T19:02:27+00:00",
        rol_de_imagen={"n_llamadas": 7, "n_items": 7, "segundos": 1.0},
    )
    for entrada in ([vieja, nueva], [nueva, vieja]):
        r = val.r3_cache_de_imagen_reusado(entrada)
        assert not r.pasa
        assert r.detalle["llamadas_nuevas_al_PuertoEmbeddingImagen"] == 7
        assert nueva.variante_id in r.poblacion
