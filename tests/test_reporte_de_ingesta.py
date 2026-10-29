from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.modulo_a.reporte_de_ingesta import ReporteDeIngesta

pytestmark = pytest.mark.nivel1

def _registro(museo="cma", **cambios) -> RegistroCanónico:
    base = dict(
        artwork_id=f"{museo}:1",
        museum=museo,
        type="Painting",
        title="T",
        artist="A",
        date_display="1865",
        year=1865,
        medium="oil",
        style=None,
        culture_raw=("France, 19th century",),
        source_url="https://x/1",
        imagen=Imagen("https://x/1.jpg", "sha256:a"),
        media_license="CC0",
        campos_de_texto=(CampoDeTexto("description", "texto", "https://x/1", "CC0"),),
        procedencia={},
    )
    base.update(cambios)
    return RegistroCanónico(**base)

def test_a9a_cuenta_cada_campo_canonico_por_museo():
    registros = [
        _registro("aic", artwork_id="aic:1", artist="Van Gogh", style="Post-Impressionism"),
        _registro("aic", artwork_id="aic:2", artist=None),
        _registro("cma", artwork_id="cma:1"),
    ]
    a9a = ReporteDeIngesta.cobertura_de_normalización(registros)
    assert a9a["por_museo"]["aic"]["n_universo_normalizado"] == 2
    assert a9a["por_museo"]["aic"]["artist"] == {"n": 1, "pct": 50.0}
    assert a9a["por_museo"]["cma"]["artist"] == {"n": 1, "pct": 100.0}
    assert a9a["total"]["n_universo_normalizado"] == 3

def test_a9a_declara_su_denominador_y_es_el_universo_no_las_indexadas():
    a9a = ReporteDeIngesta.cobertura_de_normalización([_registro()])
    assert "universo normalizado" in a9a["denominador"]
    assert "no obras indexadas" in a9a["denominador"]

def test_a9a_cuenta_culture_como_resuelta_solo_si_hay_valor():
    a9a = ReporteDeIngesta.cobertura_de_normalización(
        [_registro(artwork_id="cma:1"), _registro(artwork_id="cma:2", culture_raw=None)]
    )
    assert a9a["por_museo"]["cma"]["culture"] == {"n": 1, "pct": 50.0}

def test_a9a_incluye_media_license_y_no_license():
    campos = ReporteDeIngesta.cobertura_de_normalización([_registro()])["campos"]
    assert "media_license" in campos and "license" not in campos

def test_la_cobertura_de_texto_cuenta_texto_y_no_claves_presentes():
    con = _registro(artwork_id="cma:1")
    sin = _registro(artwork_id="cma:2", campos_de_texto=())
    cobertura = ReporteDeIngesta.cobertura_de_texto([con, sin])
    assert cobertura["cma"]["secciones"]["description"] == {"n": 1, "pct": 50.0}

def test_un_campo_con_texto_en_blanco_no_cuenta():
    vacío = _registro(
        campos_de_texto=(CampoDeTexto("description", "   ", "https://x/1", "CC0"),)
    )
    cobertura = ReporteDeIngesta.cobertura_de_texto([vacío])
    assert cobertura["cma"]["secciones"] == {}

def test_las_licencias_por_seccion_dejan_ver_el_cc_by_de_aic():
    registros = [
        _registro(
            "aic",
            artwork_id="aic:1",
            campos_de_texto=(
                CampoDeTexto("description", "t", "u", "CC-BY-4.0"),
                CampoDeTexto("provenance", "t", "u", "CC0"),
            ),
        ),
        _registro("cma", artwork_id="cma:1"),
    ]
    tabla = ReporteDeIngesta.licencias_por_seccion(registros)
    assert tabla["aic"]["description"] == {"CC-BY-4.0": 1}
    assert tabla["aic"]["provenance"] == {"CC0": 1}
    assert tabla["cma"]["description"] == {"CC0": 1}

def _reporte(cfg) -> dict:
    from tp3_rag.nucleo.credenciales import Credenciales
    from tp3_rag.puertos.catalogo_de_museo import CoberturaDePaginación

    return ReporteDeIngesta.armar(
        catalogo_id="abc123",
        ordinal=1,
        fecha_de_ingesta="2026-10-14T10:00:00+00:00",
        commit="deadbeef",
        configuración=cfg,
        credenciales_presentes=Credenciales(contacto_aic="x@y").constancia(),
        registros=[_registro()],
        coberturas_de_paginación=[CoberturaDePaginación("cma", 3950, 1, 3949)],
        preparadas=[],
        a9b={"criterio": "crudo", "total": {}, "por_museo": {}},
        version_del_normalizador="1",
        segundos={"total": 1.0},
    ).a_json()

def _claves(nodo, prefijo="") -> set[str]:
    if isinstance(nodo, dict):
        return {
            k for clave, v in nodo.items() for k in {clave} | _claves(v, f"{prefijo}{clave}.")
        }
    if isinstance(nodo, list):
        return {k for v in nodo for k in _claves(v, prefijo)}
    return set()

def test_el_reporte_no_lleva_los_conteos_de_split_ni_de_variante(cfg):
    claves = _claves(_reporte(cfg))
    ajenas = {
        "n_indexadas",
        "n_fragmentos",
        "corpus_indexado",
        "held_out",
        "split_id",
        "variante_id",
        "elegibles_con_texto_e_imagen",
    }
    assert not (claves & ajenas), f"el reporte copió conteos de otro dueño: {claves & ajenas}"

def test_el_reporte_si_lleva_n_con_imagen_real(cfg):
    r = _reporte(cfg)
    assert "n_con_imagen_real" in r["universo"]["por_museo"]["cma"]
    assert "n_sin_imagen_real" in r["universo"]["por_museo"]["cma"]

def test_la_identidad_es_de_clase_2_y_lleva_ordinal(cfg):
    r = _reporte(cfg)
    assert r["clase_de_identidad"] == 2
    assert r["ordinal"] == 1
    assert r["catalogo_id"] == "abc123"

def test_la_fecha_esta_adentro_como_procedencia_y_no_en_la_identidad(cfg):
    r = _reporte(cfg)
    assert r["procedencia"]["fecha_de_ingesta"] == "2026-10-14T10:00:00+00:00"
    assert "fecha" not in r["catalogo_id"]

def test_la_configuracion_resuelta_viaja_adentro(cfg):
    r = _reporte(cfg)
    assert r["configuracion_resuelta"]["ingesta"]["museos"] == ["aic", "cma"]
    assert r["config_id"] == cfg.config_id
    assert r["procedencia"]["capas_de_configuracion"]

def test_de_las_credenciales_solo_viaja_la_constancia(cfg):
    r = _reporte(cfg)
    assert r["procedencia"]["credenciales"] == {
        "contacto_aic_presente": True,
        "openai_api_key_presente": False,
        "langsmith_api_key_presente": False,
    }
    assert "x@y" not in str(r)

def test_el_embebido_es_null_y_lo_dice(cfg):
    r = _reporte(cfg)
    assert r["embebido"] is None
    assert "etapa aparte" in r["nota_de_embebido"]

def test_la_cobertura_de_paginacion_viaja_por_museo(cfg):
    pag = _reporte(cfg)["universo"]["por_museo"]["cma"]["paginacion"]
    assert pag["n_total_declarada"] == 3950
    assert pag["n_no_alcanzadas"] == 3949
