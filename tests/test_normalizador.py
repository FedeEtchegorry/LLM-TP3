from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import prefijar
from tp3_rag.modulo_a import normalizador as mod
from tp3_rag.modulo_a.normalizador import Normalizador, html_a_texto, parsear_artista
from tp3_rag.nucleo.errores import ErrorDeConstrucción, LicenciaNoDeclarada

pytestmark = pytest.mark.nivel1

@pytest.mark.parametrize(
    "museo,id_crudo,esperado",
    [("aic", 28560, "aic:28560"), ("cma", 124078, "cma:124078"), ("aic", "9", "aic:9")],
)
def test_prefijado_de_artwork_id(museo, id_crudo, esperado):
    assert prefijar(museo, id_crudo) == esperado

def test_los_ids_numericos_colisionan_y_el_prefijo_los_separa():
    assert prefijar("aic", 1000) != prefijar("cma", 1000)

@pytest.mark.parametrize(
    "descripción,esperado",
    [
        ("Pieter de Hooch (Dutch, 1629–1684)", "Pieter de Hooch"),
        ("Jean Baptiste Camille Corot (French, 1796–1875)", "Jean Baptiste Camille Corot"),
        ("Kitagawa Utamaro (Japanese, 1753–1806) (artist)", "Kitagawa Utamaro"),
        ("Zha Shibiao (Chinese, 1615–1698)", "Zha Shibiao"),
        ("William Merritt Chase", "William Merritt Chase"),
        (None, None),
        ("", None),
        ("   ", None),
        ("(Dutch, 1629–1684)", None),
    ],
)
def test_parsear_artista(descripción, esperado):
    assert parsear_artista(descripción) == esperado

def test_nr1_description_de_aic_es_cc_by(cfg, crudos_aic):
    normalizador = Normalizador(cfg)
    con_description = [c for c in crudos_aic if c.get("description")]
    assert con_description, "el fixture tiene que traer al menos un AIC con description"
    registro = normalizador.normalizar(con_description[0], "aic", "sha256:deadbeef")
    licencias = {c.section: c.license for c in registro.campos_de_texto}
    assert licencias["description"] == "CC-BY-4.0"
    assert all(v == "CC0" for k, v in licencias.items() if k != "description")

def test_nr1_todo_el_resto_de_cma_es_cc0(cfg, corot):
    registro = Normalizador(cfg).normalizar(corot, "cma", "sha256:deadbeef")
    assert {c.license for c in registro.campos_de_texto} == {"CC0"}

def test_nr1_un_par_sin_entrada_en_la_tabla_es_error_duro(cfg, crudos_cma):
    normalizador = Normalizador(cfg)
    normalizador.campos_de_texto = {
        **{m: dict(v) for m, v in cfg["ingesta.campos_de_texto"].items()},
        "cma": {
            **dict(cfg["ingesta.campos_de_texto.cma"]),
            "conservation_statement": {"origen": "tombstone", "render": "texto_plano"},
        },
    }
    with pytest.raises(LicenciaNoDeclarada) as e:
        normalizador.normalizar(crudos_cma[0], "cma", None)
    assert "conservation_statement" in str(e.value)
    assert "error duro" in str(e.value)

def test_media_license_tambien_se_declara_y_no_se_asume(cfg, corot):
    normalizador = Normalizador(cfg)
    normalizador.media_license = {}
    with pytest.raises(ErrorDeConstrucción, match="media_license"):
        normalizador.normalizar(corot, "cma", None)

def test_nr2_culture_raw_de_cma_se_conserva_verbatim(cfg, corot):
    assert corot["culture"] == ["France, 19th century"]
    registro = Normalizador(cfg).normalizar(corot, "cma", None)
    assert registro.culture_raw == ("France, 19th century",)
    assert isinstance(registro.culture_raw, tuple)

def test_nr2_el_escalar_de_aic_se_envuelve_en_lista_sin_colapsar(cfg, crudos_aic):
    con_lugar = [c for c in crudos_aic if c.get("place_of_origin")][0]
    registro = Normalizador(cfg).normalizar(con_lugar, "aic", None)
    assert registro.culture_raw == (con_lugar["place_of_origin"],)

def test_nr2_ninguna_pieza_escribe_un_valor_normalizado_en_culture_raw(cfg, corot):
    from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture

    registro = Normalizador(cfg).normalizar(corot, "cma", None)
    vista = NormalizadorDeCulture("primer_componente_por_coma").normalizar(registro.culture_raw)
    assert vista == ["France"]
    assert registro.culture_raw == ("France, 19th century",)

def test_nr3_un_aic_sin_description_no_tiene_la_section(cfg, aic_sin_description):
    registro = Normalizador(cfg).normalizar(aic_sin_description, "aic", None)
    assert "description" not in {c.section for c in registro.campos_de_texto}
    assert not registro.tiene_description_no_vacia

def test_nr3_cma_no_tiene_style_y_la_clave_no_aparece(cfg, corot):
    registro = Normalizador(cfg).normalizar(corot, "cma", None)
    assert registro.style is None
    assert "style" not in registro.a_json()

def test_nr3_la_serializacion_omite_las_claves_ausentes(cfg, aic_sin_description):
    d = Normalizador(cfg).normalizar(aic_sin_description, "aic", None).a_json()
    assert "" not in d.values()
    for clave in ("title", "artist", "year", "medium", "style"):
        assert d.get(clave) != ""

def test_nr3_sin_imagen_la_clave_imagen_no_existe(cfg, corot):
    registro = Normalizador(cfg).normalizar(corot, "cma", hash_de_imagen=None)
    assert registro.imagen is None
    assert "imagen" not in registro.a_json()
    assert not registro.tiene_imagen_real

def test_nr4_dos_normalizaciones_del_mismo_crudo_son_identicas(cfg, corot):
    normalizador = Normalizador(cfg)
    a = normalizador.normalizar(corot, "cma", "sha256:abc")
    b = normalizador.normalizar(corot, "cma", "sha256:abc")
    assert a.a_json() == b.a_json()

def test_nr4_el_modulo_no_importa_reloj_red_ni_azar():
    prohibidos = {"time", "datetime", "random", "requests", "urllib", "socket"}
    presentes = prohibidos & set(vars(mod))
    assert not presentes, f"el Normalizador importó {presentes}: NR4 roto"

def test_nr4_la_version_del_normalizador_es_explicita():
    assert mod.VERSIÓN_DEL_NORMALIZADOR
    assert isinstance(mod.VERSIÓN_DEL_NORMALIZADOR, str)

def test_los_dos_museos_producen_el_mismo_esquema(cfg, crudos_aic, crudos_cma):
    normalizador = Normalizador(cfg)
    a = normalizador.normalizar(crudos_aic[0], "aic", "sha256:a").a_json()
    c = normalizador.normalizar(crudos_cma[0], "cma", "sha256:c").a_json()
    obligatorias = {"artwork_id", "museum", "type", "source_url", "media_license", "campos_de_texto"}
    assert obligatorias <= set(a) and obligatorias <= set(c)

def test_source_url_es_lo_que_se_cita_y_su_ausencia_es_error_duro(cfg, crudos_cma):
    sin_url = {**crudos_cma[0]}
    sin_url.pop("url", None)
    with pytest.raises(ErrorDeConstrucción, match="source_url"):
        Normalizador(cfg).normalizar(sin_url, "cma", None)

def test_year_de_aic_sale_de_date_start_y_cma_de_sortable_date(cfg, crudos_aic, corot):
    normalizador = Normalizador(cfg)
    a = normalizador.normalizar(crudos_aic[0], "aic", None)
    assert a.year == crudos_aic[0]["date_start"]
    assert a.procedencia["year"] == ["aic", "date_start"]
    c = normalizador.normalizar(corot, "cma", None)
    assert c.year == corot["sortable_date"]
    assert c.procedencia["year"] == ["cma", "sortable_date"]

def test_la_procedencia_dice_de_que_campo_salio_cada_canonico(cfg, corot):
    p = Normalizador(cfg).normalizar(corot, "cma", None).procedencia
    assert p["artist"] == ["cma", "creators.description"]
    assert p["medium"] == ["cma", "technique"]
    assert p["source_url"] == ["cma", "url"]

@pytest.mark.parametrize(
    "crudo,esperado",
    [
        ("<p>Perhaps the most famous depiction</p>", "Perhaps the most famous depiction"),
        ("<em>Chinese Landscape Painting</em>. Cleveland", "Chinese Landscape Painting. Cleveland"),
        ("Rue de l&#39;Abreuvoir", "Rue de l'Abreuvoir"),
        ("a<br>b", "a\nb"),
        ("  espacios   colapsados  ", "espacios colapsados"),
    ],
)
def test_html_a_texto(crudo, esperado):
    assert html_a_texto(crudo) == esperado

def test_la_description_de_aic_no_queda_con_etiquetas(cfg, crudos_aic):
    normalizador = Normalizador(cfg)
    for crudo in crudos_aic:
        if not crudo.get("description"):
            continue
        registro = normalizador.normalizar(crudo, "aic", None)
        texto = next(c.texto for c in registro.campos_de_texto if c.section == "description")
        assert "<p>" not in texto and "&#" not in texto

def test_provenance_de_cma_es_una_lista_de_dicts_y_se_renderiza_a_texto(cfg, crudos_cma):
    con_prov = [c for c in crudos_cma if isinstance(c.get("provenance"), list) and c["provenance"]]
    assert con_prov, "el fixture tiene que traer un CMA con provenance"
    registro = Normalizador(cfg).normalizar(con_prov[0], "cma", None)
    texto = next(c.texto for c in registro.campos_de_texto if c.section == "provenance")
    assert isinstance(texto, str) and texto
    assert "{" not in texto and "sortorder" not in texto

def test_exhibitions_de_cma_vacio_no_produce_section(cfg, crudos_cma):
    crudo = {**crudos_cma[0], "exhibitions": {"current": [], "legacy": []}}
    registro = Normalizador(cfg).normalizar(crudo, "cma", None)
    assert "exhibitions" not in {c.section for c in registro.campos_de_texto}
