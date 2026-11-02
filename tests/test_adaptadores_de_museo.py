from __future__ import annotations

import pytest

from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos.cliente_http import CacheMiss
from tp3_rag.puertos.museos.aic import TOPE_DE_LIMIT, AdaptadorAIC
from tp3_rag.puertos.museos.cma import AdaptadorCMA

pytestmark = pytest.mark.nivel1

N = 8

def test_aic_lista_desde_el_fixture_sin_red(cliente_sin_red, cfg):
    adaptador = AdaptadorAIC(cliente_sin_red, cfg)
    ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    assert len(ids) == N
    assert ids == sorted(ids)

def test_aic_la_cobertura_separa_el_limite_del_hueco(cliente_sin_red, cfg):
    adaptador = AdaptadorAIC(cliente_sin_red, cfg)
    adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    cobertura = adaptador.cobertura_de_paginación()
    assert cobertura.museo == "aic"
    assert cobertura.n_alcanzadas == N
    assert cobertura.truncado_por_limite is True
    assert cobertura.n_no_alcanzadas == 0
    assert cobertura.n_total_declarada > N

def test_aic_el_total_declarado_sale_del_primer_probe(cliente_sin_red, cfg):
    adaptador = AdaptadorAIC(cliente_sin_red, cfg)
    adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    assert adaptador.cobertura_de_paginación().n_total_declarada == 1965

def test_aic_el_keyset_nunca_toca_el_tope_de_mil(cfg, cliente_sin_red):
    url = AdaptadorAIC(cliente_sin_red, cfg)._query(list(cfg["ingesta.tipos"]), "libre", 12345)
    assert f"limit={TOPE_DE_LIMIT}" in url and "page=1" in url
    assert TOPE_DE_LIMIT * 1 <= 1000

def test_aic_la_consulta_lleva_el_rango_de_id_y_el_orden(cfg, cliente_sin_red):
    url = AdaptadorAIC(cliente_sin_red, cfg)._query(list(cfg["ingesta.tipos"]), "libre", 500)
    assert "range%5D%5Bid%5D%5Bgte%5D=500" in url
    assert "sort%5B0%5D=id" in url
    assert "is_public_domain" in url

def test_aic_pide_fields_explicito(cfg, cliente_sin_red):
    url = AdaptadorAIC(cliente_sin_red, cfg)._query(list(cfg["ingesta.tipos"]), "libre", 0)
    assert "fields=" in url
    for campo in ("image_id", "description", "style_title", "place_of_origin"):
        assert campo in url

def test_aic_traer_sale_del_prefetch_y_no_pega_otra_vez(cliente_sin_red, cfg):
    adaptador = AdaptadorAIC(cliente_sin_red, cfg)
    ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    misses_antes = cliente_sin_red.n_misses
    crudo = adaptador.traer(ids[0])
    assert crudo["id"] == ids[0]
    assert cliente_sin_red.n_misses == misses_antes

def test_aic_una_licencia_que_no_sabe_pedir_no_se_adivina(cliente_sin_red, cfg):
    with pytest.raises(ErrorDeConstrucción, match="libre"):
        AdaptadorAIC(cliente_sin_red, cfg).listar(["Painting"], "cc-by", N)

def test_cobertura_antes_de_listar_es_un_error(cliente_sin_red, cfg):
    with pytest.raises(ErrorDeConstrucción, match="antes de listar"):
        AdaptadorAIC(cliente_sin_red, cfg).cobertura_de_paginación()

def test_cma_lista_desde_el_fixture_sin_red(cliente_sin_red, cfg):
    adaptador = AdaptadorCMA(cliente_sin_red, cfg)
    ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    assert len(ids) == N
    assert adaptador.cobertura_de_paginación().n_total_declarada > N

def test_cma_el_universo_ya_excluye_las_obras_sin_imagen(cliente_sin_red, cfg):
    url = AdaptadorCMA(cliente_sin_red, cfg)._url(["Painting"], "libre", 0, N)
    assert "has_image=1" in url and "cc0=1" in url

def test_cma_mas_de_un_tipo_falla_en_vez_de_fingir(cliente_sin_red, cfg):
    with pytest.raises(ErrorDeConstrucción, match="escalar"):
        AdaptadorCMA(cliente_sin_red, cfg)._url(["Painting", "Drawing"], "libre", 0, N)

def test_cma_traer_sale_del_prefetch(cliente_sin_red, cfg):
    adaptador = AdaptadorCMA(cliente_sin_red, cfg)
    ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
    assert adaptador.traer(ids[0])["id"] == ids[0]

def test_una_obra_fuera_del_fixture_falla_en_vez_de_pedir(cliente_sin_red, cfg):
    with pytest.raises(CacheMiss):
        AdaptadorCMA(cliente_sin_red, cfg).traer(999999999)

def test_el_fixture_trae_un_registro_con_culture_lista(corot):
    assert isinstance(corot["culture"], list)
    assert corot["culture"] == ["France, 19th century"]

def test_el_fixture_trae_un_registro_sin_description(aic_sin_description):
    assert not aic_sin_description.get("description")

def test_los_dos_adaptadores_cumplen_el_mismo_puerto(cliente_sin_red, cfg):
    for fábrica in (AdaptadorAIC, AdaptadorCMA):
        adaptador = fábrica(cliente_sin_red, cfg)
        for operación in ("listar", "traer", "traer_imagen", "cobertura_de_paginación"):
            assert callable(getattr(adaptador, operación))
        assert adaptador.museo and adaptador.version_del_adaptador
