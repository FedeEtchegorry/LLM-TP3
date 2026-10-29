from __future__ import annotations

from pathlib import Path

import pytest

from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.puertos.cliente_http import (
    CacheMiss,
    ClienteHTTP,
    RecursoNoEncontrado,
    RespuestaHTTP,
)

pytestmark = pytest.mark.nivel1

HEADERS = {
    "artic.edu": {"AIC-User-Agent": "proyecto (mail)", "User-Agent": "proyecto"},
    "clevelandart.org": {"User-Agent": "proyecto"},
}

def _cliente(tmp_path: Path, **kw) -> ClienteHTTP:
    return ClienteHTTP(
        cache_dir=tmp_path / "cache", headers_por_dominio=HEADERS, dormir=lambda s: None, **kw
    )

@pytest.mark.parametrize(
    "url,header_esperado",
    [
        ("https://api.artic.edu/api/v1/artworks", "AIC-User-Agent"),
        ("https://www.artic.edu/iiif/2/abc/full/843,/0/default.jpg", "AIC-User-Agent"),
    ],
)
def test_r6_el_header_de_aic_va_en_todos_los_dominios_de_aic(tmp_path, url, header_esperado):
    assert header_esperado in _cliente(tmp_path, red_habilitada=False).headers_de(url)

def test_el_match_es_por_sufijo_de_dominio(tmp_path):
    cliente = _cliente(tmp_path, red_habilitada=False)
    assert cliente.headers_de("https://api.artic.edu/x") == cliente.headers_de(
        "https://www.artic.edu/y"
    )

def test_cma_recibe_un_user_agent_que_no_es_el_default_de_la_libreria(tmp_path):
    cliente = _cliente(tmp_path, red_habilitada=False)
    for dominio in ("openaccess-api.clevelandart.org", "openaccess-cdn.clevelandart.org"):
        headers = cliente.headers_de(f"https://{dominio}/x")
        assert headers.get("User-Agent")
        assert "urllib" not in headers["User-Agent"].lower()

def test_un_dominio_no_declarado_no_recibe_headers(tmp_path):
    assert _cliente(tmp_path, red_habilitada=False).headers_de("https://ejemplo.com/x") == {}

def test_sin_red_un_miss_es_falla_de_test_no_una_llamada(tmp_path):
    with pytest.raises(CacheMiss, match="nivel 1"):
        _cliente(tmp_path, red_habilitada=False).get("https://api.artic.edu/api/v1/artworks")

def test_el_fixture_del_repo_responde_sin_red(cliente_sin_red, cfg):
    from tp3_rag.puertos.museos.aic import AdaptadorAIC

    adaptador = AdaptadorAIC(cliente_sin_red, cfg)
    url = adaptador._query(list(cfg["ingesta.tipos"]), "libre", 0)
    respuesta = cliente_sin_red.get(url)
    assert respuesta.desde_cache and respuesta.status == 200
    assert respuesta.json()["pagination"]["total"] > 0

def test_un_hit_no_cuenta_como_miss(cliente_sin_red, cfg):
    from tp3_rag.puertos.museos.aic import AdaptadorAIC

    url = AdaptadorAIC(cliente_sin_red, cfg)._query(list(cfg["ingesta.tipos"]), "libre", 0)
    cliente_sin_red.get(url)
    assert cliente_sin_red.n_hits == 1 and cliente_sin_red.n_misses == 0

def test_el_cache_no_guarda_el_valor_de_los_headers(cliente_sin_red, cfg):
    import json

    metas = list((cliente_sin_red.cache_dir).rglob("*.json"))
    assert metas
    for meta in metas:
        m = json.loads(meta.read_text(encoding="utf-8"))
        assert "headers_enviados" in m
        assert all(isinstance(h, str) for h in m["headers_enviados"])
        assert "@" not in json.dumps(m)

class _RespuestaFalsa:
    def __init__(self, status, contenido=b"", headers=None):
        self.status_code, self.content, self.headers = status, contenido, headers or {}

class _SesiónFalsa:
    def __init__(self, respuestas):
        self.respuestas, self.llamadas = list(respuestas), 0

    def get(self, url, headers=None, timeout=None):
        self.llamadas += 1
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

def test_el_403_no_se_reintenta_y_nombra_el_dominio(tmp_path):
    cliente = _cliente(tmp_path, reintentos=4)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(403, b"<html>Just a moment...")])
    with pytest.raises(ErrorDeConfiguración) as e:
        cliente.get("https://api.artic.edu/api/v1/artworks")
    assert cliente._sesión.llamadas == 1
    assert "api.artic.edu" in str(e.value)
    assert "AIC-User-Agent" in str(e.value)

def test_el_403_no_se_cachea(tmp_path):
    cliente = _cliente(tmp_path, reintentos=1)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(403)])
    with pytest.raises(ErrorDeConfiguración):
        cliente.get("https://api.artic.edu/x")
    assert not list((tmp_path / "cache").rglob("*.bin"))

def test_el_429_y_el_5xx_se_reintentan_con_backoff(tmp_path):
    cliente = _cliente(tmp_path, reintentos=4)
    cliente._sesión = _SesiónFalsa(
        [_RespuestaFalsa(429), _RespuestaFalsa(503), _RespuestaFalsa(200, b"ok")]
    )
    assert cliente.get("https://api.artic.edu/x").cuerpo == b"ok"
    assert cliente._sesión.llamadas == 3

def test_el_timeout_se_reintenta(tmp_path):
    import requests

    cliente = _cliente(tmp_path, reintentos=3)
    cliente._sesión = _SesiónFalsa([requests.Timeout(), _RespuestaFalsa(200, b"ok")])
    assert cliente.get("https://api.artic.edu/x").cuerpo == b"ok"

def test_el_404_se_reporta_y_la_obra_se_descarta(tmp_path):
    cliente = _cliente(tmp_path, reintentos=1)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(404)])
    with pytest.raises(RecursoNoEncontrado):
        cliente.get("https://www.artic.edu/iiif/2/x/full/843,/0/default.jpg")

def test_el_404_si_se_cachea_porque_es_un_hecho_del_remoto(tmp_path):
    cliente = _cliente(tmp_path, reintentos=1)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(404)])
    with pytest.raises(RecursoNoEncontrado):
        cliente.get("https://www.artic.edu/x.jpg")
    cliente.red_habilitada = False
    with pytest.raises(RecursoNoEncontrado, match="cacheado"):
        cliente.get("https://www.artic.edu/x.jpg")

def test_la_escritura_del_cache_es_atomica(tmp_path):
    cliente = _cliente(tmp_path, reintentos=1)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(200, b"cuerpo")])
    cliente.get("https://api.artic.edu/x")
    assert not list((tmp_path / "cache").rglob("*.parcial"))

def test_el_rate_limit_por_dominio_espera_lo_declarado(tmp_path):
    dormidas = []
    cliente = ClienteHTTP(
        cache_dir=tmp_path / "c",
        headers_por_dominio=HEADERS,
        rate_limit_por_dominio={"api.artic.edu": 60, "www.artic.edu": 0},
        reintentos=1,
        dormir=dormidas.append,
    )
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(200, b"1"), _RespuestaFalsa(200, b"2")])
    cliente.get("https://api.artic.edu/a")
    cliente.get("https://api.artic.edu/b")
    assert dormidas and 0 < dormidas[-1] <= 1.0

    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(200, b"3")])
    antes = len(dormidas)
    cliente.get("https://www.artic.edu/img.jpg")
    assert len(dormidas) == antes

def test_la_respuesta_expone_si_vino_del_cache(tmp_path):
    cliente = _cliente(tmp_path, reintentos=1)
    cliente._sesión = _SesiónFalsa([_RespuestaFalsa(200, b"x")])
    assert cliente.get("https://api.artic.edu/x").desde_cache is False
    assert cliente.get("https://api.artic.edu/x").desde_cache is True

def test_la_respuesta_es_inmutable():
    with pytest.raises(Exception):
        RespuestaHTTP("u", 200, b"", {}, False).status = 404
