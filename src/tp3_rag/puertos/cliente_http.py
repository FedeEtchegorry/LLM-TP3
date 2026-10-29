from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import requests

from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.nucleo.identidad import clave_de_cache_http, json_canónico

ESTADOS_CACHEABLES = (200, 404)

class RecursoNoEncontrado(ErrorDelProyecto):
    pass

class CacheMiss(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class RespuestaHTTP:
    url: str
    status: int
    cuerpo: bytes
    headers: dict[str, str]
    desde_cache: bool

    def json(self):
        import json

        return json.loads(self.cuerpo.decode("utf-8"))

class ClienteHTTP:
    def __init__(
        self,
        *,
        cache_dir: Path,
        headers_por_dominio: Mapping[str, Mapping[str, str]],
        rate_limit_por_dominio: Mapping[str, int] | None = None,
        reintentos: int = 4,
        backoff_base: float = 1.5,
        timeout: float = 60.0,
        red_habilitada: bool = True,
        dormir=time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.headers_por_dominio = {d: dict(h) for d, h in headers_por_dominio.items()}
        self.rate_limit_por_dominio = dict(rate_limit_por_dominio or {})
        self.reintentos = reintentos
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.red_habilitada = red_habilitada
        self._dormir = dormir
        self._última_llamada: dict[str, float] = {}
        self._sesión = requests.Session() if red_habilitada else None
        self.n_hits = 0
        self.n_misses = 0

    def headers_de(self, url: str) -> dict[str, str]:
        host = urlsplit(url).hostname or ""
        resueltos: dict[str, str] = {}
        for dominio in sorted(self.headers_por_dominio, key=len):
            if host == dominio or host.endswith("." + dominio):
                resueltos.update(self.headers_por_dominio[dominio])
        return resueltos

    def _rutas(self, url: str, nombres: list[str]) -> tuple[Path, Path]:
        clave = clave_de_cache_http(url, nombres)
        d = self.cache_dir / clave[:2]
        return d / f"{clave}.bin", d / f"{clave}.json"

    def _leer_cache(self, cuerpo: Path, meta: Path, url: str) -> RespuestaHTTP | None:
        if not (cuerpo.exists() and meta.exists()):
            return None
        import json

        m = json.loads(meta.read_text(encoding="utf-8"))
        return RespuestaHTTP(url, m["status"], cuerpo.read_bytes(), m.get("headers", {}), True)

    def _escribir_cache(self, cuerpo: Path, meta: Path, r: RespuestaHTTP, nombres: list[str]) -> None:
        cuerpo.parent.mkdir(parents=True, exist_ok=True)
        tmp = cuerpo.with_suffix(".bin.parcial")
        tmp.write_bytes(r.cuerpo)
        tmp.replace(cuerpo)
        tmp_m = meta.with_suffix(".json.parcial")
        tmp_m.write_text(
            json_canónico(
                {
                    "url": r.url,
                    "status": r.status,
                    "headers": {
                        k: v
                        for k, v in r.headers.items()
                        if k.lower() in ("content-type", "content-length", "etag", "last-modified")
                    },
                    "headers_enviados": sorted(nombres),
                }
            ),
            encoding="utf-8",
        )
        tmp_m.replace(meta)

    def get(self, url: str) -> RespuestaHTTP:
        headers = self.headers_de(url)
        nombres = sorted(headers)
        cuerpo, meta = self._rutas(url, nombres)

        cacheada = self._leer_cache(cuerpo, meta, url)
        if cacheada is not None:
            self.n_hits += 1
            if cacheada.status == 404:
                raise RecursoNoEncontrado(f"404 (cacheado) {url}")
            return cacheada

        if not self.red_habilitada:
            raise CacheMiss(
                f"nivel 1: la red está deshabilitada y {url} no está en el caché "
                f"({self.cache_dir}). Un miss es falla de test, no una llamada."
            )

        self.n_misses += 1
        self._respetar_rate_limit(url)
        r = self._pedir_con_reintentos(url, headers)
        if r.status in ESTADOS_CACHEABLES:
            self._escribir_cache(cuerpo, meta, r, nombres)
        if r.status == 404:
            raise RecursoNoEncontrado(f"404 {url}")
        return r

    def _respetar_rate_limit(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        for dominio, por_minuto in self.rate_limit_por_dominio.items():
            if not (host == dominio or host.endswith("." + dominio)) or por_minuto <= 0:
                continue
            intervalo = 60.0 / por_minuto
            falta = intervalo - (time.monotonic() - self._última_llamada.get(dominio, 0.0))
            if falta > 0:
                self._dormir(falta)
            self._última_llamada[dominio] = time.monotonic()

    def _pedir_con_reintentos(self, url: str, headers: dict[str, str]) -> RespuestaHTTP:
        último: Exception | None = None
        for intento in range(self.reintentos):
            try:
                resp = self._sesión.get(url, headers=headers, timeout=self.timeout)
            except requests.Timeout as e:
                último = e
            except requests.RequestException as e:
                último = e
            else:
                if resp.status_code == 403:
                    host = urlsplit(url).hostname
                    raise ErrorDeConfiguración(
                        f"403 de {host} en {url}\n"
                        f"  headers enviados: {sorted(headers) or '(ninguno)'}\n"
                        f"  No se reintenta: el 403 es error de configuración, no de red. "
                        f"Causas conocidas:\n"
                        f"  · falta `AIC-User-Agent` en un dominio de AIC — API o imágenes;\n"
                        f"  · `User-Agent` default de la librería en CMA;\n"
                        f"  · parámetros inválidos en `/search` de AIC: `limit > 100` o "
                        f"`limit × page > 1000` también devuelven 403."
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    último = ErrorDelProyecto(f"{resp.status_code} {url}")
                else:
                    return RespuestaHTTP(
                        url, resp.status_code, resp.content, dict(resp.headers), False
                    )
            if intento < self.reintentos - 1:
                self._dormir(self.backoff_base ** (intento + 1))
        raise ErrorDelProyecto(f"{url} falló después de {self.reintentos} intentos: {último}")

    @classmethod
    def desde_configuración(cls, cfg, credenciales, *, red_habilitada: bool = True) -> "ClienteHTTP":
        from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

        proyecto = cfg["http.proyecto"]
        headers = {d: dict(h) for d, h in cfg["http.headers_por_dominio"].items()}
        for dominio, hs in headers.items():
            for clave, valor in list(hs.items()):
                if valor == "{aic_user_agent}":
                    hs[clave] = f"{proyecto} ({credenciales.exigir_contacto_aic()})"
                elif valor == "{user_agent}":
                    hs[clave] = cfg["http.user_agent"]
        return cls(
            cache_dir=RAÍZ_DEL_REPO / cfg["rutas.http_cache"],
            headers_por_dominio=headers,
            rate_limit_por_dominio=cfg["http.rate_limit_por_minuto"],
            reintentos=cfg["http.reintentos"],
            backoff_base=cfg["http.backoff_base"],
            timeout=cfg["http.timeout_segundos"],
            red_habilitada=red_habilitada,
        )
