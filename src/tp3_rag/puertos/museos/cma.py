from __future__ import annotations

from typing import Sequence
from urllib.parse import urlencode

from tp3_rag.dominio.urls import url_de_imagen
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos.catalogo_de_museo import CoberturaDePaginación

TOPE_DE_LIMIT = 1000

class AdaptadorCMA:
    museo = "cma"
    version_del_adaptador = "1"

    def __init__(self, cliente, cfg) -> None:
        self.cliente = cliente
        self.cfg = cfg
        self.base = cfg["museos.cma.base"]
        self.plantillas = cfg["ingesta.urls"]
        self._crudos: dict = {}
        self._cobertura: CoberturaDePaginación | None = None

    def _url(self, tipos: Sequence[str], licencia: str, skip: int, limit: int) -> str:
        if licencia != "libre":
            raise ErrorDeConstrucción(
                f"el adaptador de CMA solo sabe pedir licencia 'libre' (`cc0=1`); "
                f"recibió {licencia!r}"
            )
        if len(tipos) != 1:
            raise ErrorDeConstrucción(
                f"CMA toma `type` escalar y se configuraron {len(tipos)} tipos: {list(tipos)}. "
                f"Para más de uno hace falta un recorrido por tipo (no implementado )."
            )
        params = {
            "cc0": 1,
            "has_image": 1,
            "type": tipos[0],
            "limit": limit,
            "skip": skip,
        }
        return f"{self.base}?{urlencode(params)}"

    def listar(self, tipos: Sequence[str], licencia: str, límite: int | None = None) -> list:
        ids: list = []
        skip, total_declarado, requests_hechas = 0, None, 0
        por_página = TOPE_DE_LIMIT if límite is None else min(TOPE_DE_LIMIT, límite)

        while True:
            d = self.cliente.get(self._url(tipos, licencia, skip, por_página)).json()
            requests_hechas += 1
            if total_declarado is None:
                total_declarado = (d.get("info") or {}).get("total", 0)
            lote = d.get("data") or []
            if not lote:
                break
            for crudo in lote:
                self._crudos[crudo["id"]] = crudo
                ids.append(crudo["id"])
                if límite is not None and len(ids) >= límite:
                    break
            if límite is not None and len(ids) >= límite:
                break
            skip += por_página

        únicos = list(dict.fromkeys(ids))
        truncado = límite is not None and len(únicos) >= límite
        self._cobertura = CoberturaDePaginación(
            museo=self.museo,
            n_total_declarada=total_declarado or 0,
            n_alcanzadas=len(únicos),
            n_no_alcanzadas=0 if truncado else max(0, (total_declarado or 0) - len(únicos)),
            truncado_por_limite=truncado,
            nota=(
                f"skip/limit={por_página}: {requests_hechas} requests. El universo de CMA "
                f"ya excluye las obras sin imagen (`has_image=1`)."
            ),
        )
        return únicos

    def traer(self, id_del_museo) -> dict:
        crudo = self._crudos.get(id_del_museo)
        if crudo is not None:
            return crudo
        crudo = self.cliente.get(f"{self.base}{id_del_museo}").json()["data"]
        self._crudos[id_del_museo] = crudo
        return crudo

    def traer_imagen(self, id_del_museo) -> bytes:
        url = url_de_imagen(self.museo, self.traer(id_del_museo), self.plantillas)
        if not url:
            raise ErrorDeConstrucción(f"cma:{id_del_museo} no tiene `images.web.url`")
        return self.cliente.get(url).cuerpo

    def cobertura_de_paginación(self) -> CoberturaDePaginación:
        if self._cobertura is None:
            raise ErrorDeConstrucción("cobertura_de_paginación() antes de listar()")
        return self._cobertura
