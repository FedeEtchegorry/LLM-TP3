from __future__ import annotations

from typing import Sequence
from urllib.parse import urlencode

from tp3_rag.dominio.urls import url_de_imagen
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos.catalogo_de_museo import CoberturaDePaginación

TOPE_DE_LIMIT = 100

class AdaptadorAIC:
    museo = "aic"
    version_del_adaptador = "1"

    def __init__(self, cliente, cfg) -> None:
        self.cliente = cliente
        self.cfg = cfg
        self.base = cfg["museos.aic.base"]
        self.campos = list(cfg["museos.aic.campos"])
        self.plantillas = cfg["ingesta.urls"]
        self._crudos: dict = {}
        self._cobertura: CoberturaDePaginación | None = None

    def _query(self, tipos: Sequence[str], licencia: str, desde_id: int) -> str:
        if licencia != "libre":
            raise ErrorDeConstrucción(
                f"el adaptador de AIC solo sabe pedir licencia 'libre' "
                f"(`is_public_domain=true`); recibió {licencia!r}"
            )
        params: list[tuple[str, object]] = [
            ("query[bool][must][0][term][is_public_domain]", "true"),
        ]
        for i, tipo in enumerate(tipos):
            params.append((f"query[bool][must][1][terms][artwork_type_title.keyword][{i}]", tipo))
        params.append(("query[bool][must][2][range][id][gte]", desde_id))
        params += [
            ("sort[0]", "id"),
            ("limit", TOPE_DE_LIMIT),
            ("page", 1),
            ("fields", ",".join(self.campos)),
        ]
        return f"{self.base}/search?{urlencode(params)}"

    def listar(self, tipos: Sequence[str], licencia: str, límite: int | None = None) -> list:
        ids: list = []
        cursor, total_declarado, requests_hechas = 0, None, 0

        while True:
            d = self.cliente.get(self._query(tipos, licencia, cursor)).json()
            requests_hechas += 1
            if total_declarado is None:
                total_declarado = d["pagination"]["total"]
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
            cursor = lote[-1]["id"] + 1
            if len(lote) < TOPE_DE_LIMIT:
                break

        únicos = list(dict.fromkeys(ids))
        truncado = límite is not None and len(únicos) >= límite
        self._cobertura = CoberturaDePaginación(
            museo=self.museo,
            n_total_declarada=total_declarado or 0,
            n_alcanzadas=len(únicos),
            n_no_alcanzadas=0 if truncado else max(0, (total_declarado or 0) - len(únicos)),
            truncado_por_limite=truncado,
            nota=(
                f"keyset sobre id (sort[0]=id, limit={TOPE_DE_LIMIT}, page=1): "
                f"{requests_hechas} requests. El tope de 1.000 de /search no se toca."
            ),
        )
        return únicos

    def traer(self, id_del_museo) -> dict:
        crudo = self._crudos.get(id_del_museo)
        if crudo is not None:
            return crudo
        url = f"{self.base}/{id_del_museo}?fields={','.join(self.campos)}"
        crudo = self.cliente.get(url).json()["data"]
        self._crudos[id_del_museo] = crudo
        return crudo

    def traer_imagen(self, id_del_museo) -> bytes:
        url = url_de_imagen(self.museo, self.traer(id_del_museo), self.plantillas)
        if not url:
            raise ErrorDeConstrucción(
                f"aic:{id_del_museo} no tiene `image_id`: no hay imagen que traer"
            )
        return self.cliente.get(url).cuerpo

    def cobertura_de_paginación(self) -> CoberturaDePaginación:
        if self._cobertura is None:
            raise ErrorDeConstrucción("cobertura_de_paginación() antes de listar()")
        return self._cobertura
