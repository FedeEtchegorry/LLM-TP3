from __future__ import annotations

from typing import Any, Mapping

from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.nucleo.observabilidad import NodoDeTraza, PuertoTrazasNulo, Span

PROVEEDORES: tuple[str, ...] = ("langsmith", "nulo")

CLAVES_DE_USAGE: tuple[str, ...] = ("input_tokens", "output_tokens", "total_tokens")

def usage_metadata_de(tokens: Mapping[str, Any] | None) -> dict:
    entrada = int((tokens or {}).get("tokens_in") or 0)
    salida = int((tokens or {}).get("tokens_out") or 0)
    if not entrada and not salida:
        return {}
    return {
        "usage_metadata": {
            "input_tokens": entrada,
            "output_tokens": salida,
            "total_tokens": entrada + salida,
        }
    }

class TrazasLangSmith:
    def __init__(
        self,
        *,
        api_key: str,
        proyecto: str,
        timeout_segundos: float = 10.0,
        url: str = "https://api.smith.langchain.com",
    ) -> None:
        from langsmith import Client

        if not api_key:
            raise ErrorDeConfiguración(
                "falta `LANGSMITH_API_KEY` (en `.env` o en el entorno). Es la key propia "
                "del proyecto y la necesita el `PuertoTrazas`. "
                "Sin ella el pipeline corre igual: `[trazas] proveedor = \"nulo\"` "
                "cumple el mismo contrato y los tokens salen del artefacto "
                "de corrida, no del servicio."
            )
        self._cliente = Client(
            api_url=url, api_key=api_key, timeout_ms=int(timeout_segundos * 1000)
        )
        self.proyecto = proyecto
        self.url = url
        self.spans: list[Span] = []
        self.nodos: list[NodoDeTraza] = []
        self.fallas = 0
        self.n_posteados = 0

    def span(self, span: Span) -> None:
        self.spans.append(span)

    def nodo(self, nodo: NodoDeTraza) -> None:
        self.nodos.append(nodo)
        try:
            self._postear(nodo)
            self.n_posteados += 1
        except Exception:
            self.fallas += 1

    def cerrar(self) -> None:
        try:
            self._cliente.flush()
        except Exception:  # pragma: no cover - ídem, no propaga
            self.fallas += 1

    def _postear(self, nodo: NodoDeTraza) -> None:
        tokens = dict(nodo.metadata.get("tokens") or {})
        extra = {
            "metadata": {
                **{k: v for k, v in nodo.metadata.items() if k != "tokens"},
                **{f"tokens_{k.split('_')[-1]}": v for k, v in tokens.items() if v},
                **usage_metadata_de(tokens),
                "politica_de_trazado": nodo.politica,
                "campos_censurados": nodo.campos_censurados,
            }
        }
        self._cliente.create_run(
            name=nodo.nombre,
            run_type=nodo.tipo,
            inputs=dict(nodo.entradas),
            outputs=dict(nodo.salidas) or None,
            id=nodo.id,
            trace_id=nodo.trace_id,
            parent_run_id=nodo.padre,
            dotted_order=nodo.dotted_order,
            start_time=nodo.inicio,
            end_time=nodo.fin,
            error=nodo.error,
            project_name=self.proyecto,
            extra=extra,
        )

    def url_del_proyecto(self) -> str:
        return f"https://smith.langchain.com/  →  proyecto {self.proyecto!r}"

def armar(cfg, credenciales) -> Any:
    proveedor = cfg["trazas.proveedor"]
    if proveedor not in PROVEEDORES:
        raise ErrorDeConfiguración(
            f"`trazas.proveedor = {proveedor!r}` no es uno de {list(PROVEEDORES)}."
        )
    if proveedor == "nulo":
        return PuertoTrazasNulo()
    return TrazasLangSmith(
        api_key=credenciales.exigir_langsmith(),
        proyecto=cfg["trazas.proyecto"],
        timeout_segundos=float(cfg["trazas.timeout_segundos"]),
    )
