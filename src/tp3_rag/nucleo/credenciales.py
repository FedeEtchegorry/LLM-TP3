from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tp3_rag.nucleo.errores import ErrorDeConfiguración

def _leer_dotenv(ruta: Path) -> dict[str, str]:
    if not ruta.exists():
        return {}
    pares: dict[str, str] = {}
    for línea in ruta.read_text(encoding="utf-8").splitlines():
        línea = línea.strip()
        if not línea or línea.startswith("#") or "=" not in línea:
            continue
        clave, _, valor = línea.partition("=")
        pares[clave.strip()] = valor.strip().strip("\"'")
    return pares

VARIABLES_DE_LA_KEY = ("LLM_API_KEY", "OPENAI_API_KEY")

@dataclass(frozen=True)
class Credenciales:
    contacto_aic: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    langsmith_api_key: str | None = field(default=None, repr=False)
    variable_de_la_key: str | None = None

    @classmethod
    def desde_entorno(cls, raíz: Path | None = None) -> "Credenciales":
        from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

        entorno = {**_leer_dotenv((raíz or RAÍZ_DEL_REPO) / ".env"), **os.environ}
        variable = next((v for v in VARIABLES_DE_LA_KEY if entorno.get(v)), None)
        return cls(
            contacto_aic=entorno.get("TP3_CONTACTO_AIC") or None,
            openai_api_key=entorno.get(variable) if variable else None,
            langsmith_api_key=entorno.get("LANGSMITH_API_KEY") or None,
            variable_de_la_key=variable,
        )

    def exigir_contacto_aic(self) -> str:
        if not self.contacto_aic:
            raise ErrorDeConfiguración(
                "falta `TP3_CONTACTO_AIC` (en `.env` o en el entorno). Es el contacto "
                "del header `AIC-User-Agent: <proyecto> (<email>)`, que hace falta en "
                "todas las requests a dominios de AIC — API e imágenes. Sin él, AIC "
                "devuelve 403 con el challenge de Cloudflare y el síntoma no apunta a la "
                "causa."
            )
        return self.contacto_aic

    def exigir_openai(self) -> str:
        if not self.openai_api_key:
            raise ErrorDeConfiguración(
                "falta `LLM_API_KEY` (en `.env` o en el entorno; se acepta "
                "`OPENAI_API_KEY` como alias, y si están las dos gana la primera). "
                "Es la virtual key del proxy LiteLLM de la cátedra y la necesita el "
                "`PuertoEmbeddingTexto`. Con la key sola no alcanza: el host sale de "
                "`proveedor.base_url`, porque esta key contra `api.openai.com` devuelve "
                "401. "
                "Sin ella sí se puede construir la colección `artworks` —sale del caché "
                "de embeddings— y fragmentar: `poetry run tp3-indexar --hasta "
                "fragmentos`. Lo que no se puede es `knowledge`, y por lo tanto tampoco "
                "registrar: que toda obra indexada tenga al menos un fragmento en "
                "`knowledge` es uno de los invariantes de la transacción de registración, "
                "y una variante que no los pasa no se registra."
            )
        return self.openai_api_key

    def exigir_langsmith(self) -> str:
        if not self.langsmith_api_key:
            raise ErrorDeConfiguración(
                "falta `LANGSMITH_API_KEY` (en `.env` o en el entorno). Es la key "
                "propia del proyecto —no la de la cátedra— y la pide el "
                "`PuertoTrazas`. "
                "El pipeline corre sin ella: `--set trazas.proveedor=nulo` cumple el "
                "mismo contrato y los tokens de la tabla de costo salen del artefacto de "
                "corrida. Lo que se pierde es la traza, que es un extra entregable — y "
                "por eso no se degrada en silencio: pedir `nulo` es una decisión "
                "declarada que queda escrita en el `config_id`."
            )
        return self.langsmith_api_key

    def constancia(self) -> dict[str, bool]:
        return {
            "contacto_aic_presente": bool(self.contacto_aic),
            "openai_api_key_presente": bool(self.openai_api_key),
            "langsmith_api_key_presente": bool(self.langsmith_api_key),
        }
