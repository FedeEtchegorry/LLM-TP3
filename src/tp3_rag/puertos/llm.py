from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto

MÁXIMO_DE_IMÁGENES = 2

class FallaDeEtapa(ErrorDelProyecto):
    def __init__(self, causa: str, detalle: str) -> None:
        super().__init__(f"{causa}: {detalle}")
        self.causa = causa
        self.detalle = detalle

class RechazoDeContenido(FallaDeEtapa):
    pass

@dataclass(frozen=True)
class RespuestaDeLLM:
    texto: str
    uso: dict = field(default_factory=dict)
    muestreo_efectivo: dict = field(default_factory=dict)
    modelo_efectivo: str = ""
    n_intentos: int = 1
    n: int = 1

    def json(self) -> Any:
        return json.loads(self.texto)

    def a_json(self) -> dict:
        return {
            "uso": dict(self.uso),
            "muestreo_efectivo": dict(self.muestreo_efectivo),
            "modelo_efectivo": self.modelo_efectivo,
            "n_intentos": self.n_intentos,
            "nota_de_bd3": (
                "`muestreo_efectivo` es **lo que el puerto envió**, no lo que el binding "
                "pidió: *registrar «lo configurado» en vez de «lo usado» es la "
                "forma elegante de no cumplir el muestreo fijado, registrado y con su variabilidad medida*."
            ),
        }

def _validar_modelo(modelo: str) -> str:
    if "/" not in modelo:
        raise ErrorDeConfiguración(
            f"el modelo {modelo!r} no lleva prefijo de proveedor. 🔴 **el prefijo del proveedor en el nombre del modelo/el registro de decisiones: "
            f"el prefijo `openai/` es obligatorio** —el proxy LiteLLM de la cátedra "
            f"responde `400 · no healthy deployments` al nombre pelado—. Se valida antes "
            f"de llamar porque ese síntoma no apunta a esta causa. Declaralo como "
            f"`openai/{modelo}` en `[bindings.<rol>] modelo`."
        )
    return modelo

def _cuerpo(binding, mensajes: Sequence[Mapping[str, Any]], imagenes: Sequence[bytes]) -> dict:
    if len(imagenes) > MÁXIMO_DE_IMÁGENES:
        raise ErrorDeConfiguración(
            f"se pidieron {len(imagenes)} imágenes y el contrato de los puertos es **0..2**. El "
            f"tope es del diseño y no del proveedor: la forma de C4 que el registro de decisiones midió "
            f"—y con la que se eligió `gpt-4.1-mini`— es **dos** imágenes en una llamada."
        )
    contenido: list[dict] = []
    for m in mensajes:
        if m.get("rol") == "sistema":
            continue
        contenido.append({"type": "text", "text": m["texto"]})
    for datos in imagenes:
        b64 = base64.b64encode(datos).decode("ascii")
        contenido.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    sistema = [m["texto"] for m in mensajes if m.get("rol") == "sistema"]
    openai_mensajes: list[dict] = []
    if sistema:
        openai_mensajes.append({"role": "system", "content": "\n\n".join(sistema)})
    openai_mensajes.append({"role": "user", "content": contenido})

    cuerpo: dict[str, Any] = {
        "model": binding.modelo,
        "messages": openai_mensajes,
        **binding.muestreo.a_json(),
    }
    if binding.esquema_de_salida == "json":
        cuerpo["response_format"] = {"type": "json_object"}
    return cuerpo

def _muestreo_enviado(cuerpo: Mapping[str, Any]) -> dict:
    claves = ("temperature", "top_p", "max_tokens", "seed", "response_format")
    return {k: cuerpo[k] for k in claves if k in cuerpo}

class LLMOpenAI:
    OPERACIONES = ("invocar",)

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_segundos: float = 120.0,
        backoff_base: float = 1.5,
    ) -> None:
        from openai import OpenAI

        if not base_url:
            raise ErrorDeConfiguración(
                "`base_url` es obligatorio. Sin él el SDK apunta a "
                "`api.openai.com` y la key de la cátedra —una *virtual key* del proxy "
                "LiteLLM de ITBA— devuelve **401**. Sale de `[proveedor] base_url`, que es "
                "**una sola sección para todos los puertos que hablan con el proveedor** "
                ": el host **no** sale del binding."
            )
        self._cliente = OpenAI(
            api_key=api_key, base_url=base_url, timeout=float(timeout_segundos)
        )
        self.base_url = base_url
        self.backoff_base = float(backoff_base)

    def invocar(
        self,
        binding,
        mensajes: Sequence[Mapping[str, Any]],
        imagenes: Sequence[bytes] = (),
    ) -> RespuestaDeLLM:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        _validar_modelo(binding.modelo)
        cuerpo = _cuerpo(binding, mensajes, imagenes)
        enviado = _muestreo_enviado(cuerpo)

        último: Exception | None = None
        for intento in range(1, int(binding.reintentos) + 2):
            try:
                respuesta = self._cliente.chat.completions.create(**cuerpo)
            except RateLimitError as e:
                último = e
                if intento > int(binding.reintentos):
                    raise FallaDeEtapa("429 agotado", str(e)) from None
                time.sleep(self.backoff_base**intento)
                continue
            except APIConnectionError as e:
                raise FallaDeEtapa(
                    "red caída",
                    f"no se pudo llegar al proveedor ({e}). En la **consulta** esto es "
                    f"falla de etapa clase 3 y la política la aplica el "
                    f"`EjecutorDeEtapas`, no este puerto: un puerto que degradara "
                    f"por su cuenta haría que el indexado heredara una política de la "
                    f"consulta.",
                ) from None
            except APIStatusError as e:
                código = getattr(e, "status_code", None)
                if código == 400 and "healthy deployments" in str(e):
                    raise ErrorDeConfiguración(
                        f"el proxy respondió `400 · no healthy deployments` para "
                        f"{binding.modelo!r}. El modelo lleva prefijo pero el proxy "
                        f"no lo sirve: revisá el nombre contra la guía de la cátedra."
                    ) from None
                if código and código >= 500 and intento <= int(binding.reintentos):
                    time.sleep(self.backoff_base**intento)
                    último = e
                    continue
                raise FallaDeEtapa(f"HTTP {código}", str(e)) from None

            elección = respuesta.choices[0]
            razón = getattr(elección, "finish_reason", None)
            if razón == "content_filter":
                raise RechazoDeContenido(
                    "filtro de contenido",
                    "el proveedor rechazó la llamada. el contrato de los puertos lo manda como **falla de "
                    "etapa**, y es el caso de la imagen fuera de dominio: el diseño "
                    "sigue **abierto** como decisión de producto y este hito no lo cierra "
                    "— lo vuelve nombrable y contable.",
                )
            texto = (elección.message.content or "").strip()
            uso = getattr(respuesta, "usage", None)
            salida = RespuestaDeLLM(
                texto=texto,
                uso={
                    "tokens_in": getattr(uso, "prompt_tokens", 0) or 0,
                    "tokens_out": getattr(uso, "completion_tokens", 0) or 0,
                },
                muestreo_efectivo=enviado,
                modelo_efectivo=getattr(respuesta, "model", binding.modelo) or binding.modelo,
                n_intentos=intento,
            )
            if binding.esquema_de_salida != "json":
                return salida
            try:
                salida.json()
            except json.JSONDecodeError as e:
                último = e
                if intento > int(binding.reintentos):
                    raise FallaDeEtapa(
                        "JSON inválido",
                        f"el modelo devolvió algo que no parsea después de "
                        f"{intento} intento(s) ({e}). el contrato de los puertos: *reintenta hasta "
                        f"`binding.reintentos`, después **falla la etapa***. "
                        f"Primeros 200 caracteres: {texto[:200]!r}",
                    ) from None
                continue
            return salida

        raise FallaDeEtapa("reintentos agotados", str(último))

EXPLICACIÓN_VÁLIDA = {
    "obras": [
        {
            "artwork_id": "cma:109240",
            "afirmaciones": [
                {
                    "texto": "La obra es un paisaje rural pintado al óleo sobre lienzo.",
                    "tipo": "dato",
                    "refs": ["cma:109240::description::0"],
                },
                {
                    "texto": "La paleta se mantiene en verdes y grises fríos, más apagada "
                    "que la de la imagen que subiste.",
                    "tipo": "observacion",
                    "refs": [],
                },
            ],
        }
    ]
}

EXPLICACIÓN_CON_REF_COLGADA = {
    "obras": [
        {
            "artwork_id": "cma:109240",
            "afirmaciones": [
                {
                    "texto": "La obra es un paisaje rural pintado al óleo sobre lienzo.",
                    "tipo": "dato",
                    "refs": ["cma:109240::description::0"],
                },
                {
                    "texto": "La obra se expuso en París en 1890.",
                    "tipo": "dato",
                    "refs": ["cma:109240::exhibitions::0"],
                },
            ],
        }
    ]
}

EXPLICACIÓN_TODA_COLGADA = {
    "obras": [
        {
            "artwork_id": "cma:109240",
            "afirmaciones": [
                {
                    "texto": "La obra se expuso en París en 1890.",
                    "tipo": "dato",
                    "refs": ["cma:109240::exhibitions::0"],
                }
            ],
        }
    ]
}

class LLMGrabado:
    OPERACIONES = ("invocar",)

    def __init__(
        self,
        respuestas: Mapping[str, Sequence[Any]] | None = None,
        *,
        fallas: Mapping[str, Exception] | None = None,
    ) -> None:
        self._respuestas = {k: list(v) for k, v in (respuestas or {}).items()}
        self._fallas = dict(fallas or {})
        self.llamadas: list[dict] = []

    def invocar(
        self,
        binding,
        mensajes: Sequence[Mapping[str, Any]],
        imagenes: Sequence[bytes] = (),
    ) -> RespuestaDeLLM:
        _validar_modelo(binding.modelo)
        cuerpo = _cuerpo(binding, mensajes, imagenes)
        self.llamadas.append(
            {"rol": binding.rol, "n_imagenes": len(imagenes), "cuerpo": cuerpo}
        )
        if binding.rol in self._fallas:
            raise self._fallas[binding.rol]
        cola = self._respuestas.get(binding.rol)
        if not cola:
            raise FallaDeEtapa(
                "sin respuesta grabada",
                f"`LLMGrabado` no tiene respuesta para el rol {binding.rol!r}. el contrato de los puertos pide "
                f"**respuestas grabadas por rol**: registrá una en vez de devolver algo "
                f"genérico, porque un doble que improvisa hace pasar tests contra un "
                f"contrato que el adaptador real no cumple.",
            )
        salida = cola[0] if len(cola) == 1 else cola.pop(0)
        texto = salida if isinstance(salida, str) else json.dumps(salida, ensure_ascii=False)
        return RespuestaDeLLM(
            texto=texto,
            uso={
                "tokens_in": sum(len(str(m["texto"]).split()) for m in mensajes),
                "tokens_out": len(texto.split()),
            },
            muestreo_efectivo=_muestreo_enviado(cuerpo),
            modelo_efectivo=binding.modelo,
        )
