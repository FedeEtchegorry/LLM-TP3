from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDeConfiguración

ROLES: tuple[str, ...] = ("planificador", "reranker", "generador", "juez")

HITO_DEL_ROL = {
    "planificador": " (C2 · pre-retrieval con LLM) — existe",
    "reranker": " (C4 · reranking multimodal, 1 candidato por llamada) — existe",
    "generador": " (C1 = v1) — existe",
    "juez": " (Módulo C) · y su binding solo es construible desde ahí (un tipo de binding por rol)",
}

ESQUEMAS_DE_SALIDA: tuple[str, ...] = ("json", "texto")

@dataclass(frozen=True)
class Muestreo:
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None = None

    def a_json(self) -> dict:
        d = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    @classmethod
    def desde_configuración(cls, d: Mapping[str, Any], *, clave: str) -> "Muestreo":
        faltantes = sorted({"temperature", "top_p", "max_tokens"} - set(d))
        if faltantes:
            raise ErrorDeConfiguración(
                f"`{clave}` no declara {faltantes}. El muestreo va explícito, fijado y "
                f"registrado: un default del SDK que no se declare haría que el "
                f"artefacto registre 'lo configurado' en vez de 'lo usado', que es la "
                f"forma elegante de no registrar nada."
            )
        semilla = d.get("seed")
        return cls(
            temperature=float(d["temperature"]),
            top_p=float(d["top_p"]),
            max_tokens=int(d["max_tokens"]),
            seed=int(semilla) if semilla is not None else None,
        )

@dataclass(frozen=True)
class _Binding:
    ROL: str | None = None
    CLAVE_DE_CONFIGURACIÓN: str | None = None

    modelo: str = ""
    prompt_id: str = ""
    version_del_prompt: str = ""
    muestreo: Muestreo | None = None
    esquema_de_salida: str = "json"
    reintentos: int = 2

    _puerto: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self) is _Binding or self.ROL is None:
            raise ErrorDeConfiguración(
                "`_Binding` no se instancia: los cuatro bindings son tipos distintos "
                "con claves de configuración distintas, y el del juez solo es "
                "construible desde el módulo de evaluación. Un binding genérico "
                "parametrizado por rol en runtime es exactamente el atajo que esto "
                "cierra — y con él, que el juez no sea el reranker volvería a ser "
                "disciplina en vez de estructura."
            )
        if self.ROL not in ROLES:
            raise ErrorDeConfiguración(
                f"`ROL = {self.ROL!r}` no es uno de los cuatro roles: {list(ROLES)}."
            )
        if self.esquema_de_salida not in ESQUEMAS_DE_SALIDA:
            raise ErrorDeConfiguración(
                f"`{self.CLAVE_DE_CONFIGURACIÓN}.esquema_de_salida` = "
                f"{self.esquema_de_salida!r} no es uno de {list(ESQUEMAS_DE_SALIDA)}."
            )
        if not self.modelo:
            raise ErrorDeConfiguración(
                f"`{self.CLAVE_DE_CONFIGURACIÓN}.modelo` está vacío. Se declara por rol "
                f"y el prefijo `openai/` es obligatorio: el proxy rechaza el nombre "
                f"pelado con `400 · no healthy deployments`."
            )
        if not self.prompt_id or not self.version_del_prompt:
            raise ErrorDeConfiguración(
                f"`{self.CLAVE_DE_CONFIGURACIÓN}` no declara `prompt_id` + "
                f"`version_del_prompt`. **El prompt es parte de la identidad del binding, "
                f"no un argumento de llamada**: sin versión, cambiar el texto del "
                f"prompt dejaría dos planes distintos compartiendo `plan_id`."
            )
        if self.muestreo is None:
            raise ErrorDeConfiguración(
                f"`{self.CLAVE_DE_CONFIGURACIÓN}.muestreo` falta. No hay default: el muestreo fijado, registrado y con su variabilidad medida."
            )

    @property
    def rol(self) -> str:
        return self.ROL  # type: ignore[return-value]

    def identidad(self) -> dict:
        return {
            "rol": self.rol,
            "modelo": self.modelo,
            "prompt_id": self.prompt_id,
            "version_del_prompt": self.version_del_prompt,
            "muestreo": self.muestreo.a_json(),  # type: ignore[union-attr]
        }

    def a_json(self) -> dict:
        return {
            **self.identidad(),
            "esquema_de_salida": self.esquema_de_salida,
            "reintentos": self.reintentos,
            "clave_de_configuracion": self.CLAVE_DE_CONFIGURACIÓN,
            "nota": (
                "el binding como único camino al puerto: el binding es el **único camino al puerto** y el prompt es parte de "
                "su identidad, no un argumento. la atribución de cada llamada a su rol: la llamada se atribuye a `rol`, no al "
                "nombre de la etapa. El `base_url` **no** es ingrediente: sale de "
                "`[proveedor]` (que el host del proxy es obligatorio, )."
            ),
        }

    def invocar(self, mensajes: Sequence[Mapping[str, Any]], imagenes: Sequence[bytes] = ()):
        if self._puerto is None:
            raise ErrorDeConfiguración(
                f"el binding de `{self.rol}` no tiene puerto. Se construye con "
                f"`desde_configuración(cfg, puerto=…, observabilidad=…)`: el puerto va "
                f"**adentro del binding** porque el binding como único camino al puerto dice que es el único camino, y un "
                f"binding sin puerto es un binding que una etapa no puede usar — que es "
                f"mejor que una etapa que puede usar el puerto sin binding."
            )
        return self._puerto.invocar(self, mensajes, imagenes)

    @classmethod
    def desde_configuración(cls, cfg, *, puerto=None, observabilidad=None):
        clave = cls.CLAVE_DE_CONFIGURACIÓN
        d = cfg[clave]
        envuelto = puerto
        if puerto is not None and observabilidad is not None:
            envuelto = observabilidad.instrumentar(puerto, rol=cls.ROL)
        return cls(
            modelo=d["modelo"],
            prompt_id=d["prompt_id"],
            version_del_prompt=str(d["version_del_prompt"]),
            muestreo=Muestreo.desde_configuración(d["muestreo"], clave=f"{clave}.muestreo"),
            esquema_de_salida=d["esquema_de_salida"],
            reintentos=int(d["reintentos"]),
            _puerto=envuelto,
        )

@dataclass(frozen=True)
class BindingDelGenerador(_Binding):
    ROL: str = "generador"
    CLAVE_DE_CONFIGURACIÓN: str = "bindings.generador"

@dataclass(frozen=True)
class BindingDelPlanificador(_Binding):
    ROL: str = "planificador"
    CLAVE_DE_CONFIGURACIÓN: str = "bindings.planificador"

@dataclass(frozen=True)
class BindingDelReranker(_Binding):
    ROL: str = "reranker"
    CLAVE_DE_CONFIGURACIÓN: str = "bindings.reranker"

def binding_faltante(rol: str) -> str:
    if rol not in ROLES:
        return f"{rol!r} no es uno de los cuatro roles: {list(ROLES)}"
    return (
        f"el binding de `{rol}` no existe todavía: llega con {HITO_DEL_ROL[rol]}. "
        f"El pipeline base tiene que estar completo antes de agregar nada avanzado, "
        f"así que esto no es una pieza que falte sino una que todavía no corresponde."
    )
