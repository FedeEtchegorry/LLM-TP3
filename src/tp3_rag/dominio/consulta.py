from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeConfiguración

PROCEDENCIAS: tuple[str, ...] = ("set", "ad_hoc")

@dataclass(frozen=True)
class PolíticaDeTrazado:
    procedencia: str
    captura_entradas: bool
    por_que: str

    @classmethod
    def de(cls, procedencia: str) -> "PolíticaDeTrazado":
        if procedencia == "set":
            return cls(
                procedencia="set",
                captura_entradas=True,
                por_que=(
                    "Las 30 consultas de evaluación se trazan **completas**: "
                    "la política de trazado`). No son dato de un tercero: son obras de nuestro propio "
                    "held-out, versionadas con el ground truth porque son el "
                    "**instrumento de medición**. Trazarlas enteras "
                    "cumple las técnicas avanzadas aisladas por medición, la tabla de tokens, costo y la figura y el criterio de costo sin tocar el compromiso de privacidad."
                ),
            )
        if procedencia == "ad_hoc":
            return cls(
                procedencia="ad_hoc",
                captura_entradas=False,
                por_que=(
                    "Una consulta ad-hoc traza **estructura, latencia y tokens, y NO "
                    "captura entradas** — ni el pedido ni la imagen. El "
                    "default de LangSmith devuelve `inputs` y `outputs` completos desde "
                    "el servidor, o sea que **persiste el dato del usuario en la nube de "
                    "un tercero**: justo lo que el registro de decisiones compromete que no pasa, en "
                    "un informe del Ej. 3 que ya está entregable."
                ),
            )
        raise ErrorDeConfiguración(
            f"`procedencia = {procedencia!r}` no es una de {list(PROCEDENCIAS)}. La "
            f"política de trazado del registro de decisiones **se deriva de la procedencia** y no se "
            f"configura: una procedencia desconocida no puede caer en un default, porque "
            f"el default es el que persiste el dato del usuario."
        )

    def censurar(self, entradas: Mapping[str, Any] | None) -> tuple[dict, int]:
        if not entradas:
            return {}, 0
        if self.captura_entradas:
            return dict(entradas), 0
        return {}, len(entradas)

    def a_json(self) -> dict:
        return {
            "procedencia": self.procedencia,
            "captura_entradas": self.captura_entradas,
            "decision": "el registro de decisiones la política de trazado",
            "por_que": self.por_que,
        }

@dataclass(frozen=True)
class PolíticaDeRetenciónDelPlan:
    procedencia: str
    retiene_pedido: bool
    por_que: str

    @classmethod
    def de(cls, procedencia: str) -> "PolíticaDeRetenciónDelPlan":
        if procedencia == "set":
            return cls(
                procedencia="set",
                retiene_pedido=True,
                por_que=(
                    "La consulta sale del `SetDeConsultas`: **no es dato de un tercero**, "
                    "es una obra de nuestro propio held-out versionada con el ground "
                    "truth porque es el **instrumento de medición**. El "
                    "plan se guarda entero para que el informe lo pueda leer, que es la "
                    "misma frontera que el registro de decisiones trazó para las trazas."
                ),
            )
        if procedencia == "ad_hoc":
            return cls(
                procedencia="ad_hoc",
                retiene_pedido=False,
                por_que=(
                    "🔴 el registro de decisiones: **el texto del pedido de un usuario real no se "
                    "guarda**. El pedido **es ingrediente del `plan_id` y eso no obliga a "
                    "guardarlo** — es la regla *«no se guarda ≠ no es ingrediente»* de "
                    "la tabla de identidades, la misma que el plan guarda el hash y no los bytes aplica a los bytes de la imagen. El reuso "
                    "del reuso del plan al barrer el peso de fusión funciona **por dirección**: el id se computa de las entradas "
                    "en el momento, no se lee del artefacto."
                ),
            )
        raise ErrorDeConfiguración(
            f"`procedencia = {procedencia!r}` no es una de {list(PROCEDENCIAS)}. Qué guarda "
            f"el plan **se deriva de la procedencia** y no se configura: una procedencia "
            f"desconocida no puede caer en un default, porque el default es el que persiste "
            f"el pedido del usuario."
        )

    def a_json(self) -> dict:
        return {
            "procedencia": self.procedencia,
            "retiene_pedido": self.retiene_pedido,
            "decision": "la política de retención (corregida el 2026-10-25)",
            "por_que": self.por_que,
        }

@dataclass(frozen=True)
class EntradaDeConsulta:
    consulta_id: str
    imagen_hash: str
    pedido: str
    procedencia: str
    datos_de_imagen: bytes = field(default=b"", compare=False, repr=False)
    artwork_id_de_la_imagen: str | None = None
    categoria: str | None = None

    def __post_init__(self) -> None:
        if self.procedencia not in PROCEDENCIAS:
            raise ErrorDeConfiguración(
                f"`procedencia = {self.procedencia!r}` no es una de {list(PROCEDENCIAS)}."
            )
        if not self.consulta_id:
            raise ErrorDeConfiguración("una consulta sin `consulta_id` no es una consulta")
        if not str(self.imagen_hash).startswith("sha256:"):
            raise ErrorDeConfiguración(
                f"`imagen_hash = {self.imagen_hash!r}` sin prefijo de algoritmo. El prefijo "
                f"es de `identidad.hash_de_bytes` y existe para que un cambio de algoritmo "
                f"lo diga en vez de colisionar."
            )
        if self.procedencia == "set" and identidad.parece_hash_corto(self.consulta_id):
            raise ErrorDeConfiguración(
                f"`consulta_id = {self.consulta_id!r}` tiene forma de `hash_corto` y la "
                f"procedencia dice `set`. **Los 30 del set se declaran, no se derivan de "
                f"los bytes** (el set de consultas, consecuencia 1): un id derivado se mueve cuando el "
                f"museo reemplaza un JPEG, y con él se huerfanan los ~540 juicios."
            )

    @property
    def política_de_trazado(self) -> PolíticaDeTrazado:
        return PolíticaDeTrazado.de(self.procedencia)

    @property
    def política_de_retención_del_plan(self) -> PolíticaDeRetenciónDelPlan:
        return PolíticaDeRetenciónDelPlan.de(self.procedencia)

    @property
    def es_del_set(self) -> bool:
        return self.procedencia == "set"

    def a_json(self, *, retener_pedido: bool = False) -> dict:
        return {
            "consulta_id": self.consulta_id,
            "imagen_hash": self.imagen_hash,
            "procedencia": self.procedencia,
            "artwork_id_de_la_imagen": self.artwork_id_de_la_imagen,
            "categoria": self.categoria,
            "pedido": self.pedido if retener_pedido else None,
            "pedido_retenido": bool(retener_pedido),
            "imagen_retenida": False,
            "nota_de_retencion": (
                "el registro de decisiones (cierra la política de retención del dato del usuario): **no se persiste nada de un "
                "tercero**. Los bytes de la imagen no se guardan nunca —no hay clave que "
                "los active— y el `pedido` solo si `[retencion] pedido = true`. Lo que se "
                "vacía son **campos**, y que estaban vaciados queda registrado: "
                "la corrida conserva su `corrida_id`, su ordinal, sus rankings y sus "
                "degradaciones, porque borrarla reasignaría el ordinal y un informe que "
                "cita la corrida #1 apuntaría a otra ejecución."
            ),
        }

    @classmethod
    def del_set(cls, consulta, *, datos_de_imagen: bytes) -> "EntradaDeConsulta":
        return cls(
            consulta_id=consulta.consulta_id,
            imagen_hash=consulta.imagen_hash,
            pedido=consulta.pedido,
            procedencia="set",
            datos_de_imagen=datos_de_imagen,
            artwork_id_de_la_imagen=consulta.artwork_id,
            categoria=consulta.categoria,
        )

    @classmethod
    def ad_hoc(cls, *, datos_de_imagen: bytes, pedido: str) -> "EntradaDeConsulta":
        if not str(pedido).strip():
            raise ErrorDeConfiguración(
                "una consulta ad-hoc sin pedido no es una consulta: el pedido es la mitad "
                "de la entrada (`Consulta` = imagen + pedido, el plan de consulta). Para una búsqueda "
                "puramente visual lo que corresponde es `α = 1`, que es C0 y es otra "
                "configuración — no un pedido vacío."
            )
        hash_bytes = identidad.hash_de_bytes(datos_de_imagen)
        return cls(
            consulta_id=identidad.consulta_id_ad_hoc(imagen_hash=hash_bytes, pedido=pedido),
            imagen_hash=hash_bytes,
            pedido=pedido,
            procedencia="ad_hoc",
            datos_de_imagen=datos_de_imagen,
        )
