from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from tp3_rag.dominio.consulta import PROCEDENCIAS, PolíticaDeRetenciónDelPlan
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDelProyecto

VERSIÓN_DEL_PLAN = "1"

CLAVES_OBLIGATORIAS = (
    "artefacto",
    "version_del_plan",
    "plan_id",
    "procedencia",
    "imagen_hash",
    "consulta_visual",
    "consulta_conceptual",
    "pl3",
    "binding",
    "uso",
)

CLÁUSULAS_DE_PL3: tuple[str, ...] = (
    "no_vacias",
    "distintas_del_pedido_crudo",
    "distintas_entre_si",
)

def resumen_de_pl3(planes) -> dict:
    planes = tuple(planes)
    conteo: dict = {"de": len(planes)}
    for cláusula in CLÁUSULAS_DE_PL3:
        conteo[cláusula] = sum(1 for p in planes if p.pl3.get(cláusula))
    conteo["las_tres"] = sum(1 for p in planes if p.cumple_pl3)
    return conteo

class PlanInválido(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class PlanDeConsulta:
    plan_id: str
    consulta_id: str
    imagen_hash: str
    pedido: str
    consulta_visual: str
    consulta_conceptual: str
    binding: Mapping[str, Any]
    uso: Mapping[str, int]
    procedencia: str = "ad_hoc"
    n_intentos: int = 1
    pl3: Mapping[str, bool] = field(default_factory=dict)

    reusado: bool = False
    produccion_descartada: bool = False

    def __post_init__(self) -> None:
        if self.procedencia not in PROCEDENCIAS:
            raise PlanInválido(
                f"`procedencia = {self.procedencia!r}` no es una de {list(PROCEDENCIAS)}. "
                f"El plan la necesita porque decide qué se guarda."
            )
        if not self.pl3:
            object.__setattr__(self, "pl3", self.evaluar_pl3())

    def evaluar_pl3(self) -> dict:
        visual = str(self.consulta_visual).strip()
        conceptual = str(self.consulta_conceptual).strip()
        crudo = str(self.pedido).strip()
        return {
            "no_vacias": bool(visual) and bool(conceptual),
            "distintas_del_pedido_crudo": bool(crudo) and crudo not in (visual, conceptual),
            "distintas_entre_si": bool(visual) and visual != conceptual,
        }

    def validar(self) -> None:
        veredicto = self.evaluar_pl3()
        if not veredicto["no_vacias"]:
            raise PlanInválido(
                "primera cláusula: una de las dos consultas está vacía. Si el modelo "
                "devuelve una vacía o las devuelve iguales, es falla de etapa y aplica "
                "su política de degradación — que para el planificador es pedido crudo "
                "al text encoder, o sea el comportamiento del pipeline base."
            )
        if not veredicto["distintas_entre_si"]:
            raise PlanInválido(
                "las tres cláusulas de las consultas del plan, tercera cláusula: `consulta_visual` y `consulta_conceptual` son "
                "**iguales**. Es la que **distingue *expansion* de *rewriting***: sin ella, "
                "dos reescrituras idénticas pasarían por expansión, y las técnicas avanzadas aisladas por medición reclama las "
                "dos técnicas con el artefacto que las evidencia."
            )
        if not veredicto["distintas_del_pedido_crudo"]:
            raise PlanInválido(
                "las tres cláusulas de las consultas del plan, segunda cláusula: una de las dos consultas derivadas es igual al "
                "**pedido crudo**. Que difieran del pedido *es* el *rewriting* que las técnicas avanzadas aisladas por medición "
                "reclama; si no difieren, el pre-retrieval no hizo nada y **C2 correría "
                "como C1 con una llamada de más**."
            )

    @property
    def cumple_pl3(self) -> bool:
        return all(bool(self.pl3.get(c)) for c in CLÁUSULAS_DE_PL3)

    @property
    def política_de_retención(self) -> PolíticaDeRetenciónDelPlan:
        return PolíticaDeRetenciónDelPlan.de(self.procedencia)

    def contenido_del_artefacto(self) -> dict:
        política = self.política_de_retención
        return {
            "artefacto": "PlanDeConsulta",
            "version_del_plan": VERSIÓN_DEL_PLAN,
            "clase_de_identidad": "1/2",
            "plan_id": self.plan_id,
            "consulta_id": self.consulta_id,
            "procedencia": self.procedencia,
            "imagen_hash": self.imagen_hash,
            "pedido": self.pedido if política.retiene_pedido else None,
            "pedido_retenido": política.retiene_pedido,
            "imagen_retenida": False,
            "consulta_visual": self.consulta_visual,
            "consulta_conceptual": self.consulta_conceptual,
            "pl3": dict(self.pl3),
            "cumple_pl3": self.cumple_pl3,
            "binding": dict(self.binding),
            "uso": dict(self.uso),
            "n_intentos": self.n_intentos,
            "nota_de_identidad": (
                "Artefacto híbrido: se direcciona como caché de entradas —cinco "
                "corridas de un barrido del peso de fusión reusando un plan— y su "
                "contenido es la salida no determinista de un LLM. Por eso no se le "
                "puede pedir que el mismo id determine el mismo contenido, y lo que rige "
                "es escritura única. El peso de fusión, N, la estrategia de fusión y el "
                "variante_id NO son ingredientes. El prompt más su versión, el modelo y "
                "el muestreo SÍ."
            ),
            "nota_de_retencion": política.a_json(),
            "nota_de_pl3": (
                "Las **tres** cláusulas van escritas porque son **tres cosas**: que las dos "
                "consultas existan, que difieran del pedido crudo (*rewriting*) y que "
                "difieran entre sí (lo que distingue *expansion* de *rewriting*). El "
                "veredicto viaja adentro para que las tres se puedan contar también "
                "sobre una consulta ad-hoc, cuyo pedido no se guarda."
            ),
        }

    def a_json(self) -> dict:
        return {
            **self.contenido_del_artefacto(),
            "reusado": self.reusado,
            "produccion_descartada": self.produccion_descartada,
            "nota_de_pl5": (
                "**la escritura única del plan**: un `plan_id` se escribe **una sola vez**; si ya existe se **lee** "
                "y la corrida registra si lo **produjo o lo reusó**. Sin ese registro no "
                "habría forma de saber **qué corrida pagó la llamada que la tabla de tokens, costo y la figura cuenta**."
            ),
        }

    @classmethod
    def desde_artefacto(cls, contenido: Mapping[str, Any]) -> "PlanDeConsulta":
        return cls(
            plan_id=contenido["plan_id"],
            consulta_id=contenido.get("consulta_id") or "",
            imagen_hash=contenido["imagen_hash"],
            pedido=contenido.get("pedido") or "",
            consulta_visual=contenido["consulta_visual"],
            consulta_conceptual=contenido["consulta_conceptual"],
            binding=dict(contenido.get("binding") or {}),
            uso=dict(contenido.get("uso") or {}),
            procedencia=contenido.get("procedencia") or "ad_hoc",
            n_intentos=int(contenido.get("n_intentos") or 1),
            pl3=dict(contenido.get("pl3") or {}),
        )

class AlmacenDePlanes:
    def __init__(self, almacén, raíz: str | Path) -> None:
        self.almacén = almacén
        self.raíz = Path(raíz)

    @staticmethod
    def ruta_de(raíz: str | Path, plan_id: str) -> Path:
        return Path(raíz) / f"{plan_id}.json"

    def ruta(self, plan_id: str) -> Path:
        return self.ruta_de(self.raíz, plan_id)

    def existe(self, plan_id: str) -> bool:
        return self.almacén.existe(self.ruta(plan_id))

    def leer(self, plan_id: str) -> PlanDeConsulta:
        contenido = self.almacén.leer_json(
            self.ruta(plan_id), claves_obligatorias=CLAVES_OBLIGATORIAS
        )
        if contenido["plan_id"] != plan_id:
            raise ErrorDeArtefacto(
                f"el plan {plan_id}.json declara adentro `plan_id = "
                f"{contenido['plan_id']!r}`. El nombre y el contenido no pueden discrepar: "
                f"la identidad va en el nombre."
            )
        return PlanDeConsulta.desde_artefacto(contenido)

    def leer_si_existe(self, plan_id: str) -> PlanDeConsulta | None:
        return self.leer(plan_id) if self.existe(plan_id) else None

    def existentes(self) -> tuple[str, ...]:
        carpeta = self.almacén.ruta(self.raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.json")))

    def todos(self) -> tuple[PlanDeConsulta, ...]:
        return tuple(self.leer(pid) for pid in self.existentes())

    def obtener_o_producir(
        self, plan_id: str, producir: Callable[[], PlanDeConsulta]
    ) -> PlanDeConsulta:
        existente = self.leer_si_existe(plan_id)
        if existente is not None:
            return replace(existente, reusado=True)

        plan = producir()
        if plan.plan_id != plan_id:
            raise ErrorDelProyecto(
                f"el planificador produjo un plan con `plan_id = {plan.plan_id!r}` y se le "
                f"había pedido {plan_id!r}. El id se computa antes de llamar —es función "
                f"de las entradas, no de la salida—, así que esto es un bug del llamador "
                f"y no una respuesta rara del modelo."
            )
        ruta = self.ruta(plan_id)
        try:
            self.almacén.reservar(ruta)
        except ErrorDeArtefacto:
            if not self.existe(plan_id):
                raise
            return replace(self.leer(plan_id), reusado=True, produccion_descartada=True)
        self.almacén.escribir_json(ruta, plan.contenido_del_artefacto())
        return replace(plan, reusado=False)
