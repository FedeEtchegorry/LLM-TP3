from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.plan import PlanDeConsulta
from tp3_rag.nucleo.errores import ErrorDelProyecto

CLAVE_DE_ETAPA: dict[str, str] = {
    "planificador": "plan",
    "recuperador_visual": 'fuentes_de_ranking["visual"]',
    "recuperador_textual": 'fuentes_de_ranking["texto"]',
    "fusionador": "ranking_fusionado",
    "reranker": "rerank",
    "recolector_de_evidencia": "evidencia",
    "generador": "explicacion",
    "renderizador": "salida",
}

NO_OMITIBLES: tuple[str, ...] = ("recuperador_visual", "renderizador")

ETAPAS_DE_RANKING: tuple[str, ...] = (
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "reranker",
)

ETAPAS_QUE_DEJAN_TOP5: tuple[str, ...] = (
    "recuperador_visual",
    "fusionador",
    "reranker",
)

class FormaDeclaradaRota(ErrorDelProyecto):
    pass

class SobrescrituraProhibida(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class Degradación:
    etapa: str
    causa: str
    politica: str
    planificada: bool = False

    def a_json(self) -> dict:
        return {
            "etapa": self.etapa,
            "causa": self.causa,
            "politica": self.politica,
            "planificada": self.planificada,
        }

@dataclass(frozen=True)
class Candidata:
    artwork_id: str
    rank: int
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def a_json(self) -> dict:
        return {"artwork_id": self.artwork_id, "rank": self.rank, "score": self.score}

@dataclass(frozen=True)
class EstadoDeConsulta:
    entrada: EntradaDeConsulta
    plan: PlanDeConsulta | None = None
    fuentes_de_ranking: Mapping[str, tuple[Candidata, ...]] = field(default_factory=dict)
    ranking_fusionado: tuple[Any, ...] | None = None
    rerank: tuple[Any, ...] | None = None
    top5: tuple[str, ...] | None = None
    evidencia: Mapping[str, tuple[Any, ...]] | None = None
    explicacion: tuple[Any, ...] | None = None
    salida: Any | None = None

    degradaciones: tuple[Degradación, ...] = ()
    afirmaciones_descartadas: tuple[Mapping[str, Any], ...] = ()
    fallida: Mapping[str, str] | None = None
    sellos: Mapping[str, Any] = field(default_factory=dict)

    def con(self, **claves) -> "EstadoDeConsulta":
        for clave, valor in claves.items():
            if not hasattr(self, clave):
                raise SobrescrituraProhibida(
                    f"`{clave}` no es una clave del `EstadoDeConsulta`. El estado "
                    f"es cerrado a propósito: una clave nueva es una **etapa nueva**, y "
                    f"entonces la forma declarada del artefacto necesita saber de quién es (`CLAVE_DE_ETAPA`)."
                )
            actual = getattr(self, clave)
            if clave in ("fuentes_de_ranking", "evidencia"):
                continue
            if actual is not None and actual != () and clave not in (
                "degradaciones",
                "afirmaciones_descartadas",
                "fallida",
                "sellos",
                "top5",
            ):
                raise SobrescrituraProhibida(
                    f"se rompió la no destrucción de lo ya registrado: se intentó sobrescribir `{clave}`, que ya está poblada. "
                    f"Una etapa **agrega**, no pisa. Si esto fuera legal, el reranker "
                    f"podría pisar el `score_rrf` del fusionado y **`C3−C2` y `C4−C3` "
                    f"dejarían de ser legibles del mismo artefacto** (las técnicas avanzadas aisladas por medición pide los dos)."
                )
        nuevas = dict(claves)
        if "fuentes_de_ranking" in nuevas:
            agregadas = dict(nuevas["fuentes_de_ranking"])
            repetidas = sorted(set(agregadas) & set(self.fuentes_de_ranking))
            if repetidas:
                raise SobrescrituraProhibida(
                    f"se rompió la no destrucción de lo ya registrado: la fuente de ranking {repetidas} ya estaba registrada. "
                    f"Una rama es **una fuente nombrada**: pisarla borraría sus rangos de "
                    f"origen, que son la evidencia que las técnicas avanzadas aisladas por medición pide."
                )
            nuevas["fuentes_de_ranking"] = {**dict(self.fuentes_de_ranking), **agregadas}
        if "evidencia" in nuevas and self.evidencia:
            nuevas["evidencia"] = {**dict(self.evidencia), **dict(nuevas["evidencia"])}
        return replace(self, **nuevas)

    def ids_propuestos(self) -> frozenset[str]:
        propuestos: set[str] = set(self.top5 or ())
        for candidatas in self.fuentes_de_ranking.values():
            propuestos.update(c.artwork_id for c in candidatas)
        for fila in self.ranking_fusionado or ():
            propuestos.add(getattr(fila, "artwork_id", None) or fila["artwork_id"])
        for fila in self.rerank or ():
            propuestos.add(
                getattr(fila, "artwork_id", None)
                or (fila["artwork_id"] if isinstance(fila, Mapping) else None)
            )
        return frozenset(p for p in propuestos if p)

    def proyectar_top5(self, nuevo: Sequence[str], *, por: str) -> "EstadoDeConsulta":
        propuestos = self.ids_propuestos()
        intrusas = sorted(set(nuevo) - propuestos) if propuestos else []
        if intrusas:
            raise SobrescrituraProhibida(
                f"`{por}` puso en el `top5` obras que **ninguna etapa de ranking propuso**: "
                f"{intrusas}. Lo propuesto en esta consulta son {len(propuestos)} obras "
                f"(fuentes de ranking, ranking fusionado, rerank y el `top5` previo). la no destrucción de lo ya registrado "
                f"permite **proyectar el propio ranking**; traer una obra de la nada haría "
                f"que el `%` de ítems sin juzgar de la tabla de ablación la contara como no juzgada sin que "
                f"nadie supiera de dónde salió."
            )
        return replace(self, top5=tuple(nuevo))

    def degradar(self, etapa: str, causa: str, politica: str, *, planificada: bool = False):
        return replace(
            self,
            degradaciones=self.degradaciones
            + (Degradación(etapa, causa, politica, planificada),),
        )

    def marcar_fallida(self, etapa: str, obligacion: str) -> "EstadoDeConsulta":
        return replace(self, fallida={"etapa": etapa, "obligacion": obligacion})

    def descartar_afirmaciones(self, descartes: Sequence[Mapping[str, Any]]):
        return replace(
            self, afirmaciones_descartadas=self.afirmaciones_descartadas + tuple(descartes)
        )

    @property
    def consulta_visual(self) -> str:
        return self.plan.consulta_visual if self.plan else self.entrada.pedido

    @property
    def consulta_conceptual(self) -> str:
        return self.plan.consulta_conceptual if self.plan else self.entrada.pedido

    @property
    def degradó(self) -> bool:
        return bool(self.degradaciones)

    @property
    def degradaciones_no_planificadas(self) -> tuple[Degradación, ...]:
        return tuple(d for d in self.degradaciones if not d.planificada)

    def etapas_con_clave_poblada(self) -> set[str]:
        pobladas: set[str] = set()
        if self.plan is not None:
            pobladas.add("planificador")
        if "visual" in self.fuentes_de_ranking:
            pobladas.add("recuperador_visual")
        if "texto" in self.fuentes_de_ranking:
            pobladas.add("recuperador_textual")
        if self.ranking_fusionado is not None:
            pobladas.add("fusionador")
        if self.rerank is not None:
            pobladas.add("reranker")
        if self.evidencia is not None:
            pobladas.add("recolector_de_evidencia")
        if self.explicacion is not None:
            pobladas.add("generador")
        if self.salida is not None:
            pobladas.add("renderizador")
        return pobladas

    def ultimo_ranking(self, cadena: Sequence[str]) -> tuple[str, tuple[str, ...]] | None:
        degradadas = {d.etapa for d in self.degradaciones}
        for etapa in reversed([e for e in cadena if e in ETAPAS_QUE_DEJAN_TOP5]):
            if etapa in degradadas:
                continue
            ids = self._ranking_de(etapa)
            if ids is not None:
                return etapa, ids
        return None

    def _ranking_de(self, etapa: str) -> tuple[str, ...] | None:
        if etapa == "recuperador_visual":
            fuente = self.fuentes_de_ranking.get("visual")
            return tuple(c.artwork_id for c in fuente) if fuente else None
        if etapa == "recuperador_textual":
            fuente = self.fuentes_de_ranking.get("texto")
            return tuple(c.artwork_id for c in fuente) if fuente else None
        if etapa == "fusionador":
            if self.ranking_fusionado is None:
                return None
            return tuple(x.artwork_id for x in self.ranking_fusionado)
        if etapa == "reranker":
            if self.rerank is None:
                return None
            return tuple(x["artwork_id"] for x in self.rerank)
        return None

    def verificar_et6(self, cadena: Sequence[str], *, top_k: int = 5) -> dict:
        cadena = tuple(cadena)
        pobladas = self.etapas_con_clave_poblada()
        degradadas = {d.etapa for d in self.degradaciones}
        fallida = (self.fallida or {}).get("etapa")

        mudas = [
            e
            for e in cadena
            if e not in pobladas and e not in degradadas and e != fallida
        ]
        if mudas:
            raise FormaDeclaradaRota(
                f"la forma declarada del artefacto, primera dirección: las etapas {mudas} **están en la cadena** y no "
                f"dejaron ni su clave ({[CLAVE_DE_ETAPA[e] for e in mudas]}) ni una entrada "
                f"en `degradaciones`. Una etapa que **no hizo nada y no registró nada** "
                f"produce un artefacto con la forma de una omisión legítima, y **la "
                f"C3 que corre como C2 entra por esa puerta** — la que la precondición declarada no mira, "
                f"porque la cadena estaba bien armada. Es además el detector de la ablación por configuración."
            )

        intrusas = sorted(pobladas - set(cadena))
        if intrusas:
            raise FormaDeclaradaRota(
                f"la forma declarada del artefacto, segunda dirección: las etapas {intrusas} **no están en la cadena** "
                f"y dejaron su clave ({[CLAVE_DE_ETAPA[e] for e in intrusas]}). El "
                f"`config_id` declara {list(cadena)}: un artefacto con una clave de más "
                f"dice ser una configuración y es otra, y la fila de la tabla saldría con "
                f"la etiqueta equivocada."
            )

        verificación_del_top5: dict[str, Any] = {"verificado": False}
        último = self.ultimo_ranking(cadena)
        if self.top5 is not None and último is not None:
            etapa, ids = último
            esperado = tuple(ids[:top_k])
            if tuple(self.top5) != esperado:
                raise FormaDeclaradaRota(
                    f"la forma declarada del artefacto · el `top5` no son los primeros {top_k} del ranking que dejó "
                    f"**{etapa}**, que es la última etapa de ranking de la cadena que no "
                    f"degradó. Declarado {list(self.top5)}, esperado {list(esperado)}. "
                    f"`top5` **no es la clave de ninguna etapa**: *toda etapa de "
                    f"ranking lo deja como proyección de su propio ranking*, así que la forma declarada del artefacto "
                    f"no lo atribuye a nadie — lo verifica."
                )
            verificación_del_top5 = {
                "verificado": True,
                "contra": etapa,
                "n": len(self.top5),
            }

        return {
            "cumple_et6": True,
            "cadena_declarada": list(cadena),
            "claves_pobladas": sorted(pobladas),
            "etapas_degradadas": sorted(degradadas),
            "etapa_fallida": fallida,
            "top5": verificación_del_top5,
            "nota": (
                "Las **dos** direcciones son error de corrida. la precondición declarada mira la cadena "
                "**antes** de correr y la degradación registrada registra lo que falló; sin la forma declarada del artefacto, una etapa muda "
                "pasa por omisión legítima. Lo verifica el ejecutor al cerrar y lo vuelve "
                "a verificar `MétricasDeRanking` al leer."
            ),
        }

    def a_json(self, *, retener_pedido: bool = False) -> dict:
        return {
            "entrada": self.entrada.a_json(retener_pedido=retener_pedido),
            "plan": self.plan.a_json() if self.plan else None,
            "fuentes_de_ranking": {
                nombre: [c.a_json() for c in candidatas]
                for nombre, candidatas in self.fuentes_de_ranking.items()
            },
            "ranking_fusionado": (
                [x.a_json() for x in self.ranking_fusionado]
                if self.ranking_fusionado is not None
                else None
            ),
            "rerank": list(self.rerank) if self.rerank is not None else None,
            "top5": list(self.top5) if self.top5 is not None else None,
            "evidencia": (
                {
                    obra: [f.a_json() for f in fragmentos]
                    for obra, fragmentos in self.evidencia.items()
                }
                if self.evidencia is not None
                else None
            ),
            "explicacion": (
                [e.a_json() for e in self.explicacion]
                if self.explicacion is not None
                else None
            ),
            "salida": self.salida.a_json() if self.salida is not None else None,
            "degradaciones": [d.a_json() for d in self.degradaciones],
            "afirmaciones_descartadas": [dict(d) for d in self.afirmaciones_descartadas],
            "fallida": dict(self.fallida) if self.fallida else None,
            "sellos": dict(self.sellos),
        }

class Etapa:
    NOMBRE: str = ""
    PRECONDICIONES: tuple[str, ...] = ()
    CLAVES_DE_CONFIGURACIÓN: tuple[str, ...] = ()
    DEGRADA: bool = True
    POLÍTICA: str = ""

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        return f"<{type(self).__name__} {self.NOMBRE}>"
