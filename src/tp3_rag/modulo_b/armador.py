from __future__ import annotations

from typing import Mapping, Sequence

from tp3_rag.dominio.binding import (
    BindingDelGenerador,
    BindingDelPlanificador,
    BindingDelReranker,
    binding_faltante,
)
from tp3_rag.dominio.salida import FormatoDeSalida
from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA, NO_OMITIBLES, Etapa
from tp3_rag.modulo_b.fusionador import Fusionador
from tp3_rag.modulo_b.generador import Generador
from tp3_rag.modulo_b.planificador import PlanificadorDeConsulta
from tp3_rag.modulo_b.recolector_de_evidencia import RecolectorDeEvidencia
from tp3_rag.modulo_b.recuperador_textual import RecuperadorTextual
from tp3_rag.modulo_b.recuperador_visual import RecuperadorVisual
from tp3_rag.modulo_b.renderizador import RenderizadorDeSalida
from tp3_rag.modulo_b.reranker import Reranker
from tp3_rag.nucleo.errores import ErrorDeConfiguración

CLAVES_DE_LA_ENTRADA: tuple[str, ...] = ("entrada",)

PRODUCE: dict[str, tuple[str, ...]] = {
    "planificador": ("plan",),
    "recuperador_visual": ("fuentes_de_ranking", "top5"),
    "recuperador_textual": ("fuentes_de_ranking",),
    "fusionador": ("ranking_fusionado", "top5"),
    "reranker": ("rerank", "top5"),
    "recolector_de_evidencia": ("evidencia",),
    "generador": ("explicacion",),
    "renderizador": ("salida",),
}

OMITIBLES_POR_CONFIGURACIÓN: dict[str, frozenset[str]] = {
    "C0": frozenset(),
    "C1": frozenset({"recolector_de_evidencia", "generador"}),
    "C2": frozenset({"planificador", "recolector_de_evidencia", "generador"}),
    "C3": frozenset(
        {
            "planificador",
            "recuperador_textual",
            "fusionador",
            "recolector_de_evidencia",
            "generador",
        }
    ),
    "C4": frozenset(
        {
            "planificador",
            "recuperador_textual",
            "fusionador",
            "reranker",
            "recolector_de_evidencia",
            "generador",
        }
    ),
}

CONSTRUIBLES: tuple[str, ...] = (
    "planificador",
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "reranker",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
)

CLASE_DE_ETAPA: dict[str, type[Etapa]] = {
    "planificador": PlanificadorDeConsulta,
    "recuperador_visual": RecuperadorVisual,
    "recuperador_textual": RecuperadorTextual,
    "fusionador": Fusionador,
    "reranker": Reranker,
    "recolector_de_evidencia": RecolectorDeEvidencia,
    "generador": Generador,
    "renderizador": RenderizadorDeSalida,
}

HITO_DE_LA_ETAPA: dict[str, str] = {}

class CadenaImposible(ErrorDeConfiguración):
    pass

class Pipeline:
    def __init__(
        self,
        etapas: Sequence[Etapa],
        *,
        configuracion: str,
        plan_de_falla: Mapping[str, str],
        top_k: int,
    ) -> None:
        self.etapas = tuple(etapas)
        self.configuracion = configuracion
        self.plan_de_falla = dict(plan_de_falla)
        self.top_k = int(top_k)

    @property
    def cadena(self) -> tuple[str, ...]:
        return tuple(e.NOMBRE for e in self.etapas)

    def a_json(self) -> dict:
        return {
            "configuracion": self.configuracion,
            "cadena": list(self.cadena),
            "claves_por_etapa": {e: CLAVE_DE_ETAPA[e] for e in self.cadena},
            "plan_de_falla": dict(self.plan_de_falla),
            "top_k": self.top_k,
            "etapas": [e.a_json() for e in self.etapas if hasattr(e, "a_json")],
            "nota": (
                "**Las cinco configuraciones son subsecuencias de una sola cadena** (la clausura de la etapa, "
                "la ablación por configuración), no cinco pipelines. la precondición declarada las rechaza antes de correr por dos "
                "caminos: precondiciones insatisfechas, y una cadena que no coincide con la "
                "fila que `[consulta] configuracion` dice ser."
            ),
        }

class ArmadorDePipeline:
    def __init__(
        self,
        cfg,
        *,
        puerto_de_imagen=None,
        indice=None,
        puerto_de_llm=None,
        embebedor_de_texto=None,
        almacen_de_planes=None,
        bytes_de_la_imagen=None,
        observabilidad=None,
    ) -> None:
        self.cfg = cfg
        self.puerto_de_imagen = puerto_de_imagen
        self.indice = indice
        self.puerto_de_llm = puerto_de_llm
        self.embebedor_de_texto = embebedor_de_texto
        self.almacen_de_planes = almacen_de_planes
        self.bytes_de_la_imagen = bytes_de_la_imagen
        self.observabilidad = observabilidad

    def validar(self) -> tuple[str, ...]:
        cadena = tuple(self.cfg["consulta.etapas"])
        configuración = self.cfg["consulta.configuracion"]
        self._validar_nombres(cadena)
        self._validar_no_omitibles(cadena)
        self._validar_etiqueta(cadena, configuración)
        self._validar_precondiciones(cadena)
        self._validar_plan_de_falla(cadena)
        return cadena

    def _validar_nombres(self, cadena: Sequence[str]) -> None:
        desconocidas = [e for e in cadena if e not in CLAVE_DE_ETAPA]
        if desconocidas:
            raise CadenaImposible(
                f"`consulta.etapas` nombra {desconocidas}, que no son etapas del contrato de etapa. Las "
                f"ocho son {list(CLAVE_DE_ETAPA)}. **Una clave desconocida es un error** "
                f": un typo en `c3.toml` no puede terminar corriendo C2 en silencio."
            )
        repetidas = sorted({e for e in cadena if list(cadena).count(e) > 1})
        if repetidas:
            raise CadenaImposible(
                f"`consulta.etapas` repite {repetidas}. Cada etapa tiene **una** clave "
                f"propia, así que la segunda intentaría sobrescribirla y la no destrucción de lo ya registrado la "
                f"rechazaría **en runtime** — a mitad de la consulta 1 de 30, en vez de "
                f"antes de correr."
            )

    def _validar_no_omitibles(self, cadena: Sequence[str]) -> None:
        faltan = [e for e in NO_OMITIBLES if e not in cadena]
        if faltan:
            raise CadenaImposible(
                f"la cadena no incluye {faltan}, y **esas dos no se omiten nunca**: "
                f"el `recuperador_visual` es **el piso** (la asimetría del piso de recuperación — *un embedding CLIP de la "
                f"imagen + una consulta a `artworks`*; si falla, no hay respuesta) y el "
                f"`renderizador` **es la salida**. *Omitible no es lo mismo que etapa*: el "
                f"resto sí, y eso es la omisión de etapas en runtime."
            )

    def _validar_etiqueta(self, cadena: Sequence[str], configuracion: str) -> None:
        if configuracion not in OMITIBLES_POR_CONFIGURACIÓN:
            raise CadenaImposible(
                f"`consulta.configuracion = {configuracion!r}` no es una fila de la tabla de "
                f"el diseño de la evaluación: son {sorted(OMITIBLES_POR_CONFIGURACIÓN)}. La etiqueta no es "
                f"decorativa — es contra lo que la precondición declarada compara la cadena."
            )
        esperadas = OMITIBLES_POR_CONFIGURACIÓN[configuracion]
        declaradas = frozenset(cadena) - frozenset(NO_OMITIBLES)
        if declaradas != esperadas:
            sobran = sorted(declaradas - esperadas)
            faltan = sorted(esperadas - declaradas)
            raise CadenaImposible(
                f"la cadena dice ser **{configuracion}** y no coincide con la fila que "
                f"el diseño de la evaluación define. Sobran {sobran}; faltan {faltan}. "
                f"Es el bug que la precondición declarada existe para impedir: *una C3 mal configurada —sin rama "
                f"de texto— que corre en silencio **como C2** y entra a la tabla con la "
                f"etiqueta equivocada*. Si lo que querés es un **brazo declarado** y no una "
                f"fila de la tabla, declaralo con su propia etiqueta en vez de prestarle la "
                f"de otra fila."
            )

    def _validar_precondiciones(self, cadena: Sequence[str]) -> None:
        disponibles = set(CLAVES_DE_LA_ENTRADA)
        for etapa in cadena:
            necesarias = _precondiciones_de(etapa)
            faltan = sorted(set(necesarias) - disponibles)
            if faltan:
                raise CadenaImposible(
                    f"la precondición declarada: la etapa `{etapa}` necesita {faltan} del estado y **ninguna "
                    f"etapa anterior de la cadena lo produce** (disponibles: "
                    f"{sorted(disponibles)}). Se rechaza **antes de correr**, que es la "
                    f"política de la clase 1 de falla: descubrirlo en runtime sería "
                    f"una degradación que parece legítima."
                )
            disponibles.update(PRODUCE.get(etapa, ()))

    def _validar_plan_de_falla(self, cadena: Sequence[str]) -> None:
        crudo = self.cfg["consulta.plan_de_falla"]
        if not isinstance(crudo, Mapping):
            raise CadenaImposible(
                f"`consulta.plan_de_falla` llegó como {type(crudo).__name__} "
                f"({crudo!r:.60}) y la regla de qué corridas agregan a la tabla lo declara como **una tabla** "
                f"`{{ etapa → modo }}`. Una tabla entera **no es expresable por `--set`** "
                f"—`_interpretar` convierte números, booleanos y listas, no tablas—, y no "
                f"hace falta que lo sea: el plan de falla es **tabla abierta**, así "
                f"que se pasa **una clave por bandera**:\n"
                f" --set consulta.plan_de_falla.reranker=excepcion\n"
                f"Los modos son `excepcion` y `vacio`, y el armador verifica además que la "
                f"etapa **esté en la cadena** y que **degrade**."
            )
        plan = dict(crudo)
        fuera = sorted(set(plan) - set(cadena))
        if fuera:
            raise CadenaImposible(
                f"`consulta.plan_de_falla` pide inyectar falla en {fuera}, que **no están en "
                f"la cadena**. el criterio de respuesta garantizada se verifica *"
                f"corriendo las 30 consultas una vez por etapa con fallback*: una "
                f"etapa que no corre no puede fallar, y el reporte diría que se inyectó una "
                f"falla que nunca pasó."
            )
        for etapa in plan:
            if etapa in NO_OMITIBLES:
                raise CadenaImposible(
                    f"`consulta.plan_de_falla` pide inyectar falla en `{etapa}`, que **no "
                    f"degrada** (la asimetría del piso de recuperación / las obligaciones de la cita). los criterios de entrega el criterio de respuesta garantizada compromete 30/30 contra "
                    f"falla inyectada en *las etapas post-retrieval que **tienen "
                    f"fallback***, y estas dos no están en esa lista: el piso porque *si "
                    f"falla, no hay respuesta*, y el renderizador porque sus condiciones de "
                    f"falla son **obligaciones**. Inyectar acá no mide el criterio de respuesta garantizada: mide otra cosa "
                    f"y la reporta como el criterio de respuesta garantizada."
                )

    def armar(self) -> Pipeline:
        cadena = self.validar()
        no_construibles = [e for e in cadena if e not in CONSTRUIBLES]
        if no_construibles:  # pragma: no cover - las ocho son construibles
            detalle = "; ".join(
                f"`{e}` → {HITO_DE_LA_ETAPA.get(e, 'hito no declarado')}"
                for e in no_construibles
            )
            raise CadenaImposible(
                f"la cadena pide etapas que **todavía no existen**: {detalle}. "
                f"el enunciado exige que **v1 esté completo antes** de "
                f"agregar nada avanzado, y los criterios de entrega el pipeline base completo y andando compromete que *los commits "
                f"de C2–C4 son posteriores* al tag `v1`. No es una pieza que falte: es una "
                f"que todavía no corresponde. Y si lo que falta es su **binding**, el "
                f"mensaje lo dice aparte: {binding_faltante('juez')}"
            )

        self.cfg.olvidar_lecturas()
        alfa = float(self.cfg["consulta.alfa"])
        top_k = int(self.cfg["consulta.top_k"])
        construidas: list[Etapa] = []
        for nombre in cadena:
            construidas.append(self._construir(nombre, alfa=alfa, top_k=top_k))
        sin_declarar = verificar_el_alcance(self.cfg, cadena, self.cfg.lecturas())
        if sin_declarar:
            raise CadenaImposible(
                f"🔴 fuera del alcance declarado: armar la cadena {list(cadena)} leyó "
                f"{list(sin_declarar)} de "
                f"la configuración, y **ninguna etapa de la cadena declara esas claves** "
                f"(`Etapa.CLAVES_DE_CONFIGURACIÓN`). El `config_id` se computa sobre el "
                f"alcance declarado, así que tal como está **dos corridas con esas claves "
                f"distintas tendrían el mismo `config_id` y contenido distinto** — que "
                f"la regla de que un id determina su contenido declara error duro. Se arregla declarándolas en la etapa que las "
                f"lee, o —si de verdad no determinan lo que la corrida produce— agregándolas "
                f"al bloque de (`NO_DETERMINAN_EL_CONFIG_ID`) **con su razón**. Lo que "
                f"no se puede es dejarlas afuera de las dos listas: *una identidad que "
                f"incluye de menos no se detecta, se cobra al interpretar la tabla*."
            )
        return Pipeline(
            construidas,
            configuracion=self.cfg["consulta.configuracion"],
            plan_de_falla=dict(self.cfg["consulta.plan_de_falla"]),
            top_k=top_k,
        )

    def _construir(self, nombre: str, *, alfa: float, top_k: int) -> Etapa:
        if nombre == "planificador":
            return PlanificadorDeConsulta(
                binding=BindingDelPlanificador.desde_configuración(
                    self.cfg,
                    puerto=_exigir(self.puerto_de_llm, "PuertoLLM", nombre),
                    observabilidad=self.observabilidad,
                ),
                almacen_de_planes=_exigir(
                    self.almacen_de_planes, "AlmacénDePlanes", nombre
                ),
            )
        if nombre == "recuperador_visual":
            return RecuperadorVisual(
                puerto_de_imagen=_exigir(self.puerto_de_imagen, "PuertoEmbeddingImagen", nombre),
                indice=_exigir(self.indice, "PuertoÍndiceVectorial", nombre),
                alfa=alfa,
                n_candidatos=int(self.cfg["consulta.n_candidatos"]),
                top_k=top_k,
            )
        if nombre == "recuperador_textual":
            return RecuperadorTextual(
                indice=_exigir(self.indice, "PuertoÍndiceVectorial", nombre),
                embebedor_de_texto=_exigir(
                    self.embebedor_de_texto, "PuertoEmbeddingTexto", nombre
                ),
                n_fragmentos=int(self.cfg["recuperacion_textual.n_fragmentos_recuperados"]),
                agregacion=self.cfg["recuperacion_textual.agregacion"],
            )
        if nombre == "fusionador":
            return Fusionador(
                estrategia=self.cfg["fusion.estrategia"],
                k=int(self.cfg["fusion.k"]),
                top_k=top_k,
            )
        if nombre == "reranker":
            return Reranker(
                binding=BindingDelReranker.desde_configuración(
                    self.cfg,
                    puerto=_exigir(self.puerto_de_llm, "PuertoLLM", nombre),
                    observabilidad=self.observabilidad,
                ),
                bytes_de_la_imagen=_exigir(
                    self.bytes_de_la_imagen, "AlmacénDeImágenes", nombre
                ),
                n_candidatos=int(self.cfg["consulta.n_candidatos"]),
                top_k=top_k,
            )
        if nombre == "recolector_de_evidencia":
            return RecolectorDeEvidencia(
                indice=_exigir(self.indice, "PuertoÍndiceVectorial", nombre),
                n_fragmentos_por_obra=int(self.cfg["evidencia.n_fragmentos_por_obra"]),
                orden=self.cfg["evidencia.orden"],
                campos_de_knowledge=tuple(self.cfg["fragmentacion.campos_de_knowledge"]),
                embebedor_de_texto=self.embebedor_de_texto,
            )
        if nombre == "generador":
            return Generador(
                binding=BindingDelGenerador.desde_configuración(
                    self.cfg,
                    puerto=_exigir(self.puerto_de_llm, "PuertoLLM", nombre),
                    observabilidad=self.observabilidad,
                )
            )
        if nombre == "renderizador":
            return RenderizadorDeSalida(formato=FormatoDeSalida.desde_configuración(self.cfg))
        raise CadenaImposible(  # pragma: no cover - `validar` ya lo cubrió
            f"`{nombre}` pasó la validación y no se sabe construir. Es un bug del armador: "
            f"`CONSTRUIBLES` y `_construir` tienen que decir lo mismo."
        )

def _precondiciones_de(etapa: str) -> tuple[str, ...]:
    return CLASE_DE_ETAPA[etapa].PRECONDICIONES

def claves_de_configuracion_de(etapa: str) -> tuple[str, ...]:
    return CLASE_DE_ETAPA[etapa].CLAVES_DE_CONFIGURACIÓN

def alcance_de_la_cadena(cadena: Sequence[str]) -> tuple[str, ...]:
    desconocidas = [e for e in cadena if e not in CLASE_DE_ETAPA]
    if desconocidas:
        raise CadenaImposible(
            f"`consulta.etapas` nombra {desconocidas}, que no son etapas del contrato de etapa, y el "
            f"**`config_id` no se puede computar** sin saber qué lee la cadena. Las "
            f"ocho son {list(CLASE_DE_ETAPA)}. Es clase 1 de falla y llega antes que "
            f"la precondición declarada a propósito: una identidad que no se puede derivar no se puede corregir "
            f"más tarde — la corrida ya se habría escrito bajo un id equivocado."
        )
    claves: set[str] = set()
    for etapa in cadena:
        claves.update(claves_de_configuracion_de(etapa))
    return tuple(sorted(claves))

def verificar_el_alcance(cfg, cadena: Sequence[str], leídas: Sequence[str]) -> tuple[str, ...]:
    from tp3_rag.nucleo.configuracion import fuera_del_alcance

    return fuera_del_alcance(leídas, alcance=alcance_de_la_cadena(cadena))

def _exigir(puerto, nombre_del_puerto: str, etapa: str):
    if puerto is None:
        raise CadenaImposible(
            f"la etapa `{etapa}` necesita el `{nombre_del_puerto}` y no se le inyectó "
            f"ninguno. la configuración en capas: *toda pieza recibe configuración —y puertos— inyectados*; un "
            f"puerto que la etapa construyera por su cuenta sería un camino al proveedor que "
            f"el decorador de observabilidad no ve."
        )
    return puerto
