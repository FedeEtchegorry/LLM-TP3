from __future__ import annotations

import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

@dataclass(frozen=True)
class Span:
    rol: str
    operación: str
    segundos: float
    resultado: str
    tokens: dict = field(default_factory=dict)
    n: int = 1
    etapa: str = ""

    def a_json(self) -> dict:
        d = {
            "rol": self.rol,
            "operacion": self.operación,
            "segundos": round(self.segundos, 4),
            "resultado": self.resultado,
            "n": self.n,
        }
        if self.tokens:
            d["tokens"] = dict(self.tokens)
        if self.etapa:
            d["etapa"] = self.etapa
        return d

TIPOS_DE_NODO: tuple[str, ...] = ("chain", "llm", "retriever", "embedding", "tool")

@dataclass
class NodoDeTraza:
    id: str
    trace_id: str
    dotted_order: str
    nombre: str
    tipo: str
    inicio: datetime
    padre: str | None = None
    fin: datetime | None = None
    entradas: dict = field(default_factory=dict)
    salidas: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    campos_censurados: int = 0
    politica: str = "sin_politica"

    @property
    def segundos(self) -> float:
        if self.fin is None:
            return 0.0
        return (self.fin - self.inicio).total_seconds()

    def a_json(self) -> dict:
        return {
            "nombre": self.nombre,
            "tipo": self.tipo,
            "padre": self.padre,
            "segundos": round(self.segundos, 4),
            "tokens": {
                k: v for k, v in (self.metadata.get("tokens") or {}).items() if v
            },
            "error": self.error,
            "campos_censurados": self.campos_censurados,
            "captura_entradas": bool(self.entradas),
        }

class PuertoTrazasNulo:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.nodos: list[NodoDeTraza] = []
        self.fallas = 0

    def span(self, span: Span) -> None:
        try:
            self.spans.append(span)
        except Exception:  # pragma: no cover - no hay forma de que falle en memoria
            self.fallas += 1

    def nodo(self, nodo: NodoDeTraza) -> None:
        try:
            self.nodos.append(nodo)
        except Exception:  # pragma: no cover
            self.fallas += 1

    def cerrar(self) -> None:
        pass

@dataclass
class Observabilidad:
    trazas: PuertoTrazasNulo = field(default_factory=PuertoTrazasNulo)
    eventos: Counter = field(default_factory=Counter)
    silencioso: bool = False
    _politica: Any = None
    _pila: list = field(default_factory=list)

    @contextmanager
    def con_politica_de_trazado(self, politica):
        previa = self._politica
        self._politica = politica
        try:
            yield politica
        finally:
            self._politica = previa

    def _censurar(self, entradas: Mapping[str, Any] | None) -> tuple[dict, int]:
        if not entradas:
            return {}, 0
        if self._politica is None:
            return {}, len(entradas)
        return self._politica.censurar(entradas)

    @contextmanager
    def nodo(self, nombre: str, *, tipo: str = "chain", entradas=None, metadata=None):
        if tipo not in TIPOS_DE_NODO:
            raise ValueError(
                f"tipo de nodo {tipo!r} desconocido; los de LangSmith son "
                f"{list(TIPOS_DE_NODO)}. Se valida acá y no en el adaptador para que el "
                f"puerto nulo arme la misma estructura."
            )
        padre = self._pila[-1] if self._pila else None
        inicio = datetime.now(timezone.utc)
        nodo = NodoDeTraza(
            id=str(uuid.uuid4()),
            trace_id=padre.trace_id if padre else "",
            dotted_order="",
            nombre=nombre,
            tipo=tipo,
            inicio=inicio,
            padre=padre.id if padre else None,
            entradas=dict(entradas or {}),
            metadata=dict(metadata or {}),
        )
        nodo.trace_id = nodo.trace_id or nodo.id
        sello = f"{inicio.strftime('%Y%m%dT%H%M%S%f')}Z{nodo.id}"
        nodo.dotted_order = f"{padre.dotted_order}.{sello}" if padre else sello
        self._pila.append(nodo)
        try:
            yield nodo
        except Exception as e:
            nodo.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._pila.pop()
            self.cerrar_nodo(nodo)

    def cerrar_nodo(self, nodo: NodoDeTraza) -> None:
        nodo.fin = datetime.now(timezone.utc)
        nodo.entradas, n_entradas = self._censurar(nodo.entradas)
        nodo.salidas, n_salidas = self._censurar(nodo.salidas)
        nodo.campos_censurados = n_entradas + n_salidas
        nodo.politica = (
            "ad_hoc_sin_entradas"
            if self._politica is None
            else ("completa" if self._politica.captura_entradas else "sin_entradas")
        )
        if nodo.campos_censurados:
            self.contar("trazas.campos_censurados", nodo.campos_censurados)
        self.trazas.nodo(nodo)

    def etapa_actual(self) -> str:
        for nodo in reversed(self._pila):
            etapa = nodo.metadata.get("etapa")
            if etapa:
                return str(etapa)
        return ""

    def nodos(self, desde: tuple[int, int] | None = None) -> list[NodoDeTraza]:
        return list(self.trazas.nodos[desde[1] :] if desde else self.trazas.nodos)

    def arbol_de_trazas(self, politica=None, desde: tuple[int, int] | None = None) -> dict:
        nodos = self.nodos(desde)
        return {
            "n_nodos": len(nodos),
            "por_tipo": dict(Counter(n.tipo for n in nodos)),
            "etapas_trazadas": [n.nombre for n in nodos if n.tipo == "chain"],
            "campos_censurados": sum(n.campos_censurados for n in nodos),
            "nodos_con_entradas": sum(1 for n in nodos if n.entradas),
            "politica": (
                (politica or self._politica).a_json()
                if (politica or self._politica) is not None
                else None
            ),
            "fallas_del_puerto": getattr(self.trazas, "fallas", 0),
            "nota": (
                "El árbol lo arman el `EjecutorDeEtapas` (corrida y etapa) y el decorador "
                "de puertos. Ninguna etapa emite su propio span: el lugar correcto para "
                "el desglose por etapa es el ejecutor, que ya las envuelve. "
                "`campos_censurados > 0` es la política de retención actuando: en una "
                "consulta ad-hoc no viaja ni el pedido ni la imagen."
            ),
        }

    def instrumentar(self, puerto: Any, *, rol: str, operaciones=None) -> "PuertoInstrumentado":
        declaradas = operaciones or getattr(puerto, "OPERACIONES", ())
        if not declaradas:
            raise ValueError(
                f"{type(puerto).__name__} no declara `OPERACIONES`, así que no hay "
                f"superficie que instrumentar"
            )
        return PuertoInstrumentado(puerto, rol=rol, operaciones=tuple(declaradas), obs=self)

    def marca(self) -> tuple[int, int]:
        return (len(self.trazas.spans), len(self.trazas.nodos))

    def spans(self, desde: tuple[int, int] | None = None) -> list[Span]:
        return list(self.trazas.spans[desde[0] :] if desde else self.trazas.spans)

    def libro_mayor(self, desde: tuple[int, int] | None = None) -> dict:
        return self._proyectar(lambda s: s.rol, desde)

    def libro_mayor_por_etapa(self, desde: tuple[int, int] | None = None) -> dict:
        return self._proyectar(lambda s: s.etapa or "(fuera de una etapa)", desde)

    def _proyectar(self, clave, desde: tuple[int, int] | None = None) -> dict:
        acumulado: dict[str, dict] = {}
        for s in self.spans(desde):
            r = acumulado.setdefault(
                clave(s),
                {"n_llamadas": 0, "n_items": 0, "segundos": 0.0, "tokens_in": 0, "tokens_out": 0,
                 "resultados": Counter(), "roles": Counter()},
            )
            r["n_llamadas"] += 1
            r["n_items"] += s.n
            r["segundos"] += s.segundos
            r["tokens_in"] += s.tokens.get("tokens_in", 0)
            r["tokens_out"] += s.tokens.get("tokens_out", 0)
            r["resultados"][s.resultado] += 1
            r["roles"][s.rol] += 1
        return {
            k: {
                **v,
                "segundos": round(v["segundos"], 3),
                "resultados": dict(v["resultados"]),
                "roles": dict(v["roles"]),
            }
            for k, v in sorted(acumulado.items())
        }

    def contar(self, evento: str, n: int = 1) -> None:
        self.eventos[evento] += n

    def decir(self, mensaje: str) -> None:
        if not self.silencioso:
            print(mensaje, flush=True)

    def progreso(self, mensaje: str) -> None:
        if self.silencioso:
            return
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K  {mensaje}")
        else:
            sys.stdout.write(f"  {mensaje}\n")
        sys.stdout.flush()

    def cerrar_progreso(self) -> None:
        if not self.silencioso and sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

    @contextmanager
    def cronómetro(self, nombre: str):
        t0 = time.perf_counter()
        yield
        self.eventos[f"segundos.{nombre}"] = round(time.perf_counter() - t0, 3)

class PuertoInstrumentado:
    def __init__(self, puerto, *, rol: str, operaciones: tuple[str, ...], obs: Observabilidad) -> None:
        self._puerto = puerto
        self._rol = rol
        self._operaciones = operaciones
        self._obs = obs

    def __getattr__(self, nombre: str):
        atributo = getattr(self._puerto, nombre)
        if nombre not in self._operaciones or not callable(atributo):
            return atributo

        def envuelto(*args, **kwargs):
            t0 = time.perf_counter()
            etapa = self._obs.etapa_actual()
            with self._obs.nodo(
                f"{self._rol}.{nombre}", tipo=_TIPO_DE_NODO.get(self._rol, "tool")
            ) as nodo:
                try:
                    salida = atributo(*args, **kwargs)
                except Exception as e:
                    self._obs.trazas.span(
                        Span(
                            self._rol,
                            nombre,
                            time.perf_counter() - t0,
                            f"error:{type(e).__name__}",
                            etapa=etapa,
                        )
                    )
                    raise
                uso = dict(getattr(salida, "uso", None) or {})
                n = getattr(salida, "n", None) or _largo(salida)
                self._obs.trazas.span(
                    Span(
                        rol=self._rol,
                        operación=nombre,
                        segundos=time.perf_counter() - t0,
                        resultado="ok",
                        tokens=uso,
                        n=n,
                        etapa=etapa,
                    )
                )
                nodo.metadata.update({"tokens": uso, "n": n, "rol": self._rol})
                if isinstance(getattr(salida, "texto", None), str):
                    nodo.salidas["texto"] = salida.texto
                return salida

        return envuelto

_TIPO_DE_NODO = {
    "planificador": "llm",
    "reranker": "llm",
    "generador": "llm",
    "juez": "llm",
    "embedding_texto": "embedding",
    "embedding_imagen": "embedding",
    "indice": "retriever",
}

def _largo(salida) -> int:
    try:
        if salida and isinstance(next(iter(salida)), (int, float)) and not isinstance(
            salida, (str, bytes)
        ):
            return 1
    except TypeError:
        return 1
    try:
        return len(salida)
    except TypeError:
        return 1
