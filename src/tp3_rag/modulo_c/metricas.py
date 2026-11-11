from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import log2
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.puertos.indice_vectorial import AUSENTE, SEPARADOR_DE_LISTA

CATEGORÍAS: tuple[str, ...] = (
    "visual_cromatico",
    "conceptual_cultural",
    "ambiguo",
)

@dataclass(frozen=True)
class Agregables:
    agregan: tuple[Any, ...]
    planificadas: tuple[Any, ...]
    espontaneas: tuple[Any, ...]
    fallidas: tuple[Any, ...]

    def a_json(self) -> dict:
        return {
            "agregan": len(self.agregan),
            "con_degradacion_planificada": len(self.planificadas),
            "con_degradacion_espontanea": len(self.espontaneas),
            "fallidas_por_obligacion": len(self.fallidas),
            "causas_espontaneas": sorted(
                {
                    f"{d['etapa']}: {d['causa'][:90]}"
                    for c in self.espontaneas
                    for d in c.degradaciones_no_planificadas
                }
            ),
            "nota": (
                "la regla de qué corridas agregan a la tabla: una corrida con degradación **no planificada** no agrega —*una C4 "
                "que degradó en silencio entraría etiquetada como C4 comportándose como "
                "C3*— y una con degradación **planificada** tampoco: es el criterio de respuesta garantizada y se reporta "
                "aparte. Una etapa **ausente por configuración** no degrada: no corre, y "
                "por eso C0–C3 sí agregan."
            ),
        }

def separar_por_9_5(corridas: Sequence[Any]) -> Agregables:
    agregan, planificadas, espontáneas, fallidas = [], [], [], []
    for c in corridas:
        if c.fallida is not None:
            fallidas.append(c)
        if c.degradaciones_no_planificadas:
            espontáneas.append(c)
        elif c.degradaciones:
            planificadas.append(c)
        elif c.fallida is None:
            agregan.append(c)
    return Agregables(
        agregan=tuple(agregan),
        planificadas=tuple(planificadas),
        espontaneas=tuple(espontáneas),
        fallidas=tuple(fallidas),
    )

def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    conjunto_a, conjunto_b = set(a), set(b)
    union = conjunto_a | conjunto_b
    return len(conjunto_a & conjunto_b) / len(union) if union else 1.0

def qs(top5_por_pedido: Sequence[Sequence[str]]) -> float | None:
    pares = list(combinations([list(t) for t in top5_por_pedido], 2))
    if not pares:
        return None
    return 1.0 - sum(jaccard(a, b) for a, b in pares) / len(pares)

def qs_por_imagen(corridas: Sequence[Any], *, imagen_de) -> dict[str, float | None]:
    por_imagen: dict[str, list[list[str]]] = {}
    for c in corridas:
        imagen = imagen_de(c.consulta_id)
        if imagen is None:
            continue
        por_imagen.setdefault(imagen, []).append(list(c.top5))
    return {img: qs(tops) for img, tops in sorted(por_imagen.items())}

def culturas_del(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    crudo = metadata.get("culture")
    if crudo in (None, AUSENTE):
        return ()
    return tuple(p.strip() for p in str(crudo).split(SEPARADOR_DE_LISTA) if p.strip())

@dataclass(frozen=True)
class CoberturaCultural:
    consulta_id: str
    categoria: str
    cultura_pedida: tuple[str, ...]
    n_en_el_top5: int
    de: int
    sin_culture: int
    culturas_del_top5: tuple[str, ...] = ()

    def a_json(self) -> dict:
        return {
            "consulta_id": self.consulta_id,
            "categoria": self.categoria,
            "cultura_pedida": list(self.cultura_pedida),
            "n_en_el_top5": self.n_en_el_top5,
            "de": self.de,
            "obras_sin_culture": self.sin_culture,
            "culturas_del_top5": list(self.culturas_del_top5),
        }

def cobertura_cultural(
    corridas: Sequence[Any],
    *,
    culturas_por_consulta: Mapping[str, Sequence[str]],
    categoria_de: Mapping[str, str],
    metadata_por_obra: Mapping[str, Mapping[str, Any]],
) -> tuple[CoberturaCultural, ...]:
    salida = []
    for c in corridas:
        etiquetas = culturas_por_consulta.get(c.consulta_id)
        if not etiquetas:
            continue
        pedidas = {e.lower() for e in etiquetas}
        aciertos, sin_culture, vistas = 0, 0, []
        for artwork_id in c.top5:
            culturas = culturas_del(metadata_por_obra.get(artwork_id) or {})
            vistas += list(culturas)
            if not culturas:
                sin_culture += 1
                continue
            if any(k.lower() in pedidas for k in culturas):
                aciertos += 1
        salida.append(
            CoberturaCultural(
                consulta_id=c.consulta_id,
                categoria=categoria_de.get(c.consulta_id, "?"),
                cultura_pedida=tuple(etiquetas),
                n_en_el_top5=aciertos,
                de=len(c.top5),
                sin_culture=sin_culture,
                culturas_del_top5=tuple(dict.fromkeys(vistas)),
            )
        )
    return tuple(salida)

@dataclass
class DeltaPorCategoria:
    categoria: str
    n_consultas: int
    churn_medio: float
    churn_por_consulta: dict[str, int] = field(default_factory=dict)
    cobertura_a: float | None = None
    cobertura_b: float | None = None

    @property
    def delta_de_cobertura(self) -> float | None:
        if self.cobertura_a is None or self.cobertura_b is None:
            return None
        return self.cobertura_b - self.cobertura_a

    def a_json(self) -> dict:
        return {
            "categoria": self.categoria,
            "n_consultas": self.n_consultas,
            "churn_medio_del_top5": round(self.churn_medio, 3),
            "cobertura_cultural_media_a": self.cobertura_a,
            "cobertura_cultural_media_b": self.cobertura_b,
            "delta_de_cobertura": self.delta_de_cobertura,
        }

def delta_por_categoria(
    corridas_a: Sequence[Any],
    corridas_b: Sequence[Any],
    *,
    categoria_de: Mapping[str, str],
    cobertura_a: Sequence[CoberturaCultural] = (),
    cobertura_b: Sequence[CoberturaCultural] = (),
) -> tuple[DeltaPorCategoria, ...]:
    por_id_a = {c.consulta_id: c for c in corridas_a}
    por_id_b = {c.consulta_id: c for c in corridas_b}
    comunes = sorted(set(por_id_a) & set(por_id_b))
    cob_a = {c.consulta_id: c for c in cobertura_a}
    cob_b = {c.consulta_id: c for c in cobertura_b}

    salida = []
    for categoria in CATEGORÍAS:
        ids = [i for i in comunes if categoria_de.get(i) == categoria]
        if not ids:
            continue
        churn = {
            i: len(set(por_id_b[i].top5) - set(por_id_a[i].top5)) for i in ids
        }
        salida.append(
            DeltaPorCategoria(
                categoria=categoria,
                n_consultas=len(ids),
                churn_medio=sum(churn.values()) / len(ids),
                churn_por_consulta=churn,
                cobertura_a=_media([cob_a[i].n_en_el_top5 for i in ids if i in cob_a]),
                cobertura_b=_media([cob_b[i].n_en_el_top5 for i in ids if i in cob_b]),
            )
        )
    return tuple(salida)

def _media(valores: Sequence[float]) -> float | None:
    return round(sum(valores) / len(valores), 3) if valores else None

VALORES_DE_N: tuple[int, ...] = (10, 20, 50)

ARCHIVO_POR_FILA: dict[str, str | None] = {
    "C0": "config/c0.toml",
    "C1": None,
    "C2": "config/c2.toml",
    "C3": "config/c3.toml",
    "C4": "config/c4.toml",
}

def config_ids_de_la_tabla(*, raiz=None) -> dict[str, str]:
    from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración

    raíz = raiz or RAÍZ_DEL_REPO
    salida: dict[str, str] = {}
    for fila, archivo in ARCHIVO_POR_FILA.items():
        ruta = (raíz / archivo) if archivo else None
        if ruta is not None and not ruta.exists():
            continue
        cfg = Configuración.cargar(archivo=ruta)
        declarada = cfg["consulta.configuracion"]
        if declarada != fila:
            raise ErrorDelProyecto(
                f"`{archivo or 'config/default.toml'}` declara "
                f"`consulta.configuracion = {declarada!r}` y esta tabla lo mapea a la fila "
                f"**{fila}**. Las dos no pueden discrepar: la etiqueta es contra lo que "
                f"la precondición declarada compara la cadena, y acá es contra lo que se agrupa la población de "
                f"un número publicado."
            )
        salida[fila] = cfg.config_id
    return salida

ROLES_LOCALES: tuple[str, ...] = ("embedding_imagen", "indice")

@dataclass(frozen=True)
class CostoMedido:
    configuracion: str
    n_corridas: int
    n_candidatos: int | None
    por_rol: Mapping[str, Mapping[str, int]]
    por_etapa: Mapping[str, Mapping[str, int]]
    modelo_por_rol: Mapping[str, str | None] = field(default_factory=dict)
    precios_in: Mapping[str, float] = field(default_factory=dict)
    precios_out: Mapping[str, float] = field(default_factory=dict)
    plan_nominal: Mapping[str, int] = field(default_factory=dict)
    planes_reusados: int = 0

    @property
    def tokens_in(self) -> int:
        return sum(int(d.get("tokens_in", 0)) for d in self.por_rol.values())

    @property
    def tokens_out(self) -> int:
        return sum(int(d.get("tokens_out", 0)) for d in self.por_rol.values())

    @property
    def llamadas_al_llm(self) -> int:
        from tp3_rag.modulo_c.validadores import ROLES_DEL_LLM

        return sum(
            int(self.por_rol.get(rol, {}).get("n_llamadas", 0)) for rol in ROLES_DEL_LLM
        )

    @property
    def llamadas_al_llm_con_el_plan(self) -> int:
        return self.llamadas_al_llm + self.planes_reusados

    def _usd(self, rol: str, tokens_in: int, tokens_out: int) -> float | None:
        if rol in ROLES_LOCALES:
            return 0.0
        modelo = self.modelo_por_rol.get(rol)
        if not modelo:
            return None
        precio_in = self.precios_in.get(modelo)
        if precio_in is None:
            return None
        precio_out = self.precios_out.get(modelo)
        if tokens_out and precio_out is None:
            return None
        return tokens_in / 1e6 * float(precio_in) + tokens_out / 1e6 * float(
            precio_out or 0.0
        )

    @property
    def usd_por_rol(self) -> dict[str, float | None]:
        return {
            rol: self._usd(rol, int(d.get("tokens_in", 0)), int(d.get("tokens_out", 0)))
            for rol, d in sorted(self.por_rol.items())
        }

    @property
    def usd_pagado(self) -> float | None:
        valores = self.usd_por_rol.values()
        if any(v is None for v in valores):
            return None
        return sum(v for v in valores if v is not None)

    @property
    def usd_del_plan(self) -> float | None:
        if not self.plan_nominal:
            return 0.0
        return self._usd(
            "planificador",
            int(self.plan_nominal.get("tokens_in", 0)),
            int(self.plan_nominal.get("tokens_out", 0)),
        )

    @property
    def usd_nominal(self) -> float | None:
        sin_plan = {
            rol: v for rol, v in self.usd_por_rol.items() if rol != "planificador"
        }
        if any(v is None for v in sin_plan.values()):
            return None
        plan = self.usd_del_plan
        if plan is None:
            return None
        return sum(v for v in sin_plan.values() if v is not None) + plan

    @property
    def usd_por_consulta(self) -> float | None:
        total = self.usd_nominal
        if total is None or not self.n_corridas:
            return None
        return total / self.n_corridas

    @property
    def tokens_in_por_consulta_con_el_plan(self) -> int | None:
        if not self.n_corridas:
            return None
        del_libro = self.tokens_in - int(
            (self.por_rol.get("planificador") or {}).get("tokens_in", 0)
        )
        return (del_libro + int(self.plan_nominal.get("tokens_in", 0))) // self.n_corridas

    @property
    def tokens_por_llamada_del_reranker(self) -> int | None:
        d = self.por_rol.get("reranker") or {}
        llamadas = int(d.get("n_llamadas", 0))
        return int(d.get("tokens_in", 0)) // llamadas if llamadas else None

    def a_json(self) -> dict:
        return {
            "configuracion": self.configuracion,
            "n_corridas": self.n_corridas,
            "n_candidatos": self.n_candidatos,
            "llamadas_al_llm": self.llamadas_al_llm,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_in_por_consulta": (
                self.tokens_in // self.n_corridas if self.n_corridas else None
            ),
            "tokens_in_por_llamada_del_reranker": self.tokens_por_llamada_del_reranker,
            "por_rol": {k: dict(v) for k, v in self.por_rol.items()},
            "por_etapa": {k: dict(v) for k, v in self.por_etapa.items()},
            "llamadas_al_llm_con_el_plan": self.llamadas_al_llm_con_el_plan,
            "planes_reusados_r11": self.planes_reusados,
            "plan_nominal": dict(self.plan_nominal),
            "tokens_in_por_consulta_con_el_plan": self.tokens_in_por_consulta_con_el_plan,
            "modelo_por_rol": dict(self.modelo_por_rol),
            "usd_por_rol": self.usd_por_rol,
            "usd_pagado": (
                round(self.usd_pagado, 6) if self.usd_pagado is not None else None
            ),
            "usd_nominal": (
                round(self.usd_nominal, 6) if self.usd_nominal is not None else None
            ),
            "usd_por_consulta": (
                round(self.usd_por_consulta, 6)
                if self.usd_por_consulta is not None
                else None
            ),
            "precio_usd_por_millon_de_tokens": {
                "entrada": dict(self.precios_in),
                "salida": dict(self.precios_out),
                "fuente": "https://developers.openai.com/api/docs/pricing",
                "leido": "2026-11-03",
            },
            "nota_del_usd": (
                "🔴 **Los tokens son MEDIDOS; el USD es DERIVADO de un precio DECLARADO.** El "
                "precio es el de **lista de OpenAI** leído el 2026-11-03, no la tarifa del "
                "proxy de ITBA, que nadie puede leer (`/key/info` bloqueado para virtual "
                "keys). *Un modelo sin precio da **ausente**, no cero* sigue "
                "vigente. ⚠️ **Y las dos cifras del gasto total conviven sin "
                "promediarse**: derivado de los artefactos a precio de lista "
                "da **USD 12,43** y el "
                "dashboard del proxy facturó **USD 10,0654** (la decisión de presupuesto). La hipótesis "
                "—el proxy factura un **19 % por debajo de lista**, base: lista— es el "
                "**descuento por entrada cacheada**, 0,10 contra 0,40 en "
                "`gpt-4.1-mini`, y el reranker repite el mismo prompt N veces por consulta—, "
                "**no verificable con nuestros artefactos**: el libro mayor guarda "
                "`prompt_tokens` y no `cached_tokens`."
            ),
            "nota_del_plan_r11": (
                "🔴 **`usd_por_consulta` NO es `usd_pagado / n`.** Un plan **reusado** no paga "
                "la llamada del planificador (el reuso del plan al barrer el peso de fusión: cinco brazos de un barrido comparten un "
                "plan), así que el libro mayor de `C2` tiene **cero** llamadas de "
                "planificador. El caché del plan es un artefacto de **nuestra ablación**, no "
                "de producción: una consulta de un usuario paga su planificador **siempre**. "
                "Por eso lo nominal suma **un plan por corrida** desde `estado.plan.uso` y "
                "resta el planificador del libro mayor, que si no se contaría dos veces en "
                "las corridas que sí lo crearon."
            ),
        }

def corridas_de_la_tabla(corridas_de, *, raiz=None, filas=None) -> dict:
    config_ids = {**config_ids_de_la_tabla(raiz=raiz), **dict(filas or {})}
    de_la_tabla = {v: k for k, v in config_ids.items()}
    por_clave: dict[tuple[str, str], Any] = {}
    huérfanas: dict[str, int] = {}
    de_p7: dict[str, int] = {}
    ad_hoc: dict[str, int] = {}
    en_disco = 0
    for cid in corridas_de.existentes():
        try:
            corrida = corridas_de.leer(cid)
        except ErrorDelProyecto:
            continue
        en_disco += 1
        if corrida.config_id in de_la_tabla and not corrida.es_del_set:
            ad_hoc[de_la_tabla[corrida.config_id]] = (
                ad_hoc.get(de_la_tabla[corrida.config_id], 0) + 1
            )
            continue
        if corrida.config_id not in de_la_tabla:
            etiqueta = (corrida.contenido.get("pipeline") or {}).get("configuracion", "?")
            if (corrida.contenido.get("pipeline") or {}).get("plan_de_falla"):
                de_p7[etiqueta] = de_p7.get(etiqueta, 0) + 1
            else:
                huérfanas[etiqueta] = huérfanas.get(etiqueta, 0) + 1
            continue
        clave = (corrida.config_id, corrida.consulta_id)
        previa = por_clave.get(clave)
        if previa is None or corrida.ordinal > previa.ordinal:
            por_clave[clave] = corrida
    corridas = tuple(por_clave.values())
    por_fila: dict[str, int] = {}
    for c in corridas:
        fila = de_la_tabla[c.config_id]
        por_fila[fila] = por_fila.get(fila, 0) + 1
    return {
        "corridas": corridas,
        "por_fila": dict(sorted(por_fila.items())),
        "config_ids": config_ids,
        "huerfanas": dict(sorted(huérfanas.items())),
        "de_p7": dict(sorted(de_p7.items())),
        "ad_hoc": dict(sorted(ad_hoc.items())),
        "en_disco": en_disco,
        "faltan": [f for f in config_ids if f not in por_fila],
    }

ROLES_QUE_PAGAN_AL_PROXY: tuple[str, ...] = (
    "planificador",
    "generador",
    "reranker",
    "juez",
    "embedding_texto",
)

def llamadas_al_proxy(corridas_de, *, artefactos_extra: Sequence[Mapping[str, Any]] = ()) -> dict:
    por_rol: dict[str, int] = {}
    por_fuente: dict[str, int] = {"corridas": 0, "otros_artefactos": 0}
    n_corridas = 0
    for cid in corridas_de.existentes():
        try:
            corrida = corridas_de.leer(cid)
        except ErrorDelProyecto:
            continue
        n_corridas += 1
        for rol, valores in (corrida.contenido.get("libro_mayor") or {}).items():
            if rol not in ROLES_QUE_PAGAN_AL_PROXY:
                continue
            n = int(valores.get("n_llamadas", 0) or 0)
            por_rol[rol] = por_rol.get(rol, 0) + n
            por_fuente["corridas"] += n
    for artefacto in artefactos_extra:
        for rol, valores in (artefacto.get("libro_mayor") or {}).items():
            if rol not in ROLES_QUE_PAGAN_AL_PROXY:
                continue
            n = int(valores.get("n_llamadas", 0) or 0)
            por_rol[rol] = por_rol.get(rol, 0) + n
            por_fuente["otros_artefactos"] += n
    total = sum(por_rol.values())
    return {
        "total": total,
        "por_rol": dict(sorted(por_rol.items())),
        "por_fuente": por_fuente,
        "n_corridas_leidas": n_corridas,
        "roles_locales_excluidos": ["embedding_imagen (CLIP)", "indice (Chroma)"],
        "nota": (
            "**Requests al proxy**, que es la unidad que el dashboard cobra. Población: "
            "**todas** las corridas en disco —huérfanas, del criterio de respuesta garantizada, ad-hoc y brazos de barrido "
            "incluidos— más los artefactos con libro mayor que no son corridas. Es otra "
            "población que la tabla de la tabla de ablación, que solo lleva las cinco filas declaradas, y "
            "sumar solo esa **subestimaría el gasto**. `embedding_imagen` (CLIP) e `indice` "
            "(Chroma) **no entran**: son locales."
        ),
    }

def modelos_declarados_por_rol(resuelta: Mapping[str, Any]) -> dict[str, str]:
    salida: dict[str, str] = {}
    for rol, binding in (resuelta.get("bindings") or {}).items():
        modelo = (binding or {}).get("modelo")
        if modelo:
            salida[str(rol)] = str(modelo)
    modelo_de_texto = (resuelta.get("embebido_texto") or {}).get("modelo")
    if modelo_de_texto:
        salida["embedding_texto"] = str(modelo_de_texto)
    return salida

def costo_por_configuracion(
    corridas: Sequence[Any],
    *,
    precios_in: Mapping[str, float] | None = None,
    precios_out: Mapping[str, float] | None = None,
) -> tuple[CostoMedido, ...]:
    acumulado: dict[str, dict] = {}
    for corrida in corridas:
        etiqueta = (corrida.contenido.get("pipeline") or {}).get("configuracion", "?")
        d = acumulado.setdefault(
            etiqueta,
            {
                "n": 0,
                "n_candidatos": None,
                "por_rol": {},
                "por_etapa": {},
                "modelos": {},
                "plan": {"n_llamadas": 0, "tokens_in": 0, "tokens_out": 0},
                "reusados": 0,
            },
        )
        d["n"] += 1
        resuelta = corrida.contenido.get("configuracion_resuelta") or {}
        d["n_candidatos"] = (resuelta.get("consulta") or {}).get("n_candidatos")
        for rol, modelo in modelos_declarados_por_rol(resuelta).items():
            d["modelos"].setdefault(rol, set()).add(modelo)
        for clave, destino in (
            ("libro_mayor", "por_rol"),
            ("libro_mayor_por_etapa", "por_etapa"),
        ):
            for grupo, valores in (corrida.contenido.get(clave) or {}).items():
                acum = d[destino].setdefault(
                    grupo, {"n_llamadas": 0, "tokens_in": 0, "tokens_out": 0}
                )
                for campo in acum:
                    acum[campo] += int(valores.get(campo, 0) or 0)
        plan = (corrida.estado or {}).get("plan") or {}
        if plan:
            uso = plan.get("uso") or {}
            d["plan"]["n_llamadas"] += 1
            d["plan"]["tokens_in"] += int(uso.get("tokens_in", 0) or 0)
            d["plan"]["tokens_out"] += int(uso.get("tokens_out", 0) or 0)
            if plan.get("reusado"):
                d["reusados"] += 1
    return tuple(
        CostoMedido(
            configuracion=etiqueta,
            n_corridas=d["n"],
            n_candidatos=d["n_candidatos"],
            por_rol={k: v for k, v in sorted(d["por_rol"].items())},
            por_etapa={k: v for k, v in sorted(d["por_etapa"].items())},
            modelo_por_rol={
                rol: (next(iter(ms)) if len(ms) == 1 else None)
                for rol, ms in sorted(d["modelos"].items())
            },
            precios_in=dict(precios_in or {}),
            precios_out=dict(precios_out or {}),
            plan_nominal=dict(d["plan"]),
            planes_reusados=d["reusados"],
        )
        for etiqueta, d in sorted(acumulado.items())
    )

def derivar_n(
    rerank: Sequence[Mapping[str, Any]], *, valores: Sequence[int] = VALORES_DE_N, top_k: int = 5
) -> dict[int, tuple[str, ...]]:
    if any(f.get("estado") != "puntuado" for f in rerank):
        return {}
    salida: dict[int, tuple[str, ...]] = {}
    for n in valores:
        dentro = [f for f in rerank if int(f.get("rango_fusionado") or 0) <= n]
        salida[n] = tuple(f["artwork_id"] for f in dentro[:top_k])
    return salida

def barrido_anidado_de_n(
    corridas: Sequence[Any],
    *,
    imagen_de,
    culturas_por_consulta: Mapping[str, Sequence[str]],
    categoria_de: Mapping[str, str],
    metadata_por_obra: Mapping[str, Mapping[str, Any]],
    valores: Sequence[int] = VALORES_DE_N,
    top_k: int = 5,
) -> dict:
    medido = max(valores)
    derivados: dict[int, dict[str, tuple[str, ...]]] = {n: {} for n in valores}
    salteadas, segundos, llamadas, tokens_in, tokens_out = [], 0.0, 0, 0, 0
    for corrida in corridas:
        rerank = (corrida.estado.get("rerank") or ())
        por_n = derivar_n(rerank, valores=valores, top_k=top_k)
        if not por_n:
            salteadas.append(corrida.consulta_id)
            continue
        for n, top5 in por_n.items():
            derivados[n][corrida.consulta_id] = top5
        libro = (corrida.contenido.get("libro_mayor") or {}).get("reranker") or {}
        segundos += float(libro.get("segundos", 0) or 0)
        llamadas += int(libro.get("n_llamadas", 0) or 0)
        tokens_in += int(libro.get("tokens_in", 0) or 0)
        tokens_out += int(libro.get("tokens_out", 0) or 0)

    por_llamada_s = segundos / llamadas if llamadas else 0.0
    por_llamada_in = tokens_in // llamadas if llamadas else 0
    por_llamada_out = tokens_out // llamadas if llamadas else 0
    n_consultas = len(derivados[medido])

    filas = {}
    for n in sorted(valores):
        tops = derivados[n]
        por_imagen: dict[str, list[list[str]]] = {}
        for consulta_id, top5 in tops.items():
            imagen = imagen_de(consulta_id)
            if imagen is not None:
                por_imagen.setdefault(imagen, []).append(list(top5))
        definidas = [v for v in (qs(t) for t in por_imagen.values()) if v is not None]
        aciertos = de = 0
        for consulta_id, top5 in tops.items():
            etiquetas = culturas_por_consulta.get(consulta_id)
            if not etiquetas:
                continue
            pedidas = {e.lower() for e in etiquetas}
            de += len(top5)
            for artwork_id in top5:
                culturas = culturas_del(metadata_por_obra.get(artwork_id) or {})
                if culturas and any(k.lower() in pedidas for k in culturas):
                    aciertos += 1
        churn = [
            len(set(tops[c]) - set(derivados[medido][c])) for c in tops if c in derivados[medido]
        ]
        filas[n] = {
            "n": n,
            "medido": n == medido,
            "consultas": len(tops),
            "qs": round(sum(definidas) / len(definidas), 3) if definidas else None,
            "imagenes_con_qs": len(definidas),
            "cobertura": f"{aciertos}/{de}" if de else None,
            "cobertura_media": round(aciertos / (de / top_k), 3) if de else None,
            "churn_medio_contra_n_mayor": (
                round(sum(churn) / len(churn), 3) if churn else None
            ),
            "llamadas_de_reranking_por_consulta": n,
            "llamadas_al_llm_por_consulta": 2 + n,
            "segundos_de_reranking_por_consulta": round(por_llamada_s * n, 1),
            "tokens_in_por_consulta_de_reranking": por_llamada_in * n,
            "llamadas_de_reranking_sobre_las_30": n * n_consultas,
        }
    return {
        "medido_con_n": medido,
        "n_consultas": n_consultas,
        "consultas_salteadas": tuple(salteadas),
        "por_llamada_medido": {
            "segundos": round(por_llamada_s, 2),
            "tokens_in": por_llamada_in,
            "tokens_out": por_llamada_out,
            "llamadas": llamadas,
        },
        "por_n": filas,
        "ahorro_en_llamadas_de_reranking": sum(valores) * n_consultas - medido * n_consultas,
        "nota_del_anidado": (
            "🔴 **Una corrida, tres valores de N.** El reranker puntúa **un candidato por "
            "llamada** y el mensaje de un candidato es **idéntico** carácter por carácter "
            "dentro de un fusionado de 50 y de 20 —verificado en "
            "`tests/test_reranker.py::test_m10_el_mensaje_de_un_candidato_no_depende_de_n`, y "
            "ni N ni el rango entran al texto—, así que derivar N es filtrar por **rango "
            "fusionado ≤ N** y reordenar con **esos mismos puntajes**. "
            "🔑 **Y es más correcto, no solo más barato:** tres corridas separadas "
            "compararían también la variabilidad del reranker, que midió en **0,8 "
            "obras de 5** — del orden de lo que el propio eje mueve. Anidado, **la única "
            "diferencia entre las tres filas es N**."
        ),
        "nota_de_la_latencia": (
            "⚠️ **La latencia por N es una proyección y la etiqueta lo dice.** Lo **medido** "
            "es el segundo por llamada de esta corrida; las llamadas son secuenciales (un "
            "candidato por llamada, el diseño de la arquitectura), así que `segundos(N) = por_llamada × N`. "
            "Es la misma distinción *medido / proyectado* y la séptima vez que el "
            "proyecto la mantiene a propósito."
        ),
    }

def proyeccion_de_n(
    costo: CostoMedido, *, valores: Sequence[int] = VALORES_DE_N, n_consultas: int = 30
) -> dict:
    por_llamada = costo.tokens_por_llamada_del_reranker
    reranker = costo.por_rol.get("reranker") or {}
    llamadas_medidas = int(reranker.get("n_llamadas", 0))
    out_por_llamada = (
        int(reranker.get("tokens_out", 0)) // llamadas_medidas if llamadas_medidas else 0
    )
    sin_reranker_in = costo.tokens_in - int(reranker.get("tokens_in", 0))
    sin_reranker_out = costo.tokens_out - int(reranker.get("tokens_out", 0))
    por_consulta_sin_reranker = (
        sin_reranker_in // costo.n_corridas if costo.n_corridas else 0
    )
    proyecciones = {}
    for n in valores:
        reranking_in = (por_llamada or 0) * n
        proyecciones[str(n)] = {
            "llamadas_al_llm_por_consulta": 2 + n,
            "tokens_in_por_consulta": por_consulta_sin_reranker + reranking_in,
            "tokens_in_sobre_las_30": (por_consulta_sin_reranker + reranking_in)
            * n_consultas,
            "llamadas_de_reranking_sobre_las_30": n * n_consultas,
            "tokens_out_por_consulta": (
                (sin_reranker_out // costo.n_corridas if costo.n_corridas else 0)
                + out_por_llamada * n
            ),
            "medido": n == costo.n_candidatos,
        }
    return {
        "medido_con_n": costo.n_candidatos,
        "tokens_in_por_llamada_del_reranker": por_llamada,
        "llamadas_del_reranker_medidas": llamadas_medidas,
        "tokens_in_por_consulta_sin_reranking": por_consulta_sin_reranker,
        "por_n": proyecciones,
        "nota_del_supuesto": (
            "⚠️ **Una fila está medida y las otras dos son proyección**, y la etiqueta "
            "`medido` lo dice fila por fila. El supuesto es que el **costo medio por "
            "llamada se mantenga** al pedir más candidatos: la llamada es *imagen del "
            "usuario + imagen del candidato*, y el peso de la del candidato varía por obra "
            "(mediana 248 KB, máximo 1.015 KB). La media de las llamadas medidas es "
            "el mejor estimador que hay **y no es una medición** — es la sexta vez que el "
            "proyecto distingue medido de proyectado."
        ),
        "nota_de_d13": (
            "🔑 **El contraste con el registro de decisiones es el control de esta cifra.** el modelo del LLM por rol midió "
            "**a mano** 1.946 tokens de entrada por par de imágenes con `gpt-4.1-mini`, y "
            "dijo que *la cifra publicable sale de la tabla de costo que emite el pipeline*. "
            "Esta es esa tabla: si el número de acá se despega del que declara el modelo del LLM por rol, **el modelo o el "
            "`detail` cambiaron** —o la medición a mano no tenía la forma de la llamada "
            "real— y hay que volver a medir **antes** de barrer N."
        ),
    }

SECCIÓN_SIN_EVIDENCIA = "(colgada · sin evidencia)"

@dataclass(frozen=True)
class GeneraciónMedida:
    configuracion: str
    config_id: str
    variante_id: str
    campos_de_knowledge: tuple[str, ...]
    n_corridas: int
    n_obras: int
    n_explicaciones: int
    n_abstenciones: int
    n_fragmentos_de_evidencia: int
    evidencia_por_seccion: dict[str, int]
    n_referencias: int
    n_obras_con_referencia: int
    referencias_por_seccion: dict[str, int]
    n_referencias_colgadas: int
    n_afirmaciones_descartadas: int
    n_citas_byte_a_byte: int
    obras_por_seccion: dict[str, int]
    obras_con_evidencia_por_seccion: dict[str, int] = field(default_factory=dict)

    estrategia_de_chunking: str = ""
    n_fragmentos_que_mezclan_licencias: int = 0
    n_referencias_que_mezclan_licencias: int = 0
    evidencia_por_combinacion_de_licencias: dict[str, int] = field(default_factory=dict)
    referencias_por_combinacion_de_licencias: dict[str, int] = field(default_factory=dict)
    ejemplos_de_mezcla_citada: tuple = ()
    tokens_de_evidencia: int | None = None
    caracteres_de_evidencia: int = 0
    obras_por_cantidad_de_fragmentos: dict[int, int] = field(default_factory=dict)

    @property
    def tokens_de_evidencia_por_obra(self) -> float | None:
        if self.tokens_de_evidencia is None or not self.n_obras:
            return None
        return round(self.tokens_de_evidencia / self.n_obras, 1)

    @property
    def caracteres_de_evidencia_por_obra(self) -> float:
        return round(self.caracteres_de_evidencia / self.n_obras, 1) if self.n_obras else 0.0

    @property
    def referencias_por_obra(self) -> float:
        return round(self.n_referencias / self.n_obras, 3) if self.n_obras else 0.0

    @property
    def fragmentos_por_obra(self) -> float:
        return (
            round(self.n_fragmentos_de_evidencia / self.n_obras, 3)
            if self.n_obras
            else 0.0
        )

    def tasa_de_uso_por_seccion(self) -> dict[str, float | None]:
        salida: dict[str, float | None] = {}
        for seccion in sorted(set(self.evidencia_por_seccion) | set(self.referencias_por_seccion)):
            ofrecidos = self.evidencia_por_seccion.get(seccion, 0)
            citados = self.referencias_por_seccion.get(seccion, 0)
            salida[seccion] = round(citados / ofrecidos, 3) if ofrecidos else None
        return salida

    def a_json(self) -> dict:
        return {
            "configuracion": self.configuracion,
            "config_id": self.config_id,
            "variante_id": self.variante_id,
            "campos_de_knowledge": list(self.campos_de_knowledge),
            "n_corridas": self.n_corridas,
            "n_obras_del_top5": self.n_obras,
            "n_explicaciones": self.n_explicaciones,
            "n_abstenciones": self.n_abstenciones,
            "n_fragmentos_de_evidencia": self.n_fragmentos_de_evidencia,
            "fragmentos_de_evidencia_por_obra": self.fragmentos_por_obra,
            "evidencia_por_seccion": dict(self.evidencia_por_seccion),
            "obras_con_evidencia_por_seccion": dict(self.obras_con_evidencia_por_seccion),
            "n_referencias_publicadas": self.n_referencias,
            "referencias_por_obra": self.referencias_por_obra,
            "n_obras_con_al_menos_una_referencia": self.n_obras_con_referencia,
            "referencias_por_seccion": dict(self.referencias_por_seccion),
            "obras_con_al_menos_una_referencia_por_seccion": dict(self.obras_por_seccion),
            "tasa_de_uso_por_seccion": self.tasa_de_uso_por_seccion(),
            "n_referencias_colgadas": self.n_referencias_colgadas,
            "n_afirmaciones_descartadas_por_sa5": self.n_afirmaciones_descartadas,
            "n_citas_byte_a_byte": self.n_citas_byte_a_byte,
            "estrategia_de_chunking": self.estrategia_de_chunking,
            "n_fragmentos_de_evidencia_que_mezclan_licencias": (
                self.n_fragmentos_que_mezclan_licencias
            ),
            "n_referencias_que_mezclan_licencias": self.n_referencias_que_mezclan_licencias,
            "evidencia_por_combinacion_de_licencias": dict(
                sorted(self.evidencia_por_combinacion_de_licencias.items())
            ),
            "referencias_por_combinacion_de_licencias": dict(
                sorted(self.referencias_por_combinacion_de_licencias.items())
            ),
            "ejemplos_de_mezcla_citada": list(self.ejemplos_de_mezcla_citada),
            "tokens_de_evidencia": self.tokens_de_evidencia,
            "tokens_de_evidencia_por_obra": self.tokens_de_evidencia_por_obra,
            "caracteres_de_evidencia": self.caracteres_de_evidencia,
            "caracteres_de_evidencia_por_obra": self.caracteres_de_evidencia_por_obra,
            "obras_por_cantidad_de_fragmentos": {
                str(k): v for k, v in sorted(self.obras_por_cantidad_de_fragmentos.items())
            },
            "nota_de_mg_2": (
                "🔴 **: el presupuesto de evidencia es fijo** "
                "(`evidencia.n_fragmentos_por_obra`), así que dos brazos con "
                "`fragmentos/obra` distintos **no reciben la misma cantidad de texto**. Si "
                "un brazo gana en *groundedness*, estas dos cifras son lo único que "
                "distingue *ganó por los bordes* de *ganó por el volumen*. Sin ellas el "
                "resultado no es interpretable y es el **modo de falla 3** del proyecto: "
                "una medición sesgada presentada como medición."
            ),
            "nota": (
                "Todo se lee **del artefacto de corrida** y nada se recalcula (el módulo de evaluación). "
                "*Groundedness* fino necesita un humano y es la muestra de 25 del criterio de honestidad: **no "
                "está acá**, y se dice en vez de omitirse."
            ),
        }

def pares_comunes(
    corridas_a: Sequence[Any], corridas_b: Sequence[Any]
) -> frozenset[tuple[str, str]]:
    por_id_a = {c.consulta_id: c for c in corridas_a}
    por_id_b = {c.consulta_id: c for c in corridas_b}
    return frozenset(
        (consulta_id, artwork_id)
        for consulta_id in set(por_id_a) & set(por_id_b)
        for artwork_id in por_id_a[consulta_id].top5
        if artwork_id in set(por_id_b[consulta_id].top5)
    )

def _licencias_de(trozo: Mapping[str, Any]) -> tuple[str, ...]:
    licencias = trozo.get("licencias") or ()
    if isinstance(licencias, str):
        licencias = [p.strip() for p in licencias.split("|") if p.strip()]
    return tuple(licencias) or (trozo["license"],)

def _combinacion(licencias: Sequence[str]) -> str:
    return " + ".join(sorted(set(licencias)))

def generacion_medida(
    corridas: Sequence[Any],
    *,
    pares: frozenset[tuple[str, str]] | None = None,
    medidor: Any = None,
) -> GeneraciónMedida:
    if not corridas:
        raise ErrorDelProyecto(
            "no hay corridas sobre las que medir la generación. El estudio secundario se "
            "mide **del lado de la salida** (`la evaluación § El pool se congela`), así que "
            "primero hace falta correr el brazo: `poetry run tp3-responder --config "
            "config/c3.toml --variante-id <variante> set`."
        )
    primera = corridas[0]
    n_obras = n_explicaciones = n_abstenciones = 0
    n_fragmentos = n_referencias = n_obras_con_ref = 0
    n_colgadas = n_descartadas = n_byte_a_byte = 0
    evidencia_por_seccion: dict[str, int] = {}
    referencias_por_seccion: dict[str, int] = {}
    obras_por_seccion: dict[str, int] = {}
    obras_con_evidencia: dict[str, int] = {}
    n_mezclan_evidencia = n_mezclan_refs = 0
    caracteres = 0
    tokens = 0 if medidor is not None else None
    combinaciones_evidencia: dict[str, int] = {}
    combinaciones_refs: dict[str, int] = {}
    por_cantidad: dict[int, int] = {}
    ejemplos: list[dict] = []
    for corrida in corridas:
        evidencia = (corrida.estado.get("evidencia") or {}) if corrida.estado else {}
        seccion_de = {}
        texto_de = {}
        fragmento_de = {}
        for artwork_id, fragmentos in evidencia.items():
            incluida = pares is None or (corrida.consulta_id, artwork_id) in pares
            de_esta_obra = set()
            for f in fragmentos or ():
                seccion_de[f["fragment_id"]] = f.get("section") or SECCIÓN_SIN_EVIDENCIA
                texto_de[f["fragment_id"]] = f["texto"]
                fragmento_de[f["fragment_id"]] = f
                if not incluida:
                    continue
                n_fragmentos += 1
                seccion = seccion_de[f["fragment_id"]]
                evidencia_por_seccion[seccion] = evidencia_por_seccion.get(seccion, 0) + 1
                de_esta_obra.add(seccion)
                licencias = _licencias_de(f)
                combinacion = _combinacion(licencias)
                combinaciones_evidencia[combinacion] = (
                    combinaciones_evidencia.get(combinacion, 0) + 1
                )
                if len(set(licencias)) > 1:
                    n_mezclan_evidencia += 1
                caracteres += len(f["texto"])
                if tokens is not None:
                    tokens += medidor(f["texto"])
            for seccion in de_esta_obra:
                obras_con_evidencia[seccion] = obras_con_evidencia.get(seccion, 0) + 1
            if incluida:
                cuantos = len(fragmentos or ())
                por_cantidad[cuantos] = por_cantidad.get(cuantos, 0) + 1
        n_descartadas += sum(
            1
            for d in corrida.afirmaciones_descartadas
            if pares is None or (corrida.consulta_id, d.get("artwork_id")) in pares
        )
        for obra in ((corrida.salida or {}).get("obras") or ()):
            if pares is not None and (corrida.consulta_id, obra["artwork_id"]) not in pares:
                continue
            n_obras += 1
            if any(linea.strip() for linea in (obra.get("lineas") or ())):
                n_explicaciones += 1
            if obra.get("abstencion"):
                n_abstenciones += 1
            referencias = tuple(obra.get("referencias") or ())
            if referencias:
                n_obras_con_ref += 1
            vistas_en_la_obra = set()
            for ref in referencias:
                n_referencias += 1
                seccion = seccion_de.get(ref["fragment_id"], SECCIÓN_SIN_EVIDENCIA)
                referencias_por_seccion[seccion] = referencias_por_seccion.get(seccion, 0) + 1
                vistas_en_la_obra.add(seccion)
                if ref["fragment_id"] not in seccion_de:
                    n_colgadas += 1
                elif ref["texto_citado"] == texto_de[ref["fragment_id"]]:
                    n_byte_a_byte += 1
                licencias = _licencias_de(ref)
                combinacion = _combinacion(licencias)
                combinaciones_refs[combinacion] = combinaciones_refs.get(combinacion, 0) + 1
                if len(set(licencias)) > 1:
                    n_mezclan_refs += 1
                    if len(ejemplos) < 3:
                        ejemplos.append(
                            {
                                "consulta_id": corrida.consulta_id,
                                "artwork_id": obra["artwork_id"],
                                "fragment_id": ref["fragment_id"],
                                "campos": list(
                                    ref.get("campos")
                                    or fragmento_de.get(ref["fragment_id"], {}).get("campos")
                                    or ()
                                ),
                                "licencias": list(licencias),
                                "license_publicado": ref["license"],
                                "atribucion": ref.get("atribucion"),
                                "texto_citado": (
                                    ref["texto_citado"]
                                    if len(ref["texto_citado"]) <= 300
                                    else ref["texto_citado"][:150]
                                    + " […] "
                                    + ref["texto_citado"][-150:]
                                ),
                            }
                        )
            for seccion in vistas_en_la_obra:
                obras_por_seccion[seccion] = obras_por_seccion.get(seccion, 0) + 1
    return GeneraciónMedida(
        configuracion=primera.contenido["pipeline"]["configuracion"],
        config_id=primera.config_id,
        variante_id=primera.contenido["sellos"]["variante_id"],
        campos_de_knowledge=tuple(
            primera.contenido["configuracion_resuelta"]["fragmentacion"][
                "campos_de_knowledge"
            ]
        ),
        n_corridas=len(corridas),
        n_obras=n_obras,
        n_explicaciones=n_explicaciones,
        n_abstenciones=n_abstenciones,
        n_fragmentos_de_evidencia=n_fragmentos,
        evidencia_por_seccion=dict(sorted(evidencia_por_seccion.items())),
        n_referencias=n_referencias,
        n_obras_con_referencia=n_obras_con_ref,
        referencias_por_seccion=dict(sorted(referencias_por_seccion.items())),
        n_referencias_colgadas=n_colgadas,
        n_afirmaciones_descartadas=n_descartadas,
        n_citas_byte_a_byte=n_byte_a_byte,
        obras_por_seccion=dict(sorted(obras_por_seccion.items())),
        obras_con_evidencia_por_seccion=dict(sorted(obras_con_evidencia.items())),
        estrategia_de_chunking=(
            primera.contenido["configuracion_resuelta"]["fragmentacion"].get("estrategia", "")
        ),
        n_fragmentos_que_mezclan_licencias=n_mezclan_evidencia,
        n_referencias_que_mezclan_licencias=n_mezclan_refs,
        evidencia_por_combinacion_de_licencias=combinaciones_evidencia,
        referencias_por_combinacion_de_licencias=combinaciones_refs,
        ejemplos_de_mezcla_citada=tuple(ejemplos),
        tokens_de_evidencia=tokens,
        caracteres_de_evidencia=caracteres,
        obras_por_cantidad_de_fragmentos=por_cantidad,
    )

@dataclass(frozen=True)
class ÍtemsSinJuzgar:
    config_id: str
    configuracion: str
    pool_id: str
    estado_del_pool: str
    formo_el_pool: bool
    n_items: int
    n_sin_juzgar: int
    sin_juzgar_por_consulta: dict[str, int]
    obras_sin_juzgar: tuple[str, ...]

    @property
    def pct_sin_juzgar(self) -> float:
        return round(100 * self.n_sin_juzgar / self.n_items, 2) if self.n_items else 0.0

    def a_json(self) -> dict:
        return {
            "configuracion": self.configuracion,
            "config_id": self.config_id,
            "pool_id": self.pool_id,
            "estado_del_pool": self.estado_del_pool,
            "formo_el_pool": self.formo_el_pool,
            "n_items_del_top5": self.n_items,
            "n_sin_juzgar": self.n_sin_juzgar,
            "pct_sin_juzgar": self.pct_sin_juzgar,
            "n_obras_distintas_sin_juzgar": len(self.obras_sin_juzgar),
            "sin_juzgar_por_consulta": dict(self.sin_juzgar_por_consulta),
            "lectura": (
                "**0 por construcción**: esta configuración formó el pool, así que todo par "
                "de su top-5 está adentro por definición. El 0 no dice que la "
                "comparación se sostenga: dice que la pregunta no aplica."
                if self.formo_el_pool
                else "**Número real.** Esta configuración **no** formó el pool, así que sus "
                "ítems sí pueden estar sin juzgar — y por la convención de TREC que "
                "la evaluación adopta, un ítem sin juzgar **cuenta como no relevante**, o sea "
                "que su nDCG@5 estaría **subestimado** y hay que decirlo."
            ),
        }

def items_del_top5_sin_juzgar(corridas: Sequence[Any], *, pool) -> ÍtemsSinJuzgar:
    pares_del_pool = pool.pares
    config_ids_del_pool = {c["config_id"] for c in pool.configuraciones}
    n_items = 0
    sin_juzgar: dict[str, int] = {}
    obras: list[str] = []
    for corrida in corridas:
        for artwork_id in corrida.top5:
            n_items += 1
            if (corrida.consulta_id, artwork_id) not in pares_del_pool:
                sin_juzgar[corrida.consulta_id] = sin_juzgar.get(corrida.consulta_id, 0) + 1
                obras.append(artwork_id)
    primera = corridas[0]
    return ÍtemsSinJuzgar(
        config_id=primera.config_id,
        configuracion=primera.contenido["pipeline"]["configuracion"],
        pool_id=pool.pool_id,
        estado_del_pool=pool.estado,
        formo_el_pool=primera.config_id in config_ids_del_pool,
        n_items=n_items,
        n_sin_juzgar=sum(sin_juzgar.values()),
        sin_juzgar_por_consulta=dict(sorted(sin_juzgar.items())),
        obras_sin_juzgar=tuple(sorted(set(obras))),
    )

POBLACIÓN_COMPLETA = "completa"
POBLACIÓN_RESTRINGIDA = "restringida"

POLÍTICAS_DE_SIN_GRADO: tuple[str, ...] = ("cero", "condensada")

K_DE_LA_TABLA = 5

GRADO_RELEVANTE = 1
GRADO_ESTRICTO = 2

@dataclass(frozen=True)
class GroundTruthLeído:
    pool_id: str
    rubrica_id: str
    poblacion: str
    resoluciones: tuple[str, ...]
    grados: Mapping[tuple[str, str], int]
    por_resolucion: Mapping[str, int]
    n_filas_del_consenso: int
    n_sin_grado: int
    archivo: str

    @property
    def n(self) -> int:
        return len(self.grados)

    def grado_de(self, par: tuple[str, str]) -> int | None:
        return self.grados.get(par)

    def ideales_de(self, consulta_id: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                (g for (c, _), g in self.grados.items() if c == consulta_id),
                reverse=True,
            )
        )

    def a_json(self) -> dict:
        return {
            "pool_id": self.pool_id,
            "rubrica_id": self.rubrica_id,
            "poblacion": self.poblacion,
            "resoluciones_conservadas": list(self.resoluciones),
            "n_pares_con_grado": self.n,
            "n_filas_del_consenso": self.n_filas_del_consenso,
            "n_filas_sin_grado": self.n_sin_grado,
            "por_resolucion": dict(sorted(self.por_resolucion.items())),
            "archivo": self.archivo,
        }

def leer_consenso(
    almacén,
    raíz,
    *,
    pool_id: str,
    rubrica_id: str,
    resoluciones: Sequence[str] = ("acuerdo", "desempate"),
    poblacion: str = POBLACIÓN_COMPLETA,
) -> GroundTruthLeído:
    from tp3_rag.dominio import ground_truth as gt
    from tp3_rag.modulo_c.acuerdo import ruta_del_consenso

    ruta = ruta_del_consenso(raíz, pool_id, rubrica_id)
    if not almacén.existe(ruta):
        raise ErrorDelProyecto(
            f"no hay consenso en {ruta}. Las cuatro métricas de `la evaluación §A` **son sobre el "
            f"set anotado**: sin consenso no hay grados, y publicar la tabla sin ellos sería "
            f"publicar QS sola llamándola nDCG.\n"
            f"  Se escribe con `poetry run tp3-anotar kappa --designacion <…> --rubrica "
            f"{rubrica_id}`, que es una **lectura** (no gasta: no llama al proxy)."
        )
    _, filas = gt.partir_csv(almacén.leer_texto(ruta), ruta=ruta)
    conservadas = tuple(resoluciones)
    grados: dict[tuple[str, str], int] = {}
    por_resolución: dict[str, int] = {}
    sin_grado = 0
    for fila in filas:
        resuelto = (fila.get("como_se_resolvio") or "").strip()
        por_resolución[resuelto] = por_resolución.get(resuelto, 0) + 1
        crudo = (fila.get("grado") or "").strip()
        if not crudo:
            sin_grado += 1
            continue
        if resuelto not in conservadas:
            continue
        par = ((fila.get("consulta_id") or "").strip(), (fila.get("artwork_id") or "").strip())
        grados[par] = int(crudo)
    return GroundTruthLeído(
        pool_id=pool_id,
        rubrica_id=rubrica_id,
        poblacion=poblacion,
        resoluciones=conservadas,
        grados=grados,
        por_resolucion=por_resolución,
        n_filas_del_consenso=len(filas),
        n_sin_grado=sin_grado,
        archivo=str(ruta),
    )

def dcg(ganancias: Sequence[int | float]) -> float:
    return sum(g / log2(i + 1) for i, g in enumerate(ganancias, start=1))

def ndcg_at_k(
    ganancias: Sequence[int | float], ideales: Sequence[int | float], *, k: int = K_DE_LA_TABLA
) -> float | None:
    ideal = dcg(sorted(ideales, reverse=True)[:k])
    if ideal <= 0:
        return None
    return dcg(list(ganancias)[:k]) / ideal

def p_at_k(
    ganancias: Sequence[int | float], *, k: int = K_DE_LA_TABLA, umbral: int = GRADO_RELEVANTE
) -> float:
    return sum(1 for g in list(ganancias)[:k] if g >= umbral) / k

def rr(
    ganancias: Sequence[int | float], *, k: int = K_DE_LA_TABLA, umbral: int = GRADO_RELEVANTE
) -> float:
    for i, g in enumerate(list(ganancias)[:k], start=1):
        if g >= umbral:
            return 1.0 / i
    return 0.0

@dataclass(frozen=True)
class RankingJuzgado:
    consulta_id: str
    categoria: str
    politica: str
    ganancias: tuple[int, ...]
    sin_grado: int
    ideales: tuple[int, ...]

    @property
    def ndcg(self) -> float | None:
        return ndcg_at_k(self.ganancias, self.ideales)

    @property
    def p5(self) -> float:
        return p_at_k(self.ganancias)

    @property
    def p5_estricta(self) -> float:
        return p_at_k(self.ganancias, umbral=GRADO_ESTRICTO)

    @property
    def rr(self) -> float:
        return rr(self.ganancias)

    def a_json(self) -> dict:
        return {
            "consulta_id": self.consulta_id,
            "categoria": self.categoria,
            "grados_del_top5": list(self.ganancias),
            "n_sin_grado": self.sin_grado,
            "ndcg_at_5": None if self.ndcg is None else round(self.ndcg, 4),
            "p_at_5": round(self.p5, 4),
            "p_at_5_estricta": round(self.p5_estricta, 4),
            "rr": round(self.rr, 4),
            "ideal": list(self.ideales[:K_DE_LA_TABLA]),
        }

def juzgar_ranking(
    corrida: Any, *, gt: GroundTruthLeído, categoria: str, politica: str = "cero"
) -> RankingJuzgado:
    if politica not in POLÍTICAS_DE_SIN_GRADO:
        raise ErrorDelProyecto(
            f"`politica = {politica!r}` no es una de {list(POLÍTICAS_DE_SIN_GRADO)}. "
            f"`cero` es la convención de TREC que la evaluación adopta —*un ítem sin juzgar "
            f"cuenta como no relevante*—; `condensada` lo saca del ranking, y es la que "
            f"corresponde cuando el ítem **sí se juzgó** y la población lo excluye."
        )
    ganancias: list[int] = []
    sin_grado = 0
    for artwork_id in list(corrida.top5)[:K_DE_LA_TABLA]:
        grado = gt.grado_de((corrida.consulta_id, artwork_id))
        if grado is None:
            sin_grado += 1
            if politica == "cero":
                ganancias.append(0)
            continue
        ganancias.append(grado)
    return RankingJuzgado(
        consulta_id=corrida.consulta_id,
        categoria=categoria,
        politica=politica,
        ganancias=tuple(ganancias),
        sin_grado=sin_grado,
        ideales=gt.ideales_de(corrida.consulta_id),
    )

@dataclass(frozen=True)
class FilaDeRanking:
    configuracion: str
    config_id: str
    poblacion: str
    politica: str
    n_consultas: int
    n_items: int
    n_sin_grado: int
    rankings: tuple[RankingJuzgado, ...]

    @property
    def definidas(self) -> tuple[RankingJuzgado, ...]:
        return tuple(r for r in self.rankings if r.ndcg is not None)

    @property
    def sin_definir(self) -> tuple[str, ...]:
        return tuple(r.consulta_id for r in self.rankings if r.ndcg is None)

    @property
    def ndcg_sobre_definidas(self) -> float | None:
        d = self.definidas
        return sum(r.ndcg for r in d) / len(d) if d else None

    @property
    def ndcg_sobre_todas(self) -> float | None:
        if not self.rankings:
            return None
        return sum((r.ndcg or 0.0) for r in self.rankings) / len(self.rankings)

    @property
    def p5(self) -> float | None:
        return _promedio([r.p5 for r in self.rankings])

    @property
    def p5_estricta(self) -> float | None:
        return _promedio([r.p5_estricta for r in self.rankings])

    @property
    def mrr(self) -> float | None:
        return _promedio([r.rr for r in self.rankings])

    @property
    def pct_sin_grado(self) -> float:
        return round(100 * self.n_sin_grado / self.n_items, 2) if self.n_items else 0.0

    def por_categoria(self) -> dict[str, "FilaDeRanking"]:
        salida: dict[str, FilaDeRanking] = {}
        for categoria in CATEGORÍAS:
            de_la_categoria = tuple(r for r in self.rankings if r.categoria == categoria)
            if not de_la_categoria:
                continue
            salida[categoria] = FilaDeRanking(
                configuracion=self.configuracion,
                config_id=self.config_id,
                poblacion=self.poblacion,
                politica=self.politica,
                n_consultas=len(de_la_categoria),
                n_items=len(de_la_categoria) * K_DE_LA_TABLA,
                n_sin_grado=sum(r.sin_grado for r in de_la_categoria),
                rankings=de_la_categoria,
            )
        return salida

    def a_json(self) -> dict:
        return {
            "configuracion": self.configuracion,
            "config_id": self.config_id,
            "poblacion_del_ground_truth": self.poblacion,
            "politica_de_sin_grado": self.politica,
            "n_consultas": self.n_consultas,
            "n_items_del_top5": self.n_items,
            "ndcg_at_5_sobre_definidas": _r(self.ndcg_sobre_definidas),
            "n_consultas_con_ndcg_definido": len(self.definidas),
            "consultas_sin_ndcg_definido": list(self.sin_definir),
            "ndcg_at_5_sobre_todas": _r(self.ndcg_sobre_todas),
            "p_at_5": _r(self.p5),
            "p_at_5_estricta": _r(self.p5_estricta),
            "mrr": _r(self.mrr),
            "n_items_sin_grado": self.n_sin_grado,
            "pct_items_sin_grado": self.pct_sin_grado,
            "nota_del_ndcg": (
                "🔴 **Dos promedios, un numerador.** `sobre_definidas` divide por las "
                "consultas donde `IDCG > 0`; `sobre_todas` divide por todas y cuenta 0 las "
                "indefinidas, que es lo que la **decisión del acuerdo entre anotadores** declara para `c05-ambiguo` "
                "(*«no aporta a nDCG@5 y sí al denominador»*). El informe tiene que decir "
                "cuál publica."
            ),
        }

def _promedio(valores: Sequence[float]) -> float | None:
    return sum(valores) / len(valores) if valores else None

def _r(valor: float | None, n: int = 4) -> float | None:
    return None if valor is None else round(valor, n)

def fila_de_ranking(
    corridas: Sequence[Any],
    *,
    gt: GroundTruthLeído,
    categoria_de: Mapping[str, str],
    politica: str = "cero",
) -> FilaDeRanking:
    if not corridas:
        raise ErrorDelProyecto(
            "no hay corridas sobre las que medir el ranking. Las cuatro métricas de "
            "`la evaluación §A` son **sobre el set anotado**, así que hacen falta las dos cosas: "
            "las corridas de la fila y el consenso del pool."
        )
    rankings = tuple(
        juzgar_ranking(
            c, gt=gt, categoria=categoria_de.get(c.consulta_id, "?"), politica=politica
        )
        for c in sorted(corridas, key=lambda c: c.consulta_id)
    )
    primera = corridas[0]
    return FilaDeRanking(
        configuracion=primera.contenido["pipeline"]["configuracion"],
        config_id=primera.config_id,
        poblacion=gt.poblacion,
        politica=politica,
        n_consultas=len(rankings),
        n_items=len(rankings) * K_DE_LA_TABLA,
        n_sin_grado=sum(r.sin_grado for r in rankings),
        rankings=rankings,
    )

def orden_de_las_filas(
    filas: Sequence[FilaDeRanking], *, por: str = "ndcg_sobre_todas"
) -> tuple[str, ...]:
    def clave(fila: FilaDeRanking):
        valor = getattr(fila, por)
        return (-(valor if valor is not None else -1.0), fila.configuracion)

    return tuple(f.configuracion for f in sorted(filas, key=clave))

def comparar_poblaciones(
    completa: Sequence[FilaDeRanking],
    restringida: Sequence[FilaDeRanking],
    *,
    por: str = "ndcg_sobre_todas",
) -> dict:
    orden_a = orden_de_las_filas(completa, por=por)
    orden_b = orden_de_las_filas(restringida, por=por)
    por_nombre_a = {f.configuracion: f for f in completa}
    por_nombre_b = {f.configuracion: f for f in restringida}
    deltas = {}
    for nombre in sorted(set(por_nombre_a) & set(por_nombre_b)):
        a = getattr(por_nombre_a[nombre], por)
        b = getattr(por_nombre_b[nombre], por)
        deltas[nombre] = {
            "completa": _r(a),
            "restringida": _r(b),
            "delta": None if (a is None or b is None) else round(b - a, 4),
            "pct_sin_grado_en_la_restringida": por_nombre_b[nombre].pct_sin_grado,
        }
    return {
        "metrica": por,
        "orden_en_la_completa": list(orden_a),
        "orden_en_la_restringida": list(orden_b),
        "el_orden_cambia": orden_a != orden_b,
        "por_fila": deltas,
        "lectura": (
            "🔴 **El orden CAMBIA entre las dos poblaciones**, y eso *es* el resultado: la "
            "tabla de C0–C4 depende de los 133 ítems que resolvió el tercer anotador, que "
            "coincide con `ia-claude` en el 71 % y con `ia-codex` en el 22 % (el desempate). La "
            "limitación no queda acotada: queda **medida**."
            if orden_a != orden_b
            else "✅ **El orden NO cambia** entre la población completa (379 ítems, con los 133 "
            "del desempate) y la restringida (246 de acuerdo entre los dos primeros). La "
            "limitación del desempate **queda acotada**: el ordenamiento de C0–C4 no depende del "
            "tercio disputado, que es donde el consenso es de hecho el juicio de `ia-claude`."
        ),
        "nota": (
            "⚠️ **No se publica una de las dos.** No hay un tercer anotador neutral disponible "
            "—el proxy de la cátedra sirve solo OpenAI, y `ia-claude` es de otra familia—, así "
            "que **una sensibilidad es lo único honesto** que se puede decir del ground truth "
            "del tercio disputado."
        ),
    }

def piso_de_ruido(corridas_de, *, config_ids: Mapping[str, str]) -> dict:
    from collections import defaultdict

    de_la_tabla = {v: k for k, v in config_ids.items()}
    por_clave: dict[tuple[str, str], list] = defaultdict(list)
    for cid in corridas_de.existentes():
        try:
            corrida = corridas_de.leer(cid)
        except ErrorDelProyecto:
            continue
        if not corrida.es_del_set:
            continue
        por_clave[(corrida.config_id, corrida.consulta_id)].append(corrida)

    por_config: dict[str, dict] = {}
    for (config_id, _), lista in por_clave.items():
        if len(lista) < 2:
            continue
        ordenadas = sorted(lista, key=lambda c: c.ordinal)
        primera, última = ordenadas[0], ordenadas[-1]
        d = por_config.setdefault(
            config_id,
            {"n_pares": 0, "churn": 0, "identicos": 0, "jaccards": [], "ordinales": set()},
        )
        d["n_pares"] += 1
        d["churn"] += len([o for o in última.top5 if o not in set(primera.top5)])
        d["identicos"] += int(list(primera.top5) == list(última.top5))
        d["jaccards"].append(jaccard(primera.top5, última.top5))
        d["ordinales"] |= {c.ordinal for c in ordenadas}

    filas: dict[str, dict] = {}
    for fila, config_id in sorted(config_ids.items()):
        d = por_config.get(config_id)
        filas[fila] = {
            "config_id": config_id,
            "es_el_config_id_de_la_fila": True,
            "n_pares_con_dos_ordinales": (d or {}).get("n_pares", 0),
            "ordinales": sorted((d or {}).get("ordinales", ())),
            "top5_identico": f"{d['identicos']}/{d['n_pares']}" if d else None,
            "churn_medio_de_5": round(d["churn"] / d["n_pares"], 3) if d else None,
            "piso_de_qs": (
                round(1.0 - sum(d["jaccards"]) / len(d["jaccards"]), 4) if d else None
            ),
            "estado": "medido" if d else "sin medir en el `config_id` de la fila",
        }
    otros = {
        config_id: {
            "n_pares_con_dos_ordinales": d["n_pares"],
            "top5_identico": f"{d['identicos']}/{d['n_pares']}",
            "churn_medio_de_5": round(d["churn"] / d["n_pares"], 3),
            "piso_de_qs": round(1.0 - sum(d["jaccards"]) / len(d["jaccards"]), 4),
            "fila_que_declara_la_etiqueta": de_la_tabla.get(config_id),
        }
        for config_id, d in sorted(por_config.items())
        if config_id not in set(config_ids.values())
    }
    return {
        "por_fila": filas,
        "de_otros_config_id": otros,
        "nota": (
            "🔴 ** medido, y el piso NO es el mismo para las cinco filas.** el muestreo fijado, registrado y con su variabilidad medida "
            "empareja por `(config_id, consulta_id)`: una fila sin dos ordinales **no tiene "
            "piso medido** y sale `None`, no 0. ⚠️ **Y el 0,249 del hallazgo está medido "
            "sobre otro `config_id` que el de la fila C4 publicada** —las corridas con dos "
            "ordinales de C4 son de un pipeline anterior—, así que se reporta aparte en "
            "`de_otros_config_id` en vez de imputárselo a la fila."
        ),
        "nota_de_la_inclusion": (
            "🔑 **Para las filas sin medición hay un argumento, y es otra cosa que una "
            "medición.** Las cadenas de C0, C1 y C2 son **prefijos** de la de C3, y el piso "
            "de C3 está medido en **0,000 exacto** (top-5 idéntico en los 30 pares), así que "
            "los de C0–C2 quedan **acotados por inclusión** — no medidos. La fuente de no "
            "determinismo que el proyecto midió es el reranker, y el reranker "
            "solo está en C4. Las dos etiquetas no se mezclan: `medido` ≠ "
            "`acotado_por_inclusion`."
        ),
    }
