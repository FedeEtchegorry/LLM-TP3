from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
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
                "Una corrida con una degradación que nadie inyectó no agrega: entraría "
                "etiquetada como la fila que dice ser mientras se comportó como la "
                "anterior. Una con degradación inyectada a propósito tampoco agrega: es "
                "una prueba de respuesta garantizada y se reporta aparte. Una etapa "
                "ausente por configuración no degrada —no corre—, y por eso las filas "
                "que no tienen esa etapa sí agregan. Los conteos de `fallidas` y de "
                "degradación pueden solaparse: una corrida que falló y además degradó "
                "aparece en los dos."
            ),
        }

def separar_por_degradacion(corridas: Sequence[Any]) -> Agregables:
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
                f"{fila}. Las dos no pueden discrepar: la etiqueta es contra lo que "
                f"la precondición declarada compara la cadena, y acá es contra lo que se agrupa la población de "
                f"un número publicado."
            )
        salida[fila] = cfg.config_id
    return salida

@dataclass(frozen=True)
class CostoMedido:
    configuracion: str
    n_corridas: int
    n_candidatos: int | None
    por_rol: Mapping[str, Mapping[str, int]]
    por_etapa: Mapping[str, Mapping[str, int]]

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
            "usd": None,
            "nota_del_usd": (
                "⚠️ USD ausente, no cero (, ): `[proveedor]` solo declara el "
                "precio de `openai/text-embedding-3-small`. *El día que alguien lea el precio "
                "de `gpt-4.1-mini` con su fecha y su URL, se anota ahí y en ningún otro "
                "lado.* Lo que está medido son los tokens."
            ),
        }

def corridas_de_la_tabla(corridas_de, *, raiz=None) -> dict:
    config_ids = config_ids_de_la_tabla(raiz=raiz)
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
            "Requests al proxy, que es la unidad que se factura. La población es "
            "todas las corridas en disco —huérfanas, de falla inyectada, ad-hoc y "
            "brazos de barrido incluidos— más los artefactos con libro mayor que no "
            "son corridas. Es otra población que la tabla de ablación, que solo lleva "
            "las cinco filas declaradas, y sumar solo esa subestimaría el gasto. El "
            "embebido de imágenes y el índice no entran: corren local."
        ),
    }

def costo_por_configuracion(corridas: Sequence[Any]) -> tuple[CostoMedido, ...]:
    acumulado: dict[str, dict] = {}
    for corrida in corridas:
        etiqueta = (corrida.contenido.get("pipeline") or {}).get("configuracion", "?")
        d = acumulado.setdefault(
            etiqueta,
            {"n": 0, "n_candidatos": None, "por_rol": {}, "por_etapa": {}},
        )
        d["n"] += 1
        resuelta = corrida.contenido.get("configuracion_resuelta") or {}
        d["n_candidatos"] = (resuelta.get("consulta") or {}).get("n_candidatos")
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
    return tuple(
        CostoMedido(
            configuracion=etiqueta,
            n_corridas=d["n"],
            n_candidatos=d["n_candidatos"],
            por_rol={k: v for k, v in sorted(d["por_rol"].items())},
            por_etapa={k: v for k, v in sorted(d["por_etapa"].items())},
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
            "Una corrida, tres valores de N. El reranker puntúa una candidata por "
            "llamada y el mensaje de una candidata es idéntico carácter por carácter "
            "dentro de un fusionado de cincuenta y de veinte —hay un test que lo "
            "verifica, y ni N ni el rango entran al texto—, así que derivar N es "
            "filtrar por rango fusionado y reordenar con esos mismos puntajes. Y es "
            "más correcto, no solo más barato: tres corridas separadas compararían "
            "también la variabilidad del reranker, que es del orden de lo que el "
            "propio eje mueve. Anidado, la única diferencia entre las tres filas es N."
        ),
        "nota_de_la_latencia": (
            "⚠️ La latencia por N es una proyección y la etiqueta lo dice. Lo medido "
            "es el segundo por llamada de esta corrida; las llamadas son secuenciales (un "
            "candidato por llamada, el diseño de la arquitectura), así que `segundos(N) = por_llamada × N`. "
            "Es la misma distinción *medido / proyectado* que y la séptima vez que el "
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
            "Una fila está medida y las otras dos son proyección, y la etiqueta "
            "`medido` lo dice fila por fila. El supuesto es que el costo medio por "
            "llamada se mantenga al pedir más candidatas: la llamada lleva la imagen de "
            "la persona más la de la candidata, y el peso de la segunda varía por obra. "
            "La media de las llamadas medidas es "
            "el mejor estimador que hay y no es una medición — es la sexta vez que el "
            "proyecto distingue *medido* de *proyectado*."
        ),
        "nota_de_d13": (
            "El contraste con la medición a mano es el control de esta cifra. El modelo "
            "se eligió midiendo a mano un par de imágenes, y la cifra publicable es la "
            "que emite el pipeline. Esta es esa cifra: si se despega de la de "
            "referencia, cambió el modelo o el nivel de detalle con que se mandan las "
            "imágenes —o la medición a mano no tenía la forma de la llamada real— y hay "
            "que volver a medir antes de barrer N."
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
            "nota": (
                "Todo se lee del artefacto de corrida y nada se recalcula. Evaluar si "
                "la explicación se apoya de verdad en la evidencia necesita una persona "
                "y se hace sobre una muestra aparte: no está acá, y se dice en vez de "
                "omitirse."
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

def generacion_medida(
    corridas: Sequence[Any], *, pares: frozenset[tuple[str, str]] | None = None
) -> GeneraciónMedida:
    if not corridas:
        raise ErrorDelProyecto(
            "no hay corridas sobre las que medir la generación. El estudio secundario se "
            "mide del lado de la salida, así que "
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
    for corrida in corridas:
        evidencia = (corrida.estado.get("evidencia") or {}) if corrida.estado else {}
        seccion_de = {}
        texto_de = {}
        for artwork_id, fragmentos in evidencia.items():
            incluida = pares is None or (corrida.consulta_id, artwork_id) in pares
            de_esta_obra = set()
            for f in fragmentos or ():
                seccion_de[f["fragment_id"]] = f.get("section") or SECCIÓN_SIN_EVIDENCIA
                texto_de[f["fragment_id"]] = f["texto"]
                if not incluida:
                    continue
                n_fragmentos += 1
                seccion = seccion_de[f["fragment_id"]]
                evidencia_por_seccion[seccion] = evidencia_por_seccion.get(seccion, 0) + 1
                de_esta_obra.add(seccion)
            for seccion in de_esta_obra:
                obras_con_evidencia[seccion] = obras_con_evidencia.get(seccion, 0) + 1
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
                "0 por construcción: esta configuración formó el pool, así que todo par "
                "de su top-5 está adentro por definición (`MP-3`). El 0 no dice que la "
                "comparación se sostenga: dice que la pregunta no aplica."
                if self.formo_el_pool
                else "Número real. Esta configuración no formó el pool, así que sus "
                "ítems sí pueden estar sin juzgar — y por la convención de TREC, un "
                "ítem sin juzgar cuenta como no relevante, o sea que su nDCG estaría "
                "subestimado y hay que decirlo."
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
