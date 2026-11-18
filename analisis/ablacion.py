"""Ablación de C0 a C4 (cascada con filtro por metadata): métricas y gráficos. No gasta.

    python analisis/ablacion.py                 # tabla + JSON + figuras sobre fondo blanco
    python analisis/ablacion.py --tema oscuro   # las mismas figuras con el fondo de la presentación
    python analisis/ablacion.py --sin-figuras   # solo la tabla y el JSON

Lee las corridas y los juicios que ya están en `artifacts/` (los mismos que `cascada.py`)
y escribe en `artifacts/analisis/`:

  ablacion.json            todas las cifras de la tabla
  ablacion-metricas.png    nDCG@5 con IC, saltos pareados, precisión, MRR y QS
  ablacion-<metrica>.png   cada métrica sola, para una diapositiva
  ablacion-por-tipo.png    nDCG@5 por tipo de consulta y configuración
  ablacion-costo.png       USD por rol, latencia por etapa y calidad contra costo

Las cinco filas:
  C0  solo imagen (CLIP)            C3  + cascada con filtro por metadata
  C1  + pedido crudo y generación   C4  + reranker con LLM (arquitectura final)
  C2  + query rewriting

C0 a C2 no tienen filtro por metadata porque el filtro vive en la rama de texto, que
aparece recién en C3. Si una configuración tiene varios `config_id` completos se usa el
de la corrida más reciente.

Costo: el libro mayor de una corrida no registra la llamada del pre-retrieval cuando el
plan salió de caché. Para que las filas sean comparables, el uso de esa llamada se toma
del plan guardado (`artifacts/planes/<plan_id>.json`) y se suma. La latencia no se puede
reconstruir así: queda la medida, sin esa llamada, y se declara.
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
from pathlib import Path

import cascada as c

FILAS = [  # (clave, nombre corto, qué agrega)
    ("C0", "C0", "solo imagen"),
    ("C1", "C1", "+ pedido crudo"),
    ("C2", "C2", "+ query rewriting"),
    ("C3·filtro100·metadata", "C3", "+ cascada con filtro"),
    ("C4·filtro100·metadata·rerank20", "C4", "+ reranker"),
]
ROLES_LLM = ("planificador", "aclarador", "reranker", "generador")
PLANES = c.RAIZ / "artifacts" / "planes"
GRUPOS_DE_ETAPAS = {  # para la latencia apilada: lo que no es LLM va junto
    "Recuperación y armado": ("recuperador_visual", "recuperador_textual", "fusionador", "recolector_de_evidencia", "renderizador", "planificador"),
    "Generador": ("generador",),
    "Reranker": ("reranker",),
}


# ---------------------------------------------------------------- carga

def _clave(cfg: dict) -> str | None:
    nombre = cfg["consulta"]["configuracion"]
    if nombre in ("C0", "C1", "C2"):
        return nombre
    if (cfg.get("fusion") or {}).get("estrategia") != "cascada":
        return None
    return c.etiqueta(cfg)


def _uso_del_plan(estado: dict) -> dict:
    plan_id = (estado.get("plan") or {}).get("plan_id")
    archivo = PLANES / f"{plan_id}.json"
    if not plan_id or not archivo.exists():
        return {}
    with open(archivo, encoding="utf-8") as f:
        return json.load(f).get("uso") or {}


def cargar() -> tuple[dict[str, dict[str, dict]], dict]:
    """{clave: {consulta_id: corrida resumida}} y la tabla de precios de la fila final."""
    grupos: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
    buscadas = {clave for clave, _, _ in FILAS}
    for archivo in sorted(glob.glob(str(c.RAIZ / "artifacts" / "corridas" / "*" / "corrida.json"))):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        cfg, estado = corrida["configuracion_resuelta"], corrida["estado"]
        if corrida["consulta_id"].endswith("-mixto"):
            continue  # el set mixto es complementario y se mide aparte
        clave = _clave(cfg)
        if clave not in buscadas or not corrida.get("agrega_a_la_tabla", True) or not estado.get("top5"):
            continue
        if (estado.get("entrada") or {}).get("procedencia") == "ad_hoc":
            continue
        if any(r.get("estado") == "sin_puntaje" for r in (estado.get("rerank") or [])):
            continue  # reranker fallido: no es una medicion del reranker
        fila = {
            "ordinal": int(corrida.get("ordinal", 1)),
            "fecha": corrida.get("fecha") or "",
            "top5": [x["artwork_id"] if isinstance(x, dict) else x for x in estado["top5"]],
            "libro": {r: l for r, l in (corrida.get("libro_mayor") or {}).items() if isinstance(l, dict)},
            "segundos": (corrida.get("latencia") or {}).get("segundos_de_la_corrida"),
            "por_etapa": (corrida.get("latencia") or {}).get("por_etapa") or {},
            "modelos": {r: b.get("modelo") for r, b in (cfg.get("bindings") or {}).items()},
            "modelo_de_embedding": (cfg.get("embebido_texto") or {}).get("modelo"),
            "precios": cfg.get("proveedor") or {},
            "planifica": "planificador" in cfg["consulta"]["etapas"],
            "uso_del_plan": _uso_del_plan(estado),
        }
        previo = grupos[(clave, corrida["config_id"])].get(corrida["consulta_id"])
        if previo is None or fila["ordinal"] > previo["ordinal"]:
            grupos[(clave, corrida["config_id"])][corrida["consulta_id"]] = fila
    elegidos: dict[str, dict[str, dict]] = {}
    for (clave, _), corridas in grupos.items():
        if len(corridas) < 30:
            continue
        mas_nueva = max(x["fecha"] for x in corridas.values())
        if clave not in elegidos or mas_nueva > max(x["fecha"] for x in elegidos[clave].values()):
            elegidos[clave] = corridas
    final = elegidos.get(FILAS[-1][0]) or {}
    precios = next(iter(final.values()))["precios"] if final else {}
    return elegidos, precios


# ---------------------------------------------------------------- métricas

def mrr(ev, consulta, ids, umbral=1) -> float:
    """1 / posición de la primera obra con grado >= umbral dentro del top-5 (0 si no hay)."""
    for i, artwork_id in enumerate(ids[: c.K], start=1):
        if ev.gt.get((consulta, artwork_id), 0) >= umbral:
            return 1 / i
    return 0.0


def qs(corridas) -> float:
    """1 − media de Jaccard entre los top-5 de pedidos distintos sobre la misma imagen."""
    por_imagen = collections.defaultdict(list)
    for consulta, fila in corridas.items():
        por_imagen[consulta.split("-")[0]].append(set(fila["top5"]))
    valores = []
    for tops in por_imagen.values():
        pares = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
        if pares:
            valores.append(1 - c.media(pares))
    return c.media(valores)


def percentil(valores, p) -> float:
    valores = sorted(valores)
    k = (len(valores) - 1) * p
    i = int(k)
    return valores[i] + (valores[min(i + 1, len(valores) - 1)] - valores[i]) * (k - i)


def uso(fila, precios) -> dict:
    """Tokens, llamadas LLM y USD de una corrida, con la llamada del plan sumada si salió de caché."""
    entrada = precios.get("precio_usd_por_millon_de_tokens") or {}
    salida = precios.get("precio_usd_por_millon_de_tokens_de_salida") or {}
    libros = dict(fila["libro"])
    plan_de_cache = fila["planifica"] and "planificador" not in libros and bool(fila["uso_del_plan"])
    if plan_de_cache:
        libros["planificador"] = {**fila["uso_del_plan"], "n_llamadas": 1}
    total = {"tokens_in": 0, "tokens_out": 0, "llamadas_llm": 0, "usd": 0.0, "usd_por_rol": {}}
    for rol, libro in libros.items():
        modelo = fila["modelos"].get(rol) or (fila["modelo_de_embedding"] if rol == "embedding_texto" else None)
        usd = libro.get("tokens_in", 0) * entrada.get(modelo, 0) / 1e6 + libro.get("tokens_out", 0) * salida.get(modelo, 0) / 1e6
        total["tokens_in"] += libro.get("tokens_in", 0)
        total["tokens_out"] += libro.get("tokens_out", 0)
        total["usd"] += usd
        if usd:
            total["usd_por_rol"][rol] = usd
        if rol in ROLES_LLM:
            total["llamadas_llm"] += libro.get("n_llamadas", 0)
    total["plan_de_cache"] = plan_de_cache
    return total


def tabla(ev, elegidos, precios) -> dict:
    salida = {}
    previa = None
    for clave, nombre, agrega in FILAS:
        corridas = elegidos.get(clave)
        if not corridas:
            continue
        consultas = sorted(corridas)
        top = lambda q: corridas[q]["top5"]  # noqa: E731
        usos = [uso(corridas[q], precios) for q in consultas]
        segundos = [corridas[q]["segundos"] for q in consultas if corridas[q]["segundos"] is not None]
        roles = sorted({r for u in usos for r in u["usd_por_rol"]})
        fila = {
            "clave": clave,
            "agrega": agrega,
            "consultas": len(consultas),
            "ndcg": c.por_categoria(consultas, lambda q: ev.ndcg(q, top(q))),
            "ic95": c.ic95(ev.ndcg(q, top(q)) for q in consultas),
            "ndcg_por_consulta": {cat: [ev.ndcg(q, top(q)) for q in consultas if c.categoria(q) == cat] for cat in c.CATEGORIAS},
            "p5": round(c.media(ev.p5(q, top(q)) for q in consultas), 3),
            "p5_estricta": round(c.media(ev.p5(q, top(q), 2) for q in consultas), 3),
            "mrr": round(c.media(mrr(ev, q, top(q)) for q in consultas), 3),
            "qs": round(qs(corridas), 3),
            "grado2_por_top5": round(c.media(c.K * ev.p5(q, top(q), 2) for q in consultas), 2),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, top(q))) for q in consultas) / (c.K * len(consultas)), 1),
            "usd_por_consulta": round(c.media(u["usd"] for u in usos), 4),
            "usd_por_rol": {r: round(c.media(u["usd_por_rol"].get(r, 0) for u in usos), 4) for r in roles},
            "tokens_in": round(c.media(u["tokens_in"] for u in usos)),
            "tokens_out": round(c.media(u["tokens_out"] for u in usos)),
            "llamadas_llm": round(c.media(u["llamadas_llm"] for u in usos), 2),
            "segundos_media": round(c.media(segundos), 1) if segundos else None,
            "segundos_p95": round(percentil(segundos, 0.95), 1) if segundos else None,
            "plan_de_cache": any(u["plan_de_cache"] for u in usos),
            "segundos_por_grupo": {g: round(c.media(sum(corridas[q]["por_etapa"].get(e, 0) for e in etapas) for q in consultas), 1)
                                   for g, etapas in GRUPOS_DE_ETAPAS.items()},
            "fecha": max(corridas[q]["fecha"] for q in consultas)[:10],
        }
        if previa is not None:
            deltas = [ev.ndcg(q, top(q)) - ev.ndcg(q, elegidos[previa][q]["top5"]) for q in consultas if q in elegidos[previa]]
            fila["delta_ndcg_contra_la_anterior"] = {"delta": round(c.media(deltas), 3), "ic95": c.ic95(deltas)}
        salida[nombre] = fila
        previa = clave
    return salida


# ---------------------------------------------------------------- figuras

TEMAS = {
    "oscuro": {"fondo": "#1a1612", "texto": "#f6f0e4", "secundario": "#bdb5a6", "linea": "#4a433a",
               "barra": "#8f8778", "final": "#ddb15c",
               "tipos": {"conceptual": "#3987e5", "visual": "#d95926", "ambiguo": "#199e70", "mixto": "#a77fe0"}},
    "claro": {"fondo": "#ffffff", "texto": "#0b0b0b", "secundario": "#52514e", "linea": "#d9d8d2",
              "barra": "#a3a29a", "final": "#b8860b",
              "tipos": {"conceptual": "#2a78d6", "visual": "#eb6834", "ambiguo": "#1baf7a", "mixto": "#8a5cc2"}},
}
METRICAS = [  # (clave, título, archivo, decimales, máximo del eje)
    ("ndcg", "nDCG@5", "ndcg", 3, 1.0),
    ("p5", "P@5 (grado ≥ 1)", "p5", 3, 1.0),
    ("p5_estricta", "P@5 estricta (grado 2)", "p5-estricta", 3, 1.0),
    ("mrr", "MRR", "mrr", 3, 1.0),
    ("qs", "QS (cuánto cambia el top-5 con el pedido)", "qs", 3, 1.0),
]  # «obras de grado 2 por top-5» no va: es 5 × P@5 estricta.
CORTO = {"C0": "solo imagen", "C1": "+ pedido", "C2": "+ reescritura", "C3": "+ cascada", "C4": "+ reranker"}
COSTOS = [
    ("usd_por_consulta", "USD por consulta", 4),
    ("tokens_in", "Tokens de entrada por consulta", 0),
    ("segundos_media", "Latencia media (s)", 1),
    ("llamadas_llm", "Llamadas LLM por consulta", 1),
]


def numero(valor, decimales) -> str:
    if decimales == 0:
        return f"{valor:,.0f}".replace(",", ".")
    return f"{valor:.{decimales}f}".replace(".", ",")


def _valor(fila, clave):
    return fila["ndcg"]["todas"] if clave == "ndcg" else fila[clave]


def _eje(ax, t, titulo):
    ax.set_facecolor(t["fondo"])
    ax.set_title(titulo, color=t["texto"], fontsize=12, fontweight="bold", loc="left", pad=10)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(t["linea"])
    ax.tick_params(axis="x", colors=t["texto"], length=0, labelsize=10)
    ax.tick_params(axis="y", colors=t["secundario"], length=0, labelsize=8)
    ax.yaxis.grid(True, color=t["linea"], linewidth=0.6)
    ax.set_axisbelow(True)


def _barras(ax, t, tab, clave, titulo, decimales, maximo=None, con_delta=True, detalle=True):
    """Una barra por configuración; la final va resaltada. Arriba, el valor y la diferencia con la anterior."""
    nombres = list(tab)
    valores = [_valor(tab[n], clave) or 0 for n in nombres]
    colores = [t["final"] if n == nombres[-1] else t["barra"] for n in nombres]
    _eje(ax, t, titulo)
    ax.bar(range(len(nombres)), valores, width=0.56, color=colores, edgecolor=t["fondo"], linewidth=1.5)
    techo = maximo or (max(valores) * 1.28 if max(valores) else 1)
    ax.set_ylim(0, techo)
    if clave == "ndcg":
        for i, n in enumerate(nombres):
            bajo, alto = tab[n]["ic95"]
            ax.plot([i, i], [bajo, alto], color=t["texto"], linewidth=1.2, solid_capstyle="butt")
    for i, valor in enumerate(valores):
        base = tab[nombres[i]]["ic95"][1] if clave == "ndcg" else valor
        ax.text(i, base + techo * 0.02, numero(valor, decimales), ha="center", va="bottom", color=t["texto"], fontsize=10, fontweight="bold")
        if con_delta and i:
            delta = valor - valores[i - 1]
            signo = "+" if delta >= 0 else "−"
            ax.text(i, base + techo * 0.085, f"{signo}{numero(abs(delta), decimales)}", ha="center", va="bottom", color=t["secundario"], fontsize=8)
    # En los paneles chicos la segunda línea se pisa: ahí va solo el nombre y la leyenda al pie.
    ax.set_xticks(range(len(nombres)), [f"{n}\n{CORTO.get(n, tab[n]['agrega'])}" if detalle else n for n in nombres])
    ax.yaxis.set_major_formatter(lambda y, _: numero(y, 3 if techo < 0.1 else (1 if techo <= 1 else 0)))


def _saltos(ax, t, tab):
    """Salto pareado de nDCG@5 de cada configuración contra la anterior, con IC 95 %. Es mejora si el IC no cruza 0."""
    nombres = [n for n in tab if tab[n].get("delta_ndcg_contra_la_anterior")]
    _eje(ax, t, "Salto de nDCG@5 contra la anterior")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=t["linea"], linewidth=0.6)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", colors=t["texto"], labelsize=10)
    ax.tick_params(axis="x", colors=t["secundario"], labelsize=8)
    ax.axvline(0, color=t["secundario"], linewidth=0.9)
    for y, n in enumerate(nombres):
        d = tab[n]["delta_ndcg_contra_la_anterior"]
        bajo, alto = d["ic95"]
        mejora = bajo > 0
        color = t["final"] if mejora else t["barra"]
        ax.plot([bajo, alto], [y, y], color=color, linewidth=2.2, solid_capstyle="butt")
        ax.plot(d["delta"], y, "o", color=color, markersize=8)
        signo = "+" if d["delta"] >= 0 else "−"
        ax.text(alto + 0.008, y, f"{signo}{numero(abs(d['delta']), 3)}", va="center", color=t["texto"],
                fontsize=10, fontweight="bold" if mejora else "normal")
    ax.set_yticks(range(len(nombres)), [f"{n} {CORTO.get(n, tab[n]['agrega'])}" for n in nombres])
    ax.set_ylim(len(nombres) - 0.4, -0.5)
    ax.set_xlim(-0.1, 0.33)
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.xaxis.set_major_formatter(lambda x, _: ("+" if x > 0 else "") + numero(x, 1))


def _precision(ax, t, tab):
    """P@5 laxa y estricta lado a lado: la distancia entre las dos es cuánto del top-5 es solo «algo relevante»."""
    nombres = list(tab)
    _eje(ax, t, "Precisión en el top-5")
    series = [("p5", "grado ≥ 1", t["barra"]), ("p5_estricta", "grado 2", t["final"])]
    ancho = 0.38
    for j, (clave, etiqueta, color) in enumerate(series):
        xs = [i + (j - 0.5) * ancho for i in range(len(nombres))]
        ys = [tab[n][clave] for n in nombres]
        ax.bar(xs, ys, width=ancho, color=color, edgecolor=t["fondo"], linewidth=1.2, label=etiqueta)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, numero(y, 2), ha="center", va="bottom", color=t["texto"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(nombres)), nombres)
    ax.yaxis.set_major_formatter(lambda y, _: numero(y, 1))
    leyenda = ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
    for texto in leyenda.get_texts():
        texto.set_color(t["texto"])


def _rotulo_de_uso(nombre, fila) -> str:
    llamadas = round(fila["llamadas_llm"])
    if not llamadas:
        return f"{nombre}\nSin LLM"
    return f"{nombre}\n{llamadas} llamada{'s' if llamadas > 1 else ''}\n{numero(fila['tokens_in'] / 1000, 0)} mil tokens"


def _guardar(fig, t, archivo) -> str:
    fig.patch.set_facecolor(t["fondo"])
    fig.savefig(c.SALIDA / archivo, dpi=200, facecolor=t["fondo"])
    return archivo


def figuras(tab, tema, juicios) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Sin matplotlib no hay figuras: `poetry install --with figuras`.")
        return []
    t = TEMAS[tema]
    hechas = []
    filas = "   ".join(f"{n}: {f['agrega'].lstrip('+ ')}" for n, f in tab.items())
    miles = numero(juicios, 0)

    # Arriba, la métrica principal y lo que aporta cada componente; abajo, las secundarias.
    fig = plt.figure(figsize=(16, 9))
    grilla = GridSpec(2, 6, figure=fig, height_ratios=(1.15, 1))
    ax = fig.add_subplot(grilla[0, :3])
    _barras(ax, t, tab, "ndcg", "nDCG@5 (IC 95 %)", 3, 1.0, con_delta=False)
    _saltos(fig.add_subplot(grilla[0, 3:]), t, tab)
    _precision(fig.add_subplot(grilla[1, :2]), t, tab)
    _barras(fig.add_subplot(grilla[1, 2:4]), t, tab, "mrr", "MRR", 2, 1.0, con_delta=False, detalle=False)
    _barras(fig.add_subplot(grilla[1, 4:]), t, tab, "qs", "QS (diversidad entre pedidos)", 2, 1.0, con_delta=False, detalle=False)
    pie = (f"30 consultas (10 por tipo) · {miles} juicios 0/1/2; obra sin juicio = 0 · IC 95 %: bootstrap sobre consultas "
           "(4.000 remuestreos); saltos pareados por consulta · en dorado, salto con IC 95 % pareado mayor que 0\n"
           "C1: α = 0,5 imagen + texto · C4: gpt-4.1-mini, 20 candidatas, prompt v1, corrida más reciente de tres · "
           "QS: 1 − Jaccard medio entre los top-5 de los tres pedidos de cada imagen")
    fig.text(0.012, 0.012, pie, color=t["secundario"], fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.2, w_pad=2.5)
    hechas.append(_guardar(fig, t, "ablacion-metricas.png")); plt.close(fig)

    for clave, titulo, archivo, decimales, maximo in METRICAS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _barras(ax, t, tab, clave, titulo, decimales, maximo)
        fig.tight_layout()
        hechas.append(_guardar(fig, t, f"ablacion-{archivo}.png")); plt.close(fig)

    # nDCG@5 por tipo: una línea por tipo a lo largo de la ablación. Las mixtas van en su
    # propio panel porque se juzgan con otra rúbrica y su nivel no se compara con el resto.
    nombres = list(tab)
    xs = list(range(len(nombres)))
    mixtas = {}
    if (c.SALIDA / "mixtas.json").exists():
        mixtas = json.loads((c.SALIDA / "mixtas.json").read_text(encoding="utf-8"))["configuraciones"]
    con_mixtas = all(n in mixtas for n in nombres)
    paneles = [(tipo, tipo.capitalize(), [tab[n]["ndcg"][tipo] for n in nombres],
                [c.ic95(tab[n]["ndcg_por_consulta"][tipo]) for n in nombres]) for tipo in c.CATEGORIAS]
    if con_mixtas:
        paneles.append(("mixto", "Mixto", [mixtas[n]["ndcg"] for n in nombres], [mixtas[n]["ic95"] for n in nombres]))
    fig, ejes = plt.subplots(1, len(paneles), figsize=(4.1 * len(paneles), 5.2), sharey=True)
    for k, (eje, (tipo, titulo, ys, ics)) in enumerate(zip(ejes, paneles)):
        color = t["tipos"][tipo]
        _eje(eje, t, titulo)
        bajos, altos = zip(*ics)
        eje.fill_between(xs, bajos, altos, color=color, alpha=0.13, linewidth=0)
        eje.plot(xs, ys, marker="o", markersize=7, linewidth=2.4, color=color)
        for x, y in zip(xs, ys):
            eje.annotate(numero(y, 2), (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, color=t["texto"], fontweight="bold")
        eje.set_xticks(xs, nombres, fontsize=10)
        eje.set_xlim(-0.4, len(nombres) - 0.6)
        eje.set_ylim(0, 1.05)
        eje.yaxis.set_major_formatter(lambda y, _: numero(y, 1))
        if tipo == "mixto":
            eje.axvline(-0.4, color=t["linea"], linewidth=1)
    filas = "   ".join(f"{n}: {CORTO.get(n, tab[n]['agrega']).lstrip('+ ')}" for n in nombres)
    pie = (f"{filas}\nBanda sombreada: IC 95 % por bootstrap sobre las 10 consultas de cada tipo · {miles} juicios 0/1/2 en el set de 30; obra sin juicio = 0.")
    if con_mixtas:
        pie += ("\nMixto: pedidos que combinan cambio visual, restricción de metadata y adjetivo vago; grado = cambio visual (A) + restricción verificada en metadata (B). "
                "Su nivel no se compara con el de los otros tipos.")
    fig.text(0.012, 0.015, pie, color=t["secundario"], fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.13 if con_mixtas else 0.09, 1, 1), w_pad=2.5)
    hechas.append(_guardar(fig, t, "ablacion-por-tipo.png")); plt.close(fig)

    # Costo: de dónde sale la plata, de dónde sale el tiempo y qué se compra con eso.
    fig, (ax_usd, ax_seg, ax_cal) = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": (1, 1, 1.1)})
    nombres = list(tab)
    xs = range(len(nombres))
    claro, oscuro = "#cfcec7", "#6f6e67"
    roles = [("planificador", "Planificador", claro), ("generador", "Generador", oscuro), ("reranker", "Reranker", t["final"])]
    _eje(ax_usd, t, "USD por consulta")
    base = [0.0] * len(nombres)
    for rol, etiqueta, color in roles:
        ys = [tab[n]["usd_por_rol"].get(rol, 0) for n in nombres]
        ax_usd.bar(xs, ys, bottom=base, width=0.56, color=color, edgecolor=t["fondo"], linewidth=1, label=etiqueta)
        base = [b + y for b, y in zip(base, ys)]
    for i, n in enumerate(nombres):
        f = tab[n]
        ax_usd.text(i, base[i] + 0.0006, numero(f["usd_por_consulta"], 4), ha="center", va="bottom", color=t["texto"], fontsize=10, fontweight="bold")
    ax_usd.set_ylim(0, max(base) * 1.18)
    ax_usd.set_xticks(xs, [_rotulo_de_uso(n, tab[n]) for n in nombres], fontsize=9)
    ax_usd.yaxis.set_major_formatter(lambda y, _: numero(y, 3))

    _eje(ax_seg, t, "Latencia media (s)")
    base = [0.0] * len(nombres)
    for grupo, color in zip(GRUPOS_DE_ETAPAS, (claro, oscuro, t["final"])):
        ys = [tab[n]["segundos_por_grupo"][grupo] for n in nombres]
        ax_seg.bar(xs, ys, bottom=base, width=0.56, color=color, edgecolor=t["fondo"], linewidth=1, label=grupo)
        base = [b + y for b, y in zip(base, ys)]
    for i, n in enumerate(nombres):
        ax_seg.text(i, base[i] + 0.6, numero(tab[n]["segundos_media"], 1), ha="center", va="bottom", color=t["texto"], fontsize=10, fontweight="bold")
    ax_seg.set_ylim(0, max(base) * 1.18)
    ax_seg.set_xticks(xs, nombres)
    ax_seg.yaxis.set_major_formatter(lambda y, _: numero(y, 0))
    for ax in (ax_usd, ax_seg):
        leyenda = ax.legend(frameon=False, loc="upper left", fontsize=9)
        for texto in leyenda.get_texts():
            texto.set_color(t["texto"])

    _eje(ax_cal, t, "Calidad contra costo")
    ax_cal.xaxis.grid(True, color=t["linea"], linewidth=0.6)
    ax_cal.tick_params(axis="x", colors=t["secundario"], labelsize=8)
    for n in nombres:
        f = tab[n]
        x, y = f["usd_por_consulta"], f["ndcg"]["todas"]
        color = t["final"] if n == nombres[-1] else t["barra"]
        ax_cal.plot([x, x], f["ic95"], color=color, linewidth=1.2)
        ax_cal.plot(x, y, "o", color=color, markersize=9)
        ax_cal.annotate(n, (x, y), xytext=(7, 0), textcoords="offset points", va="center", color=t["texto"], fontsize=10, fontweight="bold")
    ax_cal.set_xlim(-0.002, 0.032)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("USD por consulta", color=t["secundario"], fontsize=9)
    ax_cal.set_ylabel("nDCG@5 (IC 95 %)", color=t["secundario"], fontsize=9)
    ax_cal.set_xticks([0, 0.01, 0.02, 0.03])
    ax_cal.xaxis.set_major_formatter(lambda x, _: numero(x, 2))
    ax_cal.yaxis.set_major_formatter(lambda y, _: numero(y, 1))

    nota = ("Media de 30 consultas. Precios de lista de OpenAI: gpt-4.1-mini (USD 0,40 / 1,60 por millón de tokens de entrada / salida) en los tres roles; text-embedding-3-small < USD 0,0001; CLIP sin costo de API. "
            "\nEl plan se reutiliza de caché en C2–C4: su costo se suma desde el plan guardado, pero su latencia no se midió. C0 no genera explicación. C0–C2 corrieron el 27/10 y C3–C4 el 16/11: "
            "la diferencia de latencia entre C2 y C3 es del generador (8,6 s contra 6,4 s), no de la cascada.")
    fig.text(0.012, 0.012, nota, color=t["secundario"], fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=3)
    hechas.append(_guardar(fig, t, "ablacion-costo.png")); plt.close(fig)
    return hechas


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juicios-extra", type=Path, action="append", default=[], help="CSV con consulta_id, artwork_id, grado (repetible)")
    p.add_argument("--tema", choices=sorted(TEMAS), default="claro", help="claro: fondo blanco (por defecto); oscuro: fondo de la presentación")
    p.add_argument("--sin-figuras", action="store_true")
    args = p.parse_args()

    a_mano = sorted(c.SALIDA.glob("juicios-*.csv"))
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *a_mano, *args.juicios_extra]))
    elegidos, precios = cargar()
    tab = tabla(ev, elegidos, precios)
    faltan = [nombre for clave, nombre, _ in FILAS if clave not in elegidos]

    c.SALIDA.mkdir(parents=True, exist_ok=True)
    (c.SALIDA / "ablacion.json").write_text(json.dumps({"juicios": len(ev.gt), "filas": tab}, ensure_ascii=False, indent=2), encoding="utf-8")
    hechas = [] if args.sin_figuras else figuras(tab, args.tema, len(ev.gt))

    print(f"Juicios: {len(ev.gt)}" + (f" · sin corridas completas: {', '.join(faltan)}" if faltan else "") + "\n")
    print(f"{'fila':<5}{'agrega':<24}{'nDCG@5':>7}  {'IC95':<16}{'concep':>7}{'visual':>7}{'ambig':>7}{'P@5':>7}{'P@5 2':>7}{'MRR':>7}{'QS':>7}{'G2/5':>6}{'USD':>8}{'tok in':>8}{'seg':>6}{'LLM':>6}")
    for nombre, f in tab.items():
        print(f"{nombre:<5}{f['agrega']:<24}{f['ndcg']['todas']:>7.3f}  {str(f['ic95']):<16}{f['ndcg']['conceptual']:>7.3f}{f['ndcg']['visual']:>7.3f}{f['ndcg']['ambiguo']:>7.3f}"
              f"{f['p5']:>7.3f}{f['p5_estricta']:>7.3f}{f['mrr']:>7.3f}{f['qs']:>7.3f}{f['grado2_por_top5']:>6.2f}{f['usd_por_consulta']:>8.4f}{f['tokens_in']:>8}{(f['segundos_media'] or 0):>6.1f}{f['llamadas_llm']:>6.1f}")
    for nombre, f in tab.items():
        d = f.get("delta_ndcg_contra_la_anterior")
        if d:
            print(f"\nΔ nDCG@5 de {nombre} contra la anterior: {d['delta']:+.3f}  IC95 {d['ic95']}")
    print(f"\nEscrito en {c.SALIDA.relative_to(c.RAIZ)}: ablacion.json" + (", " + ", ".join(hechas) if hechas else ""))


if __name__ == "__main__":
    main()
