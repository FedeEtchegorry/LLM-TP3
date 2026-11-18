"""Figuras del deck que no salen de cascada.py ni de ablacion.py. No gasta.

    python analisis/cascada.py && python analisis/ablacion.py && python analisis/mixtas.py
    python analisis/ruido.py && python analisis/confianza.py
    python analisis/figuras_deck.py

Lee resultados.json, ablacion.json, mixtas.json, ruido.json, confianza.json, aclaraciones-v3.json,
validacion-v3.json y la revisión de explicaciones (claude-c1-5x5-v2.json), y escribe en
artifacts/analisis: ruido.png, modelos-calidad.png, modelos-costo.png, v1-vs-v2.png,
filtro-metadata.png, honestidad.png, aviso.png y prefiltro.png. Todas comparten estilo: fondo
blanco con recuadro, coma decimal, arena para la base y colores fijos por tipo de consulta.
"""
from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cascada as c  # noqa: E402

S = c.SALIDA
C3, C4 = "C3·filtro100·metadata", "C4·filtro100·metadata·rerank20"


def _es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def _estilo(ax, titulo):
    """Mismo estilo que las figuras de ablacion.py (tema claro)."""
    ax.set_title(titulo, color="#0b0b0b", fontsize=12, fontweight="bold", loc="left", pad=10)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d2")
    ax.tick_params(length=0)
    ax.tick_params(axis="y", colors="#52514e", labelsize=8.5)
    ax.yaxis.grid(True, color="#d9d8d2", linewidth=0.6)
    ax.set_axisbelow(True)


def ruido():
    """Izquierda: el nDCG@5 medio no se mueve entre repeticiones. Derecha: el de cada consulta sí, sobre todo en las visuales."""
    r = json.loads((S / "ruido.json").read_text(encoding="utf-8"))
    gris, oro, tinta, sec = "#a3a29a", "#b8860b", "#0b0b0b", "#52514e"
    colores = {"conceptual": "#2a78d6", "visual": "#eb6834", "ambiguo": "#1baf7a"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})

    _estilo(a, "nDCG@5 medio en cada repetición")
    for clave, nombre, color, dy in ((C4, "C4 (con reranker)", oro, 10), (C3, "C3 (sin reranker)", gris, -18)):
        ys = [f["ndcg"] for f in r[clave]["filas"]]
        xs = list(range(1, len(ys) + 1))
        a.plot(xs, ys, marker="o", markersize=9, linewidth=2.5, color=color)
        for x, y in zip(xs, ys):
            a.annotate(_es(y, 3), (x, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=10.5, fontweight="bold", color=tinta)
        a.annotate(nombre, (xs[-1], ys[-1]), textcoords="offset points", xytext=(14, 0), va="center", fontsize=10.5, fontweight="bold", color=color)
    a.set_xticks([1, 2, 3], ["Repetición 1", "Repetición 2", "Repetición 3"], fontsize=10)
    a.set_xlim(0.6, 4.3)
    a.set_ylim(0, 1.0)
    a.yaxis.set_major_formatter(lambda y, _: _es(y, 1))

    _estilo(b, "Cuánto cambia el nDCG@5 de una consulta en C4")
    d = r[C4]
    tipos = list(colores)
    rangos, cambian = [], []
    for tipo in tipos:
        qs = [q for q in d["ndcg_por_consulta"] if c.categoria(q) == tipo]
        rangos.append(sum(max(d["ndcg_por_consulta"][q]) - min(d["ndcg_por_consulta"][q]) for q in qs) / len(qs))
        cambian.append((sum(d["top5_distinto_por_consulta"][q] for q in qs), len(qs)))
    barras = b.bar(range(len(tipos)), rangos, width=0.55, color=[colores[t] for t in tipos])
    for barra, v in zip(barras, rangos):
        b.text(barra.get_x() + barra.get_width() / 2, v + 0.003, _es(v, 3), ha="center", va="bottom", fontsize=11, fontweight="bold", color=tinta)
    b.set_xticks(range(len(tipos)), [f"{t.capitalize()}\ntop-5 distinto en {k} de {n}" for t, (k, n) in zip(tipos, cambian)], fontsize=10)
    b.set_ylim(0, max(rangos) * 1.3)
    b.set_ylabel("Mejor − peor repetición (media por consulta)", color=sec, fontsize=9)
    b.yaxis.set_major_formatter(lambda y, _: _es(y, 2))

    fig.text(0.012, 0.015,
             "30 consultas (10 por tipo), tres repeticiones de cada configuración con el mismo plan. "
             "C4: reranker gpt-4.1-mini, 20 candidatas, prompt v1. C3 devuelve el mismo top-5 en las tres repeticiones.\n"
             f"Entre repeticiones de C4, la diferencia de nDCG@5 medio no es significativa (IC 95 % pareado, las tres cruzan 0). "
             f"Jaccard medio del top-5 entre repeticiones: {_es(d['jaccard_medio_del_top5'])}.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=4)
    fig.savefig(S / "ruido.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("ruido.png")


MODELOS = (  # (nombre, sufijo de la etiqueta, color): mini es la base, luna el de mejor resultado
    ("gpt-4.1-mini", "", "#d6cdbd"),
    ("gpt-6-luna", "·gpt-6-luna", "#8a6a2f"),
    ("gpt-6.1-terra", "·gpt-6.1-terra", "#7f9c96"),
)


CORTO_MODELO = {"gpt-4.1-mini": "mini", "gpt-6-luna": "luna", "gpt-6.1-terra": "terra"}


def _recuadro(ax):
    for lado in ("top", "right", "left", "bottom"):
        ax.spines[lado].set_visible(True)
        ax.spines[lado].set_color("#a9a59b")
        ax.spines[lado].set_linewidth(1)


def modelos(e):
    """nDCG@5 de C4 con cada modelo de reranker, por tipo de consulta, en el set de 30 y en las mixtas."""
    from matplotlib.patches import Patch
    tinta, sec, linea = "#0b0b0b", "#52514e", "#d9d8d2"
    mixtas = json.loads((S / "mixtas.json").read_text(encoding="utf-8"))["configuraciones"] if (S / "mixtas.json").exists() else {}
    en_mixtas = {"": "C4", "·gpt-6-luna": "C4 luna", "·gpt-6.1-terra": "C4 terra"}
    presentes = [(n, C4 + suf, col, suf) for n, suf, col in MODELOS if C4 + suf in e]
    grupos = [("conceptual", "Conceptual"), ("visual", "Visual"), ("ambiguo", "Ambiguo"), ("todas", "Set de 30")]
    if mixtas:
        grupos.append(("mixtas", "Mixtas"))
    xs = [0, 1, 2, 3.5, 5][: len(grupos)]
    ancho = 0.8 / len(presentes)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    _estilo(ax, "")
    _recuadro(ax)
    for k, (nombre, clave, color, suf) in enumerate(presentes):
        for x, (g, _) in zip(xs, grupos):
            v = (mixtas.get(en_mixtas[suf]) or {}).get("ndcg") if g == "mixtas" else e[clave]["ndcg"].get(g)
            if v is None:
                continue
            xb = x + (k - (len(presentes) - 1) / 2) * ancho
            ax.bar(xb, v, width=ancho * 0.94, color=color, edgecolor="white", linewidth=1)
            ax.text(xb, v + 0.012, _es(v, 2), ha="center", va="bottom", fontsize=9.5, color=tinta)
    for x in (2.75, 4.25)[: len(grupos) - 3]:
        ax.axvline(x, color=linea, linewidth=1)
    ax.set_xticks(xs, [n for _, n in grupos], fontsize=10.5, color=tinta)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(lambda y, _: _es(y, 1))
    ax.set_ylabel("nDCG@5", color=sec, fontsize=9)
    leyenda = ax.legend(handles=[Patch(color=col, label=n) for n, _, col, _ in presentes], frameon=False, loc="upper left",
                        fontsize=10, ncols=len(presentes), bbox_to_anchor=(0, 1.0))
    for t in leyenda.get_texts():
        t.set_color(tinta)
    fig.text(0.012, 0.015,
             "C4 con 20 candidatas y prompt v1; solo cambia el modelo del reranker. nDCG@5 medio; set de 30 (10 consultas por tipo), todos los pares juzgados.\n"
             "Mixtas: 10 consultas con otra rúbrica (cambio visual + restricción verificada en metadata); se comparan entre modelos, no con el set de 30.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(S / "modelos-calidad.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("modelos-calidad.png")
    presentes = [(n, k, col) for n, k, col, _ in presentes]

    modelos_costo(e, presentes)


def modelos_costo(e, presentes):
    """USD por consulta (reranker sobre el resto), latencia por etapa y calidad contra costo, por modelo de reranker."""
    from matplotlib.patches import Patch
    tinta, sec, linea, neutro = "#0b0b0b", "#52514e", "#d9d8d2", "#d5d9dc"
    consumo = _consumo({k for _, k, _ in presentes})
    xs = list(range(len(presentes)))
    fig, (a_usd, a_seg, a_cal) = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": (1, 1, 1.15)})

    _estilo(a_usd, "USD por consulta")
    _recuadro(a_usd)
    for x, (n, k, col) in zip(xs, presentes):
        d = consumo[k]
        resto = d["usd"] - d["usd_reranker"]
        a_usd.bar(x, resto, width=0.58, color=neutro, edgecolor="white")
        a_usd.bar(x, d["usd_reranker"], bottom=resto, width=0.58, color=col, edgecolor="white")
        a_usd.text(x, d["usd"] + 0.0012, _es(d["usd"], 4), ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=tinta)
    a_usd.set_xticks(xs, [f"{CORTO_MODELO[n]}\n{_es(consumo[k]['tokens_in'] / 1000, 0)} mil entrada\n{_es(consumo[k]['tokens_out'] / 1000, 1)} mil salida"
                          for n, k, _ in presentes], fontsize=9.5, color=tinta)
    a_usd.set_ylim(0, max(consumo[k]["usd"] for _, k, _ in presentes) * 1.2)
    a_usd.yaxis.set_major_formatter(lambda y, _: _es(y, 2))

    _estilo(a_seg, "Latencia media (s)")
    _recuadro(a_seg)
    for x, (n, k, col) in zip(xs, presentes):
        g = consumo[k]["segundos_por_grupo"]
        base = 0.0
        for grupo, color in (("Recuperación y armado", neutro), ("Generador", "#9ea4a8"), ("Reranker", col)):
            a_seg.bar(x, g[grupo], bottom=base, width=0.58, color=color, edgecolor="white")
            base += g[grupo]
        a_seg.text(x, consumo[k]["segundos"] + 1.5, _es(consumo[k]["segundos"], 1), ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=tinta)
    a_seg.set_xticks(xs, [CORTO_MODELO[n] for n, _, _ in presentes], fontsize=10, color=tinta)
    a_seg.set_ylim(0, max(consumo[k]["segundos"] for _, k, _ in presentes) * 1.2)
    a_seg.yaxis.set_major_formatter(lambda y, _: _es(y, 0))
    leyenda = a_seg.legend(handles=[Patch(color=neutro, label="Recuperación y armado"), Patch(color="#9ea4a8", label="Generador"),
                                    Patch(facecolor="white", edgecolor=sec, label="Reranker (color del modelo)")],
                           frameon=False, loc="upper left", fontsize=8.5)
    for t in leyenda.get_texts():
        t.set_color(tinta)

    _estilo(a_cal, "Calidad contra costo")
    _recuadro(a_cal)
    a_cal.xaxis.grid(True, color=linea, linewidth=0.6)
    a_cal.tick_params(axis="x", colors=sec, labelsize=8.5)
    for n, k, col in presentes:
        x, y = consumo[k]["usd"], e[k]["ndcg"]["todas"]
        a_cal.plot([x, x], e[k]["ic95"], color=col, linewidth=1.6)
        a_cal.plot(x, y, "o", color=col, markersize=11, markeredgecolor="#6b675f")
        a_cal.annotate(CORTO_MODELO[n], (x, y), textcoords="offset points", xytext=(10, 0), va="center", fontsize=10.5, fontweight="bold", color=tinta)
    a_cal.set_xlim(0, max(consumo[k]["usd"] for _, k, _ in presentes) * 1.25)
    a_cal.set_ylim(0.5, 1.0)
    a_cal.set_xlabel("USD por consulta", color=sec, fontsize=9)
    a_cal.set_ylabel("nDCG@5, set de 30 (IC 95 %)", color=sec, fontsize=9)
    a_cal.xaxis.set_major_formatter(lambda x, _: _es(x, 2))
    a_cal.yaxis.set_major_formatter(lambda y, _: _es(y, 1))

    fig.text(0.012, 0.015,
             "C4 con 20 candidatas y prompt v1, media de 30 consultas; costo y latencia de toda la cadena. En USD, en gris el resto de la cadena (planificador y generador, siempre gpt-4.1-mini). "
             "Precios por millón de tokens (entrada / salida):\nmini 0,40 / 1,60 (lista de OpenAI); luna 0,10 / 0,50 y terra 1,00 / 5,00 (comparadores, no factura). "
             "Los tokens de salida de luna y terra incluyen razonamiento. Latencia medida a través de un proxy compartido.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=3)
    fig.savefig(S / "modelos-costo.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("modelos-costo.png")


def _consumo(etiquetas) -> dict:
    """Media por consulta de USD, tokens y latencia de la corrida más nueva de cada consulta."""
    import glob
    import os

    import ablacion as ab

    ultimas: dict = {}
    for archivo in glob.glob(str(c.RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        cfg, estado = corrida["configuracion_resuelta"], corrida["estado"]
        if corrida["consulta_id"].endswith("-mixto") or (cfg.get("fusion") or {}).get("estrategia") != "cascada" or not estado.get("top5"):
            continue
        if any(r.get("estado") == "sin_puntaje" for r in (estado.get("rerank") or [])):
            continue
        etiqueta = c.etiqueta(cfg)
        if etiqueta not in etiquetas:
            continue
        fila = {
            "fecha": os.path.getmtime(archivo),
            "libro": {r: l for r, l in (corrida.get("libro_mayor") or {}).items() if isinstance(l, dict)},
            "segundos": (corrida.get("latencia") or {}).get("segundos_de_la_corrida"),
            "por_etapa": (corrida.get("latencia") or {}).get("por_etapa") or {},
            "modelos": {r: b.get("modelo") for r, b in (cfg.get("bindings") or {}).items()},
            "modelo_de_embedding": (cfg.get("embebido_texto") or {}).get("modelo"),
            "precios": cfg.get("proveedor") or {},
            "planifica": "planificador" in cfg["consulta"]["etapas"],
            "uso_del_plan": ab._uso_del_plan(estado),
        }
        clave = (etiqueta, corrida["consulta_id"])
        if clave not in ultimas or fila["fecha"] > ultimas[clave]["fecha"]:
            ultimas[clave] = fila
    salida = {}
    for etiqueta in etiquetas:
        filas = [f for (e, _), f in ultimas.items() if e == etiqueta]
        usos = [ab.uso(f, f["precios"]) for f in filas]
        segundos = [f["segundos"] for f in filas if f["segundos"]]
        salida[etiqueta] = {
            "n": len(filas),
            "usd": c.media(u["usd"] for u in usos),
            "tokens_in": c.media(u["tokens_in"] for u in usos),
            "tokens_out": c.media(u["tokens_out"] for u in usos),
            "segundos": c.media(segundos) if segundos else 0.0,
            "usd_reranker": c.media(u["usd_por_rol"].get("reranker", 0) for u in usos),
            "segundos_por_grupo": {g: c.media(sum(f["por_etapa"].get(x, 0) for x in etapas) for f in filas)
                                   for g, etapas in ab.GRUPOS_DE_ETAPAS.items()},
        }
    return salida


def _barras_agrupadas(series, archivo, pie):
    """series: [(nombre, color, {grupo: valor})]. Grupos: tres tipos | set de 30 | mixtas, con el estilo del deck."""
    from matplotlib.patches import Patch
    tinta, sec, linea = "#0b0b0b", "#52514e", "#d9d8d2"
    grupos = [("conceptual", "Conceptual"), ("visual", "Visual"), ("ambiguo", "Ambiguo"), ("todas", "Set de 30")]
    if any(v.get("mixtas") is not None for _, _, v in series):
        grupos.append(("mixtas", "Mixtas"))
    xs = [0, 1, 2, 3.5, 5][: len(grupos)]
    ancho = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    _estilo(ax, "")
    _recuadro(ax)
    for k, (nombre, color, valores) in enumerate(series):
        for x, (g, _) in zip(xs, grupos):
            v = valores.get(g)
            if v is None:
                continue
            xb = x + (k - (len(series) - 1) / 2) * ancho
            ax.bar(xb, v, width=ancho * 0.94, color=color, edgecolor="white", linewidth=1)
            ax.text(xb, v + 0.012, _es(v, 2), ha="center", va="bottom", fontsize=9.5, color=tinta)
    for x in (2.75, 4.25)[: len(grupos) - 3]:
        ax.axvline(x, color=linea, linewidth=1)
    ax.set_xticks(xs, [n for _, n in grupos], fontsize=10.5, color=tinta)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(lambda y, _: _es(y, 1))
    ax.set_ylabel("nDCG@5", color=sec, fontsize=9)
    leyenda = ax.legend(handles=[Patch(color=col, label=n) for n, col, _ in series], frameon=False, loc="upper left",
                        fontsize=10, ncols=len(series))
    for t in leyenda.get_texts():
        t.set_color(tinta)
    fig.text(0.012, 0.015, pie, color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(S / archivo, dpi=200, facecolor="white")
    plt.close(fig)
    print(archivo)


def v1_vs_v2(e, pareadas):
    """nDCG@5 de C4 (gpt-4.1-mini) con el prompt v1 y el v2 del reranker."""
    v2 = C4 + "·prompt2"
    mixtas = json.loads((S / "mixtas.json").read_text(encoding="utf-8"))["configuraciones"] if (S / "mixtas.json").exists() else {}
    series = []
    for nombre, clave, color, en_mixtas in (("Prompt v1", C4, "#d6cdbd", "C4"), ("Prompt v2", v2, "#6d84a0", "C4 prompt v2")):
        valores = dict(e[clave]["ndcg"])
        valores["mixtas"] = (mixtas.get(en_mixtas) or {}).get("ndcg")
        series.append((nombre, color, valores))
    _barras_agrupadas(series, "v1-vs-v2.png",
                      "C4 con gpt-4.1-mini y 20 candidatas; solo cambia el prompt del reranker. v2 obliga a comparar la obra original y la candidata en el aspecto pedido antes de puntuar.\n"
                      "nDCG@5 medio; set de 30 (10 consultas por tipo), todos los pares juzgados. Mixtas: 10 consultas con otra rúbrica; se comparan entre prompts, no con el set de 30.")


def filtro(e):
    """Con y sin filtro de metadata, en C3 y en C4: un punto por variante y la diferencia pareada a la derecha.

    Para cada consulta del set de 30 el filtro se considera activo si cambia la lista que entra a CLIP.
    Las mixtas salen de mixtas.py (otra rúbrica) y van separadas.
    """
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(S.glob("juicios-*.csv"))]))
    ex = c.cargar_corridas()
    mixtas = json.loads((S / "mixtas.json").read_text(encoding="utf-8")) if (S / "mixtas.json").exists() else None
    tinta, sec, linea, gris, oscuro = "#0b0b0b", "#52514e", "#d9d8d2", "#d6cdbd", "#8a6a2f"
    s3, c3 = ex["C3·filtro100"], ex[C3]
    qs = sorted(c3)
    activo = {q: [x["artwork_id"] for x in c3[q]["ranking_fusionado"]] != [x["artwork_id"] for x in s3[q]["ranking_fusionado"]] for q in qs}

    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, sin, con, titulo, m_sin, m_con in (
        (ejes[0], "C3·filtro100", C3, "C3: sin reranker", "C3 sin filtro", "C3"),
        (ejes[1], "C4·filtro100·rerank20", C4, "C4: con reranker", "C4 sin filtro", "C4"),
    ):
        filas = []
        for tipo in (*c.CATEGORIAS, "todas"):
            sel = [q for q in qs if tipo == "todas" or c.categoria(q) == tipo]
            v_sin = c.media(ev.ndcg(q, ex[sin][q]["top5"]) for q in sel)
            v_con = c.media(ev.ndcg(q, ex[con][q]["top5"]) for q in sel)
            dif = [ev.ndcg(q, ex[con][q]["top5"]) - ev.ndcg(q, ex[sin][q]["top5"]) for q in sel]
            nombre = "Set de 30" if tipo == "todas" else f"{tipo.capitalize()} (filtro en {sum(activo[q] for q in sel)} de {len(sel)})"
            filas.append((nombre, v_sin, v_con, c.media(dif), c.ic95(dif)))
        if mixtas:
            d = mixtas["pareadas"][f"{m_con} − {m_sin}"]
            filas.append(("Mixtas (filtro en 9 de 10)", mixtas["configuraciones"][m_sin]["ndcg"], mixtas["configuraciones"][m_con]["ndcg"], d["delta"], tuple(d["ic95"])))
        _estilo(ax, titulo)
        for lado in ("top", "right", "left", "bottom"):  # recuadro alrededor de cada panel
            ax.spines[lado].set_visible(True)
            ax.spines[lado].set_color("#a9a59b")
            ax.spines[lado].set_linewidth(1)
        xs = [0, 1, 2, 3.5, 5][: len(filas)]
        ancho = 0.36
        for x, (nombre, v_sin, v_con, _delta, _ic) in zip(xs, filas):
            ax.bar(x - ancho / 2, v_sin, width=ancho, color=gris, edgecolor="white", linewidth=1)
            ax.bar(x + ancho / 2, v_con, width=ancho, color=oscuro, edgecolor="white", linewidth=1)
            for dx, v in ((-ancho / 2, v_sin), (ancho / 2, v_con)):
                ax.text(x + dx, v + 0.012, _es(v, 2), ha="center", va="bottom", fontsize=9, color=tinta)
        for x in (2.75, 4.25)[: len(filas) - 3]:
            ax.axvline(x, color=linea, linewidth=1)
        etiquetas = [f[0].replace(" (filtro en ", "\nfiltro ").replace(" de 10)", "/10") for f in filas]
        ax.set_xticks(xs, etiquetas, fontsize=9.5, color=tinta)
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(lambda y, _: _es(y, 1))
    ejes[0].set_ylabel("nDCG@5", color=sec, fontsize=9)
    from matplotlib.patches import Patch
    leyenda = ejes[0].legend(handles=[Patch(color=gris, label="Sin filtro"), Patch(color=oscuro, label="Con filtro")],
                             frameon=False, loc="upper left", fontsize=9.5, ncols=2, bbox_to_anchor=(0, 1.02))
    for t in leyenda.get_texts():
        t.set_color(tinta)
    fig.text(0.012, 0.015,
             "nDCG@5 medio. "
             "«Filtro k/n»: consultas en las que el filtro cambia la lista que entra a CLIP.\n"
             "Mixtas: 10 consultas con otra rúbrica (cambio visual + restricción verificada en metadata); se comparan con y sin filtro entre sí, no con el set de 30.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=3)
    fig.savefig(S / "filtro-metadata.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("filtro-metadata.png")


def honestidad():
    """Revisión manual de explicaciones: barras al 100 % por eje (bien, a medias, mal)."""
    d = json.loads((c.RAIZ / "artifacts" / "ground-truth" / "revision-explicaciones" / "claude-c1-5x5-v2.json").read_text(encoding="utf-8"))
    r, auto = d["conteos"], d["verificaciones_automaticas"]
    e1, e2, e3 = r["eje_1_clasificacion"], r["eje_2_la_cita_se_sostiene"], r["eje_3_la_observacion_es_defendible"]
    tinta, sec = "#0b0b0b", "#52514e"
    bien, medio, mal = "#7f9c96", "#d6cdbd", "#c0634a"
    filas = [
        (f"¿Dato u observación,\nbien clasificado?\n({e1['n']} afirmaciones)", [("Bien", e1["bien_clasificadas"], bien), ("Mal", e1["mal_clasificadas"], mal)]),
        (f"¿La cita sostiene el dato?\n({e2['n']} datos con cita)", [("Sí", e2["sostiene"], bien), ("En parte", e2["sostiene_parcialmente"], medio), ("No", e2["no_sostiene"], mal)]),
        (f"¿La observación visual\nes defendible?\n({e3['n_con_asercion_visual_verificable']} verificables)",
         [("Sí", e3["defendible"], bien), ("Discutible", e3["discutible"], medio), ("Falsa", e3["falsa_a_la_vista"], mal)]),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    _recuadro(ax)
    for y, (_, partes) in enumerate(filas):
        total, izq = sum(n for _, n, _ in partes), 0.0
        for nombre, n, color in partes:
            ancho = 100 * n / total
            ax.barh(y, ancho, left=izq, color=color, edgecolor="white", linewidth=2.5, height=0.62)
            ax.text(izq + ancho / 2, y, f"{nombre}\n{n}", ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white" if color in (bien, mal) else tinta)
            izq += ancho
    ax.set_yticks(range(len(filas)), [f for f, _ in filas], fontsize=10, color=tinta)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{x} %" for x in range(0, 101, 20)])
    ax.tick_params(length=0, colors=sec, labelsize=9)
    ax.set_xlabel("Porcentaje de las afirmaciones de cada pregunta", color=sec, fontsize=9)
    p3, p8, p4 = auto["P3_a_referencias_colgadas"], auto["P8_contrato_de_idioma"], auto["P4_atribucion_CC_BY"]
    fig.text(0.012, 0.015,
             f"Revisión hecha por un asistente de IA (Claude), sin validación humana, de {r['n_explicaciones']} explicaciones de C1 (5 consultas × 5 obras; el generador es el mismo en C1–C4), {r['n_afirmaciones']} afirmaciones: "
             f"{r['n_afirmaciones_dato']} datos con cita y {r['n_afirmaciones_observacion']} observaciones.\n"
             f"Verificaciones automáticas: {p3['n_colgadas']} de {p3['n_referencias']} referencias colgadas; {p8['byte_a_byte_contra_el_fragmento']} de {p8['n_citas']} citas textuales al fragmento; "
             f"{p4['con_atribucion']} de {p4['n_refs_CC_BY_4_0']} textos CC BY con atribución.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(S / "honestidad.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("honestidad.png")


def aviso():
    """Izquierda: la confianza de cada consulta, separando listas buenas y malas. Derecha: cobertura y precisión según el umbral."""
    d = json.loads((S / "confianza.json").read_text(encoding="utf-8"))
    tinta, sec, linea = "#0b0b0b", "#52514e", "#d9d8d2"
    mala_c, buena_c, avisa_c = "#c0634a", "#7f9c96", "#f3ece4"
    u = d["umbral_publicado"]["umbral"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": (1.25, 1)})

    _estilo(a, "Confianza de cada consulta")
    _recuadro(a)
    a.yaxis.grid(False)
    a.xaxis.grid(True, color=linea, linewidth=0.6)
    a.axvspan(0, u, color=avisa_c, linewidth=0, zorder=0)
    a.axvline(u, color=sec, linewidth=1.2, linestyle="--")
    a.text(u - 0.1, 1.42, "avisa", ha="right", va="center", fontsize=10, color=sec, fontweight="bold")
    a.text(u + 0.1, 1.42, "no avisa", ha="left", va="center", fontsize=10, color=sec, fontweight="bold")
    filas = sorted(d["por_consulta"].items(), key=lambda kv: kv[1]["confianza"])
    contados = {}
    for q, f in filas:
        mala = f["obras_de_grado_2"] < 3
        y = 0 if mala else 1
        clave = (y, f["confianza"])
        contados[clave] = contados.get(clave, 0) + 1
        dy = (contados[clave] - 1) * 0.09  # apila las consultas con la misma confianza
        a.plot(f["confianza"], y + dy - 0.1, "o", markersize=10, color=mala_c if mala else buena_c, markeredgecolor="white", zorder=3)
    n_malas = d["listas_malas"]
    a.set_yticks([0, 1], [f"Listas malas ({n_malas})", f"Listas buenas ({d['consultas'] - n_malas})"], fontsize=10.5, color=tinta)
    a.set_ylim(-0.5, 1.6)
    a.set_xlim(0, 10)
    a.set_xticks(range(0, 11))
    a.set_xlabel("Puntaje medio del reranker en el top-5 (0 a 10)", color=sec, fontsize=9)
    a.tick_params(axis="x", colors=sec, labelsize=9)
    a.tick_params(axis="y", length=0)

    _estilo(b, "Cobertura y precisión según el umbral")
    _recuadro(b)
    barrido = d["barrido_de_umbrales"]
    xs = [f["umbral"] for f in barrido]
    for clave, nombre, color in (("cobertura", "Cobertura: listas malas que avisa", "#6d84a0"),
                                 ("precision", "Precisión: avisos que eran listas malas", "#8a6a2f")):
        b.plot(xs, [f[clave] for f in barrido], marker="o", markersize=7, linewidth=2.4, color=color, label=nombre)
    b.axvline(u, color=sec, linewidth=1.2, linestyle="--")
    b.set_ylim(0, 1.08)
    b.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    b.xaxis.set_major_formatter(lambda x, _: _es(x, 1))
    b.yaxis.set_major_formatter(lambda y, _: _es(y, 1))
    b.set_xlabel("Umbral", color=sec, fontsize=9)
    b.tick_params(axis="x", colors=sec, labelsize=9)
    leyenda = b.legend(frameon=False, loc="lower left", fontsize=9.5)
    for t in leyenda.get_texts():
        t.set_color(tinta)

    t = d["umbral_publicado"]
    fig.text(0.012, 0.015,
             f"C4 (gpt-4.1-mini, prompt v1), 30 consultas. Lista mala: {d['definicion_de_lista_mala']}. Umbral {_es(u, 0)}: avisa en {t['avisa']}, "
             f"detecta las {t['listas_malas_avisadas']} malas y da {t['avisos_sobre_listas_buenas']} falsas alarmas. AUC {_es(d['auc_confianza_para_separar_buenas_de_malas'])}.\n"
             f"El umbral se eligió con las mismas 30 consultas; dejando una consulta afuera, la exactitud es {_es(d['exactitud_dejando_una_consulta_afuera'])}.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=3)
    fig.savefig(S / "aviso.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("aviso.png")


def prefiltro():
    """Matrices de confusión del prefiltro: las 30 consultas por tipo y 12 pedidos de validación con nivel esperado."""
    from matplotlib.colors import LinearSegmentedColormap
    sugeridas, validacion = S / "aclaraciones-v3.json", S / "validacion-v3.json"
    if not sugeridas.exists():
        print("prefiltro.png: falta aclaraciones-v3.json (correr aclarar.py sugerir)")
        return
    tinta, sec = "#0b0b0b", "#52514e"
    mapa = LinearSegmentedColormap.from_list("arena", ["#faf8f4", "#d6cdbd", "#8a6a2f"])
    niveles = ("concreto", "medio", "vago")
    a = json.loads(sugeridas.read_text(encoding="utf-8"))
    v = json.loads(validacion.read_text(encoding="utf-8")) if validacion.exists() else []
    paneles = [("Set de 30", "Tipo de consulta", [(t.capitalize(), [x["nivel"] for x in a.values() if x["categoria"] == t]) for t in c.CATEGORIAS])]
    if v:
        paneles.append((f"Validación ({len(v)} pedidos escritos aparte)", "Nivel esperado",
                        [(n.capitalize(), [x["nivel"] for x in v if x["esperado"] == n]) for n in niveles]))
    fig, ejes = plt.subplots(1, len(paneles), figsize=(6.2 * len(paneles), 4.6), squeeze=False)
    for ax, (titulo, eje_y, filas) in zip(ejes[0], paneles):
        m = [[obt.count(n) for n in niveles] for _, obt in filas]
        tope = max(max(f) for f in m)
        ax.imshow(m, cmap=mapa, vmin=0, vmax=tope, aspect="auto")
        for r, fila in enumerate(m):
            for k, val in enumerate(fila):
                ax.text(k, r, str(val), ha="center", va="center", fontsize=15, fontweight="bold",
                        color="white" if val > tope * 0.6 else (tinta if val else "#b9b4aa"))
        ax.set_xticks(range(3), [n.capitalize() for n in niveles], fontsize=10.5, color=tinta)
        ax.set_yticks(range(len(filas)), [nombre for nombre, _ in filas], fontsize=10.5, color=tinta)
        ax.set_xlabel("Clasificado por el prefiltro", color=sec, fontsize=9.5)
        ax.set_ylabel(eje_y, color=sec, fontsize=9.5)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for lado in ax.spines.values():
            lado.set_color("#a9a59b")
        ax.set_xticks([x - 0.5 for x in range(1, 3)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(filas))], minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", length=0)
        ax.set_title(titulo, color=tinta, fontsize=12, fontweight="bold", loc="left", pad=34)
    fig.text(0.012, 0.015,
             "El prefiltro (gpt-4.1-mini, prompt v3) clasifica cada pedido como concreto, medio o vago antes del planificador; si es vago, sugiere una versión más concreta (solo sugiere).\n"
             "La validación son 12 pedidos escritos por el grupo con el nivel esperado de antemano (4 por nivel). No se midió si buscar con la sugerencia mejora el resultado.",
             color=sec, fontsize=8.5, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=4)
    fig.savefig(S / "prefiltro.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("prefiltro.png")


def main():
    r = json.loads((S / "resultados.json").read_text(encoding="utf-8"))
    e = r["experimentos"]
    ruido()
    modelos(e)
    v1_vs_v2(e, r["pareadas_contra_C4"])
    filtro(e)
    honestidad()
    aviso()
    prefiltro()


if __name__ == "__main__":
    main()
