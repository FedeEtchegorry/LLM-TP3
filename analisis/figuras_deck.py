"""Figuras del deck que no salen de cascada.py ni de ablacion.py. No gasta.

    python analisis/cascada.py && python analisis/ruido.py && python analisis/confianza.py
    python analisis/figuras_deck.py

Lee resultados.json, ruido.json, confianza.json y la revisión de explicaciones, y escribe en
artifacts/analisis: ruido.png, modelos-calidad.png, modelos-costo.png, v1-vs-v2.png,
filtro-metadata.png, honestidad.png y aviso.png.
"""
from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cascada as c  # noqa: E402

S = c.SALIDA
TIPOS = ("conceptual", "visual", "ambiguo", "todas")
AZUL, NARANJA, VERDE, GRIS, TINTA = "#1f77b4", "#ff7f0e", "#2ca02c", "#9a9a9a", "#333333"
C3, C4 = "C3·filtro100·metadata", "C4·filtro100·metadata·rerank20"
RUIDO = 0.02


def _es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def _guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(S / nombre, dpi=150)
    plt.close(fig)
    print(nombre)


def _barras_por_tipo(ax, series, tipos=TIPOS):
    """series: [(nombre, color, {tipo: valor})]. Un valor ausente o NaN no se dibuja."""
    ancho = 0.8 / len(series)
    for i, (nombre, color, valores) in enumerate(series):
        xs, ys = [], []
        for j, t in enumerate(tipos):
            v = valores.get(t)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                xs.append(j + (i - (len(series) - 1) / 2) * ancho)
                ys.append(v)
        barras = ax.bar(xs, ys, ancho * 0.92, label=nombre, color=color)
        for b, y in zip(barras, ys):
            ax.annotate(_es(y), (b.get_x() + b.get_width() / 2, y), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(range(len(tipos)), tipos)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("nDCG@5")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def ruido():
    r = json.loads((S / "ruido.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [3, 2]})
    for clave, nombre, color in ((C3, "C3 (sin reranker)", AZUL), (C4, "C4 (con reranker)", NARANJA)):
        filas = r[clave]["filas"]
        xs, ys = [f["repeticion"] for f in filas], [f["ndcg"] for f in filas]
        a.plot(xs, ys, marker="o", color=color, linewidth=2, label=nombre)
        for x, y in zip(xs, ys):
            a.annotate(_es(y, 3), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    media = sum(f["ndcg"] for f in r[C4]["filas"]) / len(r[C4]["filas"])
    a.axhspan(media - RUIDO, media + RUIDO, color=NARANJA, alpha=0.12, label=f"±{_es(RUIDO)} alrededor de la media de C4")
    a.set_xticks([f["repeticion"] for f in r[C4]["filas"]])
    a.set_xlabel("Repetición de la misma configuración")
    a.set_ylabel("nDCG@5 (30 consultas)")
    a.set_ylim(0.55, 0.82)
    a.grid(axis="y", alpha=0.25)
    a.legend(fontsize=8, loc="lower right")
    a.set_title("La media se mueve poco…")
    n = r[C4]["consultas"]
    cambian = [r[C3]["consultas_con_top5_distinto"], r[C4]["consultas_con_top5_distinto"]]
    barras = b.bar(["C3", "C4"], cambian, color=[AZUL, NARANJA], width=0.55)
    for barra, v in zip(barras, cambian):
        b.annotate(f"{v} de {n}", (barra.get_x() + barra.get_width() / 2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
    b.set_ylim(0, n + 3)
    b.set_ylabel("Consultas con top-5 distinto entre repeticiones")
    b.grid(axis="y", alpha=0.25)
    b.set_axisbelow(True)
    b.set_title("…pero el top-5 cambia")
    fig.suptitle("Ruido entre repeticiones: C3 es determinista, C4 no", fontsize=12)
    _guardar(fig, "ruido.png")


def modelos(e):
    luna, terra = C4 + "·gpt-6-luna", C4 + "·gpt-6.1-terra"
    presentes = [(n, k, col) for n, k, col in (("gpt-6-luna", luna, AZUL), ("gpt-4.1-mini", C4, NARANJA), ("gpt-6.1-terra", terra, VERDE)) if k in e]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    _barras_por_tipo(ax, [(n, col, e[k]["ndcg"] if e[k]["consultas"] == 30 else {"conceptual": e[k]["ndcg"]["conceptual"]}) for n, k, col in presentes])
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    corto = {"gpt-6-luna": "luna", "gpt-4.1-mini": "mini", "gpt-6.1-terra": "terra"}
    nota = "Pares sin juzgar, que cuentan como 0: " + ", ".join(f"{corto[n]} {_es(e[k]['pct_sin_juzgar'], 0)} %" for n, k, _ in presentes) + "."
    if terra in e and e[terra]["consultas"] != 30:
        nota += f" Terra se corrió solo en las {e[terra]['consultas']} conceptuales."
    ax.set_title("Calidad por modelo de reranker (C4, 20 candidatas)", fontsize=11, pad=22)
    ax.text(0.5, 1.02, nota, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#555555")
    _guardar(fig, "modelos-calidad.png")

    # Costo y consumo: cada medida en su panel, porque las escalas no se comparan.
    consumo = _consumo({k for _, k, _ in presentes})
    paneles = [("usd", "USD por consulta", 4), ("tokens_in", "Tokens de entrada por consulta", 0),
               ("tokens_out", "Tokens de salida por consulta", 0), ("segundos", "Latencia media (s)", 1)]
    fig, ejes = plt.subplots(2, 2, figsize=(12, 7.2))
    nombres = [f"{n}\n({consumo[k]['n']} consultas)" for n, k, _ in presentes]
    for ax, (clave, titulo, dec) in zip(ejes.flat, paneles):
        valores = [consumo[k][clave] for _, k, _ in presentes]
        barras = ax.bar(nombres, valores, color=[col for _, _, col in presentes], width=0.6)
        base = consumo[C4][clave]
        for barra, v, (_, k, _) in zip(barras, valores, presentes):
            texto = _es(v, dec) if dec else f"{v:,.0f}".replace(",", ".")
            if k != C4 and base:
                texto += f"\n({_es(v / base, 1)}× mini)"
            ax.annotate(texto, (barra.get_x() + barra.get_width() / 2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
        ax.set_ylim(0, max(valores) * 1.28)
        ax.set_title(titulo, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Costo y consumo por modelo de reranker (C4, 20 candidatas, toda la cadena)", fontsize=12)
    _guardar(fig, "modelos-costo.png")


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
        }
    return salida


def v1_vs_v2(e, pareadas):
    v2 = C4 + "·prompt2"
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _barras_por_tipo(ax, [("prompt v1", GRIS, e[C4]["ndcg"]), ("prompt v2", AZUL, e[v2]["ndcg"])])
    d = pareadas[f"{v2} − {C4}"]
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_title(
        f"Prompt del reranker: v1 contra v2\nDiferencia en todas: {d['delta']:+.3f}".replace(".", ",")
        + f" (IC 95 %: {_es(d['ic95'][0], 3)} a {_es(d['ic95'][1], 3)}) · v2 tiene {_es(e[v2]['pct_sin_juzgar'], 0)} % de pares sin juzgar",
        fontsize=10,
    )
    _guardar(fig, "v1-vs-v2.png")


def filtro(e):
    # Las mixtas salen de mixtas.py y entran como un grupo más. Sin las corridas de C3 y C4 sin filtro no hay barra gris.
    mixtas = {}
    if (S / "mixtas.json").exists():
        mixtas = json.loads((S / "mixtas.json").read_text(encoding="utf-8"))["configuraciones"]
    tipos = (*TIPOS, "mixtas") if mixtas else TIPOS
    fig, ejes = plt.subplots(1, 2, figsize=(12.5 if mixtas else 11, 4.4), sharey=True)
    for ax, sin, con, titulo, c_sin, c_con in (
        (ejes[0], "C3·filtro100", C3, "C3: sin reranker", "C3 sin filtro", "C3"),
        (ejes[1], "C4·filtro100·rerank20", C4, "C4: con reranker", "C4 sin filtro", "C4"),
    ):
        v_sin, v_con = dict(e[sin]["ndcg"]), dict(e[con]["ndcg"])
        if mixtas:
            v_sin["mixtas"] = (mixtas.get(c_sin) or {}).get("ndcg")
            v_con["mixtas"] = (mixtas.get(c_con) or {}).get("ndcg")
        _barras_por_tipo(ax, [("sin filtro de metadata", GRIS, v_sin), ("con filtro de metadata", AZUL, v_con)], tipos)
        if mixtas and v_sin["mixtas"] is None:
            ax.text(len(tipos) - 1 - 0.2, 0.02, "sin corrida", rotation=90, ha="center", va="bottom", fontsize=8, color=GRIS)
        elif mixtas and (mixtas[c_sin].get("pct_sin_juzgar") or 0) > 0:  # pares sin juicio cuentan 0: el nDCG es cota inferior
            ax.set_xlabel(f"mixtas sin filtro: {_es(mixtas[c_sin]['pct_sin_juzgar'], 0)} % de pares sin juzgar, nDCG es cota inferior", fontsize=8, color=GRIS)
        ax.set_title(titulo, fontsize=11)
    ejes[1].set_ylabel("")
    ejes[0].legend(fontsize=9, loc="upper right")
    ejes[0].set_ylim(0, 1.2)  # lugar para la leyenda sin tapar las etiquetas
    ejes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle("Efecto del filtro de metadata en las 30 consultas y en las mixtas" if mixtas else "Efecto del filtro de metadata en las 30 consultas", fontsize=11)
    _guardar(fig, "filtro-metadata.png")


def honestidad():
    r = json.loads((c.RAIZ / "artifacts" / "ground-truth" / "revision-explicaciones" / "m13b-c1-5x5.json").read_text(encoding="utf-8"))["conteos"]
    e1, e2, e3 = r["eje_1_clasificacion"], r["eje_2_la_cita_se_sostiene"], r["eje_3_la_observacion_es_defendible"]
    filas = [
        (f"¿Dato u observación bien tipados?\n({e1['n']} afirmaciones)", [("bien", e1["bien_clasificadas"], AZUL), ("mal", e1["mal_clasificadas"], NARANJA)]),
        (f"¿La cita sostiene el dato?\n({e2['n']} datos con cita)", [("sí", e2["sostiene"], AZUL), ("en parte", e2["sostiene_parcialmente"], GRIS), ("no", e2["no_sostiene"], NARANJA)]),
        (f"¿La observación visual es defendible?\n({e3['n_con_asercion_visual_verificable']} observaciones verificables)",
         [("sí", e3["defendible"], AZUL), ("discutible", e3["discutible"], GRIS), ("falsa", e3["falsa_a_la_vista"], NARANJA)]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for y, (_, partes) in enumerate(filas):
        total, izq = sum(n for _, n, _ in partes), 0.0
        for nombre, n, color in partes:
            ancho = 100 * n / total
            ax.barh(y, ancho, left=izq, color=color, edgecolor="white", linewidth=2, height=0.6)
            if ancho > 5:
                ax.text(izq + ancho / 2, y, f"{nombre}: {n}", ha="center", va="center", fontsize=9, color="white" if color != GRIS else TINTA)
            izq += ancho
    ax.set_yticks(range(len(filas)), [f for f, _ in filas], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de las afirmaciones de cada eje")
    ax.set_title(f"Revisión manual de {r['n_explicaciones']} explicaciones ({r['n_afirmaciones']} afirmaciones)", fontsize=11)
    _guardar(fig, "honestidad.png")


def aviso():
    d = json.loads((S / "confianza.json").read_text(encoding="utf-8"))
    filas = d["barrido_de_umbrales"]
    xs = [f["umbral"] for f in filas]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for clave, nombre, color in (("cobertura", "Cobertura: listas malas que avisa", AZUL), ("precision", "Precisión: avisos que eran listas malas", NARANJA)):
        ax.plot(xs, [f[clave] for f in filas], marker="o", color=color, linewidth=2, label=nombre)
    u = d["umbral_publicado"]
    ax.axvline(u["umbral"], color=TINTA, linestyle="--", linewidth=1)
    ax.annotate(f"umbral {_es(u['umbral'], 0)}: avisa {u['avisa']} de {d['consultas']},\ncobertura {_es(u['cobertura'])}, precisión {_es(u['precision'])}",
                (u["umbral"], 0.18), textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Umbral de confianza (puntaje medio del reranker en el top-5)")
    ax.set_ylabel("Proporción")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(f"Aviso por confianza: {d['listas_malas']} de {d['consultas']} listas son malas ({d['definicion_de_lista_mala']})", fontsize=10)
    _guardar(fig, "aviso.png")


def main():
    r = json.loads((S / "resultados.json").read_text(encoding="utf-8"))
    e = r["experimentos"]
    ruido()
    modelos(e)
    v1_vs_v2(e, r["pareadas_contra_C4"])
    filtro(e)
    honestidad()
    aviso()


if __name__ == "__main__":
    main()
