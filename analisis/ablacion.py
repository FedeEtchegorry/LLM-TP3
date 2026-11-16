"""Ablación de C0 a C4 (cascada con filtro por metadata): métricas y gráficos. No gasta.

    python analisis/ablacion.py                 # tabla + JSON + figuras sobre fondo blanco
    python analisis/ablacion.py --tema oscuro   # las mismas figuras con el fondo de la presentación
    python analisis/ablacion.py --sin-figuras   # solo la tabla y el JSON

Lee las corridas y los juicios que ya están en `artifacts/` (los mismos que `cascada.py`)
y escribe en `artifacts/analisis/`:

  ablacion.json            todas las cifras de la tabla
  ablacion-metricas.png    seis métricas de calidad, un panel por métrica
  ablacion-<metrica>.png   cada métrica sola, para una diapositiva
  ablacion-por-tipo.png    nDCG@5 por tipo de consulta y configuración
  ablacion-costo.png       USD, tokens, latencia y llamadas LLM por consulta

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
        clave = _clave(cfg)
        if clave not in buscadas or not corrida.get("agrega_a_la_tabla", True) or not estado.get("top5"):
            continue
        if (estado.get("entrada") or {}).get("procedencia") == "ad_hoc":
            continue
        fila = {
            "ordinal": int(corrida.get("ordinal", 1)),
            "fecha": corrida.get("fecha") or "",
            "top5": [x["artwork_id"] if isinstance(x, dict) else x for x in estado["top5"]],
            "libro": {r: l for r, l in (corrida.get("libro_mayor") or {}).items() if isinstance(l, dict)},
            "segundos": (corrida.get("latencia") or {}).get("segundos_de_la_corrida"),
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
               "tipos": {"conceptual": "#3987e5", "visual": "#d95926", "ambiguo": "#199e70"}},
    "claro": {"fondo": "#ffffff", "texto": "#0b0b0b", "secundario": "#52514e", "linea": "#d9d8d2",
              "barra": "#a3a29a", "final": "#b8860b",
              "tipos": {"conceptual": "#2a78d6", "visual": "#eb6834", "ambiguo": "#1baf7a"}},
}
METRICAS = [  # (clave, título, archivo, decimales, máximo del eje)
    ("ndcg", "nDCG@5", "ndcg", 3, 1.0),
    ("p5", "P@5 (grado ≥ 1)", "p5", 3, 1.0),
    ("p5_estricta", "P@5 estricta (grado 2)", "p5-estricta", 3, 1.0),
    ("mrr", "MRR", "mrr", 3, 1.0),
    ("qs", "QS (cuánto cambia el top-5 con el pedido)", "qs", 3, 1.0),
    ("grado2_por_top5", "Obras de grado 2 por top-5", "grado2", 2, 5.0),
]
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
    ax.set_xticks(range(len(nombres)), [f"{n}\n{tab[n]['agrega']}" if detalle else n for n in nombres])
    ax.yaxis.set_major_formatter(lambda y, _: numero(y, 3 if techo < 0.1 else (1 if techo <= 1 else 0)))


def _guardar(fig, t, archivo) -> str:
    fig.patch.set_facecolor(t["fondo"])
    fig.savefig(c.SALIDA / archivo, dpi=200, facecolor=t["fondo"])
    return archivo


def figuras(tab, tema) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Sin matplotlib no hay figuras: `poetry install --with figuras`.")
        return []
    t = TEMAS[tema]
    hechas = []
    filas = "   ".join(f"{n}: {f['agrega'].lstrip('+ ')}" for n, f in tab.items())
    pie = "30 consultas · 634 juicios · en gris, la diferencia con la configuración anterior"

    fig, ejes = plt.subplots(2, 3, figsize=(16, 8.4))
    for ax, (clave, titulo, _, decimales, maximo) in zip(ejes.flat, METRICAS):
        _barras(ax, t, tab, clave, titulo, decimales, maximo, detalle=False)
    fig.text(0.012, 0.04, filas, color=t["texto"], fontsize=10)
    fig.text(0.012, 0.012, pie + " · línea en nDCG@5: IC95", color=t["secundario"], fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2.5, w_pad=2.5)
    hechas.append(_guardar(fig, t, "ablacion-metricas.png")); plt.close(fig)

    for clave, titulo, archivo, decimales, maximo in METRICAS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _barras(ax, t, tab, clave, titulo, decimales, maximo)
        fig.tight_layout()
        hechas.append(_guardar(fig, t, f"ablacion-{archivo}.png")); plt.close(fig)

    # nDCG@5 por tipo de consulta: un grupo por configuración, un color por tipo.
    fig, ax = plt.subplots(figsize=(11, 5))
    _eje(ax, t, "nDCG@5 por tipo de consulta")
    nombres, ancho = list(tab), 0.25
    for j, tipo in enumerate(c.CATEGORIAS):
        xs = [i + (j - 1) * ancho for i in range(len(nombres))]
        ys = [tab[n]["ndcg"][tipo] for n in nombres]
        ax.bar(xs, ys, width=ancho, color=t["tipos"][tipo], edgecolor=t["fondo"], linewidth=1.5, label=tipo.capitalize())
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, numero(y, 2), ha="center", va="bottom", color=t["texto"], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(nombres)), [f"{n}\n{tab[n]['agrega']}" for n in nombres])
    ax.yaxis.set_major_formatter(lambda y, _: numero(y, 1))
    leyenda = ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=10)
    for texto in leyenda.get_texts():
        texto.set_color(t["texto"])
    fig.text(0.012, 0.015, "10 consultas por tipo · 634 juicios", color=t["secundario"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    hechas.append(_guardar(fig, t, "ablacion-por-tipo.png")); plt.close(fig)

    # Costo y consumo: cada medida en su propio panel, porque las escalas no se comparan.
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8.4))
    for ax, (clave, titulo, decimales) in zip(ejes.flat, COSTOS):
        _barras(ax, t, tab, clave, titulo, decimales, con_delta=False, detalle=False)
    nota = "Por consulta, media de 30. USD y tokens incluyen la llamada del pre-retrieval tomada del plan guardado."
    if any(f["plan_de_cache"] for f in tab.values()):
        nota += " La latencia no incluye esa llamada cuando el plan salió de caché."
    fig.text(0.012, 0.04, filas, color=t["texto"], fontsize=10)
    fig.text(0.012, 0.012, nota, color=t["secundario"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2.5, w_pad=2.5)
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
    hechas = [] if args.sin_figuras else figuras(tab, args.tema)

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
