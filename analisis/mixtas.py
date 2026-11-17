"""Resultados del set mixto (10 consultas, una por imagen): tabla, JSON y figura. No gasta.

    python analisis/mixtas.py                 # tabla + mixtas.json + mixtas.png
    python analisis/mixtas.py --sin-figuras

Lee las corridas `cNN-mixto` y los `juicios-*.csv` que ya están en `artifacts/`, y escribe en
`artifacts/analisis/`: `mixtas.json` y `mixtas.png`. Aparte de `cascada.py` y `ablacion.py`
porque el set es complementario y sus juicios llevan los componentes A, B y C (rúbrica en
`docs/rubrica-mixta.md`). Por (configuración, consulta) se usa la corrida de mayor ordinal y, si
empatan, la más reciente. Un par sin juicio cuenta como 0, así que `pct_sin_juzgar` acota cuánto
puede estar subestimada cada fila.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import ablacion as ab
import cascada as c

DECLARACION = c.RAIZ / "consultas-g2-mixto.toml"
BASE = [  # (etiqueta de cascada.etiqueta o nombre de C0 a C2, nombre corto)
    ("C0", "C0"),
    ("C1", "C1"),
    ("C2", "C2"),
    ("C3·filtro100·metadata", "C3"),
    ("C4·filtro100·metadata·rerank20", "C4"),
]
VARIANTES = [
    ("C3·filtro100", "C3 sin filtro"),  # config/c3.toml, hoy sin corridas sobre las mixtas
    ("C4·filtro100·rerank20", "C4 sin filtro"),  # config/c4.toml, ídem
    ("C4·filtro100·metadata·rerank20·gpt-6-luna", "C4 luna"),
    ("C4·filtro100·metadata·rerank20·prompt2", "C4 prompt v2"),
    ("C4·filtro100·metadata·rerank40", "C4 40 candidatas"),
    ("C4·filtro100·metadata·rerank40·gpt-6-luna", "C4 luna 40 candidatas"),
    ("C4·filtro100·metadata·rerank20·gpt-6.1-terra", "C4 terra"),
]
PASOS = [  # (nombre, base)
    ("C1", "C0"), ("C2", "C1"), ("C3", "C2"), ("C4", "C3"),
    ("C3", "C3 sin filtro"), ("C4", "C4 sin filtro"),
    ("C4 luna", "C4"), ("C4 prompt v2", "C4"), ("C4 40 candidatas", "C4"), ("C4 terra", "C4"),
    ("C4 luna 40 candidatas", "C4 40 candidatas"),
]
CONTROL = "c06-mixto"  # sin restricción: no entra en la tasa de B


def _clave(cfg: dict) -> str | None:
    nombre = cfg["consulta"]["configuracion"]
    if nombre in ("C0", "C1", "C2"):
        return nombre
    return c.etiqueta(cfg) if (cfg.get("fusion") or {}).get("estrategia") == "cascada" else None


def cargar() -> dict[str, dict[str, dict]]:
    """{etiqueta: {consulta_id: top5, _usd}} con las corridas válidas del set mixto."""
    buscadas = {e for e, _ in BASE + VARIANTES}
    elegidas: dict[str, dict[str, tuple]] = {e: {} for e in buscadas}
    for archivo in sorted(glob.glob(str(c.RAIZ / "artifacts" / "corridas" / "*" / "corrida.json"))):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        if not corrida["consulta_id"].endswith("-mixto"):
            continue
        estado = corrida["estado"]
        clave = _clave(corrida["configuracion_resuelta"])
        if clave not in buscadas or not corrida.get("agrega_a_la_tabla", True) or not estado.get("top5"):
            continue
        if (estado.get("entrada") or {}).get("procedencia") == "ad_hoc":
            continue
        if any(r.get("estado") == "sin_puntaje" for r in (estado.get("rerank") or [])):
            continue
        if any(d.get("etapa") == "recuperador_textual" for d in (estado.get("degradaciones") or [])):
            continue  # rama de texto caída (bug del año en el filtro): la corrida no midió el filtro
        orden = (int(corrida.get("ordinal", 1)), corrida.get("fecha") or "")
        previo = elegidas[clave].get(corrida["consulta_id"])
        if previo is None or orden > previo[0]:
            top5 = [x["artwork_id"] if isinstance(x, dict) else x for x in estado["top5"]]
            elegidas[clave][corrida["consulta_id"]] = (orden, {"top5": top5, "usd": c._usd(corrida)})
    return {e: {q: v[1] for q, v in por_q.items()} for e, por_q in elegidas.items() if por_q}


def componentes() -> dict[tuple[str, str], dict[str, int]]:
    """{(consulta, obra): {a, b, c}}: el primer archivo que juzga un par gana, como en cascada.py."""
    salida: dict[tuple[str, str], dict[str, int]] = {}
    for archivo in sorted(glob.glob(str(c.SALIDA / "juicios-mixtas-*.csv"))):
        with open(archivo, encoding="utf-8", newline="") as f:
            filas = [linea for linea in f if not linea.startswith("#")]
        for fila in csv.DictReader(filas):
            clave = (fila["consulta_id"], fila["artwork_id"])
            if clave in salida or fila.get("cambio_visual") in (None, ""):
                continue
            salida[clave] = {k: int(fila[col]) for k, col in (("a", "cambio_visual"), ("b", "restriccion"), ("c", "adjetivo")) if fila.get(col) not in (None, "")}
    return salida


def tasa(comp, corridas, clave) -> float | None:
    """Parte de las obras juzgadas del top-5 que cumplen el componente (B sin el control)."""
    valores = [
        comp[(q, o)][clave] for q, fila in corridas.items() for o in fila["top5"]
        if (q, o) in comp and clave in comp[(q, o)] and not (clave == "b" and q == CONTROL)
    ]
    return round(c.media(valores), 3) if valores else None


def tabla(ev, comp, elegidas) -> dict:
    salida = {}
    for etiqueta, nombre in BASE + VARIANTES:
        corridas = elegidas.get(etiqueta)
        if not corridas:
            continue
        consultas = sorted(corridas)
        ndcg = [ev.ndcg(q, corridas[q]["top5"]) for q in consultas]
        costos = [corridas[q]["usd"] for q in consultas if corridas[q]["usd"]]
        salida[nombre] = {
            "etiqueta": etiqueta,
            "consultas": len(consultas),
            "ndcg": round(c.media(ndcg), 3),
            "ic95": c.ic95(ndcg),
            "p5_estricta": round(c.media(ev.p5(q, corridas[q]["top5"], 2) for q in consultas), 3),
            "grado2_por_top5": round(c.media(c.K * ev.p5(q, corridas[q]["top5"], 2) for q in consultas), 2),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, corridas[q]["top5"])) for q in consultas) / (c.K * len(consultas)), 1),
            "cumple_cambio_visual": tasa(comp, corridas, "a"),
            "cumple_restriccion": tasa(comp, corridas, "b"),
            "usd_por_consulta": round(c.media(costos), 4) if costos else None,
            "ndcg_por_consulta": {q: round(n, 3) for q, n in zip(consultas, ndcg)},
        }
    return salida


def pareadas(ev, elegidas, nombres) -> dict:
    por_nombre = {n: elegidas[e] for e, n in BASE + VARIANTES if e in elegidas}
    salida = {}
    for nombre, base in PASOS:
        if nombre not in por_nombre or base not in por_nombre:
            continue
        comunes = sorted(set(por_nombre[nombre]) & set(por_nombre[base]))
        deltas = [ev.ndcg(q, por_nombre[nombre][q]["top5"]) - ev.ndcg(q, por_nombre[base][q]["top5"]) for q in comunes]
        salida[f"{nombre} − {base}"] = {
            "consultas": len(comunes), "delta": round(c.media(deltas), 3), "ic95": c.ic95(deltas),
            "mejoran": sum(d > 1e-9 for d in deltas), "empatan": sum(abs(d) <= 1e-9 for d in deltas), "empeoran": sum(d < -1e-9 for d in deltas),
        }
    return salida


def figura(tab, tema="claro") -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Sin matplotlib no hay figuras: `poetry install --with figuras`.")
        return []
    t = ab.TEMAS[tema]
    nombres = [n for _, n in BASE if n in tab]
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 5))
    ab._eje(izq, t, "nDCG@5 en las consultas mixtas")
    ys = [tab[n]["ndcg"] for n in nombres]
    colores = [t["final"] if n == nombres[-1] else t["barra"] for n in nombres]
    izq.bar(range(len(nombres)), ys, color=colores, edgecolor=t["fondo"], width=0.6)
    for i, n in enumerate(nombres):
        bajo, alto = tab[n]["ic95"]
        izq.plot([i, i], [bajo, alto], color=t["texto"], linewidth=1.2)
        izq.text(i + 0.08, tab[n]["ndcg"] + 0.012, ab.numero(tab[n]["ndcg"], 3), color=t["texto"], fontsize=9)
    izq.set_ylim(0, 1.08)
    izq.set_xticks(range(len(nombres)), nombres)
    izq.yaxis.set_major_formatter(lambda y, _: ab.numero(y, 1))

    ab._eje(der, t, "Obras del top-5 que cumplen cada componente")
    ancho = 0.35
    for j, (clave, nombre, color) in enumerate((("cumple_cambio_visual", "Cambio visual (A)", t["tipos"]["conceptual"]),
                                                ("cumple_restriccion", "Restricción por metadata (B)", t["tipos"]["visual"]))):
        xs = [i + (j - 0.5) * ancho for i in range(len(nombres))]
        vs = [tab[n][clave] or 0 for n in nombres]
        der.bar(xs, vs, width=ancho, color=color, edgecolor=t["fondo"], linewidth=1.5, label=nombre)
        for x, v in zip(xs, vs):
            der.text(x, v + 0.012, ab.numero(v, 2), ha="center", va="bottom", color=t["texto"], fontsize=8)
    der.set_ylim(0, 1.08)
    der.set_xticks(range(len(nombres)), nombres)
    der.yaxis.set_major_formatter(lambda y, _: ab.numero(y, 1))
    leyenda = der.legend(frameon=False, ncols=2, loc="upper left", fontsize=9)
    for texto in leyenda.get_texts():
        texto.set_color(t["texto"])
    sj = max(tab[n]["pct_sin_juzgar"] for n in nombres)
    juzgado = "todos los pares del top-5 están juzgados" if sj == 0 else f"pares sin juzgar en el top-5: hasta {ab.numero(sj, 1)} %"
    fig.text(0.012, 0.015, f"10 consultas mixtas · línea en nDCG@5: IC95 · {juzgado}", color=t["secundario"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return [ab._guardar(fig, t, "mixtas.png")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-figuras", action="store_true")
    p.add_argument("--tema", choices=sorted(ab.TEMAS), default="claro")
    args = p.parse_args()
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(c.SALIDA.glob("juicios-*.csv"))]))
    elegidas, comp = cargar(), componentes()
    tab = tabla(ev, comp, elegidas)
    pares = pareadas(ev, elegidas, tab)
    (c.SALIDA / "mixtas.json").write_text(json.dumps({"configuraciones": tab, "pareadas": pares}, ensure_ascii=False, indent=2), encoding="utf-8")
    hechas = [] if args.sin_figuras else figura(tab, args.tema)
    print(f"{'configuración':<24}{'n':>3}{'nDCG@5':>8}  {'IC95':<14}{'P@5 2':>7}{'A':>6}{'B':>6}{'s/j %':>7}{'USD':>8}")
    for nombre, f in tab.items():
        print(f"{nombre:<24}{f['consultas']:>3}{f['ndcg']:>8.3f}  {str(f['ic95']):<14}{f['p5_estricta']:>7.3f}"
              f"{f['cumple_cambio_visual'] or 0:>6.2f}{f['cumple_restriccion'] or 0:>6.2f}{f['pct_sin_juzgar']:>7.1f}{f['usd_por_consulta'] or 0:>8.4f}")
    print()
    for nombre, f in pares.items():
        print(f"{nombre:<34}{f['delta']:>+7.3f}  {str(f['ic95']):<16}{f['mejoran']} mejoran, {f['empatan']} empatan, {f['empeoran']} empeoran")
    print(f"\nEscrito en artifacts/analisis/mixtas.json" + (", " + ", ".join(Path(h).name for h in hechas) if hechas else ""))


if __name__ == "__main__":
    main()
