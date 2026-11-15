"""Análisis de la cascada (el texto filtra, CLIP ordena, el reranker decide el top-5).

Lee las corridas y el ground truth que ya están en `artifacts/` y NO llama a ningún
proveedor. Uso, desde la raíz del repo:

    python analisis/cascada.py                      # resumen + JSON + figuras
    python analisis/cascada.py --juicios-extra X.csv  # sumar juicios nuevos (repetible)

Qué calcula:
  1. Embudo: cuántas obras relevantes sobreviven a cada etapa.
  2. Tamaño del filtro (simulado): quedarse con las N obras mejor rankeadas por texto.
  3. Profundidad del reranker (simulado): rerankear solo las k primeras.
  4. Tabla de experimentos: una fila por configuración corrida, con IC95 por bootstrap.
  5. Diferencias pareadas contra la base y política adaptativa por tipo de pedido.
  6. Pares sin juzgar de cada experimento, listos para `tp3-anotar juez --solo-pares`.

Un par (consulta, obra) sin juicio cuenta como no relevante (convención TREC), así que
toda cifra va con su porcentaje sin juzgar: con mucho sin juzgar, es una cota inferior.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
from math import log2
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
GT = RAIZ / "artifacts" / "ground-truth"
SALIDA = RAIZ / "artifacts" / "analisis"
SPLIT = "5b6cba32126fed61"
JUICIOS_BASE = [
    GT / f"consenso-{SPLIT}-definitivo-v1.csv",
    GT / "juicios" / f"{SPLIT}-v1-ia-cascada.csv",
    GT / "juicios" / f"{SPLIT}-v1-ia-incremental.csv",
]
CATEGORIAS = ("conceptual", "visual", "ambiguo")
BASE_C3 = "C3·filtro100"
BASE_C4 = "C4·filtro100·rerank20"
K = 5


# ---------------------------------------------------------------- carga

def cargar_juicios(archivos) -> dict[tuple[str, str], int]:
    """El primer archivo que juzga un par gana: el consenso nunca se pisa."""
    gt: dict[tuple[str, str], int] = {}
    for archivo in archivos:
        with open(archivo, encoding="utf-8", newline="") as f:
            filas = [linea for linea in f if not linea.startswith("#")]
        for fila in csv.DictReader(filas):
            clave = (fila["consulta_id"], fila["artwork_id"])
            if clave not in gt and fila.get("grado") not in (None, ""):
                gt[clave] = int(fila["grado"])
    return gt


def etiqueta(cfg: dict) -> str:
    consulta = cfg["consulta"]
    nombre = consulta["configuracion"]
    partes = [nombre, f"filtro{cfg['recuperacion_textual']['n_fragmentos_recuperados']}"]
    if cfg["recuperacion_textual"].get("filtro_por_metadata"):
        partes.append("metadata")
    if "reranker" in consulta["etapas"]:
        partes.append(f"rerank{consulta['n_candidatos']}")
        version = str(((cfg.get("bindings") or {}).get("reranker") or {}).get("version_del_prompt", "1"))
        if version != "1":
            partes.append(f"prompt{version}")
    if float(consulta["alfa"]) != 0.5:
        partes.append(f"alfa{consulta['alfa']}")
    return "·".join(partes)


def cargar_corridas() -> dict[str, dict[str, dict]]:
    """{experimento: {consulta_id: estado}} — solo cascada, solo corridas que agregan a la tabla."""
    experimentos: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for archivo in sorted(glob.glob(str(RAIZ / "artifacts" / "corridas" / "*" / "corrida.json"))):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        cfg = corrida["configuracion_resuelta"]
        if (cfg.get("fusion") or {}).get("estrategia") != "cascada":
            continue
        if not corrida.get("agrega_a_la_tabla", True):
            continue
        if ((corrida["estado"].get("entrada") or {}).get("procedencia")) == "ad_hoc":
            continue
        estado = corrida["estado"]
        if not estado.get("ranking_fusionado") or not estado.get("top5"):
            continue
        estado["_usd"] = _usd(corrida)
        estado["_ordinal"] = int(corrida.get("ordinal", 1))
        previo = experimentos[etiqueta(cfg)].get(corrida["consulta_id"])
        if previo is None or estado["_ordinal"] > previo["_ordinal"]:  # el último ordinal, como el pool
            experimentos[etiqueta(cfg)][corrida["consulta_id"]] = estado
    return dict(experimentos)


def _usd(corrida: dict) -> float | None:
    cfg = corrida["configuracion_resuelta"]["proveedor"]
    entrada = cfg.get("precio_usd_por_millon_de_tokens") or {}
    salida = cfg.get("precio_usd_por_millon_de_tokens_de_salida") or {}
    bindings = corrida["configuracion_resuelta"].get("bindings") or {}
    total = 0.0
    for rol, libro in (corrida.get("libro_mayor") or {}).items():
        if not isinstance(libro, dict) or "tokens_in" not in libro:
            continue
        modelo = (bindings.get(rol) or {}).get("modelo") or (
            corrida["configuracion_resuelta"]["embebido_texto"]["modelo"] if rol == "embedding_texto" else None
        )
        if modelo is None:
            continue
        total += libro.get("tokens_in", 0) * entrada.get(modelo, 0) / 1e6
        total += libro.get("tokens_out", 0) * salida.get(modelo, 0) / 1e6
    return round(total, 5) if total else None


# ---------------------------------------------------------------- métricas

def categoria(consulta_id: str) -> str:
    return consulta_id.split("-", 1)[1]


def dcg(grados) -> float:
    return sum(g / log2(i + 1) for i, g in enumerate(grados, start=1))


class Evaluador:
    def __init__(self, gt):
        self.gt = gt
        self.ideales = collections.defaultdict(list)
        for (consulta, _), grado in gt.items():
            self.ideales[consulta].append(grado)

    def ndcg(self, consulta, ids) -> float:
        ideal = dcg(sorted(self.ideales[consulta], reverse=True)[:K])
        if ideal <= 0:
            return 0.0  # como `ndcg_sobre_todas` de la tabla publicada
        return dcg([self.gt.get((consulta, a), 0) for a in ids[:K]]) / ideal

    def p5(self, consulta, ids, umbral=1) -> float:
        return sum(1 for a in ids[:K] if self.gt.get((consulta, a), 0) >= umbral) / K

    def sin_juzgar(self, consulta, ids) -> list[str]:
        return [a for a in ids[:K] if (consulta, a) not in self.gt]

    def relevantes(self, consulta, umbral=2) -> set[str]:
        return {a for (c, a), g in self.gt.items() if c == consulta and g >= umbral}


def media(valores) -> float:
    valores = list(valores)
    return sum(valores) / len(valores) if valores else float("nan")


def ic95(valores, n=4000, semilla=7369) -> tuple[float, float]:
    """Bootstrap: se remuestrean consultas con reposición y se recalcula la media."""
    azar = random.Random(semilla)
    valores = list(valores)
    medias = sorted(media(azar.choices(valores, k=len(valores))) for _ in range(n))
    return round(medias[int(0.025 * n)], 3), round(medias[int(0.975 * n)], 3)


def por_categoria(consultas, funcion) -> dict[str, float]:
    salida = {c: round(media(funcion(q) for q in consultas if categoria(q) == c), 3) for c in CATEGORIAS}
    salida["todas"] = round(media(funcion(q) for q in consultas), 3)
    return salida


# ---------------------------------------------------------------- análisis

def embudo(ev, c3, c4, umbral=2) -> dict:
    filas = collections.defaultdict(collections.Counter)
    for consulta, estado in c4.items():
        relevantes = ev.relevantes(consulta, umbral)
        elegibles = [x["artwork_id"] for x in estado["ranking_fusionado"]]
        n = len(estado.get("rerank") or []) or 20
        visual = {c["artwork_id"] for c in estado["fuentes_de_ranking"]["visual"]}
        for clave in (categoria(consulta), "todas"):
            f = filas[clave]
            f["consultas"] += 1
            f["conocidas"] += len(relevantes)
            f["pasan_el_filtro"] += len(relevantes & set(elegibles))
            f["top_n_de_clip"] += len(relevantes & set(elegibles[:n]))
            if consulta in c3:
                f["top5_sin_reranker"] += len(relevantes & set(c3[consulta]["top5"]))
            f["top5_con_reranker"] += len(relevantes & set(estado["top5"]))
            f["descartadas_por_el_filtro_que_la_rama_visual_tenia"] += len((relevantes & visual) - set(elegibles))
    return {
        clave: {k: round(v / f["consultas"], 2) for k, v in f.items() if k != "consultas"}
        for clave, f in filas.items()
    }


def top5_con_filtro(estado, n) -> list[str]:
    return [x["artwork_id"] for x in estado["ranking_fusionado"] if x["rangos_de_origen"]["texto"] <= n][:K]


def barrido_del_filtro(ev, c3, tamanos=(5, 10, 15, 20, 30, 50, 75, 100)) -> list[dict]:
    consultas = sorted(c3)
    filas = []
    for n in tamanos:
        tops = {q: top5_con_filtro(c3[q], n) for q in consultas}
        filas.append({
            "obras_elegibles": n,
            "ndcg": por_categoria(consultas, lambda q: ev.ndcg(q, tops[q])),
            "ic95": ic95(ev.ndcg(q, tops[q]) for q in consultas),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, tops[q])) for q in consultas) / (K * len(consultas)), 1),
        })
    referencias = {
        "solo_texto": {q: [c["artwork_id"] for c in c3[q]["fuentes_de_ranking"]["texto"]][:K] for q in consultas},
        "solo_visual": {q: [c["artwork_id"] for c in c3[q]["fuentes_de_ranking"]["visual"]][:K] for q in consultas},
    }
    for nombre, tops in referencias.items():
        filas.append({
            "obras_elegibles": nombre,
            "ndcg": por_categoria(consultas, lambda q: ev.ndcg(q, tops[q])),
            "ic95": ic95(ev.ndcg(q, tops[q]) for q in consultas),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, tops[q])) for q in consultas) / (K * len(consultas)), 1),
        })
    return filas


def top5_rerankeando(estado, k) -> list[str]:
    """Mismo orden que `modulo_b/reranker.py::ordenar`: score desc, rango fusionado asc."""
    vistas = [x for x in estado["rerank"] if x["rango_fusionado"] <= k]
    vistas.sort(key=lambda x: (-(x["score_0_10"] if x["score_0_10"] is not None else -1), x["rango_fusionado"]))
    return [x["artwork_id"] for x in vistas[:K]]


def barrido_del_reranker(ev, c4) -> list[dict]:
    consultas = sorted(c4)
    maximo = min(len(c4[q]["rerank"]) for q in consultas)
    filas = []
    for k in [k for k in (5, 8, 10, 12, 15, 20, 25, 30, 35, 40) if k <= maximo]:
        tops = {q: top5_rerankeando(c4[q], k) for q in consultas}
        filas.append({
            "candidatas": k,
            "ndcg": por_categoria(consultas, lambda q: ev.ndcg(q, tops[q])),
            "ic95": ic95(ev.ndcg(q, tops[q]) for q in consultas),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, tops[q])) for q in consultas) / (K * len(consultas)), 1),
        })
    return filas


def calibracion_del_reranker(ev, c4) -> dict:
    por_grado = collections.defaultdict(list)
    for consulta, estado in c4.items():
        for x in estado["rerank"]:
            grado = ev.gt.get((consulta, x["artwork_id"]))
            if grado is not None and x["score_0_10"] is not None:
                por_grado[grado].append(x["score_0_10"])
    return {str(g): {"n": len(v), "score_medio": round(media(v), 2)} for g, v in sorted(por_grado.items())}


def tabla_de_experimentos(ev, experimentos) -> dict:
    tabla = {}
    for nombre, corridas in sorted(experimentos.items()):
        consultas = sorted(corridas)
        costos = [corridas[q]["_usd"] for q in consultas if corridas[q].get("_usd")]
        tabla[nombre] = {
            "consultas": len(consultas),
            "ndcg": por_categoria(consultas, lambda q: ev.ndcg(q, corridas[q]["top5"])),
            "ic95": ic95(ev.ndcg(q, corridas[q]["top5"]) for q in consultas),
            "p5": round(media(ev.p5(q, corridas[q]["top5"]) for q in consultas), 3),
            "p5_estricta": round(media(ev.p5(q, corridas[q]["top5"], 2) for q in consultas), 3),
            "elegibles_medio": round(media(len(corridas[q]["ranking_fusionado"]) for q in consultas), 1),
            "pct_sin_juzgar": round(100 * sum(len(ev.sin_juzgar(q, corridas[q]["top5"])) for q in consultas) / (K * len(consultas)), 1),
            "usd_por_consulta": round(media(costos), 4) if costos else None,
        }
    return tabla


def pareadas(ev, experimentos, base) -> dict:
    if base not in experimentos:
        return {}
    salida = {}
    for nombre, corridas in experimentos.items():
        comunes = sorted(set(corridas) & set(experimentos[base]))
        if nombre == base or not comunes:
            continue
        deltas = [ev.ndcg(q, corridas[q]["top5"]) - ev.ndcg(q, experimentos[base][q]["top5"]) for q in comunes]
        salida[f"{nombre} − {base}"] = {"consultas": len(comunes), "delta": round(media(deltas), 3), "ic95": ic95(deltas)}
    return salida


def politica_adaptativa(ev, experimentos) -> dict:
    """Elige una configuración por tipo de pedido entre las que comparten cadena de etapas.

    Es un techo optimista: la categoría se toma del set de evaluación (un oráculo) y la
    elección se hace sobre las mismas consultas con las que se mide.
    """
    salida = {}
    for prefijo in ("C3", "C4"):
        candidatos = {n: c for n, c in experimentos.items() if n.startswith(prefijo) and len(c) == 30}
        if len(candidatos) < 2:
            continue
        eleccion, valores = {}, []
        for cat in CATEGORIAS:
            mejor = max(candidatos, key=lambda n: media(ev.ndcg(q, candidatos[n][q]["top5"]) for q in candidatos[n] if categoria(q) == cat))
            eleccion[cat] = mejor
            valores += [ev.ndcg(q, candidatos[mejor][q]["top5"]) for q in candidatos[mejor] if categoria(q) == cat]
        salida[prefijo] = {"eleccion": eleccion, "ndcg": round(media(valores), 3), "ic95": ic95(valores)}
    return salida


def pares_sin_juzgar(ev, experimentos) -> dict[str, list[list[str]]]:
    return {
        nombre: [[q, a] for q in sorted(corridas) for a in ev.sin_juzgar(q, corridas[q]["top5"])]
        for nombre, corridas in experimentos.items()
    }


# ---------------------------------------------------------------- figuras

def figuras(resultados) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    hechas = []
    colores = {"conceptual": "#1f77b4", "visual": "#ff7f0e", "ambiguo": "#2ca02c", "todas": "#333333"}

    emb = resultados["embudo_grado_2"]
    if emb:
        etapas = ["conocidas", "pasan_el_filtro", "top_n_de_clip", "top5_sin_reranker", "top5_con_reranker"]
        nombres = ["Conocidas", "Pasan el filtro", "Top-N de CLIP", "Top-5 sin reranker", "Top-5 con reranker"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ancho = 0.25
        for i, cat in enumerate(CATEGORIAS):
            ax.bar([x + (i - 1) * ancho for x in range(len(etapas))], [emb[cat].get(e, 0) for e in etapas], ancho, label=cat, color=colores[cat])
        ax.set_xticks(range(len(etapas)), nombres)
        ax.set_ylabel("Obras de grado 2 por consulta")
        ax.set_title("Embudo de la cascada: dónde se pierden las obras relevantes")
        ax.legend()
        fig.tight_layout(); fig.savefig(SALIDA / "embudo.png", dpi=150); plt.close(fig)
        hechas.append("embudo.png")

    for clave, eje_x, titulo, archivo in (
        ("barrido_del_filtro", "obras_elegibles", "nDCG@5 según el tamaño del filtro (sin reranker, simulado)", "filtro.png"),
        ("barrido_del_reranker", "candidatas", "nDCG@5 según cuántas candidatas ve el reranker", "reranker.png"),
    ):
        filas = [f for f in resultados[clave] if isinstance(f[eje_x], int)]
        if not filas:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [f[eje_x] for f in filas]
        for cat in (*CATEGORIAS, "todas"):
            ax.plot(xs, [f["ndcg"][cat] for f in filas], marker="o", label=cat, color=colores[cat], linewidth=2.5 if cat == "todas" else 1.5)
        ax.fill_between(xs, [f["ic95"][0] for f in filas], [f["ic95"][1] for f in filas], color="#333333", alpha=0.1, label="IC95 (todas)")
        if clave == "barrido_del_filtro":
            for f in filas:
                ax.annotate(f"{f['pct_sin_juzgar']:.0f}% s/j", (f[eje_x], f["ndcg"]["todas"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7)
        ax.set_xlabel("Obras elegibles" if clave == "barrido_del_filtro" else "Candidatas rerankeadas")
        ax.set_ylabel("nDCG@5")
        ax.set_title(titulo)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(SALIDA / archivo, dpi=150); plt.close(fig)
        hechas.append(archivo)

    tabla = {n: f for n, f in resultados["experimentos"].items() if f["usd_por_consulta"] and f["consultas"] == 30}
    if len(tabla) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for nombre, f in tabla.items():
            ax.errorbar(f["usd_por_consulta"], f["ndcg"]["todas"], yerr=[[f["ndcg"]["todas"] - f["ic95"][0]], [f["ic95"][1] - f["ndcg"]["todas"]]], fmt="o", capsize=3)
            ax.annotate(nombre, (f["usd_por_consulta"], f["ndcg"]["todas"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("USD por consulta (precio de lista, sin la llamada del plan)")
        ax.set_ylabel("nDCG@5 (IC95)")
        ax.set_title("Calidad contra costo, por experimento")
        fig.tight_layout(); fig.savefig(SALIDA / "costo.png", dpi=150); plt.close(fig)
        hechas.append("costo.png")
    return hechas


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--juicios-extra", type=Path, action="append", default=[], help="CSV con consulta_id, artwork_id, grado (repetible)")
    p.add_argument("--sin-figuras", action="store_true")
    args = p.parse_args()

    # Los juicios hechos a mano con la planilla se suman solos: después del consenso y de
    # los juicios de IA, así que nunca pisan un par que ya estaba juzgado.
    a_mano = sorted(SALIDA.glob("juicios-*.csv"))
    args.juicios_extra = [*a_mano, *[a for a in args.juicios_extra if a.resolve() not in {m.resolve() for m in a_mano}]]
    ev = Evaluador(cargar_juicios([*JUICIOS_BASE, *args.juicios_extra]))
    experimentos = cargar_corridas()
    c3 = experimentos.get(BASE_C3, {})
    c4 = experimentos.get(BASE_C4, {})
    mas_profundo = max((e for n, e in experimentos.items() if n.startswith("C4·filtro100·rerank") and len(e) == 30),
                       key=lambda e: min(len(x["rerank"]) for x in e.values()), default=c4)

    resultados = {
        "juicios": len(ev.gt),
        "juicios_extra": [str(a.relative_to(RAIZ)) if a.is_absolute() and RAIZ in a.parents else str(a) for a in args.juicios_extra],
        "experimentos": tabla_de_experimentos(ev, experimentos),
        "embudo_grado_2": embudo(ev, c3, c4, 2) if c4 else {},
        "embudo_grado_1_o_mas": embudo(ev, c3, c4, 1) if c4 else {},
        "barrido_del_filtro": barrido_del_filtro(ev, c3) if c3 else [],
        "barrido_del_reranker": barrido_del_reranker(ev, mas_profundo) if mas_profundo else [],
        "calibracion_del_reranker": calibracion_del_reranker(ev, mas_profundo) if mas_profundo else {},
        "pareadas_contra_C3": pareadas(ev, {n: e for n, e in experimentos.items() if n.startswith("C3")}, BASE_C3),
        "pareadas_contra_C4": pareadas(ev, {n: e for n, e in experimentos.items() if n.startswith("C4")}, BASE_C4),
        "C4_menos_C3": pareadas(ev, {BASE_C3: c3, BASE_C4: c4}, BASE_C3) if c3 and c4 else {},
        "politica_adaptativa": politica_adaptativa(ev, experimentos),
    }
    pendientes = pares_sin_juzgar(ev, experimentos)
    union = sorted({tuple(par) for pares in pendientes.values() for par in pares})

    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    (SALIDA / "pares-sin-juzgar.json").write_text(json.dumps([list(par) for par in union], ensure_ascii=False), encoding="utf-8")
    hechas = [] if args.sin_figuras else figuras(resultados)

    print(f"Juicios: {len(ev.gt)} · experimentos en cascada: {len(experimentos)}\n")
    print(f"{'experimento':<34}{'n':>3}{'nDCG@5':>8}  {'IC95':<16}{'concep':>7}{'visual':>7}{'ambig':>7}{'P@5 2':>7}{'s/j %':>7}{'USD':>8}")
    for nombre, f in resultados["experimentos"].items():
        print(f"{nombre:<34}{f['consultas']:>3}{f['ndcg']['todas']:>8.3f}  {str(f['ic95']):<16}{f['ndcg']['conceptual']:>7.3f}{f['ndcg']['visual']:>7.3f}{f['ndcg']['ambiguo']:>7.3f}{f['p5_estricta']:>7.3f}{f['pct_sin_juzgar']:>7.1f}{(f['usd_por_consulta'] or 0):>8.4f}")
    for titulo in ("C4_menos_C3", "pareadas_contra_C3", "pareadas_contra_C4"):
        for nombre, f in resultados[titulo].items():
            print(f"\nΔ {nombre}: {f['delta']:+.3f}  IC95 {f['ic95']}  ({f['consultas']} consultas)")
    if resultados["politica_adaptativa"]:
        print("\nPolítica adaptativa (techo optimista):", json.dumps(resultados["politica_adaptativa"], ensure_ascii=False))
    print(f"\nPares sin juzgar: {len(union)} en total →", {n: len(v) for n, v in pendientes.items() if v})
    print(f"Escrito en {SALIDA.relative_to(RAIZ)}: resultados.json, pares-sin-juzgar.json" + (", " + ", ".join(hechas) if hechas else ""))


if __name__ == "__main__":
    main()
