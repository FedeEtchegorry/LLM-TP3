"""Diagnóstico del reranker sobre la arquitectura final. No llama a ningún proveedor.

    python analisis/diagnostico.py

1. Confianza: ¿el puntaje medio que el reranker le da a su top-5 predice la calidad?
2. Calibración: ¿qué puntaje reciben las obras de cada grado, por tipo de pedido?
3. Desempate: ¿el orden de CLIP, que desempata los puntajes iguales, lleva información?
"""
from __future__ import annotations

import collections
import json

import cascada as c

FINAL = "C4·filtro100·metadata·rerank20"
UMBRAL = 7.0


def _rangos(v):
    orden = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and v[orden[j + 1]] == v[orden[i]]:
            j += 1
        for k in range(i, j + 1):
            r[orden[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = _rangos(a), _rangos(b)
    ma, mb = c.media(ra), c.media(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def auc(positivos, negativos) -> float:
    if not positivos or not negativos:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in positivos for n in negativos) / (len(positivos) * len(negativos))


def main() -> None:
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(c.SALIDA.glob("juicios-*.csv"))]))
    corridas = c.cargar_corridas()[FINAL]
    consultas = sorted(corridas)

    confianza, ndcg, aciertos = [], [], []
    for q in consultas:
        puntajes = {x["artwork_id"]: x["score_0_10"] or 0 for x in corridas[q]["rerank"]}
        confianza.append(c.media(puntajes[a] for a in corridas[q]["top5"]))
        ndcg.append(ev.ndcg(q, corridas[q]["top5"]))
        aciertos.append(5 * ev.p5(q, corridas[q]["top5"], 2))
    bajo = [i for i, s in enumerate(confianza) if s < UMBRAL]
    alto = [i for i, s in enumerate(confianza) if s >= UMBRAL]
    resultado = {
        "experimento": FINAL,
        "confianza": {
            "spearman_con_ndcg": round(spearman(confianza, ndcg), 2),
            "spearman_con_obras_de_grado_2": round(spearman(confianza, aciertos), 2),
            "umbral": UMBRAL,
            "debajo_del_umbral": {
                "consultas": len(bajo),
                "ndcg": round(c.media(ndcg[i] for i in bajo), 3),
                "obras_de_grado_2_por_top5": round(c.media(aciertos[i] for i in bajo), 2),
                "por_categoria": dict(collections.Counter(c.categoria(consultas[i]) for i in bajo)),
            },
            "sobre_el_umbral": {
                "consultas": len(alto),
                "ndcg": round(c.media(ndcg[i] for i in alto), 3),
                "obras_de_grado_2_por_top5": round(c.media(aciertos[i] for i in alto), 2),
                "por_categoria": dict(collections.Counter(c.categoria(consultas[i]) for i in alto)),
            },
            "por_consulta": {q: {"confianza": round(s, 1), "ndcg": round(n, 2)} for q, s, n in zip(consultas, confianza, ndcg)},
        },
        "calibracion_por_categoria": {},
    }
    for cat in c.CATEGORIAS:
        puntaje, clip = collections.defaultdict(list), collections.defaultdict(list)
        for q in consultas:
            if c.categoria(q) != cat:
                continue
            for x in corridas[q]["rerank"]:
                grado = ev.gt.get((q, x["artwork_id"]))
                if grado is not None and x["score_0_10"] is not None:
                    puntaje[grado].append(x["score_0_10"])
                    clip[grado].append(-x["rango_fusionado"])
        resultado["calibracion_por_categoria"][cat] = {
            "puntaje_medio_por_grado": {str(g): round(c.media(puntaje[g]), 1) for g in (0, 1, 2) if puntaje[g]},
            "auc_del_reranker_grado2_vs_grado0": round(auc(puntaje[2], puntaje[0]), 2),
            "auc_del_orden_de_clip_grado2_vs_resto": round(auc(clip[2], clip[0] + clip[1]), 2),
        }
    (c.SALIDA / "diagnostico.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    conf = resultado["confianza"]
    print(f"Confianza vs calidad: Spearman {conf['spearman_con_ndcg']} (nDCG), {conf['spearman_con_obras_de_grado_2']} (obras de grado 2)")
    for clave in ("debajo_del_umbral", "sobre_el_umbral"):
        print(f"  {clave} ({UMBRAL}): {conf[clave]}")
    for cat, fila in resultado["calibracion_por_categoria"].items():
        print(f"  {cat}: {fila}")
    print("Escrito en artifacts/analisis/diagnostico.json")


if __name__ == "__main__":
    main()
