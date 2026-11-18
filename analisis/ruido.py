"""Ruido entre repeticiones de la misma configuracion (mismo config_id, misma consulta).

Ordena las corridas de cada consulta por fecha y compara repeticion 1, 2, 3...: nDCG@5 medio,
pares sin juzgar y estabilidad del top-5 (Jaccard entre repeticiones). No gasta nada.

  python analisis/ruido.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
import cascada as c  # noqa: E402

CONFIGS = {
    "C3·filtro100·metadata": "C3 + metadata (sin reranker)",
    "C4·filtro100·metadata·rerank20": "C4 + metadata (reranker mini, 20)",
}


def main() -> None:
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(c.SALIDA.glob("juicios-*.csv"))]))
    grupos = collections.defaultdict(lambda: collections.defaultdict(list))  # etiqueta -> consulta -> [(mtime, top5)]
    for archivo in glob.glob(str(c.RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        corrida = json.load(open(archivo, encoding="utf-8"))
        cfg, estado = corrida["configuracion_resuelta"], corrida["estado"]
        if (cfg.get("fusion") or {}).get("estrategia") != "cascada" or not estado.get("top5"):
            continue
        if any(r.get("estado") == "sin_puntaje" for r in (estado.get("rerank") or [])):
            continue
        etiqueta = c.etiqueta(cfg)
        if etiqueta in CONFIGS and not corrida["consulta_id"].endswith("-mixto"):  # las mixtas se miden aparte
            top5 = [x["artwork_id"] if isinstance(x, dict) else x for x in estado["top5"]]
            grupos[etiqueta][corrida["consulta_id"]].append((os.path.getmtime(archivo), top5))

    salida = {}
    for etiqueta, nombre in CONFIGS.items():
        por_consulta = {q: [t for _, t in sorted(v)] for q, v in grupos[etiqueta].items()}
        n = min(len(v) for v in por_consulta.values())
        filas, jacc = [], []
        for rep in range(n):
            nd = [ev.ndcg(q, v[rep]) for q, v in por_consulta.items()]
            sj = sum(len(ev.sin_juzgar(q, v[rep])) for q, v in por_consulta.items())
            filas.append({"repeticion": rep + 1, "ndcg": round(c.media(nd), 3), "pares_sin_juzgar": sj})
        for q, v in por_consulta.items():
            for a, b in combinations(v[:n], 2):
                jacc.append(len(set(a) & set(b)) / len(set(a) | set(b)))
        # desvio de nDCG por consulta entre repeticiones
        por_q = [max(ev.ndcg(q, t) for t in v[:n]) - min(ev.ndcg(q, t) for t in v[:n]) for q, v in por_consulta.items()]
        cambian = sum(1 for v in por_consulta.values() if len({tuple(t) for t in v[:n]}) > 1)
        medias = [f["ndcg"] for f in filas]
        salida[etiqueta] = {
            "nombre": nombre, "consultas": len(por_consulta), "repeticiones": n, "filas": filas,
            "rango_de_la_media": round(max(medias) - min(medias), 3),
            "jaccard_medio_del_top5": round(c.media(jacc), 3),
            "consultas_con_top5_distinto": cambian,
            "rango_medio_de_ndcg_por_consulta": round(c.media(por_q), 3),
            "top5_distinto_por_consulta": {q: len({tuple(t) for t in v[:n]}) > 1 for q, v in sorted(por_consulta.items())},
            "ndcg_por_consulta": {q: [round(ev.ndcg(q, t), 3) for t in v[:n]] for q, v in sorted(por_consulta.items())},
        }
    (c.SALIDA / "ruido.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    for e, d in salida.items():
        print(f"\n{d['nombre']}  ({d['consultas']} consultas, {d['repeticiones']} repeticiones)")
        for f in d["filas"]:
            print(f"  rep {f['repeticion']}: nDCG@5 {f['ndcg']:.3f}   pares sin juzgar {f['pares_sin_juzgar']}")
        print(f"  rango de la media {d['rango_de_la_media']:.3f} | Jaccard medio del top-5 {d['jaccard_medio_del_top5']:.3f} | "
              f"consultas con top-5 distinto {d['consultas_con_top5_distinto']}/{d['consultas']} | "
              f"rango medio por consulta {d['rango_medio_de_ndcg_por_consulta']:.3f}")


if __name__ == "__main__":
    main()
