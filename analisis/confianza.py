"""Calidad del detector de confianza. No gasta.

    python analisis/confianza.py [--experimento "C4·filtro100·metadata·rerank20·prompt2"]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

import cascada as c
from diagnostico import auc

sys.path.insert(0, str(c.RAIZ / "src"))
from tp3_rag.modulo_b.confianza import UMBRAL_DE_CONFIANZA, confianza_del_top5  # noqa: E402

FINAL = "C4·filtro100·metadata·rerank20"
MINIMO_DE_OBRAS_BUENAS = 3
UMBRALES = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]


def _tabla(filas, umbral):
    """Matriz de confusión. Positivo: el detector avisa."""
    avisa = [f["confianza"] < umbral for f in filas]
    mala = [f["mala"] for f in filas]
    tp = sum(a and m for a, m in zip(avisa, mala))
    fp = sum(a and not m for a, m in zip(avisa, mala))
    fn = sum((not a) and m for a, m in zip(avisa, mala))
    tn = sum((not a) and (not m) for a, m in zip(avisa, mala))
    return {
        "umbral": umbral,
        "avisa": tp + fp,
        "listas_malas_avisadas": tp,
        "avisos_sobre_listas_buenas": fp,
        "listas_malas_sin_aviso": fn,
        "listas_buenas_sin_aviso": tn,
        "precision": round(tp / (tp + fp), 2) if tp + fp else None,
        "cobertura": round(tp / (tp + fn), 2) if tp + fn else None,
        "exactitud": round((tp + tn) / len(filas), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experimento", default=FINAL)
    args = ap.parse_args()

    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(c.SALIDA.glob("juicios-*.csv"))]))
    experimentos = c.cargar_corridas()
    if args.experimento not in experimentos:
        sys.exit(f"No hay corridas de {args.experimento!r}. Hay: {sorted(experimentos)}")
    corridas = experimentos[args.experimento]

    filas = []
    for q in sorted(corridas):
        conf = confianza_del_top5(corridas[q]["rerank"], corridas[q]["top5"])
        if conf is None:
            continue
        buenas = sum(1 for a in corridas[q]["top5"] if ev.gt.get((q, a), 0) >= 2)
        filas.append(
            {
                "consulta": q,
                "categoria": c.categoria(q),
                "confianza": conf.puntaje_medio,
                "obras_de_grado_2": buenas,
                "mala": buenas < MINIMO_DE_OBRAS_BUENAS,
                "ndcg": ev.ndcg(q, corridas[q]["top5"]),
            }
        )
    malas = [f["confianza"] for f in filas if f["mala"]]
    buenas = [f["confianza"] for f in filas if not f["mala"]]

    # dejando una afuera: el umbral más bajo que avisa todas las listas malas
    afuera, elegidos = [], []
    for i, fila in enumerate(filas):
        resto = filas[:i] + filas[i + 1 :]
        elegido = min((u for u in UMBRALES if _tabla(resto, u)["cobertura"] == 1.0), default=max(UMBRALES))
        elegidos.append(elegido)
        afuera.append((fila["confianza"] < elegido) == fila["mala"])

    por_categoria = {}
    for cat in c.CATEGORIAS:
        sub = [f for f in filas if f["categoria"] == cat]
        por_categoria[cat] = {
            "consultas": len(sub),
            "avisa": sum(f["confianza"] < UMBRAL_DE_CONFIANZA for f in sub),
            "listas_malas": sum(f["mala"] for f in sub),
        }

    resultado = {
        "experimento": args.experimento,
        "definicion_de_lista_mala": f"menos de {MINIMO_DE_OBRAS_BUENAS} obras de grado 2 en el top-5",
        "consultas": len(filas),
        "listas_malas": len(malas),
        "auc_confianza_para_separar_buenas_de_malas": round(auc(buenas, malas), 2),
        "umbral_publicado": _tabla(filas, UMBRAL_DE_CONFIANZA),
        "barrido_de_umbrales": [_tabla(filas, u) for u in UMBRALES],
        "exactitud_dejando_una_consulta_afuera": round(sum(afuera) / len(afuera), 2),
        "umbrales_elegidos_dejando_una_consulta_afuera": dict(sorted(collections.Counter(elegidos).items())),
        "por_categoria": por_categoria,
        "por_consulta": {
            f["consulta"]: {
                "confianza": round(f["confianza"], 1),
                "obras_de_grado_2": f["obras_de_grado_2"],
                "ndcg": round(f["ndcg"], 2),
                "avisa": f["confianza"] < UMBRAL_DE_CONFIANZA,
            }
            for f in filas
        },
    }
    destino = c.SALIDA / ("confianza.json" if args.experimento == FINAL else "confianza-otro.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    t = resultado["umbral_publicado"]
    print(f"{args.experimento}: {len(filas)} consultas, {len(malas)} con lista mala (menos de {MINIMO_DE_OBRAS_BUENAS} obras de grado 2)")
    print(f"AUC del puntaje medio: {resultado['auc_confianza_para_separar_buenas_de_malas']}")
    print(
        f"Umbral {UMBRAL_DE_CONFIANZA}: avisa {t['avisa']}, acierta {t['listas_malas_avisadas']}, "
        f"avisa de más {t['avisos_sobre_listas_buenas']}, no avisa y era mala {t['listas_malas_sin_aviso']}, "
        f"precisión {t['precision']}, cobertura {t['cobertura']}"
    )
    print(
        f"Exactitud dejando una consulta afuera: {resultado['exactitud_dejando_una_consulta_afuera']} "
        f"(umbrales elegidos: {resultado['umbrales_elegidos_dejando_una_consulta_afuera']})"
    )
    for cat, fila in por_categoria.items():
        print(f"  {cat}: avisa {fila['avisa']} de {fila['consultas']} (listas malas: {fila['listas_malas']})")
    print(f"Escrito en {destino.relative_to(c.RAIZ)}")


if __name__ == "__main__":
    main()
