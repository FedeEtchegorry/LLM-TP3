"""Acuerdo entre la revisión de explicaciones del asistente y la de una persona. No gasta.

    python analisis/acuerdo_explicaciones.py

Lee `artifacts/analisis/juicios-explicaciones-*.csv` (de planilla_explicaciones.py) y compara con
`claude-c1-5x5-v2.json` (asistente) y `m13b-c1-5x5.json` (revisión anterior, también con IA).
Reporta el acuerdo por pregunta, por separado para las afirmaciones al azar (estimación del acuerdo)
y para las de desacuerdo entre las dos revisiones con IA (desempate).
"""
from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

import planilla_explicaciones as pe

RAIZ = Path(__file__).resolve().parents[1]


def kappa(pares) -> float | None:
    pares = list(pares)
    if not pares:
        return None
    n = len(pares)
    po = sum(a == b for a, b in pares) / n
    ca, cb = collections.Counter(a for a, _ in pares), collections.Counter(b for _, b in pares)
    pe_ = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return round((po - pe_) / (1 - pe_), 2) if pe_ < 1 else 1.0


def main() -> None:
    grupo = {f'{it["obra"]["artwork_id"]}#{it["af"]["n"]}': it["grupo"] for it in pe.elegir()}
    nuevo = json.loads((pe.REV / "claude-c1-5x5-v2.json").read_text(encoding="utf-8"))
    viejo = json.loads((pe.REV / "m13b-c1-5x5.json").read_text(encoding="utf-8"))
    ia = {f'{o["artwork_id"]}#{a["n"]}': a for o in nuevo["obras"] for a in o["afirmaciones"]}
    previa = {f'{o["artwork_id"]}#{a["n"]}': a for o in viejo["obras"] for a in o["afirmaciones"]}
    norm = {"falla_A": "A", "falla_B": "B", "falla_C": "C", "no_sostiene": "no", "falsa_a_la_vista": "falsa", None: "no_visual"}
    archivos = sorted(glob.glob(str(RAIZ / "artifacts" / "analisis" / "juicios-explicaciones-*.csv")))
    if not archivos:
        print("Todavía no hay juicios-explicaciones-*.csv en artifacts/analisis/")
        return
    resumen = {}
    for archivo in archivos:
        filas = [f for f in csv.DictReader(open(archivo, encoding="utf-8")) if f["clasificacion"]]
        por = {}
        for preg, campo_h, campo_ia, campo_prev in (("clasificacion", "clasificacion", "clasificacion", "veredicto_clasificacion"),
                                                     ("cita", "cita", "cita", "veredicto_cita"),
                                                     ("observacion", "observacion", "observacion", "veredicto_observacion")):
            for g in ("azar", "desacuerdo"):
                sel = [f for f in filas if grupo.get(f'{f["artwork_id"]}#{f["n"]}') == g and f[campo_h]]
                h = [f[campo_h] for f in sel]
                c = [ia[f'{f["artwork_id"]}#{f["n"]}'][campo_ia] for f in sel]
                m = [norm.get(previa[f'{f["artwork_id"]}#{f["n"]}'][campo_prev], previa[f'{f["artwork_id"]}#{f["n"]}'][campo_prev]) for f in sel]
                por[f"{preg}_{g}"] = {
                    "n": len(sel),
                    "acuerdo_con_claude": round(sum(x == y for x, y in zip(h, c)) / len(sel), 2) if sel else None,
                    "kappa_con_claude": kappa(zip(h, c)),
                    "acuerdo_con_m13b": round(sum(x == y for x, y in zip(h, m)) / len(sel), 2) if sel else None,
                }
        resumen[Path(archivo).stem.replace("juicios-explicaciones-", "")] = por
    destino = RAIZ / "artifacts" / "analisis" / "acuerdo-explicaciones.json"
    destino.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    for quien, por in resumen.items():
        print(f"\n{quien}")
        for k, v in por.items():
            print(f"  {k:<26} n={v['n']:>2}  acuerdo con Claude {v['acuerdo_con_claude']}  κ {v['kappa_con_claude']}  · con m13b {v['acuerdo_con_m13b']}")
    print(f"\nEscrito en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
