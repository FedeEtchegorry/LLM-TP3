"""Gasto medido de las corridas, para que un script desatendido se frene solo.

  python analisis/gasto.py --desde <epoch> [--solo-numero]   # suma de las corridas creadas desde esa hora
  python analisis/gasto.py --ultima [--solo-numero]          # USD de la corrida mas reciente
  python analisis/gasto.py --fallas-ultima [--solo-numero]   # candidatas SIN puntaje en la ultima corrida

Usa los tokens guardados en cada corrida y los precios de su propia configuracion (como cascada.py).
No incluye la llamada del plan cuando sale de cache (~USD 0,001 por consulta) ni el juez.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def usd(corrida: dict) -> float:
    conf = corrida["configuracion_resuelta"]
    prov = conf.get("proveedor") or {}
    entrada = prov.get("precio_usd_por_millon_de_tokens") or {}
    salida = prov.get("precio_usd_por_millon_de_tokens_de_salida") or {}
    bindings = conf.get("bindings") or {}
    total = 0.0
    for rol, libro in (corrida.get("libro_mayor") or {}).items():
        if not isinstance(libro, dict) or "tokens_in" not in libro:
            continue
        modelo = (bindings.get(rol) or {}).get("modelo") or (
            (conf.get("embebido_texto") or {}).get("modelo") if rol == "embedding_texto" else None
        )
        if modelo is None:
            continue
        total += libro.get("tokens_in", 0) * entrada.get(modelo, 0) / 1e6
        total += libro.get("tokens_out", 0) * salida.get(modelo, 0) / 1e6
    return total


def corridas():
    for archivo in glob.glob(str(RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        try:
            with open(archivo, encoding="utf-8") as f:
                yield os.path.getmtime(archivo), archivo, json.load(f)
        except (OSError, ValueError):
            continue


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=float)
    p.add_argument("--ultima", action="store_true")
    p.add_argument("--fallas-ultima", action="store_true")
    p.add_argument("--solo-numero", action="store_true")
    a = p.parse_args()
    if a.fallas_ultima:
        todas = sorted(corridas(), key=lambda t: t[0])
        rerank = (todas[-1][2].get("estado") or {}).get("rerank") or [] if todas else []
        valor = float(sum(1 for r in rerank if r.get("estado") == "sin_puntaje"))
        etiqueta = "candidatas sin puntaje en la ultima corrida"
    elif a.ultima:
        todas = sorted(corridas(), key=lambda t: t[0])
        valor = usd(todas[-1][2]) if todas else 0.0
        etiqueta = "ultima corrida"
    elif a.desde is not None:
        valor = sum(usd(c) for t, _, c in corridas() if t >= a.desde)
        etiqueta = "gasto desde el inicio"
    else:
        p.error("falta --desde o --ultima")
    print(f"{valor:.4f}" if a.solo_numero else f"{etiqueta}: {valor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
