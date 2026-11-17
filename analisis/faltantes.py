"""Consultas del set de 30 que todavía no tienen una corrida válida de una configuración. No gasta.

    python analisis/faltantes.py "C4·filtro100·metadata·rerank20·gpt-6.1-terra"
    python analisis/faltantes.py "C4.filtro100.metadata.rerank20.gpt-6.1-terra"   # «.» en vez de «·»

Imprime un consulta_id por línea. Una corrida es válida si tiene top-5 y el reranker puntuó todas
las candidatas: una corrida cortada por la key o por falta de saldo no cuenta y se vuelve a pedir.
"""
from __future__ import annotations

import glob
import json
import re
import sys

import cascada as c


def main() -> None:
    etiqueta = sys.argv[1]
    todas = re.findall(r'consulta_id\s*=\s*"([^"]+)"', (c.RAIZ / "consultas-g2.toml").read_text(encoding="utf-8"))
    hechas = set()
    for archivo in glob.glob(str(c.RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        estado, cfg = corrida["estado"], corrida["configuracion_resuelta"]
        if (cfg.get("fusion") or {}).get("estrategia") != "cascada":
            continue
        # el argumento es una expresión regular: «.» cubre el «·» cuando la consola no lo pasa bien
        if not re.fullmatch(etiqueta, c.etiqueta(cfg)) or not estado.get("top5"):
            continue
        if any(r.get("estado") == "sin_puntaje" for r in (estado.get("rerank") or [])):
            continue
        hechas.add(corrida["consulta_id"])
    for consulta in todas:
        if consulta not in hechas:
            print(consulta)


if __name__ == "__main__":
    main()
