"""Datos y exportación de los juicios manuales de la UI."""
from __future__ import annotations

import csv
import glob
import json
import random
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

RAIZ = Path(__file__).resolve().parent.parent


def cargar():
    datos = tomllib.loads((RAIZ / "consultas-g2.toml").read_text(encoding="utf-8"))
    consultas = {c["consulta_id"]: {**c, "imagen": imagen["artwork_id"]}
                 for imagen in datos["imagenes"] for c in imagen["consultas"]}
    fichas, pares = {}, set()
    for archivo in glob.glob(str(RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        estado = corrida["estado"]
        for obra in ((estado.get("salida") or {}).get("obras") or []):
            fichas.setdefault(obra["artwork_id"], {**(obra.get("ficha") or {}), "source_url": obra.get("source_url")})
        consulta = corrida.get("consulta_id")
        if consulta in consultas and corrida.get("agrega_a_la_tabla", True) and not estado.get("fallida"):
            obras = estado.get("top5") or [o["artwork_id"] for o in ((estado.get("salida") or {}).get("obras") or [])]
            pares.update((consulta, obra) for obra in obras)
    if not pares:
        return [], consultas, fichas, "Todavía no hay obras para evaluar en las corridas del set de consultas."
    pares = sorted(pares)
    random.Random(7369).shuffle(pares)
    pares.sort(key=lambda p: p[0])
    return pares, consultas, fichas, ""


def exportar(pares, estado, anotador):
    filas = [(c, o, estado.get(f"{c}|{o}", {})) for c, o in pares]
    filas = [(c, o, v) for c, o, v in filas if v.get("grado") in (0, 1, 2)]
    if not filas:
        return None
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False) as f:
        escritor = csv.writer(f)
        escritor.writerow(("consulta_id", "artwork_id", "grado", "nota", "anotador"))
        escritor.writerows((c, o, v["grado"], v.get("nota", ""), anotador or "") for c, o, v in filas)
        return f.name
