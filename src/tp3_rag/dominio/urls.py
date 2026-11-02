from __future__ import annotations

from typing import Any, Mapping

def _plantilla(plantillas: Mapping[str, Any], museo: str, cual: str) -> str | None:
    return (plantillas.get(museo) or {}).get(cual) or None

def url_de_imagen(museo: str, crudo: Mapping, plantillas: Mapping[str, Any]) -> str | None:
    conf = plantillas.get(museo) or {}
    campo = conf.get("campo_de_imagen")
    plantilla = conf.get("imagen")
    if not campo:
        return None

    valor: Any = crudo
    for parte in campo.split("."):
        if not isinstance(valor, Mapping):
            return None
        valor = valor.get(parte)
    if not valor:
        return None
    if not plantilla:
        return str(valor)
    return plantilla.format(**{campo.split(".")[-1]: valor})

def url_de_ficha(museo: str, crudo: Mapping, plantillas: Mapping[str, Any]) -> str | None:
    conf = plantillas.get(museo) or {}
    campo = conf.get("campo_de_ficha")
    plantilla = _plantilla(plantillas, museo, "ficha")
    valor = crudo.get(campo) if campo else None
    if not valor:
        return None
    if not plantilla:
        return str(valor)
    return plantilla.format(**{campo: valor})
