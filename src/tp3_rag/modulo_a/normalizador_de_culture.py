from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

_PARENTÉTICO = re.compile(r"\s*\([^()]*\)\s*")

def _crudo(valores: Sequence[str]) -> list[str]:
    return [v.strip() for v in valores if v.strip()]

def _primer_componente_por_coma(valores: Sequence[str]) -> list[str]:
    salida = []
    for v in valores:
        cabeza = _PARENTÉTICO.sub(" ", v).split(",")[0].strip(" .;")
        if cabeza:
            salida.append(cabeza)
    return salida

def _primer_componente_sin_parentesis_ni_periodo(valores: Sequence[str]) -> list[str]:
    sufijos = ("dynasty", "period", "century", "kingdom", "era")
    salida = []
    for v in _primer_componente_por_coma(valores):
        palabras = v.split()
        while len(palabras) > 2 and palabras[-1].lower().strip(".,") in sufijos:
            palabras = palabras[:-2]
        salida.append(" ".join(palabras) if palabras else v)
    return salida

CRITERIOS: dict[str, Callable[[Sequence[str]], list[str]]] = {
    "crudo": _crudo,
    "primer_componente_por_coma": _primer_componente_por_coma,
    "primer_componente_sin_parentesis_ni_periodo": _primer_componente_sin_parentesis_ni_periodo,
}

class NormalizadorDeCulture:
    def __init__(self, criterio: str) -> None:
        if criterio not in CRITERIOS:
            raise ValueError(f"criterio {criterio!r} desconocido; hay {sorted(CRITERIOS)}")
        self.criterio = criterio
        self._fn = CRITERIOS[criterio]

    def normalizar(self, culture_raw: Iterable[str] | None) -> list[str]:
        if not culture_raw:
            return []
        return list(dict.fromkeys(self._fn(list(culture_raw))))

    def a9b(self, registros) -> dict:
        antes: set[str] = set()
        después: set[str] = set()
        por_museo: dict[str, dict] = {}
        for r in registros:
            a = {v.strip() for v in (r.culture_raw or ()) if v.strip()}
            d = set(self.normalizar(r.culture_raw))
            antes |= a
            después |= d
            m = por_museo.setdefault(r.museum, {"antes": set(), "despues": set(), "n_obras": 0})
            m["antes"] |= a
            m["despues"] |= d
            m["n_obras"] += 1
        return {
            "criterio": self.criterio,
            "nota": (
                "El criterio definitivo sigue abierto: lo implementado es el punto de "
                "variación y lo que se publica es el par número + criterio, no el valor."
            ),
            "total": {
                "distintos_antes": len(antes),
                "distintos_despues": len(después),
                "colapsados": len(antes) - len(después),
            },
            "por_museo": {
                museo: {
                    "n_obras": v["n_obras"],
                    "distintos_antes": len(v["antes"]),
                    "distintos_despues": len(v["despues"]),
                    "colapsados": len(v["antes"]) - len(v["despues"]),
                }
                for museo, v in sorted(por_museo.items())
            },
        }
