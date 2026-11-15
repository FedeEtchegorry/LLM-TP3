"""Confianza de una respuesta: el promedio del reranker sobre el top-5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

UMBRAL_DE_CONFIANZA = 7.0


@dataclass(frozen=True)
class Confianza:
    puntaje_medio: float
    n_obras: int

    @property
    def baja(self) -> bool:
        return self.puntaje_medio < UMBRAL_DE_CONFIANZA


def confianza_del_top5(rerank: Sequence[Mapping[str, Any]] | None, top5: Sequence[str]) -> Confianza | None:
    """`None` si falta el top-5 o el puntaje de alguna obra."""
    puntajes = {f["artwork_id"]: f.get("score_0_10") for f in rerank or ()}
    if not top5 or any(puntajes.get(a) is None for a in top5):
        return None
    return Confianza(sum(puntajes[a] for a in top5) / len(top5), len(top5))
