from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

@dataclass(frozen=True)
class CoberturaDePaginación:
    museo: str
    n_total_declarada: int
    n_alcanzadas: int
    n_no_alcanzadas: int
    truncado_por_limite: bool = False
    nota: str | None = None

    def a_json(self) -> dict:
        d = {
            "museo": self.museo,
            "n_total_declarada": self.n_total_declarada,
            "n_alcanzadas": self.n_alcanzadas,
            "n_no_alcanzadas": self.n_no_alcanzadas,
            "truncado_por_limite": self.truncado_por_limite,
        }
        if self.nota:
            d["nota"] = self.nota
        return d

class PuertoCatálogoDeMuseo(Protocol):
    museo: str
    version_del_adaptador: str

    def listar(
        self, tipos: Sequence[str], licencia: str, límite: int | None = None
    ) -> list: ...
    def traer(self, id_del_museo) -> dict: ...
    def traer_imagen(self, id_del_museo) -> bytes: ...
    def cobertura_de_paginación(self) -> CoberturaDePaginación: ...
