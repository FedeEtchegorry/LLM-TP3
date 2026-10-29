from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.nucleo.errores import ErrorDeConfiguración

CONDICIONES: dict[str, Callable[[RegistroCanónico], bool]] = {
    "description_no_vacia": lambda r: r.tiene_description_no_vacia,
    "imagen_real": lambda r: r.tiene_imagen_real,
}

@dataclass(frozen=True)
class Elegibles:
    criterio: dict
    registros: tuple[RegistroCanónico, ...]
    n_universo: int
    descartadas_por_condicion: dict

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.artwork_id for r in self.registros)

    def a_json(self) -> dict:
        return {
            "criterio_de_inclusion": self.criterio,
            "n_universo_normalizado": self.n_universo,
            "n_elegibles_con_texto_e_imagen": len(self.registros),
            "descartadas_por_condicion": self.descartadas_por_condicion,
            "nota": (
                "`n_elegibles_con_texto_e_imagen` es el conjunto **pre-split**: el "
                "held-out todavía está adentro. El corpus indexado sale del "
                "`ManifiestoDeSplit`, que es su dueño."
            ),
        }

class SelectorDeCorpus:
    def __init__(self, criterio: Mapping[str, bool]) -> None:
        desconocidas = sorted(set(criterio) - set(CONDICIONES))
        if desconocidas:
            raise ErrorDeConfiguración(
                f"`ingesta.criterio_de_inclusion` tiene condiciones que no existen: "
                f"{desconocidas}. Disponibles: {sorted(CONDICIONES)}. Una condición mal "
                f"escrita no puede leerse como `false`: sería un criterio más amplio en "
                f"silencio, y el criterio es lo que entra en el `split_id`."
            )
        self.criterio = {k: bool(v) for k, v in sorted(criterio.items())}
        self._exigidas = tuple(k for k, v in self.criterio.items() if v)

    def incluye(self, registro: RegistroCanónico) -> bool:
        return all(CONDICIONES[nombre](registro) for nombre in self._exigidas)

    def seleccionar(self, registros: Sequence[RegistroCanónico]) -> Elegibles:
        elegidos, descartes = [], {nombre: 0 for nombre in self._exigidas}
        for r in registros:
            fallan = [n for n in self._exigidas if not CONDICIONES[n](r)]
            if fallan:
                for n in fallan:
                    descartes[n] += 1
            else:
                elegidos.append(r)
        elegidos.sort(key=lambda r: r.artwork_id)
        return Elegibles(
            criterio=dict(self.criterio),
            registros=tuple(elegidos),
            n_universo=len(registros),
            descartadas_por_condicion=dict(sorted(descartes.items())),
        )
