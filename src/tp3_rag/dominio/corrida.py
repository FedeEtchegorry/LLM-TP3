from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DE_LA_CORRIDA = "1"

CLAVES_OBLIGATORIAS = (
    "artefacto",
    "version_de_la_corrida",
    "corrida_id",
    "config_id",
    "consulta_id",
    "ordinal",
    "estado",
    "sellos",
    "configuracion_resuelta",
)

ARCHIVO = "corrida.json"

@dataclass(frozen=True)
class Corrida:
    contenido: dict

    @property
    def corrida_id(self) -> str:
        return self.contenido["corrida_id"]

    @property
    def config_id(self) -> str:
        return self.contenido["config_id"]

    @property
    def consulta_id(self) -> str:
        return self.contenido["consulta_id"]

    @property
    def ordinal(self) -> int:
        return int(self.contenido["ordinal"])

    @property
    def estado(self) -> dict:
        return self.contenido["estado"]

    @property
    def cadena(self) -> list[str]:
        return list(self.contenido["pipeline"]["cadena"])

    @property
    def degradaciones(self) -> list[dict]:
        return list(self.estado.get("degradaciones") or [])

    @property
    def degradaciones_no_planificadas(self) -> list[dict]:
        return [d for d in self.degradaciones if not d.get("planificada")]

    @property
    def fallida(self) -> dict | None:
        return self.estado.get("fallida")

    @property
    def top5(self) -> list[str]:
        return list(self.estado.get("top5") or [])

    @property
    def plan(self) -> dict | None:
        return self.estado.get("plan")

    @property
    def plan_id(self) -> str | None:
        return (self.plan or {}).get("plan_id")

    @property
    def plan_reusado(self) -> bool | None:
        plan = self.plan
        return None if plan is None else bool(plan.get("reusado"))

    @property
    def salida(self) -> dict | None:
        return self.estado.get("salida")

    @property
    def afirmaciones_descartadas(self) -> list[dict]:
        return list(self.estado.get("afirmaciones_descartadas") or [])

    @property
    def procedencia(self) -> str | None:
        return (self.estado.get("entrada") or {}).get("procedencia")

    @property
    def es_del_set(self) -> bool:
        return self.procedencia == "set"

    @property
    def agrega_a_la_tabla(self) -> bool:
        return not self.degradaciones and self.fallida is None

    def a_json(self) -> dict:
        return self.contenido

    @staticmethod
    def ruta_de(raíz: str | Path, corrida_id: str) -> Path:
        return Path(raíz) / corrida_id / ARCHIVO

    @classmethod
    def leer(cls, almacén, raíz: str | Path, corrida_id: str) -> "Corrida":
        contenido = almacén.leer_json(
            cls.ruta_de(raíz, corrida_id), claves_obligatorias=CLAVES_OBLIGATORIAS
        )
        corrida = cls(contenido)
        corrida.validar()
        if corrida.corrida_id != corrida_id:
            raise ErrorDeArtefacto(
                f"la corrida de {corrida_id}/ declara adentro "
                f"`corrida_id = {corrida.corrida_id}`. El nombre y el contenido no pueden "
                f"discrepar: la identidad va en el nombre."
            )
        return corrida

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.name for p in carpeta.iterdir() if (p / ARCHIVO).exists()))

    def validar(self) -> None:
        esperado = identidad.corrida_id(
            config_id=self.config_id,
            consulta_id=self.consulta_id,
            commit=self.contenido["sellos"].get("commit"),
            ordinal=self.ordinal,
        )
        if self.corrida_id != esperado:
            raise ErrorDeArtefacto(
                f"`corrida_id` dice {self.corrida_id!r} y sus ingredientes "
                f"(`config_id`, `consulta_id`, commit, ordinal {self.ordinal}) dan "
                f"{esperado!r}. La identidad fija las entradas, así que una que no "
                f"cierra contra ellas no dice de qué ejecución habla."
            )
        self._verificar_et6_al_leer()

    def _verificar_et6_al_leer(self) -> None:
        from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA

        cadena = self.cadena
        pobladas = set(self.contenido["estado"].get("claves_pobladas") or [])
        degradadas = {d["etapa"] for d in self.degradaciones}
        fallida = (self.fallida or {}).get("etapa")
        mudas = [e for e in cadena if e not in pobladas and e not in degradadas and e != fallida]
        if mudas:
            raise ErrorDeArtefacto(
                f"la forma del artefacto {self.corrida_id} no coincide con su cadena: las "
                f"etapas {mudas} están en la cadena declarada y no dejaron ni su clave "
                f"({[CLAVE_DE_ETAPA[e] for e in mudas]}) ni una degradación. Se verifica al "
                f"leer y no solo al escribir: leer artefactos es el único vínculo del "
                f"módulo de evaluación con el pipeline, así que no confía en que el "
                f"ejecutor lo haya verificado."
            )

class AlmacenDeCorridas:
    def __init__(self, almacén, raíz: str | Path) -> None:
        self.almacén = almacén
        self.raíz = Path(raíz)

    def reservar(self, *, config_id: str, consulta_id: str, commit: str | None):
        base = identidad.corrida_id(
            config_id=config_id, consulta_id=consulta_id, commit=commit, ordinal=1
        ).rsplit("-", 1)[0]
        ordinal, directorio = self.almacén.reservar_ordinal(
            str(self.raíz / f"{base}-{{ordinal}}"), directorio=True
        )
        return f"{base}-{ordinal}", ordinal, directorio

    def escribir(self, corrida_id: str, contenido: Mapping[str, Any]) -> Path:
        return self.almacén.escribir_json(Corrida.ruta_de(self.raíz, corrida_id), dict(contenido))

    def leer(self, corrida_id: str) -> Corrida:
        return Corrida.leer(self.almacén, self.raíz, corrida_id)

    def existentes(self) -> tuple[str, ...]:
        return Corrida.existentes(self.almacén, self.raíz)

    def del_config(self, config_id: str) -> tuple[Corrida, ...]:
        salida = []
        for cid in self.existentes():
            try:
                corrida = self.leer(cid)
            except ErrorDeArtefacto:
                continue
            if corrida.config_id == config_id:
                salida.append(corrida)
        return tuple(salida)
