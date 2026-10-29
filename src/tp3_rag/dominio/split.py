from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DEL_MANIFIESTO = "1"

CLAVES_OBLIGATORIAS = (
    "split_id",
    "criterio_de_inclusion",
    "semilla",
    "proporcion_held_out",
    "definicion_de_estrato",
    "conteos",
    "corpus_indexado",
    "held_out",
)

@dataclass(frozen=True)
class ManifiestoDeSplit:
    contenido: dict

    @property
    def split_id(self) -> str:
        return self.contenido["split_id"]

    @property
    def ids_corpus(self) -> tuple[str, ...]:
        return tuple(self.contenido["corpus_indexado"]["ids"])

    @property
    def ids_held_out(self) -> tuple[str, ...]:
        return tuple(self.contenido["held_out"]["ids"])

    @property
    def conteos(self) -> dict:
        return dict(self.contenido["conteos"])

    @property
    def catalogo_id(self) -> str:
        return self.contenido["procedencia"]["catalogo_id"]

    def a_json(self) -> dict:
        return self.contenido

    def es_held_out(self, artwork_id: str) -> bool:
        return artwork_id in set(self.ids_held_out)

    def cobertura_contra(self, ids_del_catalogo) -> tuple[str, ...]:
        return tuple(sorted(set(self.ids_corpus) - set(ids_del_catalogo)))

    @staticmethod
    def ruta_de(raíz_de_splits: str | Path, split_id: str) -> Path:
        return Path(raíz_de_splits) / f"{split_id}.json"

    @classmethod
    def leer(cls, almacén, raíz_de_splits: str | Path, split_id: str) -> "ManifiestoDeSplit":
        contenido = almacén.leer_json(
            cls.ruta_de(raíz_de_splits, split_id), claves_obligatorias=CLAVES_OBLIGATORIAS
        )
        manifiesto = cls(contenido)
        manifiesto.validar()
        if manifiesto.split_id != split_id:
            raise ErrorDeArtefacto(
                f"el manifiesto de {split_id}.json declara adentro "
                f"`split_id = {manifiesto.split_id}`. El nombre y el contenido no pueden "
                f"discrepar: la identidad va en el nombre."
            )
        return manifiesto

    @classmethod
    def existentes(cls, almacén, raíz_de_splits: str | Path) -> tuple[str, ...]:
        raíz = almacén.ruta(raíz_de_splits)
        if not raíz.exists():
            return ()
        return tuple(sorted(p.stem for p in raíz.glob("*.json")))

    def validar(self) -> None:
        corpus, held = set(self.ids_corpus), set(self.ids_held_out)
        c = self.conteos
        if len(corpus) != len(self.ids_corpus) or len(held) != len(self.ids_held_out):
            raise ErrorDeArtefacto(
                f"{self.split_id}: hay ids repetidos en un lado del split."
            )
        solapados = corpus & held
        if solapados:
            raise ErrorDeArtefacto(
                f"{self.split_id}: {len(solapados)} ids están en los dos lados del split "
                f"(p. ej. {sorted(solapados)[:3]}). la exclusión del held-out de toda colección sería falso desde el artefacto."
            )
        esperados = {
            "corpus_indexado": len(corpus),
            "held_out": len(held),
            "elegibles_con_texto_e_imagen": len(corpus) + len(held),
        }
        for clave, esperado in esperados.items():
            if c.get(clave) != esperado:
                raise ErrorDeArtefacto(
                    f"{self.split_id}: `conteos.{clave}` dice {c.get(clave)} y los ids "
                    f"dicen {esperado}. Este artefacto es el dueño de los tres conteos: "
                    f"un conteo que no cierra contra sus propios ids no se completa, se "
                    f"rechaza."
                )

    def distribucion_del_held_out(self) -> dict:
        return dict(self.contenido["distribucion_del_held_out"])

    def estratos(self) -> Mapping[str, Any]:
        return self.contenido["estratos"]
