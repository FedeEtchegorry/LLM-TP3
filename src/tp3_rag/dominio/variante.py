from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DEL_REGISTRO = "1"

CLAVES_OBLIGATORIAS = (
    "variante_id",
    "receta",
    "conteos",
    "colecciones",
    "procedencia",
    "invariantes_de_9_6",
)

@dataclass(frozen=True)
class RegistroDeÍndices:
    contenido: dict

    @property
    def variante_id(self) -> str:
        return self.contenido["variante_id"]

    @property
    def receta(self) -> dict:
        return dict(self.contenido["receta"])

    @property
    def conteos(self) -> dict:
        return dict(self.contenido["conteos"])

    @property
    def split_id(self) -> str:
        return self.contenido["receta"]["split_id"]

    @property
    def cita_al_embebido(self) -> dict:
        return dict(self.contenido["procedencia"]["embebido"])

    def a_json(self) -> dict:
        return self.contenido

    @staticmethod
    def ruta_de(raíz: str | Path, variante_id: str) -> Path:
        return Path(raíz) / f"{variante_id}.json"

    @classmethod
    def leer(cls, almacén, raíz: str | Path, variante_id: str) -> "RegistroDeÍndices":
        contenido = almacén.leer_json(
            cls.ruta_de(raíz, variante_id), claves_obligatorias=CLAVES_OBLIGATORIAS
        )
        registro = cls(contenido)
        registro.validar()
        if registro.variante_id != variante_id:
            raise ErrorDeArtefacto(
                f"el registro de {variante_id}.json declara adentro "
                f"`variante_id = {registro.variante_id}`. La identidad va en el nombre."
            )
        return registro

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.json")))

    def validar(self) -> None:
        receta = self.receta
        faltantes = sorted(set(INGREDIENTES_DEL_ID) - set(receta))
        if faltantes:
            raise ErrorDeArtefacto(
                f"{self.variante_id}: la receta no declara {faltantes}, que son "
                f"ingredientes del `variante_id`. La receta tiene que ser un "
                f"superconjunto de los ingredientes: una variante cuya receta no dice "
                f"con qué se construyó es una variante que los informes no pueden "
                f"describir."
            )
        sin_valor = sorted(k for k in INGREDIENTES_DEL_ID if receta.get(k) in (None, "", [], {}))
        if sin_valor:
            raise ErrorDeArtefacto(
                f"{self.variante_id}: los ingredientes {sin_valor} están en la receta "
                f"sin valor, y ninguno puede quedar sin valor."
            )
        c = self.conteos
        for clave in ("n_indexadas", "n_fragmentos"):
            if not isinstance(c.get(clave), int):
                raise ErrorDeArtefacto(
                    f"{self.variante_id}: `conteos.{clave}` tiene que ser un entero y es "
                    f"{c.get(clave)!r}. Este artefacto es su dueño."
                )
        if "n_embebidas" in c:
            raise ErrorDeArtefacto(
                f"{self.variante_id}: `conteos` trae `n_embebidas`, que no es de este "
                f"artefacto: es del universo y la determina otra identidad. "
                f"Se cita el reporte del embebido por `catalogo_id` + ordinal, no se "
                f"copia el número — una cifra con dos copias es una cifra que se "
                f"bifurca."
            )

INGREDIENTES_DEL_ID = (
    "catalogo_id",
    "split_id",
    "criterio_de_inclusion",
    "checkpoint_de_clip",
    "version_del_preprocesado",
    "estrategia_de_chunking",
    "campos_de_knowledge",
    "modelo_de_embedding_de_texto",
    "criterio_de_culture",
)
