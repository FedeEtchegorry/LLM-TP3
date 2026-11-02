from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tp3_rag.dominio.catalogo import CampoDeTexto, RegistroCanónico
from tp3_rag.nucleo.errores import ErrorDeConstrucción

SEPARADOR_DE_ID = "::"

ORDEN_DE_RESTRICCIÓN: tuple[str, ...] = ("CC0", "CC-BY-4.0")

def licencia_más_restrictiva(licencias: Sequence[str]) -> str:
    desconocidas = sorted(set(licencias) - set(ORDEN_DE_RESTRICCIÓN))
    if desconocidas:
        raise ErrorDeConstrucción(
            f"las licencias {desconocidas} no están en el orden de restricción declarado "
            f"{list(ORDEN_DE_RESTRICCIÓN)}, así que no se sabe cuál de la unión es la más "
            f"restrictiva. **No hay default**: elegir la primera podría poner `CC0` como "
            f"`license` de un chunk que contiene texto `CC-BY`, y eso publica una cita sin "
            f"atribuir (los modos de falla del proyecto). Es la regla de la licencia declarada y no asumida aplicada a la unión."
        )
    return max(licencias, key=ORDEN_DE_RESTRICCIÓN.index)

@dataclass(frozen=True)
class Fragmento:
    artwork_id: str
    museum: str
    section: str
    chunk_index: int
    texto: str
    source_url: str
    license: str
    title: str | None
    artist: str | None

    campos: tuple[str, ...] | None = None
    licencias: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.campos is not None and self.section not in self.campos:
            raise ErrorDeConstrucción(
                f"{self.fragment_id}: `section = {self.section!r}` no está entre los "
                f"`campos` que el fragmento declara abarcar ({list(self.campos)}). El "
                f"singular tiene que ser uno de la lista: es la mitad que leen el criterio de atribución y la "
                f"ficha."
            )
        if self.licencias is not None and self.license not in self.licencias:
            raise ErrorDeConstrucción(
                f"{self.fragment_id}: `license = {self.license!r}` no está en la unión "
                f"{list(self.licencias)}. El singular es **el más restrictivo de la "
                f"unión**, nunca uno de afuera: si no, un lector del singular podría "
                f"publicar una cita sin atribuir (los modos de falla del proyecto)."
            )

    @property
    def fragment_id(self) -> str:
        return f"{self.artwork_id}{SEPARADOR_DE_ID}{self.section}{SEPARADOR_DE_ID}{self.chunk_index}"

    @property
    def campos_abarcados(self) -> tuple[str, ...]:
        return self.campos or (self.section,)

    @property
    def licencias_abarcadas(self) -> tuple[str, ...]:
        return self.licencias or (self.license,)

    @property
    def mezcla_licencias(self) -> bool:
        return len(set(self.licencias_abarcadas)) > 1

    @property
    def combinacion_de_licencias(self) -> str:
        return " + ".join(sorted(set(self.licencias_abarcadas)))

    def clave_de_valor(self) -> tuple:
        return (
            self.artwork_id,
            self.section,
            self.chunk_index,
            self.texto,
            self.source_url,
            self.license,
            self.campos_abarcados,
            self.licencias_abarcadas,
        )

    def a_metadata(self) -> dict:
        metadata = {
            "artwork_id": self.artwork_id,
            "museum": self.museum,
            "section": self.section,
            "title": self.title,
            "artist": self.artist,
            "source_url": self.source_url,
            "license": self.license,
            "chunk_index": self.chunk_index,
        }
        if len(self.campos_abarcados) > 1 or len(set(self.licencias_abarcadas)) > 1:
            metadata["campos"] = list(self.campos_abarcados)
            metadata["licencias"] = list(self.licencias_abarcadas)
        return metadata

    def a_json(self) -> dict:
        return {"fragment_id": self.fragment_id, "texto": self.texto, **self.a_metadata()}

    @classmethod
    def desde_campo(
        cls,
        registro: RegistroCanónico,
        campo: CampoDeTexto,
        *,
        texto: str,
        chunk_index: int,
    ) -> "Fragmento":
        if not texto:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id}::{campo.section}: un fragmento vacío no es un "
                f"fragmento. Si el campo quedó vacío al renderizar, está ausente (la ausencia tratada como dato)."
            )
        if texto not in campo.texto:
            raise ErrorDeConstrucción(
                f"la verificación de subcadena roto en {registro.artwork_id}::{campo.section}::{chunk_index}: el "
                f"texto del fragmento **no es subcadena literal** del `CampoDeTexto` "
                f"normalizado. El chunking corta, nunca reescribe (la verificación de subcadena, "
                f"que es literal). Fragmento: {texto[:80]!r}"
            )
        return cls(
            artwork_id=registro.artwork_id,
            museum=registro.museum,
            section=campo.section,
            chunk_index=chunk_index,
            texto=texto,
            source_url=campo.source_url,
            license=campo.license,
            title=registro.title,
            artist=registro.artist,
        )

    @classmethod
    def desde_concatenacion(
        cls,
        registro: RegistroCanónico,
        campos: Sequence[CampoDeTexto],
        *,
        texto: str,
        chunk_index: int,
        concatenacion: str,
    ) -> "Fragmento":
        if not campos:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id}: un fragmento ciego sin ningún campo de origen no "
                f"tiene de dónde copiar `license` ni `source_url` (la regla). Un trozo que no "
                f"solapa ningún campo solo puede salir de un separador, y el splitter los "
                f"descarta: llegar acá es un bug del mapeo de tramos."
            )
        if not texto:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id}: un fragmento vacío no es un fragmento."
            )
        if texto not in concatenacion:
            raise ErrorDeConstrucción(
                f"la verificación de subcadena roto en {registro.artwork_id} (chunk {chunk_index}, brazo **ciego**): "
                f"el texto del fragmento **no es subcadena literal de la concatenación "
                f"declarada**. En este brazo la concatenación es el texto de origen "
                f"y el chunking sigue cortando y nunca "
                f"reescribiendo. Fragmento: {texto[:80]!r}"
            )
        fichas = sorted({c.source_url for c in campos})
        if len(fichas) > 1:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id}: los campos que este chunk abarca declaran "
                f"{len(fichas)} `source_url` distintos ({fichas}). Un chunk que mezcla dos "
                f"fichas no tiene **una** fuente que citar, y la regla prohíbe inventarla "
                f"desde el registro."
            )
        licencias = tuple(dict.fromkeys(c.license for c in campos))
        return cls(
            artwork_id=registro.artwork_id,
            museum=registro.museum,
            section=campos[0].section,
            chunk_index=chunk_index,
            texto=texto,
            source_url=campos[0].source_url,
            license=licencia_más_restrictiva(licencias),
            title=registro.title,
            artist=registro.artist,
            campos=tuple(c.section for c in campos),
            licencias=licencias,
        )
