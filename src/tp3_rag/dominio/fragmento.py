from __future__ import annotations

from dataclasses import dataclass

from tp3_rag.dominio.catalogo import CampoDeTexto, RegistroCanónico
from tp3_rag.nucleo.errores import ErrorDeConstrucción

SEPARADOR_DE_ID = "::"

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

    @property
    def fragment_id(self) -> str:
        return f"{self.artwork_id}{SEPARADOR_DE_ID}{self.section}{SEPARADOR_DE_ID}{self.chunk_index}"

    def clave_de_valor(self) -> tuple:
        return (
            self.artwork_id,
            self.section,
            self.chunk_index,
            self.texto,
            self.source_url,
            self.license,
        )

    def a_metadata(self) -> dict:
        return {
            "artwork_id": self.artwork_id,
            "museum": self.museum,
            "section": self.section,
            "title": self.title,
            "artist": self.artist,
            "source_url": self.source_url,
            "license": self.license,
            "chunk_index": self.chunk_index,
        }

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
                f"en {registro.artwork_id}::{campo.section}::{chunk_index}: el "
                f"texto del fragmento no es subcadena literal del campo de texto "
                f"normalizado. El chunking corta, nunca reescribe. "
                f"Fragmento: {texto[:80]!r}"
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
