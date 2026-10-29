from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tp3_rag.nucleo.errores import ErrorDeArtefacto

def prefijar(museo: str, id_del_museo) -> str:
    if not museo:
        raise ValueError("el museo no puede ser vacío: el prefijo del prefijo de museo en el id es obligatorio")
    return f"{museo}:{id_del_museo}"

def museo_de(artwork_id: str) -> str:
    return artwork_id.split(":", 1)[0]

@dataclass(frozen=True)
class CampoDeTexto:
    section: str
    texto: str
    source_url: str
    license: str

    def a_json(self) -> dict:
        return {
            "section": self.section,
            "texto": self.texto,
            "source_url": self.source_url,
            "license": self.license,
        }

@dataclass(frozen=True)
class Imagen:
    image_url: str
    hash_bytes: str

    def a_json(self) -> dict:
        return {"image_url": self.image_url, "hash_bytes": self.hash_bytes}

@dataclass(frozen=True)
class RegistroCanónico:
    artwork_id: str
    museum: str
    type: str
    title: str | None
    artist: str | None
    date_display: str | None
    year: int | None
    medium: str | None
    style: str | None
    culture_raw: tuple[str, ...] | None
    source_url: str
    imagen: Imagen | None
    media_license: str
    campos_de_texto: tuple[CampoDeTexto, ...]
    procedencia: dict

    CAMPOS_DE_A9A = ("artist", "year", "medium", "culture", "source_url", "media_license")

    def a_json(self) -> dict:
        salida: dict = {
            "artwork_id": self.artwork_id,
            "museum": self.museum,
            "type": self.type,
            "source_url": self.source_url,
            "media_license": self.media_license,
            "campos_de_texto": [c.a_json() for c in self.campos_de_texto],
            "procedencia": self.procedencia,
        }
        for clave in ("title", "artist", "date_display", "year", "medium", "style"):
            valor = getattr(self, clave)
            if valor is not None:
                salida[clave] = valor
        if self.culture_raw is not None:
            salida["culture_raw"] = list(self.culture_raw)
        if self.imagen is not None:
            salida["imagen"] = self.imagen.a_json()
        return salida

    @classmethod
    def desde_json(cls, d: dict) -> "RegistroCanónico":
        img = d.get("imagen")
        return cls(
            artwork_id=d["artwork_id"],
            museum=d["museum"],
            type=d["type"],
            title=d.get("title"),
            artist=d.get("artist"),
            date_display=d.get("date_display"),
            year=d.get("year"),
            medium=d.get("medium"),
            style=d.get("style"),
            culture_raw=tuple(d["culture_raw"]) if "culture_raw" in d else None,
            source_url=d["source_url"],
            imagen=Imagen(img["image_url"], img["hash_bytes"]) if img else None,
            media_license=d["media_license"],
            campos_de_texto=tuple(CampoDeTexto(**c) for c in d.get("campos_de_texto", [])),
            procedencia=d.get("procedencia", {}),
        )

    def campo_de_a9a_resuelto(self, campo: str) -> bool:
        if campo == "culture":
            return bool(self.culture_raw)
        return getattr(self, campo) is not None

    @property
    def tiene_imagen_real(self) -> bool:
        return self.imagen is not None

    @property
    def tiene_description_no_vacia(self) -> bool:
        return any(c.section == "description" and c.texto.strip() for c in self.campos_de_texto)

    def seccion(self, section: str) -> CampoDeTexto | None:
        for campo in self.campos_de_texto:
            if campo.section == section:
                return campo
        return None

@dataclass(frozen=True)
class CatálogoCanónico:
    catalogo_id: str
    registros: tuple[RegistroCanónico, ...]
    ruta: Path

    @classmethod
    def cargar(cls, almacén, base: str | Path, *, verificar: bool = True) -> "CatálogoCanónico":
        from tp3_rag.nucleo.identidad import hash_de_registros

        base = Path(base)
        declarado = almacén.leer_json(
            base / "catalogo.json",
            claves_obligatorias=("catalogo_id", "hash_de_registros", "n_registros"),
        )
        filas = list(almacén.leer_jsonl(base / "registros.jsonl"))

        if verificar:
            if len(filas) != declarado["n_registros"]:
                raise ErrorDeArtefacto(
                    f"{base}: `catalogo.json` declara {declarado['n_registros']} registros y "
                    f"`registros.jsonl` tiene {len(filas)}. El `catalogo_id` se "
                    f"deriva del contenido: mismo id, mismo contenido."
                )
            medido = hash_de_registros(filas)
            if medido != declarado["hash_de_registros"]:
                raise ErrorDeArtefacto(
                    f"{base}: el hash de los registros no coincide con el declarado en "
                    f"`catalogo.json` (declarado {declarado['hash_de_registros'][:16]}…, "
                    f"medido {medido[:16]}…). El contenido cambió sin que cambiara el "
                    f"`catalogo_id`, que es error duro: re-ingestá en vez de editar."
                )

        return cls(
            catalogo_id=declarado["catalogo_id"],
            registros=tuple(RegistroCanónico.desde_json(f) for f in filas),
            ruta=almacén.ruta(base),
        )

    @staticmethod
    def único(almacén, raíz_de_catálogos: str | Path) -> Path:
        raíz = almacén.ruta(raíz_de_catálogos)
        candidatos = sorted(d.name for d in raíz.iterdir() if (d / "catalogo.json").exists()) if raíz.exists() else []
        if not candidatos:
            raise ErrorDeArtefacto(
                f"no hay ningún catálogo en {raíz}. Corré `poetry run tp3-ingesta` primero."
            )
        if len(candidatos) > 1:
            raise ErrorDeArtefacto(
                f"hay {len(candidatos)} catálogos en {raíz}: {candidatos}. "
                f"Pasá `--catalogo-id` para decir cuál: con más de uno, elegir por fecha "
                f"de modificación sería tomar una decisión de alcance por `mtime`."
            )
        return Path(raíz_de_catálogos) / candidatos[0]
