from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tp3_rag.nucleo.errores import ErrorDeArtefacto
from tp3_rag.nucleo.identidad import json_canónico

class YaExiste(ErrorDeArtefacto):
    pass

class AlmacenamientoEnArchivos:
    def __init__(self, raíz: Path) -> None:
        self.raíz = Path(raíz)

    def ruta(self, relativa: str | Path) -> Path:
        return self.raíz / relativa

    def escribir_bytes(self, relativa: str | Path, datos: bytes) -> Path:
        destino = self.ruta(relativa)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_name(destino.name + ".parcial")
        with tmp.open("wb") as fh:
            fh.write(datos)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(destino)
        return destino

    def escribir_json(self, relativa: str | Path, artefacto: Any, *, legible: bool = True) -> Path:
        texto = (
            json.dumps(artefacto, ensure_ascii=False, indent=2, sort_keys=True)
            if legible
            else json_canónico(artefacto)
        )
        return self.escribir_bytes(relativa, (texto + "\n").encode("utf-8"))

    def escribir_texto(self, relativa: str | Path, texto: str) -> Path:
        return self.escribir_bytes(relativa, texto.encode("utf-8"))

    def escribir_jsonl(self, relativa: str | Path, filas) -> Path:
        cuerpo = "".join(json_canónico(f) + "\n" for f in filas)
        return self.escribir_bytes(relativa, cuerpo.encode("utf-8"))

    def leer_json(self, relativa: str | Path, claves_obligatorias=()) -> Any:
        destino = self.ruta(relativa)
        if not destino.exists():
            raise ErrorDeArtefacto(f"no existe el artefacto {destino}")
        try:
            artefacto = json.loads(destino.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ErrorDeArtefacto(f"{destino} no es JSON válido: {e}") from None
        faltantes = [k for k in claves_obligatorias if k not in artefacto]
        if faltantes:
            raise ErrorDeArtefacto(
                f"{destino} no valida: le faltan {faltantes}. Un artefacto que no valida "
                f"se rechaza; no se completa con defaults."
            )
        return artefacto

    def leer_texto(self, relativa: str | Path) -> str:
        destino = self.ruta(relativa)
        if not destino.exists():
            raise ErrorDeArtefacto(f"no existe el artefacto {destino}")
        return destino.read_text(encoding="utf-8")

    def leer_jsonl(self, relativa: str | Path):
        destino = self.ruta(relativa)
        if not destino.exists():
            raise ErrorDeArtefacto(f"no existe el artefacto {destino}")
        with destino.open(encoding="utf-8") as fh:
            for n, línea in enumerate(fh, 1):
                línea = línea.strip()
                if not línea:
                    continue
                try:
                    yield json.loads(línea)
                except json.JSONDecodeError as e:
                    raise ErrorDeArtefacto(f"{destino}:{n} no es JSON válido: {e}") from None

    def existe(self, relativa: str | Path) -> bool:
        return self.ruta(relativa).exists()

    def reservar(self, relativa: str | Path) -> Path:
        destino = self.ruta(relativa)
        destino.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(destino, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise YaExiste(str(destino)) from None
        os.close(fd)
        return destino

    def reservar_directorio(self, relativa: str | Path) -> Path:
        destino = self.ruta(relativa)
        destino.parent.mkdir(parents=True, exist_ok=True)
        try:
            destino.mkdir()
        except FileExistsError:
            raise YaExiste(str(destino)) from None
        return destino

    def reservar_ordinal(
        self,
        plantilla: str,
        desde: int = 1,
        tope: int = 10_000,
        *,
        directorio: bool = False,
    ) -> tuple[int, Path]:
        reservar = self.reservar_directorio if directorio else self.reservar
        for ordinal in range(desde, tope):
            try:
                return ordinal, reservar(plantilla.format(ordinal=ordinal))
            except YaExiste:
                continue
        raise ErrorDeArtefacto(f"no se pudo reservar un ordinal para {plantilla} (tope {tope})")
