from __future__ import annotations

import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConfiguración

if sys.version_info >= (3, 11):  # pragma: no cover - el lock fija 3.10
    import tomllib
else:
    import tomli as tomllib

RAÍZ = "artifacts/ground-truth"

CONSULTAS = "consultas"
RUBRICAS = "rubricas"
POOL = "pool"
JUICIOS = "juicios"

PLANILLAS = "planillas"

@dataclass(frozen=True)
class RutasDelGroundTruth:
    raiz: Path
    consultas: Path
    rubricas: Path
    pool: Path
    juicios: Path
    planillas: Path

    @property
    def acuerdo(self) -> Path:
        return self.raiz

def rutas(cfg) -> RutasDelGroundTruth:
    declarada = Path(cfg["rutas.ground_truth_consultas"])
    esperada = Path(RAÍZ) / CONSULTAS
    if declarada != esperada:
        raise ErrorDeConfiguración(
            f"`rutas.ground_truth_consultas` = {str(declarada)!r} y la raíz del almacén "
            f"de ground truth es {str(esperada)!r}. Las rutas de la anotación "
            f"—`rubricas/`, `pool/`, `juicios/`— se declaran en código y no en "
            f"`config/default.toml` (ver el docstring de este módulo). Lo "
            f"que las ancla es esta igualdad: si la configuración mueve `consultas/`, el "
            f"ground truth se partiría en dos árboles y nadie avisaría."
        )
    raiz = Path(RAÍZ)
    return RutasDelGroundTruth(
        raiz=raiz,
        consultas=declarada,
        rubricas=raiz / RUBRICAS,
        pool=raiz / POOL,
        juicios=raiz / JUICIOS,
        planillas=Path(PLANILLAS),
    )

MARCA_DE_PREÁMBULO = "#"

def armar_csv(
    *, procedencia: Mapping[str, Any], columnas: Sequence[str], filas: Sequence[Mapping[str, Any]]
) -> str:
    líneas = [f"{MARCA_DE_PREÁMBULO} {l}" if l else MARCA_DE_PREÁMBULO
              for l in _toml_de(procedencia)]
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=list(columnas), lineterminator="\n")
    escritor.writeheader()
    for fila in filas:
        escritor.writerow({c: fila.get(c, "") for c in columnas})
    return "\n".join(líneas) + "\n" + buffer.getvalue()

def partir_csv(texto: str, *, ruta=None) -> tuple[dict, list[dict]]:
    crudas = texto.splitlines()
    preámbulo = [l for l in crudas if l.startswith(MARCA_DE_PREÁMBULO)]
    cuerpo = [l for l in crudas if not l.startswith(MARCA_DE_PREÁMBULO)]
    if not preámbulo:
        raise ErrorDeArtefacto(
            f"{ruta or 'el CSV'} no tiene preámbulo de procedencia. Los `.csv` del ground "
            f"truth llevan su procedencia adentro como TOML comentado —lo que no entra "
            f"en la identidad se declara adentro—: sin él no se sabe de qué "
            f"`pool_id` ni de qué rúbrica salieron estos juicios."
        )
    sin_marca = "\n".join(l[len(MARCA_DE_PREÁMBULO):].lstrip(" ") for l in preámbulo)
    try:
        procedencia = tomllib.loads(sin_marca)
    except Exception as e:
        raise ErrorDeArtefacto(
            f"{ruta or 'el CSV'}: el preámbulo no es TOML válido ({e}). Es TOML comentado "
            f"a propósito; la prosa va con `##`."
        ) from None
    filas = list(csv.DictReader(io.StringIO("\n".join(cuerpo))))
    return procedencia, filas

def _toml_de(valor: Mapping[str, Any], prefijo: str = "") -> list[str]:
    escalares, tablas, líneas = [], [], []
    for clave, v in valor.items():
        (tablas if isinstance(v, Mapping) else escalares).append((clave, v))
    for clave, v in escalares:
        líneas.append(f"{_clave(clave)} = {_literal(v)}")
    for clave, v in tablas:
        ruta = f"{prefijo}{_clave(clave)}"
        líneas += ["", f"[{ruta}]", *_toml_de(v, prefijo=f"{ruta}.")]
    return líneas

def _clave(clave: Any) -> str:
    texto = str(clave)
    if texto and all(c.isalnum() and c.isascii() or c in "_-" for c in texto):
        return texto
    return f'"{texto}"'

def _literal(valor: Any) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, (int, float)):
        return repr(valor)
    if valor is None:
        return '"(ausente)"'
    if isinstance(valor, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in valor) + "]"
    texto = str(valor).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{texto}"'
