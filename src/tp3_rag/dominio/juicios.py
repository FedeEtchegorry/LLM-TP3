from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tp3_rag.dominio import ground_truth as gt
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DEL_APORTE = "1"

GRADOS: tuple[int, ...] = (0, 1, 2)

SIGNIFICADO_DE_LA_ESCALA: Mapping[int, str] = {
    2: "Cumple claramente el pedido respecto de la imagen original",
    1: "Cumple parcialmente o es defendible",
    0: "No cumple",
}

COLUMNAS = (
    "split_id",
    "rubrica_id",
    "anotador",
    "consulta_id",
    "artwork_id",
    "grado",
    "nota",
)

@dataclass(frozen=True)
class Juicio:
    consulta_id: str
    artwork_id: str
    grado: int
    nota: str = ""

    @property
    def par(self) -> tuple[str, str]:
        return (self.consulta_id, self.artwork_id)

@dataclass(frozen=True)
class AporteDeAnotador:
    procedencia: dict
    juicios: tuple[Juicio, ...]

    @property
    def identidad(self) -> str:
        return self.procedencia["identidad"]

    @property
    def split_id(self) -> str:
        return self.procedencia["split_id"]

    @property
    def rubrica_id(self) -> str:
        return self.procedencia["rubrica_id"]

    @property
    def anotador(self) -> str:
        return self.procedencia["anotador"]

    @property
    def pool_id(self) -> str:
        return self.procedencia["pool_id"]

    @property
    def hash_de_la_rubrica(self) -> str | None:
        return (self.procedencia.get("rubrica") or {}).get("hash_del_texto")

    @property
    def pares(self) -> frozenset[tuple[str, str]]:
        return frozenset(j.par for j in self.juicios)

    def grado_de(self, par: tuple[str, str]) -> int | None:
        for j in self.juicios:
            if j.par == par:
                return j.grado
        return None

    def por_grado(self) -> dict[str, int]:
        return {f"grado_{g}": sum(1 for j in self.juicios if j.grado == g) for g in GRADOS}

    @staticmethod
    def ruta_de(raíz: str | Path, identidad_del_aporte: str) -> Path:
        return Path(raíz) / f"{identidad_del_aporte}.csv"

    def a_texto(self) -> str:
        return gt.armar_csv(
            procedencia=self.procedencia,
            columnas=COLUMNAS,
            filas=[
                {
                    "split_id": self.split_id,
                    "rubrica_id": self.rubrica_id,
                    "anotador": self.anotador,
                    "consulta_id": j.consulta_id,
                    "artwork_id": j.artwork_id,
                    "grado": j.grado,
                    "nota": j.nota,
                }
                for j in self.juicios
            ],
        )

    @classmethod
    def desde_texto(cls, texto: str, *, ruta=None) -> "AporteDeAnotador":
        procedencia, filas = gt.partir_csv(texto, ruta=ruta)
        aporte = cls(procedencia=procedencia, juicios=tuple(_juicios_de(filas, ruta=ruta)))
        aporte.validar()
        return aporte

    @classmethod
    def leer(cls, almacén, raíz: str | Path, identidad_del_aporte: str) -> "AporteDeAnotador":
        ruta = cls.ruta_de(raíz, identidad_del_aporte)
        aporte = cls.desde_texto(almacén.leer_texto(ruta), ruta=ruta)
        if aporte.identidad != identidad_del_aporte:
            raise ErrorDeArtefacto(
                f"el aporte de {identidad_del_aporte}.csv declara adentro "
                f"`identidad = {aporte.identidad}`. El nombre y el contenido no pueden "
                f"discrepar — y acá la discrepancia es peor que en otros "
                f"artefactos: el nombre junta **tres etiquetas con `-`** y dos de ellas "
                f"pueden contener `-`, así que el contenido es lo único que desambigua de "
                f"qué rúbrica y de qué anotador son estos juicios."
            )
        return aporte

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.csv")))

    @classmethod
    def todos(cls, almacén, raíz: str | Path, *, split_id: str | None = None,
              rubrica_id: str | None = None) -> tuple["AporteDeAnotador", ...]:
        salida = []
        for ident in cls.existentes(almacén, raíz):
            aporte = cls.leer(almacén, raíz, ident)
            if split_id is not None and aporte.split_id != split_id:
                continue
            if rubrica_id is not None and aporte.rubrica_id != rubrica_id:
                continue
            salida.append(aporte)
        return tuple(salida)

    def validar(self) -> None:
        faltantes = [
            k for k in ("identidad", "split_id", "rubrica_id", "anotador", "pool_id")
            if not self.procedencia.get(k)
        ]
        if faltantes:
            raise ErrorDeArtefacto(
                f"el aporte no declara {faltantes} en su preámbulo. Los tres primeros son "
                f"la identidad, y el `pool_id` es la procedencia que hace falta para "
                f"preguntar de qué pool salió cada par."
            )
        esperada = identidad.identidad_del_aporte(
            split_id=self.split_id, rubrica_id=self.rubrica_id, anotador=self.anotador
        )
        if self.identidad != esperada:
            raise ErrorDeArtefacto(
                f"`identidad` dice {self.identidad!r} y sus tres ingredientes dan "
                f"{esperada!r}. En un ancla de trabajo humano la identidad es el "
                f"compromiso: si no cierra, no se sabe con qué rúbrica se juzgó — y el "
                f"acuerdo entre anotadores es siempre sobre una rúbrica."
            )
        if not self.juicios:
            raise ErrorDeArtefacto(
                f"{self.identidad}: el aporte no tiene ni un juicio. Un aporte vacío es un "
                f"artefacto de clase 3 sin el trabajo humano que lo justifica: lo que se "
                f"congela es la anotación, no el archivo."
            )
        pares = [j.par for j in self.juicios]
        repetidos = sorted({p for p in pares if pares.count(p) > 1})
        if repetidos:
            raise ErrorDeArtefacto(
                f"{self.identidad}: los pares {repetidos[:3]} están **juzgados dos veces** "
                f"por el mismo anotador. Un `Juicio` es `(consulta_id, artwork_id) → Grado` "
                f"— dos grados para el mismo par no son dos juicios, "
                f"son uno sin resolver, y κ tomaría el que el orden del archivo deje último."
            )
        for j in self.juicios:
            if j.grado not in GRADOS:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: el par {j.par} tiene `grado = {j.grado!r}` y la "
                    f"escala es {list(GRADOS)}. La escala es compartida; lo que "
                    f"significa cada valor lo fija la rúbrica."
                )
        self._validar_conteos()

    def _validar_conteos(self) -> None:
        conteos = self.procedencia.get("conteos") or {}
        if conteos.get("n_juicios") != len(self.juicios):
            raise ErrorDeArtefacto(
                f"{self.identidad}: `conteos.n_juicios` dice "
                f"{conteos.get('n_juicios')!r} y hay {len(self.juicios)} filas. Un conteo "
                f"que no cierra contra su propio contenido no se completa, se rechaza "
                f"— y acá el conteo es cuántos juicios humanos hay."
            )
        por_grado = conteos.get("por_grado") or {}
        if por_grado and por_grado != self.por_grado():
            raise ErrorDeArtefacto(
                f"{self.identidad}: `conteos.por_grado` dice {dict(por_grado)} y el "
                f"contenido dice {self.por_grado()}."
            )

    def filas_para_mostrar(self) -> Iterable[tuple[str, ...]]:
        for j in self.juicios:
            yield (j.consulta_id, j.artwork_id, str(j.grado), j.nota or "—")

def _juicios_de(filas: Sequence[Mapping[str, Any]], *, ruta=None) -> list[Juicio]:
    salida = []
    for n, fila in enumerate(filas, 1):
        faltan = [c for c in ("consulta_id", "artwork_id", "grado") if not (fila.get(c) or "")]
        if faltan:
            raise ErrorDeArtefacto(
                f"{ruta or 'el CSV'}: la fila {n} no tiene {faltan}. Un juicio es "
                f"`(consulta_id, artwork_id) → Grado`: sin una de las tres no hay juicio."
            )
        salida.append(
            Juicio(
                consulta_id=fila["consulta_id"].strip(),
                artwork_id=fila["artwork_id"].strip(),
                grado=_grado(fila["grado"], fila=n, ruta=ruta),
                nota=(fila.get("nota") or "").strip(),
            )
        )
    return salida

def _grado(valor: Any, *, fila: int, ruta=None) -> int:
    texto = str(valor).strip()
    if texto not in {str(g) for g in GRADOS}:
        raise ErrorDeArtefacto(
            f"{ruta or 'el CSV'}: la fila {fila} tiene `grado = {texto!r}` y los únicos "
            f"valores de la escala son {list(GRADOS)}. No hay default: un grado "
            f"ilegible no se lee como 0 — un 0 es 'no cumple', que es "
            f"un juicio, y lo que hay acá es una celda que nadie llenó bien."
        )
    return int(texto)

def nota_de_identidad() -> str:
    return (
        "Ancla de trabajo humano. La identidad son TRES ingredientes —split_id + "
        "rubrica_id + anotador— y el del medio es la lección más cara del proyecto: sin "
        "él, la ronda piloto y la campaña definitiva comparten nombre de archivo y "
        "re-anotar después de refinar la rúbrica es impagable. El pool_id NO es "
        "ingrediente y va acá adentro: un juicio es sobre el par (consulta, obra), no "
        "sobre el pool que lo trajo, y eso es lo que vuelve legítima la importación de "
        "los juicios de la ronda piloto. No se reescribe y no se recalcula: refinar la "
        "rúbrica deriva OTRO aporte y no borra este."
    )
