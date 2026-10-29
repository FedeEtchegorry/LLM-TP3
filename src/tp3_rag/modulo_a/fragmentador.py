from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tp3_rag.dominio.catalogo import CampoDeTexto, RegistroCanónico
from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDeConstrucción

VERSIÓN_DEL_FRAGMENTADOR = "1"

def medidor_de_caracteres(_tokenizador: str, _caché: Path | None) -> Callable[[str], int]:
    return len

def medidor_de_tokens(tokenizador: str, caché: Path | None) -> Callable[[str], int]:
    if caché is not None:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(caché))
        Path(caché).mkdir(parents=True, exist_ok=True)
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - está en el grupo `modelos`
        raise ErrorDeConfiguración(
            "`fragmentacion.unidad = \"tokens\"` necesita `tiktoken`, que viene con el "
            "grupo `modelos`. La alternativa declarada es `caracteres`."
        ) from None
    try:
        codificador = tiktoken.get_encoding(tokenizador)
    except Exception as e:
        raise ErrorDeConfiguración(
            f"`fragmentacion.tokenizador = {tokenizador!r}` no se pudo cargar ({e}). Si la "
            f"máquina está sin red y el BPE no está en `rutas.tokenizadores`, la unidad "
            f"declarada `caracteres` no necesita ninguno."
        ) from None
    return lambda texto: len(codificador.encode(texto, disallowed_special=()))

MEDIDORES: dict[str, Callable[[str, Path | None], Callable[[str], int]]] = {
    "caracteres": medidor_de_caracteres,
    "tokens": medidor_de_tokens,
}

ESTRATEGIAS = ("por_campo_semantico",)

@dataclass(frozen=True)
class EstrategiaDeChunking:
    nombre: str
    tamano_objetivo: int
    solapamiento: int
    unidad: str
    tokenizador: str
    separadores: tuple[str, ...]
    version_del_fragmentador: str = VERSIÓN_DEL_FRAGMENTADOR

    @classmethod
    def desde_configuración(cls, cfg) -> "EstrategiaDeChunking":
        estrategia = cls(
            nombre=cfg["fragmentacion.estrategia"],
            tamano_objetivo=int(cfg["fragmentacion.tamano_objetivo"]),
            solapamiento=int(cfg["fragmentacion.solapamiento"]),
            unidad=cfg["fragmentacion.unidad"],
            tokenizador=cfg["fragmentacion.tokenizador"],
            separadores=tuple(cfg["fragmentacion.separadores"]),
        )
        estrategia.validar()
        return estrategia

    def validar(self) -> None:
        if self.nombre not in ESTRATEGIAS:
            raise ErrorDeConfiguración(
                f"`fragmentacion.estrategia` = {self.nombre!r} no existe. Hay "
                f"{list(ESTRATEGIAS)}. ⚠️ El brazo **ciego** de la ablación secundaria "
                f" todavía no está: concatenar campos produce chunks "
                f"que no son subcadena de ningún campo (la subcadena literal) y que no tienen de dónde "
                f"copiar `license` (la licencia copiada del campo de origen) — `description` de AIC es CC-BY-4.0 y el resto "
                f"CC0. Es una decisión abierta, no una implementación pendiente."
            )
        if self.unidad not in MEDIDORES:
            raise ErrorDeConfiguración(
                f"`fragmentacion.unidad` = {self.unidad!r} no existe. Hay "
                f"{sorted(MEDIDORES)}."
            )
        if self.tamano_objetivo <= 0:
            raise ErrorDeConfiguración("`fragmentacion.tamano_objetivo` tiene que ser > 0")
        if not 0 <= self.solapamiento < self.tamano_objetivo:
            raise ErrorDeConfiguración(
                f"`fragmentacion.solapamiento` = {self.solapamiento} tiene que estar en "
                f"[0, tamano_objetivo). Un solapamiento ≥ el tamaño no termina de cortar."
            )
        if not self.separadores or self.separadores[-1] != "":
            raise ErrorDeConfiguración(
                "`fragmentacion.separadores` tiene que terminar en `\"\"`: es el último "
                "recurso del splitter recursivo. Sin él, un campo sin ningún separador "
                "—una lista de `provenance` en una línea— **no se puede subdividir** y el "
                "fragmento se pasa del tamaño objetivo en silencio."
            )

    def a_json(self) -> dict:
        return {
            "nombre": self.nombre,
            "tamano_objetivo": self.tamano_objetivo,
            "solapamiento": self.solapamiento,
            "unidad": self.unidad,
            "tokenizador": self.tokenizador,
            "separadores": list(self.separadores),
            "version_del_fragmentador": self.version_del_fragmentador,
        }

@dataclass
class ConteoDeSección:
    n_obras_con_el_campo: int = 0
    n_fragmentos: int = 0
    n_campos_subdivididos: int = 0
    unidades_totales: int = 0
    unidades_del_mayor: int = 0

    def a_json(self) -> dict:
        return {
            "n_obras_con_el_campo": self.n_obras_con_el_campo,
            "n_fragmentos": self.n_fragmentos,
            "n_campos_subdivididos": self.n_campos_subdivididos,
            "pct_subdivididos": (
                round(100 * self.n_campos_subdivididos / self.n_obras_con_el_campo, 1)
                if self.n_obras_con_el_campo
                else 0.0
            ),
            "unidades_promedio": (
                round(self.unidades_totales / self.n_obras_con_el_campo, 1)
                if self.n_obras_con_el_campo
                else 0.0
            ),
            "unidades_del_mayor": self.unidades_del_mayor,
            "fragmentos_por_obra": (
                round(self.n_fragmentos / self.n_obras_con_el_campo, 2)
                if self.n_obras_con_el_campo
                else 0.0
            ),
        }

class Fragmentador:
    def __init__(self, cfg, *, medidor: Callable[[str], int] | None = None) -> None:
        self.estrategia = EstrategiaDeChunking.desde_configuración(cfg)
        self.campos_de_knowledge = tuple(cfg["fragmentacion.campos_de_knowledge"])
        if not self.campos_de_knowledge:
            raise ErrorDeConfiguración(
                "`fragmentacion.campos_de_knowledge` está vacío: `knowledge` no tendría "
                "ni un fragmento y el invariante 3 de los invariantes de la registración rechazaría la variante."
            )
        self.medidor = medidor or MEDIDORES[self.estrategia.unidad](
            self.estrategia.tokenizador, _caché_de_tokenizadores(cfg)
        )
        self._splitter = None
        self.secciones: dict[str, ConteoDeSección] = {}

    def _subdividir(self, texto: str) -> list[str]:
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            self._splitter = RecursiveCharacterTextSplitter(
                separators=list(self.estrategia.separadores),
                chunk_size=self.estrategia.tamano_objetivo,
                chunk_overlap=self.estrategia.solapamiento,
                length_function=self.medidor,
                is_separator_regex=False,
                keep_separator=False,
                strip_whitespace=True,
            )
        return [t for t in self._splitter.split_text(texto) if t.strip()]

    def fragmentar(self, registro: RegistroCanónico) -> list[Fragmento]:
        fragmentos: list[Fragmento] = []
        for section in self.campos_de_knowledge:
            campo = registro.seccion(section)
            if campo is None or not campo.texto.strip():
                continue
            fragmentos.extend(self._fragmentar_campo(registro, campo))
        return fragmentos

    def _fragmentar_campo(
        self, registro: RegistroCanónico, campo: CampoDeTexto
    ) -> list[Fragmento]:
        conteo = self.secciones.setdefault(campo.section, ConteoDeSección())
        conteo.n_obras_con_el_campo += 1
        unidades = self.medidor(campo.texto)
        conteo.unidades_totales += unidades
        conteo.unidades_del_mayor = max(conteo.unidades_del_mayor, unidades)

        if unidades <= self.estrategia.tamano_objetivo:
            trozos = [campo.texto]
        else:
            trozos = self._subdividir(campo.texto)
            conteo.n_campos_subdivididos += 1
            if len(trozos) < 2:
                raise ErrorDeConstrucción(
                    f"{registro.artwork_id}::{campo.section}: el campo mide {unidades} "
                    f"{self.estrategia.unidad} —más que el objetivo de "
                    f"{self.estrategia.tamano_objetivo}— y el splitter devolvió "
                    f"{len(trozos)} trozo(s). Un fragmento más grande que el objetivo es "
                    f"el camino al `400` por input largo del `PuertoEmbeddingTexto`, que "
                    f"el contrato de los puertos declara **error del fragmentador**."
                )

        fragmentos = [
            Fragmento.desde_campo(registro, campo, texto=trozo, chunk_index=i)
            for i, trozo in enumerate(trozos)
        ]
        conteo.n_fragmentos += len(fragmentos)
        return fragmentos

    def fragmentar_muchos(
        self, registros: Sequence[RegistroCanónico]
    ) -> dict[str, list[Fragmento]]:
        self.secciones = {}
        return {r.artwork_id: self.fragmentar(r) for r in registros}

    def a_json(self) -> dict:
        return {
            "estrategia": self.estrategia.a_json(),
            "campos_de_knowledge": list(self.campos_de_knowledge),
            "nota_de_p2": (
                "la ablación de campos de la colección de texto sigue **abierto**: el conjunto de campos es un punto de "
                "variación, no un valor. El default son los campos marcados como aptos "
                "van a knowledge`; `exhibitions` y `citations` son el eje de la ablación "
                "secundaria y entran agregándolos a la lista."
            ),
            "nota_de_v1": (
                "Chunking **por campo semántico**: desvío / el registro de decisiones, **no "
                "consultado**. La entrega aprobada dice *fragmentos de "
                "unos 500 tokens con solapamiento*; está mitigado porque el chunking "
                "ciego es un brazo de la ablación, así que se decide con números."
            ),
        }

    def conteos_por_seccion(self) -> dict:
        return {s: c.a_json() for s, c in sorted(self.secciones.items())}

def _caché_de_tokenizadores(cfg) -> Path | None:
    ruta = cfg.get("rutas.tokenizadores")
    return Path(ruta) if ruta else None

def informe_de_secciones(
    cfg,
    registros: Sequence[RegistroCanónico],
    *,
    medidor: Callable[[str], int] | None = None,
) -> dict:
    configuradas = tuple(cfg["fragmentacion.campos_de_knowledge"])
    todas = sorted({c.section for r in registros for c in r.campos_de_texto} | set(configuradas))

    def medir(subconjunto: Sequence[RegistroCanónico]) -> dict:
        frag = Fragmentador(_ConfiguraciónConCampos(cfg, todas), medidor=medidor)
        for r in subconjunto:
            frag.fragmentar(r)
        return frag.conteos_por_seccion()

    return {
        "unidad": cfg["fragmentacion.unidad"],
        "tamano_objetivo": int(cfg["fragmentacion.tamano_objetivo"]),
        "secciones_configuradas": list(configuradas),
        "secciones_medidas": list(todas),
        "candidatas_de_p2": [s for s in todas if s not in configuradas],
        "n_obras": len(registros),
        "total": medir(registros),
        "por_museo": {
            museo: {
                "n_obras": sum(1 for r in registros if r.museum == museo),
                "secciones": medir([r for r in registros if r.museum == museo]),
            }
            for museo in sorted({r.museum for r in registros})
        },
        "nota": (
            "Medido sobre **todas** las secciones del catálogo, no solo las que la "
            "variante indexa: es el insumo de la ablación de campos de la colección de texto, que sigue abierto. `n "
            "fragmentos` de la variante es otro número y vive en el `RegistroDeÍndices`."
        ),
    }

class _ConfiguraciónConCampos:
    def __init__(self, cfg, campos: Sequence[str]) -> None:
        self._cfg = cfg
        self._campos = list(campos)

    def __getitem__(self, ruta: str):
        if ruta == "fragmentacion.campos_de_knowledge":
            return self._campos
        return self._cfg[ruta]

    def get(self, ruta: str, default=None):
        if ruta == "fragmentacion.campos_de_knowledge":
            return self._campos
        return self._cfg.get(ruta, default)
