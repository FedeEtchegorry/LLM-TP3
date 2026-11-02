from __future__ import annotations

import os
from dataclasses import dataclass, field
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

ESTRATEGIAS = ("por_campo_semantico", "ciego")

CIEGO = "ciego"

SEPARADOR_DE_CONCATENACIÓN = "\n\n"

@dataclass(frozen=True)
class EstrategiaDeChunking:
    nombre: str
    tamano_objetivo: int
    solapamiento: int
    unidad: str
    tokenizador: str
    separadores: tuple[str, ...]
    version_del_fragmentador: str = VERSIÓN_DEL_FRAGMENTADOR
    orden_de_concatenacion: tuple[str, ...] = ()
    separador_de_concatenacion: str = SEPARADOR_DE_CONCATENACIÓN

    @classmethod
    def desde_configuración(cls, cfg) -> "EstrategiaDeChunking":
        nombre = cfg["fragmentacion.estrategia"]
        estrategia = cls(
            nombre=nombre,
            tamano_objetivo=int(cfg["fragmentacion.tamano_objetivo"]),
            solapamiento=int(cfg["fragmentacion.solapamiento"]),
            unidad=cfg["fragmentacion.unidad"],
            tokenizador=cfg["fragmentacion.tokenizador"],
            separadores=tuple(cfg["fragmentacion.separadores"]),
            orden_de_concatenacion=(
                tuple(cfg["fragmentacion.campos_de_knowledge"]) if nombre == CIEGO else ()
            ),
        )
        estrategia.validar()
        return estrategia

    @property
    def es_ciega(self) -> bool:
        return self.nombre == CIEGO

    def validar(self) -> None:
        if self.nombre not in ESTRATEGIAS:
            raise ErrorDeConfiguración(
                f"`fragmentacion.estrategia` = {self.nombre!r} no existe. Hay "
                f"{list(ESTRATEGIAS)}: `por_campo_semantico` es el desvío chunking "
                f"(el registro de decisiones) y `ciego` es el brazo que la entrega aprobada pedía — "
                f"el que vuelve ese desvío mitigado, porque se decide con números."
            )
        if self.es_ciega:
            if not self.orden_de_concatenacion:
                raise ErrorDeConfiguración(
                    "la estrategia `ciego` necesita su `orden_de_concatenacion`, que sale "
                    "de `fragmentacion.campos_de_knowledge` **tal como está declarado**. "
                    "Sin él la concatenación no es determinista, y la verificación de subcadena ancla a ella "
                    "— un orden implícito haría que el mismo "
                    "`variante_id` describiera dos textos distintos."
                )
            if not self.separador_de_concatenacion:
                raise ErrorDeConfiguración(
                    "el `separador_de_concatenacion` del brazo ciego no puede ser vacío: "
                    "dos campos pegados inventan un borde de palabra —*…his home.Sold en "
                    "1923*— y la cita lo publicaría así, que es el criterio de idioma midiendo nuestra "
                    "edición en vez de la ficha."
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
        base = {
            "nombre": self.nombre,
            "tamano_objetivo": self.tamano_objetivo,
            "solapamiento": self.solapamiento,
            "unidad": self.unidad,
            "tokenizador": self.tokenizador,
            "separadores": list(self.separadores),
            "version_del_fragmentador": self.version_del_fragmentador,
        }
        if not self.es_ciega:
            return base
        return {
            **base,
            "orden_de_concatenacion": list(self.orden_de_concatenacion),
            "separador_de_concatenacion": self.separador_de_concatenacion,
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

@dataclass
class ConteoDeConcatenación:
    n_obras: int = 0
    n_fragmentos: int = 0
    n_concatenaciones_subdivididas: int = 0
    unidades_totales: int = 0
    unidades_del_mayor: int = 0
    n_fragmentos_que_mezclan_licencias: int = 0
    campos_abarcados_totales: int = 0
    combinaciones_de_licencias: dict = field(default_factory=dict)
    fragmentos_por_cantidad_de_campos: dict = field(default_factory=dict)
    ejemplo_de_mezcla: dict | None = None

    def a_json(self) -> dict:
        return {
            "n_obras": self.n_obras,
            "n_fragmentos": self.n_fragmentos,
            "fragmentos_por_obra": (
                round(self.n_fragmentos / self.n_obras, 2) if self.n_obras else 0.0
            ),
            "campos_por_fragmento": (
                round(self.campos_abarcados_totales / self.n_fragmentos, 2)
                if self.n_fragmentos
                else 0.0
            ),
            "n_concatenaciones_subdivididas": self.n_concatenaciones_subdivididas,
            "pct_subdivididas": (
                round(100 * self.n_concatenaciones_subdivididas / self.n_obras, 1)
                if self.n_obras
                else 0.0
            ),
            "unidades_promedio": (
                round(self.unidades_totales / self.n_obras, 1) if self.n_obras else 0.0
            ),
            "unidades_del_mayor": self.unidades_del_mayor,
            "n_fragmentos_que_mezclan_licencias": self.n_fragmentos_que_mezclan_licencias,
            "pct_que_mezclan_licencias": (
                round(100 * self.n_fragmentos_que_mezclan_licencias / self.n_fragmentos, 1)
                if self.n_fragmentos
                else 0.0
            ),
            "combinaciones_de_licencias": dict(sorted(self.combinaciones_de_licencias.items())),
            "fragmentos_por_cantidad_de_campos": {
                str(k): v for k, v in sorted(self.fragmentos_por_cantidad_de_campos.items())
            },
            "ejemplo_de_mezcla": self.ejemplo_de_mezcla,
            "nota": (
                "🔑 **El costo medido del chunking que la entrega aprobada pedía** "
                "— un fragmento que mezcla `CC0` con `CC-BY-4.0` "
                "fuerza atribución CC-BY sobre texto CC0. El chunking **por campo** no "
                "puede producir uno, y eso es un argumento nuevo a favor de chunking — el "
                "desvío no solo evita mezclar análisis con bibliografía, **también evita "
                "mezclar licencias**. ⚠️ `n_fragmentos` de esta tabla es el de la variante; "
                "el `frag` de la tabla por `section` cuenta **cuántos fragmentos abarcan** "
                "cada campo, así que suma más que el total a propósito."
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
        self.concatenacion = ConteoDeConcatenación()

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
        if self.estrategia.es_ciega:
            return self._fragmentar_ciego(registro)
        fragmentos: list[Fragmento] = []
        for section in self.campos_de_knowledge:
            campo = registro.seccion(section)
            if campo is None or not campo.texto.strip():
                continue
            fragmentos.extend(self._fragmentar_campo(registro, campo))
        return fragmentos

    def _campos_presentes(self, registro: RegistroCanónico) -> list[CampoDeTexto]:
        presentes = []
        for section in self.estrategia.orden_de_concatenacion:
            campo = registro.seccion(section)
            if campo is None or not campo.texto.strip():
                continue
            presentes.append(campo)
        return presentes

    def _fragmentar_ciego(self, registro: RegistroCanónico) -> list[Fragmento]:
        presentes = self._campos_presentes(registro)
        if not presentes:
            return []

        separador = self.estrategia.separador_de_concatenacion
        concatenacion = separador.join(c.texto for c in presentes)

        tramos: list[tuple[int, int, CampoDeTexto]] = []
        cursor = 0
        for campo in presentes:
            tramos.append((cursor, cursor + len(campo.texto), campo))
            cursor += len(campo.texto) + len(separador)

        for campo in presentes:
            conteo = self.secciones.setdefault(campo.section, ConteoDeSección())
            conteo.n_obras_con_el_campo += 1
            unidades_del_campo = self.medidor(campo.texto)
            conteo.unidades_totales += unidades_del_campo
            conteo.unidades_del_mayor = max(conteo.unidades_del_mayor, unidades_del_campo)
            if unidades_del_campo > self.estrategia.tamano_objetivo:
                conteo.n_campos_subdivididos += 1

        unidades = self.medidor(concatenacion)
        self.concatenacion.n_obras += 1
        self.concatenacion.unidades_totales += unidades
        self.concatenacion.unidades_del_mayor = max(
            self.concatenacion.unidades_del_mayor, unidades
        )
        if unidades <= self.estrategia.tamano_objetivo:
            trozos = [concatenacion]
        else:
            trozos = self._subdividir(concatenacion)
            self.concatenacion.n_concatenaciones_subdivididas += 1
            if len(trozos) < 2:
                raise ErrorDeConstrucción(
                    f"{registro.artwork_id} (brazo ciego): la concatenación mide "
                    f"{unidades} {self.estrategia.unidad} —más que el objetivo de "
                    f"{self.estrategia.tamano_objetivo}— y el splitter devolvió "
                    f"{len(trozos)} trozo(s). Un fragmento más grande que el objetivo es "
                    f"el camino al `400` por input largo del `PuertoEmbeddingTexto`, que "
                    f"el contrato de los puertos declara **error del fragmentador**."
                )

        fragmentos: list[Fragmento] = []
        buscar_desde = 0
        for i, trozo in enumerate(trozos):
            inicio = concatenacion.find(trozo, buscar_desde)
            if inicio < 0:
                raise ErrorDeConstrucción(
                    f"la verificación de subcadena roto en {registro.artwork_id} (brazo ciego, trozo {i}): el "
                    f"splitter devolvió un texto que **no aparece en la concatenación** "
                    f"desde la posición {buscar_desde}. En este brazo la concatenación es "
                    f"el texto de origen, así que el chunking corta y nunca "
                    f"reescribe. Trozo: {trozo[:80]!r}"
                )
            buscar_desde = inicio + 1
            fin = inicio + len(trozo)
            abarcados = [c for (a, b, c) in tramos if a < fin and b > inicio]
            fragmento = Fragmento.desde_concatenacion(
                registro,
                abarcados,
                texto=trozo,
                chunk_index=i,
                concatenacion=concatenacion,
            )
            fragmentos.append(fragmento)
            self._contar_fragmento_ciego(fragmento, trozo, inicio, fin, tramos)

        self.concatenacion.n_fragmentos += len(fragmentos)
        for fragmento in fragmentos:
            for section in fragmento.campos_abarcados:
                self.secciones.setdefault(section, ConteoDeSección()).n_fragmentos += 1
        return fragmentos

    def _contar_fragmento_ciego(
        self,
        fragmento: Fragmento,
        trozo: str,
        inicio: int,
        fin: int,
        tramos: Sequence[tuple[int, int, CampoDeTexto]],
    ) -> None:
        c = self.concatenacion
        n_campos = len(fragmento.campos_abarcados)
        c.campos_abarcados_totales += n_campos
        c.fragmentos_por_cantidad_de_campos[n_campos] = (
            c.fragmentos_por_cantidad_de_campos.get(n_campos, 0) + 1
        )
        combinacion = fragmento.combinacion_de_licencias
        c.combinaciones_de_licencias[combinacion] = (
            c.combinaciones_de_licencias.get(combinacion, 0) + 1
        )
        if not fragmento.mezcla_licencias:
            return
        c.n_fragmentos_que_mezclan_licencias += 1
        if c.ejemplo_de_mezcla is not None:
            return
        partes = []
        for a, b, campo in tramos:
            if not (a < fin and b > inicio):
                continue
            recorte = trozo[max(a - inicio, 0) : max(b - inicio, 0)]
            partes.append(
                {
                    "section": campo.section,
                    "license": campo.license,
                    "caracteres_en_el_chunk": len(recorte),
                    "texto_en_el_borde": (
                        recorte if len(recorte) <= 160 else recorte[:80] + " […] " + recorte[-80:]
                    ),
                }
            )
        c.ejemplo_de_mezcla = {
            "artwork_id": fragmento.artwork_id,
            "fragment_id": fragmento.fragment_id,
            "campos": list(fragmento.campos_abarcados),
            "licencias": list(fragmento.licencias_abarcadas),
            "license_singular": fragmento.license,
            "partes": partes,
            "lectura": (
                "Un solo fragmento, una sola licencia publicable, y adentro texto de dos "
                "licencias distintas. La unión incluye `CC-BY-4.0`, así que la atribución "
                "se aplica **también** a los "
                + str(
                    sum(
                        p["caracteres_en_el_chunk"]
                        for p in partes
                        if p["license"] == "CC0"
                    )
                )
                + " caracteres que son `CC0`. **Ése es el costo medido del chunking "
                "ciego**, y el chunking por campo no lo puede producir."
            ),
        }

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
        self.concatenacion = ConteoDeConcatenación()
        return {r.artwork_id: self.fragmentar(r) for r in registros}

    def a_json(self) -> dict:
        return {
            "estrategia": self.estrategia.a_json(),
            "campos_de_knowledge": list(self.campos_de_knowledge),
            "nota_de_p2": (
                "la ablación de campos de la colección de texto sigue **abierto**: el conjunto de campos es un punto de "
                "variación, no un valor. El default son los ✅ de `el relevamiento § Qué campos "
                "van a knowledge`; `exhibitions` y `citations` son el eje de la ablación "
                "secundaria y entran agregándolos a la lista."
            ),
            "nota_de_v1": (
                "Chunking **por campo semántico**: desvío  / el chunking por campo semántico, **no "
                "consultado. La entrega aprobada dice *fragmentos de "
                "unos 500 tokens con solapamiento*; está mitigado porque el chunking "
                "ciego es un brazo de la ablación, así que se decide con números."
                if not self.estrategia.es_ciega
                else (
                    "Chunking **ciego**: es el brazo que la entrega aprobada pedía "
                    "literalmente (*fragmentos de unos 500 tokens con solapamiento*) y el "
                    "que vuelve el desvío de chunking un desvío **mitigado**. Concatena los campos de "
                    "`orden_de_concatenacion` con `separador_de_concatenacion` y corta al "
                    "mismo tamaño objetivo que el otro brazo, para que la única diferencia "
                    "sea **dónde caen los bordes**."
                )
            ),
        }

    def conteos_por_seccion(self) -> dict:
        return {s: c.a_json() for s, c in sorted(self.secciones.items())}

    def conteos_de_concatenacion(self) -> dict:
        if not self.estrategia.es_ciega:
            return {}
        return self.concatenacion.a_json()

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
