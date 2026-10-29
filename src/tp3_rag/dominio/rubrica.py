from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto

if sys.version_info >= (3, 11):  # pragma: no cover - el lock fija 3.10
    import tomllib
else:
    import tomli as tomllib

VERSIÓN_DE_LA_RÚBRICA = "1"

DELIMITADOR = "+++"

ESTADOS: tuple[str, ...] = ("plantilla", "completa")

CLAVES_OBLIGATORIAS = ("rubrica_id", "estado")
CLAVES_OPCIONALES = ("refina", "escrita_por", "fecha", "nota")

MARCA = "REEMPLAZAR"

_SECCIÓN = re.compile(r"^##\s+(?P<titulo>.+?)\s*$", re.MULTILINE)
_GRADO = re.compile(r"^Grado\s+(?P<grado>[012])\b")
_EJEMPLOS = re.compile(r"^Ejemplos\b", re.IGNORECASE)
_DESEMPATE = re.compile(r"^Desempate\b", re.IGNORECASE)

@dataclass(frozen=True)
class Rúbrica:
    front_matter: dict
    cuerpo: str
    texto: str

    @property
    def rubrica_id(self) -> str:
        return self.front_matter["rubrica_id"]

    @property
    def estado(self) -> str:
        return self.front_matter["estado"]

    @property
    def refina(self) -> str | None:
        return self.front_matter.get("refina")

    @property
    def es_usable_para_anotar(self) -> bool:
        return self.estado == "completa"

    @property
    def hash_del_texto(self) -> str:
        return identidad.hash_corto({"texto": self.texto})

    def definicion_de(self, grado: int) -> str:
        return self.secciones()[f"grado_{grado}"]

    def secciones(self) -> dict[str, str]:
        encontradas: dict[str, str] = {}
        marcas = list(_SECCIÓN.finditer(self.cuerpo))
        for n, marca in enumerate(marcas):
            título = marca.group("titulo")
            fin = marcas[n + 1].start() if n + 1 < len(marcas) else len(self.cuerpo)
            contenido = self.cuerpo[marca.end():fin].strip()
            grado = _GRADO.match(título)
            if grado:
                encontradas[f"grado_{grado.group('grado')}"] = contenido
            elif _EJEMPLOS.match(título):
                encontradas["ejemplos"] = contenido
            elif _DESEMPATE.match(título):
                encontradas["desempate"] = contenido
        return encontradas

    def cita(self, raíz: str | Path | None = None) -> dict:
        from tp3_rag.dominio import ground_truth as gt

        base = Path(raíz) if raíz is not None else Path(gt.RAÍZ) / gt.RUBRICAS
        return {
            "rubrica_id": self.rubrica_id,
            "estado": self.estado,
            "archivo": str(self.ruta_de(base, self.rubrica_id)),
            "hash_del_texto": self.hash_del_texto,
            **({"refina": self.refina} if self.refina else {}),
            "nota": (
                "El hash es **procedencia, no identidad**: hashear el texto como "
                "identidad descartaría los ~540 juicios por una coma corregida. Como "
                "procedencia es el detector de que la rúbrica se movió sin cambiar de "
                "designación, que es lo que el diseño de la evaluación castiga con *se descartan los juicios*."
            ),
        }

    @staticmethod
    def ruta_de(raíz: str | Path, rubrica_id: str) -> Path:
        return Path(raíz) / f"{rubrica_id}.md"

    @classmethod
    def desde_texto(cls, texto: str, *, ruta=None) -> "Rúbrica":
        front_matter, cuerpo = _partir(texto, ruta=ruta)
        rúbrica = cls(front_matter=front_matter, cuerpo=cuerpo, texto=texto)
        rúbrica.validar()
        return rúbrica

    @classmethod
    def leer(cls, almacén, raíz: str | Path, rubrica_id: str) -> "Rúbrica":
        ruta = cls.ruta_de(raíz, rubrica_id)
        rúbrica = cls.desde_texto(almacén.leer_texto(ruta), ruta=ruta)
        if rúbrica.rubrica_id != rubrica_id:
            raise ErrorDeArtefacto(
                f"la rúbrica de {rubrica_id}.md declara adentro "
                f"`rubrica_id = {rúbrica.rubrica_id}`. El nombre y el contenido no pueden "
                f"discrepar: la identidad va en el nombre, y acá el nombre es lo "
                f"que los juicios citan."
            )
        return rúbrica

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.md")))

    def validar(self) -> None:
        self._validar_front_matter()
        self._validar_secciones()
        self._validar_estado()

    def _validar_front_matter(self) -> None:
        faltantes = [k for k in CLAVES_OBLIGATORIAS if not self.front_matter.get(k)]
        if faltantes:
            raise ErrorDeArtefacto(
                f"el front matter de la rúbrica no declara {faltantes}. `rubrica_id` **es** "
                f"la identidad (la tabla de identidades, clase 3: una designación que asigna una persona) y "
                f"`estado` es lo que distingue una plantilla de un instrumento."
            )
        de_más = sorted(set(self.front_matter) - set(CLAVES_OBLIGATORIAS) - set(CLAVES_OPCIONALES))
        if de_más:
            raise ErrorDeArtefacto(
                f"el front matter trae claves que no se guardan: {de_más}. Lo que no está "
                f"en el esquema no existe, y en un artefacto de clase 3 una clave "
                f"ignorada en silencio es trabajo humano que se pierde. Las opcionales son "
                f"{list(CLAVES_OPCIONALES)}."
            )
        identidad.rubrica_id(self.rubrica_id)
        if self.refina is not None:
            identidad.rubrica_id(self.refina)
            if self.refina == self.rubrica_id:
                raise ErrorDeArtefacto(
                    f"{self.rubrica_id}: `refina` apunta a sí misma. Refinar es **derivar "
                    f"otra** rúbrica (la regla de que un id determina su contenido, clase 3: *no se reescribe*); una rúbrica que "
                    f"se refina a sí misma es la segunda escritura que el acuerdo entre anotadores necesitaba poder "
                    f"evitar."
                )

    def _validar_secciones(self) -> None:
        secciones = self.secciones()
        esperadas = [f"grado_{g}" for g in sorted(GRADOS, reverse=True)] + [
            "ejemplos", "desempate"
        ]
        faltan = [s for s in esperadas if s not in secciones]
        if faltan:
            raise ErrorDeArtefacto(
                f"{self.front_matter.get('rubrica_id')}: la rúbrica no tiene las secciones "
                f"{faltan}. Las cinco son obligatorias porque son **lo que la tabla de identidades dice que "
                f"una rúbrica es**: las definiciones de los tres valores del `Grado` "
                f"(`## Grado 2`, `## Grado 1`, `## Grado 0`), sus ejemplos (`## Ejemplos`) y "
                f"las instrucciones de desempate (`## Desempate`). Una rúbrica sin "
                f"desempate produce el κ bajo que el piloto existe para encontrar."
            )
        vacías = [s for s in esperadas if not secciones[s]]
        if vacías:
            raise ErrorDeArtefacto(
                f"{self.rubrica_id}: las secciones {vacías} están **vacías**. Un "
                f"encabezado sin texto no es una definición: κ se calcula sobre lo que los "
                f"dos anotadores leyeron."
            )

    def _validar_estado(self) -> None:
        if self.estado not in ESTADOS:
            raise ErrorDeArtefacto(
                f"{self.rubrica_id}: `estado` = {self.estado!r} no es uno de "
                f"{list(ESTADOS)}. `plantilla` es el esqueleto que entrega; `completa` "
                f"es la que el grupo escribió y con la que se puede anotar."
            )
        marcas = self.texto.count(MARCA)
        if self.estado == "completa" and marcas:
            raise ErrorDeArtefacto(
                f"{self.rubrica_id}: `estado = completa` y quedan {marcas} marcas "
                f"`{MARCA}` en el texto. **El texto de la rúbrica lo escribe el grupo** "
                f", y una rúbrica a medio llenar declarada completa es peor que una "
                f"plantilla: los dos anotadores leerían un hueco y κ mediría el hueco. "
                f"Llenala, o dejala en `estado = \"plantilla\"`, que es honesto y lo dice."
            )
        if self.estado == "plantilla" and not marcas:
            raise ErrorDeArtefacto(
                f"{self.rubrica_id}: `estado = plantilla` y no queda ni una marca "
                f"`{MARCA}`. Si ya está escrita, declarala `completa`: un instrumento "
                f"terminado que se declara borrador **no se puede usar para anotar** y el "
                f"trabajo de escribirlo no serviría de nada."
            )

def _partir(texto: str, *, ruta=None) -> tuple[dict, str]:
    líneas = texto.splitlines()
    if not líneas or líneas[0].strip() != DELIMITADOR:
        raise ErrorDeArtefacto(
            f"{ruta or 'la rúbrica'} no empieza con `{DELIMITADOR}`. El front matter es "
            f"**TOML entre `{DELIMITADOR}`** y es donde vive la identidad: un `.md` sin "
            f"front matter es un texto sin `rubrica_id`, y entonces los juicios no pueden "
            f"citar qué leyeron (la tabla de identidades, )."
        )
    try:
        cierre = next(n for n, l in enumerate(líneas[1:], 1) if l.strip() == DELIMITADOR)
    except StopIteration:
        raise ErrorDeArtefacto(
            f"{ruta or 'la rúbrica'}: el front matter abre con `{DELIMITADOR}` y no cierra."
        ) from None
    try:
        front_matter = tomllib.loads("\n".join(líneas[1:cierre]))
    except Exception as e:
        raise ErrorDeArtefacto(
            f"{ruta or 'la rúbrica'}: el front matter no es TOML válido ({e})."
        ) from None
    return front_matter, "\n".join(líneas[cierre + 1:])

def plantilla(*, designacion: str = "piloto", refina: str | None = None) -> str:
    identidad.rubrica_id(designacion)
    front = [
        DELIMITADOR,
        f'rubrica_id = "{designacion}"',
        'estado     = "plantilla"      # plantilla | completa  ← pasalo a `completa` al terminar',
    ]
    if refina:
        identidad.rubrica_id(refina)
        front.append(
            f'refina     = "{refina}"'
            f'      # procedencia: de qué rúbrica viene este refinamiento'
        )
    else:
        front.append(
            '# refina = "piloto" # si esta rúbrica REFINA otra (el acuerdo entre anotadores, κ < 0.6), decilo acá'
        )
    front += [
        f'escrita_por = "{MARCA}" # quiénes la escribieron (es trabajo humano, el módulo de evaluación)',
        f'fecha      = "{MARCA}"        # AAAA-MM-DD',
        DELIMITADOR,
        "",
    ]

    cuerpo = [
        f"# Rúbrica `{designacion}` — cómo se asigna un grado 0 / 1 / 2",
        "",
        "> 🔴 **Este archivo es una plantilla y el texto lo escribe el grupo.**",
        f"> Reemplazá cada `{MARCA}`. Mientras quede uno, `estado` tiene que seguir en",
        "> `plantilla`, y con `estado = \"plantilla\"` **no se puede exportar una planilla**:",
        "> gastar anotación humana bajo una rúbrica a medio escribir es exactamente lo que",
        "> la ronda piloto existe para no hacer con los ~540.",
        ">",
        "> **Por qué importa tanto un texto:** κ es siempre *κ sobre una rúbrica*",
        ">. Dos anotadores que no leyeron la misma no produjeron",
        "> juicios comparables, y si este texto cambia, **los juicios hechos con el anterior",
        "> se descartan**. Eso es lo que el piloto de 20 juicios compra: pagar",
        "> ese descarte cuando cuesta 20 y no 540.",
        "",
        "## La pregunta que se contesta en cada juicio",
        "",
        "Dada **la imagen original** de la consulta y **el pedido**, ¿cuánto cumple",
        "**esta obra recuperada** el pedido *respecto de la imagen original*?",
        "",
        "⚠️ **No es «se parece»**: el diseño de la evaluación dice que *si el usuario pide «versiones más",
        "cálidas» de un paisaje gris, la respuesta correcta probablemente no comparte",
        "estilo con el original*. Premiar el parecido sería premiar al sistema por ignorar",
        "al usuario.",
        "",
    ]
    for grado in sorted(GRADOS, reverse=True):
        cuerpo += [
            f"## Grado {grado} — {SIGNIFICADO_DE_LA_ESCALA[grado]}",
            "",
            "*(El renglón de arriba es el del diseño de la evaluación y **ya está decidido**: no",
            "se cambia. Lo que falta es la definición operativa.)*",
            "",
            f"{MARCA}: cuándo exactamente se pone {grado}. Escribilo de forma que dos",
            f"personas que lean esto solas pongan el mismo número. Si para decidirlo hace",
            f"falta mirar algo que la planilla no muestra, decilo acá.",
            "",
        ]
    cuerpo += [
        "## Ejemplos",
        "",
        "*(Al menos uno por grado. Un ejemplo con el par concreto —imagen, pedido, obra—",
        "vale más que una definición más larga: es lo que se mira cuando hay duda.)*",
        "",
        "| grado | imagen original | pedido | obra recuperada | por qué |",
        "|---|---|---|---|---|",
        f"| 2 | {MARCA} | {MARCA} | {MARCA} | {MARCA} |",
        f"| 1 | {MARCA} | {MARCA} | {MARCA} | {MARCA} |",
        f"| 0 | {MARCA} | {MARCA} | {MARCA} | {MARCA} |",
        "",
        "## Desempate",
        "",
        "*(La regla para cuando el juicio queda entre dos grados. el diseño de la evaluación fija quién",
        "resuelve un desacuerdo **entre anotadores** —el tercer integrante, y ese desempate",
        "es **una anotación más**, no una corrección del consenso—; lo que falta acá es la",
        "regla para la duda de **una sola persona**.)*",
        "",
        f"{MARCA}: entre 2 y 1, ¿cuál se elige? Entre 1 y 0, ¿cuál? Y qué se hace con",
        "los casos que el diseño de la evaluación ya nombra y que van a aparecer:",
        "",
        "- la obra cumple el pedido pero **es la misma escena** que la original;",
        "- el pedido es **ambiguo** (*«con más vida»*) y hay dos lecturas razonables;",
        "- el pedido es **cultural** y la obra es de la cultura pedida pero no del tema;",
        "- la obra **no tiene** el dato con el que se juzgaría (sin `culture`, sin fecha).",
        "",
        "---",
        "",
        "## Procedencia (no se edita)",
        "",
        f"- Identidad: `rubrica_id` = la **designación declarada** del front matter",
        " (la tabla de identidades, clase 3). **No es un hash del texto, a propósito**: una",
        "  coma corregida daría otra rúbrica y descartaría los ~540 juicios, y la clase 3",
        "  existe para que ese precio lo decida una persona y no un diff.",
        "- La escala (tres valores) es compartida; **lo que significa cada valor lo fija",
        "  este archivo**.",
        "- Validar: `poetry run tp3-anotar rubrica`",
        "",
    ]
    return "\n".join(front + cuerpo)

def nota_de_identidad() -> Mapping[str, Any]:
    return {
        "clase": 3,
        "entra": "una designación declarada (piloto, v1, v2…), asignada a mano",
        "excluye": (
            "el texto, los ejemplos, el desempate y la rúbrica que refina —los cuatro "
            "adentro—, y el split_id, el pool_id y el anotador, que son de quien la usa"
        ),
        "por_que_no_se_hashea_el_texto": (
            "la tabla de identidades: una coma corregida daría otra rúbrica y descartaría los ~540 juicios. "
            "El hash va como procedencia (cita()), que es donde detecta en vez de invalidar."
        ),
    }
