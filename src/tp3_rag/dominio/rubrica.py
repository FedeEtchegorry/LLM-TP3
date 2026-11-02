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
                "El hash es **procedencia, no identidad** (la tabla de identidades): hashear el texto como "
                "identidad descartaría los ~540 juicios por una coma corregida. Como "
                "procedencia es el detector de que la rúbrica se movió sin cambiar de "
                "designación, que es lo que la evaluación castiga con *se descartan los juicios*."
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
                f"discrepar: la identidad va en el nombre (la regla de que la identidad va en el nombre), y acá el nombre es lo "
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
                f"en el esquema no existe (la configuración en capas), y en un artefacto de clase 3 una clave "
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
                f"{list(ESTADOS)}. `plantilla` es el esqueleto que entrega; completa "
                f"es la que el grupo escribió y con la que se puede anotar."
            )
        marcas = self.texto.count(MARCA)
        if self.estado == "completa" and marcas:
            raise ErrorDeArtefacto(
                f"{self.rubrica_id}: `estado = completa` y quedan {marcas} marcas "
                f"`{MARCA}` en el texto. **El texto de la rúbrica lo escribe el grupo** "
                f"(el módulo de evaluación), y una rúbrica a medio llenar declarada completa es peor que una "
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
            f"citar qué leyeron (la tabla de identidades, el compromiso con la rúbrica que se juzgó)."
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
            f'      # procedencia (la tabla de identidades): de qué rúbrica viene este refinamiento'
        )
    else:
        front.append(
            '# refina   = "piloto"      # si esta rúbrica REFINA otra (el acuerdo entre anotadores, κ < 0.6), decilo acá'
        )
    front += [
        f'escrita_por = "{MARCA}"       # quiénes la escribieron (es trabajo humano, el módulo de evaluación)',
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
        "> la ronda piloto existe para no hacer con los ~540 (la evaluación).",
        ">",
        "> **Por qué importa tanto un texto:** κ es siempre *κ sobre una rúbrica*",
        "> (el módulo de evaluación). Dos anotadores que no leyeron la misma no produjeron",
        "> juicios comparables, y si este texto cambia, **los juicios hechos con el anterior",
        "> se descartan** (la evaluación). Eso es lo que el piloto de 20 juicios compra: pagar",
        "> ese descarte cuando cuesta 20 y no 540.",
        "",
        "## La pregunta que se contesta en cada juicio",
        "",
        "Dada **la imagen original** de la consulta y **el pedido**, ¿cuánto cumple",
        "**esta obra recuperada** el pedido *respecto de la imagen original*?",
        "",
        "⚠️ **No es «se parece»**: la evaluación dice que *si el usuario pide «versiones más",
        "cálidas» de un paisaje gris, la respuesta correcta probablemente no comparte",
        "estilo con el original*. Premiar el parecido sería premiar al sistema por ignorar",
        "al usuario.",
        "",
    ]
    for grado in sorted(GRADOS, reverse=True):
        cuerpo += [
            f"## Grado {grado} — {SIGNIFICADO_DE_LA_ESCALA[grado]}",
            "",
            "*(El renglón de arriba es el de `la evaluación § Escala` y **ya está decidido**: no",
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
        "*(La regla para cuando el juicio queda entre dos grados. la evaluación fija quién",
        "resuelve un desacuerdo **entre anotadores** —el tercer integrante, y ese desempate",
        "es **una anotación más**, no una corrección del consenso—; lo que falta acá es la",
        "regla para la duda de **una sola persona**.)*",
        "",
        f"{MARCA}: entre 2 y 1, ¿cuál se elige? Entre 1 y 0, ¿cuál? Y qué se hace con",
        "los casos que la evaluación ya nombra y que van a aparecer:",
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
        "  (la tabla de identidades, clase 3). **No es un hash del texto, a propósito**: una",
        "  coma corregida daría otra rúbrica y descartaría los ~540 juicios, y la clase 3",
        "  existe para que ese precio lo decida una persona y no un diff.",
        "- La escala (tres valores) es compartida; **lo que significa cada valor lo fija",
        "  este archivo** (el núcleo compartido).",
        "- Validar: `poetry run tp3-anotar rubrica`",
        "",
    ]
    return "\n".join(front + cuerpo)

SUFIJO_DE_LA_DECISIÓN = "-decision-de-a5.toml"

SALIDAS_DE_A5: tuple[str, ...] = ("limitacion_declarada", "refinada")

CLAVES_OBLIGATORIAS_DE_LA_DECISIÓN = (
    "rubrica_id",
    "salida",
    "kappa_leido",
    "pool_del_kappa",
    "hash_del_texto",
    "decidida_por",
    "fecha",
)
CLAVES_OPCIONALES_DE_LA_DECISIÓN = ("limitacion", "refinada_en", "nota")

DECIMALES_DEL_KAPPA = 3

@dataclass(frozen=True)
class DecisiónDeA5:
    datos: dict
    texto: str

    @property
    def rubrica_id(self) -> str:
        return self.datos["rubrica_id"]

    @property
    def salida(self) -> str:
        return self.datos["salida"]

    @property
    def kappa_leido(self) -> float:
        return float(self.datos["kappa_leido"])

    @property
    def pool_del_kappa(self) -> str:
        return self.datos["pool_del_kappa"]

    @property
    def hash_del_texto(self) -> str:
        return self.datos["hash_del_texto"]

    @property
    def limitacion(self) -> str:
        return (self.datos.get("limitacion") or "").strip()

    @property
    def refinada_en(self) -> str | None:
        return self.datos.get("refinada_en")

    @property
    def decidida_por(self) -> str:
        return self.datos["decidida_por"]

    @property
    def fecha(self) -> str:
        return self.datos["fecha"]

    @property
    def autoriza_exportar(self) -> bool:
        return self.salida == "limitacion_declarada"

    def problemas_contra(self, *, kappas, hash_del_texto: str) -> list[str]:
        problemas: list[str] = []
        if hash_del_texto != self.hash_del_texto:
            problemas.append(
                f"la decisión cita `hash_del_texto = {self.hash_del_texto}` y la rúbrica en "
                f"disco hashea {hash_del_texto}: **el texto se movió después de decidir**. "
                f"Una limitación declarada sobre un texto no vale para otro — es  "
                f"sobre la autorización. Volvé a declararla con el hash nuevo, o restaurá "
                f"el texto (`git diff`)."
            )
        cubiertos = [
            k
            for k in kappas
            if k.pool_id == self.pool_del_kappa
            and k.kappa is not None
            and round(k.kappa, DECIMALES_DEL_KAPPA)
            == round(self.kappa_leido, DECIMALES_DEL_KAPPA)
        ]
        if not cubiertos:
            medidos = {k.pool_id: k.kappa for k in kappas}
            problemas.append(
                f"la decisión dice haber leído `kappa_leido = {self.kappa_leido}` sobre el "
                f"pool {self.pool_del_kappa!r} y lo que hay medido es {medidos}. **Una "
                f"autorización que no cita la medición que autoriza vale para cualquier κ "
                f"futuro**, y entonces este chequeo se desactivaría una vez y para siempre."
            )
        sin_cubrir = [
            k.pool_id
            for k in kappas
            if k.esta_bajo_el_umbral and k.pool_id != self.pool_del_kappa
        ]
        if sin_cubrir:
            problemas.append(
                f"hay otro(s) κ bajo el umbral que esta decisión **no** menciona: "
                f"{sorted(set(sin_cubrir))}. Cada medición baja necesita su decisión: "
                f"cubrir una y heredar el resto es el agujero que el chequeo cierra."
            )
        if not self.autoriza_exportar:
            problemas.append(
                f"`salida = {self.salida!r}` **no autoriza a exportar con esta rúbrica**. "
                f"Refinar es derivar otra: la salida se ejerce exportando "
                + (
                    f"con `--rubrica {self.refinada_en}`."
                    if self.refinada_en
                    else "con la rúbrica refinada."
                )
            )
        return problemas

    def a_json(self) -> dict:
        return {
            "rubrica_id": self.rubrica_id,
            "salida": self.salida,
            "autoriza_exportar": self.autoriza_exportar,
            "kappa_leido": self.kappa_leido,
            "pool_del_kappa": self.pool_del_kappa,
            "hash_del_texto": self.hash_del_texto,
            "decidida_por": self.datos.get("decidida_por"),
            "fecha": self.datos.get("fecha"),
            "limitacion": self.limitacion or None,
            "refinada_en": self.refinada_en,
            "archivo": str(self.ruta_de("artifacts/ground-truth/rubricas", self.rubrica_id)),
            "nota": (
                "el acuerdo entre anotadores declara dos salidas para κ < umbral y **solo la limitación explícita "
                "autoriza a exportar con esta rúbrica**: refinar es derivar otro rubrica_id "
                ", así que esa salida se ejerce exportando con la otra. La decisión cita "
                "el κ, su pool y el hash del texto: si se re-mide o se edita la rúbrica, deja "
                "de cubrir y el chequeo vuelve a bloquear."
            ),
        }

    @staticmethod
    def ruta_de(raíz: str | Path, rubrica_id: str) -> Path:
        return Path(raíz) / f"{rubrica_id}{SUFIJO_DE_LA_DECISIÓN}"

    @classmethod
    def desde_texto(cls, texto: str, *, ruta=None) -> "DecisiónDeA5":
        try:
            datos = tomllib.loads(texto)
        except Exception as e:
            raise ErrorDeArtefacto(
                f"{ruta or 'la decisión del acuerdo entre anotadores'} no es TOML válido ({e}). Es TOML como todo "
                f"lo declarado del proyecto (la configuración en capas)."
            ) from None
        decisión = cls(datos=datos, texto=texto)
        decisión.validar(ruta=ruta)
        return decisión

    @classmethod
    def leer_si_existe(cls, almacén, raíz: str | Path, rubrica_id: str) -> "DecisiónDeA5 | None":
        ruta = cls.ruta_de(raíz, rubrica_id)
        if not almacén.existe(ruta):
            return None
        decisión = cls.desde_texto(almacén.leer_texto(ruta), ruta=almacén.ruta(ruta))
        if decisión.rubrica_id != rubrica_id:
            raise ErrorDeArtefacto(
                f"{almacén.ruta(ruta)} declara adentro `rubrica_id = "
                f"{decisión.rubrica_id!r}`. El nombre y el contenido no pueden discrepar "
                f"(la regla de que la identidad va en el nombre): acá el nombre es lo que la exportación busca."
            )
        return decisión

    def validar(self, *, ruta=None) -> None:
        dónde = ruta or "la decisión del acuerdo entre anotadores"
        faltantes = [
            k for k in CLAVES_OBLIGATORIAS_DE_LA_DECISIÓN if self.datos.get(k) in (None, "")
        ]
        if faltantes:
            raise ErrorDeArtefacto(
                f"{dónde}: la decisión del acuerdo entre anotadores no declara {faltantes}. Las siete son "
                f"obligatorias porque son lo que la vuelve **auditable**: qué rúbrica, cuál "
                f"de las dos salidas del acuerdo entre anotadores, **qué κ se leyó y de qué pool**, sobre qué "
                f"texto, quién decidió y cuándo. Una autorización sin el número que autoriza "
                f"valdría para cualquier κ futuro."
            )
        de_más = sorted(
            set(self.datos)
            - set(CLAVES_OBLIGATORIAS_DE_LA_DECISIÓN)
            - set(CLAVES_OPCIONALES_DE_LA_DECISIÓN)
        )
        if de_más:
            raise ErrorDeArtefacto(
                f"{dónde}: trae claves que no se guardan: {de_más}. Lo que no está en el "
                f"esquema no existe (la configuración en capas), y en un artefacto de clase 3 una clave "
                f"ignorada en silencio es trabajo humano que se pierde. Las opcionales son "
                f"{list(CLAVES_OPCIONALES_DE_LA_DECISIÓN)}."
            )
        identidad.rubrica_id(self.rubrica_id)
        if self.salida not in SALIDAS_DE_A5:
            raise ErrorDeArtefacto(
                f"{dónde}: `salida = {self.salida!r}` no es una de las dos que el acuerdo entre anotadores declara, "
                f"{list(SALIDAS_DE_A5)}. No hay una tercera: *se refina la rúbrica y se "
                f"re-anota, o se reporta como limitación explícita*."
            )
        if self.datos.get("limitacion") and MARCA in str(self.datos["limitacion"]):
            raise ErrorDeArtefacto(
                f"{dónde}: la `limitacion` todavía tiene marcas `{MARCA}`. **Es el párrafo "
                f"que va al informe**: una limitación a medio escribir declarada como "
                f"decidida es peor que no tenerla, porque desbloquea la exportación sin que "
                f"nadie haya escrito la limitación."
            )
        if self.autoriza_exportar and not self.limitacion:
            raise ErrorDeArtefacto(
                f"{dónde}: `salida = \"limitacion_declarada\"` y `limitacion` está vacía. "
                f"**La limitación *es* la salida del acuerdo entre anotadores**, no una nota al pie: lo que el acuerdo entre anotadores "
                f"compromete es *reportarlo como limitación explícita*, y sin el texto no hay "
                f"nada que reportar."
            )
        if self.salida == "refinada":
            if not self.refinada_en:
                raise ErrorDeArtefacto(
                    f"{dónde}: `salida = \"refinada\"` y no dice `refinada_en`. Refinar es "
                    f"derivar otra rúbrica, así que la decisión tiene que nombrar "
                    f"cuál — es lo que vuelve el refinamiento auditable en vez de un archivo "
                    f"nuevo sin historia (la tabla de identidades)."
                )
            identidad.rubrica_id(self.refinada_en)
            if self.refinada_en == self.rubrica_id:
                raise ErrorDeArtefacto(
                    f"{dónde}: `refinada_en` apunta a la misma rúbrica. Refinar es derivar "
                    f"**otra** designación (la regla de que un id determina su contenido, clase 3: *no se reescribe*)."
                )

def plantilla_de_decision_de_a5(
    *, rubrica_id: str, kappa: float | None, pool_id: str, hash_del_texto: str, umbral: float
) -> str:
    identidad.rubrica_id(rubrica_id)
    return "\n".join(
        [
            f"#  Decisión del acuerdo entre anotadores sobre la rúbrica `{rubrica_id}` — clase 3, la escribe el grupo",
            "#",
            f"#  κ medido = {kappa} < {umbral}, así que el acuerdo entre anotadores pide elegir **una de",
            "#  dos** y este archivo es dónde queda elegida:",
            "#",
            '#    · "refinada"              → derivar OTRA rúbrica y re-anotar. ⚠️ NO autoriza',
            "#                                a exportar con esta: la salida se ejerce",
            "#                                exportando con la refinada (`--rubrica …`).",
            '#    · "limitacion_declarada"  → reportarlo como limitación explícita. Autoriza a',
            "#                                exportar, y el texto de abajo **va al informe**.",
            "#",
            "#  🔴 Mientras este archivo no exista, `tp3-anotar exportar` **falla** con esta",
            "# rúbrica, y lo que impide es gastar anotación bajo un instrumento",
            "#  cuyo propio κ dice que es ambiguo — que es, literalmente, lo que la ronda",
            "#  piloto existe para no hacer.",
            "",
            f'rubrica_id     = "{rubrica_id}"',
            f'salida         = "{MARCA}"   # limitacion_declarada | refinada',
            "",
            "#  ── La medición que se está decidiendo. La emite el pipeline: no se",
            "#     transcribe a mano, y si se re-mide o se edita el texto de la rúbrica, esta",
            "#     decisión **deja de cubrir** y el chequeo vuelve a bloquear.",
            f"kappa_leido    = {kappa}",
            f'pool_del_kappa = "{pool_id}"',
            f'hash_del_texto = "{hash_del_texto}"',
            "",
            f'decidida_por   = "{MARCA}"   # quiénes lo decidieron (es trabajo humano, el módulo de evaluación)',
            f'fecha          = "{MARCA}"   # AAAA-MM-DD',
            "",
            "#  ── Si `salida = \"refinada\"`: cuál es la rúbrica derivada ──────────────────",
            '# refinada_en  = "v2"',
            "",
            "#  ── Si `salida = \"limitacion_declarada\"`: EL PÁRRAFO DEL INFORME ───────────",
            "#",
            "#     No es una nota al pie: el acuerdo entre anotadores compromete *reportarlo como limitación",
            "#     explícita*, así que esto es lo que se publica. Decí qué significa el",
            "#     número, sobre qué población se midió y qué NO se puede concluir del",
            "#     ground truth construido con este instrumento.",
            "limitacion = \"\"\"",
            f"{MARCA}",
            '"""',
            "",
        ]
    )

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
