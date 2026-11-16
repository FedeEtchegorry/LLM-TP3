"""Filtro por metadata en pre-retrieval (Clase 7.2: la metadata «limita el scope del
retriever permitiendo hacer filtros previos»).

Qué hace: si el **pedido** de la persona nombra una cultura o un país, la búsqueda de
texto se restringe a las obras cuya `culture` cumple con eso. La restricción se aplica
como un `where` de ChromaDB **antes** de la búsqueda por similitud, no después.

Tres decisiones, y por qué:

1. **Se lee el pedido, no el plan.** La `consulta_conceptual` del pre-retrieval nombra a
   veces la cultura de la *imagen* aunque el pedido no la pida (medido: 6 de los 20
   planes no culturales del set). Un filtro duro sobre eso restringiría resultados que
   la persona no pidió restringir.
2. **Vocabulario cerrado.** Un `where` es una igualdad exacta, así que solo sirven los
   valores que existen en el índice. Los términos se resuelven contra los valores de
   `culture` de la colección: «India» cubre también `Mughal India`, `Northern India`, etc.
3. **Si no hay con qué filtrar, no se filtra.** Sin cultura nombrada, con un término que
   no está en el índice o con menos de `MÍNIMO_DE_OBRAS` obras, la búsqueda corre como
   antes y el motivo queda registrado en la corrida.

Además de la cultura, el pedido puede acotar por **técnica** (familia de `medium`), por
**artista** (nombre del índice) y por **período** (rango de `year`: un siglo o un movimiento).
Cada eje se resuelve igual que la cultura y los ejes nombrados se intersecan. `style` no se
usa: le falta el 74 % de las obras.

Límite declarado: los términos se reconocen con una tabla de nombres en español e inglés
(`ALIAS_DE_CULTURA`). Una cultura que no esté en la tabla no filtra; no rompe nada, pero
tampoco ayuda.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Sequence

CAMPO = "culture"

# Menos obras que esto y el filtro no se aplica: el reranker ve 20 candidatas y el
# sistema compromete 5 obras; un conjunto elegible más chico que eso no deja elegir.
MÍNIMO_DE_OBRAS = 20

_PAISES_BAJOS = (
    "paises bajos", "holanda", "holandes", "holandesa", "holandeses", "holandesas",
    "neerlandes", "neerlandesa", "netherlands", "holland", "dutch",
)
_ESTADOS_UNIDOS = (
    "estados unidos", "estadounidense", "estadounidenses", "norteamericano", "norteamericana",
    "norteamericanos", "norteamericanas", "united states", "american",
)

# término canónico (como aparece en `culture`) -> formas en que el pedido puede nombrarlo.
# Las formas van en minúsculas y sin tildes; se comparan por palabra entera.
ALIAS_DE_CULTURA: Mapping[str, tuple[str, ...]] = {
    "France": ("francia", "frances", "francesa", "franceses", "francesas", "france", "french"),
    "China": ("china", "chino", "chinos", "chinas", "chinese"),
    "Japan": ("japon", "japones", "japonesa", "japoneses", "japonesas", "japan", "japanese"),
    "India": ("india", "indio", "indios", "indias", "hindu", "indian"),
    "Mughal": ("mogol", "mogola", "mogoles", "mogolas", "mughal"),
    "Italy": ("italia", "italiano", "italiana", "italianos", "italianas", "italy", "italian"),
    "Netherlands": _PAISES_BAJOS,
    "Holland": _PAISES_BAJOS,
    "Flanders": ("flandes", "flamenco", "flamenca", "flamencos", "flamencas", "flanders", "flemish"),
    "United States": _ESTADOS_UNIDOS,
    "America": _ESTADOS_UNIDOS,
    "Korea": ("corea", "coreano", "coreana", "coreanos", "coreanas", "korea", "korean"),
    "Germany": ("alemania", "aleman", "alemana", "alemanes", "alemanas", "germany", "german"),
    "England": ("inglaterra", "ingles", "inglesa", "ingleses", "inglesas", "britanico", "britanica", "britanicos", "britanicas", "england", "english", "british"),
    "United Kingdom": ("reino unido", "britanico", "britanica", "britanicos", "britanicas", "united kingdom", "british"),
    "Spain": ("espana", "espanol", "espanola", "espanoles", "espanolas", "spain", "spanish"),
    "Nepal": ("nepal", "nepali", "nepales", "nepalesa", "nepalese"),
    "Tibet": ("tibet", "tibetano", "tibetana", "tibetanos", "tibetanas", "tibetan"),
    "Austria": ("austria", "austriaco", "austriaca", "austriacos", "austriacas", "austrian"),
    "Belgium": ("belgica", "belga", "belgas", "belgium", "belgian"),
    "Egypt": ("egipto", "egipcio", "egipcia", "egipcios", "egipcias", "egypt", "egyptian"),
    "Greece": ("grecia", "griego", "griega", "griegos", "griegas", "greece", "greek"),
}

# Artista: pocas obras por artista, así que el mínimo es el top_k y no las 20 del reranker.
MÍNIMO_DE_OBRAS_ARTISTA = 5

# familias de `familia_de_tecnica` -> formas en que el pedido nombra la técnica o el soporte.
# Formas que también son color («tonos pastel», «colores frescos») piden la preposición.
ALIAS_DE_TECNICA: Mapping[tuple[str, ...], tuple[str, ...]] = {
    ("oleo",): ("oleo", "al oleo", "oil"),
    ("temple_sobre_papel", "temple_sobre_tabla"): ("tempera", "temple"),
    ("tinta_sobre_papel_o_seda",): ("tinta", "acuarela", "acuarelas", "ink", "watercolor"),
    ("pastel_o_gouache",): ("al pastel", "gouache", "guache"),
    ("fresco_o_muro",): ("al fresco", "mural", "murales", "fresco mural"),
}

# movimiento -> (desde, hasta) de `year`. Fechas aproximadas, con margen en los bordes.
_RENACIMIENTO, _BARROCO, _ROCOCO = (1400, 1600), (1600, 1750), (1720, 1790)
_NEOCLASICISMO, _ROMANTICISMO, _REALISMO = (1760, 1830), (1800, 1860), (1840, 1890)
_IMPRESIONISMO, _POSTIMPRESIONISMO = (1860, 1895), (1880, 1910)
# «realista» y «romántico» no están: «más realista» y «más romántico» describen una imagen.
MOVIMIENTOS: Mapping[str, tuple[int, int]] = {
    **dict.fromkeys(("renacimiento", "renacentista", "renacentistas", "renaissance"), _RENACIMIENTO),
    **dict.fromkeys(("barroco", "barroca", "barrocos", "barrocas", "baroque"), _BARROCO),
    **dict.fromkeys(("rococo",), _ROCOCO),
    **dict.fromkeys(
        ("neoclasicismo", "neoclasico", "neoclasica", "neoclasicos", "neoclasicas", "neoclassical", "neoclassicism"),
        _NEOCLASICISMO,
    ),
    **dict.fromkeys(("romanticismo", "romanticism"), _ROMANTICISMO),
    **dict.fromkeys(("realismo", "realism"), _REALISMO),
    **dict.fromkeys(
        ("impresionismo", "impresionista", "impresionistas", "impressionism", "impressionist"),
        _IMPRESIONISMO,
    ),
    **dict.fromkeys(("manierismo", "manierista", "manieristas", "mannerism"), (1520, 1600)),
    **dict.fromkeys(("modernismo", "modernism"), (1860, 1940)),
    **dict.fromkeys(
        ("puntillismo", "puntillista", "neoimpresionismo", "pointillism"), (1880, 1905)
    ),
    **dict.fromkeys(("prerrafaelita", "prerrafaelitas", "pre-raphaelite", "pre raphaelite"), (1848, 1900)),
    **dict.fromkeys(("sintetismo", "sintetista", "synthetism", "synthetist"), (1888, 1900)),
}

# Épocas del dataset (salen de `culture_raw`, por ejemplo «Japan, Edo period (1615–1868)»).
# El índice solo conserva el primer componente de `culture` («Japan»), así que una época se
# resuelve como cultura ∩ rango de `year`. forma -> (culturas, desde, hasta). Las formas que
# son palabras comunes en inglés («song», «tang») piden «dinastía» o «dynasty».
_EDO, _MUROMACHI, _MOMOYAMA = (1615, 1868), (1392, 1573), (1573, 1615)
EPOCAS: Mapping[str, tuple[tuple[str, ...], int, int]] = {
    **dict.fromkeys(("edo", "periodo edo", "epoca edo"), (("Japan",), *_EDO)),
    **dict.fromkeys(("muromachi",), (("Japan",), *_MUROMACHI)),
    **dict.fromkeys(("momoyama",), (("Japan",), *_MOMOYAMA)),
    **dict.fromkeys(("kamakura",), (("Japan",), 1185, 1333)),
    **dict.fromkeys(("heian",), (("Japan",), 794, 1185)),
    **dict.fromkeys(("meiji",), (("Japan",), 1868, 1912)),
    **dict.fromkeys(("taisho",), (("Japan",), 1912, 1926)),
    **dict.fromkeys(("qing", "dinastia qing", "qing dynasty"), (("China",), 1644, 1911)),
    **dict.fromkeys(("ming", "dinastia ming", "ming dynasty"), (("China",), 1368, 1644)),
    **dict.fromkeys(("yuan", "dinastia yuan", "yuan dynasty"), (("China",), 1271, 1368)),
    **dict.fromkeys(("dinastia song", "song dynasty", "dinastia sung"), (("China",), 960, 1279)),
    **dict.fromkeys(("dinastia tang", "tang dynasty"), (("China",), 618, 907)),
    **dict.fromkeys(("joseon", "choson"), (("Korea",), 1392, 1910)),
    **dict.fromkeys(("goryeo", "koryo"), (("Korea",), 918, 1392)),
    **dict.fromkeys(("safavi", "safavid", "safavida"), (("Iran",), 1501, 1736)),
    **dict.fromkeys(
        ("escuela de barbizon", "barbizon"), (("France",), 1830, 1875)
    ),
    **dict.fromkeys(
        ("escuela del rio hudson", "hudson river", "hudson river school"),
        (("America", "United States"), 1825, 1880),
    ),
}
_ROMANOS = {"i": 1, "v": 5, "x": 10}

# Expresiones donde el gentilicio no nombra una cultura. Se quitan antes de buscar.
FALSOS_AMIGOS: tuple[str, ...] = ("tinta china",)


def _normalizar(texto: str) -> str:
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFKD", texto or "") if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", sin_tildes.lower()).strip()


def terminos_pedidos(pedido: str) -> tuple[str, ...]:
    """Los términos canónicos que el pedido nombra, en el orden de `ALIAS_DE_CULTURA`."""
    texto = _normalizar(pedido)
    for falso in FALSOS_AMIGOS:
        texto = texto.replace(falso, " ")
    return tuple(
        termino
        for termino, formas in ALIAS_DE_CULTURA.items()
        if any(re.search(rf"\b{re.escape(forma)}\b", texto) for forma in formas)
    )


def valores_que_cumplen(terminos: Sequence[str], vocabulario: Iterable[str]) -> tuple[str, ...]:
    """Los valores de `culture` del índice que satisfacen alguno de los términos.

    Un valor cumple si es el término o lo contiene como palabra entera: «India» cubre
    `Mughal India`; «Mughal» cubre solo `Mughal India`; «China» no cubre `Indochina`.
    """
    patrones = [re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE) for t in terminos]
    return tuple(
        sorted(v for v in set(vocabulario) if v and any(p.search(v) for p in patrones))
    )


def where_de(valores: Sequence[str]) -> dict:
    """El `where` de ChromaDB sobre `artworks`."""
    return {CAMPO: {"$in": list(valores)}}


def culturas_reconocidas(culturas_por_obra: Mapping[str, str]) -> list[tuple[str, int]]:
    """Culturas que activan el filtro y sus obras, de más a menos. Los alias compartidos van juntos."""
    vocabulario = set(culturas_por_obra.values())
    grupos: dict[tuple[str, ...], list[str]] = {}
    for termino, formas in ALIAS_DE_CULTURA.items():
        grupos.setdefault(formas, []).append(termino)
    filas = []
    for terminos in grupos.values():
        valores = set(valores_que_cumplen(terminos, vocabulario))
        n = sum(1 for cultura in culturas_por_obra.values() if cultura in valores)
        if n >= MÍNIMO_DE_OBRAS:
            filas.append((terminos[0], n))
    return sorted(filas, key=lambda f: (-f[1], f[0]))


def tecnicas_pedidas(pedido: str) -> tuple[str, ...]:
    """Las familias de técnica (`familia_de_tecnica`) que el pedido nombra."""
    texto = _normalizar(pedido)
    return tuple(
        familia
        for familias, formas in ALIAS_DE_TECNICA.items()
        if any(re.search(rf"\b{re.escape(f)}\b", texto) for f in formas)
        for familia in familias
    )


def _romano(texto: str) -> int | None:
    valores = [_ROMANOS.get(c) for c in texto.lower()]
    if not texto or None in valores:
        return None
    total = 0
    for actual, siguiente in zip(valores, valores[1:] + [0]):
        total += -actual if actual < siguiente else actual
    return total


def periodos_pedidos(pedido: str) -> tuple[tuple[int, int], ...]:
    """Los rangos de `year` que el pedido nombra: «siglo XVII», «17th century», «barroco»."""
    texto = _normalizar(pedido)
    siglos = [m for m in re.findall(r"\bsiglos?\s+([a-z0-9]+)\b", texto)]
    siglos += re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)\s+century\b", texto)
    siglos += re.findall(r"\b([ivx]+)\s+century\b", texto)
    rangos = []
    for token in siglos:
        n = int(token) if token.isdigit() else _romano(token)
        if n and 1 <= n <= 21:
            rangos.append(((n - 1) * 100, n * 100))
    # «post-impresionista» se quita antes para que no cuente también como impresionismo.
    if re.search(r"\bpost[- ]?impres", texto):
        rangos.append(_POSTIMPRESIONISMO)
        texto = re.sub(r"\bpost[- ]?impres\w*", " ", texto)
    for nombre, rango in MOVIMIENTOS.items():
        if re.search(rf"\b{re.escape(nombre)}\b", texto):
            rangos.append(rango)
    return tuple(dict.fromkeys(rangos))


def epocas_pedidas(pedido: str) -> tuple[tuple[tuple[str, ...], int, int], ...]:
    """Las épocas del dataset que el pedido nombra: (culturas, desde, hasta). «Edo» es Japón 1615-1868."""
    texto = _normalizar(pedido)
    return tuple(
        dict.fromkeys(
            epoca for forma, epoca in EPOCAS.items() if re.search(rf"\b{re.escape(forma)}\b", texto)
        )
    )


def artistas_pedidos(pedido: str, vocabulario: Iterable[str]) -> tuple[str, ...]:
    """Los valores de `artist` del índice que el pedido nombra.

    El nombre completo cuenta en cualquier parte. El apellido solo, precedido de «de», «del»,
    «por», «estilo» o «como», para que una palabra común que sea apellido no filtre.
    """
    texto = _normalizar(pedido)
    elegidos = []
    for artista in set(vocabulario):
        nombre = _normalizar(artista)
        if not nombre or nombre.startswith(("style of", "attributed", "master of")):
            continue
        apellido = nombre.split()[-1]
        if re.search(rf"\b{re.escape(nombre)}\b", texto) or (
            len(apellido) >= 4
            and re.search(rf"\b(?:de|del|por|estilo|como)\s+{re.escape(apellido)}\b", texto)
        ):
            elegidos.append(artista)
    return tuple(sorted(elegidos))
