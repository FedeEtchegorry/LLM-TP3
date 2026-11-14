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
