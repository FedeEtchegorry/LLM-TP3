from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDeConfiguración

LARGO_DE_ID = 16

def json_canónico(valor: Any) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def hash_completo(valor: Any) -> str:
    return hashlib.sha256(json_canónico(valor).encode("utf-8")).hexdigest()

def hash_corto(valor: Any) -> str:
    return hash_completo(valor)[:LARGO_DE_ID]

def hash_de_bytes(datos: bytes) -> str:
    return "sha256:" + hashlib.sha256(datos).hexdigest()

def hash_de_registros(registros: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for registro in sorted(registros, key=lambda r: r["artwork_id"]):
        h.update(json_canónico(registro).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def catalogo_id(
    *,
    museos: Sequence[str],
    tipos: Sequence[str],
    version_del_normalizador: str,
    registros: Iterable[dict],
) -> str:
    return hash_corto(
        {
            "museos": sorted(museos),
            "tipos": sorted(tipos),
            "version_del_normalizador": version_del_normalizador,
            "hash_de_registros": hash_de_registros(registros),
        }
    )

def split_id(
    *,
    criterio_de_inclusion: Mapping[str, Any],
    semilla: int,
    proporcion_held_out: float,
    definicion_de_estrato: Any,
    ids_corpus: Sequence[str],
    ids_held_out: Sequence[str],
) -> str:
    return hash_corto(
        {
            "criterio_de_inclusion": dict(sorted(criterio_de_inclusion.items())),
            "semilla": semilla,
            "proporcion_held_out": proporcion_held_out,
            "definicion_de_estrato": definicion_de_estrato,
            "ids_corpus": sorted(ids_corpus),
            "ids_held_out": sorted(ids_held_out),
        }
    )

def variante_id(
    *,
    catalogo_id: str,
    split_id: str,
    criterio_de_inclusion: Mapping[str, Any],
    checkpoint_de_clip: str,
    version_del_preprocesado: str,
    estrategia_de_chunking: Mapping[str, Any],
    campos_de_knowledge: Sequence[str],
    modelo_de_embedding_de_texto: str,
    criterio_de_culture: str,
) -> str:
    return hash_corto(
        {
            "catalogo_id": catalogo_id,
            "split_id": split_id,
            "criterio_de_inclusion": dict(sorted(criterio_de_inclusion.items())),
            "checkpoint_de_clip": checkpoint_de_clip,
            "version_del_preprocesado": version_del_preprocesado,
            "estrategia_de_chunking": dict(sorted(estrategia_de_chunking.items())),
            "campos_de_knowledge": sorted(campos_de_knowledge),
            "modelo_de_embedding_de_texto": modelo_de_embedding_de_texto,
            "criterio_de_culture": criterio_de_culture,
        }
    )

_ETIQUETA_DECLARADA = re.compile(r"^[a-z0-9]+(?:[-_.][a-z0-9]+)*$")
_HASH_CORTO = re.compile(r"^[0-9a-f]{16}$")

ETIQUETAS_DECLARADAS = ("autoría", "designación", "anotador")

def etiqueta_declarada(valor: str, *, campo: str = "autoría") -> str:
    if not isinstance(valor, str) or not _ETIQUETA_DECLARADA.match(valor):
        raise ErrorDeConfiguración(
            f"`{campo}` = {valor!r} no es una etiqueta declarada válida. Tiene que ser "
            f"minúsculas, dígitos y `-`/`_`/`.` (p. ej. `revision-1`, `piloto`, `v2`): va "
            f"en el nombre del artefacto, que es donde vive la identidad."
        )
    if _HASH_CORTO.match(valor):
        raise ErrorDeConfiguración(
            f"`{campo}` = {valor!r} tiene forma de `hash_corto` (16 hex). Una etiqueta de "
            f"clase 3 la asigna una persona: si tiene forma de hash, alguien la va a "
            f"derivar, y derivar la identidad de un ancla de trabajo humano descarta "
            f"trabajo que no se recalcula."
        )
    return valor

def parece_hash_corto(valor: str) -> bool:
    return bool(isinstance(valor, str) and _HASH_CORTO.match(valor))

def identidad_del_set_de_consultas(*, split_id: str, autoria: str) -> str:
    if not _HASH_CORTO.match(split_id or ""):
        raise ErrorDeConfiguración(
            f"`split_id` = {split_id!r} no tiene la forma de un id del proyecto "
            f"({LARGO_DE_ID} hex). La identidad del set lo lleva literal, así que un "
            f"split_id mal escrito produciría un artefacto que no ancla a ningún "
            f"manifiesto."
        )
    return f"{split_id}-{etiqueta_declarada(autoria, campo='autoría')}"

def partir_identidad_del_set(identidad: str) -> tuple[str, str]:
    split_id, _, autoria = (identidad or "").partition("-")
    if not autoria:
        raise ErrorDeConfiguración(
            f"{identidad!r} no tiene la forma `<split_id>-<autoría>`. Los dos "
            f"ingredientes van en el nombre: un set sin `autoría` en el nombre deja "
            f"dos sets distintos compartiendo archivo."
        )
    return split_id, autoria

def rubrica_id(designacion: str) -> str:
    return etiqueta_declarada(designacion, campo="designación")

def identidad_del_pool(*, split_id: str, designacion: str) -> str:
    if not _HASH_CORTO.match(split_id or ""):
        raise ErrorDeConfiguración(
            f"`split_id` = {split_id!r} no tiene la forma de un id del proyecto "
            f"({LARGO_DE_ID} hex). La identidad del pool lo lleva literal, así que un "
            f"`split_id` mal escrito produciría una orden de trabajo sobre un split que "
            f"no existe — y lo que cuelga de ella es anotación humana."
        )
    return f"{split_id}-{etiqueta_declarada(designacion, campo='designación')}"

def partir_identidad_del_pool(identidad: str) -> tuple[str, str]:
    split_id, _, designacion = (identidad or "").partition("-")
    if not designacion or not _HASH_CORTO.match(split_id):
        raise ErrorDeConfiguración(
            f"{identidad!r} no tiene la forma `<split_id>-<designación>`. Los dos "
            f"ingredientes van en el nombre: un pool sin designación en el nombre no "
            f"puede expresar el pooling incremental."
        )
    return split_id, designacion

def identidad_del_aporte(*, split_id: str, rubrica_id: str, anotador: str) -> str:
    if not _HASH_CORTO.match(split_id or ""):
        raise ErrorDeConfiguración(
            f"`split_id` = {split_id!r} no tiene la forma de un id del proyecto "
            f"({LARGO_DE_ID} hex). Es el ancla del aporte: los juicios de un split no "
            f"son los de otro."
        )
    return (
        f"{split_id}"
        f"-{etiqueta_declarada(rubrica_id, campo='designación')}"
        f"-{etiqueta_declarada(anotador, campo='anotador')}"
    )

def rango_de_seleccion(semilla: int, artwork_id: str) -> str:
    return hashlib.sha256(f"{semilla}|{artwork_id}".encode("utf-8")).hexdigest()

def clave_de_cache_de_embedding(
    *,
    artwork_id: str,
    hash_bytes: str,
    checkpoint: str,
    version_del_preprocesado: str,
) -> str:
    return hash_corto(
        {
            "artwork_id": artwork_id,
            "hash_bytes": hash_bytes,
            "checkpoint": checkpoint,
            "version_del_preprocesado": version_del_preprocesado,
        }
    )

def consulta_id_ad_hoc(*, imagen_hash: str, pedido: str) -> str:
    if not str(imagen_hash).startswith("sha256:"):
        raise ErrorDeConfiguración(
            f"`imagen_hash = {imagen_hash!r}` sin prefijo de algoritmo. Entra en el "
            f"`consulta_id`, así que un cambio de algoritmo tiene que decirlo en vez de "
            f"colisionar (es la regla de `hash_de_bytes`)."
        )
    return hash_corto({"imagen_hash": imagen_hash, "pedido": pedido})

NO_DETERMINAN_EL_PLAN: tuple[str, ...] = (
    "alfa",
    "n_candidatos",
    "fusion",
    "variante_id",
    "config_id",
    "consulta_id",
    "split_id",
    "commit",
)

def plan_id(
    *,
    imagen_hash: str,
    pedido: str,
    prompt_id: str,
    version_del_prompt: str,
    modelo: str,
    muestreo: Mapping[str, Any],
) -> str:
    return hash_corto(
        {
            "imagen_hash": imagen_hash,
            "pedido": pedido,
            "prompt_id": prompt_id,
            "version_del_prompt": str(version_del_prompt),
            "modelo": modelo,
            "muestreo": dict(sorted(dict(muestreo).items())),
        }
    )

def corrida_id(
    *, config_id: str, consulta_id: str, commit: str | None, ordinal: int
) -> str:
    base = hash_corto(
        {"config_id": config_id, "consulta_id": consulta_id, "commit": commit}
    )
    if int(ordinal) < 1:
        raise ErrorDeConfiguración(
            f"`ordinal = {ordinal}` y los ordinales de la clase 2 empiezan en 1: son "
            f"`1 + n` de las ejecuciones existentes, reservados con un create "
            f"exclusivo."
        )
    return f"{base}-{int(ordinal)}"

def partir_corrida_id(valor: str) -> tuple[str, int]:
    base, _, ordinal = (valor or "").rpartition("-")
    if not base or not _HASH_CORTO.match(base) or not ordinal.isdigit():
        raise ErrorDeConfiguración(
            f"{valor!r} no tiene la forma `<hash_corto>-<ordinal>`. Los dos ingredientes "
            f"van en el nombre, y el ordinal legible es lo que vuelve citable "
            f"\"la corrida 2 de esta configuración\"."
        )
    return base, int(ordinal)

def clave_de_cache_http(url: str, nombres_de_headers: Sequence[str]) -> str:
    return hash_completo({"url": url, "headers": sorted(nombres_de_headers)})
