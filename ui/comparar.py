"""Comparación de C0 a C4 sobre una misma consulta.

Sin Gradio ni red: app.py lo conecta y los tests lo importan por ruta de archivo.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

CONFIGURACIONES = ("C0", "C1", "C2", "C3", "C4")
NOMBRES = {
    "C0": "Solo imagen",
    "C1": "Pedido y evidencia citada",
    "C2": "Con planificador",
    "C3": "Con rama de texto",
    "C4": "Con reranker",
}
_ID_DE_COMPARACION = re.compile(r"[0-9a-f]{16}")
_ID_DE_CORRIDA = re.compile(r"[0-9a-f]{16}-\d+")


def correr_todas(ejecutor_de: Callable[[str], Any], entrada) -> Iterator[tuple[str, dict | None, str | None]]:
    """C0 a C4 en orden. Entrega (nombre, contenido, error) apenas termina cada una.

    Una excepción en una configuración no corta a las siguientes.
    """
    for nombre in CONFIGURACIONES:
        try:
            resultado = (ejecutor_de(nombre).correr(entrada).contenido, None)
        except Exception as e:
            resultado = (None, f"Falló: {type(e).__name__}: {e}")
        yield (nombre, *resultado)


def _id_de(ids: dict[str, str | None]) -> str:
    crudo = json.dumps([ids.get(c) for c in CONFIGURACIONES])
    return hashlib.sha256(crudo.encode()).hexdigest()[:16]


def guardar_indice(raiz: Path, ids: dict[str, str | None], fusion: str) -> str:
    """Escribe el índice de una comparación. Lleva ids de corrida y nada de la imagen ni del pedido."""
    comparacion_id = _id_de(ids)
    ruta = raiz / f"{comparacion_id}.json"
    if not ruta.exists():  # shortcut: escritura no atómica, pasar a AlmacenamientoEnArchivos si la UI deja de ser local
        raiz.mkdir(parents=True, exist_ok=True)
        contenido = {
            "comparacion_id": comparacion_id,
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fusion": fusion,
            "corridas": {c: ids.get(c) for c in CONFIGURACIONES},
        }
        ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8")
    return comparacion_id


def leer_indice(raiz: Path, comparacion_id: str) -> dict:
    if not _ID_DE_COMPARACION.fullmatch(comparacion_id or ""):
        raise ValueError(f"id de comparación inválido: {comparacion_id!r}")
    return json.loads((raiz / f"{comparacion_id}.json").read_text(encoding="utf-8"))


def listar_indices(raiz: Path) -> list[dict]:
    """Los índices legibles, el más nuevo primero. Un archivo roto se salta."""
    indices = []
    for ruta in raiz.glob("*.json") if raiz.exists() else []:
        try:
            indice = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(indice, dict) and "comparacion_id" in indice:
            indices.append(indice)
    return sorted(indices, key=lambda i: i.get("fecha", ""), reverse=True)


CSS = """
.tp3 .tp3-comp { display: grid; grid-template-columns: repeat(5, minmax(11rem, 1fr)); gap: 1rem;
  overflow-x: auto; padding-bottom: .5rem; }
.tp3 .tp3-ch { display: flex; flex-direction: column; padding-bottom: .4rem; border-bottom: 2px solid var(--tp-acento); }
.tp3 .tp3-ch b { font-size: 1.1rem; }
.tp3 .tp3-ch span { font-size: .85rem; color: var(--tp-mute); }
.tp3 .tp3-cobra { display: grid; grid-template-columns: 1.4rem minmax(0, 1fr); column-gap: .5rem;
  padding: .7rem 0; border-bottom: 1px solid var(--tp-line); }
.tp3 .tp3-cn { grid-row: 1 / span 4; font: italic 1.2rem/1.1 Georgia, "Palatino Linotype", serif; color: var(--tp-mute); }
.tp3 .tp3-cfoto { width: 100%; height: auto; display: block; border-radius: 2px; }
.tp3 .tp3-csinfoto { height: 6rem; display: flex; align-items: center; justify-content: center;
  border: 1px dashed var(--tp-line); color: var(--tp-mute); font-size: .85rem; }
.tp3 .tp3-ct { margin: .3rem 0 0; font: italic 1rem/1.25 Georgia, "Palatino Linotype", serif; }
.tp3 .tp3-ca { margin: 0; font-size: .85rem; color: var(--tp-mute); }
.tp3 .tp3-cnota { margin: .15rem 0 0; font-size: .85rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.tp3 .tp3-cdeg { margin: .4rem 0 0; font-size: .8rem; color: var(--tp-medio); }
.tp3 .tp3-cespera, .tp3 .tp3-cerror { margin: .7rem 0; font-size: .9rem; color: var(--tp-mute); }
.tp3 .tp3-cerror { color: var(--tp-mal); }
"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)  # texto de los museos y de un LLM


def _obra(o: dict, puntaje, miniatura: Callable[[str], str]) -> str:
    ficha = o.get("ficha") or {}
    titulo = ficha.get("title") or "Sin título"
    artista = ficha.get("artist")
    src = miniatura(o["artwork_id"])
    foto = (
        f'<img class="tp3-cfoto" src="{_esc(src)}" alt="{_esc(titulo)}" loading="lazy">'
        if src
        else '<div class="tp3-csinfoto">sin imagen</div>'
    )
    quien = _esc(artista) if artista else "Artista no registrado"
    nota = f'<p class="tp3-cnota">{_esc(puntaje)} de 10</p>' if puntaje is not None else ""
    return (
        f'<article class="tp3-cobra"><span class="tp3-cn">{_esc(o.get("rango", ""))}</span>{foto}'
        f'<p class="tp3-ct">{_esc(titulo)}</p><p class="tp3-ca">{quien}</p>{nota}</article>'
    )


def _cuerpo(valor: dict | str | None, miniatura: Callable[[str], str], corriendo: bool) -> str:
    if valor is None:
        return f'<p class="tp3-cespera">{"Corriendo…" if corriendo else "En espera"}</p>'
    if isinstance(valor, str):
        return f'<p class="tp3-cerror">{_esc(valor)}</p>'
    estado = valor.get("estado") or {}
    fallida = estado.get("fallida")
    if isinstance(fallida, dict):  # el pipeline marca {"etapa", "obligacion"} (marcar_fallida)
        motivo = f'en la etapa {_esc(fallida.get("etapa"))}: {_esc(fallida.get("obligacion"))}'
        return f'<p class="tp3-cerror">Falló la ejecución {motivo}</p>'
    if fallida:
        return f'<p class="tp3-cerror">Falló la ejecución: {_esc(fallida)}</p>'
    puntajes = {x["artwork_id"]: x.get("score_0_10") for x in estado.get("rerank") or []}
    obras = (estado.get("salida") or {}).get("obras") or []
    notas = "".join(
        f'<p class="tp3-cdeg">Se degradó la etapa {_esc(d.get("etapa"))}</p>' for d in estado.get("degradaciones") or []
    )
    return notas + "".join(_obra(o, puntajes.get(o["artwork_id"]), miniatura) for o in obras)


def render_grilla(
    columnas: dict[str, dict | str | None],
    miniatura: Callable[[str], str],
    en_curso: str | None = None,
) -> str:
    """Las cinco columnas. Cada valor es una corrida (dict), un error (str) o None si todavía no corrió."""
    cols = "".join(
        f'<section class="tp3-col"><header class="tp3-ch"><b>{c}</b><span>{_esc(NOMBRES[c])}</span></header>'
        f"{_cuerpo(columnas.get(c), miniatura, c == en_curso)}</section>"
        for c in CONFIGURACIONES
    )
    return f'<div class="tp3"><div class="tp3-comp">{cols}</div></div>'


def cargar_columnas(corridas: Path, indice: dict) -> dict[str, dict | str]:
    """Lee las corridas de un índice. Donde no hay corrida deja un texto que la grilla muestra como error."""
    columnas: dict[str, dict | str] = {}
    for nombre in CONFIGURACIONES:
        corrida_id = (indice.get("corridas") or {}).get(nombre)
        if not corrida_id:
            columnas[nombre] = "No corrió en esta comparación"
        elif not _ID_DE_CORRIDA.fullmatch(corrida_id):
            columnas[nombre] = "Ejecución no disponible"
        else:
            try:
                columnas[nombre] = json.loads((corridas / corrida_id / "corrida.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                columnas[nombre] = "Ejecución no disponible"
    return columnas
