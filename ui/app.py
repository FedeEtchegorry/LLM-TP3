"""Interfaz local para probar consultas.
Instalación: poetry install --with modelos,ui
Uso desde la raíz del repo: poetry run python ui/app.py  (http://127.0.0.1:7860, UI_PORT cambia el puerto).
Poetry crea el entorno en .venv (poetry.toml), así que .venv/Scripts/python ui/app.py hace lo mismo.
Arma el pipeline como `tp3-responder adhoc` y guarda la ejecución en artifacts/corridas.
Cada consulta gasta excepto C0.
"""
from __future__ import annotations

import atexit
import base64
import html
import io
import json
import os
import re
from functools import lru_cache, partial
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import gradio as gr

import comparar  # ui/ queda en sys.path al correr python ui/app.py
import clasificador
from tp3_rag.dominio.binding import BindingDelAclarador
from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.corrida import AlmacenDeCorridas
from tp3_rag.dominio.plan import AlmacenDePlanes
from tp3_rag.modulo_b import aclarador
from tp3_rag.modulo_b import confianza as conf
from tp3_rag.modulo_b import filtro_de_metadata as fm
from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.modulo_b.armador import ArmadorDePipeline
from tp3_rag.modulo_b.ejecutor import EjecutorDeEtapas
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos import trazas as trazas_mod
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.indice_vectorial import ARTWORKS

VARIANTE_ID = "c3274567371ff42b"  # variante de índice congelada
# C3 y C4 con filtro por metadata, así C4 es la final y C3 → C4 solo agrega el reranker
CAPAS = {
    "C0": "config/c0.toml",
    "C1": None,
    "C2": "config/c2.toml",
    "C3": "config/analisis/c3-filtro-metadata.toml",
    "C4": "config/analisis/c4-filtro-metadata.toml",
}
ROTULOS = {k: f"{k}  {v}" for k, v in comparar.NOMBRES.items()}
NOTAS = {
    "C0": "Usa solo CLIP local y no lee el pedido.",
    "C1": "Una llamada de generación con el pedido y la evidencia citada.",
    "C2": "Agrega el planificador.",
    "C3": "Agrega la rama de texto. Si el pedido nombra una cultura, busca solo entre sus obras.",
    "C4": "Agrega el reranker, que mira las imágenes de los candidatos.",
}
FUSIONES = [("Cascada: el texto filtra y CLIP ordena", "cascada"), ("RRF: mezcla los rankings de texto e imagen", "rrf")]
AVANZADAS = ("C3", "C4")  # tienen selector de fusión y piden confirmar el gasto
RRF = {"fusion.estrategia": "rrf", "recuperacion_textual.n_fragmentos_recuperados": "50"}

IMAGENES = RAÍZ_DEL_REPO / "data" / "imagenes"
CORRIDAS = RAÍZ_DEL_REPO / "artifacts" / "corridas"
UMBRAL_PASA, UMBRAL_BUENO = 5, 8  # colores del medidor de cada obra

COMPARACIONES = RAÍZ_DEL_REPO / "artifacts" / "comparaciones"

_compartido: dict = {}
_ejecutores: dict[tuple, EjecutorDeEtapas] = {}


def _montaje() -> tuple:
    if not _compartido:
        # el montaje de C4 sirve a las cinco capas
        cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / CAPAS["C4"])
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
        credenciales = Credenciales.desde_entorno()
        obs = Observabilidad(silencioso=True)
        obs.trazas = trazas_mod.armar(cfg, credenciales)
        m = mnt.montar(cfg, almacén, variante_id=VARIANTE_ID, observabilidad=obs, credenciales=credenciales)
        _compartido.update(m=m, obs=obs, almacén=almacén)
    return _compartido["m"], _compartido["obs"], _compartido["almacén"]


def _armar(nombre: str, fusion: str = "cascada"):
    fusion = fusion if (fusion == "rrf" and nombre in AVANZADAS) else "cascada"
    clave = (nombre, fusion)
    if clave not in _ejecutores:
        m, obs, almacén = _montaje()
        capa = CAPAS[nombre]
        overrides = dict(RRF) if fusion == "rrf" else {}
        cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / capa if capa else None, overrides=overrides)
        ArmadorDePipeline(cfg).validar()
        pipeline = ArmadorDePipeline(
            cfg,
            puerto_de_imagen=m.puerto_de_imagen,
            indice=m.indice,
            puerto_de_llm=m.puerto_de_llm,
            embebedor_de_texto=m.embebedor_de_texto,
            almacen_de_planes=AlmacenDePlanes(almacén, cfg["rutas.artefactos_planes"]),
            bytes_de_la_imagen=mnt.bytes_del_corpus(cfg, almacén),
            observabilidad=obs,
        ).armar()
        _ejecutores[clave] = EjecutorDeEtapas(
            cfg,
            pipeline=pipeline,
            almacen_de_corridas=AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"]),
            observabilidad=obs,
            variante_id=m.variante_id,
            split_id=m.split_id,
        )
    return _ejecutores[clave]


def _cerrar() -> None:
    if _compartido:
        _compartido["m"].cerrar()
        if hasattr(_compartido["obs"].trazas, "cerrar"):
            _compartido["obs"].trazas.cerrar()


atexit.register(_cerrar)


TEMA = gr.themes.Base(
    primary_hue="indigo",
    neutral_hue="slate",
    font=(gr.themes.Font("Segoe UI"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")),
)

CSS = """
.tp3 { --tp-ink: var(--body-text-color, #1b2130); --tp-mute: var(--body-text-color-subdued, #5a6273);
  --tp-line: var(--border-color-primary, #d3d7de); --tp-acento: var(--primary-600, #3f4fc9);
  --tp-ok: #1e7a4a; --tp-medio: #8a5a00; --tp-mal: #a82a2a; --tp-info: var(--tp-mute);
  color: var(--tp-ink); font-size: 15px; line-height: 1.5; }
.dark .tp3 { --tp-ok: #5fd39a; --tp-medio: #e5b454; --tp-mal: #f08a84; }
.tp3 a { color: var(--tp-acento); }
.tp3 a:focus-visible, .tp3 summary:focus-visible { outline: 2px solid var(--tp-acento); outline-offset: 2px; }

.tp3 .tp3-veredicto { --tono: var(--tp-info); padding: .8rem 1.1rem;
  margin: 0 0 .4rem; background: color-mix(in srgb, var(--tono) 10%, transparent); border-radius: 3px; }
.tp3 .tp3-veredicto.tp3-bueno { --tono: var(--tp-ok); }
.tp3 .tp3-veredicto.tp3-parcial { --tono: var(--tp-medio); }
.tp3 .tp3-veredicto.tp3-debil, .tp3 .tp3-veredicto.tp3-error { --tono: var(--tp-mal); }
.tp3 .tp3-vtitulo { margin: 0; font-size: 1.15rem; font-weight: 650; color: var(--tono); }
.tp3 .tp3-vdetalle { margin: .15rem 0 0; }
.tp3 .tp3-vnota { margin: .45rem 0 0; font-size: .86rem; color: var(--tp-mute); }
.tp3 .tp3-vlectura { margin: .45rem 0 0; font-size: .9rem; }
.tp3 .tp3-vlectura b, .tp3 .tp3-guia b { font-weight: 600; }
.tp3 .tp3-guia { margin: .7rem 0 0; padding-top: .6rem; border-top: 1px solid var(--tp-line); font-size: .9rem; }
.tp3 .tp3-gtitulo { margin: 0 0 .25rem; font-weight: 650; }
.tp3 .tp3-guia ul { margin: 0; padding-left: 1.1rem; }
.tp3 .tp3-guia li { margin: .2rem 0; }
.tp3 .tp3-meta { display: flex; flex-wrap: wrap; gap: .1rem 1.4rem; margin: .6rem 0 0; font-size: .86rem; }
.tp3 .tp3-meta span { color: var(--tp-mute); }
.tp3 .tp3-meta b { font-weight: 600; color: var(--tp-ink); }
.tp3 .tp3-vacio { margin: 0; padding: .9rem 0 1.4rem; color: var(--tp-mute); }
.tp3 .tp3-vacio b { font-weight: 600; color: var(--tp-ink); }

.tp3 .tp3-obra { display: grid; grid-template-columns: 2rem 168px minmax(0, 1fr) 17rem; column-gap: 1.4rem;
  padding: 1.15rem 0; border-top: 1px solid var(--tp-line); }
.tp3 .tp3-expl { grid-column: 2 / -1; margin-top: .9rem; }
.tp3 .tp3-num { font: italic 1.6rem/1.1 Georgia, "Palatino Linotype", serif; color: var(--tp-mute); }
.tp3 .tp3-foto { width: 168px; height: auto; display: block; border-radius: 2px; }
.tp3 .tp3-sinfoto { width: 168px; height: 112px; display: flex; align-items: center; justify-content: center;
  border: 1px dashed var(--tp-line); color: var(--tp-mute); font-size: .85rem; }
.tp3 .tp3-artista { margin: 0; font-weight: 600; }
.tp3 .tp3-artista.tp3-anonimo { font-weight: 400; color: var(--tp-mute); }
.tp3 .tp3-titulo { margin: .1rem 0 .15rem; font: italic 1.25rem/1.3 Georgia, "Palatino Linotype", serif; }
.tp3 .tp3-dato { margin: 0; font-size: .88rem; color: var(--tp-mute); }

.tp3 .tp3-ev { display: grid; grid-template-columns: 3.4rem minmax(2.5rem, 1fr) auto; gap: .3rem .6rem;
  align-content: start; align-items: center; margin: .2rem 0 0; font-size: .86rem; }
.tp3 .tp3-k { color: var(--tp-mute); }
.tp3 .tp3-v { white-space: nowrap; font-variant-numeric: tabular-nums; }
.tp3 .tp3-solo { grid-column: 2 / -1; }
.tp3 .tp3-pista { position: relative; height: 2px; background: var(--tp-line); }
.tp3 .tp3-punto { position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%;
  background: var(--tp-acento); transform: translate(-50%, -50%); }
.tp3 .tp3-punto.tp3-fuera { background: transparent; border: 2px solid var(--tp-acento); }
.tp3 .tp3-medidor { display: grid; grid-template-columns: repeat(10, 1fr); gap: 2px; }
.tp3 .tp3-medidor i { display: block; height: 9px; background: var(--tp-line); }
.tp3 .tp3-medidor.tp3-bueno i.on { background: var(--tp-ok); }
.tp3 .tp3-medidor.tp3-parcial i.on { background: var(--tp-medio); }
.tp3 .tp3-medidor.tp3-debil i.on { background: var(--tp-mal); }

.tp3 .tp3-linea { margin: .35rem 0; }
.tp3 .tp3-abst { margin: .35rem 0; font-style: italic; color: var(--tp-mute); }
.tp3 details { margin: .3rem 0; }
.tp3 summary { cursor: pointer; color: var(--tp-acento); font-size: .9rem; }
.tp3 .tp3-cita { margin: .5rem 0; }
.tp3 .tp3-cita p { margin: 0; font: 1rem/1.5 Georgia, "Palatino Linotype", serif; }
.tp3 .tp3-cita footer { margin-top: .2rem; font-size: .82rem; color: var(--tp-mute); }
.tp3 .tp3-just { margin: .4rem 0; font-size: .92rem; }

@media (max-width: 960px) {
  .tp3 .tp3-obra { grid-template-columns: 2rem 168px minmax(0, 1fr); }
  .tp3 .tp3-ev { grid-column: 3; margin-top: .8rem; max-width: 22rem; }
}

@media (max-width: 720px) {
  .tp3 .tp3-obra { grid-template-columns: 2rem minmax(0, 1fr); }
  .tp3 .tp3-fotoc, .tp3 .tp3-cab, .tp3 .tp3-ev { grid-column: 2; }
  .tp3 .tp3-cab { margin-top: .8rem; }
  .tp3 .tp3-expl { grid-column: 1 / -1; }
}

.tp3-judge-tools { align-items: center !important; gap: 10px !important; }
.tp3-judge-tools button { white-space: nowrap; height: 44px; }
.tp3-judge-layout { align-items: start !important; gap: 24px !important; }
.tp3-judge-reference { position: sticky; top: 16px; flex: 0 0 280px !important;
  border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px !important;
  background: var(--block-background-fill); gap: 12px !important; }
.tp3-judge-grid { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr));
  flex: 1 1 0 !important; min-width: 0 !important; gap: 16px !important; }
.tp3-judge-card { min-width: 0 !important; padding: 16px !important; gap: 12px !important;
  border: 1px solid var(--border-color-primary); border-radius: 12px;
  background: var(--block-background-fill); }
.tp3-judge-card.juzgada { border-color: var(--primary-400); }
.tp3-judge-status p { margin: 0 !important; font-size: 12px; letter-spacing: .06em;
  text-transform: uppercase; color: var(--body-text-color-subdued); }
.tp3-judge-caption { height: 64px; overflow: hidden; }
.tp3-judge-caption p { margin: 0 0 4px !important; }
.tp3-judge-caption p:first-child { display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden; line-height: 1.4; }
.tp3-judge-caption p + p { font-size: 13px; color: var(--body-text-color-subdued);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tp3-judge-grades .wrap { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px !important; }
.tp3-judge-grades label { margin: 0 !important; padding: 8px 4px !important; min-width: 0; white-space: nowrap; justify-content: center; }
.tp3-judge-grades label span { font-size: 12px; }
.tp3-judge-reference .prose p { overflow-wrap: anywhere; }
@media (max-width: 1350px) {
  .tp3-judge-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 960px) {
  .tp3-judge-reference { position: static; flex: 1 1 100% !important; }
  .tp3-judge-layout { flex-wrap: wrap !important; }
  .tp3-judge-grid { flex-basis: 100% !important; }
}
@media (max-width: 600px) {
  .tp3-judge-grid { grid-template-columns: minmax(0, 1fr); }
}
"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)  # Texto de un LLM y de los museos


def _miles(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def _n(x: float, decimales: int = 1) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


@lru_cache(maxsize=256)
def _miniatura(artwork_id: str) -> str:
    ruta = IMAGENES / (artwork_id.replace(":", "_") + ".jpg")
    if not ruta.exists():
        return ""
    from PIL import Image

    with Image.open(ruta) as im:
        im.thumbnail((340, 340))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _banda(p: float) -> str:
    return "tp3-bueno" if p >= UMBRAL_BUENO else "tp3-parcial" if p >= UMBRAL_PASA else "tp3-debil"


def _fila_rango(nombre: str, rango: int | None, largo: int) -> str:
    if rango:
        pos = (rango - 1) / max(largo - 1, 1) * 100
        punto, valor = f'<i class="tp3-punto" style="left:{pos:.1f}%"></i>', f"puesto {rango} de {largo}"
    else:
        punto, valor = '<i class="tp3-punto tp3-fuera" style="left:100%"></i>', f"fuera de los {largo}"
    return (
        f'<span class="tp3-k">{nombre}</span><span class="tp3-pista" role="img" aria-label="{nombre}: {valor}">{punto}</span>'
        f'<span class="tp3-v">{valor}</span>'
    )


def _obra(o: dict, rr: dict | None, rangos: dict, largos: dict, fusion: int | None) -> str:
    f, aid = o["ficha"], o["artwork_id"]
    titulo, artista = f.get("title") or "Sin título", f.get("artist")
    img = _miniatura(aid)
    foto = (
        f'<img class="tp3-foto" src="{img}" alt="{_esc(titulo)}{", " + _esc(artista) if artista else ""}" loading="lazy">'
        if img
        else '<div class="tp3-sinfoto">sin imagen</div>'
    )
    quien = (
        f'<p class="tp3-artista">{_esc(artista)}</p>'
        if artista
        else '<p class="tp3-artista tp3-anonimo">Artista no registrado</p>'
    )
    lugar = ", ".join(_esc(x) for x in (f.get("culture"), f.get("date_display")) if x)
    url = o.get("source_url") or ""
    ficha = (
        f' <a href="{_esc(url)}" target="_blank" rel="noopener">Ficha del museo</a>'
        if url.startswith(("https://", "http://"))
        else ""
    )
    soporte = (_esc(f["medium"]) if f.get("medium") else "") + ficha
    datos = "".join(f'<p class="tp3-dato">{x}</p>' for x in (lugar, soporte.strip()) if x)

    filas = []
    for rama, nombre in (("texto", "Texto"), ("visual", "Visual")):
        if rama in largos:
            filas.append(_fila_rango(nombre, rangos.get(rama), largos[rama]))
    if fusion:
        filas.append(f'<span class="tp3-k">Fusión</span><span class="tp3-v tp3-solo">puesto {fusion}</span>')
    if rr and rr.get("score_0_10") is not None:
        p = rr["score_0_10"]
        segmentos = "".join(f'<i class="{"on" if i < p else ""}"></i>' for i in range(10))
        filas.append(
            f'<span class="tp3-k">Reranker</span><span class="tp3-medidor {_banda(p)}" role="img" '
            f'aria-label="reranker: {p} de 10">{segmentos}</span><span class="tp3-v">{p} de 10</span>'
        )
    evidencia = f'<aside class="tp3-ev" aria-label="Dónde puso cada rama esta obra">{"".join(filas)}</aside>' if filas else ""

    explicacion = "".join(f'<p class="tp3-linea">{_esc(x)}</p>' for x in o.get("lineas") or [])
    if o.get("abstencion"):
        explicacion += f'<p class="tp3-abst">{_esc(o["abstencion"])}</p>'
    just = (
        f'<details><summary>Evaluación del reranker</summary><p class="tp3-just">{_esc(rr["justificacion"])}</p></details>'
        if rr and rr.get("justificacion")
        else ""
    )
    refs = ""
    for r in o.get("referencias") or []:
        cita = r.get("texto_citado") or ""
        cita = cita[:500] + ("…" if len(cita) > 500 else "")
        atribucion = f", {_esc(r['atribucion'])}" if r.get("atribucion") else ""
        refs += (
            f'<blockquote class="tp3-cita"><p>“{_esc(cita)}”</p>'
            f'<footer>Cita [{_esc(r.get("n", ""))}], licencia {_esc(r.get("license") or "sin dato")}{atribucion}</footer></blockquote>'
        )
    citas = f"<details><summary>Citas ({len(o.get('referencias') or [])})</summary>{refs}</details>" if refs else ""
    return (
        f'<article class="tp3-obra"><div class="tp3-num" aria-hidden="true">{_esc(o["rango"])}</div>'
        f'<div class="tp3-fotoc">{foto}</div><div class="tp3-cab">{quien}'
        f'<h3 class="tp3-titulo">{_esc(titulo)}</h3>{datos}</div>{evidencia}'
        f'<div class="tp3-expl">{explicacion}{just}{citas}</div></article>'
    )


def _veredicto(
    tono: str, titulo: str, detalle: str = "", nota: str = "", meta: str = "", lectura: str = "", guia: str = ""
) -> str:
    return (
        f'<div class="tp3"><div class="tp3-veredicto tp3-{tono}" role="status"><p class="tp3-vtitulo">{_esc(titulo)}</p>'
        + (f'<p class="tp3-vdetalle">{_esc(detalle)}</p>' if detalle else "")
        + (f'<p class="tp3-vnota">{_esc(nota)}</p>' if nota else "")
        + (f'<p class="tp3-vlectura"><b>Lectura del pedido.</b> {_esc(lectura)}</p>' if lectura else "")
        + guia
        + (f'<div class="tp3-meta">{meta}</div>' if meta else "")
        + "</div></div>"
    )


VACIO = '<div class="tp3"><p class="tp3-vacio"><b>Sin resultados.</b> Elegí un ejemplo o subí una pintura.</p></div>'


def _juicio(e: dict, c: conf.Confianza | None) -> tuple[str, str, str, str]:
    problemas = []
    if e.get("fallida"):
        problemas.append(f"Falló la ejecución: {e['fallida']}")
    for d in e.get("degradaciones") or []:
        problemas.append(f"Se degradó la etapa {d.get('etapa')}: {str(d.get('causa'))[:200]}")
    if c is None:
        return (
            "error" if problemas else "info",
            "Sin puntaje",
            "Solo C4 usa reranker.",
            " ".join(problemas),
        )
    if not c.baja:
        return "bueno", "Coincidencia buena", "", " ".join(problemas)
    detalle = (
        f"El reranker puntuó estas {c.n_obras} obras con un promedio de {_n(c.puntaje_medio)} sobre 10, "
        f"por debajo de {_n(conf.UMBRAL_DE_CONFIANZA, 0)}. Puede que el corpus no tenga obras que cumplan bien el pedido, "
        "o que el pedido necesite precisarse."
    )
    tono, titulo = ("debil", "Coincidencia débil") if c.puntaje_medio < UMBRAL_PASA else ("parcial", "Coincidencia parcial")
    return tono, titulo, detalle, " ".join(problemas)


def _lectura(e: dict) -> str:
    p = e.get("plan") or {}
    rotulos = (("Concepto buscado", "consulta_conceptual"), ("Aspecto visual buscado", "consulta_visual"))
    return " ".join(f"{rotulo}: «{p[clave]}»." for rotulo, clave in rotulos if p.get(clave))


@lru_cache(maxsize=1)
def _culturas_del_corpus() -> tuple[tuple[str, int], ...]:
    """Las culturas que activan el filtro, con sus obras. Vacío si no se puede leer el índice."""
    try:
        indice = _montaje()[0].indice
        fichas = indice.metadatas(ARTWORKS, list(indice.ids(ARTWORKS)))
        return tuple(fm.culturas_reconocidas({i: str(m.get(fm.CAMPO) or "") for i, m in fichas.items()}))
    except Exception:
        return ()


def _guia(e: dict, filtro_por_cultura: bool) -> str:
    """Cómo precisar un pedido con confianza baja."""
    items = []
    sello = (e.get("sellos") or {}).get("filtro_por_metadata") or {}
    if filtro_por_cultura and sello.get("aplicado"):
        nombres = ", ".join(_esc(t) for t in sello.get("terminos") or [])
        items.append(
            f"<li><b>Cultura.</b> El filtro por cultura se aplicó a {nombres} ({_miles(sello.get('n_obras', 0))} obras). "
            "Si la cultura no era lo que importaba, sacarla del pedido.</li>"
        )
    elif filtro_por_cultura and (lista := ", ".join(f"{_esc(nombre)} ({_miles(n)})" for nombre, n in _culturas_del_corpus())):
        items.append(
            "<li><b>Cultura.</b> El filtro por cultura no se aplicó. Nombrar una de estas culturas lo activa "
            f"(entre paréntesis, sus obras en el corpus): {lista}.</li>"
        )
    items.append(
        "<li><b>Aspecto a cambiar.</b> Nombrar qué debe cambiar respecto de la imagen: la paleta, la luz, "
        "la época, la técnica o el motivo.</li>"
    )
    return f'<div class="tp3-guia"><p class="tp3-gtitulo">Precisar el pedido</p><ul>{"".join(items)}</ul></div>'


def _plan(e: dict) -> str:
    p = e.get("plan")
    if not p:
        return "Sin planificador (solo C2, C3 y C4)."
    return f"**Consulta visual:** {p.get('consulta_visual')}\n\n**Consulta conceptual:** {p.get('consulta_conceptual')}"


@lru_cache(maxsize=1)
def _precios_de_lista() -> tuple[dict, dict]:
    p = tomllib.loads((RAÍZ_DEL_REPO / "config" / "default.toml").read_text(encoding="utf-8"))["proveedor"]
    return p["precio_usd_por_millon_de_tokens"], p["precio_usd_por_millon_de_tokens_de_salida"]


def _costo(c: dict) -> tuple[float | None, str, float | None]:
    r = c["configuracion_resuelta"]
    prov = r.get("proveedor") or {}
    hoy = _precios_de_lista()
    pin = {**hoy[0], **(prov.get("precio_usd_por_millon_de_tokens") or {})}
    pout = {**hoy[1], **(prov.get("precio_usd_por_millon_de_tokens_de_salida") or {})}
    completado = "precio_usd_por_millon_de_tokens_de_salida" not in prov
    modelos = {rol: b.get("modelo") for rol, b in (r.get("bindings") or {}).items()}
    modelos["embedding_texto"] = (r.get("embebido_texto") or {}).get("modelo")

    lat = c.get("latencia") or {}
    segs = lat.get("por_etapa") or {}
    mayor = c.get("libro_mayor_por_etapa") or {}
    con_tokens = {e for e, v in mayor.items() if v.get("tokens_in") or v.get("tokens_out")}
    lentas = set(sorted(segs, key=segs.get, reverse=True)[:3])
    filas, total, sin_precio = [], 0.0, False
    for etapa in sorted(con_tokens | lentas, key=lambda e: -segs.get(e, 0)):
        v = mayor.get(etapa, {})
        ti, to = v.get("tokens_in", 0), v.get("tokens_out", 0)
        if etapa in con_tokens:
            roles = [m for rol in v.get("roles", {}) if (m := modelos.get(rol))]
            m = roles[0] if len(roles) == 1 else None
            if m in pin and (m in pout or not to):
                usd = ti / 1e6 * pin[m] + to / 1e6 * pout.get(m, 0.0)
                total += usd
                usd_txt = _n(usd, 4)
            else:
                sin_precio, usd_txt = True, "sin precio"
            datos = f"{v.get('n_llamadas', 0)} | {_miles(ti)} | {_miles(to)} | {usd_txt}"
        else:
            datos = "- | - | - | -"
        filas.append(f"| {etapa} | {_n(segs[etapa]) if etapa in segs else '-'} | {datos} |")
    tabla = "\n".join(["| etapa | segundos | llamadas | tokens in | tokens out | USD |", "|---|---:|---:|---:|---:|---:|", *filas])
    if not con_tokens:
        tabla = "Sin llamadas pagas.\n\n" + tabla
    elif completado:
        tabla += "\n\n_Precio de salida no registrado. El USD usa la lista de `config/default.toml`._"
    return (total if con_tokens and not sin_precio else None), tabla, lat.get("segundos_de_la_corrida")


def render_corrida(c: dict) -> tuple[str, str, str, str]:
    e = c["estado"]
    obras = e["salida"]["obras"] if e.get("salida") else []
    rerank = {x["artwork_id"]: x for x in e.get("rerank") or []}
    fusionado = e.get("ranking_fusionado") or []
    de_fusion = {x["artwork_id"]: x for x in fusionado}
    por_rama = {k: {x["artwork_id"]: x.get("rank") for x in v} for k, v in (e.get("fuentes_de_ranking") or {}).items()}

    def rangos(aid: str) -> dict:
        origen = de_fusion.get(aid, {}).get("rangos_de_origen", {})
        return {k: r for k in ("texto", "visual") if (r := origen.get(k) or por_rama.get(k, {}).get(aid))}

    def pos_fusion(aid: str) -> int | None:
        if aid in rerank:
            return rerank[aid].get("rango_fusionado")
        return next((i + 1 for i, x in enumerate(fusionado) if x["artwork_id"] == aid), None)

    largos = {k: len(v) for k, v in por_rama.items()}
    cuerpo = "".join(
        _obra(o, rerank.get(o["artwork_id"]), rangos(o["artwork_id"]), largos, pos_fusion(o["artwork_id"])) for o in obras
    )
    total, costo, segundos = _costo(c)
    confianza = conf.confianza_del_top5(e.get("rerank"), [o["artwork_id"] for o in obras])
    filtro = bool((c["configuracion_resuelta"].get("recuperacion_textual") or {}).get("filtro_por_metadata"))
    tono, titulo, detalle, nota = _juicio(e, confianza)
    cadena = c["pipeline"].get("cadena") or []
    meta = [("Configuración", c["pipeline"]["configuracion"])]
    if "fusionador" in cadena:
        meta.append(("Fusión", (c["configuracion_resuelta"].get("fusion") or {}).get("estrategia", "")))
    if segundos:
        meta.append(("Tiempo", f"{_n(segundos)} s"))
    meta.append(("Costo", f"USD {_n(total, 4)}" if total is not None else "sin gasto"))
    meta.append(("Ejecución", c["corrida_id"]))
    meta_html = "".join(f"<div><span>{_esc(k)}</span> <b>{_esc(v)}</b></div>" for k, v in meta)
    veredicto = _veredicto(tono, titulo, detalle, nota, meta_html, lectura=_lectura(e), guia=_guia(e, filtro) if confianza and confianza.baja else "")
    return veredicto, f'<div class="tp3">{cuerpo}</div>', _plan(e), costo


def _aviso(titulo: str, detalle: str = "", tono: str = "info") -> tuple[str, str, str, str]:
    return _veredicto(tono, titulo, detalle), "", "", ""


def responder(config: str, fusion: str, imagen: str | None, pedido: str, confirmado: bool):
    if not imagen or not (pedido or "").strip():
        return _aviso("Falta la imagen o el pedido")
    if config in AVANZADAS and not confirmado:
        return _aviso(
            f"{config} requiere confirmación",
            'Marcá "Confirmo el gasto" y consultá de nuevo.',
            "parcial",
        )
    try:
        ejecutor = _armar(config, fusion)
        corrida = ejecutor.correr(
            EntradaDeConsulta.ad_hoc(datos_de_imagen=Path(imagen).read_bytes(), pedido=pedido.strip())
        )
        return render_corrida(corrida.contenido)
    except Exception as e:
        return _aviso("No se pudo consultar", f"{type(e).__name__}: {e}", "error")


def _aclarador() -> BindingDelAclarador:
    if "aclarador" not in _compartido:
        m, obs, _ = _montaje()
        cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / CAPAS["C4"])
        _compartido["aclarador"] = BindingDelAclarador.desde_configuración(cfg, puerto=m.puerto_de_llm, observabilidad=obs)
    return _compartido["aclarador"]


def consultar(config: str, fusion: str, imagen: str | None, pedido: str, confirmado: bool, sugerir: bool):
    """Si el pedido es vago, muestra una sugerencia en vez de buscar. Devuelve las salidas de `responder`, el panel y la sugerencia."""
    listo = imagen and (pedido or "").strip() and config != "C0" and (config not in AVANZADAS or confirmado)
    if sugerir and listo:
        try:
            a = aclarador.aclarar(_aclarador(), pedido=pedido.strip(), datos_de_imagen=Path(imagen).read_bytes())
        except Exception:
            a = None  # si el prefiltro falla, se busca con el pedido tal cual
        if a is not None and a.vago:
            detalle = f"{a.motivo} Se puede buscar con la versión sugerida, con una editada o con el pedido original."
            return (*_aviso("Pedido poco concreto", detalle, "parcial"), gr.update(visible=True), a.sugerencia)
    return (*responder(config, fusion, imagen, pedido, confirmado), gr.update(visible=False), gr.update())


@lru_cache(maxsize=4096)
def _etiqueta(ruta: Path, _mtime: float) -> tuple[str, str, int, str]:
    try:
        c = json.loads(ruta.read_text(encoding="utf-8"))
        e = c["estado"]
        cfg = ((c.get("configuracion_resuelta") or {}).get("consulta") or {}).get("configuracion", "?")
        origen = "set de evaluación" if (e.get("entrada") or {}).get("procedencia") != "ad_hoc" else "consulta libre"
        q = ((e.get("plan") or {}).get("consulta_conceptual") or "sin planificador").strip()
        obra = ((e.get("salida") or {}).get("obras") or [{}])[0].get("ficha", {}).get("title")
        dia, hora = ((c.get("fecha") or "")[:16] + "T").split("T")[:2]
        n = int(cfg[1:]) if re.fullmatch(r"C\d", cfg) else 99
        return dia, hora, n, f"{dia} {hora}, {cfg}, {origen}, {q[:70]}" + (f". Primera obra: {obra[:30]}" if obra else "")
    except Exception:
        return "", "", 99, f"{ruta.parent.name} (no se pudo leer)"


def _ids_de_corridas() -> list[tuple[str, str]]:
    filas = [(*_etiqueta(f, f.stat().st_mtime), f.parent.name) for f in CORRIDAS.glob("*/corrida.json")]
    filas.sort(key=lambda x: (x[0], -x[2], x[1]), reverse=True)  # día y hora descendentes, C ascendente
    return [(x[3], x[4]) for x in filas]


def abrir(corrida_id: str | None):
    if not corrida_id or not re.fullmatch(r"[0-9a-f]{16}-\d+", corrida_id):
        return _aviso("Elegí una ejecución")
    ruta = CORRIDAS / corrida_id / "corrida.json"
    if not ruta.exists():
        return _aviso("No existe la ejecución", corrida_id, "error")
    try:
        return render_corrida(json.loads(ruta.read_text(encoding="utf-8")))
    except Exception as e:
        return _aviso("No se pudo cargar la ejecución", f"{type(e).__name__}: {e}", "error")


def _consultas_del_set() -> dict[str, tuple[str, str]]:
    datos = tomllib.loads((RAÍZ_DEL_REPO / "consultas-g2.toml").read_text(encoding="utf-8"))
    return {
        f"{c['consulta_id']}  {c['pedido']}": (str(IMAGENES / (img["artwork_id"].replace(":", "_") + ".jpg")), c["pedido"])
        for img in datos["imagenes"]
        for c in img["consultas"]
    }


EJEMPLOS = _consultas_del_set()


def _cargar_ejemplo(clave: str | None):
    return EJEMPLOS.get(clave, (None, ""))


def _al_cambiar_config(c: str):
    return (
        NOTAS[c],
        gr.update(visible=c in AVANZADAS),
        gr.update(visible=c in AVANZADAS, value=False),
    )


def _panel():
    veredicto = gr.HTML(VACIO)
    obras = gr.HTML()
    with gr.Accordion("Análisis del pedido", open=False):
        plan = gr.Markdown()
    with gr.Accordion("Costo y tokens", open=False):
        costo = gr.Markdown()
    return veredicto, obras, plan, costo


def _aviso_grilla(titulo: str, detalle: str = "", tono: str = "info") -> str:
    return _veredicto(tono, titulo, detalle)


def _comparaciones() -> list[tuple[str, str]]:
    filas = []
    for ind in comparar.listar_indices(COMPARACIONES):
        rid = next((ind["corridas"][c] for c in ("C4", "C3", "C2", "C1", "C0") if (ind.get("corridas") or {}).get(c)), None)
        ruta = CORRIDAS / (rid or "-") / "corrida.json"
        etiqueta = _etiqueta(ruta, ruta.stat().st_mtime)[3] if rid and ruta.exists() else ind["comparacion_id"]
        filas.append((etiqueta, ind["comparacion_id"]))
    return filas


def comparar_todas(imagen: str | None, pedido: str, confirmado: bool):
    sin_cambio = gr.update()
    if not imagen or not (pedido or "").strip():
        yield _aviso_grilla("Falta la imagen o el pedido"), {}, sin_cambio, sin_cambio
        return
    if not confirmado:
        yield (
            _aviso_grilla("La comparación requiere confirmación", 'Marcá "Confirmo el gasto" y compará de nuevo.', "parcial"),
            {},
            sin_cambio,
            sin_cambio,
        )
        return
    columnas: dict = {c: None for c in comparar.CONFIGURACIONES}
    yield comparar.render_grilla(columnas, _miniatura, en_curso="C0"), {}, sin_cambio, sin_cambio
    try:
        _montaje()
        entrada = EntradaDeConsulta.ad_hoc(datos_de_imagen=Path(imagen).read_bytes(), pedido=pedido.strip())
    except Exception as e:
        yield _aviso_grilla("No se pudo comparar", f"{type(e).__name__}: {e}", "error"), {}, sin_cambio, sin_cambio
        return
    ids: dict = {}
    for i, (nombre, contenido, error) in enumerate(comparar.correr_todas(_armar, entrada), start=1):
        columnas[nombre] = contenido if contenido is not None else error
        if contenido is not None:
            ids[nombre] = contenido["corrida_id"]
        siguiente = comparar.CONFIGURACIONES[i] if i < len(comparar.CONFIGURACIONES) else None
        yield comparar.render_grilla(columnas, _miniatura, en_curso=siguiente), dict(ids), sin_cambio, sin_cambio
    comparacion_id = comparar.guardar_indice(COMPARACIONES, ids, "cascada") if ids else None
    yield (
        comparar.render_grilla(columnas, _miniatura),
        dict(ids),
        gr.update(value=next((c for c in reversed(comparar.CONFIGURACIONES) if c in ids), None)),
        gr.update(choices=_comparaciones(), value=comparacion_id),
    )


def detalle_de(nombre: str | None, ids: dict):
    if not ids:
        return "", "", "", ""
    if not nombre or not (ids or {}).get(nombre):
        return _aviso("Elegí una configuración que haya corrido")
    return abrir(ids[nombre])


def abrir_comparacion(comparacion_id: str | None):
    sin_cambio = gr.update()
    if not comparacion_id:
        return _aviso_grilla("Elegí una comparación"), {}, sin_cambio
    try:
        indice = comparar.leer_indice(COMPARACIONES, comparacion_id)
    except (OSError, ValueError) as e:
        return _aviso_grilla("No se pudo abrir la comparación", f"{type(e).__name__}: {e}", "error"), {}, sin_cambio
    columnas = comparar.cargar_columnas(CORRIDAS, indice)
    ids = {c: i for c, i in (indice.get("corridas") or {}).items() if i}
    return comparar.render_grilla(columnas, _miniatura), ids, gr.update(value=next((c for c in reversed(comparar.CONFIGURACIONES) if c in ids), None))


def cargar_clasificacion():
    try:
        pares, consultas, fichas, aviso = clasificador.cargar()
        datos = {"pares": pares, "consultas": consultas, "fichas": fichas}
        ids = list(dict.fromkeys(c for c, _ in pares))
        opciones = [(f"{i:02} · {consultas[c]['pedido']}", c) for i, c in enumerate(ids, 1)]
        return datos, gr.update(choices=opciones, value=ids[0] if ids else None), aviso
    except (OSError, ValueError, KeyError) as e:
        return {"pares": [], "consultas": {}, "fichas": {}}, gr.update(choices=[], value=None), f"No se pudo cargar la clasificación: {e}"


def ficha_clasificacion(f):
    url = f.get("source_url") or ""
    enlace = f' · [Ficha del museo]({url})' if url.startswith(("https://", "http://")) else ""
    ficha = " · ".join(str(f[k]) for k in ("title", "artist", "culture", "date_display", "medium") if f.get(k))
    return f"{ficha}{enlace}"


def guardar_juicio(clave, grado, nota, estado):
    estado = dict(estado or {})
    estado[clave] = {"grado": grado, "nota": nota or ""}
    return estado


def mover_consulta(consulta, datos, paso):
    consultas = list(dict.fromkeys(c for c, _ in datos["pares"]))
    if consulta not in consultas:
        return consulta
    return consultas[max(0, min(len(consultas) - 1, consultas.index(consulta) + paso))]


def exportar_juicios(datos, estado, anotador):
    ruta = clasificador.exportar(datos["pares"], estado or {}, anotador)
    if not ruta:
        raise gr.Error("Todavía no hay juicios para descargar.")
    return ruta


with gr.Blocks(title="TP3") as demo:
    gr.Markdown(
        "## Probar una consulta\n"
        "Al ingresar una imagen y un pedido se devuelven cinco obras del Art Institute of Chicago y del Cleveland Museum of Art, con explicación y citas."
    )
    with gr.Tab("Consultar"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                ejemplo = gr.Dropdown(list(EJEMPLOS), value=None, label="Ejemplo del set de evaluación")
                imagen = gr.Image(type="filepath", sources=["upload", "clipboard"], label="Pintura de referencia")
                pedido = gr.Textbox(label="Pedido", placeholder="versiones más cálidas de este paisaje")
                sugerir = gr.Checkbox(value=True, label="Sugerir un pedido más concreto si el mío es vago")
                config = gr.Radio([(ROTULOS[k], k) for k in CAPAS], value="C1", label="Configuración")
                nota = gr.Markdown(NOTAS["C1"])
                fusion = gr.Radio(FUSIONES, value="cascada", label="Combinación de texto e imagen", visible=False)
                confirma = gr.Checkbox(label="Confirmo el gasto", visible=False)
                boton = gr.Button("Consultar", variant="primary")
            with gr.Column(scale=2, min_width=420):
                with gr.Group(visible=False) as panel_aclara:
                    sugerencia = gr.Textbox(label="Pedido sugerido", info="Se puede editar antes de buscar.", lines=2)
                    with gr.Row():
                        usar = gr.Button("Buscar con el pedido sugerido", variant="primary")
                        mantener = gr.Button("Buscar con mi pedido")
                salidas = _panel()
        ejemplo.change(_cargar_ejemplo, ejemplo, [imagen, pedido])
        config.change(_al_cambiar_config, config, [nota, fusion, confirma])
        (
            boton.click(lambda: gr.update(interactive=False, value="Consultando…"), None, boton, queue=False)
            .then(
                consultar,
                [config, fusion, imagen, pedido, confirma, sugerir],
                [*salidas, panel_aclara, sugerencia],
                concurrency_id="pipeline",
                concurrency_limit=1,
            )
            .then(lambda: gr.update(interactive=True, value="Consultar"), None, boton, queue=False)
        )
        (
            usar.click(lambda texto: (gr.update(visible=False), texto), sugerencia, [panel_aclara, pedido], queue=False)
            .then(
                responder,
                [config, fusion, imagen, sugerencia, confirma],
                list(salidas),
                concurrency_id="pipeline",
                concurrency_limit=1,
            )
        )
        (
            mantener.click(lambda: gr.update(visible=False), None, panel_aclara, queue=False)
            .then(
                responder,
                [config, fusion, imagen, pedido, confirma],
                list(salidas),
                concurrency_id="pipeline",
                concurrency_limit=1,
            )
        )
    with gr.Tab("Comparar"):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                c_ejemplo = gr.Dropdown(list(EJEMPLOS), value=None, label="Ejemplo del set de evaluación")
                c_imagen = gr.Image(type="filepath", sources=["upload", "clipboard"], label="Pintura de referencia")
            with gr.Column(scale=2, min_width=420):
                c_pedido = gr.Textbox(label="Pedido", placeholder="versiones más cálidas de este paisaje")
                c_confirma = gr.Checkbox(label="Confirmo el gasto")
                c_boton = gr.Button("Comparar C0 a C4", variant="primary")
                with gr.Row():
                    c_guardadas = gr.Dropdown([], value=None, label="Comparación guardada", scale=3)
                    c_abrir = gr.Button("Abrir", scale=1)
        c_grilla = gr.HTML(VACIO)
        c_ids = gr.State({})
        c_sel = gr.Radio([(ROTULOS[k], k) for k in comparar.CONFIGURACIONES], value=None, label="Detalle de")
        c_detalle = _panel()
        c_ejemplo.change(_cargar_ejemplo, c_ejemplo, [c_imagen, c_pedido])
        (
            c_boton.click(
                lambda: (gr.update(interactive=False, value="Comparando…"), gr.update(value=None), "", "", "", ""),
                None,
                [c_boton, c_sel, *c_detalle],
                queue=False,
            )
            .then(
                comparar_todas,
                [c_imagen, c_pedido, c_confirma],
                [c_grilla, c_ids, c_sel, c_guardadas],
                concurrency_id="pipeline",
                concurrency_limit=1,
            )
            .then(detalle_de, [c_sel, c_ids], list(c_detalle))
            .then(lambda: gr.update(interactive=True, value="Comparar C0 a C4"), None, c_boton, queue=False)
        )
        c_sel.input(detalle_de, [c_sel, c_ids], list(c_detalle))
        c_abrir.click(abrir_comparacion, c_guardadas, [c_grilla, c_ids, c_sel]).then(
            detalle_de, [c_sel, c_ids], list(c_detalle)
        )
        demo.load(lambda: gr.update(choices=_comparaciones()), None, c_guardadas)
    with gr.Tab("Usar ejecución guardada"):
        elegida = gr.Dropdown([], value=None, label="Ejecución guardada")
        salidas_2 = _panel()
        elegida.input(abrir, elegida, list(salidas_2))
        demo.load(lambda: gr.update(choices=_ids_de_corridas()), None, elegida)
    with gr.Tab("Clasificar a mano") as clasificar_tab:
        datos_j = gr.State({"pares": [], "consultas": {}, "fichas": {}})
        estado_j = gr.BrowserState({}, storage_key="tp3-juicios-gradio-v1", secret="tp3-juicios-local-v1")
        with gr.Row(elem_classes="tp3-judge-tools"):
            consulta_j = gr.Dropdown(label="Imagen y pedido", show_label=False, container=False, choices=[], scale=4)
            anterior_j = gr.Button("← Anterior", scale=0, min_width=125)
            siguiente_j = gr.Button("Siguiente →", variant="primary", scale=0, min_width=125)
        with gr.Accordion("Anotador y descarga CSV", open=False):
            with gr.Row():
                anotador_j = gr.Textbox(label="Anotador")
                exportar_j = gr.Button("Preparar CSV", scale=0)
                archivo_j = gr.File(label="Descargar juicios", interactive=False)
            recuperar_j = gr.Button("Recuperar avance de la versión anterior", size="sm")
        aviso_j = gr.Markdown()

        # Gradio asigna ids globales al crear las tarjetas; serializar evita colisiones entre sesiones.
        @gr.render(inputs=[datos_j, consulta_j, estado_j], concurrency_limit=1, concurrency_id="juicios", show_progress="hidden")
        def mostrar_consulta(datos, consulta, estado):
            if not consulta or consulta not in datos["consultas"]:
                return
            c = datos["consultas"][consulta]
            obras = [o for q, o in datos["pares"] if q == consulta]
            estado = estado or {}
            hechos = sum(1 for o in obras if estado.get(f"{consulta}|{o}", {}).get("grado") in (0, 1, 2))
            gr.Markdown(f"**{hechos} de {len(obras)} evaluadas** · Los cambios se guardan en este navegador.")
            with gr.Row(elem_classes="tp3-judge-layout"):
                with gr.Column(min_width=260, elem_classes="tp3-judge-reference"):
                    gr.Markdown("Imagen original", elem_classes="tp3-judge-status")
                    ruta = IMAGENES / (c["imagen"].replace(":", "_") + ".jpg")
                    gr.Image(value=str(ruta) if ruta.is_file() else None, label="Imagen original", show_label=False,
                             interactive=False, height=260, buttons=["fullscreen"])
                    gr.Markdown(f"**Pedido**\n\n{_esc(c['pedido'])}")
                    gr.Markdown("**0** No cumple o es la imagen original.\n\n**1** Cumple parcialmente o hay dudas.\n\n**2** Cumple claramente el pedido.")
                with gr.Column(min_width=0, elem_classes="tp3-judge-grid"):
                    for obra in obras:
                        clave = f"{consulta}|{obra}"
                        juicio = estado.get(clave, {})
                        evaluada = juicio.get("grado") in (0, 1, 2)
                        with gr.Column(min_width=0, elem_classes=["tp3-judge-card", "juzgada" if evaluada else "pendiente"]):
                            gr.Markdown("Evaluada" if evaluada else "Pendiente", elem_classes="tp3-judge-status")
                            ruta = IMAGENES / (obra.replace(":", "_") + ".jpg")
                            gr.Image(value=str(ruta) if ruta.is_file() else None, label="Obra a evaluar", show_label=False,
                                     interactive=False, height=260, buttons=["fullscreen"])
                            f = datos["fichas"].get(obra, {})
                            gr.Markdown(f"**{_esc(f.get('title') or obra)}**\n\n{_esc(f.get('artist') or 'Artista no registrado')}",
                                        elem_classes="tp3-judge-caption")
                            grado = gr.Radio([("0 · No", 0), ("1 · Parcial", 1), ("2 · Sí", 2)],
                                             value=juicio.get("grado"), label="Evaluación", container=False,
                                             elem_classes="tp3-judge-grades", key=f"grado-{clave}")
                            nota = gr.Textbox(value=juicio.get("nota", ""), label="Nota opcional", placeholder="Añadir un comentario…", key=f"nota-{clave}")
                            with gr.Accordion("Ficha de la obra", open=False):
                                gr.Markdown(ficha_clasificacion(f) or obra)
                            guardar = partial(guardar_juicio, clave)
                            grado.input(guardar, [grado, nota, estado_j], estado_j, concurrency_id="juicios", concurrency_limit=1)
                            nota.blur(guardar, [grado, nota, estado_j], estado_j, concurrency_id="juicios", concurrency_limit=1)
        clasificar_tab.select(cargar_clasificacion, None, [datos_j, consulta_j, aviso_j])
        for boton_j, paso in ((anterior_j, -1), (siguiente_j, 1)):
            boton_j.click(lambda consulta, datos, paso=paso: mover_consulta(consulta, datos, paso),
                          [consulta_j, datos_j], consulta_j)
        recuperar_j.click(lambda estado: estado, estado_j, estado_j, js="""(estado) => {
            try { return {...JSON.parse(localStorage.getItem('planilla-juicios-v1') || '{}'), ...estado}; }
            catch (e) { return estado; }
        }""")
        exportar_j.click(exportar_juicios, [datos_j, estado_j, anotador_j], archivo_j)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("UI_PORT", 7860)), theme=TEMA, css=CSS + comparar.CSS
    )
