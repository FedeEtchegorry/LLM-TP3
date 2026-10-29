from __future__ import annotations

import html
import re
from typing import Any, Mapping

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico, prefijar
from tp3_rag.dominio.urls import url_de_ficha, url_de_imagen
from tp3_rag.nucleo.errores import ErrorDeConstrucción, LicenciaNoDeclarada

VERSIÓN_DEL_NORMALIZADOR = "1"

_ETIQUETA = re.compile(r"<[^>]+>")
_ESPACIOS = re.compile(r"[ \t]+")
_SALTOS = re.compile(r"\n{3,}")

_CALIFICADOR_FINAL = re.compile(r"\s*\([^()]*\)\s*$")

def parsear_artista(descripción: str | None) -> str | None:
    if not descripción:
        return None
    nombre = descripción.strip()
    anterior = None
    while nombre != anterior and "(" in nombre:
        anterior = nombre
        nombre = _CALIFICADOR_FINAL.sub("", nombre).strip()
    nombre = nombre.rstrip(",;").strip()
    return nombre or None

def html_a_texto(crudo: str) -> str:
    sin_etiquetas = _ETIQUETA.sub("", crudo.replace("<br>", "\n").replace("<br/>", "\n"))
    texto = html.unescape(sin_etiquetas)
    texto = _ESPACIOS.sub(" ", texto.replace("\r\n", "\n").replace("\r", "\n"))
    return _SALTOS.sub("\n\n", texto).strip()

def _texto_plano(crudo: Any) -> str:
    return str(crudo).strip()

def _lista_de_descripciones(crudo: Any) -> str:
    if not isinstance(crudo, list):
        return html_a_texto(str(crudo))
    entradas = sorted(
        (e for e in crudo if isinstance(e, Mapping) and e.get("description")),
        key=lambda e: e.get("sortorder") or 0,
    )
    return "\n".join(html_a_texto(str(e["description"])) for e in entradas).strip()

def _lista_de_citas(crudo: Any) -> str:
    if not isinstance(crudo, list):
        return html_a_texto(str(crudo))
    return "\n".join(
        html_a_texto(str(e["citation"]))
        for e in crudo
        if isinstance(e, Mapping) and e.get("citation")
    ).strip()

def _exhibiciones_cma(crudo: Any) -> str:
    if not isinstance(crudo, Mapping):
        return html_a_texto(str(crudo))
    partes: list[str] = []
    for grupo in ("current", "legacy", "past", "future"):
        for e in crudo.get(grupo) or []:
            if isinstance(e, Mapping) and e.get("description"):
                partes.append(html_a_texto(str(e["description"])))
            elif isinstance(e, Mapping) and e.get("title"):
                partes.append(html_a_texto(str(e["title"])))
    return "\n".join(partes).strip()

RENDERS = {
    "texto_plano": _texto_plano,
    "html_a_texto": lambda c: html_a_texto(str(c)),
    "lista_de_descripciones": _lista_de_descripciones,
    "lista_de_citas": _lista_de_citas,
    "exhibiciones_cma": _exhibiciones_cma,
}

def _leer(crudo: Mapping, campo: str) -> Any:
    valor: Any = crudo
    for parte in campo.split("."):
        if isinstance(valor, list):
            if not valor:
                return None
            valor = valor[0]
        if not isinstance(valor, Mapping):
            return None
        valor = valor.get(parte)
        if valor is None:
            return None
    return valor

class Normalizador:
    def __init__(self, cfg) -> None:
        self.tabla_de_licencias = cfg["ingesta.tabla_de_licencias"]
        self.campos_de_texto = cfg["ingesta.campos_de_texto"]
        self.campos_canonicos = cfg["ingesta.campos_canonicos"]
        self.media_license = cfg["ingesta.media_license"]
        self.urls = cfg["ingesta.urls"]
        self.version = VERSIÓN_DEL_NORMALIZADOR

    def _licencia_de(self, museo: str, section: str) -> str:
        tabla = self.tabla_de_licencias.get(museo) or {}
        if section not in tabla:
            raise LicenciaNoDeclarada(museo, section)
        return tabla[section]

    def url_de_imagen(self, museo: str, crudo: Mapping) -> str | None:
        return url_de_imagen(museo, crudo, self.urls)

    def normalizar(
        self, crudo: Mapping, museo: str, hash_de_imagen: str | None = None
    ) -> RegistroCanónico:
        mapeo = self.campos_canonicos.get(museo)
        if mapeo is None:
            raise ErrorDeConstrucción(f"no hay mapeo de campos canónicos para el museo {museo!r}")

        id_crudo = crudo.get("id")
        if id_crudo is None:
            raise ErrorDeConstrucción(f"un registro de {museo} llegó sin `id`")

        ficha = url_de_ficha(museo, crudo, self.urls)
        if not ficha:
            raise ErrorDeConstrucción(
                f"{prefijar(museo, id_crudo)} no resuelve `source_url`, que es lo que se cita"
            )

        procedencia: dict = {"source_url": [museo, mapeo.get("source_url", "url")]}

        def canónico(nombre: str, transformar=None):
            campo = mapeo.get(nombre)
            if not campo:
                return None
            valor = _leer(crudo, campo)
            if valor is None or (isinstance(valor, str) and not valor.strip()):
                return None
            procedencia[nombre] = [museo, campo]
            return transformar(valor) if transformar else valor

        title = canónico("title", lambda v: str(v).strip())
        artist = canónico("artist", parsear_artista)
        if artist is None:
            procedencia.pop("artist", None)
        date_display = canónico("date_display", lambda v: str(v).strip())
        year = canónico("year", lambda v: int(v) if str(v).lstrip("-").isdigit() else None)
        if year is None:
            procedencia.pop("year", None)
        medium = canónico("medium", lambda v: str(v).strip())
        style = canónico("style", lambda v: str(v).strip())

        culture_raw = None
        campo_culture = mapeo.get("culture")
        if campo_culture:
            bruto = _leer(crudo, campo_culture) if "." in campo_culture else crudo.get(campo_culture)
            if isinstance(bruto, (list, tuple)):
                valores = tuple(str(v) for v in bruto if v is not None and str(v).strip())
            elif bruto is not None and str(bruto).strip():
                valores = (str(bruto),)
            else:
                valores = ()
            if valores:
                culture_raw = valores
                procedencia["culture"] = [museo, campo_culture]

        if museo not in self.media_license:
            raise ErrorDeConstrucción(
                f"`ingesta.media_license.{museo}` no está declarada. Cubre imagen y "
                f"metadata, no el texto, y tampoco se asume."
            )
        media_license = self.media_license[museo]
        procedencia["media_license"] = [museo, "declarado en la configuración"]

        campos: list[CampoDeTexto] = []
        for section, regla in (self.campos_de_texto.get(museo) or {}).items():
            bruto = crudo.get(regla["origen"])
            if bruto is None or bruto == [] or bruto == {}:
                continue
            render = RENDERS.get(regla["render"])
            if render is None:
                raise ErrorDeConstrucción(
                    f"regla de render desconocida {regla['render']!r} para "
                    f"({museo}, {section}). Disponibles: {sorted(RENDERS)}"
                )
            texto = render(bruto)
            if not texto:
                continue
            campos.append(
                CampoDeTexto(
                    section=section,
                    texto=texto,
                    source_url=ficha,
                    license=self._licencia_de(museo, section),
                )
            )
            procedencia[f"campos_de_texto.{section}"] = [museo, regla["origen"]]

        imagen = None
        url_img = url_de_imagen(museo, crudo, self.urls)
        if url_img and hash_de_imagen:
            imagen = Imagen(url_img, hash_de_imagen)
            procedencia["imagen"] = [museo, (self.urls.get(museo) or {}).get("campo_de_imagen")]

        tipo = crudo.get(mapeo.get("type", "type")) or ""
        return RegistroCanónico(
            artwork_id=prefijar(museo, id_crudo),
            museum=museo,
            type=str(tipo),
            title=title,
            artist=artist,
            date_display=date_display,
            year=year,
            medium=medium,
            style=style,
            culture_raw=culture_raw,
            source_url=ficha,
            imagen=imagen,
            media_license=media_license,
            campos_de_texto=tuple(campos),
            procedencia=procedencia,
        )
