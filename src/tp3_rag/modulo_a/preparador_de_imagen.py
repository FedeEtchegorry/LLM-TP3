from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tp3_rag.nucleo.identidad import hash_de_bytes
from tp3_rag.puertos.cliente_http import RecursoNoEncontrado

Image.MAX_IMAGE_PIXELS = 200_000_000

MODOS_GRISES = ("L", "LA", "1", "I", "I;16", "I;16B", "F")

@dataclass(frozen=True)
class ImagenPreparada:
    artwork_id: str
    ok: bool
    motivo: str | None = None
    hash_bytes: str | None = None
    ruta: str | None = None
    bytes_descargados: int = 0
    modo_original: str | None = None
    es_gris: bool = False
    tiene_alfa: bool = False
    convertida_a_rgb: bool = False
    ancho: int | None = None
    alto: int | None = None
    desde_cache: bool = False

    def a_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class PreparadorDeImagen:
    def __init__(self, cliente, raíz_de_imágenes: Path, observabilidad=None) -> None:
        self.cliente = cliente
        self.raíz = Path(raíz_de_imágenes)
        self.obs = observabilidad

    def _ruta(self, artwork_id: str) -> Path:
        return self.raíz / f"{artwork_id.replace(':', '_')}.jpg"

    def preparar(self, artwork_id: str, url: str | None) -> ImagenPreparada:
        if not url:
            return ImagenPreparada(artwork_id, ok=False, motivo="sin_url_de_imagen")

        try:
            respuesta = self.cliente.get(url)
        except RecursoNoEncontrado:
            return ImagenPreparada(artwork_id, ok=False, motivo="http_404")
        except Exception as e:
            return ImagenPreparada(
                artwork_id, ok=False, motivo=f"descarga_fallida:{type(e).__name__}"
            )

        datos = respuesta.cuerpo
        if not datos:
            return ImagenPreparada(artwork_id, ok=False, motivo="cuerpo_vacio")

        try:
            with Image.open(io.BytesIO(datos)) as im:
                modo, ancho, alto = im.mode, im.width, im.height
                es_gris = modo in MODOS_GRISES
                tiene_alfa = "A" in modo or modo == "P" or "transparency" in im.info
                rgb = im.convert("RGB")
                rgb.load()
        except Exception as e:
            return ImagenPreparada(
                artwork_id,
                ok=False,
                motivo=f"falla_de_formato:{type(e).__name__}",
                bytes_descargados=len(datos),
            )

        ruta = self._ruta(artwork_id)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        if not ruta.exists():
            tmp = ruta.with_suffix(".jpg.parcial")
            tmp.write_bytes(datos)
            tmp.replace(ruta)

        return ImagenPreparada(
            artwork_id=artwork_id,
            ok=True,
            hash_bytes=hash_de_bytes(datos),
            ruta=str(ruta),
            bytes_descargados=len(datos),
            modo_original=modo,
            es_gris=es_gris,
            tiene_alfa=tiene_alfa,
            convertida_a_rgb=True,
            ancho=ancho,
            alto=alto,
            desde_cache=respuesta.desde_cache,
        )

    @staticmethod
    def a9c(preparadas) -> dict:
        ok = [p for p in preparadas if p.ok]
        fallas_de_formato = [p for p in preparadas if not p.ok and str(p.motivo).startswith("falla_de_formato")]
        return {
            "n_con_imagen_preparada": len(ok),
            "grises": sum(1 for p in ok if p.es_gris),
            "alfa_o_paleta": sum(1 for p in ok if p.tiene_alfa),
            "convertidas_a_rgb": sum(1 for p in ok if p.convertida_a_rgb),
            "fallas_de_conversion_de_formato": len(fallas_de_formato),
            "fallas_de_embedding_por_formato": None,
            "nota": (
                "`fallas_de_embedding_por_formato` es `null` a propósito: el "
                "EmbebedorDeImágenes no corre, así que esa mitad del conteo de conversiones de imagen todavía "
                "no es medible. Lo que verifica es que la conversión a RGB de las "
                f"{len(ok)} imágenes preparadas no falló "
                f"({len(fallas_de_formato)} fallas de formato)."
            ),
            "modos_originales": _conteo(p.modo_original for p in ok),
            "motivos_de_descarte": _conteo(p.motivo for p in preparadas if not p.ok),
        }

def _conteo(valores) -> dict:
    from collections import Counter

    return dict(sorted(Counter(v for v in valores if v).items()))
