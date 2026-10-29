from __future__ import annotations

import io
from dataclasses import dataclass, field

from tp3_rag.nucleo.errores import ErrorDeConstrucción

DIM_ESPERADA = 512
LÍMITE_DE_TOKENS_DE_CLIP = 77
TOLERANCIA_DE_NORMA = 1e-3

class FormatoNoSoportado(ErrorDeConstrucción):
    def __init__(self, artwork_id: str, causa: str) -> None:
        super().__init__(
            f"{artwork_id}: CLIP no pudo embeber la imagen ({causa}). el conteo de conversiones de imagen compromete "
            f"**0 fallas de embedding por formato**, así que esto no se cuenta y sigue: "
            f"la conversión a RGB es del `PreparadorDeImagen` y tendría que haberlo "
            f"atrapado antes."
        )
        self.artwork_id = artwork_id

@dataclass(frozen=True)
class Vectores:
    vectores: tuple[tuple[float, ...], ...]
    truncados: tuple[int, ...] = ()
    uso: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.vectores)

    def __len__(self) -> int:
        return len(self.vectores)

    def __getitem__(self, i):
        return self.vectores[i]

@dataclass(frozen=True)
class IdentidadDelModelo:
    checkpoint: str
    version_del_preprocesado: str
    dim: int

    def a_json(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "version_del_preprocesado": self.version_del_preprocesado,
            "dim": self.dim,
        }

class EmbeddingImagenCLIP:
    OPERACIONES = ("embeber_imagenes", "embeber_imagen", "embeber_texto", "embeber_textos")

    def __init__(
        self,
        *,
        checkpoint: str,
        version_del_preprocesado: str,
        threads: int | None = None,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        if threads:
            torch.set_num_threads(int(threads))
        self._modelo = CLIPModel.from_pretrained(checkpoint).eval()
        self._procesador = CLIPProcessor.from_pretrained(checkpoint, use_fast=True)
        self._identidad = IdentidadDelModelo(
            checkpoint=checkpoint,
            version_del_preprocesado=version_del_preprocesado,
            dim=DIM_ESPERADA,
        )

    def identidad(self) -> IdentidadDelModelo:
        return self._identidad

    def embeber_imagenes(self, items) -> Vectores:
        from PIL import Image

        imágenes = []
        for artwork_id, datos in items:
            try:
                with Image.open(io.BytesIO(datos)) as im:
                    imágenes.append(im.convert("RGB"))
            except Exception as e:
                raise FormatoNoSoportado(artwork_id, type(e).__name__) from None

        with self._torch.no_grad():
            pixeles = self._procesador(images=imágenes, return_tensors="pt")["pixel_values"]
            salida = self._modelo.get_image_features(pixel_values=pixeles)
            vectores = self._normalizar(_tensor_de(salida))
        return Vectores(vectores=vectores)

    def embeber_imagen(self, datos: bytes, *, artwork_id: str = "?") -> tuple[float, ...]:
        return self.embeber_imagenes([(artwork_id, datos)])[0]

    def embeber_textos(self, textos) -> Vectores:
        textos = list(textos)
        truncados = tuple(i for i, t in enumerate(textos) if self._excede(t))
        with self._torch.no_grad():
            entradas = self._procesador(
                text=textos, return_tensors="pt", padding=True, truncation=True
            )
            salida = self._modelo.get_text_features(
                input_ids=entradas["input_ids"], attention_mask=entradas["attention_mask"]
            )
            vectores = self._normalizar(_tensor_de(salida))
        return Vectores(vectores=vectores, truncados=truncados)

    def embeber_texto(self, texto: str) -> tuple[float, ...]:
        return self.embeber_textos([texto])[0]

    def _excede(self, texto: str) -> bool:
        ids = self._procesador.tokenizer(texto, truncation=False)["input_ids"]
        return len(ids) > LÍMITE_DE_TOKENS_DE_CLIP

    def _normalizar(self, tensor) -> tuple[tuple[float, ...], ...]:
        if tensor.shape[-1] != DIM_ESPERADA:
            raise ErrorDeConstrucción(
                f"el checkpoint {self._identidad.checkpoint!r} devolvió vectores de "
                f"{tensor.shape[-1]} dims y se esperaban {DIM_ESPERADA}. ⚠️ El caso típico "
                f"es haber leído `.last_hidden_state` (que es (N, 50, 768), "
                f"pre-proyección) en vez de `.pooler_output`."
            )
        normas = tensor.norm(dim=-1, keepdim=True)
        if bool((normas <= 0).any()):
            raise ErrorDeConstrucción(
                "la verificación de la norma de los vectores: un vector tiene norma 0, así que normalizarlo daría `nan`. Un `nan` "
                "en el índice **no falla, ordena mal**, que es peor."
            )
        unitarios = tensor / normas
        peor = float((unitarios.norm(dim=-1) - 1.0).abs().max())
        if peor > TOLERANCIA_DE_NORMA:
            raise ErrorDeConstrucción(
                f"la verificación de la norma de los vectores: después de normalizar, la norma se desvía {peor:.2e} de 1. Es "
                f"prerrequisito geométrico de `α·e_img + (1−α)·e_txt`, no una prolijidad."
            )
        return tuple(tuple(float(x) for x in fila) for fila in unitarios)

def _tensor_de(salida):
    if hasattr(salida, "pooler_output"):
        return salida.pooler_output
    if hasattr(salida, "shape"):
        return salida
    raise ErrorDeConstrucción(
        f"`get_*_features` devolvió {type(salida).__name__}, que no tiene "
        f"`pooler_output` ni `shape`. Ver el diseño de la arquitectura: la API cambió entre "
        f"`transformers` 4.x (la Clase 9) y 5.x."
    )

class EmbeddingImagenFijo:
    OPERACIONES = ("embeber_imagenes", "embeber_imagen", "embeber_texto", "embeber_textos")

    def __init__(self, *, checkpoint: str = "fijo-para-tests", version_del_preprocesado: str = "1") -> None:
        self._identidad = IdentidadDelModelo(checkpoint, version_del_preprocesado, DIM_ESPERADA)
        self.n_llamadas = 0

    def identidad(self) -> IdentidadDelModelo:
        return self._identidad

    def embeber_imagenes(self, items) -> Vectores:
        self.n_llamadas += 1
        return Vectores(tuple(_vector_fijo(datos) for _, datos in items))

    def embeber_imagen(self, datos: bytes, *, artwork_id: str = "?") -> tuple[float, ...]:
        return self.embeber_imagenes([(artwork_id, datos)])[0]

    def embeber_textos(self, textos) -> Vectores:
        self.n_llamadas += 1
        textos = list(textos)
        return Vectores(
            vectores=tuple(_vector_fijo(t.encode("utf-8")) for t in textos),
            truncados=tuple(i for i, t in enumerate(textos) if len(t.split()) > LÍMITE_DE_TOKENS_DE_CLIP),
        )

    def embeber_texto(self, texto: str) -> tuple[float, ...]:
        return self.embeber_textos([texto])[0]

def _vector_fijo(datos: bytes) -> tuple[float, ...]:
    import hashlib

    semilla = hashlib.sha256(datos).digest()
    crudo = [((semilla[i % len(semilla)] * (i + 7)) % 509) - 254 for i in range(DIM_ESPERADA)]
    norma = sum(x * x for x in crudo) ** 0.5 or 1.0
    return tuple(x / norma for x in crudo)
