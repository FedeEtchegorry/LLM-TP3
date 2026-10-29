from __future__ import annotations

import time
from dataclasses import dataclass, field

from tp3_rag.nucleo.errores import ErrorDeConstrucción

TOLERANCIA_DE_NORMA = 1e-3
DIMS_CONOCIDAS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

def dim_conocida(modelo: str) -> int:
    return DIMS_CONOCIDAS.get(modelo) or DIMS_CONOCIDAS.get(modelo.rpartition("/")[2], 0)

class InputDemasiadoLargo(ErrorDeConstrucción):
    def __init__(self, etiqueta: str, n_tokens: int, limite: int) -> None:
        super().__init__(
            f"{etiqueta}: el texto mide {n_tokens} tokens y el modelo acepta {limite}. "
            f"el contrato de los puertos adjudica este error al **`Fragmentador`**: es su `tamano_objetivo` el "
            f"que tiene que garantizar que ningún fragmento lo pase, y por eso se "
            f"detecta antes de gastar la llamada. Falla la variante entera y no se "
            f"registra (las clases de falla, clase 2)."
        )
        self.etiqueta = etiqueta

class RedCaída(ErrorDeConstrucción):
    pass

@dataclass(frozen=True)
class VectoresDeTexto:
    vectores: tuple[tuple[float, ...], ...]
    uso: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.vectores)

    def __len__(self) -> int:
        return len(self.vectores)

    def __getitem__(self, i):
        return self.vectores[i]

@dataclass(frozen=True)
class IdentidadDelModeloDeTexto:
    modelo: str
    dim: int

    def a_json(self) -> dict:
        return {"modelo": self.modelo, "dim": self.dim}

def normalizar(vector, *, etiqueta: str) -> tuple[float, ...]:
    norma = sum(float(x) * float(x) for x in vector) ** 0.5
    if norma <= 0:
        raise ErrorDeConstrucción(
            f"la verificación de la norma de los vectores: el vector de {etiqueta} tiene norma 0, así que normalizarlo daría "
            f"`nan`. Un `nan` en el índice **no falla, ordena mal**, que es peor."
        )
    unitario = tuple(float(x) / norma for x in vector)
    peor = abs(sum(x * x for x in unitario) ** 0.5 - 1.0)
    if peor > TOLERANCIA_DE_NORMA:  # pragma: no cover - aritmética, no entrada
        raise ErrorDeConstrucción(
            f"la verificación de la norma de los vectores: después de normalizar, la norma de {etiqueta} se desvía {peor:.2e} de 1."
        )
    return unitario

class EmbeddingTextoOpenAI:
    OPERACIONES = ("embeber", "embeber_uno")

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        modelo: str = "openai/text-embedding-3-small",
        dim: int | None = None,
        limite_de_tokens: int = 8191,
        reintentos: int = 4,
        backoff_base: float = 1.5,
        medidor=None,
    ) -> None:
        from openai import OpenAI

        if not base_url:
            raise ErrorDeConstrucción(
                "`base_url` es obligatorio. Sin él el SDK apunta a "
                "`api.openai.com` y la key de la cátedra —una *virtual key* del proxy "
                "LiteLLM de ITBA— devuelve **401**. Es parámetro y no default porque el "
                "host es un punto de variación declarado en `config/default.toml` "
                "(`proveedor.base_url`): el día que la cátedra mueva el proxy, moverlo "
                "no puede ser un cambio de código."
            )
        self._cliente = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self._identidad = IdentidadDelModeloDeTexto(
            modelo=modelo, dim=int(dim or dim_conocida(modelo))
        )
        self.limite_de_tokens = int(limite_de_tokens)
        self.reintentos = int(reintentos)
        self.backoff_base = float(backoff_base)
        self._medidor = medidor

    def identidad(self) -> IdentidadDelModeloDeTexto:
        return self._identidad

    def embeber(self, textos, *, etiquetas=None) -> VectoresDeTexto:
        textos = list(textos)
        if not textos:
            return VectoresDeTexto(vectores=())
        etiquetas = list(etiquetas or [f"item[{i}]" for i in range(len(textos))])
        self._verificar_largos(textos, etiquetas)

        respuesta = self._llamar(textos)
        vectores = tuple(
            normalizar(d.embedding, etiqueta=etiquetas[i])
            for i, d in enumerate(sorted(respuesta.data, key=lambda d: d.index))
        )
        self._verificar_dim(vectores)
        uso = getattr(respuesta, "usage", None)
        return VectoresDeTexto(
            vectores=vectores,
            uso={"tokens_in": getattr(uso, "prompt_tokens", 0) or 0, "tokens_out": 0},
        )

    def embeber_uno(self, texto: str, *, etiqueta: str = "consulta") -> tuple[float, ...]:
        return self.embeber([texto], etiquetas=[etiqueta])[0]

    def _verificar_largos(self, textos, etiquetas) -> None:
        if self._medidor is None:
            return
        for texto, etiqueta in zip(textos, etiquetas):
            n = self._medidor(texto)
            if n > self.limite_de_tokens:
                raise InputDemasiadoLargo(etiqueta, n, self.limite_de_tokens)

    def _verificar_dim(self, vectores) -> None:
        if not vectores or not self._identidad.dim:
            return
        if len(vectores[0]) != self._identidad.dim:
            raise ErrorDeConstrucción(
                f"el modelo {self._identidad.modelo!r} devolvió vectores de "
                f"{len(vectores[0])} dims y la configuración declara "
                f"{self._identidad.dim}. Una colección ya materializada no acepta otra "
                f"dimensión: el síntoma sería un error de Chroma a mitad del upsert y no "
                f"un modelo mal declarado (`embebido_texto.dim`)."
            )

    def _llamar(self, textos):
        from openai import APIConnectionError, APIStatusError, RateLimitError

        for intento in range(self.reintentos + 1):
            try:
                return self._cliente.embeddings.create(
                    model=self._identidad.modelo, input=textos
                )
            except RateLimitError:
                if intento >= self.reintentos:
                    raise
                time.sleep(self.backoff_base ** (intento + 1))
            except APIConnectionError as e:
                raise RedCaída(
                    f"no se pudo llegar a la API de embeddings de texto ({e}). En el "
                    f"**indexado** esto falla la variante: no se registra nada, que es "
                    f"la política de la clase 2. La omisión de la omisión de etapas en runtime es de la "
                    f"consulta y la decide el `EjecutorDeEtapas`, no este puerto."
                ) from None
            except APIStatusError as e:
                if getattr(e, "status_code", None) == 400:
                    raise ErrorDeConstrucción(
                        f"la API rechazó el batch con 400 ({e}). el contrato de los puertos adjudica el caso de "
                        f"input largo al **`Fragmentador`**; inyectá el medidor del "
                        f"fragmentador en este puerto para detectarlo antes de llamar."
                    ) from None
                if getattr(e, "status_code", 500) >= 500 and intento < self.reintentos:
                    time.sleep(self.backoff_base ** (intento + 1))
                    continue
                raise
        raise ErrorDeConstrucción("se agotaron los reintentos del puerto de texto")

class EmbeddingTextoFijo:
    OPERACIONES = ("embeber", "embeber_uno")

    def __init__(
        self,
        *,
        modelo: str = "fijo-para-tests",
        dim: int = 1536,
        limite_de_tokens: int = 8191,
        medidor=None,
    ) -> None:
        self._identidad = IdentidadDelModeloDeTexto(modelo=modelo, dim=dim)
        self.limite_de_tokens = int(limite_de_tokens)
        self._medidor = medidor
        self.n_llamadas = 0
        self.n_items = 0

    def identidad(self) -> IdentidadDelModeloDeTexto:
        return self._identidad

    def embeber(self, textos, *, etiquetas=None) -> VectoresDeTexto:
        textos = list(textos)
        etiquetas = list(etiquetas or [f"item[{i}]" for i in range(len(textos))])
        if self._medidor is not None:
            for texto, etiqueta in zip(textos, etiquetas):
                n = self._medidor(texto)
                if n > self.limite_de_tokens:
                    raise InputDemasiadoLargo(etiqueta, n, self.limite_de_tokens)
        self.n_llamadas += 1
        self.n_items += len(textos)
        return VectoresDeTexto(
            vectores=tuple(_vector_fijo(t, self._identidad.dim) for t in textos),
            uso={"tokens_in": sum(len(t.split()) for t in textos), "tokens_out": 0},
        )

    def embeber_uno(self, texto: str, *, etiqueta: str = "consulta") -> tuple[float, ...]:
        return self.embeber([texto], etiquetas=[etiqueta])[0]

def _vector_fijo(texto: str, dim: int) -> tuple[float, ...]:
    import hashlib

    semilla = hashlib.sha256(texto.encode("utf-8")).digest()
    crudo = [((semilla[i % len(semilla)] * (i + 11)) % 521) - 260 for i in range(dim)]
    return normalizar(crudo, etiqueta="fijo")
