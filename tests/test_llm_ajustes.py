"""El puerto aprende de los 400 por parametros no soportados (modelos de razonamiento)."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from tp3_rag.puertos.llm import LLMOpenAI


def _400(mensaje: str) -> openai.BadRequestError:
    req = httpx.Request("POST", "http://proxy/v1/chat/completions")
    return openai.BadRequestError(mensaje, response=httpx.Response(400, request=req), body=None)


class _Completions:
    def __init__(self, rechazos: list[str]):
        self.rechazos = list(rechazos)
        self.cuerpos: list[dict] = []

    def create(self, **cuerpo):
        self.cuerpos.append(dict(cuerpo))
        if self.rechazos:
            parametro = self.rechazos[0]
            if parametro in cuerpo:
                mensajes = {
                    "max_tokens": "Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead.",
                    "temperature": "Unsupported value: 'temperature' does not support 0.0 with this model.",
                    "seed": "Unrecognized request argument supplied: 'seed'",
                }
                self.rechazos.pop(0)
                raise _400(mensajes[parametro])
        eleccion = SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='{"puntaje": 7}')
        )
        return SimpleNamespace(
            choices=[eleccion],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model="gpt-6-luna",
        )


def _puerto(rechazos: list[str]) -> tuple[LLMOpenAI, _Completions]:
    puerto = object.__new__(LLMOpenAI)
    completions = _Completions(rechazos)
    puerto._cliente = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    puerto.base_url = "https://api.openai.com/v1/"
    puerto.backoff_base = 0.0
    puerto._ajustes = {}
    return puerto, completions


def _binding():
    muestreo = SimpleNamespace(
        a_json=lambda: {"temperature": 0.0, "top_p": 1.0, "max_tokens": 200, "seed": 7369}
    )
    return SimpleNamespace(
        rol="reranker", modelo="openai/gpt-6-luna", muestreo=muestreo,
        esquema_de_salida="json", reintentos=1,
    )


MENSAJES = [{"rol": "usuario", "texto": "hola"}]


def test_renombra_max_tokens_y_lo_deja_registrado():
    puerto, completions = _puerto(["max_tokens"])
    r = puerto.invocar(_binding(), MENSAJES)
    assert "max_tokens" not in completions.cuerpos[-1]
    assert completions.cuerpos[-1]["max_completion_tokens"] == 2000
    assert r.muestreo_efectivo["max_completion_tokens"] == 2000
    assert "max_tokens" not in r.muestreo_efectivo


def test_aprende_varios_parametros_en_la_misma_llamada():
    puerto, completions = _puerto(["max_tokens", "temperature", "seed"])
    r = puerto.invocar(_binding(), MENSAJES)
    ultimo = completions.cuerpos[-1]
    assert "temperature" not in ultimo and "seed" not in ultimo and "max_tokens" not in ultimo
    assert r.uso["tokens_in"] == 10


def test_recuerda_el_ajuste_y_no_vuelve_a_fallar():
    puerto, completions = _puerto(["max_tokens"])
    puerto.invocar(_binding(), MENSAJES)
    antes = len(completions.cuerpos)
    puerto.invocar(_binding(), MENSAJES)
    assert len(completions.cuerpos) == antes + 1  # una sola llamada, sin 400 previo
    assert "max_tokens" not in completions.cuerpos[-1]


def test_un_400_que_no_es_de_parametros_se_propaga():
    puerto, completions = _puerto([])

    def falla(**cuerpo):
        raise _400("Invalid image")

    completions.create = falla
    with pytest.raises(Exception) as e:
        puerto.invocar(_binding(), MENSAJES)
    assert "Invalid image" in str(e.value)
