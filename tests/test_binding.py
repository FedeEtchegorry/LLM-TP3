from __future__ import annotations

import pytest

from tp3_rag.dominio.binding import (
    BindingDelGenerador,
    Muestreo,
)
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.llm import EXPLICACIÓN_VÁLIDA, LLMGrabado

pytestmark = pytest.mark.nivel1

@pytest.fixture(scope="module")
def cfg_b() -> Configuración:
    return Configuración.cargar()

def test_bd1_el_puerto_no_es_alcanzable_desde_la_etapa(cfg_b):
    binding = BindingDelGenerador.desde_configuración(cfg_b, puerto=LLMGrabado({}))
    públicos = [a for a in dir(binding) if not a.startswith("_")]
    assert not any("puerto" in a for a in públicos), públicos
    assert "prompt_id" in binding.identidad()

def test_bd1_un_binding_sin_puerto_no_puede_llamar(cfg_b):
    binding = BindingDelGenerador.desde_configuración(cfg_b)
    with pytest.raises(ErrorDeConfiguración, match="no tiene puerto"):
        binding.invocar([{"rol": "usuario", "texto": "hola"}])

def test_bd4_la_llamada_se_atribuye_al_rol_no_a_la_etapa(cfg_b):
    obs = Observabilidad(silencioso=True)
    binding = BindingDelGenerador.desde_configuración(
        cfg_b, puerto=LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]}), observabilidad=obs
    )
    binding.invocar([{"rol": "usuario", "texto": "hola"}])
    assert set(obs.libro_mayor()) == {"generador"}

def test_bd3_el_muestreo_efectivo_es_el_enviado_no_el_pedido(cfg_b):
    binding = BindingDelGenerador.desde_configuración(
        cfg_b, puerto=LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]})
    )
    respuesta = binding.invocar([{"rol": "usuario", "texto": "hola"}])
    assert respuesta.muestreo_efectivo["temperature"] == binding.muestreo.temperature
    assert respuesta.muestreo_efectivo["response_format"] == {"type": "json_object"}

def test_el_base_url_no_es_ingrediente_de_la_identidad(cfg_b):
    identidad = BindingDelGenerador.desde_configuración(cfg_b).identidad()
    assert "base_url" not in str(identidad)
    assert set(identidad) == {"rol", "modelo", "prompt_id", "version_del_prompt", "muestreo"}

def test_el_muestreo_no_tiene_defaults():
    with pytest.raises(ErrorDeConfiguración, match="top_p"):
        Muestreo.desde_configuración({"temperature": 0.2, "max_tokens": 10}, clave="x")

def test_el_modelo_vacio_y_el_prompt_sin_version_fallan(cfg_b):
    with pytest.raises(ErrorDeConfiguración, match="modelo"):
        BindingDelGenerador(
            modelo="", prompt_id="p", version_del_prompt="1", muestreo=Muestreo(0.0, 1.0, 10)
        )
    with pytest.raises(ErrorDeConfiguración, match="prompt"):
        BindingDelGenerador(
            modelo="openai/x", prompt_id="p", version_del_prompt="", muestreo=Muestreo(0.0, 1.0, 10)
        )

