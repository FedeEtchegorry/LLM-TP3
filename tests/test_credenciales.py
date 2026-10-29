from __future__ import annotations

import pytest

from tp3_rag.nucleo.credenciales import VARIABLES_DE_LA_KEY, Credenciales
from tp3_rag.nucleo.errores import ErrorDeConfiguración

pytestmark = pytest.mark.nivel1

def _entorno(monkeypatch, **pares):
    for variable in (*VARIABLES_DE_LA_KEY, "LANGSMITH_API_KEY", "TP3_CONTACTO_AIC"):
        monkeypatch.delenv(variable, raising=False)
    for clave, valor in pares.items():
        monkeypatch.setenv(clave, valor)

def test_k4_gana_llm_api_key_sobre_el_alias(tmp_path, monkeypatch):
    _entorno(monkeypatch, LLM_API_KEY="de-la-catedra", OPENAI_API_KEY="del-sdk")
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    assert credenciales.openai_api_key == "de-la-catedra"
    assert credenciales.variable_de_la_key == "LLM_API_KEY"

def test_k4_openai_api_key_sigue_valiendo_como_alias(tmp_path, monkeypatch):
    _entorno(monkeypatch, OPENAI_API_KEY="solo-el-viejo")
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    assert credenciales.openai_api_key == "solo-el-viejo"
    assert credenciales.variable_de_la_key == "OPENAI_API_KEY"

def test_k4_el_orden_de_precedencia_esta_declarado_y_no_es_incidental():
    assert VARIABLES_DE_LA_KEY == ("LLM_API_KEY", "OPENAI_API_KEY")

def test_sin_ninguna_de_las_dos_el_error_nombra_la_de_la_catedra(tmp_path, monkeypatch):
    _entorno(monkeypatch)
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    assert credenciales.variable_de_la_key is None
    with pytest.raises(ErrorDeConfiguración) as e:
        credenciales.exigir_openai()
    assert "LLM_API_KEY" in str(e.value)
    assert "base_url" in str(e.value)

def test_el_mensaje_dice_que_sin_la_key_igual_se_puede_casi_todo_el_indexado(
    tmp_path, monkeypatch
):
    _entorno(monkeypatch)
    with pytest.raises(ErrorDeConfiguración, match="--hasta fragmentos"):
        Credenciales.desde_entorno(raíz=tmp_path).exigir_openai()

def test_de_la_credencial_solo_sale_la_constancia(tmp_path, monkeypatch):
    _entorno(monkeypatch, LLM_API_KEY="secreta", TP3_CONTACTO_AIC="x@y")
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    constancia = credenciales.constancia()
    assert "secreta" not in str(constancia)
    assert constancia["openai_api_key_presente"] is True
    assert "secreta" not in repr(credenciales)
