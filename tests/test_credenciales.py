from __future__ import annotations

import pytest

from tp3_rag.nucleo.credenciales import (
    CLAVE_DE_LA_KEY_EN_ARTEFACTOS,
    VARIABLES_DE_LA_KEY,
    Credenciales,
    key_presente_en,
)
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
    assert credenciales.llm_api_key == "de-la-catedra"
    assert credenciales.variable_de_la_key == "LLM_API_KEY"

def test_k4_openai_api_key_sigue_valiendo_como_alias(tmp_path, monkeypatch):
    _entorno(monkeypatch, OPENAI_API_KEY="solo-el-viejo")
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    assert credenciales.llm_api_key == "solo-el-viejo"
    assert credenciales.variable_de_la_key == "OPENAI_API_KEY"

def test_k4_el_orden_de_precedencia_esta_declarado_y_no_es_incidental():
    assert VARIABLES_DE_LA_KEY == ("LLM_API_KEY", "OPENAI_API_KEY")

def test_sin_ninguna_de_las_dos_el_error_nombra_la_de_la_catedra(tmp_path, monkeypatch):
    _entorno(monkeypatch)
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    assert credenciales.variable_de_la_key is None
    with pytest.raises(ErrorDeConfiguración) as e:
        credenciales.exigir_llm()
    assert "LLM_API_KEY" in str(e.value)
    assert "base_url" in str(e.value)

def test_el_mensaje_dice_que_sin_la_key_igual_se_puede_casi_todo_el_indexado(
    tmp_path, monkeypatch
):
    _entorno(monkeypatch)
    with pytest.raises(ErrorDeConfiguración, match="--hasta fragmentos"):
        Credenciales.desde_entorno(raíz=tmp_path).exigir_llm()

def test_de_la_credencial_solo_sale_la_constancia(tmp_path, monkeypatch):
    _entorno(monkeypatch, LLM_API_KEY="secreta", TP3_CONTACTO_AIC="x@y")
    credenciales = Credenciales.desde_entorno(raíz=tmp_path)
    constancia = credenciales.constancia()
    assert "secreta" not in str(constancia)
    assert constancia["llm_api_key_presente"] is True
    assert "secreta" not in repr(credenciales)

def test_m2c4_la_constancia_escribe_el_nombre_nuevo_y_solo_ese():
    constancia = Credenciales(llm_api_key="x").constancia()
    assert constancia[CLAVE_DE_LA_KEY_EN_ARTEFACTOS] is True
    assert "openai_api_key_presente" not in constancia

def test_m2c4_se_lee_por_los_dos_nombres():
    assert key_presente_en({"llm_api_key_presente": True}) is True
    assert key_presente_en({"openai_api_key_presente": True}) is True
    assert key_presente_en({"openai_api_key_presente": False}) is False
    assert (
        key_presente_en(
            {"llm_api_key_presente": True, "openai_api_key_presente": False}
        )
        is True
    )

def test_m2c4_una_constancia_que_no_habla_de_la_key_da_ausente_y_no_false():
    assert key_presente_en({"contacto_aic_presente": True}) is None
    assert key_presente_en({}) is None
    assert key_presente_en(None) is None

def test_m2c4_el_artefacto_sellado_de_la_ingesta_se_sigue_leyendo():
    import json

    from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

    ruta = RAÍZ_DEL_REPO / "artifacts/ingesta/e10c4e4f9cce0e74-1.json"
    if not ruta.exists():  # pragma: no cover - el artefacto no está versionado
        pytest.skip("el artefacto sellado de la ingesta no está en este árbol")
    credenciales = json.loads(ruta.read_text(encoding="utf-8"))["procedencia"][
        "credenciales"
    ]
    assert "openai_api_key_presente" in credenciales, (
        "el artefacto sellado dejó de tener la clave vieja: o se reescribió "
        "—y no se debía— o cambió de esquema"
    )
    assert key_presente_en(credenciales) is False
