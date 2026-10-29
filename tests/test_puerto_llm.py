from __future__ import annotations

import json

import pytest

from tp3_rag.dominio.binding import BindingDelGenerador, Muestreo
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.puertos.llm import (
    EXPLICACIÓN_CON_REF_COLGADA,
    EXPLICACIÓN_TODA_COLGADA,
    EXPLICACIÓN_VÁLIDA,
    MÁXIMO_DE_IMÁGENES,
    FallaDeEtapa,
    LLMGrabado,
    _cuerpo,
    _validar_modelo,
)

pytestmark = pytest.mark.nivel1

def _binding(**cambios):
    base = dict(
        modelo="openai/gpt-4.1-mini",
        prompt_id="p",
        version_del_prompt="1",
        muestreo=Muestreo(temperature=0.2, top_p=1.0, max_tokens=100, seed=7369),
        esquema_de_salida="json",
        reintentos=2,
    )
    base.update(cambios)
    return BindingDelGenerador(**base)

MENSAJES = [
    {"rol": "sistema", "texto": "las reglas"},
    {"rol": "usuario", "texto": "el pedido"},
]

def test_k2_el_nombre_pelado_se_rechaza_antes_de_llamar():
    with pytest.raises(ErrorDeConfiguración, match="prefijo de proveedor"):
        _validar_modelo("gpt-4.1-mini")
    assert _validar_modelo("openai/gpt-4.1-mini") == "openai/gpt-4.1-mini"

def test_el_doble_valida_el_prefijo_igual_que_el_adaptador_real():
    with pytest.raises(ErrorDeConfiguración, match="prefijo"):
        LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]}).invocar(
            _binding(modelo="gpt-4.1-mini"), MENSAJES
        )

def test_el_tope_de_dos_imagenes_es_del_contrato_no_del_proveedor():
    assert MÁXIMO_DE_IMÁGENES == 2
    _cuerpo(_binding(), MENSAJES, [b"a", b"b"])
    with pytest.raises(ErrorDeConfiguración, match="0..2"):
        _cuerpo(_binding(), MENSAJES, [b"a", b"b", b"c"])

def test_el_mensaje_de_sistema_va_separado_y_las_imagenes_en_el_de_usuario():
    cuerpo = _cuerpo(_binding(), MENSAJES, [b"\xff\xd8bytes"])
    assert cuerpo["messages"][0]["role"] == "system"
    assert cuerpo["messages"][0]["content"] == "las reglas"
    partes = cuerpo["messages"][1]["content"]
    assert [p["type"] for p in partes] == ["text", "image_url"]
    assert partes[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

def test_bd3_el_muestreo_viaja_en_el_cuerpo_y_de_ahi_se_lee():
    cuerpo = _cuerpo(_binding(), MENSAJES, [])
    for clave, valor in (("temperature", 0.2), ("top_p", 1.0), ("max_tokens", 100), ("seed", 7369)):
        assert cuerpo[clave] == valor
    assert cuerpo["response_format"] == {"type": "json_object"}

def test_el_esquema_texto_no_pide_json():
    cuerpo = _cuerpo(_binding(esquema_de_salida="texto"), MENSAJES, [])
    assert "response_format" not in cuerpo

def test_sin_respuesta_grabada_el_doble_falla_en_vez_de_improvisar():
    with pytest.raises(FallaDeEtapa, match="sin respuesta grabada"):
        LLMGrabado({}).invocar(_binding(), MENSAJES)

def test_la_falla_de_etapa_es_clase_3_y_no_degrada_en_el_puerto():
    falla = FallaDeEtapa("JSON inválido", "no parsea")
    assert falla.causa == "JSON inválido"
    with pytest.raises(FallaDeEtapa):
        LLMGrabado({}, fallas={"generador": falla}).invocar(_binding(), MENSAJES)

def test_las_respuestas_grabadas_cubren_el_caso_que_7_nombra():
    colgadas = EXPLICACIÓN_CON_REF_COLGADA["obras"][0]["afirmaciones"]
    assert any("exhibitions" in r for a in colgadas for r in a["refs"])
    assert len(EXPLICACIÓN_TODA_COLGADA["obras"][0]["afirmaciones"]) == 1
    assert EXPLICACIÓN_VÁLIDA["obras"][0]["afirmaciones"][1]["tipo"] == "observacion"

def test_la_respuesta_declara_uso_y_n_para_el_decorador():
    respuesta = LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]}).invocar(_binding(), MENSAJES)
    assert set(respuesta.uso) == {"tokens_in", "tokens_out"}
    assert respuesta.uso["tokens_in"] > 0
    assert respuesta.n == 1
    assert respuesta.json()["obras"]

def test_las_respuestas_se_consumen_en_orden_y_una_sola_se_repite():
    doble = LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA, EXPLICACIÓN_CON_REF_COLGADA]})
    primera = json.loads(doble.invocar(_binding(), MENSAJES).texto)
    segunda = json.loads(doble.invocar(_binding(), MENSAJES).texto)
    assert primera != segunda
    uno = LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]})
    assert uno.invocar(_binding(), MENSAJES).texto == uno.invocar(_binding(), MENSAJES).texto

def test_el_doble_registra_las_llamadas_para_que_se_puedan_contar():
    doble = LLMGrabado({"generador": [EXPLICACIÓN_VÁLIDA]})
    doble.invocar(_binding(), MENSAJES, [b"img"])
    assert doble.llamadas[0]["rol"] == "generador"
    assert doble.llamadas[0]["n_imagenes"] == 1
