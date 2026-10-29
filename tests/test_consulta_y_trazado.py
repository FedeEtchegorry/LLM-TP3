from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import (
    EntradaDeConsulta,
    PolíticaDeTrazado,
)
from tp3_rag.dominio.plan import PlanDeConsulta, PlanInválido
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.nucleo.observabilidad import Observabilidad

pytestmark = pytest.mark.nivel1

BYTES = b"\xff\xd8\xff-una-imagen"
HASH = identidad.hash_de_bytes(BYTES)

class _ConsultaDelSetFalsa:
    consulta_id = "c01-visual"
    imagen_hash = HASH
    pedido = "versiones más cálidas de este paisaje"
    artwork_id = "cma:109240"
    categoria = "visual_cromatico"

def test_no_hay_constructor_que_pida_captura_de_entradas():
    import inspect

    assert list(inspect.signature(PolíticaDeTrazado.de).parameters) == ["procedencia"]
    assert PolíticaDeTrazado.de("set").captura_entradas is True
    assert PolíticaDeTrazado.de("ad_hoc").captura_entradas is False

def test_una_procedencia_desconocida_no_cae_en_un_default():
    with pytest.raises(ErrorDeConfiguración, match="procedencia"):
        PolíticaDeTrazado.de("quizas")

def test_la_censura_descarta_y_cuenta():
    completa = PolíticaDeTrazado.de("set")
    sin = PolíticaDeTrazado.de("ad_hoc")
    assert completa.censurar({"pedido": "x", "otro": 1}) == ({"pedido": "x", "otro": 1}, 0)
    assert sin.censurar({"pedido": "x", "otro": 1}) == ({}, 2)

def test_la_politica_se_deriva_de_la_entrada_y_no_se_recibe():
    assert EntradaDeConsulta.del_set(
        _ConsultaDelSetFalsa(), datos_de_imagen=BYTES
    ).política_de_trazado.captura_entradas is True
    assert EntradaDeConsulta.ad_hoc(
        datos_de_imagen=BYTES, pedido="algo"
    ).política_de_trazado.captura_entradas is False

def test_el_default_sin_contexto_es_el_restrictivo():
    obs = Observabilidad(silencioso=True)
    with obs.nodo("suelto", entradas={"pedido": "secreto"}):
        pass
    nodo = obs.nodos()[0]
    assert nodo.entradas == {}
    assert nodo.campos_censurados == 1
    assert nodo.politica == "ad_hoc_sin_entradas"

def test_el_contexto_restaura_la_politica_anterior():
    obs = Observabilidad(silencioso=True)
    with obs.con_politica_de_trazado(PolíticaDeTrazado.de("set")):
        with obs.nodo("del-set", entradas={"pedido": "p"}):
            pass
    with obs.nodo("despues", entradas={"pedido": "p"}):
        pass
    assert obs.nodos()[0].entradas == {"pedido": "p"}
    assert obs.nodos()[1].entradas == {}

def test_la_politica_del_nodo_distingue_censurado_de_vacio():
    obs = Observabilidad(silencioso=True)
    with obs.con_politica_de_trazado(PolíticaDeTrazado.de("set")):
        with obs.nodo("sin-nada-que-llevar"):
            pass
    assert obs.nodos()[0].entradas == {}
    assert obs.nodos()[0].politica == "completa"

def test_los_bytes_viajan_y_nunca_se_serializan():
    entrada = EntradaDeConsulta.ad_hoc(datos_de_imagen=BYTES, pedido="algo")
    assert entrada.datos_de_imagen == BYTES
    for retener in (False, True):
        d = entrada.a_json(retener_pedido=retener)
        assert BYTES.decode("latin-1") not in str(d)
        assert d["imagen_retenida"] is False
    assert entrada.a_json()["pedido"] is None
    assert entrada.a_json()["pedido_retenido"] is False
    assert entrada.a_json(retener_pedido=True)["pedido"] == "algo"

def test_el_consulta_id_del_set_se_declara_y_el_ad_hoc_se_deriva():
    del_set = EntradaDeConsulta.del_set(_ConsultaDelSetFalsa(), datos_de_imagen=BYTES)
    assert del_set.consulta_id == "c01-visual"
    ad_hoc = EntradaDeConsulta.ad_hoc(datos_de_imagen=BYTES, pedido="algo")
    assert ad_hoc.consulta_id == identidad.consulta_id_ad_hoc(imagen_hash=HASH, pedido="algo")

def test_un_consulta_id_del_set_con_forma_de_hash_se_rechaza():
    class ConHash(_ConsultaDelSetFalsa):
        consulta_id = "a" * 16

    with pytest.raises(ErrorDeConfiguración, match="hash_corto"):
        EntradaDeConsulta.del_set(ConHash(), datos_de_imagen=BYTES)

def test_una_ad_hoc_sin_pedido_no_es_una_consulta():
    with pytest.raises(ErrorDeConfiguración, match="sin pedido"):
        EntradaDeConsulta.ad_hoc(datos_de_imagen=BYTES, pedido="   ")

def test_el_imagen_hash_lleva_prefijo_de_algoritmo():
    with pytest.raises(ErrorDeConfiguración, match="prefijo de algoritmo"):
        EntradaDeConsulta(
            consulta_id="x", imagen_hash="deadbeef", pedido="p", procedencia="ad_hoc"
        )

MUESTREO = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 100, "seed": 7369}

def test_pl1_alfa_n_fusion_y_variante_no_entran_en_el_plan_id():
    import inspect

    parámetros = set(inspect.signature(identidad.plan_id).parameters)
    assert parámetros == {
        "imagen_hash",
        "pedido",
        "prompt_id",
        "version_del_prompt",
        "modelo",
        "muestreo",
    }
    for prohibido in identidad.NO_DETERMINAN_EL_PLAN:
        assert prohibido not in parámetros

def test_pl2_el_prompt_el_modelo_y_el_muestreo_si_entran():
    base = dict(
        imagen_hash=HASH, pedido="p", prompt_id="pr", version_del_prompt="1",
        modelo="openai/m", muestreo=MUESTREO,
    )
    uno = identidad.plan_id(**base)
    assert uno != identidad.plan_id(**{**base, "prompt_id": "otro"})
    assert uno != identidad.plan_id(**{**base, "version_del_prompt": "2"})
    assert uno != identidad.plan_id(**{**base, "modelo": "openai/otro"})
    assert uno != identidad.plan_id(**{**base, "muestreo": {**MUESTREO, "temperature": 0.9}})
    assert uno != identidad.plan_id(**{**base, "muestreo": {**MUESTREO, "seed": 1}})
    assert uno == identidad.plan_id(
        **{**base, "muestreo": dict(reversed(list(MUESTREO.items())))}
    )

def test_pl3_tiene_tres_clausulas_y_son_tres_cosas():
    def plan(visual, conceptual, pedido="el pedido crudo"):
        return PlanDeConsulta(
            plan_id="x", consulta_id="c", imagen_hash=HASH, pedido=pedido,
            consulta_visual=visual, consulta_conceptual=conceptual,
            binding={}, uso={},
        )

    plan("un paisaje cálido", "pintura rural del siglo XIX").validar()
    with pytest.raises(PlanInválido, match="vacía"):
        plan("", "algo").validar()
    with pytest.raises(PlanInválido, match="expansion"):
        plan("lo mismo", "lo mismo").validar()
    with pytest.raises(PlanInválido, match="pedido crudo"):
        plan("el pedido crudo", "otra cosa").validar()

def test_pl4_el_plan_no_tiene_campo_de_bytes():
    campos = set(PlanDeConsulta.__dataclass_fields__)
    assert "imagen_hash" in campos
    assert not any("datos" in c or "bytes" in c for c in campos)
