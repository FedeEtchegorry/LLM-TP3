from __future__ import annotations

import pytest

from tp3_rag.dominio.binding import BindingDelPlanificador
from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.plan import AlmacenDePlanes, PlanDeConsulta, PlanInválido
from tp3_rag.modulo_b.etapa import EstadoDeConsulta
from tp3_rag.modulo_b.planificador import PlanificadorDeConsulta
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.llm import LLMGrabado

pytestmark = pytest.mark.nivel1

BYTES = b"\xff\xd8\xff-un-paisaje-gris"
PEDIDO = "versiones más cálidas de este paisaje"

PLAN_VÁLIDO = {
    "consulta_visual": "warm sunlit rural landscape with golden light, oil on canvas",
    "consulta_conceptual": (
        "nineteenth-century pastoral landscape painting, warm palette, Barbizon school"
    ),
}

def _cfg(**overrides) -> Configuración:
    return Configuración.cargar(overrides=overrides)

def _almacen(tmp_path) -> AlmacenDePlanes:
    return AlmacenDePlanes(AlmacenamientoEnArchivos(tmp_path), "artifacts/planes")

def _planificador(tmp_path, respuestas=(PLAN_VÁLIDO,), *, cfg=None, almacen=None):
    puerto = LLMGrabado({"planificador": list(respuestas)})
    binding = BindingDelPlanificador.desde_configuración(cfg or _cfg(), puerto=puerto)
    etapa = PlanificadorDeConsulta(
        binding=binding, almacen_de_planes=almacen or _almacen(tmp_path)
    )
    return etapa, puerto

def _estado(pedido: str = PEDIDO, *, procedencia: str = "ad_hoc") -> EstadoDeConsulta:
    if procedencia == "ad_hoc":
        entrada = EntradaDeConsulta.ad_hoc(datos_de_imagen=BYTES, pedido=pedido)
    else:
        entrada = EntradaDeConsulta(
            consulta_id="c01-visual",
            imagen_hash=identidad.hash_de_bytes(BYTES),
            pedido=pedido,
            procedencia="set",
            datos_de_imagen=BYTES,
            artwork_id_de_la_imagen="cma:109240",
            categoria="visual_cromatico",
        )
    return EstadoDeConsulta(entrada=entrada)

def test_una_sola_llamada_por_consulta_y_los_dos_campos_salen_del_mismo_json(tmp_path):
    etapa, puerto = _planificador(tmp_path)
    estado = etapa.correr(_estado())
    assert len(puerto.llamadas) == 1
    assert puerto.llamadas[0]["rol"] == "planificador"
    assert puerto.llamadas[0]["n_imagenes"] == 1
    assert estado.plan.consulta_visual and estado.plan.consulta_conceptual

def test_la_clave_del_estado_es_plan_y_la_llena_esta_etapa(tmp_path):
    from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA

    assert CLAVE_DE_ETAPA[PlanificadorDeConsulta.NOMBRE] == "plan"
    etapa, _ = _planificador(tmp_path)
    estado = etapa.correr(_estado())
    assert estado.etapas_con_clave_poblada() == {"planificador"}

def test_pl5_la_segunda_corrida_lee_el_plan_y_no_llama_al_llm(tmp_path):
    almacen = _almacen(tmp_path)
    primera, puerto_1 = _planificador(tmp_path, almacen=almacen)
    estado_1 = primera.correr(_estado())
    assert estado_1.plan.reusado is False
    assert len(puerto_1.llamadas) == 1

    segunda, puerto_2 = _planificador(tmp_path, almacen=almacen)
    estado_2 = segunda.correr(_estado())
    assert estado_2.plan.reusado is True
    assert puerto_2.llamadas == []
    assert estado_2.plan.plan_id == estado_1.plan.plan_id
    assert estado_2.plan.consulta_visual == estado_1.plan.consulta_visual

def test_pl5_no_se_reescribe_aunque_el_modelo_conteste_otra_cosa(tmp_path):
    almacen = _almacen(tmp_path)
    primera, _ = _planificador(tmp_path, almacen=almacen)
    primera.correr(_estado())
    otra_respuesta = {
        "consulta_visual": "a completely different caption",
        "consulta_conceptual": "a completely different curatorial query",
    }
    segunda, puerto = _planificador(tmp_path, (otra_respuesta,), almacen=almacen)
    plan = segunda.correr(_estado()).plan
    assert plan.consulta_visual == PLAN_VÁLIDO["consulta_visual"]
    assert puerto.llamadas == []

def test_pl1_el_alfa_no_mueve_el_plan_id_y_por_eso_el_barrido_reusa(tmp_path):
    almacen = _almacen(tmp_path)
    ids, llamadas = set(), 0
    for alfa in (0, 0.25, 0.5, 0.75, 1):
        etapa, puerto = _planificador(
            tmp_path, cfg=_cfg(**{"consulta.alfa": alfa}), almacen=almacen
        )
        ids.add(etapa.correr(_estado()).plan.plan_id)
        llamadas += len(puerto.llamadas)
    assert len(ids) == 1, "α entró en el `plan_id`: lo que no entra en se rompió la identidad del plan y el reuso del plan al barrer el peso de fusión con él"
    assert llamadas == 1, "el barrido pagó más de un pre-retrieval"

def test_pl2_otro_muestreo_es_otro_artefacto(tmp_path):
    almacen = _almacen(tmp_path)
    uno, _ = _planificador(tmp_path, almacen=almacen)
    id_1 = uno.correr(_estado()).plan.plan_id
    otro, puerto = _planificador(
        tmp_path,
        cfg=_cfg(**{"bindings.planificador.muestreo.temperature": 0.9}),
        almacen=almacen,
    )
    id_2 = otro.correr(_estado()).plan.plan_id
    assert id_1 != id_2
    assert len(puerto.llamadas) == 1, "se reusó un plan de otro muestreo"

def test_m4_1_un_json_valido_que_no_satisface_pl3_se_reintenta(tmp_path):
    repetida = {"consulta_visual": "same text", "consulta_conceptual": "same text"}
    etapa, puerto = _planificador(tmp_path, (repetida, PLAN_VÁLIDO))
    plan = etapa.correr(_estado()).plan
    assert len(puerto.llamadas) == 2, "no reintentó"
    assert plan.consulta_visual == PLAN_VÁLIDO["consulta_visual"]
    assert plan.n_intentos == 2
    assert plan.uso["tokens_in"] > 0

def test_agotados_los_reintentos_falla_la_etapa_y_no_se_escribe_nada(tmp_path):
    repetida = {"consulta_visual": "same", "consulta_conceptual": "same"}
    almacen = _almacen(tmp_path)
    etapa, puerto = _planificador(tmp_path, (repetida,), almacen=almacen)
    with pytest.raises(PlanInválido, match="expansion"):
        etapa.correr(_estado())
    assert len(puerto.llamadas) == int(etapa.binding.reintentos) + 1
    assert almacen.existentes() == ()

def test_un_json_sin_las_dos_claves_tambien_reintenta(tmp_path):
    incompleto = {"consulta_visual": "only one"}
    etapa, puerto = _planificador(tmp_path, (incompleto, PLAN_VÁLIDO))
    etapa.correr(_estado())
    assert len(puerto.llamadas) == 2

def test_el_veredicto_de_pl3_viaja_en_el_artefacto_con_sus_tres_clausulas(tmp_path):
    almacen = _almacen(tmp_path)
    etapa, _ = _planificador(tmp_path, almacen=almacen)
    plan = etapa.correr(_estado()).plan
    leído = almacen.leer(plan.plan_id)
    assert set(leído.pl3) == {
        "no_vacias",
        "distintas_del_pedido_crudo",
        "distintas_entre_si",
    }
    assert leído.cumple_pl3
    assert leído.pedido == ""
    assert leído.pl3["distintas_del_pedido_crudo"] is True

def test_pl4_el_artefacto_no_tiene_los_bytes_de_la_imagen(tmp_path):
    almacen = _almacen(tmp_path)
    etapa, _ = _planificador(tmp_path, almacen=almacen)
    plan = etapa.correr(_estado()).plan
    crudo = (tmp_path / "artifacts" / "planes" / f"{plan.plan_id}.json").read_text("utf-8")
    assert plan.imagen_hash in crudo
    assert "imagen_retenida" in crudo and '"imagen_retenida": false' in crudo
    assert not any("bytes" in c or "datos" in c for c in PlanDeConsulta.__dataclass_fields__)

def test_d12_una_adhoc_no_guarda_el_pedido_y_una_del_set_si(tmp_path):
    etapa, _ = _planificador(tmp_path / "a")
    artefacto = etapa.correr(_estado(procedencia="ad_hoc")).plan.contenido_del_artefacto()
    assert artefacto["pedido"] is None
    assert artefacto["pedido_retenido"] is False
    assert PEDIDO not in str(artefacto)

    etapa2, _ = _planificador(tmp_path / "b")
    del_set = etapa2.correr(_estado(procedencia="set")).plan.contenido_del_artefacto()
    assert del_set["pedido"] == PEDIDO
    assert del_set["pedido_retenido"] is True

def test_no_hay_clave_de_configuracion_que_encienda_la_retencion_del_pedido(tmp_path):
    etapa, _ = _planificador(tmp_path, cfg=_cfg(**{"retencion.pedido": True}))
    artefacto = etapa.correr(_estado()).plan.contenido_del_artefacto()
    assert artefacto["pedido"] is None

def test_el_reuso_funciona_sin_el_pedido_guardado(tmp_path):
    almacen = _almacen(tmp_path)
    primera, _ = _planificador(tmp_path, almacen=almacen)
    plan = primera.correr(_estado()).plan
    assert almacen.leer(plan.plan_id).pedido == ""
    segunda, puerto = _planificador(tmp_path, almacen=almacen)
    assert segunda.correr(_estado()).plan.reusado is True
    assert puerto.llamadas == []

def test_una_version_del_prompt_sin_texto_se_rechaza_antes_de_correr(tmp_path):
    cfg = _cfg(**{"bindings.planificador.version_del_prompt": "99"})
    with pytest.raises(PlanInválido, match="ingrediente del `plan_id`"):
        _planificador(tmp_path, cfg=cfg)

def test_el_prompt_id_del_modulo_y_el_de_la_configuracion_no_pueden_discrepar(tmp_path):
    cfg = _cfg(**{"bindings.planificador.prompt_id": "otro-prompt"})
    with pytest.raises(PlanInválido, match="identidad del binding"):
        _planificador(tmp_path, cfg=cfg)

def test_las_dos_versiones_del_prompt_existen_y_el_v1_queda_citable():
    from tp3_rag.modulo_b.planificador import PROMPTS, VERSIÓN_DEL_PROMPT, sistema

    assert set(PROMPTS) == {"1", "2"}
    assert VERSIÓN_DEL_PROMPT == "2"
    assert sistema("1") != sistema("2")
    assert "GANA EL PEDIDO" in sistema("2")
    assert "GANA EL PEDIDO" not in sistema("1")
    assert "PROHIBIDO ESCRIBIR EN NEGATIVO" in sistema("2")
    assert "PROHIBIDO ESCRIBIR EN NEGATIVO" not in sistema("1")

def test_el_v2_manda_la_cultura_a_las_dos_consultas():
    from tp3_rag.modulo_b.planificador import sistema

    assert "ESO VA EN LAS DOS" in sistema("2")
    assert "ESO VA en `consulta_conceptual`" in sistema("1")

def test_la_metadata_del_set_no_entra_al_prompt(tmp_path):
    etapa, puerto = _planificador(tmp_path)
    etapa.correr(_estado(procedencia="set"))
    cuerpo = str(puerto.llamadas[0]["cuerpo"])
    assert PEDIDO in cuerpo
    for metadata in ("c01-visual", "visual_cromatico", "cma:109240"):
        assert metadata not in cuerpo
