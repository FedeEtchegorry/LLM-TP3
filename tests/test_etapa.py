from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.modulo_b.etapa import (
    CLAVE_DE_ETAPA,
    ETAPAS_DE_RANKING,
    NO_OMITIBLES,
    Candidata,
    EstadoDeConsulta,
    FormaDeclaradaRota,
    SobrescrituraProhibida,
)
from tp3_rag.nucleo import identidad

pytestmark = pytest.mark.nivel1

BYTES = b"\x89PNG\r\nbytes-que-no-aparecen-en-ninguna-prosa"

@pytest.fixture
def estado() -> EstadoDeConsulta:
    return EstadoDeConsulta(
        entrada=EntradaDeConsulta.ad_hoc(datos_de_imagen=BYTES, pedido="un pedido")
    )

def _candidatas(*ids):
    return tuple(Candidata(i, n + 1, 1.0 - n / 100) for n, i in enumerate(ids))

def test_el_mapa_cubre_las_ocho_etapas_y_cada_una_tiene_una_sola_clave():
    assert len(CLAVE_DE_ETAPA) == 8
    assert len(set(CLAVE_DE_ETAPA.values())) == 8
    assert CLAVE_DE_ETAPA["recuperador_visual"] != CLAVE_DE_ETAPA["recuperador_textual"]
    assert "top5" not in CLAVE_DE_ETAPA.values()
    assert set(NO_OMITIBLES) == {"recuperador_visual", "renderizador"}
    assert set(ETAPAS_DE_RANKING) <= set(CLAVE_DE_ETAPA)

def test_et1_clausura_entrada_y_salida_son_el_mismo_tipo(estado):
    nuevo = estado.con(fuentes_de_ranking={"visual": _candidatas("a", "b")})
    assert isinstance(nuevo, EstadoDeConsulta)
    assert nuevo is not estado

def test_et3_no_se_puede_sobrescribir_una_clave_poblada(estado):
    con_rerank = estado.con(rerank=({"artwork_id": "a", "score_0_10": 9},))
    with pytest.raises(SobrescrituraProhibida, match="sobrescribir"):
        con_rerank.con(rerank=({"artwork_id": "b", "score_0_10": 1},))

def test_et3_una_fuente_de_ranking_no_pisa_a_la_otra_y_tampoco_a_si_misma(estado):
    visual = estado.con(fuentes_de_ranking={"visual": _candidatas("a")})
    dos = visual.con(fuentes_de_ranking={"texto": _candidatas("b")})
    assert set(dos.fuentes_de_ranking) == {"visual", "texto"}
    with pytest.raises(SobrescrituraProhibida, match="ya estaba registrada"):
        dos.con(fuentes_de_ranking={"visual": _candidatas("z")})

def test_et3_el_top5_no_puede_traer_una_obra_que_nadie_propuso(estado):
    con_top = estado.proyectar_top5(["a", "b", "c"], por="recuperador_visual")
    assert con_top.proyectar_top5(["c", "a", "b"], por="reranker").top5 == ("c", "a", "b")
    with pytest.raises(SobrescrituraProhibida, match="ninguna etapa de ranking propuso"):
        con_top.proyectar_top5(["a", "b", "z"], por="reranker")

def test_et3_una_etapa_puede_proyectar_un_conjunto_distinto_si_alguna_rama_lo_propuso(
    estado,
):
    con_ramas = estado.con(
        fuentes_de_ranking={"visual": _candidatas("a", "b"), "texto": _candidatas("z")}
    ).proyectar_top5(["a", "b"], por="recuperador_visual")
    fusionado = con_ramas.proyectar_top5(["z", "a"], por="fusionador")
    assert fusionado.top5 == ("z", "a")

def test_una_clave_inventada_se_rechaza(estado):
    with pytest.raises(SobrescrituraProhibida, match="no es una clave"):
        estado.con(inventada=1)

def test_et5_la_ausencia_de_registro_significa_que_no_degrado(estado):
    assert estado.degradaciones == ()
    assert estado.degradó is False
    espontánea = estado.degradar("generador", "boom", "sin explicación")
    planificada = estado.degradar("generador", "boom", "sin explicación", planificada=True)
    assert espontánea.degradaciones_no_planificadas
    assert not planificada.degradaciones_no_planificadas

def test_et2_omitir_y_fallar_dan_la_misma_forma(estado):
    omitida = estado
    fallada = estado.degradar("generador", "boom", "sin explicación")
    assert omitida.explicacion is None and fallada.explicacion is None
    omitida_json = omitida.a_json()
    fallada_json = fallada.a_json()
    diferencias = {k for k in omitida_json if omitida_json[k] != fallada_json[k]}
    assert diferencias == {"degradaciones"}

CADENA = ("recuperador_visual", "recolector_de_evidencia", "generador", "renderizador")

def test_et6_primera_direccion_la_etapa_muda(estado):
    solo_visual = estado.con(fuentes_de_ranking={"visual": _candidatas("a")})
    with pytest.raises(FormaDeclaradaRota, match="primera dirección"):
        solo_visual.verificar_et6(CADENA)

def test_et6_segunda_direccion_la_clave_de_una_etapa_fuera_de_la_cadena(estado):
    con_intrusa = (
        estado.con(fuentes_de_ranking={"visual": _candidatas("a")})
        .con(rerank=({"artwork_id": "a", "score_0_10": 9},))
        .con(evidencia={"a": ()})
        .con(explicacion=())
        .con(salida=object())
    )
    with pytest.raises(FormaDeclaradaRota, match="segunda dirección"):
        con_intrusa.verificar_et6(CADENA)

def test_et6_una_etapa_degradada_satisface_la_primera_direccion(estado):
    completo = (
        estado.con(fuentes_de_ranking={"visual": _candidatas("a", "b")})
        .proyectar_top5(["a", "b"], por="recuperador_visual")
        .con(evidencia={"a": ()})
        .degradar("generador", "boom", "sin explicación")
        .con(salida=object())
    )
    assert completo.verificar_et6(CADENA)["cumple_et6"] is True

def test_et6_verifica_el_top5_contra_la_ultima_de_ranking_que_no_degrado(estado):
    bueno = (
        estado.con(fuentes_de_ranking={"visual": _candidatas("a", "b", "c")})
        .proyectar_top5(["a", "b"], por="recuperador_visual")
        .con(evidencia={})
        .con(explicacion=())
        .con(salida=object())
    )
    assert bueno.verificar_et6(CADENA, top_k=2)["top5"]["contra"] == "recuperador_visual"
    mentiroso = bueno.proyectar_top5(["b", "a"], por="alguien")
    with pytest.raises(FormaDeclaradaRota, match="no son los primeros"):
        mentiroso.verificar_et6(CADENA, top_k=2)

def test_el_pedido_crudo_es_la_conducta_de_c1(estado):
    assert estado.consulta_visual == "un pedido"
    assert estado.consulta_conceptual == "un pedido"

def test_sa5_los_descartes_se_acumulan_y_se_cuentan(estado):
    con_dos = estado.descartar_afirmaciones([{"a": 1}]).descartar_afirmaciones([{"b": 2}])
    assert len(con_dos.afirmaciones_descartadas) == 2

def test_marcar_fallida_no_lanza_y_eso_es_lo_que_salva_et1(estado):
    marcado = estado.marcar_fallida("renderizador", "atribución faltante")
    assert isinstance(marcado, EstadoDeConsulta)
    assert marcado.fallida == {"etapa": "renderizador", "obligacion": "atribución faltante"}

def test_la_entrada_no_se_puede_mutar(estado):
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        estado.entrada.pedido = "otro"  # type: ignore[misc]

def test_el_json_del_estado_no_lleva_los_bytes(estado):
    serializado = identidad.json_canónico(estado.a_json())
    assert identidad.hash_de_bytes(BYTES) in serializado
    assert BYTES.decode("latin-1") not in serializado
