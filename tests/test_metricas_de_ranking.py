from __future__ import annotations

import pytest

from tp3_rag.modulo_c import metricas as met
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.puertos.indice_vectorial import ARTWORKS, ÍndiceVectorialEnMemoria

from conftest import corrida_de_prueba

pytestmark = pytest.mark.nivel1

CULTURAS = {"c01-conceptual": ("China", "Chinese"), "c02-conceptual": ("France",)}
CATEGORÍAS = {
    "c01-visual": "visual_cromatico",
    "c01-conceptual": "conceptual_cultural",
    "c01-ambiguo": "ambiguo",
    "c02-conceptual": "conceptual_cultural",
}

def test_9_5_una_corrida_degradada_no_agrega_y_las_tres_situaciones_se_separan():
    limpia = corrida_de_prueba(consulta_id="c01-visual", top5=["cma:1"])
    planificada = corrida_de_prueba(
        consulta_id="c01-conceptual",
        top5=["cma:2"],
        degradaciones=[{"etapa": "generador", "causa": "inyectada", "politica": "p",
                        "planificada": True}],
    )
    espontanea = corrida_de_prueba(
        consulta_id="c01-ambiguo",
        top5=["cma:3"],
        degradaciones=[{"etapa": "recuperador_textual", "causa": "RedCaída: timeout",
                        "politica": "p", "planificada": False}],
    )
    fallida = corrida_de_prueba(
        consulta_id="c02-conceptual",
        top5=["cma:4"],
        fallida={"etapa": "renderizador", "obligacion": "atribución"},
    )
    reparto = met.separar_por_9_5([limpia, planificada, espontanea, fallida])
    assert reparto.agregan == (limpia,)
    assert reparto.planificadas == (planificada,)
    assert reparto.espontaneas == (espontanea,)
    assert reparto.fallidas == (fallida,)
    assert any("RedCaída" in c for c in reparto.a_json()["causas_espontaneas"])

def test_qs_es_cero_cuando_el_pedido_no_mueve_nada():
    mismo = ["a", "b", "c", "d", "e"]
    assert met.qs([mismo, list(mismo), list(mismo)]) == pytest.approx(0.0)

def test_qs_es_uno_cuando_los_tres_pedidos_no_comparten_nada():
    assert met.qs([["a"], ["b"], ["c"]]) == pytest.approx(1.0)

def test_qs_con_un_solo_pedido_es_ausente_y_no_cero():
    assert met.qs([["a", "b"]]) is None
    assert met.qs([]) is None

def test_qs_por_imagen_agrupa_los_pedidos_de_cada_imagen():
    corridas = [
        corrida_de_prueba(consulta_id="c01-visual", top5=["a", "b"]),
        corrida_de_prueba(consulta_id="c01-conceptual", top5=["a", "c"]),
        corrida_de_prueba(consulta_id="c02-conceptual", top5=["z"]),
    ]
    imagen = {"c01-visual": "aic:1", "c01-conceptual": "aic:1", "c02-conceptual": "aic:2"}
    por_imagen = met.qs_por_imagen(corridas, imagen_de=imagen.get)
    assert por_imagen["aic:1"] == pytest.approx(1 - 1 / 3)
    assert por_imagen["aic:2"] is None

def _indice_con_culturas(por_obra) -> ÍndiceVectorialEnMemoria:
    indice = ÍndiceVectorialEnMemoria(variante_id="v")
    indice.upsert(
        ARTWORKS,
        ids=list(por_obra),
        vectores=[[1.0, 0.0] for _ in por_obra],
        metadatas=[{"museum": "cma", "culture": c} for c in por_obra.values()],
    )
    return indice

def test_la_metadata_por_ids_es_una_operacion_del_puerto():
    indice = _indice_con_culturas({"cma:1": ["China"], "cma:2": None})
    metadata = indice.metadatas(ARTWORKS, ["cma:1", "cma:2", "cma:no-existe"])
    assert met.culturas_del(metadata["cma:1"]) == ("China",)
    assert met.culturas_del(metadata["cma:2"]) == ()
    assert "cma:no-existe" not in metadata

def test_la_cobertura_cultural_cuenta_las_obras_de_la_cultura_pedida():
    indice = _indice_con_culturas(
        {
            "cma:1": ["China"],
            "cma:2": ["Chinese"],
            "cma:3": ["Japan"],
            "cma:4": None,
        }
    )
    corrida = corrida_de_prueba(
        consulta_id="c01-conceptual", top5=["cma:1", "cma:2", "cma:3", "cma:4"]
    )
    cobertura = met.cobertura_cultural(
        [corrida],
        culturas_por_consulta=CULTURAS,
        categoria_de=CATEGORÍAS,
        metadata_por_obra=indice.metadatas(ARTWORKS, corrida.top5),
    )
    assert len(cobertura) == 1
    assert cobertura[0].n_en_el_top5 == 2
    assert cobertura[0].de == 4
    assert cobertura[0].sin_culture == 1

def test_la_cobertura_cultural_ignora_las_categorias_sin_cultura_pedida():
    indice = _indice_con_culturas({"cma:1": ["China"]})
    corridas = [
        corrida_de_prueba(consulta_id="c01-visual", top5=["cma:1"]),
        corrida_de_prueba(consulta_id="c01-ambiguo", top5=["cma:1"]),
    ]
    cobertura = met.cobertura_cultural(
        corridas,
        culturas_por_consulta=CULTURAS,
        categoria_de=CATEGORÍAS,
        metadata_por_obra=indice.metadatas(ARTWORKS, ["cma:1"]),
    )
    assert cobertura == ()

def test_el_aplanado_del_puerto_se_deshace_al_medir():
    indice = _indice_con_culturas({"cma:1": ["France", "19th century"]})
    metadata = indice.metadatas(ARTWORKS, ["cma:1"])
    assert met.culturas_del(metadata["cma:1"]) == ("France", "19th century")

def test_el_delta_por_categoria_mide_churn_y_se_empareja_por_consulta():
    a = [
        corrida_de_prueba(consulta_id="c01-visual", top5=["a", "b", "c", "d", "e"]),
        corrida_de_prueba(consulta_id="c01-conceptual", top5=["a", "b", "c", "d", "e"]),
        corrida_de_prueba(consulta_id="c01-ambiguo", top5=["a", "b"]),
    ]
    b = [
        corrida_de_prueba(consulta_id="c01-visual", top5=["a", "b", "c", "d", "e"]),
        corrida_de_prueba(consulta_id="c01-conceptual", top5=["a", "b", "x", "y", "z"]),
    ]
    deltas = {d.categoria: d for d in met.delta_por_categoria(a, b, categoria_de=CATEGORÍAS)}
    assert deltas["visual_cromatico"].churn_medio == 0.0
    assert deltas["conceptual_cultural"].churn_medio == 3.0
    assert "ambiguo" not in deltas

def test_el_delta_de_cobertura_es_ausente_donde_no_hay_cultura_pedida():
    a = [corrida_de_prueba(consulta_id="c01-visual", top5=["a"])]
    b = [corrida_de_prueba(consulta_id="c01-visual", top5=["b"])]
    delta = met.delta_por_categoria(a, b, categoria_de=CATEGORÍAS)[0]
    assert delta.cobertura_a is None and delta.delta_de_cobertura is None

def test_el_delta_de_cobertura_sale_del_par_de_coberturas():
    a = [corrida_de_prueba(consulta_id="c01-conceptual", top5=["a"])]
    b = [corrida_de_prueba(consulta_id="c01-conceptual", top5=["b"])]
    delta = met.delta_por_categoria(
        a,
        b,
        categoria_de=CATEGORÍAS,
        cobertura_a=[
            met.CoberturaCultural("c01-conceptual", "conceptual_cultural", ("China",), 1, 5, 0)
        ],
        cobertura_b=[
            met.CoberturaCultural("c01-conceptual", "conceptual_cultural", ("China",), 4, 5, 0)
        ],
    )[0]
    assert delta.cobertura_a == 1.0 and delta.cobertura_b == 4.0
    assert delta.delta_de_cobertura == pytest.approx(3.0)

def test_m10_una_adhoc_con_el_config_id_de_una_fila_no_es_poblacion_de_la_tabla():
    assert corrida_de_prueba(consulta_id="c01-visual", top5=["a"]).es_del_set
    adhoc = corrida_de_prueba(consulta_id="78a9b1ff", top5=["a"], procedencia="ad_hoc")
    assert not adhoc.es_del_set
    assert adhoc.procedencia == "ad_hoc"

def test_m10_la_tabla_cuenta_las_adhoc_aparte_en_vez_de_sumarlas(tmp_path, monkeypatch):
    from tp3_rag.dominio.corrida import AlmacenDeCorridas
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    config_ids = met.config_ids_de_la_tabla()
    almacén = AlmacenamientoEnArchivos(tmp_path)
    corridas = AlmacenDeCorridas(almacén, "corridas")
    for c in (
        corrida_de_prueba(consulta_id="c01-visual", top5=["a"], config_id=config_ids["C1"]),
        corrida_de_prueba(
            consulta_id="78a9b1ff",
            top5=["b"],
            config_id=config_ids["C1"],
            procedencia="ad_hoc",
        ),
    ):
        (tmp_path / "corridas" / c.corrida_id).mkdir(parents=True, exist_ok=True)
        corridas.escribir(c.corrida_id, c.contenido)

    lectura = met.corridas_de_la_tabla(corridas)
    assert lectura["en_disco"] == 2
    assert lectura["por_fila"] == {"C1": 1}
    assert lectura["ad_hoc"] == {"C1": 1}
    assert not lectura["huerfanas"]

def _rerank(scores):
    filas = [
        {"artwork_id": f"aic:{1000 + i}", "score_0_10": s, "rango_fusionado": i + 1,
         "estado": "puntuado"}
        for i, s in enumerate(scores)
    ]
    return sorted(filas, key=lambda f: (-f["score_0_10"], f["rango_fusionado"]))

def test_m10_derivar_n_filtra_por_rango_fusionado_y_no_reordena():
    scores = [5] * 50
    scores[29] = 10
    scores[15] = 9
    scores[3] = 8
    por_n = met.derivar_n(_rerank(scores), valores=(10, 20, 50))
    assert por_n[50][0] == "aic:1029"
    assert por_n[20][0] == "aic:1015"
    assert por_n[10][0] == "aic:1003"
    for n, top5 in por_n.items():
        assert len(top5) == 5
        assert all(int(o.split(":")[1]) - 1000 < n for o in top5)

def test_m10_una_corrida_con_candidata_sin_puntaje_no_se_deriva():
    filas = _rerank([5] * 20)
    filas[7] = {**filas[7], "score_0_10": None, "estado": "sin_puntaje"}
    assert met.derivar_n(filas, valores=(10, 20)) == {}

def test_m10_los_tres_n_comparten_los_puntajes_y_solo_difieren_en_n():
    por_n = met.derivar_n(_rerank([7] * 50), valores=(10, 20, 50))
    assert por_n[10] == por_n[20] == por_n[50]
    assert por_n[50] == tuple(f"aic:{1000 + i}" for i in range(5))

def _con_evidencia(corrida, evidencia, citas):
    contenido = corrida.contenido
    contenido["configuracion_resuelta"].setdefault("fragmentacion", {})[
        "campos_de_knowledge"
    ] = ["description", "did_you_know", "provenance"]
    contenido["estado"]["evidencia"] = {
        artwork_id: [
            {
                "fragment_id": fid,
                "section": section,
                "texto": texto,
                "license": "CC0",
                "artwork_id": artwork_id,
            }
            for fid, section, texto in fragmentos
        ]
        for artwork_id, fragmentos in evidencia.items()
    }
    for obra in contenido["estado"]["salida"]["obras"]:
        obra["lineas"] = ["Una afirmación con dato. [1]"]
        obra["referencias"] = [
            {
                "n": n,
                "fragment_id": fid,
                "texto_citado": texto,
                "license": "CC0",
                "atribucion": None,
            }
            for n, (fid, texto) in enumerate(citas.get(obra["artwork_id"], ()), start=1)
        ]
    return corrida

def _brazo(consulta_id, top5, *, config_id, con_campos_nuevos: bool):
    corrida = corrida_de_prueba(
        consulta_id=consulta_id, top5=top5, configuracion="C3", config_id=config_id
    )
    evidencia, citas = {}, {}
    for artwork_id in top5:
        secciones = (
            ("description", "citations", "exhibitions")
            if con_campos_nuevos
            else ("description", "did_you_know", "provenance")
        )
        evidencia[artwork_id] = [
            (f"{artwork_id}::{s}::0", s, f"texto de {s} de {artwork_id}") for s in secciones
        ]
        citas[artwork_id] = [
            (f"{artwork_id}::description::0", f"texto de description de {artwork_id}")
        ]
    return _con_evidencia(corrida, evidencia, citas)

def test_la_tasa_de_uso_por_section_distingue_ofrecido_de_citado():
    corridas = [_brazo("c01-visual", ["cma:1", "cma:2"], config_id="b" * 16,
                       con_campos_nuevos=True)]
    medida = met.generacion_medida(corridas)
    assert medida.evidencia_por_seccion == {"citations": 2, "description": 2, "exhibitions": 2}
    assert medida.referencias_por_seccion == {"description": 2}
    uso = medida.tasa_de_uso_por_seccion()
    assert uso["description"] == pytest.approx(1.0)
    assert uso["citations"] == pytest.approx(0.0)
    assert uso["exhibitions"] == pytest.approx(0.0)
    assert medida.obras_con_evidencia_por_seccion == {
        "citations": 2, "description": 2, "exhibitions": 2
    }

def test_una_section_que_nadie_ofrecio_da_ausente_y_no_cero():
    corridas = [_brazo("c01-visual", ["cma:1"], config_id="b" * 16, con_campos_nuevos=False)]
    medida = met.generacion_medida(corridas)
    assert "citations" not in medida.evidencia_por_seccion
    assert medida.tasa_de_uso_por_seccion().get("citations") is None

def test_una_referencia_sin_evidencia_cuenta_como_colgada_y_no_se_le_inventa_la_section():
    corrida = corrida_de_prueba(consulta_id="c01-visual", top5=["cma:1"], configuracion="C3")
    corrida = _con_evidencia(
        corrida,
        {"cma:1": [("cma:1::description::0", "description", "el texto")]},
        {"cma:1": [("cma:1::citations::7", "un texto que nadie recuperó")]},
    )
    medida = met.generacion_medida([corrida])
    assert medida.n_referencias_colgadas == 1
    assert medida.referencias_por_seccion == {met.SECCIÓN_SIN_EVIDENCIA: 1}
    assert "citations" not in medida.referencias_por_seccion
    assert medida.n_citas_byte_a_byte == 0

def test_la_cita_que_no_es_byte_a_byte_no_se_cuenta_aunque_el_fragmento_exista():
    corrida = corrida_de_prueba(consulta_id="c01-visual", top5=["cma:1"], configuracion="C3")
    corrida = _con_evidencia(
        corrida,
        {"cma:1": [("cma:1::description::0", "description", "el texto original")]},
        {"cma:1": [("cma:1::description::0", "el texto PARAFRASEADO")]},
    )
    medida = met.generacion_medida([corrida])
    assert medida.n_referencias == 1
    assert medida.n_referencias_colgadas == 0
    assert medida.n_citas_byte_a_byte == 0

def test_sin_corridas_falla_en_vez_de_publicar_una_tabla_vacia():
    with pytest.raises(ErrorDelProyecto):
        met.generacion_medida([])

def test_los_pares_comunes_aislan_el_desplazamiento_del_cambio_de_top5():
    a = _brazo("c01-visual", ["cma:1", "cma:2"], config_id="a" * 16, con_campos_nuevos=False)
    b = _brazo("c01-visual", ["cma:1", "cma:9"], config_id="b" * 16, con_campos_nuevos=True)
    comunes = met.pares_comunes([a], [b])
    assert comunes == frozenset({("c01-visual", "cma:1")})
    assert met.generacion_medida([b]).n_obras == 2
    restringida = met.generacion_medida([b], pares=comunes)
    assert restringida.n_obras == 1
    assert restringida.evidencia_por_seccion == {
        "citations": 1, "description": 1, "exhibitions": 1
    }
    assert restringida.referencias_por_seccion == {"description": 1}

def _brazo_ciego(consulta_id, top5, *, config_id):
    corrida = corrida_de_prueba(
        consulta_id=consulta_id, top5=top5, configuracion="C3", config_id=config_id
    )
    corrida.contenido["configuracion_resuelta"].setdefault("fragmentacion", {})[
        "estrategia"
    ] = "ciego"
    evidencia, citas = {}, {}
    for artwork_id in top5:
        fid = f"{artwork_id}::description::0"
        texto = f"description de {artwork_id}\n\nprovenance de {artwork_id}"
        evidencia[artwork_id] = [(fid, "description", texto)]
        citas[artwork_id] = [(fid, texto)]
    corrida = _con_evidencia(corrida, evidencia, citas)
    for fragmentos in corrida.contenido["estado"]["evidencia"].values():
        for f in fragmentos:
            f["campos"] = ["description", "provenance"]
            f["licencias"] = ["CC-BY-4.0", "CC0"]
            f["license"] = "CC-BY-4.0"
    for obra in corrida.contenido["estado"]["salida"]["obras"]:
        for ref in obra["referencias"]:
            ref["campos"] = ["description", "provenance"]
            ref["licencias"] = ["CC-BY-4.0", "CC0"]
            ref["license"] = "CC-BY-4.0"
            ref["atribucion"] = "“t”, a. Art Institute of Chicago. Texto bajo CC BY 4.0 + CC0 1.0 — https://f"
    return corrida

def test_el_brazo_ciego_cuenta_los_fragmentos_que_mezclan_licencias():
    medida = met.generacion_medida(
        [_brazo_ciego("c01-visual", ["aic:1", "aic:2"], config_id="c" * 16)]
    )
    assert medida.estrategia_de_chunking == "ciego"
    assert medida.n_fragmentos_que_mezclan_licencias == 2
    assert medida.n_referencias_que_mezclan_licencias == 2
    assert medida.evidencia_por_combinacion_de_licencias == {"CC-BY-4.0 + CC0": 2}
    ejemplo = medida.ejemplos_de_mezcla_citada[0]
    assert ejemplo["licencias"] == ["CC-BY-4.0", "CC0"]
    assert "CC BY 4.0 + CC0 1.0" in ejemplo["atribucion"]

def test_el_brazo_por_campo_da_cero_mezcla_y_eso_es_por_construccion():
    medida = met.generacion_medida(
        [_brazo("c01-visual", ["cma:1"], config_id="d" * 16, con_campos_nuevos=False)]
    )
    assert medida.n_fragmentos_que_mezclan_licencias == 0
    assert medida.n_referencias_que_mezclan_licencias == 0
    assert medida.evidencia_por_combinacion_de_licencias == {"CC0": 3}

def test_sin_la_clave_licencias_la_union_es_el_singular():
    corrida = _brazo("c01-visual", ["cma:1"], config_id="e" * 16, con_campos_nuevos=False)
    for fragmentos in corrida.contenido["estado"]["evidencia"].values():
        for f in fragmentos:
            assert "licencias" not in f
    medida = met.generacion_medida([corrida])
    assert medida.evidencia_por_combinacion_de_licencias == {"CC0": 3}

def test_los_tokens_de_evidencia_por_obra_son_el_confundidor_de_mg_2_medido():
    corrida = _brazo_ciego("c01-visual", ["aic:1"], config_id="f" * 16)
    texto = corrida.contenido["estado"]["evidencia"]["aic:1"][0]["texto"]
    medida = met.generacion_medida([corrida], medidor=len)
    assert medida.tokens_de_evidencia == len(texto)
    assert medida.tokens_de_evidencia_por_obra == float(len(texto))
    assert medida.caracteres_de_evidencia == len(texto)
    assert medida.obras_por_cantidad_de_fragmentos == {1: 1}

def test_sin_medidor_los_tokens_son_AUSENTES_y_no_cero():
    medida = met.generacion_medida([_brazo_ciego("c01-visual", ["aic:1"], config_id="0" * 16)])
    assert medida.tokens_de_evidencia is None
    assert medida.tokens_de_evidencia_por_obra is None
    assert medida.caracteres_de_evidencia > 0

def test_m13_la_ganancia_del_ndcg_es_el_grado_sin_remapear():
    assert met.dcg([2, 1]) == pytest.approx(2.63093, abs=1e-5)
    assert met.dcg([2, 1]) != pytest.approx(3.63093, abs=1e-5)

def test_m13_el_ndcg_de_una_consulta_con_varianza_cero_es_AUSENTE_y_no_cero():
    assert met.ndcg_at_k([0, 0, 0, 0, 0], [0] * 15) is None
    assert met.ndcg_at_k([0, 0, 0, 0, 0], [2, 1, 0]) == 0.0

def test_m13_p_at_5_divide_por_5_y_no_por_lo_que_entregó():
    assert met.p_at_k([2, 1, 1]) == pytest.approx(3 / 5)
    assert met.p_at_k([2, 1, 1], umbral=2) == pytest.approx(1 / 5)

def test_m13_mrr_es_cero_cuando_no_hay_relevante_y_eso_si_es_un_valor():
    assert met.rr([0, 0, 1, 2]) == pytest.approx(1 / 3)
    assert met.rr([0, 0, 0, 0, 0]) == 0.0

def test_m13_las_dos_politicas_de_sin_grado_no_dan_lo_mismo():
    corrida = corrida_de_prueba(consulta_id="c01-visual", top5=["a", "b", "c", "d", "e"])
    gt = met.GroundTruthLeído(
        pool_id="p", rubrica_id="v1", poblacion=met.POBLACIÓN_RESTRINGIDA,
        resoluciones=("acuerdo",),
        grados={("c01-visual", "a"): 0, ("c01-visual", "d"): 2, ("c01-visual", "e"): 2},
        por_resolucion={"acuerdo": 3}, n_filas_del_consenso=5, n_sin_grado=0, archivo="x",
    )
    con_cero = met.juzgar_ranking(corrida, gt=gt, categoria="visual_cromatico", politica="cero")
    condensada = met.juzgar_ranking(
        corrida, gt=gt, categoria="visual_cromatico", politica="condensada"
    )
    assert con_cero.ganancias == (0, 0, 0, 2, 2)
    assert condensada.ganancias == (0, 2, 2)
    assert con_cero.sin_grado == condensada.sin_grado == 2
    assert condensada.ndcg > con_cero.ndcg
    assert condensada.p5 == con_cero.p5 == pytest.approx(2 / 5)

def test_m13_una_politica_desconocida_es_un_error_y_no_un_default():
    with pytest.raises(ErrorDelProyecto, match="no es una de"):
        met.juzgar_ranking(
            corrida_de_prueba(consulta_id="c01-visual", top5=["a"]),
            gt=met.GroundTruthLeído(
                pool_id="p", rubrica_id="v1", poblacion="x", resoluciones=(),
                grados={}, por_resolucion={}, n_filas_del_consenso=0, n_sin_grado=0,
                archivo="x",
            ),
            categoria="ambiguo",
            politica="promediar",
        )

def test_m13_el_ideal_se_computa_sobre_la_misma_poblacion_que_el_ranking():
    gt = met.GroundTruthLeído(
        pool_id="p", rubrica_id="v1", poblacion="x", resoluciones=(),
        grados={("q", "a"): 2, ("q", "b"): 1, ("otra", "z"): 2},
        por_resolucion={}, n_filas_del_consenso=0, n_sin_grado=0, archivo="x",
    )
    assert gt.ideales_de("q") == (2, 1)
    assert gt.ideales_de("otra") == (2,)
    assert gt.ideales_de("no-existe") == ()

def test_m13_la_fila_publica_los_dos_promedios_de_ndcg_con_su_poblacion():
    gt = met.GroundTruthLeído(
        pool_id="p", rubrica_id="v1", poblacion=met.POBLACIÓN_COMPLETA,
        resoluciones=("acuerdo", "desempate"),
        grados={
            ("c01-visual", "a"): 2, ("c01-visual", "b"): 0,
            ("c01-ambiguo", "x"): 0, ("c01-ambiguo", "y"): 0,
        },
        por_resolucion={"acuerdo": 4}, n_filas_del_consenso=4, n_sin_grado=0, archivo="x",
    )
    fila = met.fila_de_ranking(
        [
            corrida_de_prueba(consulta_id="c01-visual", top5=["a", "b"]),
            corrida_de_prueba(consulta_id="c01-ambiguo", top5=["x", "y"]),
        ],
        gt=gt,
        categoria_de=CATEGORÍAS,
    )
    assert fila.sin_definir == ("c01-ambiguo",)
    assert len(fila.definidas) == 1
    assert fila.ndcg_sobre_definidas == pytest.approx(1.0)
    assert fila.ndcg_sobre_todas == pytest.approx(0.5)
    assert set(fila.por_categoria()) == {"visual_cromatico", "ambiguo"}

def test_m13_la_sensibilidad_publica_el_ORDEN_y_dice_si_cambia():
    def fila(nombre, ndcg):
        gt = met.GroundTruthLeído(
            pool_id="p", rubrica_id="v1", poblacion="x", resoluciones=(),
            grados={("c01-visual", "a"): 2 if ndcg > 0.5 else 0},
            por_resolucion={}, n_filas_del_consenso=1, n_sin_grado=0, archivo="x",
        )
        return met.fila_de_ranking(
            [corrida_de_prueba(consulta_id="c01-visual", top5=["a"], configuracion=nombre)],
            gt=gt,
            categoria_de=CATEGORÍAS,
        )

    igual = met.comparar_poblaciones(
        [fila("C4", 0.9), fila("C0", 0.1)], [fila("C4", 0.9), fila("C0", 0.1)]
    )
    assert igual["orden_en_la_completa"] == igual["orden_en_la_restringida"] == ["C4", "C0"]
    assert igual["el_orden_cambia"] is False
    assert "queda acotada" in igual["lectura"]
    cambia = met.comparar_poblaciones(
        [fila("C4", 0.9), fila("C0", 0.1)], [fila("C4", 0.1), fila("C0", 0.9)]
    )
    assert cambia["el_orden_cambia"] is True
    assert "CAMBIA" in cambia["lectura"]

def test_m13_el_orden_desempata_por_nombre_y_no_por_el_iterador():
    gt = met.GroundTruthLeído(
        pool_id="p", rubrica_id="v1", poblacion="x", resoluciones=(),
        grados={("c01-visual", "a"): 2}, por_resolucion={}, n_filas_del_consenso=1,
        n_sin_grado=0, archivo="x",
    )
    filas = [
        met.fila_de_ranking(
            [corrida_de_prueba(consulta_id="c01-visual", top5=["a"], configuracion=n)],
            gt=gt, categoria_de=CATEGORÍAS,
        )
        for n in ("C3", "C1", "C2")
    ]
    assert met.orden_de_las_filas(filas) == ("C1", "C2", "C3")
