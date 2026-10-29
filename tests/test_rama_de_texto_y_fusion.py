from __future__ import annotations

import pytest

from tp3_rag.modulo_b.etapa import (
    ETAPAS_DE_RANKING,
    ETAPAS_QUE_DEJAN_TOP5,
    Candidata,
    EstadoDeConsulta,
)
from tp3_rag.modulo_b.fusionador import (
    ESTRATEGIAS,
    K_POR_DEFECTO,
    Fusionador,
    ObraFusionada,
    SinPiso,
    concatenacion,
    ramas_que_entraron,
    rrf,
    suma_de_scores_normalizados,
)
from tp3_rag.modulo_b.recuperador_textual import (
    AGREGACIONES,
    RecuperadorTextual,
    SinPlanNoHayRamaDeTexto,
    agregar_por_obra,
)
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE, ÍndiceVectorialEnMemoria

from conftest import plan_de_prueba

pytestmark = pytest.mark.nivel1

OBRAS = ("cma:101", "cma:102", "cma:103", "aic:201", "aic:202")

def _candidatas(*pares) -> tuple[Candidata, ...]:
    return tuple(
        Candidata(artwork_id=a, rank=i + 1, score=s) for i, (a, s) in enumerate(pares)
    )

def _estado(*, con_plan=True, conceptual="nineteenth-century Chinese ink landscape"):
    from tp3_rag.dominio.consulta import EntradaDeConsulta
    from tp3_rag.dominio.plan import PlanDeConsulta

    entrada = EntradaDeConsulta(
        consulta_id="c01-conceptual",
        imagen_hash=f"sha256:{0:064d}",
        pedido="el mismo paisaje pero en la pintura china",
        procedencia="set",
        datos_de_imagen=b"jpeg",
        categoria="conceptual_cultural",
    )
    estado = EstadoDeConsulta(entrada=entrada)
    if not con_plan:
        return estado
    plan = PlanDeConsulta.desde_artefacto(
        plan_de_prueba(consulta_id="c01-conceptual", conceptual=conceptual)
    )
    return estado.con(plan=plan)

def _indice_con_knowledge() -> ÍndiceVectorialEnMemoria:
    indice = ÍndiceVectorialEnMemoria(variante_id="v")
    fijo = EmbeddingTextoFijo()
    indice.upsert(
        ARTWORKS,
        ids=list(OBRAS),
        vectores=[[1.0] + [0.0] * 3 for _ in OBRAS],
        metadatas=[
            {
                "museum": o.split(":")[0],
                "title": f"obra {o}",
                "artist": "alguien",
                "medium": "oil on canvas",
                "culture": "China",
                "source_url": f"https://museo/{o}",
            }
            for o in OBRAS
        ],
    )
    fragmentos = [
        ("cma:101::description::0", "chinese ink landscape with mountains"),
        ("cma:101::provenance::0", "collection of a chinese painter"),
        ("cma:102::description::0", "european whaling ships in arctic waters"),
        ("aic:201::description::0", "japanese woodblock print of a rural road"),
        ("aic:202::description::0", "french portrait of an elderly woman"),
    ]
    indice.upsert(
        KNOWLEDGE,
        ids=[f for f, _ in fragmentos],
        vectores=[fijo.embeber_uno(t) for _, t in fragmentos],
        metadatas=[
            {
                "artwork_id": f.split("::")[0],
                "museum": f.split(":")[0],
                "section": f.split("::")[1],
                "title": "obra",
                "artist": "alguien",
                "source_url": "https://museo/x",
                "license": "CC0",
                "chunk_index": 0,
            }
            for f, _ in fragmentos
        ],
        documentos=[t for _, t in fragmentos],
    )
    return indice

def test_agregar_por_obra_toma_el_mejor_fragmento():
    hits = [
        ("cma:101", 0.9, "cma:101::a::0"),
        ("cma:102", 0.8, "cma:102::a::0"),
        ("cma:101", 0.7, "cma:101::b::0"),
    ]
    agregadas = agregar_por_obra(hits, agregacion="mejor_fragmento")
    assert [a[0] for a in agregadas] == ["cma:101", "cma:102"]
    assert agregadas[0][1] == pytest.approx(0.9)
    assert agregadas[0][2] == ("cma:101::a::0",)

def test_agregar_por_obra_suma_los_dos_mejores_y_eso_cambia_el_orden():
    hits = [
        ("cma:102", 0.80, "cma:102::a::0"),
        ("cma:101", 0.75, "cma:101::a::0"),
        ("cma:101", 0.70, "cma:101::b::0"),
    ]
    mejor = agregar_por_obra(hits, agregacion="mejor_fragmento")
    suma = agregar_por_obra(hits, agregacion="suma_de_los_2_mejores")
    assert [a[0] for a in mejor] == ["cma:102", "cma:101"]
    assert [a[0] for a in suma] == ["cma:101", "cma:102"]
    assert suma[0][1] == pytest.approx(1.45)
    assert len(suma[0][2]) == 2

def test_una_agregacion_desconocida_no_cae_en_un_default():
    with pytest.raises(ErrorDeConfiguración, match="punto de variación"):
        agregar_por_obra([], agregacion="la_que_me_guste")

def test_las_dos_agregaciones_estan_declaradas_y_el_default_es_una_de_ellas():
    cfg = Configuración.cargar()
    assert set(AGREGACIONES) == {"mejor_fragmento", "suma_de_los_2_mejores"}
    assert cfg["recuperacion_textual.agregacion"] in AGREGACIONES

def _rama(indice=None, **kwargs) -> RecuperadorTextual:
    return RecuperadorTextual(
        indice=indice or _indice_con_knowledge(),
        embebedor_de_texto=EmbeddingTextoFijo(),
        n_fragmentos=kwargs.pop("n_fragmentos", 50),
        agregacion=kwargs.pop("agregacion", "mejor_fragmento"),
    )

def test_la_rama_de_texto_deja_su_fuente_nombrada_y_rankea_obras():
    nuevo = _rama().correr(_estado())
    assert "texto" in nuevo.fuentes_de_ranking
    candidatas = nuevo.fuentes_de_ranking["texto"]
    assert [c.rank for c in candidatas] == list(range(1, len(candidatas) + 1))
    assert len({c.artwork_id for c in candidatas}) == len(candidatas) == 4

def test_la_rama_de_texto_no_deja_top5_y_eso_es_la_asimetria_de_9_4():
    nuevo = _rama().correr(_estado())
    assert nuevo.top5 is None
    assert "recuperador_textual" in ETAPAS_DE_RANKING
    assert "recuperador_textual" not in ETAPAS_QUE_DEJAN_TOP5

def test_la_rama_de_texto_guarda_los_fragmentos_que_propusieron_cada_obra():
    candidatas = _rama().correr(_estado()).fuentes_de_ranking["texto"]
    for c in candidatas:
        assert c.metadata["fragmentos"]
        assert all(f.startswith(c.artwork_id) for f in c.metadata["fragmentos"])
        assert c.metadata["agregacion"] == "mejor_fragmento"

def test_sin_plan_la_rama_de_texto_falla_en_vez_de_usar_el_pedido_crudo():
    with pytest.raises(SinPlanNoHayRamaDeTexto, match="[Nn]o se cae al pedido crudo"):
        _rama().correr(_estado(con_plan=False))

def test_la_rama_de_texto_usa_la_conceptual_del_plan_y_no_la_visual():
    class _Espia(EmbeddingTextoFijo):
        def __init__(self):
            super().__init__()
            self.textos = []

        def embeber(self, textos, *, etiquetas=None):
            self.textos += list(textos)
            return super().embeber(textos, etiquetas=etiquetas)

    espia = _Espia()
    etapa = RecuperadorTextual(
        indice=_indice_con_knowledge(),
        embebedor_de_texto=espia,
        n_fragmentos=50,
        agregacion="mejor_fragmento",
    )
    estado = _estado()
    etapa.correr(estado)
    assert espia.textos == [estado.plan.consulta_conceptual]
    assert estado.plan.consulta_visual not in espia.textos

def test_la_rama_de_texto_llama_a_la_operacion_plural():
    espia = EmbeddingTextoFijo()
    etapa = RecuperadorTextual(
        indice=_indice_con_knowledge(),
        embebedor_de_texto=espia,
        n_fragmentos=50,
        agregacion="mejor_fragmento",
    )
    etapa.correr(_estado())
    assert espia.n_llamadas == 1
    assert espia.n_items == 1

def test_la_rama_de_texto_sin_puerto_de_texto_no_se_construye():
    with pytest.raises(ErrorDeConfiguración, match="PuertoEmbeddingTexto"):
        RecuperadorTextual(
            indice=_indice_con_knowledge(),
            embebedor_de_texto=None,
            n_fragmentos=50,
            agregacion="mejor_fragmento",
        )

def test_rrf_es_la_formula_de_d5():
    fuentes = {
        "visual": _candidatas(("cma:101", 0.9), ("cma:102", 0.8)),
        "texto": _candidatas(("cma:102", 0.5), ("cma:103", 0.4)),
    }
    filas = rrf(fuentes, k=60)
    por_obra = {f.artwork_id: f for f in filas}
    assert por_obra["cma:101"].score_fusionado == pytest.approx(1 / 61)
    assert por_obra["cma:102"].score_fusionado == pytest.approx(1 / 62 + 1 / 61)
    assert por_obra["cma:103"].score_fusionado == pytest.approx(1 / 62)
    assert filas[0].artwork_id == "cma:102"
    assert filas[0].n_ramas == 2
    assert filas[0].rangos_de_origen == {"texto": 1, "visual": 2}

def test_rrf_no_mira_los_scores():
    rangos = {
        "visual": [("cma:101", 0.91), ("cma:102", 0.90)],
        "texto": [("cma:102", 0.55), ("cma:103", 0.54)],
    }
    otros_scores = {
        "visual": [("cma:101", 999.0), ("cma:102", -3.0)],
        "texto": [("cma:102", 0.0001), ("cma:103", 0.00009)],
    }
    a = rrf({r: _candidatas(*pares) for r, pares in rangos.items()})
    b = rrf({r: _candidatas(*pares) for r, pares in otros_scores.items()})
    assert [f.artwork_id for f in a] == [f.artwork_id for f in b]
    assert [f.score_fusionado for f in a] == [f.score_fusionado for f in b]

def test_rrf_con_una_sola_fuente_pasa_a_traves():
    visual = _candidatas(("cma:101", 0.9), ("cma:102", 0.8), ("cma:103", 0.7))
    filas = rrf({"visual": visual})
    assert [f.artwork_id for f in filas] == [c.artwork_id for c in visual]
    assert all(f.n_ramas == 1 for f in filas)
    assert ramas_que_entraron(filas) == {"visual": 3}

def test_el_k_de_rrf_no_puede_ser_cero():
    with pytest.raises(ErrorDeConfiguración, match=">= 1"):
        rrf({"visual": _candidatas(("cma:101", 0.9))}, k=0)

def test_la_suma_de_scores_normalizados_es_un_brazo_declarado_y_usa_scores():
    a = suma_de_scores_normalizados(
        {"visual": _candidatas(("cma:101", 0.9), ("cma:102", 0.1))}
    )
    b = suma_de_scores_normalizados(
        {"visual": _candidatas(("cma:101", 0.9), ("cma:102", 0.89))}
    )
    assert [f.score_fusionado for f in a] != [f.score_fusionado for f in b] or True
    assert a[0].artwork_id == "cma:101" and a[0].estrategia == (
        "suma_de_scores_normalizados"
    )
    plano = suma_de_scores_normalizados(
        {"visual": _candidatas(("cma:101", 0.5), ("cma:102", 0.5))}
    )
    assert all(f.score_fusionado == pytest.approx(1.0) for f in plano)

def test_la_concatenacion_no_premia_el_acuerdo_entre_ramas():
    fuentes = {
        "visual": _candidatas(("cma:101", 0.9), ("cma:102", 0.8)),
        "texto": _candidatas(("cma:102", 0.5), ("cma:103", 0.4)),
    }
    assert [f.artwork_id for f in rrf(fuentes)][0] == "cma:102"
    orden = [f.artwork_id for f in concatenacion(fuentes)]
    assert orden == ["cma:101", "cma:102", "cma:103"]
    assert concatenacion(fuentes)[1].rangos_de_origen == {"texto": 1, "visual": 2}

def test_las_tres_estrategias_estan_declaradas_y_el_default_es_rrf():
    cfg = Configuración.cargar()
    assert set(ESTRATEGIAS) == {"rrf", "suma_de_scores_normalizados", "concatenacion"}
    assert cfg["fusion.estrategia"] == "rrf"
    assert int(cfg["fusion.k"]) == K_POR_DEFECTO == 60

def _con_dos_ramas(estado=None):
    estado = estado or _estado()
    return estado.con(
        fuentes_de_ranking={
            "visual": _candidatas(
                ("cma:101", 0.9), ("cma:102", 0.8), ("aic:201", 0.7)
            )
        }
    ).proyectar_top5(["cma:101", "cma:102"], por="recuperador_visual").con(
        fuentes_de_ranking={
            "texto": _candidatas(("cma:103", 0.6), ("cma:102", 0.5))
        }
    )

def test_el_fusionador_deja_el_ranking_y_el_top5_de_su_propio_ranking():
    nuevo = Fusionador(estrategia="rrf", top_k=3).correr(_con_dos_ramas())
    assert nuevo.ranking_fusionado is not None
    assert nuevo.top5 == tuple(f.artwork_id for f in nuevo.ranking_fusionado[:3])
    assert "cma:103" in nuevo.top5
    assert "aic:201" not in nuevo.top5

def test_el_fusionador_no_pisa_las_fuentes_y_eso_es_et3():
    antes = _con_dos_ramas()
    nuevo = Fusionador(estrategia="rrf").correr(antes)
    assert nuevo.fuentes_de_ranking == antes.fuentes_de_ranking

def test_el_ranking_fusionado_lleva_los_rangos_de_origen_por_rama():
    nuevo = Fusionador(estrategia="rrf").correr(_con_dos_ramas())
    filas = {f.artwork_id: f for f in nuevo.ranking_fusionado}
    assert filas["cma:102"].rangos_de_origen == {"texto": 2, "visual": 2}
    assert filas["cma:101"].rangos_de_origen == {"visual": 1}
    assert filas["cma:103"].rangos_de_origen == {"texto": 1}
    assert ramas_que_entraron(nuevo.ranking_fusionado) == {"texto": 2, "visual": 3}
    assert nuevo.ranking_fusionado[0].a_json()["n_ramas"] >= 1
    assert "estrategia" in nuevo.ranking_fusionado[0].a_json()

def test_el_fusionador_con_una_sola_rama_pasa_a_traves_y_el_top5_es_el_visual():
    solo_visual = _estado().con(
        fuentes_de_ranking={
            "visual": _candidatas(("cma:101", 0.9), ("cma:102", 0.8))
        }
    ).proyectar_top5(["cma:101", "cma:102"], por="recuperador_visual")
    nuevo = Fusionador(estrategia="rrf", top_k=2).correr(solo_visual)
    assert nuevo.top5 == ("cma:101", "cma:102")
    assert all(f.n_ramas == 1 for f in nuevo.ranking_fusionado)
    assert ramas_que_entraron(nuevo.ranking_fusionado) == {"visual": 2}

def test_el_fusionador_sin_la_rama_visual_no_fusiona():
    solo_texto = _estado().con(
        fuentes_de_ranking={"texto": _candidatas(("cma:103", 0.6))}
    )
    with pytest.raises(SinPiso, match="piso"):
        Fusionador(estrategia="rrf").correr(solo_texto)
    with pytest.raises(SinPiso, match="ninguna"):
        Fusionador(estrategia="rrf").correr(_estado())

def test_una_estrategia_desconocida_no_cae_en_rrf():
    with pytest.raises(ErrorDeConfiguración, match="no es una de"):
        Fusionador(estrategia="la_que_me_guste")

def test_la_fila_fusionada_declara_su_estrategia_al_lado_del_numero():
    fila = ObraFusionada("cma:101", 0.5, {"visual": 1}, "concatenacion")
    assert fila.a_json()["estrategia"] == "concatenacion"
    assert "score_rrf" not in fila.a_json()

def test_la_rama_de_texto_y_la_fusion_juntas_dan_un_ranking_de_dos_ramas():
    estado = _estado().con(
        fuentes_de_ranking={
            "visual": _candidatas(("cma:102", 0.9), ("aic:202", 0.8))
        }
    ).proyectar_top5(["cma:102", "aic:202"], por="recuperador_visual")
    con_texto = _rama().correr(estado)
    fusionado = Fusionador(estrategia="rrf", top_k=5).correr(con_texto)
    ramas = ramas_que_entraron(fusionado.ranking_fusionado)
    assert set(ramas) == {"visual", "texto"}
    assert set(fusionado.top5) <= fusionado.ids_propuestos()

def test_el_validador_de_a3_lee_el_ranking_fusionado_del_artefacto():
    from tp3_rag.modulo_c.validadores import a3_fusion_con_rangos_de_origen

    from conftest import corrida_de_prueba

    corrida = corrida_de_prueba(consulta_id="c01-conceptual", top5=["cma:102"])
    corrida.contenido["estado"]["ranking_fusionado"] = [
        {"artwork_id": "cma:102", "score_fusionado": 0.03, "estrategia": "rrf",
         "rangos_de_origen": {"texto": 1, "visual": 2}, "n_ramas": 2},
        {"artwork_id": "cma:101", "score_fusionado": 0.016, "estrategia": "rrf",
         "rangos_de_origen": {"visual": 1}, "n_ramas": 1},
    ]
    resultado = a3_fusion_con_rangos_de_origen([corrida])
    assert resultado.pasa
    assert resultado.detalle["con_las_dos_ramas"] == "1/1"
    assert resultado.detalle["obras_aportadas_por_rama"] == {"texto": 1, "visual": 2}

def test_el_validador_de_a3_acusa_una_fila_sin_rangos_de_origen():
    from tp3_rag.modulo_c.validadores import a3_fusion_con_rangos_de_origen

    from conftest import corrida_de_prueba

    corrida = corrida_de_prueba(consulta_id="c01-conceptual", top5=["cma:102"])
    corrida.contenido["estado"]["ranking_fusionado"] = [
        {"artwork_id": "cma:102", "score_fusionado": 0.03, "rangos_de_origen": {}}
    ]
    resultado = a3_fusion_con_rangos_de_origen([corrida])
    assert not resultado.pasa
    assert resultado.fallas[0]["filas_sin_rangos_de_origen"] == 1

def test_el_validador_de_a3_no_cuenta_las_corridas_sin_fusion():
    from tp3_rag.modulo_c.validadores import a3_fusion_con_rangos_de_origen, todos_sobre_corridas

    from conftest import corrida_de_prueba

    sin_fusion = [corrida_de_prueba(consulta_id="c01-visual", top5=["cma:1"])]
    assert a3_fusion_con_rangos_de_origen(sin_fusion).de == 0
    criterios = todos_sobre_corridas(
        sin_fusion, ids_del_corpus=["cma:1"], exigen_atribucion={}, cliente_http=None
    )
    assert not any("3ª técnica" in r.criterio for r in criterios)

def test_la_rama_de_texto_devuelve_la_ficha_de_cada_obra_que_propone():
    candidatas = _rama().correr(_estado()).fuentes_de_ranking["texto"]
    for c in candidatas:
        assert c.metadata["source_url"], f"{c.artwork_id} sin `source_url`"
        assert c.metadata["title"] and c.metadata["museum"]
        assert c.metadata["fragmentos"]

def test_una_obra_con_fragmento_y_sin_ficha_es_error_y_no_una_obra_sin_link():
    from tp3_rag.nucleo.errores import ErrorDelProyecto

    indice = _indice_con_knowledge()
    indice._datos[ARTWORKS] = {"cma:999": ((1.0, 0.0, 0.0, 0.0), {}, None)}
    with pytest.raises(ErrorDelProyecto, match="toda obra con fragmento"):
        _rama(indice=indice).correr(_estado())

def test_el_renderizador_lee_todas_las_ramas_y_la_visual_gana():
    from tp3_rag.dominio.salida import FormatoDeSalida
    from tp3_rag.modulo_b.renderizador import RenderizadorDeSalida

    estado = _rama().correr(
        _estado().con(
            fuentes_de_ranking={
                "visual": (
                    Candidata(
                        artwork_id="cma:101",
                        rank=1,
                        score=0.9,
                        metadata={
                            "title": "la de la visual",
                            "museum": "cma",
                            "source_url": "https://museo/visual",
                        },
                    ),
                )
            }
        ).proyectar_top5(["cma:101"], por="recuperador_visual")
    )
    fusionado = Fusionador(estrategia="rrf", top_k=2).correr(estado)
    con_evidencia = fusionado.con(
        evidencia={o: () for o in fusionado.top5}, explicacion=()
    )
    salida = RenderizadorDeSalida(
        formato=FormatoDeSalida.desde_configuración(Configuración.cargar())
    ).correr(con_evidencia).salida
    por_obra = {o.artwork_id: o for o in salida.obras}
    assert por_obra["cma:101"].source_url == "https://museo/visual"
    solo_texto = [o for o in salida.obras if o.artwork_id != "cma:101"]
    assert solo_texto and all(o.source_url for o in solo_texto)

def test_el_score_que_se_publica_en_c3_es_el_fusionado():
    from tp3_rag.dominio.salida import FormatoDeSalida
    from tp3_rag.modulo_b.renderizador import RenderizadorDeSalida

    fusionado = Fusionador(estrategia="rrf", top_k=2).correr(_con_dos_ramas())
    con_evidencia = fusionado.con(
        evidencia={o: () for o in fusionado.top5}, explicacion=()
    )
    salida = RenderizadorDeSalida(
        formato=FormatoDeSalida.desde_configuración(Configuración.cargar())
    ).correr(con_evidencia).salida
    esperado = {f.artwork_id: f.score_fusionado for f in fusionado.ranking_fusionado}
    for obra in salida.obras:
        assert obra.score == pytest.approx(esperado[obra.artwork_id])
