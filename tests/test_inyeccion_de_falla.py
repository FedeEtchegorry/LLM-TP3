from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.corrida import AlmacenDeCorridas
from tp3_rag.dominio.plan import AlmacenDePlanes
from tp3_rag.modulo_b.armador import ArmadorDePipeline
from tp3_rag.modulo_b.ejecutor import MODOS_DE_FALLA, EjecutorDeEtapas
from tp3_rag.modulo_c import validadores as val
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import (
    ARTWORKS,
    KNOWLEDGE,
    ÍndiceVectorialEnMemoria,
)
from tp3_rag.puertos.llm import LLMGrabado

pytestmark = pytest.mark.nivel3

ETAPAS_CON_FALLBACK_EN_C1 = ("recolector_de_evidencia", "generador")

ETAPAS_CON_FALLBACK_EN_C2 = ("planificador",) + ETAPAS_CON_FALLBACK_EN_C1

ETAPAS_CON_FALLBACK_EN_C3 = (
    "recuperador_textual",
    "fusionador",
) + ETAPAS_CON_FALLBACK_EN_C2

ETAPAS_DE_C3_SIN_CASCADA = tuple(
    e for e in ETAPAS_CON_FALLBACK_EN_C3 if e != "planificador"
)

ETAPAS_CON_FALLBACK_EN_C4 = ("reranker",) + ETAPAS_CON_FALLBACK_EN_C3

ETAPAS_DE_C4_SIN_CASCADA = tuple(
    e for e in ETAPAS_CON_FALLBACK_EN_C4 if e not in ("planificador", "fusionador")
)

CADENA_C2 = (
    "planificador",
    "recuperador_visual",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
)

CADENA_C3 = (
    "planificador",
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
)

CADENA_C4 = (
    "planificador",
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "reranker",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
)

CADENA_C0 = ("recuperador_visual", "renderizador")

PLAN_GRABADO = {
    "consulta_visual": "a quiet landscape near Arkville with warm afternoon light",
    "consulta_conceptual": "nineteenth-century American pastoral landscape painting",
}

N_IMAGENES, N_PEDIDOS = 10, 3
OBRAS = [f"cma:2000{i}" for i in range(20)]

def _indice() -> ÍndiceVectorialEnMemoria:
    img, txt = EmbeddingImagenFijo(), EmbeddingTextoFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v-p7")
    idx.upsert(
        ARTWORKS,
        ids=OBRAS,
        vectores=[img.embeber_imagen(f"img-{o}".encode()) for o in OBRAS],
        metadatas=[
            {"museum": "cma", "title": f"T{o}", "artist": "A", "medium": "oil",
             "source_url": f"https://clevelandart.org/art/{o}"}
            for o in OBRAS
        ],
    )
    frs, vs, ms, ds = [], [], [], []
    for o in OBRAS:
        texto = f"A landscape near Arkville, {o}."
        frs.append(f"{o}::description::0")
        vs.append(txt.embeber_uno(texto))
        ds.append(texto)
        ms.append({
            "artwork_id": o, "museum": "cma", "section": "description", "title": f"T{o}",
            "artist": "A", "source_url": f"https://clevelandart.org/art/{o}",
            "license": "CC-BY-4.0", "chunk_index": 0,
        })
    idx.upsert(KNOWLEDGE, ids=frs, vectores=vs, metadatas=ms, documentos=ds)
    return idx

class _LLMQueExplicaLoQueRecibe(LLMGrabado):
    def invocar(self, binding, mensajes, imagenes=()):
        usuario = [m for m in mensajes if m.get("rol") == "usuario"][0]["texto"]
        if binding.rol == "reranker":
            artwork_id = usuario.split("artwork_id: ", 1)[1].splitlines()[0].strip()
            self._respuestas = {
                "reranker": [
                    {
                        "score": sum(ord(c) for c in artwork_id) % 11,
                        "justificacion": f"cumple en parte para {artwork_id}",
                    }
                ]
            }
            return super().invocar(binding, mensajes, imagenes)
        if binding.rol == "planificador":
            self._respuestas = {
                "planificador": [
                    {
                        "consulta_visual": f"{PLAN_GRABADO['consulta_visual']}:: {usuario[-12:]}",
                        "consulta_conceptual": PLAN_GRABADO["consulta_conceptual"],
                    }
                ]
            }
            return super().invocar(binding, mensajes, imagenes)
        obras = []
        for línea in usuario.splitlines():
            if línea.startswith("### Obra "):
                obras.append(línea.rsplit("artwork_id: ", 1)[1].strip())
        sin_evidencia = "sin fragmentos de evidencia" in usuario
        self._respuestas = {
            binding.rol: [{
                "obras": [
                    {
                        "artwork_id": o,
                        "afirmaciones": (
                            []
                            if sin_evidencia
                            else [{
                                "texto": "Es un paisaje cerca de Arkville.",
                                "tipo": "dato",
                                "refs": [f"{o}::description::0"],
                            }]
                        ),
                        **({"abstencion": "La evidencia no alcanza."} if sin_evidencia else {}),
                    }
                    for o in obras
                ]
            }]
        }
        return super().invocar(binding, mensajes, imagenes)

CADENAS = {
    "C0": CADENA_C0,
    "C1": None,
    "C2": CADENA_C2,
    "C3": CADENA_C3,
    "C4": CADENA_C4,
}

OVERRIDES_POR_FILA = {"C0": {"consulta.alfa": 1.0, "consulta.alfa_congelado": True}}

def _correr_las_30(tmp_path, *, plan_de_falla=None, configuracion="C1", n_candidatos=None):
    cadena = CADENAS[configuracion]
    cfg = Configuración.cargar(
        overrides={
            "evidencia.orden": "por_seccion_declarada",
            **(
                {
                    "consulta.configuracion": configuracion,
                    "consulta.etapas": list(cadena),
                }
                if cadena
                else {}
            ),
            **OVERRIDES_POR_FILA.get(configuracion, {}),
            **({"consulta.n_candidatos": n_candidatos} if n_candidatos else {}),
            **({"consulta.plan_de_falla": plan_de_falla} if plan_de_falla else {}),
        }
    )
    obs = Observabilidad(silencioso=True)
    almacén = AlmacenamientoEnArchivos(tmp_path)
    armador = ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=_indice(),
        puerto_de_llm=_LLMQueExplicaLoQueRecibe({}),
        embebedor_de_texto=EmbeddingTextoFijo(),
        almacen_de_planes=AlmacenDePlanes(almacén, "artifacts/planes"),
        bytes_de_la_imagen=lambda artwork_id: f"img-{artwork_id}".encode(),
        observabilidad=obs,
    )
    ejecutor = EjecutorDeEtapas(
        cfg,
        pipeline=armador.armar(),
        almacen_de_corridas=AlmacenDeCorridas(almacén, "artifacts/corridas"),
        observabilidad=obs,
        variante_id="v-p7",
        split_id="s-p7",
    )
    corridas = []
    for i in range(N_IMAGENES):
        for j in range(N_PEDIDOS):
            entrada = EntradaDeConsulta.ad_hoc(
                datos_de_imagen=f"img-{OBRAS[i]}".encode(),
                pedido=f"pedido {j} de la imagen {i}",
            )
            corridas.append(ejecutor.correr(entrada))
    return corridas

def test_las_30_sin_falla_inyectada_dan_5_obras_con_explicacion_citada(tmp_path):
    corridas = _correr_las_30(tmp_path)
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(c.degradaciones == [] for c in corridas)
    assert all(
        all(o["referencias"] for o in c.salida["obras"]) for c in corridas
    )
    assert all(c.agrega_a_la_tabla for c in corridas)

@pytest.mark.parametrize("etapa", ETAPAS_CON_FALLBACK_EN_C1)
@pytest.mark.parametrize("modo", MODOS_DE_FALLA)
def test_p7_30_30_devuelven_5_obras_con_falla_inyectada_por_etapa(tmp_path, etapa, modo):
    corridas = _correr_las_30(tmp_path, plan_de_falla={etapa: modo})
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(len(c.salida["obras"]) == 5 for c in corridas)
    assert all(
        [d["etapa"] for d in c.degradaciones if d["planificada"]] == [etapa]
        for c in corridas
    )
    assert all(c.degradaciones_no_planificadas == [] for c in corridas)
    assert all(c.fallida is None for c in corridas)
    assert all(c.contenido["invariante_et6"]["cumple_et6"] for c in corridas)

@pytest.mark.parametrize("etapa", ETAPAS_CON_FALLBACK_EN_C1)
def test_el_fallback_declarado_es_el_que_queda_en_el_artefacto(tmp_path, etapa):
    corrida = _correr_las_30(tmp_path, plan_de_falla={etapa: "excepcion"})[0]
    degradación = [d for d in corrida.degradaciones if d["etapa"] == etapa][0]
    assert "inyectada" in degradación["causa"]
    assert degradación["politica"]
    if etapa == "generador":
        assert corrida.estado["explicacion"] is None
        assert all(o["lineas"] == [] for o in corrida.salida["obras"])
        assert all(o["abstencion"] for o in corrida.salida["obras"])
    else:
        assert corrida.estado["evidencia"] is None
        assert all(o["abstencion"] for o in corrida.salida["obras"])
    assert all(o["source_url"] for o in corrida.salida["obras"])

def test_las_30_de_c2_dan_5_obras_con_explicacion_citada_y_un_plan_cada_una(tmp_path):
    corridas = _correr_las_30(tmp_path, configuracion="C2")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(c.degradaciones == [] for c in corridas)
    assert all(c.plan is not None for c in corridas)
    assert all(c.plan["cumple_pl3"] for c in corridas), "se rompió las tres cláusulas de las consultas del plan en alguna de las 30"
    assert all(c.agrega_a_la_tabla for c in corridas)
    assert len({c.plan_id for c in corridas}) == 30
    assert sum(1 for c in corridas if c.plan_reusado) == 0

@pytest.mark.parametrize("etapa", ETAPAS_CON_FALLBACK_EN_C2)
@pytest.mark.parametrize("modo", MODOS_DE_FALLA)
def test_p7_sobre_c2_30_30_devuelven_5_obras_con_falla_por_etapa(tmp_path, etapa, modo):
    corridas = _correr_las_30(tmp_path, plan_de_falla={etapa: modo}, configuracion="C2")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(len(c.salida["obras"]) == 5 for c in corridas)
    assert all(
        [d["etapa"] for d in c.degradaciones if d["planificada"]] == [etapa]
        for c in corridas
    )
    assert all(c.degradaciones_no_planificadas == [] for c in corridas)
    assert all(c.fallida is None for c in corridas)
    assert all(c.contenido["invariante_et6"]["cumple_et6"] for c in corridas)

def test_el_planificador_caido_degrada_exactamente_a_c1(tmp_path):
    c1 = _correr_las_30(tmp_path / "c1")
    c2_caida = _correr_las_30(
        tmp_path / "c2", plan_de_falla={"planificador": "excepcion"}, configuracion="C2"
    )
    assert [c.top5 for c in c2_caida] == [c.top5 for c in c1]
    assert all(c.plan is None for c in c2_caida)
    assert all(
        d["politica"].startswith("**pedido crudo")
        for c in c2_caida
        for d in c.degradaciones
    )

def test_c2_sana_y_c1_no_dan_el_mismo_top5_o_el_pre_retrieval_no_hizo_nada(tmp_path):
    c1 = _correr_las_30(tmp_path / "c1")
    c2 = _correr_las_30(tmp_path / "c2", configuracion="C2")
    assert [c.top5 for c in c2] != [c.top5 for c in c1]

def test_et2_una_corrida_degradada_tiene_la_misma_forma_que_una_omision(tmp_path):
    sana = _correr_las_30(tmp_path / "a")[0]
    degradada = _correr_las_30(tmp_path / "b", plan_de_falla={"generador": "excepcion"})[0]
    assert set(sana.estado) == set(degradada.estado)
    assert sana.cadena == degradada.cadena
    assert sana.degradaciones == [] and degradada.degradaciones != []

def test_las_30_de_c3_dan_5_obras_con_un_ranking_de_dos_ramas(tmp_path):
    corridas = _correr_las_30(tmp_path, configuracion="C3")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(c.degradaciones == [] for c in corridas)
    assert all(c.agrega_a_la_tabla for c in corridas)
    for c in corridas:
        fuentes = c.estado["fuentes_de_ranking"]
        assert set(fuentes) == {"visual", "texto"}
        filas = c.estado["ranking_fusionado"]
        assert filas and all(f["rangos_de_origen"] for f in filas)
        assert max(f["n_ramas"] for f in filas) == 2

def test_el_top5_de_c3_lo_deja_el_fusionador_y_et6_lo_verifica(tmp_path):
    corrida = _correr_las_30(tmp_path, configuracion="C3")[0]
    et6 = corrida.contenido["invariante_et6"]
    assert et6["cumple_et6"] is True
    assert et6["top5"]["contra"] == "fusionador"
    assert corrida.top5 == [
        f["artwork_id"] for f in corrida.estado["ranking_fusionado"][:5]
    ]

@pytest.mark.parametrize("etapa", ETAPAS_DE_C3_SIN_CASCADA)
@pytest.mark.parametrize("modo", MODOS_DE_FALLA)
def test_p7_sobre_c3_30_30_devuelven_5_obras_con_falla_por_etapa(tmp_path, etapa, modo):
    corridas = _correr_las_30(tmp_path, plan_de_falla={etapa: modo}, configuracion="C3")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(len(c.salida["obras"]) == 5 for c in corridas)
    assert all(
        [d["etapa"] for d in c.degradaciones if d["planificada"]] == [etapa]
        for c in corridas
    )
    assert all(c.degradaciones_no_planificadas == [] for c in corridas)
    assert all(c.fallida is None for c in corridas)
    assert all(c.contenido["invariante_et6"]["cumple_et6"] for c in corridas)

def test_la_rama_de_texto_caida_deja_la_fusion_pasando_a_traves_con_la_visual(tmp_path):
    solo_visual = _correr_las_30(
        tmp_path / "a", plan_de_falla={"recuperador_textual": "excepcion"},
        configuracion="C3",
    )
    for c in solo_visual:
        assert set(c.estado["fuentes_de_ranking"]) == {"visual"}
        filas = c.estado["ranking_fusionado"]
        assert filas and all(f["n_ramas"] == 1 for f in filas)
        assert all(set(f["rangos_de_origen"]) == {"visual"} for f in filas)
    sin_fusion = _correr_las_30(
        tmp_path / "b",
        plan_de_falla={"recuperador_textual": "excepcion", "fusionador": "excepcion"},
        configuracion="C3",
    )
    assert [c.top5 for c in solo_visual] == [c.top5 for c in sin_fusion]

def test_la_fusion_caida_deja_el_top5_de_la_visual_y_et6_lo_acepta(tmp_path):
    con_fusion_caida = _correr_las_30(
        tmp_path / "a", plan_de_falla={"fusionador": "excepcion"}, configuracion="C3"
    )
    c3_entera = _correr_las_30(tmp_path / "b", configuracion="C3")
    for c in con_fusion_caida:
        et6 = c.contenido["invariante_et6"]
        assert et6["cumple_et6"] is True
        assert et6["top5"]["contra"] == "recuperador_visual"
        assert set(c.estado["fuentes_de_ranking"]) == {"visual", "texto"}
        assert c.estado["ranking_fusionado"] is None
    assert [c.top5 for c in con_fusion_caida] != [c.top5 for c in c3_entera]

def test_las_30_de_c4_dan_5_obras_con_el_top5_reordenado_por_el_reranker(tmp_path):
    corridas = _correr_las_30(tmp_path, configuracion="C4")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(c.degradaciones == [] for c in corridas)
    assert all(c.agrega_a_la_tabla for c in corridas)
    for c in corridas:
        rerank = c.estado["rerank"]
        assert rerank and all(f["justificacion"] for f in rerank)
        assert all(0 <= f["score_0_10"] <= 10 for f in rerank)
        assert c.estado["ranking_fusionado"]
        assert all(f["score_fusionado"] for f in c.estado["ranking_fusionado"])
        assert c.top5 == [f["artwork_id"] for f in rerank][:5]

def test_el_top5_de_c4_lo_deja_el_reranker_y_et6_lo_verifica(tmp_path):
    corrida = _correr_las_30(tmp_path, configuracion="C4")[0]
    et6 = corrida.contenido["invariante_et6"]
    assert et6["cumple_et6"] is True
    assert et6["top5"]["contra"] == "reranker"

def test_c4_sana_y_c3_no_dan_el_mismo_top5_o_el_reranking_no_hizo_nada(tmp_path):
    c3 = _correr_las_30(tmp_path / "c3", configuracion="C3")
    c4 = _correr_las_30(tmp_path / "c4", configuracion="C4")
    assert [c.top5 for c in c4] != [c.top5 for c in c3]

def test_p6_c4_son_2_mas_n_llamadas_al_llm_por_consulta(tmp_path):
    n = 7
    corridas = _correr_las_30(tmp_path, configuracion="C4", n_candidatos=n)
    for c in corridas:
        libro = c.contenido["libro_mayor"]
        llamadas = sum(
            libro.get(rol, {}).get("n_llamadas", 0)
            for rol in ("planificador", "reranker", "generador")
        )
        assert libro["reranker"]["n_llamadas"] == n
        assert llamadas == 2 + n
        assert "embedding_texto" not in val.ROLES_DEL_LLM
    resultado = val.p6_llamadas_al_llm(corridas, n_candidatos=n)
    assert resultado.pasa
    assert resultado.detalle["por_configuracion"]["C4"][
        "llamadas_nominales_por_consulta"
    ] == 2 + n

@pytest.mark.parametrize("etapa", ETAPAS_DE_C4_SIN_CASCADA)
@pytest.mark.parametrize("modo", MODOS_DE_FALLA)
def test_p7_sobre_c4_30_30_devuelven_5_obras_con_falla_por_etapa(tmp_path, etapa, modo):
    corridas = _correr_las_30(tmp_path, plan_de_falla={etapa: modo}, configuracion="C4")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(len(c.salida["obras"]) == 5 for c in corridas)
    assert all(
        [d["etapa"] for d in c.degradaciones if d["planificada"]] == [etapa]
        for c in corridas
    )
    assert all(c.degradaciones_no_planificadas == [] for c in corridas)
    assert all(c.fallida is None for c in corridas)
    assert all(c.contenido["invariante_et6"]["cumple_et6"] for c in corridas)

def test_el_reranker_caido_entrega_exactamente_c3(tmp_path):
    c3 = _correr_las_30(tmp_path / "c3", configuracion="C3")
    c4_caido = _correr_las_30(
        tmp_path / "c4", plan_de_falla={"reranker": "excepcion"}, configuracion="C4"
    )
    assert [c.top5 for c in c4_caido] == [c.top5 for c in c3]
    assert all(c.estado["rerank"] is None for c in c4_caido)
    assert all(
        d["politica"].startswith("se entrega el top-5 del ranking fusionado")
        for c in c4_caido
        for d in c.degradaciones
    )
    assert all(
        c.contenido["invariante_et6"]["top5"]["contra"] == "fusionador" for c in c4_caido
    )

def test_la_fusion_caida_en_c4_arrastra_al_reranker_y_la_matriz_no_tiene_ese_punto(
    tmp_path,
):
    corridas = _correr_las_30(
        tmp_path, plan_de_falla={"fusionador": "excepcion"}, configuracion="C4"
    )
    assert all(len(c.top5) == 5 for c in corridas)
    for c in corridas:
        assert {d["etapa"] for d in c.degradaciones} == {"fusionador", "reranker"}
        assert [d["etapa"] for d in c.degradaciones if d["planificada"]] == ["fusionador"]
        assert [d["etapa"] for d in c.degradaciones_no_planificadas] == ["reranker"]
        causa = [d["causa"] for d in c.degradaciones if d["etapa"] == "reranker"][0]
        assert "cascada" in causa
        assert not c.agrega_a_la_tabla
        assert c.contenido["invariante_et6"]["cumple_et6"] is True
        assert c.contenido["invariante_et6"]["top5"]["contra"] == "recuperador_visual"

def test_el_planificador_caido_en_c4_deja_al_reranker_corriendo(tmp_path):
    corridas = _correr_las_30(
        tmp_path, plan_de_falla={"planificador": "excepcion"}, configuracion="C4"
    )
    assert all(len(c.top5) == 5 for c in corridas)
    for c in corridas:
        assert {d["etapa"] for d in c.degradaciones} == {
            "planificador",
            "recuperador_textual",
        }
        assert c.estado["rerank"]
        assert "reranker" not in {d["etapa"] for d in c.degradaciones}
        assert c.contenido["invariante_et6"]["top5"]["contra"] == "reranker"
        assert set(c.estado["fuentes_de_ranking"]) == {"visual"}
        assert not c.agrega_a_la_tabla

def test_c0_devuelve_5_obras_sin_gastar_una_sola_llamada_al_llm(tmp_path):
    corridas = _correr_las_30(tmp_path, configuracion="C0")
    assert len(corridas) == 30
    assert all(len(c.top5) == 5 for c in corridas)
    assert all(len(c.salida["obras"]) == 5 for c in corridas)
    assert all(c.degradaciones == [] for c in corridas)
    assert all(c.agrega_a_la_tabla for c in corridas)
    for c in corridas:
        libro = c.contenido["libro_mayor"]
        assert not {"planificador", "reranker", "generador", "juez"} & set(libro)
        assert all(o["abstencion"] and o["source_url"] for o in c.salida["obras"])
        assert all(o["lineas"] == [] for o in c.salida["obras"])

def test_p1_qs_de_c0_es_cero_exacto_y_de_c1_no(tmp_path):
    from tp3_rag.modulo_c.metricas import qs

    c0 = _correr_las_30(tmp_path / "c0", configuracion="C0")
    c1 = _correr_las_30(tmp_path / "c1", configuracion="C1")
    for i in range(N_IMAGENES):
        tramo = slice(i * N_PEDIDOS, (i + 1) * N_PEDIDOS)
        assert qs([c.top5 for c in c0[tramo]]) == 0.0
    qs_de_c1 = [
        qs([c.top5 for c in c1[i * N_PEDIDOS : (i + 1) * N_PEDIDOS]])
        for i in range(N_IMAGENES)
    ]
    assert any(v > 0 for v in qs_de_c1), (
        "si el QS de C1 fuera 0 con este índice, el control de C0 no distinguiría "
        "nada y el criterio de que el pedido cambia el resultado se cumpliría por vacuidad"
    )

def test_el_plan_de_falla_de_c0_no_tiene_donde_inyectar(tmp_path):
    from tp3_rag.modulo_b.etapa import NO_OMITIBLES

    assert set(CADENA_C0) == set(NO_OMITIBLES)

def test_el_planificador_caido_tira_tambien_la_rama_de_texto(tmp_path):
    corridas = _correr_las_30(
        tmp_path, plan_de_falla={"planificador": "excepcion"}, configuracion="C3"
    )
    for c in corridas:
        etapas = {d["etapa"] for d in c.degradaciones}
        assert etapas == {"planificador", "recuperador_textual"}
        assert [d["etapa"] for d in c.degradaciones if d["planificada"]] == [
            "planificador"
        ]
        assert [d["etapa"] for d in c.degradaciones_no_planificadas] == [
            "recuperador_textual"
        ]
        assert set(c.estado["fuentes_de_ranking"]) == {"visual"}
        assert c.estado["ranking_fusionado"] is not None
        assert not c.agrega_a_la_tabla
    assert all(len(c.top5) == 5 for c in corridas)
