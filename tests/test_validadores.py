from __future__ import annotations

import pytest

from tp3_rag.modulo_c import validadores as val
from tp3_rag.nucleo.errores import ErrorDelProyecto

pytestmark = pytest.mark.nivel2

CORPUS = ["cma:1", "cma:2", "cma:3", "cma:4", "cma:5"]
TEXTO = "A rural landscape near Arkville."
FRAG = "cma:1::description::0"

class _CorridaFalsa:
    def __init__(
        self,
        *,
        consulta_id="c01-visual",
        top5=None,
        evidencia=None,
        referencias=None,
        lineas=None,
        ordinal=1,
        configuracion="C1",
        libro_mayor=None,
        rerank=None,
        degradaciones=None,
        plan=None,
        etapas=None,
        ranking_fusionado=None,
        explicacion=None,
    ):
        self.consulta_id = consulta_id
        self.ordinal = ordinal
        self.config_id = "cfg"
        self.contenido = {
            "pipeline": {"configuracion": configuracion, "etapas": etapas or []},
            "libro_mayor": (
                libro_mayor
                if libro_mayor is not None
                else {"generador": {"n_llamadas": 1, "tokens_in": 10, "tokens_out": 5}}
            ),
        }
        top5 = top5 or ["cma:1"]
        self.estado = {
            "top5": top5,
            "evidencia": evidencia
            if evidencia is not None
            else {
                o: [{"fragment_id": f"{o}::description::0", "texto": TEXTO}] for o in top5
            },
            "afirmaciones_descartadas": [],
            "salida": None,
            "rerank": rerank,
            "ranking_fusionado": ranking_fusionado,
            "explicacion": (
                explicacion
                if explicacion is not None
                else [{"artwork_id": o, "afirmaciones": []} for o in top5]
            ),
            "plan": plan,
            "degradaciones": degradaciones or [],
        }
        obras = []
        for o in top5:
            obras.append({
                "artwork_id": o,
                "source_url": f"https://clevelandart.org/art/{o}",
                "lineas": lineas if lineas is not None else ["Es un paisaje rural. [1]"],
                "abstencion": None,
                "referencias": referencias
                if referencias is not None
                else [{
                    "n": 1,
                    "fragment_id": f"{o}::description::0",
                    "texto_citado": TEXTO,
                    "license": "CC-BY-4.0",
                    "atribucion": f"“T”, A. CMA. Texto bajo CC BY 4.0 — https://x/{o}",
                }],
            })
        render = "\n".join(
            [l for ob in obras for l in ob["lineas"]]
            + [r["texto_citado"] for ob in obras for r in ob["referencias"]]
            + [r["atribucion"] or "" for ob in obras for r in ob["referencias"]]
        )
        self.estado["salida"] = {"obras": obras, "render": render}

    @property
    def top5(self):
        return list(self.estado["top5"])

    @property
    def salida(self):
        return self.estado["salida"]

    @property
    def afirmaciones_descartadas(self):
        return list(self.estado["afirmaciones_descartadas"])

    @property
    def plan(self):
        return self.estado["plan"]

    @property
    def plan_reusado(self):
        plan = self.estado["plan"]
        return None if plan is None else bool(plan.get("reusado"))

    @property
    def degradaciones(self):
        return list(self.estado["degradaciones"])

EXIGEN = {"CC-BY-4.0": True, "CC0": False}

class _ClienteQueResuelve:
    class R:
        status = 200

    def get(self, url):
        return self.R()

def test_el_self_retrieval_rechaza_cualquier_alfa_que_no_sea_1():
    with pytest.raises(ErrorDelProyecto, match="tiene que ser 1"):
        val.self_retrieval(
            puerto_de_imagen=None, indice=None, manifiesto=None, bytes_de=None, alfa=0.5
        )

def test_el_self_retrieval_corre_sobre_el_lado_corpus_del_manifiesto():
    from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
    from tp3_rag.puertos.indice_vectorial import ARTWORKS, ÍndiceVectorialEnMemoria

    class _Manifiesto:
        split_id = "s"
        ids_corpus = tuple(CORPUS)
        ids_held_out = ("cma:99",)

    img = EmbeddingImagenFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v")
    idx.upsert(
        ARTWORKS,
        ids=CORPUS,
        vectores=[img.embeber_imagen(f"b-{o}".encode()) for o in CORPUS],
        metadatas=[{"museum": "cma"} for _ in CORPUS],
    )
    resultado = val.self_retrieval(
        puerto_de_imagen=img,
        indice=idx,
        manifiesto=_Manifiesto(),
        bytes_de=lambda o: f"b-{o}".encode(),
        n=5,
    )
    assert resultado.pasa and resultado.marcador == "5/5"
    assert "lado corpus" in resultado.poblacion
    assert resultado.detalle["alfa"] == 1.0

def test_el_self_retrieval_acusa_cuando_el_top1_no_es_la_obra():
    from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
    from tp3_rag.puertos.indice_vectorial import ARTWORKS, ÍndiceVectorialEnMemoria

    class _Manifiesto:
        split_id = "s"
        ids_corpus = tuple(CORPUS)
        ids_held_out = ()

    img = EmbeddingImagenFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v")
    idx.upsert(
        ARTWORKS,
        ids=CORPUS,
        vectores=[img.embeber_imagen(f"otro-{o}".encode()) for o in CORPUS],
        metadatas=[{"museum": "cma"} for _ in CORPUS],
    )
    resultado = val.self_retrieval(
        puerto_de_imagen=img, indice=idx, manifiesto=_Manifiesto(),
        bytes_de=lambda o: f"b-{o}".encode(), n=5,
    )
    assert not resultado.pasa
    assert resultado.fallas and "top1" in resultado.fallas[0]

def test_p2_pasa_con_las_tres_condiciones():
    corridas = [_CorridaFalsa(top5=CORPUS[:5])]
    r = val.p2_trazabilidad(
        corridas, ids_del_corpus=CORPUS, cliente_http=_ClienteQueResuelve()
    )
    assert r.pasa and r.marcador == "5/5"
    assert "corpus indexado de 5 obras" in r.poblacion

def test_p2_acusa_una_obra_que_no_esta_en_el_corpus():
    corridas = [_CorridaFalsa(top5=["cma:fuera"])]
    r = val.p2_trazabilidad(
        corridas, ids_del_corpus=CORPUS, cliente_http=_ClienteQueResuelve()
    )
    assert not r.pasa
    assert any("no está en el corpus" in p for p in r.fallas[0]["problemas"])

def test_p2_acusa_una_obra_con_0_fragmentos():
    corridas = [_CorridaFalsa(top5=["cma:1"], evidencia={"cma:1": []})]
    r = val.p2_trazabilidad(
        corridas, ids_del_corpus=CORPUS, cliente_http=_ClienteQueResuelve()
    )
    assert not r.pasa
    assert any("0 fragmentos" in p for p in r.fallas[0]["problemas"])

def test_p2_sin_cliente_no_da_por_buena_la_resolucion():
    r = val.p2_trazabilidad([_CorridaFalsa()], ids_del_corpus=CORPUS, cliente_http=None)
    assert not r.pasa
    assert any("no verificada" in p for p in r.fallas[0]["problemas"])
    assert r.detalle["n_urls_sin_verificar"] == 1

def test_p3a_pasa_con_0_colgadas_y_el_umbral_es_0():
    r = val.p3a_referencias_colgadas([_CorridaFalsa()])
    assert r.pasa and r.detalle["umbral"] == 0

def test_p3a_acusa_una_referencia_publicada_que_no_resuelve():
    corrida = _CorridaFalsa(
        referencias=[{
            "n": 1, "fragment_id": "cma:1::exhibitions::0", "texto_citado": TEXTO,
            "license": "CC0", "atribucion": None,
        }]
    )
    r = val.p3a_referencias_colgadas([corrida])
    assert not r.pasa
    assert r.detalle["n_colgadas_publicadas"] == 1

def test_los_descartes_de_sa5_se_cuentan_y_no_son_fallas_de_p3a():
    corrida = _CorridaFalsa()
    corrida.estado["afirmaciones_descartadas"] = [{"artwork_id": "cma:1", "refs_colgadas": ["x"]}]
    r = val.p3a_referencias_colgadas([corrida])
    assert r.pasa
    assert r.detalle["n_afirmaciones_descartadas_por_sa5"] == 1

def test_p4_pasa_con_la_atribucion_en_la_superficie():
    r = val.p4_atribucion([_CorridaFalsa()], exigen_atribucion=EXIGEN)
    assert r.pasa and r.marcador == "1/1"
    assert r.detalle["sin_llamar_al_llm"] is True
    assert r.detalle["verificado_contra"] == "el texto renderizado, no el campo"

def test_p4_acusa_un_campo_lleno_que_el_render_no_imprime():
    corrida = _CorridaFalsa()
    corrida.estado["salida"]["render"] = "un render que no incluye la atribución"
    r = val.p4_atribucion([corrida], exigen_atribucion=EXIGEN)
    assert not r.pasa
    assert "no aparece en la superficie" in r.fallas[0]["problema"]

def test_p4_acusa_una_licencia_no_declarada_en_vez_de_tratarla_como_cc0():
    corrida = _CorridaFalsa(
        referencias=[{
            "n": 1, "fragment_id": FRAG, "texto_citado": TEXTO,
            "license": "CC-BY-NC-3.0", "atribucion": None,
        }]
    )
    r = val.p4_atribucion([corrida], exigen_atribucion=EXIGEN)
    assert not r.pasa
    assert "no está declarada" in r.fallas[0]["problema"]

def test_p4_no_mide_sobre_los_fragmentos_de_knowledge():
    r = val.p4_atribucion([_CorridaFalsa()], exigen_atribucion=EXIGEN)
    assert "referencias publicadas" in r.poblacion
    assert "knowledge" in r.poblacion

def test_p8_pasa_con_espanol_y_cita_byte_a_byte():
    r = val.p8_idioma([_CorridaFalsa()])
    assert r.pasa
    assert r.detalle["citas_byte_a_byte"] == "1/1"
    assert r.detalle["explicaciones_en_espanol"] == "1/1"

def test_p8_acusa_una_cita_que_no_es_byte_a_byte():
    corrida = _CorridaFalsa(
        referencias=[{
            "n": 1, "fragment_id": FRAG, "texto_citado": TEXTO.upper(),
            "license": "CC0", "atribucion": None,
        }]
    )
    r = val.p8_idioma([corrida])
    assert not r.pasa
    assert "byte a byte" in r.fallas[0]["problema"]

def test_p8_acusa_una_explicacion_que_no_pasa_el_heuristico_de_espanol():
    corrida = _CorridaFalsa(lineas=["This is an English sentence about the painting. [1]"])
    r = val.p8_idioma([corrida])
    assert not r.pasa
    assert r.fallas[0]["umbral"] == val.UMBRAL_DE_ESPAÑOL

def test_el_heuristico_se_declara_como_heuristico():
    r = val.p8_idioma([_CorridaFalsa()])
    assert "heurístico declarado" in r.detalle["nota_de_la_asimetria"]
    assert "RevisiónDeExplicaciones" in r.detalle["nota_de_la_asimetria"]
    assert val.proporcion_de_palabras_del_espanol("de la el los las") == 1.0
    assert val.proporcion_de_palabras_del_espanol("the quick brown fox") == 0.0
    assert val.proporcion_de_palabras_del_espanol("") == 0.0

def test_todos_sobre_corridas_corre_los_cinco_sin_puertos():
    resultados = val.todos_sobre_corridas(
        [_CorridaFalsa()],
        ids_del_corpus=CORPUS,
        exigen_atribucion=EXIGEN,
        cliente_http=_ClienteQueResuelve(),
    )
    assert len(resultados) == 5
    assert all(r.pasa for r in resultados)
    assert [r.criterio.split("\u00b7")[0].strip() for r in resultados] == [
        "el criterio de trazabilidad",
        "el conteo de referencias colgadas",
        "el criterio de atribución",
        "el criterio de idioma",
        "el criterio de costo",
    ]

def test_todo_resultado_lleva_su_poblacion():
    for r in val.todos_sobre_corridas(
        [_CorridaFalsa()], ids_del_corpus=CORPUS, exigen_atribucion=EXIGEN,
        cliente_http=_ClienteQueResuelve(),
    ):
        assert r.poblacion and len(r.poblacion) > 20
        assert r.a_json()["marcador"] == r.marcador

MUESTREO_EFECTIVO = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 7369,
    "max_tokens": 200,
    "response_format": {"type": "json_object"},
}

def _etapa_del_reranker(muestreo_efectivo=MUESTREO_EFECTIVO):
    return {
        "etapa": "reranker",
        "binding": {"rol": "reranker", "modelo": "openai/gpt-4.1-mini"},
        "muestreo_efectivo": muestreo_efectivo,
    }

def _corrida_de_c4(
    *, consulta_id="c02-conceptual", ordinal=1, scores=(9, 7, 7, 5, 3, 1),
    muestreo_efectivo=MUESTREO_EFECTIVO, top5=None, libro_mayor=None,
):
    obras = [f"cma:{i}" for i in range(1, len(scores) + 1)]
    rerank = [
        {
            "artwork_id": o,
            "score_0_10": s,
            "justificacion": f"cumple en parte ({o})",
            "rango_fusionado": i + 1,
            "estado": "puntuado",
            "uso": {"tokens_in": 1900, "tokens_out": 40},
            "n_intentos": 1,
        }
        for i, (o, s) in enumerate(zip(obras, scores))
    ]
    rerank.sort(key=lambda f: (-f["score_0_10"], f["rango_fusionado"]))
    corrida = _CorridaFalsa(
        consulta_id=consulta_id,
        ordinal=ordinal,
        configuracion="C4",
        top5=top5 or [f["artwork_id"] for f in rerank][:5],
        rerank=rerank,
        ranking_fusionado=[
            {"artwork_id": o, "score_fusionado": 1.0 / (61 + i), "rangos_de_origen": {"visual": i + 1}}
            for i, o in enumerate(obras)
        ],
        etapas=[_etapa_del_reranker(muestreo_efectivo)],
        plan={"plan_id": "p1", "reusado": True},
        libro_mayor=libro_mayor
        if libro_mayor is not None
        else {
            "reranker": {"n_llamadas": len(scores), "tokens_in": 1900 * len(scores), "tokens_out": 40},
            "generador": {"n_llamadas": 1, "tokens_in": 3000, "tokens_out": 700},
            "embedding_texto": {"n_llamadas": 2, "tokens_in": 40, "tokens_out": 0},
        },
    )
    return corrida

def test_a3_la_cuarta_tecnica_exige_los_dos_por_candidato():
    r = val.a3_reranking_con_justificacion([_corrida_de_c4()])
    assert r.pasa and r.marcador == "1/1"
    assert r.detalle["candidatos_puntuados"] == 6
    assert r.detalle["candidatos_sin_puntaje"] == 0
    assert "4ª técnica" in r.criterio

def test_a3_acusa_un_candidato_sin_justificacion():
    corrida = _corrida_de_c4()
    corrida.estado["rerank"][0]["justificacion"] = "   "
    r = val.a3_reranking_con_justificacion([corrida])
    assert not r.pasa
    assert any("sin justificación" in p for p in r.fallas[0]["problemas"])

def test_a3_acusa_un_top5_que_no_sale_del_orden_del_rerank():
    corrida = _corrida_de_c4(top5=["cma:6", "cma:5", "cma:4", "cma:3", "cma:2"])
    r = val.a3_reranking_con_justificacion([corrida])
    assert not r.pasa
    assert any("no son los primeros" in p for p in r.fallas[0]["problemas"])

def test_a3_acusa_que_el_reranker_piso_el_ranking_fusionado():
    corrida = _corrida_de_c4()
    corrida.estado["ranking_fusionado"] = None
    r = val.a3_reranking_con_justificacion([corrida])
    assert not r.pasa
    assert any("ranking_fusionado" in p for p in r.fallas[0]["problemas"])

def test_a3_acusa_mas_llamadas_que_candidatos():
    corrida = _corrida_de_c4(
        libro_mayor={"reranker": {"n_llamadas": 99, "tokens_in": 1, "tokens_out": 1}}
    )
    r = val.a3_reranking_con_justificacion([corrida])
    assert not r.pasa
    assert any("Un candidato por llamada" in p for p in r.fallas[0]["problemas"])

def test_p6_cuenta_los_roles_del_llm_y_no_los_embeddings():
    r = val.p6_llamadas_al_llm([_corrida_de_c4()], n_candidatos=6)
    c4 = r.detalle["por_configuracion"]["C4"]
    assert c4["llamadas_nominales_por_consulta"] == 8 == 2 + 6
    assert c4["comprometido_por_p6"] == 8
    assert c4["intentos_pagados_en_total"] == 7
    assert c4["planes_reusados_r11"] == 1
    assert c4["embeddings_que_p6_no_cuenta"] == 2
    assert r.pasa

def test_p6_acusa_una_configuracion_que_no_cierra_con_lo_comprometido():
    corrida = _CorridaFalsa(configuracion="C2", plan=None)
    r = val.p6_llamadas_al_llm([corrida])
    assert not r.pasa
    assert r.fallas[0]["comprometido"] == 2 and r.fallas[0]["medido"] == [1]

def test_p6_un_reintento_de_m4_1_no_rompe_el_conteo_y_queda_publicado():
    corrida = _CorridaFalsa(
        libro_mayor={"generador": {"n_llamadas": 2, "tokens_in": 20, "tokens_out": 10}}
    )
    r = val.p6_llamadas_al_llm([corrida])
    c1 = r.detalle["por_configuracion"]["C1"]
    assert r.pasa
    assert c1["llamadas_nominales_por_consulta"] == 1 == c1["comprometido_por_p6"]
    assert c1["intentos_pagados_en_total"] == 2
    assert c1["corridas_con_reintento_m4_1"] == 1

def test_p6_mide_c0_en_cero_y_es_un_numero_que_solo_c0_puede_dar():
    c0 = _CorridaFalsa(configuracion="C0", libro_mayor={}, explicacion=[])
    r = val.p6_llamadas_al_llm([c0, _corrida_de_c4()], n_candidatos=6)
    assert r.detalle["por_configuracion"]["C0"]["llamadas_nominales_por_consulta"] == 0
    assert r.pasa

def test_r12_empareja_los_dos_ordinales_y_dice_que_cambio_y_cuanto():
    primera = _corrida_de_c4(ordinal=1, scores=(9, 7, 7, 5, 3, 1))
    segunda = _corrida_de_c4(ordinal=2, scores=(9, 8, 7, 5, 3, 1))
    r = val.r12_variabilidad([primera, segunda])
    assert r.pasa and r.marcador == "1/1"
    assert r.detalle["candidatos_comparados"] == 6
    assert r.detalle["candidatos_con_otro_score"] == 1
    assert r.detalle["diferencia_absoluta_media_del_score"] == 1.0
    assert r.detalle["top5_identico"] == "1/1"
    assert "no hay umbral" in r.detalle["nota_del_umbral"].lower()

def test_r12_se_niega_si_el_muestreo_efectivo_no_esta_registrado():
    primera = _corrida_de_c4(ordinal=1)
    segunda = _corrida_de_c4(ordinal=2, muestreo_efectivo=None)
    r = val.r12_variabilidad([primera, segunda])
    assert not r.pasa
    assert "muestreo efectivo" in r.fallas[0]["problema"]

def test_r12_se_niega_si_el_muestreo_difiere_entre_las_dos():
    primera = _corrida_de_c4(ordinal=1)
    segunda = _corrida_de_c4(
        ordinal=2, muestreo_efectivo={**MUESTREO_EFECTIVO, "temperature": 0.7}
    )
    r = val.r12_variabilidad([primera, segunda])
    assert not r.pasa
    assert "difiere" in r.fallas[0]["problema"]

def test_r12_sin_dos_ordinales_no_mide_y_lo_dice():
    r = val.r12_variabilidad([_corrida_de_c4()])
    assert not r.pasa and r.de == 0
    assert "dos o más ordinales" in r.poblacion
