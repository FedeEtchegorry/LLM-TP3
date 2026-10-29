from __future__ import annotations

import pytest

from tp3_rag.dominio.binding import BindingDelReranker, Muestreo
from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.modulo_b.etapa import Candidata, EstadoDeConsulta
from tp3_rag.modulo_b.fusionador import ObraFusionada
from tp3_rag.modulo_b.reranker import (
    PROMPT_ID,
    VERSIÓN_DEL_PROMPT,
    CandidataPuntuada,
    Reranker,
    RerankInválido,
    armar_mensajes,
    movimiento_del_top5,
    ordenar,
    resumen_del_rerank,
)
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.llm import FallaDeEtapa, LLMGrabado

pytestmark = pytest.mark.nivel1

OBRAS = ("cma:101", "cma:102", "cma:103", "aic:201", "aic:202", "aic:202b")
PEDIDO = "el mismo paisaje pero en la pintura china"

def _binding(puerto, *, observabilidad=None, reintentos=2) -> BindingDelReranker:
    return BindingDelReranker(
        modelo="openai/gpt-4.1-mini",
        prompt_id=PROMPT_ID,
        version_del_prompt=VERSIÓN_DEL_PROMPT,
        muestreo=Muestreo(temperature=0.0, top_p=1.0, max_tokens=200, seed=7369),
        esquema_de_salida="json",
        reintentos=reintentos,
        _puerto=(
            observabilidad.instrumentar(puerto, rol="reranker")
            if observabilidad is not None
            else puerto
        ),
    )

def _estado(*, obras=OBRAS, n=None) -> EstadoDeConsulta:
    obras = tuple(obras)[: n or len(obras)]
    entrada = EntradaDeConsulta(
        consulta_id="c02-conceptual",
        imagen_hash=f"sha256:{0:064d}",
        pedido=PEDIDO,
        procedencia="set",
        datos_de_imagen=b"jpeg-del-usuario",
        categoria="conceptual_cultural",
    )
    visual = tuple(
        Candidata(
            artwork_id=o,
            rank=i + 1,
            score=0.9 - i / 100,
            metadata={"title": f"T{o}", "artist": "A", "museum": o.split(":")[0]},
        )
        for i, o in enumerate(obras)
    )
    fusionado = tuple(
        ObraFusionada(
            artwork_id=o,
            score_fusionado=1.0 / (61 + i),
            rangos_de_origen={"visual": i + 1},
        )
        for i, o in enumerate(obras)
    )
    estado = EstadoDeConsulta(entrada=entrada).con(fuentes_de_ranking={"visual": visual})
    estado = estado.proyectar_top5([c.artwork_id for c in visual[:5]], por="recuperador_visual")
    return estado.con(ranking_fusionado=fusionado)

class _LLMQuePuntua(LLMGrabado):
    def __init__(self, scores, *, justificacion="cumple el pedido a medias"):
        super().__init__({})
        self.scores = list(scores)
        self.justificacion = justificacion
        self.i = 0
        self.imagenes_por_llamada: list[int] = []
        self.pedidos_vistos: list[str] = []

    def invocar(self, binding, mensajes, imagenes=()):
        self.imagenes_por_llamada.append(len(imagenes))
        self.pedidos_vistos.append(
            [m for m in mensajes if m.get("rol") == "usuario"][0]["texto"]
        )
        score = self.scores[self.i % len(self.scores)]
        self.i += 1
        self._respuestas = {
            binding.rol: [{"score": score, "justificacion": self.justificacion}]
        }
        return super().invocar(binding, mensajes, imagenes)

def _reranker(puerto, *, n_candidatos=6, top_k=5, observabilidad=None, reintentos=2):
    return Reranker(
        binding=_binding(puerto, observabilidad=observabilidad, reintentos=reintentos),
        bytes_de_la_imagen=lambda artwork_id: f"jpeg-{artwork_id}".encode(),
        n_candidatos=n_candidatos,
        top_k=top_k,
    )

def test_una_llamada_por_candidato_con_dos_imagenes():
    puerto = _LLMQuePuntua([7])
    estado = _reranker(puerto).correr(_estado())
    assert len(puerto.imagenes_por_llamada) == len(OBRAS)
    assert set(puerto.imagenes_por_llamada) == {2}
    assert len(estado.rerank) == len(OBRAS)

def test_entra_el_pedido_crudo_y_no_la_consulta_reescrita():
    puerto = _LLMQuePuntua([7])
    _reranker(puerto).correr(_estado())
    assert all(PEDIDO in texto for texto in puerto.pedidos_vistos)

def test_el_prompt_declarado_y_el_del_modulo_no_pueden_discrepar():
    puerto = _LLMQuePuntua([7])
    binding = BindingDelReranker(
        modelo="openai/gpt-4.1-mini",
        prompt_id="otro-prompt",
        version_del_prompt="1",
        muestreo=Muestreo(temperature=0.0, top_p=1.0, max_tokens=200),
        _puerto=puerto,
    )
    with pytest.raises(RerankInválido, match="identidad del binding"):
        Reranker(binding=binding, bytes_de_la_imagen=lambda _: b"x", n_candidatos=5)

def test_sin_el_almacen_de_imagenes_no_se_construye():
    with pytest.raises(ErrorDeConfiguración, match="reranking por ficha"):
        Reranker(
            binding=_binding(_LLMQuePuntua([7])),
            bytes_de_la_imagen=None,
            n_candidatos=5,
        )

def test_el_binding_del_reranker_es_un_tipo_propio_con_su_clave():
    cfg = Configuración.cargar()
    binding = BindingDelReranker.desde_configuración(cfg, puerto=_LLMQuePuntua([7]))
    assert binding.rol == "reranker"
    assert binding.CLAVE_DE_CONFIGURACIÓN == "bindings.reranker"
    assert binding.modelo == cfg["bindings.generador.modelo"]
    assert binding.muestreo.temperature == 0.0
    assert binding.muestreo.temperature < float(
        cfg["bindings.planificador.muestreo.temperature"]
    )

def test_el_top5_sale_de_los_puntajes_y_el_fusionado_queda_intacto():
    puerto = _LLMQuePuntua([1, 2, 3, 4, 5, 10])
    antes = _estado()
    después = _reranker(puerto).correr(antes)
    assert [f["artwork_id"] for f in después.rerank][0] == OBRAS[-1]
    assert después.top5[0] == OBRAS[-1]
    assert después.top5 != antes.top5
    assert después.ranking_fusionado == antes.ranking_fusionado
    assert all(f.score_fusionado for f in después.ranking_fusionado)

def test_el_orden_desempata_por_el_rango_fusionado():
    puerto = _LLMQuePuntua([7])
    antes = _estado()
    después = _reranker(puerto).correr(antes)
    assert [f["artwork_id"] for f in después.rerank] == list(OBRAS)
    assert list(después.top5) == list(antes.top5)
    puntuadas = (
        CandidataPuntuada("z", 9, "j", rango_fusionado=3),
        CandidataPuntuada("a", 9, "j", rango_fusionado=1),
        CandidataPuntuada("m", 10, "j", rango_fusionado=7),
    )
    assert [c.artwork_id for c in ordenar(puntuadas)] == ["m", "a", "z"]

def test_el_reranker_puntua_min_n_candidatos_y_eso_es_la_cota_de_p6():
    puerto = _LLMQuePuntua([7])
    _reranker(puerto, n_candidatos=3, top_k=3).correr(_estado())
    assert len(puerto.imagenes_por_llamada) == 3
    otro = _LLMQuePuntua([7])
    estado = _reranker(otro, n_candidatos=10).correr(_estado())
    assert len(otro.imagenes_por_llamada) == len(OBRAS) == len(estado.rerank)

def test_m8_2_el_top5_es_subconjunto_de_lo_propuesto_y_no_una_permutacion():
    puerto = _LLMQuePuntua([1, 2, 3, 4, 5, 10])
    antes = _estado()
    después = _reranker(puerto).correr(antes)
    assert OBRAS[-1] not in antes.top5
    assert OBRAS[-1] in después.top5
    assert set(después.top5) <= después.ids_propuestos()

def test_el_movimiento_del_top5_se_lee_de_una_sola_corrida():
    puerto = _LLMQuePuntua([1, 2, 3, 4, 5, 10])
    estado = _reranker(puerto).correr(_estado())
    movimiento = movimiento_del_top5(
        [f.a_json() for f in estado.ranking_fusionado], list(estado.rerank)
    )
    assert movimiento["obras_que_entraron"] == [OBRAS[-1]]
    assert movimiento["churn"] == 1
    assert movimiento["mismo_orden"] is False
    assert movimiento["promovidas_desde_fuera_del_top5"][0]["rango_fusionado"] == 6

@pytest.mark.parametrize(
    "salida,match",
    [
        ({"score": 11, "justificacion": "j"}, "fuera de"),
        ({"score": 7.5, "justificacion": "j"}, "no es entero"),
        ({"score": "alto", "justificacion": "j"}, "tiene que ser un número"),
        ({"score": True, "justificacion": "j"}, "tiene que ser un número"),
        ({"score": 7, "justificacion": "   "}, "vacía"),
        ({"score": 7}, "sin \\['justificacion'\\]"),
        ({"justificacion": "j"}, "sin \\['score'\\]"),
    ],
)
def test_el_esquema_del_dominio_se_verifica_en_la_pieza(salida, match):
    reranker = _reranker(LLMGrabado({"reranker": [salida]}), reintentos=0)
    with pytest.raises(RerankInválido, match=match):
        reranker._parsear(reranker.binding.invocar([{"rol": "usuario", "texto": "x"}]))

def test_el_reintento_usa_el_contador_del_binding_y_el_uso_suma_los_intentos():
    puerto = LLMGrabado(
        {"reranker": [{"score": 99, "justificacion": "j"}, {"score": 8, "justificacion": "j"}]}
    )
    estado = _reranker(puerto, n_candidatos=1, top_k=1).correr(_estado(n=1))
    fila = estado.rerank[0]
    assert fila["score_0_10"] == 8
    assert fila["n_intentos"] == 2
    assert fila["estado"] == "puntuado"
    assert fila["uso"]["tokens_in"] > 0

def test_un_candidato_sin_puntaje_degrada_y_conserva_lo_pagado():
    class _UnaFalla(LLMGrabado):
        def __init__(self):
            super().__init__({})
            self.i = 0

        def invocar(self, binding, mensajes, imagenes=()):
            self.i += 1
            if self.i == 3:
                raise FallaDeEtapa("429 agotado", "el proxy dijo basta")
            self._respuestas = {binding.rol: [{"score": 6, "justificacion": "j"}]}
            return super().invocar(binding, mensajes, imagenes)

    antes = _estado()
    después = _reranker(_UnaFalla()).correr(antes)
    assert [d.etapa for d in después.degradaciones] == ["reranker"]
    assert "sin puntaje" in después.degradaciones[0].causa
    assert list(después.top5) == list(antes.top5)
    assert len(después.rerank) == len(OBRAS)
    assert sum(1 for f in después.rerank if f["estado"] == "puntuado") == len(OBRAS) - 1
    assert all(f["justificacion"] for f in después.rerank)

def test_una_falla_de_transporte_no_se_reintenta_en_la_etapa():
    class _SiempreFalla(LLMGrabado):
        def __init__(self):
            super().__init__({})
            self.llamadas = 0

        def invocar(self, binding, mensajes, imagenes=()):
            self.llamadas += 1
            raise FallaDeEtapa("red caída", "no se pudo llegar al proveedor")

    puerto = _SiempreFalla()
    estado = _reranker(puerto, n_candidatos=2, top_k=2).correr(_estado(n=2))
    assert puerto.llamadas == 2
    assert all(f["estado"] == "sin_puntaje" for f in estado.rerank)
    assert all(f["n_intentos"] == 1 for f in estado.rerank)

def test_una_imagen_que_falta_no_cuenta_llamada_y_lo_dice():
    def _sin_imagen(artwork_id):
        raise FileNotFoundError(f"falta la imagen de {artwork_id}")

    reranker = Reranker(
        binding=_binding(_LLMQuePuntua([7])),
        bytes_de_la_imagen=_sin_imagen,
        n_candidatos=2,
        top_k=2,
    )
    estado = reranker.correr(_estado(n=2))
    assert all(f["estado"] == "sin_puntaje" for f in estado.rerank)
    assert all(f["n_intentos"] == 0 for f in estado.rerank)
    assert all("no se pudo leer la imagen" in f["justificacion"] for f in estado.rerank)
    assert [d.etapa for d in estado.degradaciones] == ["reranker"]

def test_el_artefacto_de_a3_trae_los_dos_por_candidato():
    puerto = _LLMQuePuntua([3, 5, 7, 9, 10, 0], justificacion="no es china")
    estado = _reranker(puerto).correr(_estado())
    resumen = resumen_del_rerank(list(estado.rerank))
    assert resumen["n_candidatos_puntuados"] == len(OBRAS)
    assert resumen["n_con_justificacion"] == len(OBRAS)
    assert resumen["n_sin_puntaje"] == 0
    assert (resumen["score_min"], resumen["score_max"]) == (0, 10)
    assert resumen["scores_distintos"] == 6
    assert resumen["tokens_in"] > 0

def test_el_muestreo_efectivo_queda_en_el_artefacto_de_la_etapa():
    puerto = _LLMQuePuntua([7])
    reranker = _reranker(puerto)
    reranker.correr(_estado())
    artefacto = reranker.a_json()
    assert artefacto["muestreo_efectivo"]["temperature"] == 0.0
    assert artefacto["muestreo_efectivo"]["seed"] == 7369
    assert artefacto["muestreo_efectivo"]["response_format"] == {"type": "json_object"}
    assert artefacto["modelo_efectivo"] == "openai/gpt-4.1-mini"

def test_bd4_las_n_llamadas_se_atribuyen_al_rol_reranker():
    obs = Observabilidad(silencioso=True)
    puerto = _LLMQuePuntua([7])
    _reranker(puerto, observabilidad=obs).correr(_estado())
    libro = obs.libro_mayor()
    assert set(libro) == {"reranker"}
    assert libro["reranker"]["n_llamadas"] == len(OBRAS)
    assert libro["reranker"]["tokens_in"] > 0

def test_el_prompt_pide_la_escala_y_el_json_de_los_dos_campos():
    mensajes = armar_mensajes(_estado(), artwork_id="cma:101", metadata={"title": "T"})
    sistema = [m for m in mensajes if m["rol"] == "sistema"][0]["texto"]
    usuario = [m for m in mensajes if m["rol"] == "usuario"][0]["texto"]
    assert '{"score": <entero de 0 a 10>, "justificacion": "..."}' in sistema
    assert "la primera imagen" in sistema.lower()
    assert "No la compares con otras" in sistema
    assert PEDIDO in usuario and "cma:101" in usuario
    assert "rango" not in usuario and "score" not in usuario

def test_m10_el_mensaje_de_un_candidato_no_depende_de_n():
    obras = tuple(f"aic:{1000 + i}" for i in range(50))
    estado_50 = _estado(obras=obras)
    estado_20 = _estado(obras=obras, n=20)
    assert len(estado_50.ranking_fusionado) == 50
    assert len(estado_20.ranking_fusionado) == 20
    for artwork_id in obras[:20]:
        metadata = {"title": f"T{artwork_id}", "artist": "A", "museum": "aic"}
        assert armar_mensajes(estado_50, artwork_id=artwork_id, metadata=metadata) == (
            armar_mensajes(estado_20, artwork_id=artwork_id, metadata=metadata)
        ), f"el mensaje de {artwork_id} cambió con N — hay un ingrediente escondido"

def test_m10_ni_la_n_ni_el_rango_llegan_al_texto_del_prompt():
    obras = tuple(f"aic:{1000 + i}" for i in range(50))
    textos = [
        m["texto"]
        for m in armar_mensajes(
            _estado(obras=obras),
            artwork_id="aic:1030",
            metadata={"title": "T", "artist": "A", "museum": "aic"},
        )
    ]
    completo = "\n".join(textos)
    for prohibido in ("50", "31", "de 50", "candidata 31", "rango"):
        assert prohibido not in completo, f"{prohibido!r} entró al prompt"
    assert "No la compares con otras" in completo

def test_m10_los_scores_de_n_50_derivan_n_20_y_n_10_con_los_mismos_puntajes():
    obras = tuple(f"aic:{1000 + i}" for i in range(50))
    tabla = {o: i % 11 for i, o in enumerate(obras)}

    class _PorObra(LLMGrabado):
        def __init__(self):
            super().__init__({})
            self.vistos = []

        def invocar(self, binding, mensajes, imagenes=()):
            texto = [m for m in mensajes if m.get("rol") == "usuario"][0]["texto"]
            artwork_id = next(o for o in obras if f"artwork_id: {o}\n" in texto)
            self.vistos.append(artwork_id)
            self._respuestas = {
                binding.rol: [{"score": tabla[artwork_id], "justificacion": "ok"}]
            }
            return super().invocar(binding, mensajes, imagenes)

    puerto = _PorObra()
    de_50 = _reranker(puerto, n_candidatos=50).correr(_estado(obras=obras))
    assert len(puerto.vistos) == 50
    scores_de_50 = {f["artwork_id"]: f["score_0_10"] for f in de_50.rerank}

    for n in (20, 10):
        puerto_n = _PorObra()
        de_n = _reranker(puerto_n, n_candidatos=n).correr(_estado(obras=obras, n=n))
        assert len(puerto_n.vistos) == n
        assert {o: scores_de_50[o] for o in obras[:n]} == {
            f["artwork_id"]: f["score_0_10"] for f in de_n.rerank
        }
        derivado = [
            o
            for o in sorted(obras[:n], key=lambda o: (-scores_de_50[o], obras.index(o)))
        ][:5]
        assert derivado == list(de_n.top5), f"el anidado no reproduce N = {n}"
