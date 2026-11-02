from __future__ import annotations

import pytest

from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo

pytestmark = pytest.mark.nivel1

class PuertoFalsoConUso:
    OPERACIONES = ("embeber",)

    class Salida:
        def __init__(self, n):
            self.n = n
            self.uso = {"tokens_in": 7 * n, "tokens_out": 0}

    def embeber(self, textos):
        return self.Salida(len(textos))

    def ayudante_interno(self):
        return "no me instrumentes"

@pytest.fixture
def obs() -> Observabilidad:
    return Observabilidad(silencioso=True)

def test_un_span_por_llamada_no_por_item(obs):
    puerto = obs.instrumentar(EmbeddingImagenFijo(), rol="embedding_imagen")
    puerto.embeber_imagenes([("cma:1", b"a"), ("cma:2", b"b"), ("cma:3", b"c")])
    spans = obs.spans()
    assert len(spans) == 1
    assert spans[0].n == 3
    assert spans[0].rol == "embedding_imagen"
    assert spans[0].operación == "embeber_imagenes"
    assert spans[0].resultado == "ok"

def test_los_tokens_salen_del_resultado_y_no_se_estiman(obs):
    puerto = obs.instrumentar(PuertoFalsoConUso(), rol="embedding_texto")
    puerto.embeber(["a", "b"])
    assert obs.libro_mayor()["embedding_texto"]["tokens_in"] == 14

def test_el_libro_mayor_es_uno_y_las_lecturas_dos(obs):
    obs.instrumentar(PuertoFalsoConUso(), rol="generador").embeber(["a"])
    obs.instrumentar(PuertoFalsoConUso(), rol="juez").embeber(["b", "c"])
    libro = obs.libro_mayor()
    assert set(libro) == {"generador", "juez"}
    assert sum(v["tokens_in"] for v in libro.values()) == 21

def test_la_lectura_por_etapa_sale_del_arbol_y_no_de_un_parametro(obs):
    puerto = obs.instrumentar(PuertoFalsoConUso(), rol="embedding_texto")
    with obs.nodo("corrida C4", tipo="chain"):
        with obs.nodo("recuperador_textual", tipo="chain", metadata={"etapa": "recuperador_textual"}):
            puerto.embeber(["la rama"])
        with obs.nodo(
            "recolector_de_evidencia",
            tipo="chain",
            metadata={"etapa": "recolector_de_evidencia"},
        ):
            puerto.embeber(["el orden de la evidencia"])
    assert set(obs.libro_mayor()) == {"embedding_texto"}
    assert set(obs.libro_mayor_por_etapa()) == {
        "recuperador_textual",
        "recolector_de_evidencia",
    }

def test_las_dos_lecturas_no_pueden_discrepar(obs):
    rerank = obs.instrumentar(PuertoFalsoConUso(), rol="reranker")
    texto = obs.instrumentar(PuertoFalsoConUso(), rol="embedding_texto")
    with obs.nodo("corrida C4", tipo="chain"):
        with obs.nodo("reranker", tipo="chain", metadata={"etapa": "reranker"}):
            for _ in range(3):
                rerank.embeber(["un candidato"])
        with obs.nodo("recuperador_textual", tipo="chain", metadata={"etapa": "recuperador_textual"}):
            texto.embeber(["la rama"])
    por_rol, por_etapa = obs.libro_mayor(), obs.libro_mayor_por_etapa()
    for campo in ("n_llamadas", "n_items", "tokens_in", "tokens_out"):
        assert sum(v[campo] for v in por_rol.values()) == sum(
            v[campo] for v in por_etapa.values()
        )
    assert por_rol["reranker"]["n_llamadas"] == por_etapa["reranker"]["n_llamadas"] == 3
    assert por_etapa["reranker"]["roles"] == {"reranker": 3}

def test_un_span_fuera_de_una_etapa_se_reporta_aparte_y_no_como_ninguna(obs):
    obs.instrumentar(PuertoFalsoConUso(), rol="embedding_imagen").embeber(["suelto"])
    assert obs.spans()[0].etapa == ""
    assert list(obs.libro_mayor_por_etapa()) == ["(fuera de una etapa)"]

def test_una_operacion_que_el_puerto_no_declara_no_se_instrumenta(obs):
    puerto = obs.instrumentar(PuertoFalsoConUso(), rol="x")
    assert puerto.ayudante_interno() == "no me instrumentes"
    assert obs.spans() == []

def test_un_puerto_sin_operaciones_declaradas_no_se_puede_instrumentar(obs):
    class Pelado:
        pass

    with pytest.raises(ValueError, match="OPERACIONES"):
        obs.instrumentar(Pelado(), rol="x")

def test_una_falla_queda_registrada_y_se_propaga(obs):
    class Roto:
        OPERACIONES = ("tronar",)

        def tronar(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        obs.instrumentar(Roto(), rol="x").tronar()
    assert obs.spans()[0].resultado == "error:ValueError"

def test_el_puerto_instrumentado_cumple_el_mismo_contrato(obs):
    crudo = EmbeddingImagenFijo()
    puerto = obs.instrumentar(crudo, rol="embedding_imagen")
    assert puerto.identidad() == crudo.identidad()
    assert tuple(puerto.embeber_imagen(b"a")) == tuple(crudo.embeber_imagen(b"a"))

def test_una_operacion_singular_cuenta_un_item_no_las_dimensiones(obs):
    from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo

    imagen = obs.instrumentar(EmbeddingImagenFijo(), rol="embedding_imagen")
    texto = obs.instrumentar(EmbeddingTextoFijo(), rol="embedding_texto")
    vector_de_imagen = imagen.embeber_imagen(b"unos bytes")
    vector_de_texto = texto.embeber_uno("un texto")
    assert len(vector_de_imagen) == 512 and len(vector_de_texto) == 1536
    libro = obs.libro_mayor()
    assert libro["embedding_imagen"]["n_items"] == 1
    assert libro["embedding_texto"]["n_items"] == 1

def test_un_batch_sigue_contando_sus_items(obs):
    puerto = obs.instrumentar(EmbeddingImagenFijo(), rol="embedding_imagen")
    puerto.embeber_imagenes([("cma:1", b"a"), ("cma:2", b"b")])
    assert obs.libro_mayor()["embedding_imagen"]["n_items"] == 2

def test_una_lista_de_resultados_cuenta_sus_elementos(obs):
    from tp3_rag.nucleo.observabilidad import _largo
    from tp3_rag.puertos.indice_vectorial import Resultado

    assert _largo([Resultado("a", 0.9, {}), Resultado("b", 0.8, {})]) == 2
    assert _largo((0.1, 0.2, 0.3)) == 1
    assert _largo([]) == 0
    assert _largo("texto") == 5

def test_la_operacion_plural_declara_su_uso_y_la_singular_no(obs):
    from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo

    puerto = obs.instrumentar(EmbeddingTextoFijo(), rol="embedding_texto")
    puerto.embeber_uno("cuatro palabras de prueba")
    assert obs.libro_mayor()["embedding_texto"]["tokens_in"] == 0

    otra = Observabilidad(silencioso=True)
    plural = otra.instrumentar(EmbeddingTextoFijo(), rol="embedding_texto")
    plural.embeber(["cuatro palabras de prueba"])
    assert otra.libro_mayor()["embedding_texto"]["tokens_in"] == 4

def test_m7_6_los_tokens_salen_como_usage_metadata():
    from tp3_rag.puertos.trazas import CLAVES_DE_USAGE, usage_metadata_de

    salida = usage_metadata_de({"tokens_in": 900, "tokens_out": 60})
    assert set(salida["usage_metadata"]) == set(CLAVES_DE_USAGE)
    assert salida["usage_metadata"] == {
        "input_tokens": 900,
        "output_tokens": 60,
        "total_tokens": 960,
    }

def test_m7_6_las_claves_son_exactamente_las_que_el_servidor_valida():
    from langsmith.run_trees import validate_extracted_usage_metadata

    from tp3_rag.puertos.trazas import usage_metadata_de

    validate_extracted_usage_metadata(
        usage_metadata_de({"tokens_in": 1, "tokens_out": 2})["usage_metadata"]
    )

def test_m7_6_un_nodo_sin_tokens_no_declara_un_cero():
    from tp3_rag.puertos.trazas import usage_metadata_de

    assert usage_metadata_de(None) == {}
    assert usage_metadata_de({}) == {}
    assert usage_metadata_de({"tokens_in": 0, "tokens_out": 0}) == {}

def test_m7_6_el_arbol_que_se_postea_lleva_los_tokens_en_los_dos_lugares(obs):
    from tp3_rag.nucleo.observabilidad import NodoDeTraza
    from tp3_rag.puertos.trazas import usage_metadata_de

    nodo = NodoDeTraza(
        id="x", trace_id="x", dotted_order="x", nombre="planificador.invocar",
        tipo="llm", inicio=_ahora(), metadata={"tokens": {"tokens_in": 10, "tokens_out": 3}},
    )
    metadata_posteada = {
        **{k: v for k, v in nodo.metadata.items() if k != "tokens"},
        **usage_metadata_de(nodo.metadata["tokens"]),
    }
    assert metadata_posteada["usage_metadata"]["total_tokens"] == 13
    assert nodo.a_json()["tokens"] == {"tokens_in": 10, "tokens_out": 3}

def _ahora():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
