from __future__ import annotations

import inspect

import pytest

from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos import embedding_texto
from tp3_rag.puertos.embedding_texto import (
    EmbeddingTextoFijo,
    EmbeddingTextoOpenAI,
    InputDemasiadoLargo,
    RedCaída,
    dim_conocida,
    normalizar,
)

pytestmark = pytest.mark.nivel1

PROXY = "https://sa-llmproxy.it.itba.edu.ar/"

def test_la_operacion_del_puerto_es_plural():
    assert "embeber" in EmbeddingTextoOpenAI.OPERACIONES
    assert "embeber_uno" in EmbeddingTextoOpenAI.OPERACIONES
    cuerpo = inspect.getsource(EmbeddingTextoOpenAI.embeber_uno)
    assert "self.embeber(" in cuerpo

def test_el_singular_devuelve_exactamente_lo_mismo_que_el_batch_de_uno():
    puerto = EmbeddingTextoFijo(dim=8)
    assert puerto.embeber_uno("Corot") == puerto.embeber(["Corot"])[0]

def test_un_batch_devuelve_un_vector_por_item_y_en_orden():
    puerto = EmbeddingTextoFijo(dim=8)
    vectores = puerto.embeber(["a", "b", "c"])
    assert vectores.n == 3
    assert vectores[0] == puerto.embeber_uno("a")
    assert vectores[0] != vectores[1]

def test_un_batch_vacio_no_llama_al_modelo():
    puerto = EmbeddingTextoOpenAI(api_key="sk-no-existe", base_url=PROXY, dim=8)
    assert puerto.embeber([]).n == 0

def test_el_puerto_declara_tokens_porque_a8_los_cuenta():
    vectores = EmbeddingTextoFijo(dim=8).embeber(["dos palabras"])
    assert vectores.uso["tokens_in"] > 0
    assert "tokens_out" in vectores.uso

def test_d8_los_vectores_del_doble_son_unitarios():
    for vector in EmbeddingTextoFijo(dim=16).embeber(["a", "b", "c"]):
        assert sum(x * x for x in vector) ** 0.5 == pytest.approx(1.0, abs=1e-9)

def test_d8_un_vector_de_norma_cero_es_error_y_no_un_nan():
    with pytest.raises(ErrorDeConstrucción, match="norma 0"):
        normalizar([0.0, 0.0, 0.0], etiqueta="degenerado")

def test_d8_se_verifica_al_cargar_y_no_se_asume():
    assert normalizar([3.0, 4.0], etiqueta="x") == pytest.approx((0.6, 0.8))

def test_el_400_por_input_largo_se_detecta_antes_de_llamar_y_culpa_al_fragmentador():
    puerto = EmbeddingTextoOpenAI(
        api_key="sk-no-existe",
        base_url=PROXY,
        dim=8,
        limite_de_tokens=5,
        medidor=lambda t: len(t.split()),
    )
    with pytest.raises(InputDemasiadoLargo, match="Fragmentador"):
        puerto.embeber(
            ["una dos tres cuatro cinco seis"], etiquetas=["cma:1::description::0"]
        )

def test_el_error_de_input_largo_nombra_el_fragmento():
    puerto = EmbeddingTextoFijo(dim=8, limite_de_tokens=1, medidor=lambda t: len(t.split()))
    with pytest.raises(InputDemasiadoLargo, match="cma:1::provenance::3"):
        puerto.embeber(["a b"], etiquetas=["cma:1::provenance::3"])

def test_la_red_caida_es_un_error_de_construccion_y_no_una_degradacion():
    assert issubclass(RedCaída, ErrorDeConstrucción)

def test_el_limite_de_tokens_del_doble_es_el_del_modelo_por_defecto():
    assert EmbeddingTextoFijo().limite_de_tokens == 8191

def test_la_dim_declarada_es_la_del_modelo():
    assert embedding_texto.DIMS_CONOCIDAS["text-embedding-3-small"] == 1536

def test_k1_el_base_url_es_obligatorio_y_no_tiene_default():
    with pytest.raises(TypeError, match="base_url"):
        EmbeddingTextoOpenAI(api_key="sk-no-existe", dim=8)
    with pytest.raises(ErrorDeConstrucción, match="401"):
        EmbeddingTextoOpenAI(api_key="sk-no-existe", base_url="", dim=8)

def test_k1_el_base_url_llega_al_cliente_y_no_es_el_de_openai():
    puerto = EmbeddingTextoOpenAI(api_key="sk-no-existe", base_url=PROXY, dim=8)
    assert "api.openai.com" not in str(puerto._cliente.base_url)
    assert str(puerto._cliente.base_url).startswith("https://sa-llmproxy.it.itba.edu.ar")

def test_k1_el_base_url_no_entra_en_la_identidad_del_modelo():
    identidad = EmbeddingTextoOpenAI(
        api_key="sk-no-existe", base_url=PROXY, modelo="openai/text-embedding-3-small"
    ).identidad()
    assert identidad.a_json() == {"modelo": "openai/text-embedding-3-small", "dim": 1536}
    assert "sa-llmproxy" not in str(identidad.a_json())

def test_k2_la_dim_se_conoce_con_el_prefijo_del_proveedor():
    assert dim_conocida("openai/text-embedding-3-small") == 1536
    assert dim_conocida("text-embedding-3-small") == 1536
    assert dim_conocida("openai/modelo-que-no-existe") == 0

def test_el_puerto_expone_su_identidad_porque_entra_en_el_variante_id():
    identidad = EmbeddingTextoFijo(modelo="text-embedding-3-small", dim=1536).identidad()
    assert identidad.a_json() == {"modelo": "text-embedding-3-small", "dim": 1536}

def test_importar_el_modulo_no_importa_openai():
    cabecera = inspect.getsource(embedding_texto).split("class EmbeddingTextoOpenAI")[0]
    assert "import openai" not in cabecera
    assert "from openai" not in cabecera
