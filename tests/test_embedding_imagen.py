from __future__ import annotations

import sys

import pytest

from tp3_rag.nucleo import identidad
from tp3_rag.puertos.embedding_imagen import (
    DIM_ESPERADA,
    EmbeddingImagenFijo,
    LÍMITE_DE_TOKENS_DE_CLIP,
)

pytestmark = pytest.mark.nivel1

@pytest.fixture
def puerto() -> EmbeddingImagenFijo:
    return EmbeddingImagenFijo()

@pytest.mark.parametrize("datos", [b"una", b"otra", b"tercera"])
def test_los_vectores_de_imagen_tienen_norma_1(puerto, datos):
    v = puerto.embeber_imagen(datos)
    assert len(v) == DIM_ESPERADA
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9

@pytest.mark.parametrize("texto", ["un paisaje", "retrato oscuro", "colores cálidos"])
def test_los_vectores_de_texto_tambien(puerto, texto):
    v = puerto.embeber_texto(texto)
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9

def test_el_mismo_contenido_da_el_mismo_vector(puerto):
    assert puerto.embeber_imagen(b"x") == puerto.embeber_imagen(b"x")
    assert puerto.embeber_imagen(b"x") != puerto.embeber_imagen(b"y")

def test_el_singular_es_un_batch_de_uno(puerto):
    lote = puerto.embeber_imagenes([("cma:1", b"a"), ("cma:2", b"b")])
    assert lote.n == 2
    assert tuple(lote[0]) == puerto.embeber_imagen(b"a")

def test_la_truncacion_a_77_tokens_se_reporta(puerto):
    corto, largo = "un paisaje", " ".join(["palabra"] * (LÍMITE_DE_TOKENS_DE_CLIP + 5))
    salida = puerto.embeber_textos([corto, largo])
    assert salida.truncados == (1,)

def test_el_puerto_expone_checkpoint_y_version_del_preprocesado(puerto):
    i = puerto.identidad()
    assert i.a_json() == {
        "checkpoint": "fijo-para-tests",
        "version_del_preprocesado": "1",
        "dim": DIM_ESPERADA,
    }

def test_la_clave_del_cache_excluye_el_criterio_la_variante_y_el_split():
    import inspect

    parámetros = set(inspect.signature(identidad.clave_de_cache_de_embedding).parameters)
    assert parámetros == {
        "artwork_id",
        "hash_bytes",
        "checkpoint",
        "version_del_preprocesado",
    }
    assert not parámetros & {"criterio_de_inclusion", "variante_id", "split_id"}

@pytest.mark.parametrize(
    "cambio",
    [
        {"hash_bytes": "sha256:otro"},
        {"checkpoint": "openai/clip-vit-large-patch14"},
        {"version_del_preprocesado": "2"},
        {"artwork_id": "cma:2"},
    ],
)
def test_los_cuatro_ingredientes_cambian_la_clave(cambio):
    base = dict(
        artwork_id="cma:1",
        hash_bytes="sha256:ab",
        checkpoint="openai/clip-vit-base-patch32",
        version_del_preprocesado="1",
    )
    assert identidad.clave_de_cache_de_embedding(
        **base
    ) != identidad.clave_de_cache_de_embedding(**{**base, **cambio})

def test_el_nivel_1_no_carga_torch():
    assert "torch" not in sys.modules
