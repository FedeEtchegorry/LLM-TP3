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

def test_los_vectores_de_imagen_tienen_norma_1(puerto):
    v = puerto.embeber_imagen(b"una")
    assert len(v) == DIM_ESPERADA
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9

def test_los_vectores_de_texto_tambien(puerto):
    v = puerto.embeber_texto("un paisaje")
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9

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
