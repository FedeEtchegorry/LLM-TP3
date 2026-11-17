"""El set complementario de consultas mixtas: una por imagen, aparte del set de 30."""
from __future__ import annotations

import copy

import pytest

from tp3_rag.dominio.consultas import CATEGORÍA_MIXTA, SetDeConsultas
from tp3_rag.nucleo.errores import ErrorDeArtefacto

SPLIT = "5b6cba32126fed61"


def _contenido(autoria: str, categorias_por_imagen: list[list[str]]) -> dict:
    imagenes = []
    for i, categorias in enumerate(categorias_por_imagen, 1):
        imagenes.append(
            {
                "artwork_id": f"aic:{i}",
                "imagen_hash": "sha256:" + "0" * 64,
                "archivo": f"imagenes/aic_{i}.jpg",
                "atributos": ["paisaje"],
                "es_ejemplo": False,
                "consultas": [
                    {
                        "consulta_id": f"c{i:02d}-{cat}",
                        "categoria": cat,
                        "pedido": "un pedido",
                        "es_ejemplo": False,
                    }
                    for cat in categorias
                ],
            }
        )
    d = {
        "identidad": f"{SPLIT}-{autoria}",
        "split_id": SPLIT,
        "autoria": autoria,
        "estado": "completo",
        "criterio_de_culture": "primer_componente_por_coma",
        "revision": {"n_imagenes_revisadas": 30},
        "imagenes": imagenes,
    }
    d["conteos"] = _conteos(d)
    return d


def _conteos(d: dict) -> dict:
    consultas = [c for img in d["imagenes"] for c in img["consultas"]]
    por_categoria: dict[str, int] = {}
    for c in consultas:
        por_categoria[c["categoria"]] = por_categoria.get(c["categoria"], 0) + 1
    return {
        "n_imagenes": len(d["imagenes"]),
        "n_consultas": len(consultas),
        "n_pedidos_culturales": 0,
        "por_categoria": por_categoria,
    }


def _mixto(n_imagenes: int = 10) -> dict:
    return _contenido("grupo-2-mixto", [[CATEGORÍA_MIXTA]] * n_imagenes)


def test_el_set_mixto_valida_y_es_complementario():
    s = SetDeConsultas(_mixto())
    s.validar()
    assert s.es_complementario
    assert s.por_categoria() == {"mixto": 10}


def test_el_set_de_tres_categorias_no_es_complementario():
    tres = ["visual_cromatico", "ambiguo"]  # sin la cultural: se prueba solo la forma
    s = SetDeConsultas(_contenido("grupo-2", [tres] * 10))
    assert not s.es_complementario


def test_dos_mixtas_en_la_misma_imagen_se_rechazan():
    d = _contenido("grupo-2-mixto", [[CATEGORÍA_MIXTA, CATEGORÍA_MIXTA]] + [[CATEGORÍA_MIXTA]] * 9)
    with pytest.raises(ErrorDeArtefacto):
        SetDeConsultas(d).validar()


def test_un_set_mezclado_se_rechaza():
    d = _mixto()
    d["imagenes"][3]["consultas"][0]["categoria"] = "ambiguo"
    d["conteos"] = _conteos(d)
    with pytest.raises(ErrorDeArtefacto):
        SetDeConsultas(d).validar()


def test_un_set_mixto_completo_con_nueve_consultas_se_rechaza():
    with pytest.raises(ErrorDeArtefacto):
        SetDeConsultas(_mixto(9)).validar()


def test_conteos_que_no_cierran_se_rechazan():
    d = copy.deepcopy(_mixto())
    d["conteos"]["por_categoria"] = {"mixto": 9}
    with pytest.raises(ErrorDeArtefacto):
        SetDeConsultas(d).validar()
