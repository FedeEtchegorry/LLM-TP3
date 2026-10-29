from __future__ import annotations

import pytest

from tp3_rag.dominio import ground_truth as gt
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConfiguración

pytestmark = pytest.mark.nivel1

def test_las_cinco_rutas_salen_de_la_misma_raiz(cfg):
    rutas = gt.rutas(cfg)

    assert rutas.raiz.as_posix() == "artifacts/ground-truth"
    assert rutas.consultas.as_posix() == "artifacts/ground-truth/consultas"
    assert rutas.rubricas.as_posix() == "artifacts/ground-truth/rubricas"
    assert rutas.pool.as_posix() == "artifacts/ground-truth/pool"
    assert rutas.juicios.as_posix() == "artifacts/ground-truth/juicios"
    assert rutas.acuerdo == rutas.raiz
    assert rutas.planillas.as_posix() == "planillas"

def test_si_la_configuracion_mueve_consultas_el_ground_truth_falla(cfg):
    movida = Configuración.cargar(
        overrides={"rutas.ground_truth_consultas": "otro/lugar/consultas"}
    )
    with pytest.raises(ErrorDeConfiguración, match="ground_truth_consultas"):
        gt.rutas(movida)

def test_el_config_id_no_se_mueve_por_las_rutas_de_la_anotacion(cfg):
    con_una_clave_mas = Configuración.cargar(overrides={"consulta.top_k": 6})

    assert con_una_clave_mas.config_id != cfg.config_id
    assert gt.rutas(cfg).juicios == gt.rutas(con_una_clave_mas).juicios

def test_el_preambulo_es_toml_comentado_y_vuelve_a_parsear():
    procedencia = {
        "identidad": "0123456789abcdef-piloto-ana",
        "n": 20,
        "listo": True,
        "etiquetas": ["a", "b"],
        "conteos": {"n_juicios": 20, "por_grado": {"grado_0": 7}},
    }
    texto = gt.armar_csv(
        procedencia=procedencia,
        columnas=("a", "b"),
        filas=[{"a": "1", "b": "x"}, {"a": "2", "b": "y"}],
    )
    leída, filas = gt.partir_csv(texto)

    assert leída == procedencia
    assert filas == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    assert texto.splitlines()[0].startswith("#")

def test_una_clave_que_no_es_desnuda_se_escribe_entre_comillas():
    texto = gt.armar_csv(
        procedencia={"licencias": {"CC-BY-4.0": 3, "CC0": 17}},
        columnas=("a",),
        filas=[{"a": "1"}],
    )
    leída, _ = gt.partir_csv(texto)

    assert leída["licencias"] == {"CC-BY-4.0": 3, "CC0": 17}

def test_un_csv_sin_preambulo_se_rechaza():
    with pytest.raises(ErrorDeArtefacto, match="preámbulo"):
        gt.partir_csv("a,b\n1,2\n")

def test_una_nota_con_comillas_y_saltos_sobrevive_al_ida_y_vuelta():
    nota = 'dijo "no cumple" y\nsiguió'
    leída, _ = gt.partir_csv(
        gt.armar_csv(procedencia={"nota": nota}, columnas=("a",), filas=[{"a": "1"}])
    )

    assert leída["nota"] == nota
