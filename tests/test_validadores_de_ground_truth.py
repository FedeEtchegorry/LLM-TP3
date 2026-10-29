from __future__ import annotations

import pytest

from tp3_rag.dominio.juicios import AporteDeAnotador, Juicio
from tp3_rag.modulo_c import validadores as val
from tp3_rag.modulo_c.constructor_de_pool import ConstructorDePool
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDelProyecto

from conftest import (
    CONFIG_DE_PRUEBA,
    SPLIT_DE_PRUEBA,
    VARIANTE_DE_PRUEBA,
    corridas_de_prueba,
    manifiesto_de_prueba,
    registro_de_variante_de_prueba,
    set_de_prueba,
)

pytestmark = pytest.mark.nivel1

@pytest.fixture
def montaje(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    manifiesto = manifiesto_de_prueba()
    pool = ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto,
        corridas=corridas_de_prueba(set_),
    ).pool
    return {
        "set": set_,
        "manifiesto": manifiesto,
        "pool": pool,
        "registros": {VARIANTE_DE_PRUEBA: registro_de_variante_de_prueba()},
    }

def _aporte(pool, anotador, pares, *, split_id=None):
    split_id = split_id or pool.split_id
    juicios = tuple(
        Juicio(consulta_id=c, artwork_id=o, grado=1) for c, o in sorted(pares)
    )
    return AporteDeAnotador(
        procedencia={
            "identidad": identidad.identidad_del_aporte(
                split_id=split_id, rubrica_id="piloto", anotador=anotador
            ),
            "split_id": split_id,
            "rubrica_id": "piloto",
            "anotador": anotador,
            "pool_id": pool.pool_id,
            "conteos": {"n_juicios": len(juicios)},
        },
        juicios=juicios,
    )

def test_e5_pasa_cuando_los_juicios_salen_del_pool(montaje):
    pool = montaje["pool"]
    resultado = val.e5_pares_juzgados_en_algun_pool(
        aportes=[_aporte(pool, "ana", pool.pares)], pools=[pool], split_id=pool.split_id
    )

    assert resultado.pasa
    assert resultado.marcador == "20/20"
    assert pool.pool_id in resultado.detalle["pools_del_split"]

def test_e5_caza_un_par_que_no_estuvo_en_ningun_pool(montaje):
    pool = montaje["pool"]
    inventado = {("c01-visual", "cma:999999")}
    resultado = val.e5_pares_juzgados_en_algun_pool(
        aportes=[_aporte(pool, "ana", pool.pares | inventado)],
        pools=[pool],
        split_id=pool.split_id,
    )

    assert not resultado.pasa
    assert resultado.marcador == "20/21"
    assert resultado.fallas[0]["artwork_id"] == "cma:999999"

def test_e5_el_cuantificador_es_algun_pool_y_no_el_que_lo_exporto(
    cfg, almacen_temporal, rutas_de_prueba, montaje
):
    pool = montaje["pool"]
    otro = ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
        designacion="definitivo-de-prueba",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=montaje["set"],
        manifiesto=montaje["manifiesto"],
        corridas=corridas_de_prueba(montaje["set"]),
        regla=None,
    ).pool
    aporte = _aporte(pool, "ana", pool.pares)
    resultado = val.e5_pares_juzgados_en_algun_pool(
        aportes=[aporte], pools=[otro, pool], split_id=pool.split_id
    )

    assert resultado.pasa
    assert len(resultado.detalle["pools_del_split"]) == 2

def test_e5_sin_juicios_pasa_y_lo_dice(montaje):
    resultado = val.e5_pares_juzgados_en_algun_pool(
        aportes=[], pools=[montaje["pool"]], split_id=montaje["pool"].split_id
    )

    assert resultado.pasa
    assert resultado.marcador == "0/0"
    assert resultado.detalle["n_pares_en_algun_pool"] == 20

def test_e10_pasa_con_el_pool_de_c1(montaje):
    resultado = val.e10_puerta_de_la_anotacion(
        pool=montaje["pool"],
        manifiesto=montaje["manifiesto"],
        registros_de_variantes=montaje["registros"],
    )

    assert resultado.pasa
    assert resultado.marcador == "20/20"
    assert resultado.detalle["variantes_de_origen"] == [VARIANTE_DE_PRUEBA]

def test_e10_mitad_b_caza_una_obra_del_held_out(montaje):
    pool = montaje["pool"]
    del_held_out = montaje["manifiesto"].ids_held_out[0]
    pool.procedencia["origen"].append(
        {
            "consulta_id": pool.items[0].consulta_id,
            "artwork_id": del_held_out,
            "configuracion": "C1",
            "config_id": "x",
            "corrida_id": "y",
            "rank": 1,
            "score": 0.5,
        }
    )
    from tp3_rag.dominio.pool import Pool, ÍtemDePool

    con_intrusa = Pool(
        procedencia=pool.procedencia,
        items=pool.items + (
            ÍtemDePool(fila=21, consulta_id=pool.items[0].consulta_id,
                       artwork_id=del_held_out, ficha={}),
        ),
    )
    resultado = val.e10_puerta_de_la_anotacion(
        pool=con_intrusa,
        manifiesto=montaje["manifiesto"],
        registros_de_variantes=montaje["registros"],
    )

    assert not resultado.pasa
    assert resultado.fallas[0]["mitad"] == "b"
    assert resultado.fallas[0]["artwork_id"] == del_held_out
    assert "held-out" in resultado.fallas[0]["problema"]

def test_e10_mitad_a_caza_una_variante_de_otro_split(montaje):
    resultado = val.e10_puerta_de_la_anotacion(
        pool=montaje["pool"],
        manifiesto=montaje["manifiesto"],
        registros_de_variantes={
            VARIANTE_DE_PRUEBA: registro_de_variante_de_prueba(split_id="ffffffffffffffff")
        },
    )

    assert not resultado.pasa
    assert resultado.fallas[0]["mitad"] == "a"
    assert resultado.fallas[0]["split_de_la_variante"] == "ffffffffffffffff"

def test_e10_una_variante_que_no_resuelve_no_se_da_por_buena(montaje):
    resultado = val.e10_puerta_de_la_anotacion(
        pool=montaje["pool"], manifiesto=montaje["manifiesto"], registros_de_variantes={}
    )

    assert not resultado.pasa
    assert "no se pudo resolver" in resultado.fallas[0]["problema"]

def test_e10_no_se_queja_de_la_imagen_de_consulta_que_si_es_del_held_out(montaje):
    set_ = montaje["set"]
    imágenes = {c.artwork_id for c in set_.consultas()}
    held_out = set(montaje["manifiesto"].ids_held_out)

    assert imágenes <= held_out
    assert val.e10_puerta_de_la_anotacion(
        pool=montaje["pool"],
        manifiesto=montaje["manifiesto"],
        registros_de_variantes=montaje["registros"],
    ).pasa

def test_antes_de_exportar_corre_los_dos_y_corta_si_alguno_falla(montaje):
    with pytest.raises(ErrorDelProyecto, match="No se exporta la planilla"):
        val.antes_de_exportar(
            pool=montaje["pool"],
            manifiesto=montaje["manifiesto"],
            set_de_consultas=montaje["set"],
            registros_de_variantes={},
            aportes=[],
            pools=[montaje["pool"]],
        )

def test_antes_de_exportar_devuelve_los_dos_resultados_cuando_pasan(montaje):
    resultados = val.antes_de_exportar(
        pool=montaje["pool"],
        manifiesto=montaje["manifiesto"],
        set_de_consultas=montaje["set"],
        registros_de_variantes=montaje["registros"],
        aportes=[],
        pools=[montaje["pool"]],
    )

    assert len(resultados) == 2
    assert all(r.pasa for r in resultados)
    assert all(r.detalle["momento"] == "antes de exportar la planilla" for r in resultados)

def test_antes_de_exportar_rechaza_un_set_de_otro_split(montaje):
    with pytest.raises(ErrorDelProyecto, match="ancla al split"):
        val.antes_de_exportar(
            pool=montaje["pool"],
            manifiesto=montaje["manifiesto"],
            set_de_consultas=set_de_prueba(split_id="ffffffffffffffff"),
            registros_de_variantes=montaje["registros"],
            aportes=[],
            pools=[montaje["pool"]],
        )
