from __future__ import annotations

import json

import pytest

from tp3_rag.dominio.juicios import AporteDeAnotador, Juicio
from tp3_rag.modulo_c.acuerdo import UMBRAL, Acuerdo, consenso, kappa_de_cohen
from tp3_rag.modulo_c.constructor_de_pool import ConstructorDePool
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeConstrucción

from conftest import (
    CONFIG_DE_PRUEBA,
    corridas_de_prueba,
    manifiesto_de_prueba,
    rubrica_de_prueba,
    set_de_prueba,
)

pytestmark = pytest.mark.nivel1

@pytest.fixture
def pool(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    return ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=corridas_de_prueba(set_),
    ).pool

def _aporte(pool, anotador, grados, *, rubrica=None, pares=None, hash_del_texto=None):
    rubrica = rubrica or rubrica_de_prueba()
    items = list(pool.items) if pares is None else [i for i in pool.items if i.par in pares]
    juicios = tuple(
        Juicio(consulta_id=i.consulta_id, artwork_id=i.artwork_id, grado=grados[n % len(grados)])
        for n, i in enumerate(items)
    )
    ident = identidad.identidad_del_aporte(
        split_id=pool.split_id, rubrica_id=rubrica.rubrica_id, anotador=anotador
    )
    aporte = AporteDeAnotador(
        procedencia={
            "identidad": ident,
            "split_id": pool.split_id,
            "rubrica_id": rubrica.rubrica_id,
            "anotador": anotador,
            "pool_id": pool.pool_id,
            "rubrica": {
                "rubrica_id": rubrica.rubrica_id,
                "hash_del_texto": hash_del_texto or rubrica.hash_del_texto,
            },
            "conteos": {
                "n_juicios": len(juicios),
                "por_grado": {
                    f"grado_{g}": sum(1 for j in juicios if j.grado == g) for g in (0, 1, 2)
                },
            },
        },
        juicios=juicios,
    )
    aporte.validar()
    return aporte

def test_acuerdo_perfecto_da_kappa_uno(pool):
    a = _aporte(pool, "ana", [0, 1, 2])
    b = _aporte(pool, "beto", [0, 1, 2])
    k = kappa_de_cohen(a, b)

    assert k.n == 20
    assert k.acuerdo_observado == 1.0
    assert k.kappa == pytest.approx(1.0)
    assert k.pasa_el_umbral is True

def test_un_kappa_bajo_se_calcula_y_no_falla(pool):
    a = _aporte(pool, "ana", [0, 1, 2])
    b = _aporte(pool, "beto", [1, 2, 0])
    k = kappa_de_cohen(a, b)

    assert k.kappa is not None
    assert k.kappa < UMBRAL
    assert k.pasa_el_umbral is False
    assert "hallazgo" in k.veredicto
    assert "refinarla" in k.veredicto

def test_kappa_descuenta_el_acuerdo_por_azar(pool):
    pares = sorted(pool.pares)[:10]
    a = _aporte(pool, "ana", [2], pares=pares)
    b = _aporte(pool, "beto", [2, 2, 2, 2, 2, 2, 2, 2, 2, 1], pares=pares)
    k = kappa_de_cohen(a, b)

    assert k.acuerdo_observado == pytest.approx(0.9)
    assert k.kappa is not None and k.kappa < 0.1

def test_kappa_es_ausente_y_no_cero_cuando_todos_usan_un_solo_grado(pool):
    a = _aporte(pool, "ana", [0])
    b = _aporte(pool, "beto", [0])
    k = kappa_de_cohen(a, b)

    assert k.kappa is None
    assert k.pasa_el_umbral is None
    assert k.acuerdo_observado == 1.0
    assert "no está definido" in k.razon_de_ausencia
    assert "κ **ausente**" in k.veredicto

def test_los_pares_que_juzgo_solo_uno_no_entran_al_kappa_y_se_declaran(pool):
    todos = sorted(pool.pares)
    a = _aporte(pool, "ana", [0, 1, 2])
    b = _aporte(pool, "beto", [0, 1, 2], pares=todos[:15])
    k = kappa_de_cohen(a, b)

    assert k.n == 15
    assert len(k.solo_uno["ana"]) == 5
    assert k.solo_uno["beto"] == ()
    assert "15 pares" in k.poblacion

def test_sin_pares_comunes_kappa_es_ausente_con_su_razon(pool):
    todos = sorted(pool.pares)
    a = _aporte(pool, "ana", [0], pares=todos[:10])
    b = _aporte(pool, "beto", [0], pares=todos[10:])
    k = kappa_de_cohen(a, b)

    assert k.kappa is None
    assert "no comparten ni un par" in k.razon_de_ausencia

def test_dos_rubricas_no_se_cruzan(cfg, almacen_temporal, rutas_de_prueba, pool):
    piloto = rubrica_de_prueba(designacion="piloto")
    v2 = rubrica_de_prueba(designacion="v2")
    aportes = [_aporte(pool, "ana", [0, 1, 2], rubrica=piloto),
               _aporte(pool, "beto", [0, 1, 2], rubrica=v2)]

    with pytest.raises(ErrorDeConstrucción) as e:
        Acuerdo(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).calcular(
            pool=pool, rubrica=piloto, aportes=aportes
        )
    assert "κ necesita dos" in str(e.value)
    assert "no significa nada" in str(e.value)

def test_un_mismo_rubrica_id_con_otro_texto_se_niega(
    cfg, almacen_temporal, rutas_de_prueba, pool
):
    piloto = rubrica_de_prueba()
    aportes = [
        _aporte(pool, "ana", [0, 1, 2]),
        _aporte(pool, "beto", [0, 1, 2], hash_del_texto="0000000000000000"),
    ]

    with pytest.raises(ErrorDeConstrucción) as e:
        Acuerdo(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).calcular(
            pool=pool, rubrica=piloto, aportes=aportes
        )
    assert "otro hash de texto" in str(e.value)
    assert "sin cambiar de designación" in str(e.value)

def test_un_solo_aporte_no_alcanza(cfg, almacen_temporal, rutas_de_prueba, pool):
    with pytest.raises(ErrorDeConstrucción, match="κ necesita dos"):
        Acuerdo(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).calcular(
            pool=pool, rubrica=rubrica_de_prueba(), aportes=[_aporte(pool, "ana", [0])]
        )

def test_con_tres_aportes_hay_que_declarar_la_pareja(
    cfg, almacen_temporal, rutas_de_prueba, pool
):
    aportes = [_aporte(pool, n, [0, 1, 2]) for n in ("ana", "beto", "cora")]
    with pytest.raises(ErrorDeConstrucción, match="κ de Cohen es entre"):
        Acuerdo(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).calcular(
            pool=pool, rubrica=rubrica_de_prueba(), aportes=aportes
        )

def test_el_consenso_resuelve_el_acuerdo_y_marca_lo_que_no(pool):
    a = _aporte(pool, "ana", [0, 1, 2])
    b = _aporte(pool, "beto", [0, 2, 2])
    filas = consenso(pool, [a, b], pareja=("ana", "beto"))

    por_resolución = {}
    for f in filas:
        por_resolución[f.como_se_resolvio] = por_resolución.get(f.como_se_resolvio, 0) + 1
    assert por_resolución["acuerdo"] + por_resolución["sin_consenso"] == 20
    assert por_resolución["sin_consenso"] > 0
    assert all(f.grado is None for f in filas if f.como_se_resolvio == "sin_consenso")

def test_el_tercer_aporte_desempata_y_el_consenso_sigue_siendo_derivable(pool):
    a = _aporte(pool, "ana", [0, 1, 2])
    b = _aporte(pool, "beto", [0, 2, 2])
    c = _aporte(pool, "cora", [1])
    filas = consenso(pool, [a, b, c], pareja=("ana", "beto"))

    desempatadas = [f for f in filas if f.como_se_resolvio == "desempate"]
    assert desempatadas
    assert all(f.grado == 1 for f in desempatadas)
    assert not [f for f in filas if f.como_se_resolvio == "sin_consenso"]
    assert consenso(pool, [a, b, c], pareja=("ana", "beto")) == filas

def test_escribe_consenso_y_kappa_con_la_rubrica_en_el_nombre(
    cfg, almacen_temporal, rutas_de_prueba, pool
):
    rubrica = rubrica_de_prueba()
    resultado = Acuerdo(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).calcular(
        pool=pool,
        rubrica=rubrica,
        aportes=[_aporte(pool, "ana", [0, 1, 2]), _aporte(pool, "beto", [0, 1, 1])],
    )

    assert resultado.ruta_del_consenso.name == f"consenso-{pool.pool_id}-piloto.csv"
    assert resultado.ruta_del_kappa.name == f"kappa-{pool.pool_id}-piloto.json"

    publicado = json.loads(resultado.ruta_del_kappa.read_text(encoding="utf-8"))
    assert publicado["kappa"]["umbral"] == UMBRAL
    assert publicado["kappa"]["poblacion"]
    assert publicado["rubrica"]["hash_del_texto"] == rubrica.hash_del_texto
    assert publicado["pareja_del_kappa"] == ["ana", "beto"]
    assert "no se suman ni se promedian" in publicado["nota_de_los_dos_kappa"].lower()
    from tp3_rag.dominio import ground_truth as gt

    procedencia, filas = gt.partir_csv(
        resultado.ruta_del_consenso.read_text(encoding="utf-8")
    )
    assert procedencia["pool_id"] == pool.pool_id
    assert procedencia["rubrica"]["rubrica_id"] == "piloto"
    assert len(filas) == 20
