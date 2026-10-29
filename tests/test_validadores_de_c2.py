from __future__ import annotations

import pytest

from conftest import corrida_de_prueba, plan_de_prueba
from tp3_rag.modulo_c import validadores as val

pytestmark = pytest.mark.nivel1

TOP5 = [f"cma:{100 + k}" for k in range(5)]
GRID = (0, 0.25, 0.5, 0.75, 1)

def _corrida(consulta_id="c01-visual", *, plan=None, **kw):
    return corrida_de_prueba(consulta_id=consulta_id, top5=TOP5, plan=plan, **kw)

def test_pl3_cuenta_las_tres_clausulas_por_separado():
    corridas = [
        _corrida("c01-visual", plan=plan_de_prueba(plan_id="a" * 16)),
        _corrida(
            "c02-visual",
            plan=plan_de_prueba(
                plan_id="b" * 16, consulta_id="c02-visual",
                visual="lo mismo", conceptual="lo mismo",
            ),
        ),
        _corrida(
            "c03-visual",
            plan=plan_de_prueba(
                plan_id="c" * 16, consulta_id="c03-visual",
                visual="el pedido", pedido="el pedido",
            ),
        ),
    ]
    r = val.pl3_evidencia_de_a3(corridas)
    assert not r.pasa
    assert r.marcador == "1/3"
    assert r.detalle["no_vacias"] == "3/3"
    assert r.detalle["distintas_entre_si"] == "2/3"
    assert r.detalle["distintas_del_pedido_crudo"] == "2/3"
    rotas = {f["consulta_id"]: f["clausulas_rotas"] for f in r.fallas}
    assert rotas["c02-visual"] == ["distintas_entre_si"]
    assert rotas["c03-visual"] == ["distintas_del_pedido_crudo"]

def test_pl3_no_cuenta_las_corridas_sin_plan():
    corridas = [_corrida("c01-visual"), _corrida("c02-visual")]
    r = val.pl3_evidencia_de_a3(corridas)
    assert r.de == 0 and not r.pasa

def test_pl3_y_pl5_entran_solos_a_criterios_cuando_hay_planes():
    sin_plan = [_corrida("c01-visual")]
    con_plan = [_corrida("c01-visual", plan=plan_de_prueba())]
    comunes = dict(ids_del_corpus=TOP5, exigen_atribucion={"CC0": False})
    criterios = lambda rs: " | ".join(r.criterio for r in rs)  # noqa: E731
    assert "cláusulas del plan" not in criterios(
        val.todos_sobre_corridas(sin_plan, **comunes)
    )
    con = criterios(val.todos_sobre_corridas(con_plan, **comunes))
    assert "las tres cláusulas del plan" in con
    assert "la escritura única del plan" in con

def test_pl5_un_plan_lo_produce_una_corrida_y_el_resto_lo_reusa():
    plan_id = "d" * 16
    corridas = [
        _corrida("c01-visual", plan=plan_de_prueba(plan_id=plan_id, reusado=False)),
        *(
            _corrida("c01-visual", plan=plan_de_prueba(plan_id=plan_id, reusado=True), alfa=a)
            for a in GRID[1:]
        ),
    ]
    r = val.pl5_escritura_unica(corridas)
    assert r.pasa
    assert r.detalle["producidos"] == 1
    assert r.detalle["reusados"] == 4

def test_pl5_dos_corridas_produciendo_el_mismo_plan_es_falla():
    plan_id = "e" * 16
    corridas = [
        _corrida("c01-visual", plan=plan_de_prueba(plan_id=plan_id)),
        _corrida("c02-visual", plan=plan_de_prueba(plan_id=plan_id)),
    ]
    r = val.pl5_escritura_unica(corridas)
    assert not r.pasa
    assert r.fallas[0]["plan_id"] == plan_id

def test_pl5_cero_producciones_no_es_falla():
    corridas = [
        _corrida("c01-visual", plan=plan_de_prueba(plan_id="f" * 16, reusado=True), alfa=a)
        for a in GRID
    ]
    assert val.pl5_escritura_unica(corridas).pasa

def test_r11_un_plan_id_y_cinco_corrida_id():
    plan_id = "a1b2c3d4e5f60718"
    corridas = [
        _corrida(
            "c01-visual",
            plan=plan_de_prueba(plan_id=plan_id, reusado=i > 0),
            alfa=a,
            config_id=f"{i:016d}",
        )
        for i, a in enumerate(GRID)
    ]
    r = val.r11_un_plan_cinco_corridas(corridas, consulta_id="c01-visual")
    assert r.pasa
    assert r.detalle["n_plan_id"] == 1
    assert r.detalle["n_corrida_id"] == 5
    assert r.detalle["alfas"] == sorted(GRID)
    assert r.detalle["planes_producidos"] == 1
    assert r.detalle["planes_reusados"] == 4
    assert len({c.config_id for c in corridas}) == 5

def test_r11_falla_si_el_alfa_entro_en_el_plan_id():
    corridas = [
        _corrida(
            "c01-visual",
            plan=plan_de_prueba(plan_id=f"{i:016d}"),
            alfa=a,
            config_id=f"{i:016d}",
        )
        for i, a in enumerate(GRID)
    ]
    r = val.r11_un_plan_cinco_corridas(corridas, consulta_id="c01-visual")
    assert not r.pasa
    assert r.detalle["n_plan_id"] == 5

def test_r11_no_se_cae_por_el_segundo_ordinal_que_r12_crea():
    plan_id = "c" * 16
    corridas = [
        _corrida(
            "c01-visual",
            plan=plan_de_prueba(plan_id=plan_id, reusado=i > 0),
            alfa=a,
            config_id=f"{i:016d}",
        )
        for i, a in enumerate(GRID)
    ]
    corridas.append(
        _corrida(
            "c01-visual",
            plan=plan_de_prueba(plan_id=plan_id, reusado=True),
            alfa=0.5,
            config_id=f"{2:016d}",
            ordinal=2,
        )
    )
    r = val.r11_un_plan_cinco_corridas(corridas, consulta_id="c01-visual")
    assert r.pasa
    assert r.detalle["n_corrida_id"] == 5
    assert "6 corrida(s) en disco" in r.poblacion

def test_r11_falla_si_falta_una_corrida_del_barrido():
    plan_id = "b" * 16
    corridas = [
        _corrida("c01-visual", plan=plan_de_prueba(plan_id=plan_id), alfa=a, config_id=f"{i:016d}")
        for i, a in enumerate(GRID[:4])
    ]
    r = val.r11_un_plan_cinco_corridas(corridas, consulta_id="c01-visual")
    assert not r.pasa
    assert r.marcador == "4/5"
