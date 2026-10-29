from __future__ import annotations

import pytest

from tp3_rag.modulo_a.normalizador_de_culture import CRITERIOS, NormalizadorDeCulture

pytestmark = pytest.mark.nivel1

@pytest.mark.parametrize(
    "crudo,esperado",
    [
        (["France, 19th century"], ["France"]),
        (["Netherlands"], ["Netherlands"]),
        (["China, Qing dynasty (1644-1911)"], ["China"]),
        (["France, 19th century", "Netherlands"], ["France", "Netherlands"]),
        (["France, 19th century", "France, 18th century"], ["France"]),
        (None, []),
        ([], []),
        ([""], []),
    ],
)
def test_primer_componente_por_coma(crudo, esperado):
    assert NormalizadorDeCulture("primer_componente_por_coma").normalizar(crudo) == esperado

def test_el_criterio_crudo_es_el_control_y_no_colapsa_nada():
    valores = ["France, 19th century", "France, 18th century"]
    assert NormalizadorDeCulture("crudo").normalizar(valores) == valores

def test_el_criterio_agresivo_colapsa_mas_y_se_equivoca_mas():
    agresivo = NormalizadorDeCulture("primer_componente_sin_parentesis_ni_periodo")
    assert agresivo.normalizar(["China Qing dynasty"]) == ["China"]
    assert agresivo.normalizar(["Middle Kingdom"]) == ["Middle Kingdom"]

@pytest.mark.parametrize("criterio", list(CRITERIOS))
def test_ningun_criterio_pierde_valores_en_silencio(criterio):
    n = NormalizadorDeCulture(criterio)
    for valor in ("Middle Kingdom", "France, 19th century", "Netherlands", "Qing dynasty"):
        assert n.normalizar([valor]), f"{criterio} vació {valor!r}"

def test_un_criterio_desconocido_no_corre_en_silencio():
    with pytest.raises(ValueError, match="desconocido"):
        NormalizadorDeCulture("lo_que_se_me_ocurrio")

def test_todos_los_criterios_son_puros_y_deterministas():
    valores = ["France, 19th century", "China, Qing dynasty (1644-1911)", "Netherlands"]
    for nombre in CRITERIOS:
        n = NormalizadorDeCulture(nombre)
        assert n.normalizar(valores) == n.normalizar(valores)

def test_a9b_publica_el_par_numero_mas_criterio(cfg, crudos_cma, corot):
    from tp3_rag.modulo_a.normalizador import Normalizador

    normalizador = Normalizador(cfg)
    registros = [normalizador.normalizar(c, "cma", None) for c in crudos_cma]

    a9b = NormalizadorDeCulture("primer_componente_por_coma").a9b(registros)
    assert a9b["criterio"] == "primer_componente_por_coma"
    assert a9b["total"]["distintos_antes"] >= a9b["total"]["distintos_despues"]
    assert (
        a9b["total"]["colapsados"]
        == a9b["total"]["distintos_antes"] - a9b["total"]["distintos_despues"]
    )
    assert "cma" in a9b["por_museo"]
    assert a9b["por_museo"]["cma"]["n_obras"] == len(registros)

def test_a9b_con_el_criterio_crudo_no_colapsa_nada(cfg, crudos_cma):
    from tp3_rag.modulo_a.normalizador import Normalizador

    registros = [Normalizador(cfg).normalizar(c, "cma", None) for c in crudos_cma]
    a9b = NormalizadorDeCulture("crudo").a9b(registros)
    assert a9b["total"]["colapsados"] == 0

def test_el_antes_de_a9b_es_computable_porque_el_crudo_sobrevive(cfg, corot):
    from tp3_rag.modulo_a.normalizador import Normalizador

    registro = Normalizador(cfg).normalizar(corot, "cma", None)
    for nombre in CRITERIOS:
        NormalizadorDeCulture(nombre).normalizar(registro.culture_raw)
    assert registro.culture_raw == ("France, 19th century",)
