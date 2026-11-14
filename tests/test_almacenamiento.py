from __future__ import annotations

import pytest

from tp3_rag.nucleo.errores import ErrorDeArtefacto
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos, YaExiste

pytestmark = pytest.mark.nivel1

@pytest.fixture
def almacén(tmp_path) -> AlmacenamientoEnArchivos:
    return AlmacenamientoEnArchivos(tmp_path)

def test_la_escritura_no_deja_parciales(almacén, tmp_path):
    almacén.escribir_json("a/b.json", {"x": 1})
    assert not list(tmp_path.rglob("*.parcial"))

def test_un_artefacto_que_no_valida_se_rechaza(almacén):
    almacén.escribir_json("r.json", {"ordinal": 1})
    with pytest.raises(ErrorDeArtefacto, match="config_id"):
        almacén.leer_json("r.json", claves_obligatorias=("ordinal", "config_id"))

def test_un_json_roto_se_rechaza(almacén, tmp_path):
    (tmp_path / "x.json").write_text("{no json", encoding="utf-8")
    with pytest.raises(ErrorDeArtefacto, match="JSON"):
        almacén.leer_json("x.json")

def test_reservar_es_exclusivo(almacén):
    almacén.reservar("x.json")
    with pytest.raises(YaExiste):
        almacén.reservar("x.json")

def test_lo_que_impide_la_colision_es_la_reserva_no_el_ingrediente(almacén):
    plantilla = "ingesta/abc-{ordinal}.json"
    primeros = [almacén.reservar_ordinal(plantilla)[0] for _ in range(5)]
    assert primeros == [1, 2, 3, 4, 5]

def test_el_ordinal_es_por_catalogo_id(almacén):
    assert almacén.reservar_ordinal("ingesta/aaa-{ordinal}.json")[0] == 1
    assert almacén.reservar_ordinal("ingesta/bbb-{ordinal}.json")[0] == 1

