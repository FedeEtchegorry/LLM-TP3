from __future__ import annotations

import pytest

from tp3_rag.nucleo.errores import ErrorDeArtefacto
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos, YaExiste

pytestmark = pytest.mark.nivel1

@pytest.fixture
def almacén(tmp_path) -> AlmacenamientoEnArchivos:
    return AlmacenamientoEnArchivos(tmp_path)

def test_escribir_y_leer_json(almacén):
    almacén.escribir_json("a/b.json", {"x": 1})
    assert almacén.leer_json("a/b.json") == {"x": 1}

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

def test_un_artefacto_ausente_se_rechaza(almacén):
    with pytest.raises(ErrorDeArtefacto, match="no existe"):
        almacén.leer_json("no-esta.json")

def test_reservar_es_exclusivo(almacén):
    almacén.reservar("x.json")
    with pytest.raises(YaExiste):
        almacén.reservar("x.json")

def test_el_ordinal_arranca_en_uno_y_avanza(almacén):
    plantilla = "ingesta/abc-{ordinal}.json"
    assert almacén.reservar_ordinal(plantilla)[0] == 1
    assert almacén.reservar_ordinal(plantilla)[0] == 2
    assert almacén.reservar_ordinal(plantilla)[0] == 3

def test_lo_que_impide_la_colision_es_la_reserva_no_el_ingrediente(almacén):
    plantilla = "ingesta/abc-{ordinal}.json"
    primeros = [almacén.reservar_ordinal(plantilla)[0] for _ in range(5)]
    assert primeros == [1, 2, 3, 4, 5]
    assert len(set(primeros)) == 5

def test_el_ordinal_es_por_catalogo_id(almacén):
    assert almacén.reservar_ordinal("ingesta/aaa-{ordinal}.json")[0] == 1
    assert almacén.reservar_ordinal("ingesta/bbb-{ordinal}.json")[0] == 1

def test_la_identidad_esta_en_el_nombre(almacén):
    _, ruta = almacén.reservar_ordinal("ingesta/94e82ee1-{ordinal}.json")
    assert ruta.name == "94e82ee1-1.json"

def test_jsonl_una_fila_por_linea(almacén, tmp_path):
    almacén.escribir_jsonl("c/registros.jsonl", [{"a": 1}, {"a": 2}])
    líneas = (tmp_path / "c/registros.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(líneas) == 2
