from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, CatálogoCanónico, Imagen, RegistroCanónico
from tp3_rag.nucleo.errores import ErrorDeArtefacto
from tp3_rag.nucleo.identidad import hash_de_registros
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

pytestmark = pytest.mark.nivel1

def _registro(aid: str, texto: str = "prosa"):
    return RegistroCanónico(
        artwork_id=aid, museum=aid.split(":")[0], type="Painting", title="t", artist="a",
        date_display="1800", year=1800, medium="oil", style=None, culture_raw=("France",),
        source_url="https://x/1", imagen=Imagen("https://x/1.jpg", "sha256:ab"),
        media_license="CC0",
        campos_de_texto=(CampoDeTexto("description", texto, "https://x/1", "CC0"),),
        procedencia={},
    )

@pytest.fixture
def almacén(tmp_path):
    return AlmacenamientoEnArchivos(tmp_path)

def _escribir(almacén, registros, *, catalogo_id="cat1", mentir=None):
    filas = [r.a_json() for r in registros]
    almacén.escribir_jsonl("data/catalogo/cat1/registros.jsonl", filas)
    declarado = {
        "artefacto": "CatalogoCanonico",
        "catalogo_id": catalogo_id,
        "hash_de_registros": hash_de_registros(filas),
        "n_registros": len(filas),
    }
    declarado.update(mentir or {})
    almacén.escribir_json("data/catalogo/cat1/catalogo.json", declarado)
    return "data/catalogo/cat1"

def test_se_lee_y_se_verifica(almacén):
    base = _escribir(almacén, [_registro("cma:1"), _registro("aic:2")])
    catálogo = CatálogoCanónico.cargar(almacén, base)
    assert catálogo.catalogo_id == "cat1"
    assert len(catálogo.registros) == 2
    assert catálogo.registros[0].campos_de_texto[0].license == "CC0"

def test_un_registro_editado_a_mano_se_rechaza(almacén):
    base = _escribir(almacén, [_registro("cma:1")])
    ruta = almacén.ruta(base) / "registros.jsonl"
    ruta.write_text(ruta.read_text(encoding="utf-8").replace("prosa", "otra cosa"), encoding="utf-8")
    with pytest.raises(ErrorDeArtefacto, match="no coincide"):
        CatálogoCanónico.cargar(almacén, base)

def test_un_catalogo_truncado_se_rechaza(almacén):
    base = _escribir(almacén, [_registro("cma:1"), _registro("cma:2")], mentir={"n_registros": 5})
    with pytest.raises(ErrorDeArtefacto, match="declara 5 registros"):
        CatálogoCanónico.cargar(almacén, base)

def test_con_dos_catalogos_hay_que_decir_cual(almacén):
    _escribir(almacén, [_registro("cma:1")])
    almacén.escribir_json("data/catalogo/cat2/catalogo.json", {"catalogo_id": "cat2"})
    with pytest.raises(ErrorDeArtefacto, match="--catalogo-id"):
        CatálogoCanónico.único(almacén, "data/catalogo")

def test_sin_catalogo_el_error_dice_que_correr(almacén):
    with pytest.raises(ErrorDeArtefacto, match="tp3-ingesta"):
        CatálogoCanónico.único(almacén, "data/catalogo")
