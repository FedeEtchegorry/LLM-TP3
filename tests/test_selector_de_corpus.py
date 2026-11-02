from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.modulo_a.selector_de_corpus import SelectorDeCorpus
from tp3_rag.nucleo.errores import ErrorDeConfiguración

pytestmark = pytest.mark.nivel1

ESTRICTO = {"description_no_vacia": True, "imagen_real": True}

def _registro(aid: str, *, description: str | None = "prosa", imagen: bool = True, year=1800):
    campos = []
    if description is not None:
        campos.append(CampoDeTexto("description", description, "https://x/1", "CC0"))
    campos.append(CampoDeTexto("provenance", "vendido en 1923", "https://x/1", "CC0"))
    return RegistroCanónico(
        artwork_id=aid,
        museum=aid.split(":")[0],
        type="Painting",
        title="t",
        artist="a",
        date_display="1800",
        year=year,
        medium="oil",
        style=None,
        culture_raw=("France",),
        source_url="https://x/1",
        imagen=Imagen("https://x/1.jpg", "sha256:ab") if imagen else None,
        media_license="CC0",
        campos_de_texto=tuple(campos),
        procedencia={},
    )

def test_el_predicado_exige_las_dos_condiciones():
    selector = SelectorDeCorpus(ESTRICTO)
    assert selector.incluye(_registro("cma:1"))
    assert not selector.incluye(_registro("cma:2", description=None))
    assert not selector.incluye(_registro("cma:3", imagen=False))
    assert not selector.incluye(_registro("cma:4", description=None, imagen=False))

def test_una_description_vacia_no_cuenta_como_description():
    assert not SelectorDeCorpus(ESTRICTO).incluye(_registro("cma:5", description="   "))

def test_provenance_sola_no_alcanza():
    assert not SelectorDeCorpus(ESTRICTO).incluye(_registro("cma:6", description=None))

def test_apagar_una_condicion_amplia_el_criterio():
    amplio = SelectorDeCorpus({"description_no_vacia": False, "imagen_real": True})
    assert amplio.incluye(_registro("cma:7", description=None))
    assert not amplio.incluye(_registro("cma:8", imagen=False))

def test_una_condicion_desconocida_es_un_error():
    with pytest.raises(ErrorDeConfiguración, match="descripcion_no_vacia"):
        SelectorDeCorpus({"descripcion_no_vacia": True})

def test_lo_que_produce_es_pre_split_y_lo_dice():
    elegibles = SelectorDeCorpus(ESTRICTO).seleccionar(
        [_registro("cma:1"), _registro("aic:2"), _registro("cma:3", imagen=False)]
    )
    assert len(elegibles.registros) == 2
    assert elegibles.n_universo == 3
    assert elegibles.descartadas_por_condicion == {
        "description_no_vacia": 0,
        "imagen_real": 1,
    }
    assert "pre-split" in elegibles.a_json()["nota"]
    assert "corpus" not in elegibles.a_json()

def test_el_orden_del_catalogo_no_es_el_contenido():
    registros = [_registro("cma:3"), _registro("aic:1"), _registro("cma:2")]
    a = SelectorDeCorpus(ESTRICTO).seleccionar(registros).ids
    b = SelectorDeCorpus(ESTRICTO).seleccionar(list(reversed(registros))).ids
    assert a == b == ("aic:1", "cma:2", "cma:3")
