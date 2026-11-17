"""El índice guarda `year` como texto: el filtro por período y época lo tiene que leer como entero."""
from tp3_rag.modulo_b.recuperador_textual import _año


def test_el_año_en_texto_se_lee_como_entero():
    assert _año({"year": "1885"}) == 1885
    assert _año({"year": 1885}) == 1885


def test_sin_año_o_con_año_ilegible_no_rompe():
    assert _año({}) is None
    assert _año({"year": None}) is None
    assert _año({"year": "c. 1800"}) is None


def test_el_año_en_texto_se_puede_comparar_con_un_rango():
    fichas = {"a": {"year": "1650"}, "b": {"year": "1820"}, "c": {}}
    dentro = {i for i, m in fichas.items() if _año(m) is not None and 1600 <= _año(m) <= 1700}
    assert dentro == {"a"}
