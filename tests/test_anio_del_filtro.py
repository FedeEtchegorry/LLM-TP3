"""El filtro por período y época no se cae con una obra sin año.

El índice guarda `year` como entero y deja «» en la obra que no tiene año. Comparar ese texto
contra el rango tiraba TypeError y la consulta seguía solo con la rama visual.
"""
from types import SimpleNamespace

from tp3_rag.modulo_b.recuperador_textual import RecuperadorTextual


class _Indice:
    def __init__(self, fichas):
        self.fichas = fichas

    def ids(self, _coleccion):
        return list(self.fichas)

    def metadatas(self, _coleccion, ids):
        return {i: self.fichas[i] for i in ids}

    def traer(self, _coleccion, where=None):
        (campo, condicion), = where.items()
        return [SimpleNamespace(id=i, metadata=m) for i, m in self.fichas.items() if m.get(campo) in condicion["$in"]]


def _recuperador(fichas):
    r = RecuperadorTextual.__new__(RecuperadorTextual)
    r.indice, r._fichas, r.filtro_por_metadata = _Indice(fichas), None, True
    return r


def _fichas():
    fichas = {f"s17:{i}": {"culture": "Italy", "year": 1650, "medium": "oil on canvas"} for i in range(25)}
    fichas.update({f"s19:{i}": {"culture": "France", "year": 1850, "medium": "oil on canvas"} for i in range(25)})
    fichas["sin:1"] = {"culture": "Italy", "year": "", "medium": "oil on canvas"}
    return fichas


def test_el_siglo_filtra_y_deja_afuera_la_obra_sin_año():
    where, registro = _recuperador(_fichas())._filtro("retratos del siglo XVII")
    assert registro["aplicado"] is True
    assert set(where["artwork_id"]["$in"]) == {f"s17:{i}" for i in range(25)}


def test_un_pedido_sin_ejes_no_filtra():
    where, registro = _recuperador(_fichas())._filtro("más tranquilo")
    assert where is None and registro["aplicado"] is False
