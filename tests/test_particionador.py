from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.modulo_a.particionador import (
    DefiniciónDeEstrato,
    Particionador,
    Tramo,
    cuota_de_hare,
)
from tp3_rag.modulo_a.selector_de_corpus import SelectorDeCorpus
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConfiguración, ErrorDeConstrucción
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

pytestmark = pytest.mark.nivel1

TRAMOS = (
    Tramo("hasta-1399", None, 1399),
    Tramo("1400-1599", 1400, 1599),
    Tramo("1600-1699", 1600, 1699),
    Tramo("1700-1799", 1700, 1799),
    Tramo("1800-1874", 1800, 1874),
    Tramo("1875-1949", 1875, None),
)
ESTRATO = DefiniciónDeEstrato(
    ejes=("museo", "periodo"),
    campo_de_periodo="year",
    tramos=TRAMOS,
    etiqueta_sin_periodo="sin_periodo",
)

def _registro(aid: str, year: int | None):
    return RegistroCanónico(
        artwork_id=aid,
        museum=aid.split(":")[0],
        type="Painting",
        title="t",
        artist="a",
        date_display="x",
        year=year,
        medium="oil",
        style=None,
        culture_raw=("France",),
        source_url="https://x/1",
        imagen=Imagen("https://x/1.jpg", "sha256:ab"),
        media_license="CC0",
        campos_de_texto=(CampoDeTexto("description", "prosa", "https://x/1", "CC0"),),
        procedencia={},
    )

def _elegibles(registros):
    return SelectorDeCorpus(
        {"description_no_vacia": True, "imagen_real": True}
    ).seleccionar(registros)

def test_la_cuota_de_hare_suma_exactamente_el_objetivo():
    tamaños = {"a": 126, "b": 127, "c": 89, "d": 146, "e": 198, "f": 23, "g": 1,
               "h": 613, "i": 396, "j": 424, "k": 281, "l": 221, "m": 249}
    reparto = cuota_de_hare(tamaños, 0.1)
    assert sum(reparto.values()) == 289 == sum(tamaños.values()) // 10

def test_el_reparto_es_proporcional_dentro_de_un_lugar():
    import math

    tamaños = {"a": 613, "b": 23, "c": 1, "d": 198}
    reparto = cuota_de_hare(tamaños, 0.1)
    for clave, n in tamaños.items():
        assert math.floor(n * 0.1) <= reparto[clave] <= math.floor(n * 0.1) + 1

def test_un_estrato_de_uno_no_se_lleva_su_unico_elemento():
    reparto = cuota_de_hare({"grande": 2893, "sin_periodo": 1}, 0.1)
    assert reparto["sin_periodo"] == 0

def test_el_reparto_no_depende_del_orden_del_diccionario():
    a = cuota_de_hare({"x": 15, "y": 15, "z": 15}, 0.1)
    b = cuota_de_hare({"z": 15, "y": 15, "x": 15}, 0.1)
    assert a == b

def test_una_proporcion_fuera_de_rango_es_un_error():
    with pytest.raises(ErrorDeConfiguración, match=r"\(0, 1\)"):
        cuota_de_hare({"a": 10}, 1.0)

@pytest.mark.parametrize(
    "year,esperado",
    [
        (-1295, "hasta-1399"),
        (1399, "hasta-1399"),
        (1400, "1400-1599"),
        (1600, "1600-1699"),
        (1799, "1700-1799"),
        (1874, "1800-1874"),
        (1875, "1875-1949"),
        (1949, "1875-1949"),
        (None, "sin_periodo"),
    ],
)
def test_el_periodo_cae_en_el_tramo_que_le_toca(year, esperado):
    assert ESTRATO.periodo_de(_registro("cma:1", year)) == esperado

def test_el_estrato_es_el_cruce_de_los_dos_ejes():
    assert ESTRATO.estrato_de(_registro("aic:1", 1650)) == "aic|1600-1699"

def test_un_hueco_entre_tramos_falla_antes_de_partir():
    with pytest.raises(ErrorDeConfiguración, match="hueco"):
        DefiniciónDeEstrato(
            ejes=("museo", "periodo"),
            campo_de_periodo="year",
            tramos=(Tramo("a", None, 1500), Tramo("b", 1600, None)),
            etiqueta_sin_periodo="sin_periodo",
        ).validar()

def test_dos_tramos_solapados_fallan_antes_de_partir():
    with pytest.raises(ErrorDeConfiguración, match="solapan"):
        DefiniciónDeEstrato(
            ejes=("museo", "periodo"),
            campo_de_periodo="year",
            tramos=(Tramo("a", None, 1600), Tramo("b", 1500, None)),
            etiqueta_sin_periodo="sin_periodo",
        ).validar()

def test_los_tramos_tienen_que_cubrir_sin_techo_ni_piso():
    with pytest.raises(ErrorDeConfiguración, match="sin techo ni piso"):
        DefiniciónDeEstrato(
            ejes=("museo", "periodo"),
            campo_de_periodo="year",
            tramos=(Tramo("a", 1400, 1599),),
            etiqueta_sin_periodo="sin_periodo",
        ).validar()

@pytest.fixture
def particionador(cfg, tmp_path):
    return Particionador(cfg, almacén=AlmacenamientoEnArchivos(tmp_path))

@pytest.fixture
def muchos():
    registros = []
    for museo in ("aic", "cma"):
        for i, year in enumerate([1300, 1500, 1650, 1750, 1850, 1900] * 7):
            registros.append(_registro(f"{museo}:{i}", year))
    registros.append(_registro("aic:9999", None))
    return registros

def test_el_split_es_deterministico(particionador, muchos):
    a = particionador.partir(_elegibles(muchos))
    b = particionador.partir(_elegibles(list(reversed(muchos))))
    assert a == b

def test_los_dos_lados_son_una_particion(particionador, muchos):
    elegibles = _elegibles(muchos)
    partición = particionador.partir(elegibles)
    corpus = {i for e in partición.values() for i in e["corpus"]}
    held = {i for e in partición.values() for i in e["held_out"]}
    assert not corpus & held
    assert corpus | held == set(elegibles.ids)

def test_mover_un_id_cambia_el_split_id():
    común = dict(
        criterio_de_inclusion={"description_no_vacia": True, "imagen_real": True},
        semilla=7369,
        proporcion_held_out=0.1,
        definicion_de_estrato=ESTRATO.a_json(),
    )
    a = identidad.split_id(**común, ids_corpus=["cma:1", "cma:2"], ids_held_out=["cma:3"])
    b = identidad.split_id(**común, ids_corpus=["cma:1", "cma:3"], ids_held_out=["cma:2"])
    assert a != b

def test_el_catalogo_id_no_entra_en_el_split_id():
    import inspect

    assert "catalogo_id" not in inspect.signature(identidad.split_id).parameters

def test_congelar_dos_veces_no_reescribe(particionador, muchos, tmp_path):
    elegibles = _elegibles(muchos)
    primero = particionador.congelar(elegibles, catalogo_id="cat1", version_del_normalizador="1")
    assert not primero.ya_estaba_congelado
    antes = primero.ruta.read_bytes()

    segundo = particionador.congelar(elegibles, catalogo_id="cat1", version_del_normalizador="1")
    assert segundo.ya_estaba_congelado
    assert segundo.manifiesto.split_id == primero.manifiesto.split_id
    assert primero.ruta.read_bytes() == antes

def test_derivar_otro_split_exige_decirlo(particionador, muchos):
    particionador.congelar(_elegibles(muchos), catalogo_id="c", version_del_normalizador="1")
    otros = _elegibles(muchos[:30])
    with pytest.raises(ErrorDeConstrucción, match="derivar-otro"):
        particionador.congelar(otros, catalogo_id="c", version_del_normalizador="1")
    particionador.congelar(
        otros, catalogo_id="c", version_del_normalizador="1", derivar_otro=True
    )
    assert len(ManifiestoDeSplit.existentes(particionador.almacén, particionador.raíz_de_splits)) == 2

def test_una_re_ingesta_que_no_cubre_el_corpus_se_detecta_antes_de_construir(
    particionador, muchos
):
    r = particionador.congelar(_elegibles(muchos), catalogo_id="c", version_del_normalizador="1")
    ids = set(r.manifiesto.ids_corpus) | set(r.manifiesto.ids_held_out)
    perdida = sorted(r.manifiesto.ids_corpus)[0]
    assert r.manifiesto.cobertura_contra(ids) == ()
    assert r.manifiesto.cobertura_contra(ids - {perdida}) == (perdida,)

def test_un_manifiesto_con_los_dos_lados_solapados_se_rechaza():
    with pytest.raises(ErrorDeArtefacto, match="los dos lados"):
        ManifiestoDeSplit(
            {
                "split_id": "x",
                "corpus_indexado": {"ids": ["cma:1", "cma:2"]},
                "held_out": {"ids": ["cma:2"]},
                "conteos": {
                    "elegibles_con_texto_e_imagen": 3,
                    "corpus_indexado": 2,
                    "held_out": 1,
                },
            }
        ).validar()

def test_un_conteo_que_no_cierra_contra_sus_ids_se_rechaza():
    with pytest.raises(ErrorDeArtefacto, match="conteos.corpus_indexado"):
        ManifiestoDeSplit(
            {
                "split_id": "x",
                "corpus_indexado": {"ids": ["cma:1", "cma:2"]},
                "held_out": {"ids": ["cma:3"]},
                "conteos": {
                    "elegibles_con_texto_e_imagen": 3,
                    "corpus_indexado": 2605,
                    "held_out": 1,
                },
            }
        ).validar()
