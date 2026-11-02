from __future__ import annotations

import pytest

from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos.indice_vectorial import (
    ARTWORKS,
    AUSENTE,
    KNOWLEDGE,
    ColecciónAusente,
    ÍndiceVectorialChroma,
    ÍndiceVectorialEnMemoria,
    aplanar,
    aplanar_metadata,
)

VID = "e1175c30479c2f07"

def _unitario(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v

@pytest.mark.nivel1
def test_r7_una_lista_se_aplana_a_escalar_sin_perder_elementos():
    assert aplanar(["France", "Netherlands"]) == "France | Netherlands"
    assert aplanar(["France"]) == "France"

@pytest.mark.nivel1
def test_r7_la_ausencia_se_materializa_como_vacio_y_nace_aca():
    assert aplanar(None) == AUSENTE
    assert aplanar([]) == AUSENTE
    assert aplanar(()) == AUSENTE
    assert aplanar_metadata({"style": None, "year": 1865}) == {"style": "", "year": 1865}

@pytest.mark.nivel1
def test_r7_los_escalares_que_chroma_acepta_pasan_tal_cual():
    plana = aplanar_metadata({"year": 1865, "museum": "cma", "ok": True, "x": 1.5})
    assert plana == {"year": 1865, "museum": "cma", "ok": True, "x": 1.5}
    assert isinstance(plana["year"], int)

@pytest.fixture(params=["memoria", "chroma"])
def indice(request, tmp_path):
    if request.param == "memoria":
        return ÍndiceVectorialEnMemoria(variante_id=VID)
    return ÍndiceVectorialChroma(tmp_path / "chroma", variante_id=VID)

@pytest.mark.nivel2
def test_una_coleccion_ausente_es_error_duro_y_nombra_la_variante(indice):
    with pytest.raises(ColecciónAusente, match=VID):
        indice.consultar(ARTWORKS, _unitario(0), 5)

@pytest.mark.nivel2
def test_una_coleccion_vacia_tambien_es_error_duro(indice):
    indice.crear_coleccion(ARTWORKS, VID)
    with pytest.raises(ColecciónAusente, match="vacía"):
        indice.consultar(ARTWORKS, _unitario(0), 5)

@pytest.mark.nivel2
def test_no_se_puede_construir_un_indice_sin_variante_id():
    with pytest.raises(ErrorDeConstrucción, match="variante_id"):
        ÍndiceVectorialEnMemoria(variante_id="")

@pytest.mark.nivel2
def test_una_coleccion_que_no_es_del_alcance_aprobado_es_error(indice):
    with pytest.raises(ErrorDeConstrucción, match="desconocida"):
        indice.crear_coleccion("artwork", VID)

@pytest.mark.nivel2
def test_upsert_y_consulta_devuelven_score_donde_mayor_es_mejor(indice):
    indice.crear_coleccion(ARTWORKS, VID)
    indice.upsert(
        ARTWORKS,
        ids=["cma:1", "cma:2"],
        vectores=[_unitario(0), _unitario(1)],
        metadatas=[{"museum": "cma"}, {"museum": "cma"}],
    )
    resultados = indice.consultar(ARTWORKS, _unitario(0), 2)
    assert [r.id for r in resultados] == ["cma:1", "cma:2"]
    assert resultados[0].score > resultados[1].score
    assert resultados[0].score == pytest.approx(1.0, abs=1e-6)

@pytest.mark.nivel2
def test_el_filtro_de_metadata_es_lo_que_la_etapa_5_necesita(indice):
    indice.crear_coleccion(KNOWLEDGE, VID)
    indice.upsert(
        KNOWLEDGE,
        ids=["cma:1::description::0", "cma:9::description::0"],
        vectores=[_unitario(0), _unitario(0)],
        metadatas=[{"artwork_id": "cma:1"}, {"artwork_id": "cma:9"}],
        documentos=["uno", "nueve"],
    )
    resultados = indice.consultar(
        KNOWLEDGE, _unitario(0), 10, where={"artwork_id": {"$in": ["cma:1"]}}
    )
    assert [r.id for r in resultados] == ["cma:1::description::0"]

@pytest.mark.nivel2
def test_el_upsert_desalineado_falla_antes_de_escribir(indice):
    indice.crear_coleccion(ARTWORKS, VID)
    with pytest.raises(ErrorDeConstrucción, match="desalineado"):
        indice.upsert(ARTWORKS, ids=["a", "b"], vectores=[_unitario(0)], metadatas=[{}])

@pytest.mark.nivel2
def test_los_ids_materializados_son_lo_que_el_invariante_2_compara(indice):
    indice.crear_coleccion(ARTWORKS, VID)
    indice.upsert(
        ARTWORKS,
        ids=["cma:2", "cma:1"],
        vectores=[_unitario(0), _unitario(1)],
        metadatas=[{}, {}],
    )
    assert set(indice.ids(ARTWORKS)) == {"cma:1", "cma:2"}

@pytest.mark.nivel2
def test_existir_y_estar_vacia_son_preguntas_distintas(indice):
    assert not indice.existe(ARTWORKS)
    indice.crear_coleccion(ARTWORKS, VID)
    assert indice.existe(ARTWORKS)
    assert indice.contar(ARTWORKS) == 0

@pytest.mark.nivel1
def test_el_doble_en_memoria_levanta_ante_un_operador_que_no_soporta():
    indice = ÍndiceVectorialEnMemoria(variante_id=VID)
    indice.crear_coleccion(KNOWLEDGE, VID)
    indice.upsert(
        KNOWLEDGE, ids=["a"], vectores=[_unitario(0)], metadatas=[{"year": 1865}]
    )
    with pytest.raises(ErrorDeConstrucción, match="operador"):
        indice.consultar(KNOWLEDGE, _unitario(0), 1, where={"year": {"$gt": 1800}})

@pytest.mark.nivel1
def test_consultar_con_la_dimension_del_otro_espacio_falla_nombrando_los_dos():
    indice = ÍndiceVectorialEnMemoria(variante_id=VID)
    indice.crear_coleccion(ARTWORKS, VID)
    indice.upsert(ARTWORKS, ids=["a"], vectores=[[1.0, 0.0]], metadatas=[{}])
    with pytest.raises(ErrorDeConstrucción, match="512"):
        indice.consultar(ARTWORKS, [1.0, 0.0, 0.0], 1)

@pytest.mark.nivel2
def test_chroma_guarda_la_variante_en_la_coleccion_y_la_verifica_al_abrir(tmp_path):
    raíz = tmp_path / "chroma"
    indice = ÍndiceVectorialChroma(raíz, variante_id=VID)
    indice.crear_coleccion(ARTWORKS, VID)
    indice.upsert(ARTWORKS, ids=["a"], vectores=[_unitario(0)], metadatas=[{}])
    indice.cerrar()

    assert (raíz / VID).is_dir()
    otro = ÍndiceVectorialChroma(raíz, variante_id="0000000000000000")
    with pytest.raises(ColecciónAusente):
        otro.consultar(ARTWORKS, _unitario(0), 1)

@pytest.mark.nivel2
def test_crear_con_otra_variante_sobre_un_indice_abierto_es_error(tmp_path):
    indice = ÍndiceVectorialChroma(tmp_path / "chroma", variante_id=VID)
    with pytest.raises(ErrorDeConstrucción, match="no puede discrepar"):
        indice.crear_coleccion(ARTWORKS, "0000000000000000")

@pytest.mark.nivel2
def test_el_upsert_se_batchea_por_debajo_del_tope_de_chroma(tmp_path):
    indice = ÍndiceVectorialChroma(
        tmp_path / "chroma", variante_id=VID, batch_de_upsert=7
    )
    indice.crear_coleccion(KNOWLEDGE, VID)
    n = indice.upsert(
        KNOWLEDGE,
        ids=[f"cma:{i}::description::0" for i in range(20)],
        vectores=[_unitario(i) for i in range(20)],
        metadatas=[{"artwork_id": f"cma:{i}"} for i in range(20)],
        documentos=[f"texto {i}" for i in range(20)],
    )
    assert n == 20
    assert indice.contar(KNOWLEDGE) == 20
