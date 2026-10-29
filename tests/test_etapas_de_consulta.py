from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.modulo_b.etapa import EstadoDeConsulta
from tp3_rag.modulo_b.recolector_de_evidencia import (
    RecolectorDeEvidencia,
    _fragmento_desde_chroma,
)
from tp3_rag.modulo_b.recuperador_visual import RecuperadorVisual, combinar
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import (
    ARTWORKS,
    AUSENTE,
    KNOWLEDGE,
    ÍndiceVectorialEnMemoria,
)

pytestmark = pytest.mark.nivel1

OBRAS = [f"cma:1092{i}" for i in range(6)]

def test_alfa_1_devuelve_el_embedding_de_la_imagen_sin_tocar():
    img = (0.6, 0.8)
    txt = (1.0, 0.0)
    assert combinar(img, txt, 1.0) == img
    assert combinar(img, txt, 0.0) == txt

def test_alfa_intermedio_normaliza_el_resultado():
    v = combinar((1.0, 0.0), (0.0, 1.0), 0.5)
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-9

def test_alfa_fuera_de_rango_es_error():
    with pytest.raises(ErrorDeConfiguración, match=r"\[0, 1\]"):
        combinar((1.0,), (1.0,), 1.5)

def test_dos_espacios_distintos_no_se_pueden_combinar():
    with pytest.raises(ErrorDeConfiguración, match="mismo checkpoint"):
        combinar((1.0, 0.0), (0.0, 1.0, 0.0), 0.5)

def test_la_combinacion_de_norma_0_es_error_y_no_un_nan():
    with pytest.raises(ErrorDeConfiguración, match="norma 0"):
        combinar((1.0, 0.0), (-1.0, 0.0), 0.5)

@pytest.fixture
def indice() -> ÍndiceVectorialEnMemoria:
    img, txt = EmbeddingImagenFijo(), EmbeddingTextoFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v")
    idx.upsert(
        ARTWORKS,
        ids=OBRAS,
        vectores=[img.embeber_imagen(f"img-{o}".encode()) for o in OBRAS],
        metadatas=[{"museum": "cma", "title": f"T{o}", "source_url": f"u/{o}"} for o in OBRAS],
    )
    frs, vs, ms, ds = [], [], [], []
    for o in OBRAS:
        for section, licencia in (("description", "CC-BY-4.0"), ("provenance", "CC0")):
            texto = f"Texto de {section} de {o}."
            frs.append(f"{o}::{section}::0")
            vs.append(txt.embeber_uno(texto))
            ds.append(texto)
            ms.append({
                "artwork_id": o, "museum": "cma", "section": section, "title": f"T{o}",
                "artist": "Un Artista", "source_url": f"u/{o}", "license": licencia,
                "chunk_index": 0,
            })
    idx.upsert(KNOWLEDGE, ids=frs, vectores=vs, metadatas=ms, documentos=ds)
    return idx

@pytest.fixture
def estado() -> EstadoDeConsulta:
    return EstadoDeConsulta(
        entrada=EntradaDeConsulta.ad_hoc(
            datos_de_imagen=b"img-cma:10920", pedido="versiones mas calidas"
        )
    )

def test_el_recuperador_visual_deja_su_clave_y_proyecta_el_top5(indice, estado):
    etapa = RecuperadorVisual(
        puerto_de_imagen=EmbeddingImagenFijo(), indice=indice, alfa=0.5,
        n_candidatos=6, top_k=5,
    )
    nuevo = etapa.correr(estado)
    assert len(nuevo.fuentes_de_ranking["visual"]) == 6
    assert nuevo.top5 == tuple(c.artwork_id for c in nuevo.fuentes_de_ranking["visual"][:5])
    assert [c.rank for c in nuevo.fuentes_de_ranking["visual"]] == [1, 2, 3, 4, 5, 6]

def test_el_piso_no_degrada_y_lo_declara(indice):
    assert RecuperadorVisual.DEGRADA is False
    assert "piso" in RecuperadorVisual.POLÍTICA

def test_un_top_k_mayor_que_n_candidatos_falla_al_construir(indice):
    with pytest.raises(ErrorDeConfiguración, match="top_k"):
        RecuperadorVisual(
            puerto_de_imagen=EmbeddingImagenFijo(), indice=indice, alfa=0.5,
            n_candidatos=3, top_k=5,
        )

def test_la_pieza_no_tiene_default_de_alfa():
    import inspect

    alfa = inspect.signature(RecuperadorVisual.__init__).parameters["alfa"]
    assert alfa.default is inspect.Parameter.empty

def _recolector(indice, orden, **extra):
    return RecolectorDeEvidencia(
        indice=indice,
        n_fragmentos_por_obra=2,
        orden=orden,
        campos_de_knowledge=("description", "did_you_know", "provenance"),
        **extra,
    )

def test_ev2_solo_pasan_fragmentos_del_top5(indice, estado):
    con_top = estado.proyectar_top5(OBRAS[:2], por="recuperador_visual")
    nuevo = _recolector(indice, "por_seccion_declarada").correr(con_top)
    assert set(nuevo.evidencia) == set(OBRAS[:2])
    for obra, fragmentos in nuevo.evidencia.items():
        assert all(f.artwork_id == obra for f in fragmentos)

def test_el_orden_por_seccion_es_puro_determinista_y_sin_red(indice, estado):
    con_top = estado.proyectar_top5([OBRAS[0]], por="recuperador_visual")
    recolector = _recolector(indice, "por_seccion_declarada")
    uno = recolector.correr(con_top).evidencia[OBRAS[0]]
    dos = recolector.correr(con_top).evidencia[OBRAS[0]]
    assert [f.section for f in uno] == ["description", "provenance"]
    assert uno == dos

def test_el_orden_por_similitud_exige_el_puerto_y_no_cae_al_otro(indice):
    with pytest.raises(ErrorDeConfiguración, match="por_seccion_declarada"):
        _recolector(indice, "por_similitud_al_pedido")
    _recolector(indice, "por_similitud_al_pedido", embebedor_de_texto=EmbeddingTextoFijo())

def test_ev1_la_proyeccion_es_la_identidad_y_un_campo_faltante_es_error():
    completa = {
        "artwork_id": "cma:1", "museum": "cma", "section": "description",
        "chunk_index": 0, "source_url": "u", "license": "CC0",
        "title": "T", "artist": "A",
    }
    fragmento = _fragmento_desde_chroma("cma:1::description::0", "texto", completa)
    assert fragmento.license == "CC0" and fragmento.source_url == "u"
    for falta in ("license", "source_url", "artwork_id"):
        parcial = {k: v for k, v in completa.items() if k != falta}
        with pytest.raises(ErrorDelProyecto, match="la metadata de"):
            _fragmento_desde_chroma("cma:1::description::0", "texto", parcial)

def test_el_vacio_de_chroma_se_deshace_en_ausencia():
    metadata = {
        "artwork_id": "cma:1", "museum": "cma", "section": "description",
        "chunk_index": 0, "source_url": "u", "license": "CC0",
        "title": AUSENTE, "artist": AUSENTE,
    }
    fragmento = _fragmento_desde_chroma("cma:1::description::0", "t", metadata)
    assert fragmento.title is None and fragmento.artist is None

def test_ev3_una_obra_con_0_fragmentos_es_un_evento_registrado(indice, estado):
    con_top = estado.proyectar_top5(["cma:no-existe"], por="recuperador_visual")
    nuevo = _recolector(indice, "por_seccion_declarada").correr(con_top)
    assert nuevo.evidencia["cma:no-existe"] == ()
    assert len(nuevo.degradaciones) == 1
    assert "0 fragmentos" in nuevo.degradaciones[0].causa
    assert "trazabilidad" in nuevo.degradaciones[0].causa

def test_n_fragmentos_por_obra_tiene_que_ser_al_menos_1(indice):
    with pytest.raises(ErrorDeConfiguración, match="≥ 1"):
        RecolectorDeEvidencia(
            indice=indice, n_fragmentos_por_obra=0, orden="por_seccion_declarada",
            campos_de_knowledge=("description",),
        )

def test_el_fragmento_que_vuelve_trae_su_texto_para_que_sa3_pueda_comparar(indice, estado):
    con_top = estado.proyectar_top5([OBRAS[0]], por="recuperador_visual")
    fragmentos = _recolector(indice, "por_seccion_declarada").correr(con_top).evidencia[OBRAS[0]]
    assert all(f.texto for f in fragmentos)
    assert fragmentos[0].texto == f"Texto de description de {OBRAS[0]}."
