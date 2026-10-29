from __future__ import annotations

from pathlib import Path

import pytest

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.corrida import AlmacenDeCorridas, Corrida
from tp3_rag.modulo_b.armador import ArmadorDePipeline
from tp3_rag.modulo_b.ejecutor import EjecutorDeEtapas
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos, YaExiste
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import (
    ARTWORKS,
    KNOWLEDGE,
    ÍndiceVectorialEnMemoria,
)
from tp3_rag.puertos.llm import FallaDeEtapa, LLMGrabado

pytestmark = pytest.mark.nivel2

OBRAS = [f"cma:1092{i}" for i in range(6)]
RUTA_DE_CORRIDAS = "artifacts/corridas"

def _indice() -> ÍndiceVectorialEnMemoria:
    img, txt = EmbeddingImagenFijo(), EmbeddingTextoFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v-test")
    idx.upsert(
        ARTWORKS,
        ids=OBRAS,
        vectores=[img.embeber_imagen(f"img-{o}".encode()) for o in OBRAS],
        metadatas=[
            {"museum": "cma", "title": f"T{o}", "artist": "A", "source_url": f"https://x/{o}"}
            for o in OBRAS
        ],
    )
    frs, vs, ms, ds = [], [], [], []
    for o in OBRAS:
        texto = f"A landscape, {o}."
        frs.append(f"{o}::description::0")
        vs.append(txt.embeber_uno(texto))
        ds.append(texto)
        ms.append({
            "artwork_id": o, "museum": "cma", "section": "description", "title": f"T{o}",
            "artist": "A", "source_url": f"https://x/{o}", "license": "CC-BY-4.0",
            "chunk_index": 0,
        })
    idx.upsert(KNOWLEDGE, ids=frs, vectores=vs, metadatas=ms, documentos=ds)
    return idx

def _explicacion(top5, *, colgada=False):
    obras = []
    for i, o in enumerate(top5):
        refs = [f"{o}::description::0"]
        afs = [{"texto": "Es un paisaje.", "tipo": "dato", "refs": refs}]
        if colgada and i == 0:
            afs.append({"texto": "Se expuso.", "tipo": "dato", "refs": [f"{o}::exhibitions::0"]})
        obras.append({"artwork_id": o, "afirmaciones": afs})
    return {"obras": obras}

def _armar(tmp_path: Path, *, overrides=None, llm=None, obs=None):
    cfg = Configuración.cargar(
        overrides={"evidencia.orden": "por_seccion_declarada", **(overrides or {})}
    )
    obs = obs or Observabilidad(silencioso=True)
    indice = _indice()
    armador = ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=indice,
        puerto_de_llm=llm if llm is not None else LLMGrabado({}),
        embebedor_de_texto=EmbeddingTextoFijo(),
        observabilidad=obs,
    )
    almacén = AlmacenamientoEnArchivos(tmp_path)
    ejecutor = EjecutorDeEtapas(
        cfg,
        pipeline=armador.armar(),
        almacen_de_corridas=AlmacenDeCorridas(almacén, RUTA_DE_CORRIDAS),
        observabilidad=obs,
        variante_id="v-test",
        split_id="s-test",
    )
    return cfg, ejecutor, obs, almacén

def _entrada(pedido="versiones mas calidas"):
    return EntradaDeConsulta.ad_hoc(datos_de_imagen=b"img-cma:10920", pedido=pedido)

def _top5(tmp_path):
    from tp3_rag.modulo_b.etapa import EstadoDeConsulta
    from tp3_rag.modulo_b.recuperador_visual import RecuperadorVisual

    rv = RecuperadorVisual(
        puerto_de_imagen=EmbeddingImagenFijo(), indice=_indice(), alfa=0.5,
        n_candidatos=6, top_k=5,
    )
    return list(rv.correr(EstadoDeConsulta(entrada=_entrada())).top5)

def test_una_corrida_completa_pasa_et6_y_agrega_a_la_tabla(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    assert corrida.contenido["invariante_et6"]["cumple_et6"] is True
    assert corrida.top5 == top5
    assert corrida.degradaciones == []
    assert corrida.agrega_a_la_tabla is True
    assert len(corrida.salida["obras"]) == 5

def test_la_degradacion_espontanea_saca_la_corrida_de_la_tabla(tmp_path):
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({}, fallas={"generador": FallaDeEtapa("boom", "sin respuesta")})
    )
    corrida = ejecutor.correr(_entrada())
    assert corrida.degradaciones_no_planificadas
    assert corrida.agrega_a_la_tabla is False
    assert corrida.estado["explicacion"] is None
    assert len(corrida.salida["obras"]) == 5

def test_la_falla_inyectada_queda_marcada_planificada(tmp_path):
    _, ejecutor, _, _ = _armar(
        tmp_path, overrides={"consulta.plan_de_falla": {"generador": "excepcion"}}
    )
    corrida = ejecutor.correr(_entrada())
    assert [d["planificada"] for d in corrida.degradaciones] == [True]
    assert corrida.degradaciones_no_planificadas == []
    assert "inyectada" in corrida.degradaciones[0]["causa"]
    assert len(corrida.salida["obras"]) == 5

def test_p7_la_evidencia_inyectada_deja_las_cinco_obras(tmp_path):
    _, ejecutor, _, _ = _armar(
        tmp_path,
        overrides={"consulta.plan_de_falla": {"recolector_de_evidencia": "vacio"}},
    )
    corrida = ejecutor.correr(_entrada())
    assert len(corrida.salida["obras"]) == 5
    assert all(o["abstencion"] for o in corrida.salida["obras"])

def test_el_piso_propaga_en_vez_de_degradar(tmp_path):
    class ÍndiceRoto(ÍndiceVectorialEnMemoria):
        def consultar(self, *a, **k):
            raise ErrorDelProyecto("el índice no está")

    cfg = Configuración.cargar(overrides={"evidencia.orden": "por_seccion_declarada"})
    obs = Observabilidad(silencioso=True)
    armador = ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=ÍndiceRoto(variante_id="v-test"),
        puerto_de_llm=LLMGrabado({}),
        embebedor_de_texto=EmbeddingTextoFijo(),
        observabilidad=obs,
    )
    ejecutor = EjecutorDeEtapas(
        cfg,
        pipeline=armador.armar(),
        almacen_de_corridas=AlmacenDeCorridas(AlmacenamientoEnArchivos(tmp_path), RUTA_DE_CORRIDAS),
        observabilidad=obs,
        variante_id="v-test",
        split_id="s-test",
    )
    with pytest.raises(ErrorDelProyecto, match="el índice no está"):
        ejecutor.correr(_entrada())

def test_sa5_los_descartes_quedan_contados_en_la_corrida(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5, colgada=True)]})
    )
    corrida = ejecutor.correr(_entrada())
    assert len(corrida.afirmaciones_descartadas) == 1
    assert corrida.afirmaciones_descartadas[0]["refs_colgadas"][0].endswith("exhibitions::0")
    assert corrida.fallida is None

def test_el_ordinal_sube_con_cada_corrida_de_la_misma_consulta(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    una = ejecutor.correr(_entrada())
    otra = ejecutor.correr(_entrada())
    assert (una.ordinal, otra.ordinal) == (1, 2)
    assert una.consulta_id == otra.consulta_id
    assert una.corrida_id != otra.corrida_id
    assert una.corrida_id.rsplit("-", 1)[0] == otra.corrida_id.rsplit("-", 1)[0]

def test_el_ordinal_es_legible_y_va_afuera_del_hash():
    cid = identidad.corrida_id(config_id="c", consulta_id="q", commit="abc", ordinal=7)
    assert cid.endswith("-7")
    assert identidad.partir_corrida_id(cid) == (cid.rsplit("-", 1)[0], 7)
    with pytest.raises(Exception, match="ordinal"):
        identidad.corrida_id(config_id="c", consulta_id="q", commit="abc", ordinal=0)

def test_la_reserva_de_directorio_es_exclusiva(tmp_path):
    almacén = AlmacenamientoEnArchivos(tmp_path)
    almacén.reservar_directorio("corridas/abc-1")
    with pytest.raises(YaExiste):
        almacén.reservar_directorio("corridas/abc-1")
    ordinal, _ = almacén.reservar_ordinal("corridas/abc-{ordinal}", directorio=True)
    assert ordinal == 2

def test_la_corrida_se_relee_y_se_valida(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, almacén = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    de_disco = Corrida.leer(almacén, RUTA_DE_CORRIDAS, corrida.corrida_id)
    assert de_disco.contenido == corrida.contenido

def test_et6_se_vuelve_a_verificar_al_leer(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, almacén = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    roto = dict(corrida.contenido)
    roto["estado"] = {**roto["estado"], "claves_pobladas": ["recuperador_visual"]}
    almacén.escribir_json(Corrida.ruta_de(RUTA_DE_CORRIDAS, corrida.corrida_id), roto)
    with pytest.raises(ErrorDeArtefacto, match="no coincide con su cadena"):
        Corrida.leer(almacén, RUTA_DE_CORRIDAS, corrida.corrida_id)

def test_un_corrida_id_que_no_cierra_contra_sus_ingredientes_se_rechaza(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, almacén = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    roto = {**corrida.contenido, "consulta_id": "otra-consulta"}
    almacén.escribir_json(Corrida.ruta_de(RUTA_DE_CORRIDAS, corrida.corrida_id), roto)
    with pytest.raises(ErrorDeArtefacto, match="corrida_id"):
        Corrida.leer(almacén, RUTA_DE_CORRIDAS, corrida.corrida_id)

def test_d12_el_artefacto_no_lleva_la_imagen_ni_el_pedido(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada("un pedido que no se tiene que guardar"))
    serializado = identidad.json_canónico(corrida.contenido)
    assert "un pedido que no se tiene que guardar" not in serializado
    assert corrida.estado["entrada"]["pedido"] is None
    assert corrida.estado["entrada"]["pedido_retenido"] is False
    assert corrida.estado["entrada"]["imagen_retenida"] is False
    assert corrida.estado["entrada"]["imagen_hash"].startswith("sha256:")
    assert corrida.contenido["retencion"]["imagen"] is False

def test_el_libro_mayor_del_artefacto_es_el_de_esa_corrida(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    una = ejecutor.correr(_entrada())
    otra = ejecutor.correr(_entrada())
    assert una.contenido["libro_mayor"]["generador"]["n_llamadas"] == 1
    assert otra.contenido["libro_mayor"]["generador"]["n_llamadas"] == 1

def test_d14_el_arbol_declara_la_politica_y_las_etapas_trazadas(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    árbol = corrida.contenido["arbol_de_trazas"]
    assert set(árbol["etapas_trazadas"]) >= {
        "recuperador_visual", "recolector_de_evidencia", "generador", "renderizador"
    }
    assert árbol["politica"]["procedencia"] == "ad_hoc"
    assert árbol["politica"]["captura_entradas"] is False
    assert árbol["campos_censurados"] > 0
    assert árbol["nodos_con_entradas"] == 0

def test_la_latencia_cubre_las_etapas_puras(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    por_etapa = corrida.contenido["latencia"]["por_etapa"]
    assert "renderizador" in por_etapa
    assert set(por_etapa) == set(corrida.cadena)

def test_los_sellos_llevan_las_cinco_identidades(tmp_path):
    top5 = _top5(tmp_path)
    _, ejecutor, _, _ = _armar(
        tmp_path, llm=LLMGrabado({"generador": [_explicacion(top5)]})
    )
    corrida = ejecutor.correr(_entrada())
    assert set(corrida.contenido["sellos"]) == {
        "corrida_id", "config_id", "variante_id", "split_id", "commit"
    }

def test_un_modo_de_falla_desconocido_no_cae_en_un_default(tmp_path):
    with pytest.raises(ErrorDelProyecto, match="plan_de_falla"):
        _armar(tmp_path, overrides={"consulta.plan_de_falla": {"generador": "inventado"}})
