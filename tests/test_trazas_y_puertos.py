from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import PolíticaDeTrazado
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.nucleo.observabilidad import (
    TIPOS_DE_NODO,
    Observabilidad,
    PuertoTrazasNulo,
)
from tp3_rag.puertos import trazas as trazas_mod
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import KNOWLEDGE, ÍndiceVectorialEnMemoria

pytestmark = pytest.mark.nivel1

def test_ob3_el_nulo_guarda_los_nodos_ademas_de_los_spans():
    nulo = PuertoTrazasNulo()
    for operación in ("span", "nodo", "cerrar"):
        assert hasattr(nulo, operación)
    real = trazas_mod.TrazasLangSmith.__dict__
    assert {"span", "nodo", "cerrar"} <= set(real)

def test_el_arbol_tiene_tres_niveles_y_ninguno_lo_emite_una_etapa():
    obs = Observabilidad(silencioso=True)
    puerto = obs.instrumentar(EmbeddingTextoFijo(), rol="embedding_texto")
    with obs.nodo("corrida C1", tipo="chain"):
        with obs.nodo("recolector_de_evidencia", tipo="chain"):
            puerto.embeber_uno("un texto")
    nodos = {n.nombre: n for n in obs.nodos()}
    raíz = nodos["corrida C1"]
    etapa = nodos["recolector_de_evidencia"]
    hoja = nodos["embedding_texto.embeber_uno"]
    assert etapa.padre == raíz.id
    assert hoja.padre == etapa.id
    assert hoja.trace_id == etapa.trace_id == raíz.trace_id
    assert hoja.dotted_order.startswith(etapa.dotted_order)
    assert hoja.tipo == "embedding"

def test_un_tipo_de_nodo_desconocido_se_rechaza():
    obs = Observabilidad(silencioso=True)
    assert "chain" in TIPOS_DE_NODO and "llm" in TIPOS_DE_NODO
    with pytest.raises(ValueError, match="tipo de nodo"):
        with obs.nodo("x", tipo="inventado"):
            pass

def test_el_error_de_la_etapa_queda_en_el_nodo_y_se_propaga():
    obs = Observabilidad(silencioso=True)
    with pytest.raises(RuntimeError):
        with obs.nodo("generador", tipo="chain"):
            raise RuntimeError("boom")
    assert obs.nodos()[0].error.startswith("RuntimeError")

def test_la_marca_acota_el_libro_mayor_a_una_corrida():
    obs = Observabilidad(silencioso=True)
    puerto = obs.instrumentar(EmbeddingTextoFijo(), rol="embedding_texto")
    puerto.embeber_uno("primera corrida")
    marca = obs.marca()
    puerto.embeber_uno("segunda corrida")
    assert obs.libro_mayor()["embedding_texto"]["n_llamadas"] == 2
    assert obs.libro_mayor(marca)["embedding_texto"]["n_llamadas"] == 1

def test_el_proyecto_de_trazas_se_elige_por_configuracion_y_nulo_es_legitimo():
    cfg = Configuración.cargar(overrides={"trazas.proveedor": "nulo"})
    assert isinstance(trazas_mod.armar(cfg, Credenciales()), PuertoTrazasNulo)
    with pytest.raises(ErrorDeConfiguración, match="no es uno de"):
        trazas_mod.armar(
            Configuración.cargar(overrides={"trazas.proveedor": "wandb"}), Credenciales()
        )

def test_langsmith_sin_key_no_degrada_en_silencio():
    cfg = Configuración.cargar()
    with pytest.raises(ErrorDeConfiguración, match="LANGSMITH_API_KEY"):
        trazas_mod.armar(cfg, Credenciales(langsmith_api_key=None))

def test_el_mensaje_de_la_key_de_trazas_nombra_la_salida():
    with pytest.raises(ErrorDeConfiguración) as e:
        Credenciales().exigir_langsmith()
    assert "nulo" in str(e.value) and "artefacto de corrida" in str(e.value)

def test_la_politica_se_aplica_antes_del_puerto():
    nulo = PuertoTrazasNulo()
    obs = Observabilidad(trazas=nulo, silencioso=True)
    with obs.con_politica_de_trazado(PolíticaDeTrazado.de("ad_hoc")):
        with obs.nodo("corrida", tipo="chain", entradas={"pedido": "el secreto"}) as n:
            n.salidas["render"] = "la salida"
    recibido = nulo.nodos[0]
    assert recibido.entradas == {} and recibido.salidas == {}
    assert recibido.campos_censurados == 2
    assert obs.eventos["trazas.campos_censurados"] == 2

def test_traer_filtra_por_metadata_sin_vector_y_devuelve_score_none():
    txt = EmbeddingTextoFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v")
    idx.upsert(
        KNOWLEDGE,
        ids=["cma:1::description::0", "cma:2::description::0"],
        vectores=[txt.embeber_uno("a"), txt.embeber_uno("b")],
        metadatas=[{"artwork_id": "cma:1"}, {"artwork_id": "cma:2"}],
        documentos=["texto de 1", "texto de 2"],
    )
    salida = idx.traer(KNOWLEDGE, where={"artwork_id": "cma:1"})
    assert len(salida) == 1
    assert salida[0].score is None
    assert salida[0].documento == "texto de 1"

def test_consultar_devuelve_el_documento_porque_r13_lo_necesita():
    txt = EmbeddingTextoFijo()
    idx = ÍndiceVectorialEnMemoria(variante_id="v")
    idx.upsert(
        KNOWLEDGE,
        ids=["cma:1::description::0"],
        vectores=[txt.embeber_uno("a")],
        metadatas=[{"artwork_id": "cma:1"}],
        documentos=["el texto literal del museo"],
    )
    resultado = idx.consultar(KNOWLEDGE, txt.embeber_uno("a"), 1)[0]
    assert resultado.documento == "el texto literal del museo"
    assert resultado.score is not None

def test_traer_esta_en_las_operaciones_declaradas():
    assert "traer" in ÍndiceVectorialEnMemoria.OPERACIONES

def test_reservar_directorio_es_el_mismo_primitivo_sobre_otro_tipo(tmp_path):
    almacén = AlmacenamientoEnArchivos(tmp_path)
    directorio = almacén.reservar_directorio("corridas/abc-1")
    assert directorio.is_dir()
    archivo = almacén.reservar("otros/x-1")
    assert archivo.is_file()
    assert almacén.reservar_ordinal("corridas/abc-{ordinal}", directorio=True)[0] == 2
    assert almacén.reservar_ordinal("otros/x-{ordinal}")[0] == 2
