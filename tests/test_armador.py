from __future__ import annotations

from pathlib import Path

import pytest

from tp3_rag.dominio.plan import AlmacenDePlanes
from tp3_rag.modulo_b.armador import (
    HITO_DE_LA_ETAPA,
    OMITIBLES_POR_CONFIGURACIÓN,
    ArmadorDePipeline,
    CadenaImposible,
)
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import ÍndiceVectorialEnMemoria
from tp3_rag.puertos.llm import LLMGrabado

pytestmark = pytest.mark.nivel1

CADENA_C2 = [
    "planificador",
    "recuperador_visual",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
]

CADENA_C3 = [
    "planificador",
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
]

CADENA_C4 = [
    "planificador",
    "recuperador_visual",
    "recuperador_textual",
    "fusionador",
    "reranker",
    "recolector_de_evidencia",
    "generador",
    "renderizador",
]

def _cfg(**overrides) -> Configuración:
    return Configuración.cargar(overrides=overrides)

def _armador(cfg, *, raíz=None) -> ArmadorDePipeline:
    return ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=ÍndiceVectorialEnMemoria(variante_id="v"),
        puerto_de_llm=LLMGrabado({}),
        embebedor_de_texto=EmbeddingTextoFijo(),
        almacen_de_planes=AlmacenDePlanes(
            AlmacenamientoEnArchivos(raíz or Path("/tmp/tp3-planes-de-test")), "planes"
        ),
        bytes_de_la_imagen=lambda artwork_id: f"img-{artwork_id}".encode(),
    )

def test_la_cadena_de_c1_es_valida_y_se_arma():
    pipeline = _armador(_cfg()).armar()
    assert pipeline.configuracion == "C1"
    assert pipeline.cadena == (
        "recuperador_visual", "recolector_de_evidencia", "generador", "renderizador"
    )

def test_la_tabla_de_configuraciones_sale_de_docs_05_y_es_incremental():
    orden = ["C0", "C1", "C2", "C3", "C4"]
    for anterior, siguiente in zip(orden, orden[1:]):
        assert OMITIBLES_POR_CONFIGURACIÓN[anterior] < OMITIBLES_POR_CONFIGURACIÓN[siguiente]

def test_et4_segunda_mitad_una_c3_sin_rama_de_texto_no_pasa_por_c3():
    cfg = _cfg(**{
        "consulta.configuracion": "C3",
        "consulta.etapas": [
            "recuperador_visual", "recolector_de_evidencia", "generador", "renderizador"
        ],
    })
    with pytest.raises(CadenaImposible, match="dice ser \\*\\*C3\\*\\*"):
        ArmadorDePipeline(cfg).validar()

def test_et4_primera_mitad_una_precondicion_que_nadie_produce():
    cfg = _cfg(**{
        "consulta.configuracion": "C1",
        "consulta.etapas": [
            "recuperador_visual", "generador", "recolector_de_evidencia", "renderizador"
        ],
    })
    with pytest.raises(CadenaImposible, match="necesita \\['evidencia'\\]"):
        ArmadorDePipeline(cfg).validar()

def test_las_dos_no_omitibles_son_obligatorias():
    for sin in (
        ["recolector_de_evidencia", "generador", "renderizador"],
        ["recuperador_visual", "recolector_de_evidencia", "generador"],
    ):
        with pytest.raises(CadenaImposible, match="no se omiten nunca"):
            ArmadorDePipeline(_cfg(**{"consulta.etapas": sin})).validar()

def test_una_etapa_desconocida_y_una_repetida_se_rechazan():
    with pytest.raises(CadenaImposible, match="no son etapas"):
        ArmadorDePipeline(
            _cfg(**{"consulta.etapas": ["recuperador_visual", "inventada", "renderizador"]})
        ).validar()
    with pytest.raises(CadenaImposible, match="repite"):
        ArmadorDePipeline(
            _cfg(**{
                "consulta.etapas": [
                    "recuperador_visual", "recolector_de_evidencia",
                    "recolector_de_evidencia", "generador", "renderizador",
                ]
            })
        ).validar()

def test_la_cadena_de_c2_se_arma_con_el_planificador_adelante(tmp_path):
    cfg = _cfg(**{"consulta.configuracion": "C2", "consulta.etapas": CADENA_C2})
    pipeline = _armador(cfg, raíz=tmp_path).armar()
    assert pipeline.configuracion == "C2"
    assert pipeline.cadena == tuple(CADENA_C2)
    assert pipeline.cadena[0] == "planificador"

def test_el_archivo_c2_toml_declara_exactamente_la_fila_c2(tmp_path):
    cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "c2.toml")
    assert cfg["consulta.configuracion"] == "C2"
    pipeline = _armador(cfg, raíz=tmp_path).armar()
    assert "planificador" in pipeline.cadena
    de_c1 = tuple(Configuración.cargar()["consulta.etapas"])
    assert tuple(e for e in pipeline.cadena if e != "planificador") == de_c1

def test_m4_5_las_dos_direcciones_del_detector_de_etiqueta_con_c2():
    sin_planificador = [e for e in CADENA_C2 if e != "planificador"]
    with pytest.raises(CadenaImposible, match=r"dice ser \*\*C2\*\*"):
        ArmadorDePipeline(
            _cfg(**{"consulta.configuracion": "C2", "consulta.etapas": sin_planificador})
        ).validar()
    with pytest.raises(CadenaImposible, match=r"dice ser \*\*C1\*\*"):
        ArmadorDePipeline(
            _cfg(**{"consulta.configuracion": "C1", "consulta.etapas": CADENA_C2})
        ).validar()

def test_la_cadena_de_c3_es_valida_y_se_arma():
    cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "c3.toml")
    pipeline = _armador(cfg).armar()
    assert pipeline.configuracion == "C3"
    assert pipeline.cadena == tuple(CADENA_C3)

def test_la_cadena_de_c4_es_valida_y_se_arma(tmp_path):
    cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "c4.toml")
    pipeline = _armador(cfg, raíz=tmp_path).armar()
    assert pipeline.configuracion == "C4"
    assert pipeline.cadena == tuple(CADENA_C4)
    assert tuple(e for e in pipeline.cadena if e != "reranker") == tuple(CADENA_C3)
    assert pipeline.cadena.index("reranker") < pipeline.cadena.index(
        "recolector_de_evidencia"
    )

def test_la_cadena_de_c0_es_valida_y_son_solo_las_dos_no_omitibles():
    cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "c0.toml")
    pipeline = _armador(cfg).armar()
    assert pipeline.configuracion == "C0"
    assert pipeline.cadena == ("recuperador_visual", "renderizador")
    assert float(cfg["consulta.alfa"]) == 1.0
    assert not {"planificador", "reranker", "generador"} & set(pipeline.cadena)

def test_una_c0_con_generador_se_rechaza_por_la_etiqueta():
    cfg = _cfg(
        **{
            "consulta.configuracion": "C0",
            "consulta.etapas": [
                "recuperador_visual", "recolector_de_evidencia", "generador", "renderizador"
            ],
        }
    )
    with pytest.raises(CadenaImposible, match="[Ss]obran"):
        ArmadorDePipeline(cfg).validar()

def test_una_c3_con_reranker_se_rechaza_porque_c4_menos_c3_mediria_cero():
    with pytest.raises(CadenaImposible, match=r"dice ser \*\*C3\*\*"):
        ArmadorDePipeline(
            _cfg(**{"consulta.configuracion": "C3", "consulta.etapas": CADENA_C4})
        ).validar()
    sin_reranker = [e for e in CADENA_C4 if e != "reranker"]
    with pytest.raises(CadenaImposible, match=r"dice ser \*\*C4\*\*"):
        ArmadorDePipeline(
            _cfg(**{"consulta.configuracion": "C4", "consulta.etapas": sin_reranker})
        ).validar()

def test_ya_no_queda_ninguna_etapa_por_construir_y_el_mecanismo_se_conserva():
    from tp3_rag.modulo_b.armador import CONSTRUIBLES
    from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA

    assert set(CONSTRUIBLES) == set(CLAVE_DE_ETAPA)
    assert HITO_DE_LA_ETAPA == {}

def test_el_reranker_necesita_el_almacen_de_imagenes_y_lo_dice(tmp_path):
    cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "c4.toml")
    armador = ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=ÍndiceVectorialEnMemoria(variante_id="v"),
        puerto_de_llm=LLMGrabado({}),
        embebedor_de_texto=EmbeddingTextoFijo(),
        almacen_de_planes=AlmacenDePlanes(AlmacenamientoEnArchivos(tmp_path), "planes"),
    )
    armador.validar()
    with pytest.raises(CadenaImposible, match="AlmacénDeImágenes"):
        armador.armar()

def test_una_c3_sin_fusionador_se_rechaza_por_la_etiqueta():
    cadena = [e for e in CADENA_C3 if e != "fusionador"]
    cfg = _cfg(**{"consulta.configuracion": "C3", "consulta.etapas": cadena})
    with pytest.raises(CadenaImposible, match="falta"):
        ArmadorDePipeline(cfg).validar()

def test_una_c2_con_rama_de_texto_se_rechaza_por_la_etiqueta():
    cfg = _cfg(
        **{
            "consulta.configuracion": "C2",
            "consulta.etapas": [e for e in CADENA_C3 if e != "fusionador"],
        }
    )
    with pytest.raises(CadenaImposible, match="[Ss]obran"):
        ArmadorDePipeline(cfg).validar()

def test_el_fusionador_sin_rama_de_texto_no_es_una_cadena_imposible():
    cadena = [e for e in CADENA_C3 if e != "recuperador_textual"]
    cfg = _cfg(**{"consulta.configuracion": "C3", "consulta.etapas": cadena})
    with pytest.raises(CadenaImposible, match="falta"):
        ArmadorDePipeline(cfg).validar()
    cfg_sin_etiqueta = _cfg(
        **{"consulta.configuracion": "brazo-una-rama", "consulta.etapas": cadena}
    )
    with pytest.raises(CadenaImposible, match="no es una fila de la tabla"):
        ArmadorDePipeline(cfg_sin_etiqueta).validar()

def test_el_plan_de_falla_rechaza_una_etapa_fuera_de_la_cadena():
    cfg = _cfg(**{"consulta.plan_de_falla": {"reranker": "excepcion"}})
    with pytest.raises(CadenaImposible, match="no están en"):
        ArmadorDePipeline(cfg).validar()

def test_un_plan_de_falla_pasado_como_tabla_entera_da_un_error_legible():
    cfg = _cfg(**{"consulta.plan_de_falla": '{"generador":"excepcion"}'})
    with pytest.raises(CadenaImposible, match="una clave por bandera"):
        ArmadorDePipeline(cfg).validar()
    ArmadorDePipeline(
        _cfg(**{"consulta.plan_de_falla.generador": "excepcion"})
    ).validar()

def test_el_plan_de_falla_rechaza_las_dos_que_no_degradan():
    for etapa in ("recuperador_visual", "renderizador"):
        cfg = _cfg(**{"consulta.plan_de_falla": {etapa: "excepcion"}})
        with pytest.raises(CadenaImposible, match=r"que \*\*no degrada\*\*"):
            ArmadorDePipeline(cfg).validar()

def test_una_configuracion_desconocida_se_rechaza():
    with pytest.raises(CadenaImposible, match="no es una fila de la tabla"):
        ArmadorDePipeline(_cfg(**{"consulta.configuracion": "C9"})).validar()

def test_validar_no_necesita_puertos_y_armar_si():
    ArmadorDePipeline(_cfg()).validar()
    with pytest.raises(CadenaImposible, match="PuertoEmbeddingImagen"):
        ArmadorDePipeline(_cfg()).armar()

def test_el_orden_de_evidencia_no_cae_al_otro_en_silencio():
    cfg = _cfg(**{"evidencia.orden": "por_similitud_al_pedido"})
    armador = ArmadorDePipeline(
        cfg,
        puerto_de_imagen=EmbeddingImagenFijo(),
        indice=ÍndiceVectorialEnMemoria(variante_id="v"),
        puerto_de_llm=LLMGrabado({}),
        embebedor_de_texto=None,
    )
    with pytest.raises(Exception, match="por_seccion_declarada"):
        armador.armar()
