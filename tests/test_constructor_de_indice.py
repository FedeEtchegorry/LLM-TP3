from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.dominio.variante import INGREDIENTES_DEL_ID, RegistroDeÍndices
from tp3_rag.modulo_a.constructor_de_indice import (
    CAMPOS_DE_ARTWORKS,
    ConstructorDeÍndice,
    derivar_variante_id,
)
from tp3_rag.modulo_a.fragmentador import Fragmentador
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConstrucción
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.embedding_imagen import IdentidadDelModelo
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE, ÍndiceVectorialEnMemoria

pytestmark = pytest.mark.nivel2

DIM = 512

def _registro(aid: str, *, description="Corot painted this view.", provenance="Sold in 1923.",
              imagen=True) -> RegistroCanónico:
    campos = []
    if description is not None:
        campos.append(CampoDeTexto("description", description, f"https://ficha/{aid}", "CC0"))
    if provenance is not None:
        campos.append(CampoDeTexto("provenance", provenance, f"https://ficha/{aid}", "CC0"))
    return RegistroCanónico(
        artwork_id=aid,
        museum=aid.split(":")[0],
        type="Painting",
        title=f"obra {aid}",
        artist="Corot",
        date_display="1865",
        year=1865,
        medium="oil",
        style=None,
        culture_raw=("France, 19th century", "France"),
        source_url=f"https://ficha/{aid}",
        imagen=Imagen(f"https://x/{aid}.jpg", f"sha256:{aid}") if imagen else None,
        media_license="CC0",
        campos_de_texto=tuple(campos),
        procedencia={},
    )

IDS_CORPUS = ("cma:1", "cma:2", "aic:3", "aic:4")
IDS_HELD_OUT = ("cma:5", "aic:6")

@pytest.fixture
def cfg() -> Configuración:
    return Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, provenance]",
        }
    )

@pytest.fixture
def escenario(cfg, tmp_path):
    import numpy as np

    almacén = AlmacenamientoEnArchivos(tmp_path)
    registros = [_registro(i) for i in IDS_CORPUS + IDS_HELD_OUT]

    from tp3_rag.modulo_a.embebedor_de_imagenes import CachéDeEmbeddingsDeImagen

    modelo = IdentidadDelModelo(cfg["embebido.checkpoint"], "1", DIM)
    caché = CachéDeEmbeddingsDeImagen(almacén.ruta(cfg["rutas.embeddings_imagen"]), modelo)
    for n, registro in enumerate(registros):
        vector = np.zeros(DIM, dtype=np.float32)
        vector[n] = 1.0
        caché.escribir(registro.artwork_id, registro.imagen.hash_bytes, vector)

    manifiesto = _manifiesto(cfg, almacén, IDS_CORPUS, IDS_HELD_OUT)
    return almacén, registros, manifiesto

def _manifiesto(cfg, almacén, corpus, held_out) -> ManifiestoDeSplit:
    from tp3_rag.nucleo import identidad

    criterio = dict(cfg["ingesta.criterio_de_inclusion"])
    sid = identidad.split_id(
        criterio_de_inclusion=criterio,
        semilla=7369,
        proporcion_held_out=0.1,
        definicion_de_estrato={"ejes": ["museo"]},
        ids_corpus=corpus,
        ids_held_out=held_out,
    )
    contenido = {
        "artefacto": "ManifiestoDeSplit",
        "split_id": sid,
        "criterio_de_inclusion": criterio,
        "semilla": 7369,
        "proporcion_held_out": 0.1,
        "definicion_de_estrato": {"ejes": ["museo"]},
        "conteos": {
            "corpus_indexado": len(corpus),
            "held_out": len(held_out),
            "elegibles_con_texto_e_imagen": len(corpus) + len(held_out),
        },
        "corpus_indexado": {"ids": list(corpus)},
        "held_out": {"ids": list(held_out)},
        "procedencia": {"catalogo_id": "cat0000000000000"},
    }
    almacén.escribir_json(
        ManifiestoDeSplit.ruta_de(cfg["rutas.artefactos_splits"], sid), contenido
    )
    return ManifiestoDeSplit(contenido)

def _constructor(cfg, almacén, manifiesto, *, con_texto=True, fragmentador=None):
    fragmentador = fragmentador or Fragmentador(cfg)
    vid = derivar_variante_id(
        cfg,
        catalogo_id="cat0000000000000",
        split_id=manifiesto.split_id,
        fragmentador=fragmentador,
    )
    indice = ÍndiceVectorialEnMemoria(variante_id=vid)
    return ConstructorDeÍndice(
        cfg,
        almacén=almacén,
        indice=indice,
        manifiesto=manifiesto,
        catalogo_id="cat0000000000000",
        embebedor_de_texto=EmbeddingTextoFijo(dim=8) if con_texto else None,
        fragmentador=fragmentador,
    )

def _construir_todo(constructor, registros):
    corpus = constructor.verificar_cobertura(registros)
    constructor.construir_artworks(corpus)
    por_obra = constructor.fragmentar(corpus)
    constructor.construir_knowledge(por_obra)
    return por_obra

def test_la_variante_se_registra_y_publica_las_dos_cifras_de_a6(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    resultado = constructor.registrar(
        por_obra=por_obra, cita_al_embebido=constructor.cita_al_embebido()
    )

    assert resultado.registrada
    assert resultado.n_indexadas == 4
    inv = resultado.registro.a_json()["invariantes_de_9_6"]
    assert inv["2_igualdad_de_conjuntos_y_a6"]["igualdad_de_conjuntos"] is True
    assert inv["3_toda_obra_indexada_tiene_fragmento"]["min_fragmentos_por_obra"] >= 1

def test_n_embebidas_se_cita_y_no_se_copia(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    resultado = constructor.registrar(
        por_obra=por_obra, cita_al_embebido=constructor.cita_al_embebido()
    )
    assert "n_embebidas" not in resultado.registro.conteos
    assert set(resultado.registro.conteos) >= {"n_indexadas", "n_fragmentos"}

def test_el_registro_rechaza_un_n_embebidas_metido_en_sus_conteos():
    registro = RegistroDeÍndices(
        {
            "variante_id": "x",
            "receta": {k: "v" for k in INGREDIENTES_DEL_ID},
            "conteos": {"n_indexadas": 1, "n_fragmentos": 1, "n_embebidas": 9},
            "colecciones": {},
            "procedencia": {},
            "invariantes_de_9_6": {},
        }
    )
    with pytest.raises(ErrorDeArtefacto, match="n_embebidas"):
        registro.validar()

def test_el_constructor_no_recibe_el_puerto_de_imagen(cfg, escenario):
    import inspect

    firma = inspect.signature(ConstructorDeÍndice.__init__)
    assert "embebedor_de_texto" in firma.parameters
    assert not [p for p in firma.parameters if "imagen" in p]

def test_un_vector_ausente_en_el_cache_es_error_y_no_un_re_embebido(cfg, escenario, tmp_path):
    almacén, registros, manifiesto = escenario
    for ruta in (tmp_path / "data" / "embeddings-imagen").rglob("*cma_1*.npy"):
        ruta.unlink()
    constructor = _constructor(cfg, almacén, manifiesto)
    with pytest.raises(ErrorDeConstrucción, match="tp3-embeber"):
        constructor.construir_artworks(constructor.verificar_cobertura(registros))

def test_d8_un_vector_cacheado_que_no_es_unitario_es_error_duro(cfg, escenario, tmp_path):
    import numpy as np

    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    ruta = constructor.caché.ruta("cma:1", "sha256:cma:1")
    np.save(ruta.open("wb"), np.full(DIM, 0.5, dtype=np.float32))
    with pytest.raises(ErrorDeConstrucción, match="no 1|norma"):
        constructor.construir_artworks(constructor.verificar_cobertura(registros))

def test_la_metadata_de_artworks_no_lleva_media_license(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    constructor.construir_artworks(constructor.verificar_cobertura(registros))
    metadata = constructor.indice.consultar(ARTWORKS, [1.0] + [0.0] * (DIM - 1), 1)[0].metadata
    assert "media_license" not in metadata
    assert "media_license" not in CAMPOS_DE_ARTWORKS
    assert "in_eval_holdout" not in metadata

def test_la_culture_normalizada_se_deriva_aca_y_se_aplana_en_el_puerto(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    constructor.construir_artworks(constructor.verificar_cobertura(registros))
    metadata = constructor.indice.consultar(ARTWORKS, [1.0] + [0.0] * (DIM - 1), 1)[0].metadata
    assert metadata["culture"] == "France"
    assert isinstance(metadata["culture"], str)
    assert metadata["style"] == ""

def test_invariante_1_una_obra_del_held_out_indexada_aborta_y_no_registra(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)

    constructor.indice.upsert(
        ARTWORKS, ids=["cma:5"], vectores=[[1.0] + [0.0] * (DIM - 1)], metadatas=[{"m": "x"}]
    )
    with pytest.raises(ErrorDeConstrucción, match="invariante 1"):
        constructor.registrar(por_obra=por_obra, cita_al_embebido={})
    assert RegistroDeÍndices.existentes(almacén, cfg["rutas.artefactos_indices"]) == ()

def test_invariante_2_indexar_el_universo_en_vez_del_corpus_se_atrapa(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    constructor.indice.upsert(
        ARTWORKS, ids=["cma:70", "aic:71"], vectores=[[1.0] + [0.0] * (DIM - 1)] * 2,
        metadatas=[{"m": "x"}] * 2,
    )
    with pytest.raises(ErrorDeConstrucción, match="sobran 2"):
        constructor.registrar(por_obra=por_obra, cita_al_embebido={})

def test_invariante_2_dos_errores_que_se_compensan_dan_el_mismo_numero(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    constructor.indice._datos[ARTWORKS].pop("cma:1")
    constructor.indice.upsert(
        ARTWORKS, ids=["cma:5"], vectores=[[1.0] + [0.0] * (DIM - 1)], metadatas=[{"m": "x"}]
    )
    assert len(constructor.indice.ids(ARTWORKS)) == len(manifiesto.ids_corpus)
    with pytest.raises(ErrorDeConstrucción, match="invariante"):
        constructor.registrar(por_obra=por_obra, cita_al_embebido={})

def test_la_cobertura_se_chequea_antes_de_tocar_chroma(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    sin_una = [r for r in registros if r.artwork_id != "aic:3"]
    with pytest.raises(ErrorDeConstrucción, match="aic:3"):
        constructor.verificar_cobertura(sin_una)
    assert not constructor.indice.existe(ARTWORKS)

def test_una_obra_del_corpus_sin_imagen_no_se_puede_materializar(cfg, tmp_path):
    cfg = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "ingesta.criterio_de_inclusion.imagen_real": "false",
        }
    )
    almacén = AlmacenamientoEnArchivos(tmp_path)
    registros = [_registro("cma:1", imagen=False)]
    manifiesto = _manifiesto(cfg, almacén, ("cma:1",), ())
    constructor = _constructor(cfg, almacén, manifiesto)
    with pytest.raises(ErrorDeConstrucción, match="insatisfacible"):
        constructor.construir_artworks(constructor.verificar_cobertura(registros))

def test_invariante_3_un_conjunto_de_campos_sin_description_se_rechaza_al_registrar(
    cfg, escenario, tmp_path
):
    almacén, _, manifiesto = escenario
    registros = [
        _registro(i, provenance=None) for i in IDS_CORPUS
    ] + [_registro(i) for i in IDS_HELD_OUT]
    cfg_sin = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[provenance]",
        }
    )
    constructor = _constructor(cfg_sin, almacén, manifiesto)
    corpus = constructor.verificar_cobertura(registros)
    constructor.construir_artworks(corpus)
    por_obra = constructor.fragmentar(corpus)
    assert all(not v for v in por_obra.values())
    with pytest.raises(ErrorDeConstrucción, match="invariante 3"):
        constructor.registrar(por_obra=por_obra, cita_al_embebido={})
    assert RegistroDeÍndices.existentes(almacén, cfg["rutas.artefactos_indices"]) == ()

def test_invariante_3_es_el_detector_de_r4(cfg, escenario):
    almacén, _, manifiesto = escenario
    registros = [_registro("cma:1", description=None, provenance=None)] + [
        _registro(i) for i in IDS_CORPUS[1:] + IDS_HELD_OUT
    ]
    constructor = _constructor(cfg, almacén, manifiesto)
    corpus = constructor.verificar_cobertura(registros)
    constructor.construir_artworks(corpus)
    por_obra = constructor.fragmentar(corpus)
    with pytest.raises(ErrorDeConstrucción, match="cma:1"):
        constructor.registrar(por_obra=por_obra, cita_al_embebido={})

def test_knowledge_sin_puerto_de_texto_no_se_puede_construir_y_lo_dice(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto, con_texto=False)
    corpus = constructor.verificar_cobertura(registros)
    constructor.construir_artworks(corpus)
    por_obra = constructor.fragmentar(corpus)
    with pytest.raises(ErrorDeConstrucción, match="invariante 3"):
        constructor.construir_knowledge(por_obra)

def test_knowledge_lleva_el_texto_y_la_licencia_del_fragmento(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    _construir_todo(constructor, registros)
    resultados = constructor.indice.consultar(
        KNOWLEDGE,
        EmbeddingTextoFijo(dim=8).embeber_uno("Corot painted this view."),
        1,
        where={"artwork_id": {"$in": list(IDS_CORPUS)}},
    )
    assert resultados[0].metadata["license"] == "CC0"
    assert resultados[0].metadata["artwork_id"] in IDS_CORPUS
    assert constructor.indice.documento(KNOWLEDGE, resultados[0].id)

def test_invariante_4_la_receta_contiene_los_nueve_ingredientes_del_id(cfg, escenario):
    from tp3_rag.dominio.variante import INGREDIENTES_DEL_ID

    almacén, _, manifiesto = escenario
    receta = _constructor(cfg, almacén, manifiesto).receta()
    assert set(INGREDIENTES_DEL_ID) <= set(receta)
    assert len(receta) > len(INGREDIENTES_DEL_ID)

def test_invariante_4_la_receta_guarda_lo_que_el_id_no_incluye(cfg, escenario):
    almacén, _, manifiesto = escenario
    receta = _constructor(cfg, almacén, manifiesto).receta()
    assert "limite_n_de_la_ingesta" in receta
    assert receta["tabla_de_licencias_de_nr1"]["aic"]["description"] == "CC-BY-4.0"
    assert receta["version_del_preprocesado"] == "1"

def test_registrar_dos_veces_lee_y_no_reescribe(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    primero = constructor.registrar(por_obra=por_obra, cita_al_embebido={})
    segundo = constructor.registrar(por_obra=por_obra, cita_al_embebido={})
    assert segundo.ya_estaba_registrada
    assert segundo.registro.variante_id == primero.registro.variante_id

def test_el_mismo_id_con_otra_receta_es_error_y_su_causa_es_un_ingrediente_faltante(
    cfg, escenario
):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    constructor.registrar(por_obra=por_obra, cita_al_embebido={})

    cfg_otra = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, provenance]",
            "indice.metrica": "l2",
        }
    )
    otro = _constructor(cfg_otra, almacén, manifiesto)
    assert otro.variante_id == constructor.variante_id
    otro_por_obra = _construir_todo(otro, registros)
    with pytest.raises(ErrorDeArtefacto, match="falta un ingrediente"):
        otro.registrar(por_obra=otro_por_obra, cita_al_embebido={})

def test_cambiar_un_eje_de_la_ablacion_secundaria_cambia_el_id(cfg, escenario):
    almacén, _, manifiesto = escenario
    base = _constructor(cfg, almacén, manifiesto)
    con_citations = _constructor(
        Configuración.cargar(
            overrides={
                "fragmentacion.unidad": "caracteres",
                "fragmentacion.campos_de_knowledge": "[description, provenance, citations]",
            }
        ),
        almacén,
        manifiesto,
    )
    assert base.variante_id != con_citations.variante_id

def test_el_orden_de_los_campos_en_el_toml_no_cambia_el_id(cfg, escenario):
    almacén, _, manifiesto = escenario
    al_revés = _constructor(
        Configuración.cargar(
            overrides={
                "fragmentacion.unidad": "caracteres",
                "fragmentacion.campos_de_knowledge": "[provenance, description]",
            }
        ),
        almacén,
        manifiesto,
    )
    assert al_revés.variante_id == _constructor(cfg, almacén, manifiesto).variante_id

def test_k1_mover_el_proxy_no_cambia_el_variante_id(cfg, escenario):
    almacén, _, manifiesto = escenario
    otro_proxy = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, provenance]",
            "proveedor.base_url": "https://otro-proxy.itba.edu.ar/",
        }
    )
    assert (
        _constructor(otro_proxy, almacén, manifiesto).variante_id
        == _constructor(cfg, almacén, manifiesto).variante_id
    )

def test_k1_la_receta_declara_el_base_url_como_procedencia(cfg, escenario):
    almacén, _, manifiesto = escenario
    receta = _constructor(cfg, almacén, manifiesto).receta()
    assert receta["base_url_del_proveedor"].startswith("https://sa-llmproxy")
    assert "procedencia" in receta["nota_del_base_url"].lower()

def test_k2_el_modelo_con_prefijo_si_cambia_el_id(cfg, escenario):
    almacén, _, manifiesto = escenario
    pelado = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, provenance]",
            "embebido_texto.modelo": "text-embedding-3-small",
        }
    )
    assert (
        _constructor(pelado, almacén, manifiesto).variante_id
        != _constructor(cfg, almacén, manifiesto).variante_id
    )

def test_el_costo_sale_del_libro_mayor_y_lleva_la_poblacion_al_lado(cfg, escenario):
    from tp3_rag.nucleo.observabilidad import Observabilidad

    almacén, registros, manifiesto = escenario
    obs = Observabilidad(silencioso=True)
    constructor = _constructor(cfg, almacén, manifiesto)
    constructor.obs = obs
    constructor.embebedor_de_texto = obs.instrumentar(
        EmbeddingTextoFijo(dim=8), rol="embedding_texto"
    )
    por_obra = _construir_todo(constructor, registros)

    medición = constructor.medición_del_embebido_de_texto(por_obra)
    assert medición["medido"]["hubo_llamadas"] is True
    assert medición["medido"]["tokens_in"] > 0
    assert medición["medido"]["n_fragmentos_embebidos"] == sum(
        len(v) for v in por_obra.values()
    )
    assert medición["poblacion"]["n_obras_del_corpus"] == len(IDS_CORPUS)
    assert medición["poblacion"]["campos_de_knowledge"] == ["description", "provenance"]

def test_sin_precio_declarado_el_costo_es_ausente_y_no_cero(cfg, escenario):
    almacén, registros, manifiesto = escenario
    sin_precio = Configuración.cargar(
        overrides={
            "fragmentacion.unidad": "caracteres",
            "fragmentacion.campos_de_knowledge": "[description, provenance]",
            "embebido_texto.modelo": "openai/text-embedding-3-large",
        }
    )
    constructor = _constructor(sin_precio, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    medición = constructor.medición_del_embebido_de_texto(por_obra)
    assert medición["precio_usd_por_millon_de_tokens"] is None
    assert medición["costo_usd"] is None

def test_el_registro_persiste_el_libro_mayor_porque_ob2_lo_pide(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    resultado = constructor.registrar(por_obra=por_obra, cita_al_embebido={})
    contenido = resultado.registro.a_json()
    assert "libro_mayor" in contenido
    assert contenido["costo_del_embebido_de_texto"]["poblacion"]["n_fragmentos"] > 0

def test_el_costo_medido_no_entra_en_la_comparacion_de_clase_1(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    constructor.registrar(por_obra=por_obra, cita_al_embebido={})

    from tp3_rag.nucleo.observabilidad import Observabilidad

    otro = _constructor(cfg, almacén, manifiesto)
    obs = Observabilidad(silencioso=True)
    otro.obs = obs
    otro.embebedor_de_texto = obs.instrumentar(EmbeddingTextoFijo(dim=8), rol="embedding_texto")
    otro_por_obra = _construir_todo(otro, registros)
    assert otro.registrar(por_obra=otro_por_obra, cita_al_embebido={}).ya_estaba_registrada

def test_el_ordinal_del_embebido_no_entra_en_el_variante_id(cfg, escenario):
    almacén, registros, manifiesto = escenario
    constructor = _constructor(cfg, almacén, manifiesto)
    por_obra = _construir_todo(constructor, registros)
    resultado = constructor.registrar(
        por_obra=por_obra, cita_al_embebido={"catalogo_id": "cat0000000000000", "ordinal": 7}
    )
    assert resultado.registro.cita_al_embebido["ordinal"] == 7
    assert "ordinal" not in str(resultado.registro.receta)
