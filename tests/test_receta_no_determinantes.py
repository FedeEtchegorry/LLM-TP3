from __future__ import annotations

import copy
import glob
import json

import pytest

from tp3_rag.dominio.variante import RegistroDeÍndices
from tp3_rag.modulo_a import constructor_de_indice as ci
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDeArtefacto

pytestmark = pytest.mark.nivel1

def _receta(**cambios) -> dict:
    base = {
        "catalogo_id": "cafecafecafecafe",
        "split_id": "0123456789abcdef",
        "criterio_de_inclusion": {"description_no_vacia": True, "imagen_real": True},
        "checkpoint_de_clip": "openai/clip-vit-base-patch32",
        "version_del_preprocesado": "1",
        "estrategia_de_chunking": {"nombre": "por_campo_semantico", "tamano_objetivo": 500},
        "campos_de_knowledge": ["description", "provenance"],
        "modelo_de_embedding_de_texto": "openai/text-embedding-3-small",
        "criterio_de_culture": "primer_componente_por_coma",
        "limite_n_de_la_ingesta": 0,
        "dim_del_embedding_de_texto": 1536,
        "base_url_del_proveedor": "https://sa-llmproxy.it.itba.edu.ar/",
        "nota_del_base_url": "procedencia, no ingrediente",
        ci.CLAVE_DEL_BLOQUE: list(ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO),
        "nota_de_las_no_determinantes": "la nota del bloque",
        "metrica_del_indice": "cosine",
        "campos_de_metadata_de_artworks": ["museum", "title"],
        "tabla_de_licencias_de_nr1": {"aic": {"description": "CC-BY-4.0"}},
        "nota_de_la_tabla_de_licencias": "copia como procedencia",
        "nota_de_los_ingredientes": "los nueve primeros",
        "configuracion_de_indexado_resuelta": {
            "fragmentacion": {"estrategia": "por_campo_semantico", "tamano_objetivo": 500},
            "embebido": {"checkpoint": "openai/clip-vit-base-patch32", "threads": 8, "batch": 16},
            "embebido_texto": {"modelo": "openai/text-embedding-3-small", "batch": 128},
            "indice": {"metrica": "cosine", "batch_de_upsert": 5000},
            "normalizacion": {"culture": {"criterio": "primer_componente_por_coma"}},
        },
    }
    base.update(cambios)
    return base

def _registro(receta: dict) -> RegistroDeÍndices:
    return RegistroDeÍndices(
        {
            "variante_id": "abcdef0123456789",
            "receta": receta,
            "conteos": {"n_indexadas": 60, "n_fragmentos": 140},
            "colecciones": {"artworks": "a", "knowledge": "k"},
            "procedencia": {"embebido": {"catalogo_id": "cafecafecafecafe", "ordinal": 1}},
            "invariantes_de_9_6": {},
        }
    )

def _comparar(previa: dict, nueva: dict) -> None:
    ci._exigir_mismo_contenido(_registro(previa), _registro(nueva))

def _en(receta: dict, *ruta, valor):
    copia = copy.deepcopy(receta)
    nodo = copia
    for parte in ruta[:-1]:
        nodo = nodo[parte]
    nodo[ruta[-1]] = valor
    return copia

def test_mover_el_proxy_ya_no_es_error_duro():
    _comparar(
        _receta(),
        _receta(base_url_del_proveedor="https://otro-proxy.it.itba.edu.ar/"),
    )

@pytest.mark.parametrize(
    "ruta, valor",
    [
        (("embebido", "threads"), 2),
        (("embebido", "batch"), 4),
        (("embebido_texto", "batch"), 7),
        (("indice", "batch_de_upsert"), 1000),
    ],
)
def test_los_parametros_de_maquina_tampoco_son_error_duro(ruta, valor):
    nueva = copy.deepcopy(_receta())
    nodo = nueva["configuracion_de_indexado_resuelta"]
    for parte in ruta[:-1]:
        nodo = nodo[parte]
    nodo[ruta[-1]] = valor
    _comparar(_receta(), nueva)

def test_la_nota_en_prosa_del_bloque_no_reindexa_nada():
    _comparar(_receta(), _receta(nota_de_las_no_determinantes="otra redacción"))

@pytest.mark.parametrize(
    "receta_nueva, qué",
    [
        (lambda: _receta(dim_del_embedding_de_texto=3072), "dim del embedding de texto"),
        (lambda: _receta(metrica_del_indice="l2"), "métrica del índice"),
        (lambda: _receta(campos_de_knowledge=["description"]), "campos de knowledge"),
        (lambda: _receta(criterio_de_culture="crudo"), "criterio de culture"),
        (
            lambda: _en(
                _receta(),
                "configuracion_de_indexado_resuelta", "fragmentacion", "tamano_objetivo",
                valor=2000,
            ),
            "tamaño objetivo del chunking",
        ),
        (
            lambda: _en(
                _receta(),
                "configuracion_de_indexado_resuelta", "embebido", "checkpoint",
                valor="openai/clip-vit-large-patch14",
            ),
            "checkpoint de CLIP",
        ),
        (
            lambda: _en(
                _receta(),
                "configuracion_de_indexado_resuelta", "normalizacion", "culture",
                valor={"criterio": "crudo"},
            ),
            "criterio de culture, adentro de la configuración resuelta",
        ),
    ],
)
def test_una_clave_que_determina_el_contenido_sigue_siendo_error_duro(receta_nueva, qué):
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(_receta(), receta_nueva())
    assert "determinan el contenido indexado" in str(e.value), qué

def test_el_mensaje_nombra_LAS_DOS_causas_posibles():
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(_receta(), _receta(dim_del_embedding_de_texto=3072))
    mensaje = str(e.value)
    assert "falta un ingrediente en la tabla de identidades" in mensaje
    assert "falta declarar esta clave" in mensaje
    assert ci.CLAVE_DEL_BLOQUE in mensaje
    assert "dim_del_embedding_de_texto" in mensaje
    assert "3072" in mensaje and "1536" in mensaje

def test_cuando_hay_las_dos_clases_de_diferencia_el_mensaje_las_separa():
    nueva = _receta(
        dim_del_embedding_de_texto=3072,
        base_url_del_proveedor="https://otro-proxy.it.itba.edu.ar/",
    )
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(_receta(), nueva)
    mensaje = str(e.value)
    assert "sí determinan el contenido" in mensaje
    assert "base_url_del_proveedor" in mensaje and "sí** están declaradas" in mensaje

def test_una_clave_nueva_que_nadie_declaro_es_error_duro():
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(_receta(), _receta(eje_que_nadie_penso="algo"))
    assert "eje_que_nadie_penso" in str(e.value)

def test_dos_bloques_declarados_distintos_no_se_comparan_en_silencio():
    con_dim_adentro = _receta(
        **{
            ci.CLAVE_DEL_BLOQUE: list(ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO)
            + ["dim_del_embedding_de_texto"]
        }
    )
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(_receta(), con_dim_adentro)
    assert "otro bloque de claves no determinantes" in str(e.value)
    assert "dim_del_embedding_de_texto" in str(e.value)

def test_una_receta_sin_el_bloque_se_lee_con_el_conjunto_por_defecto():
    vieja = _receta()
    del vieja[ci.CLAVE_DEL_BLOQUE]
    assert ci.bloque_declarado(vieja) == ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO
    assert ci.bloque_declarado(_receta()) == tuple(ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO)
    _comparar(vieja, _receta())
    _comparar(_receta(), vieja)
    _comparar(vieja, _receta(base_url_del_proveedor="https://otro.itba.edu.ar/"))

RECETAS_SELLADAS = sorted(glob.glob(str(RAÍZ_DEL_REPO / "artifacts" / "indices" / "*.json")))

@pytest.mark.nivel2
@pytest.mark.skipif(
    not RECETAS_SELLADAS,
    reason=(
        "0 RegistroDeÍndices en artifacts/indices/ de este árbol — las 3 recetas "
        "selladas no se compararon (el repo publicado no empaqueta `artifacts/`; en la rama de "
        "trabajo son 3). NO es un criterio verificado"
    ),
)
@pytest.mark.parametrize("ruta", RECETAS_SELLADAS, ids=lambda r: r.split("/")[-1][:16])
def test_las_recetas_selladas_del_repo_siguen_leyendose_y_comparandose(ruta):
    sellada = RegistroDeÍndices(json.loads(open(ruta, encoding="utf-8").read()))
    sellada.validar()

    assert ci.CLAVE_DEL_BLOQUE not in sellada.receta
    assert "nota_del_base_url" in sellada.receta
    assert ci.bloque_declarado(sellada.receta) == ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO
    assert (
        "campos_de_knowledge"
        in sellada.receta["configuracion_de_indexado_resuelta"]["fragmentacion"]
    )

    def en_la_forma_nueva() -> RegistroDeÍndices:
        nuevo = copy.deepcopy(sellada.contenido)
        r = nuevo["receta"]
        r[ci.CLAVE_DEL_BLOQUE] = list(ci.NO_DETERMINAN_EL_CONTENIDO_INDEXADO)
        r["nota_de_las_no_determinantes"] = "la nota del bloque"
        del r["configuracion_de_indexado_resuelta"]["fragmentacion"]["campos_de_knowledge"]
        return RegistroDeÍndices(nuevo)

    ci._exigir_mismo_contenido(sellada, en_la_forma_nueva())

    k1 = en_la_forma_nueva()
    k1.contenido["receta"]["base_url_del_proveedor"] = "https://otro-proxy.itba.edu.ar/"
    resuelta = k1.contenido["receta"]["configuracion_de_indexado_resuelta"]
    resuelta["embebido"]["threads"] = 2
    resuelta["indice"]["batch_de_upsert"] = 1000
    resuelta["embebido_texto"]["batch"] = 7
    ci._exigir_mismo_contenido(sellada, k1)

    eje = en_la_forma_nueva()
    eje.contenido["receta"]["configuracion_de_indexado_resuelta"]["fragmentacion"][
        "tamano_objetivo"
    ] = 2000
    with pytest.raises(ErrorDeArtefacto):
        ci._exigir_mismo_contenido(sellada, eje)

def test_la_copia_vieja_se_lee_pero_NO_se_arregla_en_silencio_si_difiere():
    coherente = _en(
        _receta(),
        "configuracion_de_indexado_resuelta", "fragmentacion", "campos_de_knowledge",
        valor=["description", "provenance"],
    )
    _comparar(coherente, _receta())

    incoherente = _en(
        _receta(),
        "configuracion_de_indexado_resuelta", "fragmentacion", "campos_de_knowledge",
        valor=["description"],
    )
    with pytest.raises(ErrorDeArtefacto) as e:
        _comparar(incoherente, _receta())
    assert "mismo dato declarado dos veces" in str(e.value)
    assert "incoherente consigo misma" in str(e.value)
