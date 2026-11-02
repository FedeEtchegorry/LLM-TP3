from __future__ import annotations

import pytest

from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.nucleo.errores import ErrorDeArtefacto
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

from conftest import (
    CONFIG_DE_PRUEBA,
    SPLIT_DE_PRUEBA,
    corrida_de_prueba,
    corridas_de_prueba,
    manifiesto_de_prueba,
    set_de_prueba,
)

pytestmark = pytest.mark.nivel1

def _almacen_con(tmp_path, corridas):
    from tp3_rag.dominio.corrida import AlmacenDeCorridas

    almacén = AlmacenamientoEnArchivos(tmp_path)
    de = AlmacenDeCorridas(almacén, "corridas")
    for c in corridas:
        (tmp_path / "corridas" / c.corrida_id).mkdir(parents=True, exist_ok=True)
        de.escribir(c.corrida_id, c.contenido)
    return almacén, de

class _CfgDeCriterios:
    config_id = CONFIG_DE_PRUEBA

    def __init__(self):
        self._d = {
            "rutas.artefactos_corridas": "corridas",
            "rutas.artefactos_indices": "indices",
            "salida.exigen_atribucion": {"CC-BY-4.0": True, "CC0": False},
            "consulta.n_candidatos": 20,
        }

    def __getitem__(self, clave):
        return self._d[clave]

class _ArgsDeCriterios:
    sin_red = True
    variante_id = None
    autoria = None

class _MontajeDePrueba:
    def __init__(self, manifiesto):
        self.manifiesto = manifiesto

def _correr_criterios(almacén, manifiesto):
    from tp3_rag.cli import validar as cli
    from tp3_rag.nucleo.observabilidad import Observabilidad

    obs = Observabilidad(silencioso=True)
    dicho = []
    obs.decir = dicho.append  # type: ignore[method-assign]
    resultados = cli._criterios(
        _ArgsDeCriterios(),
        _CfgDeCriterios(),
        almacén,
        obs,
        _MontajeDePrueba(manifiesto),
    )
    por_criterio = {r.criterio.split(" ·")[0]: r for r in resultados}
    return por_criterio, "\n".join(dicho)

def test_mg3_una_ad_hoc_con_el_config_id_de_la_fila_queda_afuera(tmp_path):
    manifiesto = manifiesto_de_prueba()
    set_ = set_de_prueba()
    corridas = corridas_de_prueba(set_)
    assert len(corridas) == 30
    ad_hoc = corrida_de_prueba(
        consulta_id="78a9b1ffd092486d",
        top5=list(manifiesto.ids_corpus[:5]),
        procedencia="ad_hoc",
    )
    almacén, _ = _almacen_con(tmp_path, [*corridas, ad_hoc])

    por_criterio, dicho = _correr_criterios(almacén, manifiesto)
    assert por_criterio["el criterio de trazabilidad"].de == 150
    assert "30 corridas × top-5 = 150 ítems" in por_criterio["el criterio de trazabilidad"].poblacion
    assert "30 corridas" in por_criterio["el criterio de atribución"].poblacion
    assert "Población" in dicho and "1 **ad-hoc**" in dicho
    assert "de 31 en disco" in dicho

def test_mg3_las_30_del_set_se_incluyen_todas(tmp_path):
    manifiesto = manifiesto_de_prueba()
    almacén, _ = _almacen_con(tmp_path, corridas_de_prueba(set_de_prueba()))
    por_criterio, dicho = _correr_criterios(almacén, manifiesto)
    assert por_criterio["el criterio de trazabilidad"].de == 150
    assert "30 corridas × top-5 = 150 ítems" in por_criterio["el criterio de trazabilidad"].poblacion
    assert "de 30 en disco" in dicho
    assert "Población" not in dicho, "sin ad-hoc no hay recuento que publicar"

def test_mg3_el_ultimo_ordinal_sigue_ganando_despues_del_filtro(tmp_path):
    manifiesto = manifiesto_de_prueba()
    corridas = corridas_de_prueba(set_de_prueba())
    primera = corridas[0]
    segunda = corrida_de_prueba(
        consulta_id=primera.consulta_id,
        top5=list(manifiesto.ids_corpus[10:15]),
        ordinal=2,
    )
    almacén, _ = _almacen_con(tmp_path, [*corridas, segunda])
    por_criterio, dicho = _correr_criterios(almacén, manifiesto)
    assert por_criterio["el criterio de trazabilidad"].de == 150, "31 corridas del set → 30 ítems × 5"
    assert "de 31 en disco" in dicho
    assert "Población" not in dicho, "las dos son del set: ninguna es ad-hoc"

def test_mg3_una_ad_hoc_y_un_segundo_ordinal_juntos(tmp_path):
    manifiesto = manifiesto_de_prueba()
    corridas = corridas_de_prueba(set_de_prueba())
    extras = [
        corrida_de_prueba(
            consulta_id=corridas[0].consulta_id,
            top5=list(manifiesto.ids_corpus[10:15]),
            ordinal=2,
        ),
        corrida_de_prueba(
            consulta_id="78a9b1ffd092486d",
            top5=list(manifiesto.ids_corpus[:5]),
            procedencia="ad_hoc",
        ),
        corrida_de_prueba(
            consulta_id="78a9b1ffd092486d",
            top5=list(manifiesto.ids_corpus[5:10]),
            procedencia="ad_hoc",
            ordinal=2,
        ),
    ]
    almacén, _ = _almacen_con(tmp_path, [*corridas, *extras])
    por_criterio, dicho = _correr_criterios(almacén, manifiesto)
    assert por_criterio["el criterio de trazabilidad"].de == 150
    assert "2 **ad-hoc**" in dicho and "de 33 en disco" in dicho

def test_mg3_todo_ad_hoc_es_error_duro_y_no_un_marcador_sobre_1_corrida(tmp_path):
    from tp3_rag.nucleo.errores import ErrorDelProyecto

    manifiesto = manifiesto_de_prueba()
    almacén, _ = _almacen_con(
        tmp_path,
        [
            corrida_de_prueba(
                consulta_id="78a9b1ffd092486d",
                top5=list(manifiesto.ids_corpus[:5]),
                procedencia="ad_hoc",
            )
        ],
    )
    with pytest.raises(ErrorDelProyecto, match="todas ad-hoc"):
        _correr_criterios(almacén, manifiesto)

TRES = ["description", "did_you_know", "provenance"]
CINCO = [*TRES, "exhibitions", "citations"]

VARIANTES = {
    "c3274567371ff42b": (TRES, {"nombre": "por_campo_semantico"}, 6232),
    "a02f862565637f06": (CINCO, {"nombre": "por_campo_semantico"}, 12772),
    "d8da543eb94034ca": (
        TRES,
        {"nombre": "ciego", "orden_de_concatenacion": list(TRES)},
        2775,
    ),
}

class _CfgFalsa:
    def __init__(self, campos, estrategia):
        self._d = {
            "fragmentacion.campos_de_knowledge": campos,
            "fragmentacion.estrategia": estrategia,
        }

    def __getitem__(self, clave):
        return self._d[clave]

def _registro(variante_id):
    from tp3_rag.dominio.variante import INGREDIENTES_DEL_ID, RegistroDeÍndices

    campos, estrategia, n_fragmentos = VARIANTES[variante_id]
    receta = {k: f"valor-{k}" for k in INGREDIENTES_DEL_ID}
    receta["split_id"] = SPLIT_DE_PRUEBA
    receta["campos_de_knowledge"] = list(campos)
    receta["estrategia_de_chunking"] = dict(estrategia)
    registro = RegistroDeÍndices(
        {
            "variante_id": variante_id,
            "receta": receta,
            "conteos": {"n_indexadas": 2605, "n_fragmentos": n_fragmentos},
            "colecciones": {"artworks": "x", "knowledge": "y"},
            "procedencia": {
                "embebido": {"catalogo_id": "e10c4e4f9cce0e74", "ordinal": 2}
            },
            "invariantes_de_9_6": {},
        }
    )
    registro.validar()
    return registro

@pytest.mark.parametrize("variante_id", sorted(VARIANTES))
def test_mg5_el_par_coherente_pasa(variante_id):
    campos, estrategia, _ = VARIANTES[variante_id]
    cfg = _CfgFalsa(list(campos), estrategia["nombre"])
    registro = mnt.exigir_receta_coherente(cfg, _registro(variante_id))
    assert registro.variante_id == variante_id

def test_mg5_la_ampliada_sin_el_set_de_sus_campos_es_error_duro():
    cfg = _CfgFalsa(list(TRES), "por_campo_semantico")
    with pytest.raises(ErrorDeArtefacto) as e:
        mnt.exigir_receta_coherente(cfg, _registro("a02f862565637f06"))
    mensaje = str(e.value)
    assert "configuración" in mensaje and "receta" in mensaje
    assert "exhibitions" in mensaje and "citations" in mensaje
    assert "12772" in mensaje, "y el tamaño del corpus que se habría leído"
    assert "otros campos de `knowledge`" in mensaje

def test_mg5_la_congelada_declarando_los_cinco_campos_es_error_duro():
    cfg = _CfgFalsa(list(CINCO), "por_campo_semantico")
    with pytest.raises(ErrorDeArtefacto, match="sobran en la configuración"):
        mnt.exigir_receta_coherente(cfg, _registro("c3274567371ff42b"))

def test_mg5_el_brazo_ciego_sin_su_set_de_estrategia_es_error_duro():
    cfg = _CfgFalsa(list(TRES), "por_campo_semantico")
    with pytest.raises(ErrorDeArtefacto) as e:
        mnt.exigir_receta_coherente(cfg, _registro("d8da543eb94034ca"))
    mensaje = str(e.value)
    assert "por_campo_semantico" in mensaje and "ciego" in mensaje
    assert "2775" in mensaje

def test_mg5_la_congelada_declarando_ciego_es_error_duro():
    cfg = _CfgFalsa(list(TRES), "ciego")
    with pytest.raises(ErrorDeArtefacto, match="otra estrategia de chunking"):
        mnt.exigir_receta_coherente(cfg, _registro("c3274567371ff42b"))

def test_mg5_reordenar_los_campos_del_brazo_ciego_es_error_duro():
    cfg = _CfgFalsa(list(reversed(TRES)), "ciego")
    with pytest.raises(ErrorDeArtefacto, match="orden de la concatenación es el texto"):
        mnt.exigir_receta_coherente(cfg, _registro("d8da543eb94034ca"))

def test_mg5_reordenar_los_campos_del_brazo_por_campo_NO_es_error():
    cfg = _CfgFalsa(list(reversed(TRES)), "por_campo_semantico")
    registro = mnt.exigir_receta_coherente(cfg, _registro("c3274567371ff42b"))
    assert registro.variante_id == "c3274567371ff42b"

def test_mg5_corre_en_la_resolucion_de_la_variante_y_no_en_cada_entrypoint():
    import inspect

    fuente = inspect.getsource(mnt.registro_de_indices)
    assert fuente.count("return exigir_receta_coherente(") == 2, (
        "las dos ramas: la del `--variante-id` explícito y la de la única "
        "registrada. Si una sola llamara, el camino sin flag quedaría sin chequeo"
    )
    assert fuente.count("    return ") == 2
