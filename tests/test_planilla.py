from __future__ import annotations

import csv
import io

import pytest

from tp3_rag.dominio.juicios import AporteDeAnotador
from tp3_rag.modulo_c.constructor_de_pool import ConstructorDePool
from tp3_rag.modulo_c.planilla import COLUMNAS, PlanillaDeAnotación
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConstrucción

from conftest import (
    CONFIG_DE_PRUEBA,
    corridas_de_prueba,
    manifiesto_de_prueba,
    registro_de_variante_de_prueba,
    rubrica_de_prueba,
    set_de_prueba,
)

pytestmark = pytest.mark.nivel1

@pytest.fixture
def camino(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    manifiesto = manifiesto_de_prueba()
    pool = ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto,
        corridas=corridas_de_prueba(set_),
    ).pool
    rubrica = rubrica_de_prueba()
    almacen_temporal.escribir_texto(
        rutas_de_prueba.rubricas / f"{rubrica.rubrica_id}.md", rubrica.texto
    )
    planilla = PlanillaDeAnotación(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba)
    return {
        "set": set_,
        "manifiesto": manifiesto,
        "pool": pool,
        "rubrica": rubrica,
        "planilla": planilla,
        "almacen": almacen_temporal,
        "rutas": rutas_de_prueba,
    }

def _exportar(camino, anotador="ana"):
    return camino["planilla"].exportar(
        pool=camino["pool"],
        rubrica=camino["rubrica"],
        anotador=anotador,
        set_de_consultas=camino["set"],
        manifiesto=camino["manifiesto"],
        registros_de_variantes={
            registro_de_variante_de_prueba().variante_id: registro_de_variante_de_prueba()
        },
    )

def _llenar(ruta, grados=None, *, transformar=None):
    filas = list(csv.DictReader(io.StringIO(ruta.read_text(encoding="utf-8"))))
    for n, fila in enumerate(filas):
        fila["grado"] = str((grados or [0, 1, 2])[n % len(grados or [0, 1, 2])])
    if transformar is not None:
        filas = transformar(filas)
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=list(COLUMNAS), lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(filas)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")
    return ruta

def _importar(camino, ruta):
    return camino["planilla"].importar(
        planilla=ruta,
        set_de_consultas=camino["set"],
        buscar_pool=lambda _: camino["pool"],
        buscar_rubrica=lambda _: camino["rubrica"],
    )

def test_exportar_e_importar_congela_el_aporte_con_la_rubrica_en_la_identidad(camino):
    exportación = _exportar(camino)
    assert exportación.n_filas == 20

    resultado = _importar(camino, _llenar(exportación.ruta))
    aporte = resultado.aporte

    assert resultado.n_juicios == 20
    assert aporte.identidad == f"{camino['pool'].split_id}-piloto-ana"
    assert aporte.rubrica_id == "piloto"
    assert aporte.pool_id == camino["pool"].pool_id
    assert aporte.hash_de_la_rubrica == camino["rubrica"].hash_del_texto
    assert resultado.ruta.name.endswith("-piloto-ana.csv")

def test_el_aporte_congelado_se_vuelve_a_leer_igual(camino):
    exportación = _exportar(camino)
    escrito = _importar(camino, _llenar(exportación.ruta)).aporte
    leído = AporteDeAnotador.leer(camino["almacen"], camino["rutas"].juicios, escrito.identidad)

    assert leído.identidad == escrito.identidad
    assert leído.pares == escrito.pares
    assert leído.por_grado() == escrito.por_grado()
    assert leído.procedencia["pool_id"] == escrito.pool_id

def test_importar_dos_veces_lo_mismo_no_reescribe(camino):
    ruta = _llenar(_exportar(camino).ruta)
    primero = _importar(camino, ruta)
    segundo = _importar(camino, ruta)

    assert not primero.ya_estaba
    assert segundo.ya_estaba

def test_importar_otros_juicios_sobre_la_misma_identidad_se_rechaza(camino):
    ruta = _exportar(camino).ruta
    _importar(camino, _llenar(ruta, [0]))
    with pytest.raises(ErrorDeArtefacto, match="con otros juicios"):
        _importar(camino, _llenar(ruta, [2]))

def test_exportar_cuando_el_aporte_ya_existe_avisa_antes_de_gastar(camino):
    _importar(camino, _llenar(_exportar(camino).ruta))
    with pytest.raises(ErrorDeConstrucción, match="ya hay un aporte congelado"):
        _exportar(camino)

def test_d14_agregar_una_fila_es_error_de_import(camino):
    ruta = _exportar(camino).ruta

    def agregar(filas):
        copia = dict(filas[0])
        copia["artwork_id"] = "cma:999999"
        copia["fila"] = "21"
        return filas + [copia]

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=agregar))
    assert "21 filas" in str(e.value) and "20 ítems" in str(e.value)
    assert "no está en el pool" in str(e.value)

def test_d14_cambiar_un_artwork_id_es_error_de_import(camino):
    ruta = _exportar(camino).ruta
    original = None

    def retocar(filas):
        nonlocal original
        original = filas[3]["artwork_id"]
        filas[3]["artwork_id"] = "cma:123456"
        return filas

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=retocar))
    mensaje = str(e.value)
    assert "cma:123456" in mensaje and original in mensaje
    assert "retocado" in mensaje

def test_d14_reordenar_es_error_de_import(camino):
    ruta = _exportar(camino).ruta

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(
            camino,
            _llenar(ruta, transformar=lambda filas: sorted(filas, key=lambda f: f["artwork_id"])),
        )
    mensaje = str(e.value)
    assert "reordenada" in mensaje
    assert "Ordená por `fila`" in mensaje

def test_d14_pegar_una_fila_de_otra_ronda_se_caza_por_su_propio_contenido(camino):
    ruta = _exportar(camino).ruta

    def pegar(filas):
        filas[5]["rubrica_id"] = "otra-rubrica"
        return filas

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=pegar))
    assert "otra-rubrica" in str(e.value)
    assert "rúbrica" in str(e.value)

def test_d14_borrar_una_fila_tambien_es_error(camino):
    ruta = _exportar(camino).ruta
    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=lambda filas: filas[:-1]))
    assert "estaba en el pool y no está en la planilla" in str(e.value)

def test_una_columna_de_menos_se_rechaza(camino):
    ruta = _exportar(camino).ruta
    texto = ruta.read_text(encoding="utf-8").replace("pedido,", "", 1)
    ruta.write_text(texto, encoding="utf-8")
    with pytest.raises(ErrorDeArtefacto, match="las columnas no son"):
        _importar(camino, ruta)

def test_el_orden_de_las_columnas_no_importa_y_el_de_las_filas_si(camino):
    ruta = _exportar(camino).ruta
    _llenar(ruta)
    filas = list(csv.DictReader(io.StringIO(ruta.read_text(encoding="utf-8"))))
    invertidas = list(reversed(COLUMNAS))
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=invertidas, lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(filas)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")

    assert _importar(camino, ruta).n_juicios == 20

def test_una_celda_vacia_no_se_lee_como_cero_y_dice_cuantas_faltan(camino):
    ruta = _exportar(camino).ruta

    def vaciar(filas):
        filas[7]["grado"] = ""
        filas[11]["grado"] = "  "
        return filas

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=vaciar))
    assert "2 filas sin `grado`" in str(e.value)
    assert "no se lee como" in str(e.value)

def test_un_grado_fuera_de_la_escala_se_rechaza_con_la_escala_a_la_vista(camino):
    ruta = _exportar(camino).ruta
    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(ruta, transformar=_poner(2, "3")))
    assert "[0, 1, 2]" in str(e.value)
    assert "Cumple claramente" in str(e.value)

def _poner(indice, valor):
    def transformar(filas):
        filas[indice]["grado"] = valor
        return filas

    return transformar

def test_no_se_exporta_contra_una_rubrica_plantilla(cfg, camino):
    from tp3_rag.dominio.rubrica import Rúbrica, plantilla

    camino["rubrica"] = Rúbrica.desde_texto(plantilla(designacion="borrador"))
    with pytest.raises(ErrorDeConstrucción, match="no es usable para anotar"):
        _exportar(camino)

def test_un_consulta_id_que_no_resuelve_es_e4(camino):
    ruta = _llenar(_exportar(camino).ruta)
    movido = camino["pool"].consultas[0]
    for imagen in camino["set"].contenido["imagenes"]:
        for consulta in imagen["consultas"]:
            if consulta["consulta_id"] == movido:
                consulta["consulta_id"] = "c99-movida"

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, ruta)
    assert movido in str(e.value)
    assert "no resuelve" in str(e.value)

def test_la_planilla_lleva_lo_que_hace_falta_para_juzgar_sin_leer_codigo(camino):
    exportación = _exportar(camino)
    filas = list(csv.DictReader(io.StringIO(exportación.ruta.read_text(encoding="utf-8"))))

    assert filas[0]["pedido"]
    assert filas[0]["categoria"]
    assert filas[0]["imagen_de_consulta"].endswith(".jpg")
    assert filas[0]["imagen_de_la_obra"].endswith(".jpg")
    assert filas[0]["obra"] and filas[0]["source_url"]
    instrucciones = exportación.ruta_de_instrucciones.read_text(encoding="utf-8")
    assert "Cómo anotar" in instrucciones
    assert "no reordenes" in instrucciones.lower()
    assert "tp3-anotar importar" in instrucciones
    assert "Hacen falta **dos** personas" in instrucciones
    assert "tp3-anotar kappa" in instrucciones
