from __future__ import annotations

import csv
import io

import pytest

from tp3_rag.dominio.juicios import AporteDeAnotador
from tp3_rag.modulo_c.constructor_de_pool import ConstructorDePool
from tp3_rag.modulo_c.planilla import (
    ALCANCE,
    ALCANCE_DESEMPATE,
    COLUMNAS,
    COLUMNAS_EXPORTADAS,
    FUERA_DEL_ALCANCE,
    PlanillaDeAnotación,
)
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

def _exportar(camino, anotador="ana", **extra):
    if "alcance" not in extra and anotador.endswith("tercero"):
        extra["alcance"] = ALCANCE_DESEMPATE
    return _exportar_con_alcance(camino, anotador=anotador, **extra)

def _exportar_con_alcance(camino, *, anotador="ana", **extra):
    return camino["planilla"].exportar(
        pool=camino["pool"],
        rubrica=camino["rubrica"],
        anotador=anotador,
        **extra,
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
    escritor = csv.DictWriter(
        buffer, fieldnames=list(COLUMNAS_EXPORTADAS), lineterminator="\n"
    )
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
    assert "el compromiso con la rúbrica que se juzgó" in str(e.value)

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
    invertidas = list(reversed(COLUMNAS_EXPORTADAS))
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
    assert "no resuelve" in str(e.value) and "la referencia que resuelve" in str(e.value)

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

def _consenso_y_kappa(camino, *, pareja=("ana", "beto"), sin_consenso=4):
    from tp3_rag.dominio import ground_truth as gt
    from tp3_rag.modulo_c import acuerdo as ac

    pool = camino["pool"]
    rubrica_id = camino["rubrica"].rubrica_id
    disputados = list(pool.items)[:sin_consenso]
    camino["almacen"].escribir_texto(
        ac.ruta_del_consenso(camino["rutas"].acuerdo, pool.pool_id, rubrica_id),
        gt.armar_csv(
            procedencia={"artefacto": "Consenso", "pool_id": pool.pool_id},
            columnas=ac.COLUMNAS_DEL_CONSENSO,
            filas=[
                {
                    "pool_id": pool.pool_id,
                    "rubrica_id": rubrica_id,
                    "consulta_id": i.consulta_id,
                    "artwork_id": i.artwork_id,
                    "grado": "" if i in disputados else 1,
                    "como_se_resolvio": "sin_consenso" if i in disputados else "acuerdo",
                    "grados_por_anotador": "",
                }
                for i in pool.items
            ],
        ),
    )
    camino["almacen"].escribir_json(
        ac.ruta_del_kappa(camino["rutas"].acuerdo, pool.pool_id, rubrica_id),
        {
            "artefacto": "Kappa",
            "pool_id": pool.pool_id,
            "rubrica": {
                "rubrica_id": rubrica_id,
                "hash_del_texto": camino["rubrica"].hash_del_texto,
            },
            "pareja_del_kappa": list(pareja),
            "kappa": {"kappa": 0.81, "umbral": 0.6, "n_pares_comparados": len(pool.items)},
        },
    )
    return frozenset(i.par for i in disputados)

def _sin_columna_de_alcance(ruta):
    filas = list(csv.DictReader(io.StringIO(ruta.read_text(encoding="utf-8"))))
    buffer = io.StringIO()
    escritor = csv.DictWriter(buffer, fieldnames=list(COLUMNAS), lineterminator="\n")
    escritor.writeheader()
    escritor.writerows({c: f[c] for c in COLUMNAS} for f in filas)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")
    return ruta

def _llenar_solo(ruta, pares):
    filas = list(csv.DictReader(io.StringIO(ruta.read_text(encoding="utf-8"))))
    for n, fila in enumerate(filas):
        par = (fila["consulta_id"], fila["artwork_id"])
        fila["grado"] = str(n % 3) if par in pares else ""
    buffer = io.StringIO()
    escritor = csv.DictWriter(
        buffer, fieldnames=list(COLUMNAS_EXPORTADAS), lineterminator="\n"
    )
    escritor.writeheader()
    escritor.writerows(filas)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")
    return ruta

def test_td4_el_tercero_importa_su_aporte_parcial_y_el_alcance_queda_adentro(camino):
    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    resultado = _importar(camino, _llenar_solo(exportación.ruta, disputados))

    assert resultado.n_juicios == 4
    assert resultado.alcance.tipo == ALCANCE_DESEMPATE
    assert resultado.alcance.n_del_pool == 20
    assert resultado.alcance.pareja_del_kappa == ("ana", "beto")
    alcance = resultado.aporte.procedencia["alcance"]
    assert alcance["alcance"] == ALCANCE_DESEMPATE
    assert (alcance["n_en_alcance"], alcance["n_fuera_del_alcance"]) == (4, 16)
    assert alcance["pareja_del_kappa"] == ["ana", "beto"]
    leído = AporteDeAnotador.leer(
        camino["almacen"], camino["rutas"].juicios, resultado.aporte.identidad
    )
    assert leído.procedencia["alcance"]["alcance"] == ALCANCE_DESEMPATE
    assert len(leído.juicios) == 4

def test_td4_la_planilla_de_desempate_declara_su_alcance_fila_por_fila(camino):
    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    filas = list(csv.DictReader(io.StringIO(exportación.ruta.read_text(encoding="utf-8"))))

    marcadas = {
        (f["consulta_id"], f["artwork_id"]) for f in filas if f[ALCANCE] == ALCANCE_DESEMPATE
    }
    assert marcadas == disputados
    assert sum(1 for f in filas if f[ALCANCE] == FUERA_DEL_ALCANCE) == 16
    instrucciones = exportación.ruta_de_instrucciones.read_text(encoding="utf-8")
    assert "Tu alcance es el DESEMPATE: 4 de las 20" in instrucciones
    assert "no una corrección del de nadie" in instrucciones

def test_td4_un_anotador_de_la_pareja_del_kappa_no_puede_tener_alcance_parcial(camino):
    disputados = _consenso_y_kappa(camino, pareja=("ana", "beto"))
    exportación = _exportar(camino, anotador="ana")
    ruta = _sin_columna_de_alcance(_llenar_solo(exportación.ruta, disputados))
    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, ruta)
    mensaje = str(e.value)
    assert "es de la pareja del κ" in mensaje
    assert "menos pares en silencio" in mensaje
    assert "16 filas sin `grado`" in mensaje

def test_td4_exportar_un_desempate_para_la_pareja_falla_antes_de_gastar(camino):
    _consenso_y_kappa(camino, pareja=("ana", "beto"))
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar_con_alcance(camino, anotador="beto", alcance=ALCANCE_DESEMPATE)
    assert "es de la pareja del κ" in str(e.value)

def test_td4_sin_kappa_medido_no_hay_pareja_declarada_y_el_parcial_no_entra(camino):
    exportación = _exportar_con_alcance(camino, anotador="ia-tercero")
    pares = frozenset(i.par for i in list(camino["pool"].items)[:4])
    ruta = _sin_columna_de_alcance(_llenar_solo(exportación.ruta, pares))
    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, ruta)
    mensaje = str(e.value)
    assert "no hay un κ ni un consenso declarados" in mensaje
    assert "tp3-anotar kappa" in mensaje

def test_td4_un_juicio_de_mas_fuera_del_alcance_tambien_se_rechaza(camino):
    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    de_mas = next(i.par for i in camino["pool"].items if i.par not in disputados)
    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar_solo(exportación.ruta, disputados | {de_mas}))
    mensaje = str(e.value)
    assert "FUERA del alcance" in mensaje
    assert "corrigiendo" in mensaje or "corrección del consenso" in mensaje

def test_td4_una_planilla_completa_sigue_exigiendo_las_filas_completas(camino):
    _consenso_y_kappa(camino, pareja=("ana", "beto"))
    exportación = _exportar(camino, anotador="ana")

    def vaciar(filas):
        filas[3]["grado"] = ""
        return filas

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, _llenar(exportación.ruta, transformar=vaciar))
    assert "1 filas sin `grado`" in str(e.value)

def test_td4_la_columna_alcance_no_puede_declarar_dos_alcances(camino):
    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    ruta = _llenar_solo(exportación.ruta, disputados)
    filas = list(csv.DictReader(io.StringIO(ruta.read_text(encoding="utf-8"))))
    filas[0][ALCANCE] = "completa"
    buffer = io.StringIO()
    escritor = csv.DictWriter(
        buffer, fieldnames=list(COLUMNAS_EXPORTADAS), lineterminator="\n"
    )
    escritor.writeheader()
    escritor.writerows(filas)
    ruta.write_text(buffer.getvalue(), encoding="utf-8")

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, ruta)
    assert "declara ['completa', 'desempate'] a la vez" in str(e.value)

def test_td4_el_import_del_desempate_sigue_pasando_despues_de_recalcular_el_consenso(camino):
    from tp3_rag.modulo_c import acuerdo as ac

    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    ruta = _sin_columna_de_alcance(_llenar_solo(exportación.ruta, disputados))
    assert _importar(camino, ruta).n_juicios == 4

    destino = ac.ruta_del_consenso(
        camino["rutas"].acuerdo, camino["pool"].pool_id, camino["rubrica"].rubrica_id
    )
    camino["almacen"].escribir_texto(
        destino,
        camino["almacen"].leer_texto(destino).replace("sin_consenso", "desempate"),
    )
    assert ac.pares_en_disputa(
        camino["almacen"], camino["rutas"].acuerdo,
        pool_id=camino["pool"].pool_id, rubrica_id=camino["rubrica"].rubrica_id,
    ) is not None
    segundo = _importar(camino, ruta)
    assert segundo.ya_estaba and segundo.n_juicios == 4
    assert segundo.alcance.tipo == ALCANCE_DESEMPATE

def _primer_desempate_derivado(camino, disputados, anotador="ia-tercero"):
    ruta = _sin_columna_de_alcance(
        _llenar_solo(_exportar(camino, anotador=anotador).ruta, disputados)
    )
    resultado = _importar(camino, ruta)
    assert resultado.alcance.tipo == ALCANCE_DESEMPATE
    assert "**derivado** del consenso" in resultado.alcance.declarado_por
    return resultado, ruta

def test_mx1_un_segundo_desempate_sin_la_columna_ya_no_puede_derivar(camino):
    disputados = _consenso_y_kappa(camino)
    _primer_desempate_derivado(camino, disputados)

    segunda = _exportar(camino, anotador="ia-cuarto", alcance=ALCANCE_DESEMPATE)
    ruta = _sin_columna_de_alcance(_llenar_solo(segunda.ruta, disputados))

    with pytest.raises(ErrorDeArtefacto) as e:
        _importar(camino, ruta)
    mensaje = str(e.value)
    assert "ya hay un desempate congelado" in mensaje
    assert "ia-tercero" in mensaje
    assert "es circular" in mensaje
    assert "--alcance desempate" in mensaje

def test_mx1_el_segundo_desempate_entra_si_el_alcance_viaja_en_la_planilla(camino):
    disputados = _consenso_y_kappa(camino)
    _primer_desempate_derivado(camino, disputados)

    segunda = _exportar(camino, anotador="ia-cuarto", alcance=ALCANCE_DESEMPATE)
    resultado = _importar(camino, _llenar_solo(segunda.ruta, disputados))

    assert resultado.n_juicios == len(disputados)
    assert resultado.alcance.tipo == ALCANCE_DESEMPATE
    assert f"la columna `{ALCANCE}`" in resultado.alcance.declarado_por
    assert "derivado" not in resultado.aporte.procedencia["alcance"]["declarado_por"]

def test_mx1_reimportar_el_mismo_desempate_derivado_sigue_siendo_idempotente(camino):
    disputados = _consenso_y_kappa(camino)
    primero, ruta = _primer_desempate_derivado(camino, disputados)

    segundo = _importar(camino, ruta)

    assert segundo.ya_estaba and not primero.ya_estaba
    assert segundo.alcance.tipo == ALCANCE_DESEMPATE

def test_td4_una_planilla_sin_la_columna_deriva_el_alcance_y_dice_cual_eligio(camino):
    disputados = _consenso_y_kappa(camino)
    exportación = _exportar(camino, anotador="ia-tercero")
    ruta = _sin_columna_de_alcance(_llenar_solo(exportación.ruta, disputados))

    resultado = _importar(camino, ruta)
    assert resultado.n_juicios == 4
    assert resultado.alcance.tipo == ALCANCE_DESEMPATE
    assert "**derivado** del consenso" in resultado.alcance.declarado_por
    assert "derivado" in resultado.aporte.procedencia["alcance"]["declarado_por"]
