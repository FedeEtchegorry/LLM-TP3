from __future__ import annotations

import pytest

from tp3_rag.dominio.rubrica import DecisiónDeA5, Rúbrica, plantilla_de_decision_de_a5
from tp3_rag.modulo_c import acuerdo as ac
from tp3_rag.modulo_c.constructor_de_pool import ConstructorDePool
from tp3_rag.modulo_c.planilla import PlanillaDeAnotación
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
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

POOL_DEL_KAPPA = "0123456789abcdef-piloto"

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
    return {
        "set": set_,
        "manifiesto": manifiesto,
        "pool": pool,
        "rubrica": rubrica,
        "planilla": PlanillaDeAnotación(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba),
        "almacen": almacen_temporal,
        "rutas": rutas_de_prueba,
    }

def _escribir_kappa(
    camino,
    *,
    kappa,
    pool_id=POOL_DEL_KAPPA,
    rubrica_id="piloto",
    hash_del_texto=None,
    razon_de_ausencia=None,
    n=20,
):
    contenido = {
        "artefacto": "Kappa",
        "pool_id": pool_id,
        "rubrica": {
            "rubrica_id": rubrica_id,
            "estado": "completa",
            "hash_del_texto": hash_del_texto or camino["rubrica"].hash_del_texto,
        },
        "kappa": {
            "kappa": kappa,
            "umbral": ac.UMBRAL,
            "n_pares_comparados": n,
            "razon_de_ausencia": razon_de_ausencia,
        },
    }
    return camino["almacen"].escribir_json(
        ac.ruta_del_kappa(camino["rutas"].acuerdo, pool_id, rubrica_id), contenido
    )

def _escribir_decision(camino, **claves):
    datos = {
        "rubrica_id": "piloto",
        "salida": "limitacion_declarada",
        "kappa_leido": 0.419,
        "pool_del_kappa": POOL_DEL_KAPPA,
        "hash_del_texto": camino["rubrica"].hash_del_texto,
        "decidida_por": "el grupo",
        "fecha": "2026-10-30",
        "limitacion": "El ground truth se construyó con una rúbrica de acuerdo moderado.",
        **claves,
    }
    líneas = []
    for clave, valor in datos.items():
        if valor is None:
            continue
        líneas.append(
            f"{clave} = {valor!r}" if isinstance(valor, (int, float)) and not isinstance(
                valor, bool
            ) else f'{clave} = "{valor}"'
        )
    return camino["almacen"].escribir_texto(
        DecisiónDeA5.ruta_de(camino["rutas"].rubricas, datos["rubrica_id"]),
        "\n".join(líneas) + "\n",
    )

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

def test_un_kappa_medido_bajo_el_umbral_no_exporta(camino):
    _escribir_kappa(camino, kappa=0.419)
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    mensaje = str(e.value)
    assert "0.419" in mensaje and "0.6" in mensaje
    assert "refinar" in mensaje and "limitación explícita" in mensaje
    assert "decision-de-a5" in mensaje
    assert "usable por estado" in mensaje and "ambigua por medición" in mensaje

def test_un_kappa_ausente_no_bloquea_y_no_se_lee_como_cero(camino):
    _escribir_kappa(
        camino,
        kappa=None,
        razon_de_ausencia="el acuerdo esperado por azar es 1: los dos usaron un solo grado",
    )
    resultado = _exportar(camino)
    assert resultado.veredicto_de_a5.estado == "ausente"
    leído = ac.kappas_de_la_rubrica(
        camino["almacen"], camino["rutas"].acuerdo, "piloto"
    )[0]
    assert leído.es_ausente
    assert leído.kappa is None, "ausente tiene que llegar como None, no como 0"
    assert not leído.esta_bajo_el_umbral, "ausente no está bajo el umbral ni sobre él"

def test_sin_ninguna_medicion_no_bloquea(camino):
    resultado = _exportar(camino)
    assert resultado.veredicto_de_a5.estado == "no_medido"
    assert resultado.n_filas > 0

def test_un_kappa_sobre_el_umbral_exporta(camino):
    _escribir_kappa(camino, kappa=0.81)
    resultado = _exportar(camino)
    assert resultado.veredicto_de_a5.estado == "sustancial"

def test_ausente_y_bajo_se_distinguen_en_el_mismo_veredicto(camino):
    _escribir_kappa(camino, kappa=0.419, pool_id="0123456789abcdef-piloto")
    _escribir_kappa(
        camino,
        kappa=None,
        pool_id="0123456789abcdef-definitivo",
        razon_de_ausencia="pe = 1",
    )
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    mensaje = str(e.value)
    assert "0123456789abcdef-piloto" in mensaje
    assert "ausente **no es 0**" in mensaje
    assert "0123456789abcdef-definitivo" in mensaje

def test_un_kappa_de_otro_texto_bloquea_por_m5_5_y_no_por_mp_2(camino):
    _escribir_kappa(camino, kappa=0.95, hash_del_texto="0000deadbeef0000")
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    mensaje = str(e.value)
    assert "sobre **otro texto**" in mensaje
    assert "dos textos con un nombre" in mensaje
    assert "decision-de-a5" not in mensaje, "la decisión del acuerdo entre anotadores no es la salida de este caso"

def test_la_limitacion_declarada_desbloquea_y_el_briefing_lleva_el_puntero(camino):
    _escribir_kappa(camino, kappa=0.419)
    _escribir_decision(camino)
    resultado = _exportar(camino)
    assert resultado.veredicto_de_a5.estado == "bajo_con_decision"
    instrucciones = resultado.ruta_de_instrucciones.read_text(encoding="utf-8")
    assert "κ medido bajo el umbral" in instrucciones
    assert "0.419" in instrucciones
    assert "limitacion_declarada" in instrucciones
    assert "piloto-decision-de-a5.toml" in instrucciones
    assert "leela después de anotar" in instrucciones
    assert "acuerdo moderado" not in instrucciones, (
        "la limitación del acuerdo entre anotadores no se copia en el briefing del anotador — "
        "lleva el κ por categoría y la asimetría de métodos, y eso pre-juzga"
    )

def test_ka6_el_mensaje_del_comando_no_promete_la_copia_que_el_briefing_no_hace():
    import inspect

    from tp3_rag.cli import anotar as cli

    fuente = inspect.getsource(cli._exportar)
    assert "lleva **un puntero**" in fuente
    assert "la limitación es correcta en el artefacto" in fuente
    assert "La limitación va **adentro del `-como-anotar.md`**" not in fuente

def test_una_decision_que_cita_otro_kappa_no_cubre(camino):
    _escribir_kappa(camino, kappa=0.312)
    _escribir_decision(camino, kappa_leido=0.419)
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    assert "no cita la medición que autoriza" in str(e.value)

def test_una_decision_sobre_otro_texto_de_la_rubrica_no_cubre(camino):
    _escribir_kappa(camino, kappa=0.419)
    _escribir_decision(camino, hash_del_texto="0000deadbeef0000")
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    assert "el texto se movió después de decidir" in str(e.value)

def test_una_decision_que_cubre_un_pool_no_hereda_el_otro(camino):
    _escribir_kappa(camino, kappa=0.419, pool_id=POOL_DEL_KAPPA)
    _escribir_kappa(camino, kappa=0.21, pool_id="0123456789abcdef-definitivo")
    _escribir_decision(camino)
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    assert "no** menciona" in str(e.value)
    assert "0123456789abcdef-definitivo" in str(e.value)

def test_refinada_no_autoriza_a_exportar_con_esta_rubrica(camino):
    _escribir_kappa(camino, kappa=0.419)
    _escribir_decision(camino, salida="refinada", refinada_en="v2", limitacion=None)
    with pytest.raises(ErrorDeConstrucción) as e:
        _exportar(camino)
    assert "no autoriza a exportar con esta rúbrica" in str(e.value)
    assert "--rubrica v2" in str(e.value)

def test_la_plantilla_de_la_decision_no_valida_hasta_que_la_llenen():
    texto = plantilla_de_decision_de_a5(
        rubrica_id="piloto",
        kappa=0.419,
        pool_id=POOL_DEL_KAPPA,
        hash_del_texto="300fb59a61bb4566",
        umbral=0.6,
    )
    assert "kappa_leido    = 0.419" in texto
    assert '"300fb59a61bb4566"' in texto
    with pytest.raises(ErrorDeArtefacto):
        DecisiónDeA5.desde_texto(texto)

def test_una_limitacion_declarada_sin_limitacion_se_rechaza():
    with pytest.raises(ErrorDeArtefacto) as e:
        DecisiónDeA5.desde_texto(
            'rubrica_id = "piloto"\n'
            'salida = "limitacion_declarada"\n'
            "kappa_leido = 0.419\n"
            'pool_del_kappa = "p"\n'
            'hash_del_texto = "h"\n'
            'decidida_por = "x"\n'
            'fecha = "2026-10-30"\n'
        )
    assert "La limitación *es* la salida" in str(e.value)

def test_refinada_sin_decir_en_cual_se_rechaza():
    with pytest.raises(ErrorDeArtefacto) as e:
        DecisiónDeA5.desde_texto(
            'rubrica_id = "piloto"\nsalida = "refinada"\nkappa_leido = 0.4\n'
            'pool_del_kappa = "p"\nhash_del_texto = "h"\ndecidida_por = "x"\n'
            'fecha = "2026-10-30"\n'
        )
    assert "refinada_en" in str(e.value)

def test_una_tercera_salida_no_existe():
    with pytest.raises(ErrorDeArtefacto) as e:
        DecisiónDeA5.desde_texto(
            'rubrica_id = "piloto"\nsalida = "la dejamos asi"\nkappa_leido = 0.4\n'
            'pool_del_kappa = "p"\nhash_del_texto = "h"\ndecidida_por = "x"\n'
            'fecha = "2026-10-30"\n'
        )
    assert "no es una de las dos" in str(e.value)

def test_una_clave_de_mas_en_la_decision_se_rechaza():
    with pytest.raises(ErrorDeArtefacto) as e:
        DecisiónDeA5.desde_texto(
            'rubrica_id = "piloto"\nsalida = "refinada"\nrefinada_en = "v2"\n'
            'kappa_leido = 0.4\npool_del_kappa = "p"\nhash_del_texto = "h"\n'
            'decidida_por = "x"\nfecha = "2026-10-30"\nurgente = true\n'
        )
    assert "urgente" in str(e.value)

def test_una_decision_que_declara_otra_rubrica_que_su_nombre_se_rechaza(camino):
    camino["almacen"].escribir_texto(
        DecisiónDeA5.ruta_de(camino["rutas"].rubricas, "piloto"),
        'rubrica_id = "v2"\nsalida = "refinada"\nrefinada_en = "v3"\nkappa_leido = 0.4\n'
        'pool_del_kappa = "p"\nhash_del_texto = "h"\ndecidida_por = "x"\n'
        'fecha = "2026-10-30"\n',
    )
    with pytest.raises(ErrorDeArtefacto) as e:
        DecisiónDeA5.leer_si_existe(camino["almacen"], camino["rutas"].rubricas, "piloto")
    assert "no pueden discrepar" in str(e.value)

def test_la_decision_no_se_confunde_con_la_rubrica_al_listar(camino):
    _escribir_decision(camino)
    assert Rúbrica.existentes(camino["almacen"], camino["rutas"].rubricas) == ("piloto",)

def test_el_kappa_bajo_no_bloquea_el_import(camino):
    import csv
    import io

    from tp3_rag.modulo_c.planilla import COLUMNAS_EXPORTADAS

    resultado = _exportar(camino)
    _escribir_kappa(camino, kappa=0.419)

    filas = list(csv.DictReader(io.StringIO(resultado.ruta.read_text(encoding="utf-8"))))
    for n, fila in enumerate(filas):
        fila["grado"] = str(n % 3)
    buffer = io.StringIO()
    escritor = csv.DictWriter(
        buffer, fieldnames=list(COLUMNAS_EXPORTADAS), lineterminator="\n"
    )
    escritor.writeheader()
    escritor.writerows(filas)
    resultado.ruta.write_text(buffer.getvalue(), encoding="utf-8")

    importado = camino["planilla"].importar(
        planilla=resultado.ruta,
        set_de_consultas=camino["set"],
        buscar_pool=lambda _: camino["pool"],
        buscar_rubrica=lambda _: camino["rubrica"],
    )
    assert importado.n_juicios == len(filas)

KAPPA_DEL_PILOTO = (
    RAÍZ_DEL_REPO / "artifacts" / "ground-truth" / "kappa-5b6cba32126fed61-piloto-piloto.json"
)

@pytest.mark.nivel2
@pytest.mark.skipif(
    not KAPPA_DEL_PILOTO.exists(),
    reason="el κ del piloto no está en disco (artifacts/ground-truth lo escribe el camino de anotación)",
)
def test_el_kappa_real_del_piloto_se_clasifica_como_bajo_y_no_como_ausente():
    import json

    from tp3_rag.dominio import ground_truth as gt
    from tp3_rag.nucleo.configuracion import Configuración
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    cfg = Configuración.cargar()
    almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
    rutas = gt.rutas(cfg)
    kappas = ac.kappas_de_la_rubrica(almacén, rutas.acuerdo, "piloto")
    assert kappas, "el κ del piloto está en disco y el lector no lo encontró"
    k = kappas[0]
    en_disco = json.loads(KAPPA_DEL_PILOTO.read_text(encoding="utf-8"))
    assert k.kappa == en_disco["kappa"]["kappa"]
    assert k.esta_bajo_el_umbral and not k.es_ausente
    assert round(k.kappa, 3) == 0.419 and k.umbral == 0.6
    assert k.pool_id.endswith("-piloto")
