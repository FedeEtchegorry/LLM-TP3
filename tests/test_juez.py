from __future__ import annotations

import pytest

from tp3_rag.dominio.binding import Muestreo
from tp3_rag.modulo_c import juez as jz
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.puertos.llm import LLMGrabado

from conftest import SPLIT_DE_PRUEBA, rubrica_de_prueba, set_de_prueba

pytestmark = pytest.mark.nivel1

def _binding(puerto=None, *, modelo="openai/gpt-4o", reintentos=2) -> jz.BindingDelJuez:
    return jz.BindingDelJuez(
        modelo=modelo,
        prompt_id=jz.PROMPT_ID,
        version_del_prompt=jz.VERSIÓN_DEL_PROMPT,
        muestreo=Muestreo(temperature=0.0, top_p=1.0, max_tokens=300, seed=7369),
        esquema_de_salida="json",
        reintentos=reintentos,
        _puerto=puerto,
    )

class _PoolFalso:
    def __init__(self, pares, *, split_id=SPLIT_DE_PRUEBA, pool_id=None):
        from tp3_rag.dominio.pool import ÍtemDePool

        self.items = tuple(
            ÍtemDePool(
                fila=n,
                consulta_id=c,
                artwork_id=a,
                ficha={"obra": f"obra {a}", "artista": "alguien", "cultura": "France"},
            )
            for n, (c, a) in enumerate(pares, 1)
        )
        self.split_id = split_id
        self.pool_id = pool_id or f"{split_id}-definitivo"
        self.estado = "congelado"
        self.procedencia = {
            "cita_al_set_de_consultas": {"identidad": f"{split_id}-grupo-de-prueba"}
        }

def test_bd2_el_binding_del_juez_tiene_rol_y_clave_propios():
    b = _binding()
    assert b.rol == "juez"
    assert b.CLAVE_DE_CONFIGURACIÓN == "bindings.juez"
    assert b.identidad()["rol"] == "juez"
    assert "base_url" not in b.identidad()

def test_bd2_el_binding_del_juez_se_construye_desde_la_configuracion_real(cfg):
    b = jz.BindingDelJuez.desde_configuración(cfg)
    assert b.modelo == "openai/gpt-4o"
    assert b.modelo != cfg["bindings.reranker.modelo"], (
        "🔴 que el juez no sea el reranker / mitigación 3 de la evaluación: *el juez no es el mismo modelo que el "
        "reranker*. Con el mismo modelo en las dos puntas, el κ del acuerdo entre anotadores mide un evaluador "
        "contra sí mismo y el delta `C4−C3` queda inflado por una cantidad desconocida."
    )
    assert len(b.binding_juez) == 16

def test_el_binding_del_juez_no_determina_el_config_id_de_ninguna_fila(cfg):
    from tp3_rag.nucleo.configuracion import NO_DETERMINAN_EL_CONFIG_ID, _cubre

    assert _cubre(NO_DETERMINAN_EL_CONFIG_ID, "bindings.juez.modelo")
    assert not _cubre(NO_DETERMINAN_EL_CONFIG_ID, "bindings.reranker.modelo"), (
        "el del **reranker** sí determina C4: es una etapa de la cadena"
    )

def test_el_prompt_del_binding_y_el_del_modulo_no_pueden_discrepar():
    with pytest.raises(ErrorDelProyecto, match="identidad del binding"):
        jz.BindingDelJuez(
            modelo="openai/gpt-4o",
            prompt_id="otro-prompt",
            version_del_prompt="1",
            muestreo=Muestreo(temperature=0.0, top_p=1.0, max_tokens=300),
        )

def test_un_binding_sin_puerto_no_se_puede_invocar():
    with pytest.raises(ErrorDeConfiguración, match="no tiene puerto"):
        _binding().invocar([{"rol": "usuario", "texto": "hola"}])

def test_el_prompt_no_se_puede_formatear_porque_contiene_el_json_de_la_respuesta():
    assert '{"grado"' in jz.SISTEMA
    with pytest.raises(KeyError):
        jz.SISTEMA.format(rubrica="x", rubrica_id="v1", hash_del_texto="h")
    rubrica = rubrica_de_prueba()
    armado = jz._sistema_con(rubrica=rubrica)
    assert rubrica.texto in armado
    assert not any(h in armado for h in jz.HUECOS_DEL_SISTEMA)

def test_el_juez_ve_lo_MISMO_que_vio_el_anotador_en_la_planilla():
    from tp3_rag.modulo_c.planilla import COLUMNAS as COLUMNAS_DE_LA_PLANILLA

    for campo in jz.CAMPOS_DE_LA_FICHA:
        assert campo in COLUMNAS_DE_LA_PLANILLA, (
            f"el juez le pasa `{campo}` al modelo y la planilla no se lo dio al anotador"
        )
    mensajes = jz.armar_mensajes(
        pedido="con más vida",
        categoria="ambiguo",
        artwork_id="cma:1",
        ficha={"obra": "Paisaje", "artista": "Corot", "cultura": "France"},
        rubrica=rubrica_de_prueba(),
    )
    usuario = mensajes[1]["texto"]
    assert "con más vida" in usuario, "🔴 el pedido va **crudo** (el diseño de la arquitectura)"
    assert "ambiguo" in usuario
    assert "Corot" in usuario

@pytest.mark.parametrize(
    "salida",
    [
        {"grado": 3, "nota": "fuera de la escala"},
        {"grado": "dos", "nota": "no es un entero"},
        {"nota": "sin grado"},
        "esto no es json",
        [0, 1, 2],
    ],
)
def test_una_salida_que_no_satisface_la_escala_se_reintenta_y_termina_sin_grado(salida):
    puerto = LLMGrabado({"juez": [salida]})
    juicio = _juzgar_uno(puerto, reintentos=1)
    assert juicio.estado == jz.ESTADO_SIN_GRADO
    assert juicio.grado is None
    assert juicio.n_intentos == 2

def test_un_par_sin_grado_no_es_un_cero_y_no_entra_al_kappa():
    aporte = _aporte(
        [("q1", "a", 2), ("q1", "b", None)],
    )
    assert aporte.n_sin_grado == 1
    assert aporte.pares == frozenset({("q1", "a")})
    assert aporte.grado_de(("q1", "b")) is None
    assert aporte.por_grado() == {"grado_0": 0, "grado_1": 0, "grado_2": 1}

def test_una_imagen_que_falta_no_es_un_juez_que_no_concuerda():
    puerto = LLMGrabado({"juez": [{"grado": 2, "nota": "n"}]})
    juez = _juez(puerto, de_la_obra=lambda a: (_ for _ in ()).throw(FileNotFoundError(a)))
    pool = _PoolFalso([("c01-ambiguo", "cma:1")])
    resultado = juez.juzgar(pool=pool, rubrica=rubrica_de_prueba(), set_de_consultas=_set())
    juicio = resultado.aporte.juicios[0]
    assert juicio.estado == jz.ESTADO_SIN_GRADO
    assert juicio.n_intentos == 0
    assert "no se pudo leer" in juicio.nota
    assert puerto.llamadas == [], "la llamada **no se hizo**: no hay tokens que contar"

def test_las_dos_imagenes_van_en_orden_y_son_dos():
    puerto = LLMGrabado({"juez": [{"grado": 1, "nota": "n"}]})
    juez = _juez(puerto, de_consulta=lambda c: b"ORIGINAL", de_la_obra=lambda a: b"CANDIDATA")
    juez.juzgar(
        pool=_PoolFalso([("c01-ambiguo", "cma:1")]),
        rubrica=rubrica_de_prueba(),
        set_de_consultas=_set(),
    )
    assert puerto.llamadas[0]["n_imagenes"] == 2

def test_el_aporte_del_juez_no_vive_en_la_carpeta_de_los_juicios_humanos(cfg):
    from tp3_rag.dominio import ground_truth as gt

    rutas = gt.rutas(cfg)
    assert rutas.juicios_juez != rutas.juicios
    assert rutas.juicios_juez.name == "juicios-juez"
    from tp3_rag.dominio.juicios import AporteDeAnotador

    assert not issubclass(jz.AporteDelJuez, AporteDeAnotador)

def test_la_identidad_del_aporte_del_juez_son_TRES_ingredientes():
    ident = jz.identidad_del_aporte_del_juez(
        split_id="abc", binding_juez="def0123456789abc", ordinal=2
    )
    assert ident == "abc-def0123456789abc-2"
    aporte = _aporte([("q1", "a", 1)])
    assert aporte.rubrica_id == "piloto"
    assert aporte.ordinal == 1

def test_dos_corridas_del_juez_son_DOS_aportes_y_el_ordinal_lo_dice(almacen_temporal):
    almacén = almacen_temporal
    puerto = LLMGrabado({"juez": [{"grado": 2, "nota": "n"}]})
    pool = _PoolFalso([("c01-ambiguo", "cma:1")])
    rutas = _rutas_temporales()
    primera = _juez(puerto, almacén=almacén, rutas=rutas).juzgar(
        pool=pool, rubrica=rubrica_de_prueba(), set_de_consultas=_set()
    )
    segunda = _juez(puerto, almacén=almacén, rutas=rutas).juzgar(
        pool=pool, rubrica=rubrica_de_prueba(), set_de_consultas=_set()
    )
    assert primera.aporte.ordinal == 1
    assert segunda.aporte.ordinal == 2
    assert primera.aporte.identidad != segunda.aporte.identidad
    assert almacén.existe(primera.ruta.relative_to(almacén.raíz))

def test_el_aporte_va_y_vuelve_de_su_csv_con_la_procedencia_adentro(almacen_temporal):
    almacén = almacen_temporal
    rutas = _rutas_temporales()
    puerto = LLMGrabado({"juez": [{"grado": 2, "nota": "cumple el eje"}]})
    resultado = _juez(puerto, almacén=almacén, rutas=rutas).juzgar(
        pool=_PoolFalso([("c01-ambiguo", "cma:1"), ("c01-visual", "cma:2")]),
        rubrica=rubrica_de_prueba(),
        set_de_consultas=_set(),
    )
    leído = jz.AporteDelJuez.leer(almacén, rutas.juicios_juez, resultado.aporte.identidad)
    assert leído.identidad == resultado.aporte.identidad
    assert len(leído.juicios) == 2
    assert leído.pool_id == f"{SPLIT_DE_PRUEBA}-definitivo"
    assert leído.modelo == "openai/gpt-4o"
    assert leído.hash_de_la_rubrica == rubrica_de_prueba().hash_del_texto
    assert leído.procedencia["libro_mayor"]["juez"]["n_llamadas"] == 2

def test_un_aporte_parcial_declara_que_es_parcial(almacen_temporal):
    almacén = almacen_temporal
    rutas = _rutas_temporales()
    puerto = LLMGrabado({"juez": [{"grado": 0, "nota": "n"}]})
    resultado = _juez(puerto, almacén=almacén, rutas=rutas).juzgar(
        pool=_PoolFalso([("c01-ambiguo", "cma:1"), ("c01-visual", "cma:2")]),
        rubrica=rubrica_de_prueba(),
        set_de_consultas=_set(),
        limite=1,
    )
    conteos = resultado.aporte.procedencia["conteos"]
    assert conteos["n_juicios"] == 1
    assert conteos["n_items_del_pool"] == 2
    assert "parcial" in conteos["alcance"]

def test_el_mismo_par_dos_veces_en_un_aporte_se_rechaza():
    with pytest.raises(ErrorDeArtefacto, match="dos veces"):
        _aporte([("q1", "a", 1), ("q1", "a", 2)])

def test_un_aporte_sin_juicios_se_rechaza():
    with pytest.raises(ErrorDeArtefacto, match="ni un juicio"):
        _aporte([])

def test_una_rubrica_plantilla_no_es_un_instrumento():
    puerto = LLMGrabado({"juez": [{"grado": 2, "nota": "n"}]})
    with pytest.raises(ErrorDelProyecto):
        _juez(puerto).juzgar(
            pool=_PoolFalso([("c01-ambiguo", "cma:1")]),
            rubrica=rubrica_de_prueba(estado="plantilla"),
            set_de_consultas=_set(),
        )

def test_un_consulta_id_que_no_resuelve_es_E4_y_no_un_juicio_huerfano():
    puerto = LLMGrabado({"juez": [{"grado": 2, "nota": "n"}]})
    with pytest.raises(ErrorDelProyecto, match="la referencia que resuelve"):
        _juez(puerto).juzgar(
            pool=_PoolFalso([("no-existe", "cma:1")]),
            rubrica=rubrica_de_prueba(),
            set_de_consultas=_set(),
        )

def test_los_tres_kappa_salen_y_los_estratos_no_son_disjuntos():
    aporte = _aporte([("q1", "a", 2), ("q1", "b", 0), ("q2", "c", 1), ("q2", "d", 2)])
    consenso = {("q1", "a"): 2, ("q1", "b"): 1, ("q2", "c"): 1, ("q2", "d"): 0}
    tres = jz.kappas_del_juez(
        aporte,
        consenso=consenso,
        aportes_humanos={
            "ia-claude": {("q1", "a"): 2, ("q1", "b"): 0, ("q2", "c"): 1, ("q2", "d"): 2},
            "ia-codex": {("q1", "a"): 0, ("q1", "b"): 0, ("q2", "c"): 0, ("q2", "d"): 0},
        },
        origen_por_par={
            ("q1", "a"): ("C0", "C4"),
            ("q1", "b"): ("C4",),
            ("q2", "c"): ("C0",),
            ("q2", "d"): ("C4",),
        },
        rubrica_id="piloto",
    )
    assert set(tres["contra_cada_anotador"]) == {"ia-claude", "ia-codex"}
    assert set(tres["por_configuracion_de_origen"]) == {"C0", "C4"}
    assert sum(k.n for k in tres["por_configuracion_de_origen"].values()) > len(consenso)
    assert "no son disjuntos" in tres["nota_de_los_estratos"]
    assert tres["contra_cada_anotador"]["ia-claude"].kappa == pytest.approx(1.0)
    assert tres["contra_cada_anotador"]["ia-codex"].kappa < 1.0

def test_el_adaptador_del_consenso_NO_es_un_aporte_de_anotador():
    from tp3_rag.dominio.juicios import AporteDeAnotador

    adaptador = jz._ComoAporte(anotador="consenso", grados={("q", "a"): 1})
    assert not isinstance(adaptador, AporteDeAnotador)
    assert not hasattr(adaptador, "identidad")
    assert not hasattr(adaptador, "procedencia")

def test_el_kappa_del_juez_se_computa_solo_sobre_los_pares_que_juzgó():
    aporte = _aporte([("q1", "a", 2), ("q1", "b", None)])
    tres = jz.kappas_del_juez(
        aporte,
        consenso={("q1", "a"): 2, ("q1", "b"): 1, ("q1", "c"): 0},
        aportes_humanos={},
        origen_por_par={},
        rubrica_id="piloto",
    )
    k = tres["contra_el_consenso"]
    assert k.n == 1
    assert len(k.solo_uno["consenso"]) == 2

def _set():
    return set_de_prueba(split_id=SPLIT_DE_PRUEBA)

def _rutas_temporales():
    from pathlib import Path

    from tp3_rag.dominio import ground_truth as gt

    raíz = Path("gt")
    return gt.RutasDelGroundTruth(
        raiz=raíz,
        consultas=raíz / "consultas",
        rubricas=raíz / "rubricas",
        pool=raíz / "pool",
        juicios=raíz / "juicios",
        planillas=Path("planillas"),
        juicios_juez=raíz / "juicios-juez",
    )

def _juez(puerto, *, almacén=None, rutas=None, de_consulta=None, de_la_obra=None, reintentos=2):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    if almacén is None:
        almacén = AlmacenamientoEnArchivos(Path(tempfile.mkdtemp()))
    if rutas is None:
        rutas = _rutas_temporales()
    return jz.Juez(
        SimpleNamespace(capas=("config/default.toml",)),
        almacén=almacén,
        rutas=rutas,
        binding=_binding(puerto, reintentos=reintentos),
        bytes_de_la_imagen_de_consulta=de_consulta or (lambda c: b"imagen-de-consulta"),
        bytes_de_la_imagen_de_la_obra=de_la_obra or (lambda a: b"imagen-de-la-obra"),
    )

def _juzgar_uno(puerto, *, reintentos=2):
    from tp3_rag.dominio.pool import ÍtemDePool

    juez = _juez(puerto, reintentos=reintentos)
    item = ÍtemDePool(fila=1, consulta_id="c01-ambiguo", artwork_id="cma:1", ficha={})
    consulta = next(c for c in _set().consultas())
    return juez._juzgar_uno(item, consulta=consulta, rubrica=rubrica_de_prueba())

def _aporte(pares, *, ordinal=1):
    juicios = tuple(
        jz.JuicioAutomático(
            consulta_id=c,
            artwork_id=a,
            grado=g,
            estado=jz.ESTADO_JUZGADO if g is not None else jz.ESTADO_SIN_GRADO,
        )
        for c, a, g in pares
    )
    binding_juez = _binding().binding_juez
    aporte = jz.AporteDelJuez(
        procedencia={
            "identidad": jz.identidad_del_aporte_del_juez(
                split_id=SPLIT_DE_PRUEBA, binding_juez=binding_juez, ordinal=ordinal
            ),
            "split_id": SPLIT_DE_PRUEBA,
            "binding_juez": binding_juez,
            "ordinal": ordinal,
            "pool_id": f"{SPLIT_DE_PRUEBA}-definitivo",
            "rubrica_id": "piloto",
            "binding": _binding().a_json(),
            "conteos": {"n_juicios": len(juicios)},
        },
        juicios=juicios,
    )
    aporte.validar()
    return aporte

def test_solo_pares_juzga_solo_esos_y_lo_declara_parcial(almacen_temporal):
    almacén = almacen_temporal
    rutas = _rutas_temporales()
    puerto = LLMGrabado({"juez": [{"grado": 1, "nota": "n"}]})
    resultado = _juez(puerto, almacén=almacén, rutas=rutas).juzgar(
        pool=_PoolFalso(
            [("c01-ambiguo", "cma:1"), ("c01-visual", "cma:2"), ("c01-visual", "cma:3")]
        ),
        rubrica=rubrica_de_prueba(),
        set_de_consultas=_set(),
        solo_pares=frozenset({("c01-visual", "cma:2")}),
    )
    assert [j.par for j in resultado.aporte.juicios] == [("c01-visual", "cma:2")]
    conteos = resultado.aporte.procedencia["conteos"]
    assert conteos["alcance"].startswith("parcial (1 de 3)")
