from __future__ import annotations

import pytest

from tp3_rag.dominio.consultas import CATEGORÍAS
from tp3_rag.dominio.pool import Pool
from tp3_rag.modulo_c.constructor_de_pool import (
    FILAS_DE_LA_TABLA,
    N_CONSULTAS_DEL_PILOTO,
    ConstructorDePool,
    ReglaDeSelección,
    ReglaDeTodasLasConsultas,
)
from tp3_rag.nucleo import identidad
from tp3_rag.modulo_c import metricas as met
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConstrucción

from conftest import (
    CONFIG_DE_PRUEBA,
    CONFIG_IDS_DE_PRUEBA,
    FIXTURES,
    corrida_de_prueba,
    corridas_de_las_cinco_filas,
    corridas_de_prueba,
    manifiesto_de_prueba,
    set_de_prueba,
    top5_de_prueba,
)

pytestmark = pytest.mark.nivel1

@pytest.fixture
def ingredientes():
    set_ = set_de_prueba()
    return set_, manifiesto_de_prueba(), corridas_de_prueba(set_)

def _derivar(cfg, almacén, rutas, ingredientes, **kwargs):
    set_, manifiesto, corridas = ingredientes
    return ConstructorDePool(cfg, almacén=almacén, rutas=rutas).derivar(
        designacion=kwargs.pop("designacion", "piloto"),
        configuracion_esperada=kwargs.pop("configuracion_esperada", "C1"),
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto,
        corridas=kwargs.pop("corridas", corridas),
        **kwargs,
    )

def test_la_regla_elige_una_por_categoria_mas_una_repetida():
    set_ = set_de_prueba()
    elegidas = ReglaDeSelección().elegir(set_, semilla=7369)

    assert len(elegidas) == N_CONSULTAS_DEL_PILOTO == 4
    categorías = [c.categoria for c in elegidas]
    for categoría in CATEGORÍAS:
        assert categoría in categorías, f"falta una consulta de {categoría}"
    assert categorías.count("ambiguo") == 2

def test_la_regla_es_determinista_y_se_puede_re_derivar():
    set_ = set_de_prueba()
    una = ReglaDeSelección().elegir(set_, semilla=7369)
    otra = ReglaDeSelección().elegir(set_, semilla=7369)
    con_otra_semilla = ReglaDeSelección().elegir(set_, semilla=1)

    assert [c.consulta_id for c in una] == [c.consulta_id for c in otra]
    assert [c.consulta_id for c in una] != [c.consulta_id for c in con_otra_semilla]

def test_el_ranking_es_el_hash_declarado_y_no_el_orden_del_archivo():
    set_ = set_de_prueba()
    elegidas = ReglaDeSelección().elegir(set_, semilla=7369)
    visual = next(c for c in elegidas if c.categoria == "visual_cromatico")
    candidatas = [c for c in set_.consultas() if c.categoria == "visual_cromatico"]
    menor = min(candidatas, key=lambda c: identidad.rango_de_seleccion(7369, c.consulta_id))

    assert visual.consulta_id == menor.consulta_id

def test_las_cuatro_consultas_son_de_cuatro_imagenes_distintas():
    elegidas = ReglaDeSelección().elegir(set_de_prueba(), semilla=7369)
    assert len({c.artwork_id for c in elegidas}) == 4

def test_se_puede_aflojar_la_restriccion_de_imagenes_y_queda_declarado():
    regla = ReglaDeSelección(imagenes_distintas=False)
    elegidas = regla.elegir(set_de_prueba(), semilla=7369)
    declarado = regla.a_json(semilla=7369, elegidas=elegidas)

    assert declarado["imagenes_distintas"] is False
    assert declarado["por_que_imagenes_distintas"]

def test_la_categoria_repetida_tiene_que_ser_una_de_las_tres():
    with pytest.raises(ErrorDeConstrucción, match="categoria_repetida"):
        ReglaDeSelección(categoria_repetida="inventada")

def test_las_consultas_de_ejemplo_no_entran_al_pool():
    set_ = set_de_prueba()
    for img in set_.contenido["imagenes"]:
        for c in img["consultas"]:
            c["es_ejemplo"] = True
    with pytest.raises(ErrorDeConstrucción, match="es_ejemplo"):
        ReglaDeSelección().elegir(set_, semilla=7369)

def test_la_regla_declarada_viaja_adentro_del_pool(cfg, almacen_temporal, rutas_de_prueba,
                                                   ingredientes):
    resultado = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)
    regla = resultado.pool.regla

    assert regla["n_consultas"] == 4
    assert regla["categoria_repetida"] == "ambiguo"
    assert regla["semilla"] == 7369
    assert "ronda piloto" in regla["fuente"]
    assert len(regla["elegidas"]) == 4
    assert regla["por_que_esa_categoria_se_repite"]
    assert "sha256" in regla["ranking_dentro_de_la_categoria"]

def test_la_identidad_es_split_mas_designacion_y_el_estado_va_adentro(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    pool = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes).pool

    assert pool.pool_id == f"{pool.split_id}-piloto"
    assert pool.estado == "piloto"
    assert not pool.es_congelado
    assert "piloto" in pool.procedencia["nota_de_estado"]

def test_derivar_dos_veces_no_reescribe(cfg, almacen_temporal, rutas_de_prueba, ingredientes):
    primero = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)
    segundo = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)

    assert not primero.ya_estaba
    assert segundo.ya_estaba
    assert segundo.pool.pares == primero.pool.pares

def test_dos_designaciones_del_mismo_split_conviven(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes, designacion="piloto")
    otro = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes, designacion="piloto-2")

    assert otro.pool.pool_id.endswith("-piloto-2")
    assert set(Pool.del_split(almacen_temporal, rutas_de_prueba.pool, otro.pool.split_id)) == {
        f"{otro.pool.split_id}-piloto", f"{otro.pool.split_id}-piloto-2"
    }

def test_un_pool_congelado_de_una_sola_configuracion_se_rechaza(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    with pytest.raises(ErrorDeConstrucción, match="sesgaría"):
        _derivar(
            cfg, almacen_temporal, rutas_de_prueba, ingredientes,
            designacion="definitivo", estado="congelado",
        )

def test_un_set_plantilla_no_puede_derivar_un_pool(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    set_.contenido["estado"] = "plantilla"
    with pytest.raises(ErrorDeConstrucción, match="no es usable para medir"):
        ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
            designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
            set_de_consultas=set_,
            manifiesto=manifiesto_de_prueba(),
            corridas=corridas_de_prueba(set_de_prueba()),
        )

def test_las_corridas_degradadas_no_entran(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    corridas = corridas_de_prueba(
        set_, degradaciones=({"etapa": "generador", "planificada": True},)
    )
    with pytest.raises(ErrorDeConstrucción, match="no hay corrida \\*\\*limpia\\*\\*"):
        ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
            designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
            set_de_consultas=set_,
            manifiesto=manifiesto_de_prueba(),
            corridas=corridas,
        )

def test_se_toma_el_ultimo_ordinal_y_queda_citado(
    cfg, almacen_temporal, rutas_de_prueba
):
    from conftest import corrida_de_prueba

    set_ = set_de_prueba()
    corridas = corridas_de_prueba(set_)
    elegida = ReglaDeSelección().elegir(set_, semilla=7369)[0]
    segunda = corrida_de_prueba(
        consulta_id=elegida.consulta_id,
        top5=[f"cma:{n}" for n in range(150, 155)],
        ordinal=2,
    )
    resultado = ConstructorDePool(
        cfg, almacén=almacen_temporal, rutas=rutas_de_prueba
    ).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=corridas + [segunda],
    )
    por_consulta = resultado.pool.por_consulta()
    assert all(len(items) == 5 for items in por_consulta.values())
    origen = resultado.pool.origen_de(por_consulta[elegida.consulta_id][0].par)
    assert origen["corrida_id"] == segunda.corrida_id
    assert origen["rank"] in range(1, 6)
    assert resultado.corridas_descartadas["ordinales_viejos"] == 1
    assert {i.artwork_id for i in por_consulta[elegida.consulta_id]} == {
        f"cma:{n}" for n in range(150, 155)
    }

def test_una_corrida_sin_ficha_no_entra(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    with pytest.raises(ErrorDeConstrucción, match="no hay corrida"):
        ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
            designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
            set_de_consultas=set_,
            manifiesto=manifiesto_de_prueba(),
            corridas=corridas_de_prueba(set_, sin_ficha=True),
        )

def test_la_regla_no_se_ajusta_a_lo_que_hay_en_disco(cfg, almacen_temporal, rutas_de_prueba):
    set_ = set_de_prueba()
    elegidas = ReglaDeSelección().elegir(set_, semilla=7369)
    corridas = [
        c for c in corridas_de_prueba(set_)
        if c.consulta_id != elegidas[0].consulta_id
    ]
    with pytest.raises(ErrorDeConstrucción, match=elegidas[0].consulta_id):
        ConstructorDePool(cfg, almacén=almacen_temporal, rutas=rutas_de_prueba).derivar(
            designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
            set_de_consultas=set_,
            manifiesto=manifiesto_de_prueba(),
            corridas=corridas,
        )

def test_el_csv_no_lleva_ni_la_configuracion_ni_el_rank(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    resultado = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)
    csv = resultado.ruta.read_text(encoding="utf-8")
    encabezado = csv.splitlines()[0]

    assert "rank" not in encabezado
    assert "configuracion" not in encabezado
    assert "C1" not in csv
    origen = resultado.pool.origen_de(resultado.pool.items[0].par)
    assert origen["configuracion"] == "C1"
    assert 1 <= origen["rank"] <= 5

def test_el_pool_se_lee_entero_o_no_se_lee(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    resultado = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)
    resultado.ruta_de_procedencia.unlink()
    with pytest.raises(ErrorDeArtefacto):
        Pool.leer(almacen_temporal, rutas_de_prueba.pool, resultado.pool.pool_id)

def test_el_pedido_no_se_copia_al_pool_y_la_ficha_si(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    resultado = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes)
    csv = resultado.ruta.read_text(encoding="utf-8")

    assert "pedido" not in csv.splitlines()[0]
    assert "obra" in csv.splitlines()[0]
    assert resultado.pool.items[0].ficha["obra"]

def test_los_veinte_items_son_cuatro_por_cinco_sin_repetir_par(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    pool = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes).pool

    assert len(pool.items) == 20
    assert len(pool.pares) == 20
    assert len(pool.consultas) == 4
    assert sorted(i.fila for i in pool.items) == list(range(1, 21))
    assert pool.procedencia["conteos"]["n_juicios_por_anotador"] == 20

def test_m10_dos_config_id_con_la_misma_etiqueta_no_entran_al_mismo_pool(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    congelada = corridas_de_prueba(set_, config_id=CONFIG_DE_PRUEBA)
    otro_brazo = corridas_de_prueba(set_, config_id="0" * 16, ordinal=9)
    resultado = ConstructorDePool(
        cfg, almacén=almacen_temporal, rutas=rutas_de_prueba
    ).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=list(otro_brazo) + list(congelada),
    )
    assert resultado.corridas_descartadas["otro_config_id"] == len(otro_brazo)
    citados = {o["corrida_id"] for o in resultado.pool.procedencia["origen"]}
    assert citados <= {c.corrida_id for c in congelada}
    assert citados.isdisjoint({c.corrida_id for c in otro_brazo})

def test_m10_una_adhoc_de_la_configuracion_congelada_no_entra_al_pool(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    corridas = list(corridas_de_prueba(set_, config_id=CONFIG_DE_PRUEBA))
    corridas.append(
        corrida_de_prueba(
            consulta_id="78a9b1ffd092486d",
            top5=[f"cma:{100 + k}" for k in range(5)],
            config_id=CONFIG_DE_PRUEBA,
            procedencia="ad_hoc",
        )
    )
    resultado = ConstructorDePool(
        cfg, almacén=almacen_temporal, rutas=rutas_de_prueba
    ).derivar(
        designacion="piloto",
        config_id_esperado=CONFIG_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=corridas,
    )
    assert resultado.corridas_descartadas["ad_hoc"] == 1
    assert all(i.consulta_id != "78a9b1ffd092486d" for i in resultado.pool.items)

def _congelado(cfg, almacén, rutas, *, set_=None, corridas=None, **kwargs):
    set_ = set_ or set_de_prueba()
    return ConstructorDePool(cfg, almacén=almacén, rutas=rutas).derivar(
        designacion=kwargs.pop("designacion", "definitivo"),
        estado="congelado",
        config_ids_esperados=CONFIG_IDS_DE_PRUEBA,
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=corridas if corridas is not None else corridas_de_las_cinco_filas(set_),
        **kwargs,
    )

def test_m11_el_pool_congelado_es_la_union_deduplicada_de_las_cinco(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    resultado = _congelado(cfg, almacen_temporal, rutas_de_prueba, set_=set_)
    pool = resultado.pool
    conteos = pool.procedencia["conteos"]

    por_fila = {
        fila: {
            (c.consulta_id, aid)
            for n, c in enumerate(set_.consultas())
            for aid in dict.fromkeys(top5_de_prueba(fila, n))
        }
        for fila in FILAS_DE_LA_TABLA
    }
    unión = set().union(*por_fila.values())

    assert pool.es_congelado
    assert pool.nombres_de_configuracion() == FILAS_DE_LA_TABLA
    assert len(pool.items) == len(unión) == conteos["n_items"]
    assert pool.pares == unión
    assert conteos["n_consultas"] == 30
    assert conteos["aporte_por_configuracion"] == {
        fila: len(pares) for fila, pares in por_fila.items()
    }
    assert conteos["suma_de_los_top5_sin_deduplicar"] == sum(
        len(p) for p in por_fila.values()
    )
    assert conteos["deduplicados"] == (
        conteos["suma_de_los_top5_sin_deduplicar"] - conteos["n_items"]
    )
    assert conteos["deduplicados"] > 0

def test_m11_los_exclusivos_son_los_que_trajo_solo_esa_configuracion(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    consulta = next(iter(set_.consultas())).consulta_id
    compartidas = [f"cma:{100 + k}" for k in range(4)]
    corridas = [
        corrida_de_prueba(
            consulta_id=consulta, top5=compartidas + ["cma:150"],
            configuracion="C1", config_id=CONFIG_IDS_DE_PRUEBA["C1"], alfa_congelado=True,
        ),
        corrida_de_prueba(
            consulta_id=consulta, top5=compartidas + ["cma:151"],
            configuracion="C4", config_id=CONFIG_IDS_DE_PRUEBA["C4"], alfa_congelado=True,
        ),
    ]
    pool = _congelado(
        cfg, almacen_temporal, rutas_de_prueba,
        set_=set_, corridas=corridas,
        configuraciones_esperadas=("C1", "C4"),
        regla=_ReglaDeUnaConsulta(consulta),
    ).pool
    conteos = pool.procedencia["conteos"]

    assert conteos["n_items"] == 6
    assert conteos["aporte_por_configuracion"] == {"C1": 5, "C4": 5}
    assert conteos["deduplicados"] == 4
    assert conteos["exclusivos_por_configuracion"] == {"C1": 1, "C4": 1}
    assert pool.exclusivos_por_configuracion() == {"C1": 1, "C4": 1}

class _ReglaDeUnaConsulta(ReglaDeTodasLasConsultas):
    def __init__(self, consulta_id: str) -> None:
        super().__init__()
        object.__setattr__(self, "_consulta_id", consulta_id)

    def elegir(self, set_de_consultas, *, semilla):
        return (set_de_consultas.consulta(self._consulta_id),)

def test_m11_un_item_declara_todas_las_configuraciones_que_lo_trajeron(
    cfg, almacen_temporal, rutas_de_prueba
):
    pool = _congelado(cfg, almacen_temporal, rutas_de_prueba).pool

    compartido = next(
        i for i in pool.items if len(pool.origenes_de(i.par)) == len(FILAS_DE_LA_TABLA)
    )
    orígenes = pool.origenes_de(compartido.par)

    assert [o["configuracion"] for o in orígenes] == list(FILAS_DE_LA_TABLA)
    assert all(1 <= o["rank"] <= 5 for o in orígenes)
    assert all(o["corrida_id"] for o in orígenes)
    assert pool.origen_de(compartido.par) == orígenes[0]
    por_fila = {c["configuracion"]: set(c["corridas"]) for c in pool.configuraciones}
    assert all(o["corrida_id"] in por_fila[o["configuracion"]] for o in orígenes)

def test_m11_el_csv_sigue_siendo_ciego_con_cinco_configuraciones(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado = _congelado(cfg, almacen_temporal, rutas_de_prueba)
    csv = resultado.ruta.read_text(encoding="utf-8")
    encabezado = csv.splitlines()[0]

    assert "rank" not in encabezado
    assert "configuracion" not in encabezado
    for fila in FILAS_DE_LA_TABLA:
        assert fila not in csv, f"el CSV ciego nombra {fila}"
    for config_id in CONFIG_IDS_DE_PRUEBA.values():
        assert config_id not in csv
    json = resultado.ruta_de_procedencia.read_text(encoding="utf-8")
    assert all(fila in json for fila in FILAS_DE_LA_TABLA)
    assert '"rank"' in json

def test_m11_el_ultimo_ordinal_se_toma_por_config_id_y_no_entre_filas(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    corridas = list(corridas_de_las_cinco_filas(set_))
    extra = [
        corrida_de_prueba(
            consulta_id=c.consulta_id,
            top5=[f"cma:{100 + (n + k) % 40}" for k in range(5)],
            configuracion="C3", config_id=CONFIG_IDS_DE_PRUEBA["C3"],
            ordinal=ordinal, alfa_congelado=True,
        )
        for ordinal in (2, 3)
        for n, c in enumerate(set_.consultas())
    ]
    pool = _congelado(
        cfg, almacen_temporal, rutas_de_prueba, set_=set_, corridas=corridas + extra
    ).pool

    elegidas = {c["configuracion"]: set(c["corridas"]) for c in pool.configuraciones}
    assert all(len(v) == 30 for v in elegidas.values())
    assert all(cid.endswith("-3") for cid in elegidas["C3"])
    assert all(cid.endswith("-1") for cid in elegidas["C4"])

def test_m11_una_corrida_pre_congelamiento_no_forma_un_pool_congelado(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    corridas = list(corridas_de_las_cinco_filas(set_))
    provisoria = corrida_de_prueba(
        consulta_id=next(iter(set_.consultas())).consulta_id,
        top5=top5_de_prueba("C3", 0),
        configuracion="C3", config_id=CONFIG_IDS_DE_PRUEBA["C3"],
        ordinal=2, alfa_congelado=False,
    )
    with pytest.raises(ErrorDeConstrucción, match="alfa_congelado"):
        _congelado(
            cfg, almacen_temporal, rutas_de_prueba,
            set_=set_, corridas=corridas + [provisoria],
        )

def test_m11_un_pool_piloto_si_admite_corridas_sin_alfa_congelado(
    cfg, almacen_temporal, rutas_de_prueba, ingredientes
):
    pool = _derivar(cfg, almacen_temporal, rutas_de_prueba, ingredientes).pool

    assert pool.estado == "piloto"
    assert pool.configuraciones[0]["alfa_congelado"] is False

def test_m11_si_falta_una_consulta_en_una_sola_fila_es_error(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    faltante = next(iter(set_.consultas())).consulta_id
    corridas = [
        c for c in corridas_de_las_cinco_filas(set_)
        if not (c.consulta_id == faltante and c.config_id == CONFIG_IDS_DE_PRUEBA["C2"])
    ]
    with pytest.raises(ErrorDeConstrucción, match=f"C2.*{faltante}|{faltante}.*C2"):
        _congelado(cfg, almacen_temporal, rutas_de_prueba, set_=set_, corridas=corridas)

def test_m11_una_corrida_degradada_de_una_fila_tampoco_entra(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    corridas = []
    for c in corridas_de_las_cinco_filas(set_):
        degradar = c.config_id == CONFIG_IDS_DE_PRUEBA["C4"] and c.consulta_id.startswith(
            "c01"
        )
        corridas.append(
            corrida_de_prueba(
                consulta_id=c.consulta_id,
                top5=c.top5,
                configuracion=(c.contenido["pipeline"]["configuracion"]),
                config_id=c.config_id,
                alfa_congelado=True,
                degradaciones=({"etapa": "reranker", "planificada": False},)
                if degradar
                else (),
            )
        )
    with pytest.raises(ErrorDeConstrucción, match="C4"):
        _congelado(cfg, almacen_temporal, rutas_de_prueba, set_=set_, corridas=corridas)

def test_m11_la_ficha_no_puede_depender_de_que_configuracion_trajo_la_obra(
    cfg, almacen_temporal, rutas_de_prueba
):
    set_ = set_de_prueba()
    consulta = next(iter(set_.consultas())).consulta_id
    top5 = [f"cma:{100 + k}" for k in range(5)]
    corridas = [
        corrida_de_prueba(
            consulta_id=consulta, top5=top5, configuracion="C1",
            config_id=CONFIG_IDS_DE_PRUEBA["C1"], alfa_congelado=True,
        ),
        corrida_de_prueba(
            consulta_id=consulta, top5=top5, configuracion="C4",
            config_id=CONFIG_IDS_DE_PRUEBA["C4"], alfa_congelado=True,
            ficha_extra={"culture": "otra cultura"},
        ),
    ]
    with pytest.raises(ErrorDeConstrucción, match="cultura"):
        _congelado(
            cfg, almacen_temporal, rutas_de_prueba,
            set_=set_, corridas=corridas,
            configuraciones_esperadas=("C1", "C4"),
            regla=_ReglaDeUnaConsulta(consulta),
        )

def test_m11_la_regla_de_todas_las_consultas_toma_las_30_y_excluye_los_ejemplos():
    set_ = set_de_prueba()
    set_.contenido["imagenes"][0]["consultas"][0]["es_ejemplo"] = True
    elegidas = ReglaDeTodasLasConsultas().elegir(set_, semilla=7369)
    declarado = ReglaDeTodasLasConsultas().a_json(semilla=7369, elegidas=elegidas)

    assert len(elegidas) == 29
    assert [c.consulta_id for c in elegidas] == sorted(c.consulta_id for c in elegidas)
    assert declarado["seleccion_de_consultas"] == "todas"
    assert declarado["n_consultas"] == 29
    assert "es_ejemplo" in declarado["excluye"]
    assert "sha256" in declarado["mezcla_dentro_de_la_consulta"]
    assert declarado["semilla"] == 7369

def test_m11_el_pool_congelado_se_mezcla_dentro_de_cada_consulta(
    cfg, almacen_temporal, rutas_de_prueba
):
    pool = _congelado(cfg, almacen_temporal, rutas_de_prueba).pool

    por_consulta = pool.por_consulta()
    for items in por_consulta.values():
        filas = [i.fila for i in items]
        assert filas == list(range(min(filas), max(filas) + 1))
    desordenada = any(
        [pool.origen_de(i.par)["rank"] for i in items]
        != sorted(pool.origen_de(i.par)["rank"] for i in items)
        for items in por_consulta.values()
    )
    assert desordenada, "ninguna consulta quedó mezclada: el orden sería el del ranking"

def test_m11_los_conteos_de_la_union_se_vuelven_a_derivar_al_leer(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado = _congelado(cfg, almacen_temporal, rutas_de_prueba)
    pool_id = resultado.pool.pool_id
    ruta = resultado.ruta_de_procedencia
    import json as _json

    contenido = _json.loads(ruta.read_text(encoding="utf-8"))
    contenido["conteos"]["exclusivos_por_configuracion"]["C4"] += 1
    ruta.write_text(_json.dumps(contenido), encoding="utf-8")

    with pytest.raises(ErrorDeArtefacto, match="exclusivos_por_configuracion"):
        Pool.leer(almacen_temporal, rutas_de_prueba.pool, pool_id)

def test_m11_un_congelado_con_una_sola_configuracion_no_se_puede_ni_leer(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado = _congelado(cfg, almacen_temporal, rutas_de_prueba)
    ruta = resultado.ruta_de_procedencia
    import json as _json

    contenido = _json.loads(ruta.read_text(encoding="utf-8"))
    contenido["configuraciones"] = contenido["configuraciones"][:1]
    ruta.write_text(_json.dumps(contenido), encoding="utf-8")

    with pytest.raises(ErrorDeArtefacto, match="sesgaría"):
        Pool.leer(almacen_temporal, rutas_de_prueba.pool, resultado.pool.pool_id)

def test_m11_el_presupuesto_de_anotacion_sale_del_pool_y_no_de_una_proyeccion(
    cfg, almacen_temporal, rutas_de_prueba
):
    pool = _congelado(cfg, almacen_temporal, rutas_de_prueba).pool
    conteos = pool.procedencia["conteos"]

    assert conteos["n_juicios_por_anotador"] == len(pool.items)
    assert conteos["duplicado_del_20_por_ciento_de_a5"] == round(len(pool.items) * 0.20)
    assert conteos["minutos_por_anotador_al_ritmo_de_docs_05"] == round(
        len(pool.items) * 15 / 60
    )
    assert set(conteos["items_del_top5_sin_juzgar_por_configuracion"].values()) == {0}
    assert "no un subconjunto elegido" in conteos["nota_del_duplicado"]
    assert "100 %" in conteos["nota_del_duplicado"]

def test_el_pool_sellado_del_piloto_sigue_leyendose():
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    almacén = AlmacenamientoEnArchivos(FIXTURES)
    pool = Pool.leer(almacén, "pool-sellado", "5b6cba32126fed61-piloto")

    assert pool.estado == "piloto"
    assert len(pool.items) == 20
    assert pool.procedencia["version_del_pool"] == "1"
    assert len(pool.origenes_de(pool.items[0].par)) == 1
    assert pool.aporte_por_configuracion() == {"C1": 20}
    assert pool.exclusivos_por_configuracion() == {"C1": 20}

def _pool_congelado(cfg, almacén, rutas):
    set_ = set_de_prueba()
    return ConstructorDePool(cfg, almacén=almacén, rutas=rutas).derivar(
        designacion="definitivo",
        set_de_consultas=set_,
        manifiesto=manifiesto_de_prueba(),
        corridas=corridas_de_las_cinco_filas(set_),
        estado="congelado",
        configuraciones_esperadas=list(FILAS_DE_LA_TABLA),
        config_ids_esperados=CONFIG_IDS_DE_PRUEBA,
    ), set_

def test_los_items_sin_juzgar_dan_cero_por_construccion_para_una_fila_del_pool(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado, set_ = _pool_congelado(cfg, almacen_temporal, rutas_de_prueba)
    corridas_de_c3 = corridas_de_las_cinco_filas(set_, filas=["C3"])
    sin_juzgar = met.items_del_top5_sin_juzgar(corridas_de_c3, pool=resultado.pool)
    assert sin_juzgar.formo_el_pool is True
    assert sin_juzgar.n_sin_juzgar == 0
    assert sin_juzgar.pct_sin_juzgar == 0.0
    assert "por construcción" in sin_juzgar.a_json()["lectura"]
    assert "la pregunta no aplica" in sin_juzgar.a_json()["lectura"]

def test_los_items_sin_juzgar_son_un_numero_real_para_una_sexta_configuracion(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado, set_ = _pool_congelado(cfg, almacen_temporal, rutas_de_prueba)
    primera = set_.consultas()[0].consulta_id
    sexta = corrida_de_prueba(
        consulta_id=primera,
        top5=["cma:900", "cma:901", "cma:902", "cma:903", "cma:904"],
        configuracion="C3",
        config_id="5e5e5e5e5e5e5e5e",
    )
    sin_juzgar = met.items_del_top5_sin_juzgar([sexta], pool=resultado.pool)
    assert sin_juzgar.formo_el_pool is False
    assert sin_juzgar.n_sin_juzgar == 5
    assert sin_juzgar.pct_sin_juzgar == 100.0
    assert sin_juzgar.sin_juzgar_por_consulta == {primera: 5}
    assert "subestimado" in sin_juzgar.a_json()["lectura"]

def test_la_etiqueta_no_alcanza_para_saber_si_formo_el_pool(
    cfg, almacen_temporal, rutas_de_prueba
):
    resultado, set_ = _pool_congelado(cfg, almacen_temporal, rutas_de_prueba)
    por_etiqueta = corrida_de_prueba(
        consulta_id=set_.consultas()[0].consulta_id,
        top5=top5_de_prueba("C3", 0),
        configuracion="C3",
        config_id="otro0000otro0000",
    )
    sin_juzgar = met.items_del_top5_sin_juzgar([por_etiqueta], pool=resultado.pool)
    assert sin_juzgar.configuracion == "C3"
    assert sin_juzgar.formo_el_pool is False
    assert sin_juzgar.n_sin_juzgar == 0
