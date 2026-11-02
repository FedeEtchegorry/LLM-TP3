from __future__ import annotations

import json

import pytest

from tp3_rag.cli import validar as cli
from tp3_rag.dominio.corrida import AlmacenDeCorridas, Corrida
from tp3_rag.modulo_c import validadores as val
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

from conftest import corrida_de_prueba, plan_de_prueba

pytestmark = pytest.mark.nivel1

CORPUS_ESTRICTO = ("aic:1", "cma:2", "cma:3")
CORPUS_AMPLIO = CORPUS_ESTRICTO + ("aic:9",)

def _variante(
    *,
    criterio=None,
    ids_corpus=CORPUS_ESTRICTO,
    ids_indexados=None,
    registrada=True,
    rechazo=None,
    obras_sin_fragmento=(),
    split_id="a1b2c3d4e5f60718",
    variante_id="f0e1d2c3b4a59687",
) -> val.VarianteDelBanco:
    return val.VarianteDelBanco(
        criterio=dict(criterio or val.CRITERIO_ESTRICTO),
        split_id=split_id,
        variante_id=variante_id,
        ids_corpus=tuple(ids_corpus),
        ids_indexados=tuple(ids_corpus if ids_indexados is None else ids_indexados),
        obras_sin_fragmento=tuple(obras_sin_fragmento),
        n_fragmentos=len(tuple(ids_corpus)) * 2,
        registrada=registrada,
        rechazo=rechazo,
    )

def _amplia(**kwargs) -> val.VarianteDelBanco:
    base = dict(
        criterio=val.CRITERIO_AMPLIO,
        ids_corpus=CORPUS_AMPLIO,
        split_id="b1b2c3d4e5f60719",
        variante_id="e0e1d2c3b4a59688",
    )
    base.update(kwargs)
    return _variante(**base)

def test_r4_pasa_con_las_tres_clausulas_y_declara_su_poblacion():
    r = val.r4_corpus_parametrizado([_variante(), _amplia()])
    assert r.pasa and (r.n, r.de) == (3, 3)
    assert r.detalle["a_ids_indexados_es_el_lado_corpus"] == "2/2"
    assert r.detalle["b_la_del_criterio_amplio_registra"] is True
    assert r.detalle["c_los_criterios_dan_corpus_distintos"] is True
    assert "description_no_vacia" in r.poblacion
    assert "3 corpus" in r.poblacion and "4 corpus" in r.poblacion
    assert "cero llamadas al proveedor" in r.poblacion
    assert r.detalle["splits_derivados"] == ["a1b2c3d4e5f60718", "b1b2c3d4e5f60719"]

def test_r4_atrapa_un_indice_que_no_es_el_lado_corpus_de_su_manifiesto():
    r = val.r4_corpus_parametrizado(
        [_variante(), _amplia(ids_indexados=CORPUS_ESTRICTO + ("cma:77",))]
    )
    assert not r.pasa and r.n == 2
    assert r.detalle["a_ids_indexados_es_el_lado_corpus"] == "1/2"
    (falla,) = [f for f in r.fallas if f["clausula"] == "a"]
    assert falla["faltan_en_el_indice"] == ["aic:9"]
    assert falla["sobran_en_el_indice"] == ["cma:77"]
    assert "igualdad de **conjuntos**" in falla["problema"]

def test_r4_atrapa_que_la_variante_del_criterio_amplio_no_registre():
    r = val.r4_corpus_parametrizado(
        [
            _variante(),
            _amplia(
                registrada=False,
                obras_sin_fragmento=("aic:9",),
                rechazo="invariante 3 de los invariantes de la registración: aic:9 quedó con 0 fragmentos",
            ),
        ]
    )
    assert not r.pasa and r.n == 2
    assert r.detalle["b_la_del_criterio_amplio_registra"] is False
    (falla,) = [f for f in r.fallas if f["clausula"] == "b"]
    assert falla["obras_sin_fragmento"] == ["aic:9"]
    assert "duplicando el criterio de inclusión" in falla["problema"]
    assert "Fragmentador" in falla["problema"]

def test_r4_no_pasa_si_los_dos_criterios_dieron_el_mismo_corpus():
    r = val.r4_corpus_parametrizado(
        [_variante(), _amplia(ids_corpus=CORPUS_ESTRICTO)]
    )
    assert not r.pasa and r.n == 2
    assert r.detalle["c_los_criterios_dan_corpus_distintos"] is False
    (falla,) = [f for f in r.fallas if f["clausula"] == "c"]
    assert "el detector no midió nada" in falla["problema"]
    assert falla["n_corpus"] == [3, 3]

def test_r4_con_una_sola_variante_no_es_verificable_y_lo_dice():
    r = val.r4_corpus_parametrizado([_variante()])
    assert not r.pasa and (r.n, r.de) == (0, 3)
    assert "necesita dos" in r.poblacion
    assert r.fallas == ({"n_variantes": 1, "pide": 2},)

ÁRBOLES_PELIGROSOS = ("artifacts/ground-truth", "artifacts", "data/chroma", "data/catalogo")

def test_el_guardia_se_niega_a_correr_el_banco_sobre_el_repo(tmp_path):
    for árbol in ÁRBOLES_PELIGROSOS:
        raíz = tmp_path / árbol.replace("/", "-")
        (raíz / árbol).mkdir(parents=True)
        almacén = AlmacenamientoEnArchivos(raíz)
        with pytest.raises(ErrorDelProyecto) as e:
            val.exigir_banco_aislado(almacén)
        assert árbol in str(e.value), f"se niega y no nombra lo que encontró: {árbol}"

    raíz = tmp_path / "como-el-repo"
    for árbol in ÁRBOLES_PELIGROSOS:
        (raíz / árbol).mkdir(parents=True, exist_ok=True)
    almacén = AlmacenamientoEnArchivos(raíz)
    with pytest.raises(ErrorDelProyecto) as e:
        val.exigir_banco_aislado(almacén)
    mensaje = str(e.value)
    assert "artifacts/ground-truth" in mensaje and "artifacts" in mensaje
    assert "dos splits congelados" in mensaje
    assert val.rechazo_del_banco(almacén) == mensaje

def test_el_guardia_no_se_opone_a_un_almacen_vacio(tmp_path):
    almacén = AlmacenamientoEnArchivos(tmp_path)
    assert val.exigir_banco_aislado(almacén) == tmp_path
    assert val.rechazo_del_banco(almacén) is None
    (tmp_path / "artifacts" / "ground-truth").mkdir(parents=True)
    assert val.rechazo_del_banco(almacén) is not None

def test_r4_desde_el_entrypoint_declara_que_no_es_ejecutable(capsys):
    assert cli.main(["r4"]) == 0
    salida = capsys.readouterr().out
    assert "no ejecutable todavía" in salida
    assert "0 de las 2 variantes que el corpus parametrizado pide" in salida
    assert "ninguna fue evaluada" in salida
    assert "artifacts/ground-truth" in salida
    assert "este entrypoint no construye" in salida
    assert "tmp_path" in salida and "nivel 2" in salida
    assert "⊘" in salida
    assert "no evaluable(s)" in salida
    assert "0/0 criterios **evaluables** pasan" in salida
    marcadores = [l for l in salida.splitlines() if l.strip().startswith("✗")]
    assert not marcadores, f"nadie midió nada y hay un ✗: {marcadores}"

def test_lt3_un_criterio_no_evaluable_no_tapa_una_falla_de_verdad(capsys):
    r4 = val.r4_sin_banco(raíz=RAÍZ_DEL_REPO, rechazo=None)
    falla = val.Resultado(
        criterio="uno que SÍ se midió y no pasa", pasa=False, n=1, de=2, poblacion="2 cosas"
    )
    assert r4.marca == "⊘" and falla.marca == "✗"
    assert r4.a_json()["estado"] == "no_evaluable"
    assert falla.a_json()["estado"] == "falla"
    assert all(r.pasa for r in [r4] if r.evaluable) is True
    assert all(r.pasa for r in [r4, falla] if r.evaluable) is False

def test_lt3_todo_incluye_r4_para_que_no_quede_invisible():
    import inspect

    fuente = inspect.getsource(cli.main)
    assert 'args.comando in ("r4", "todo")' in fuente

def test_r4_desde_el_entrypoint_no_escribe_un_solo_archivo(tmp_path, monkeypatch):
    antes = sorted(p.name for p in tmp_path.rglob("*"))
    monkeypatch.setattr(cli, "RAÍZ_DEL_REPO", tmp_path)
    assert cli.main(["--silencioso", "r4"]) == 0
    assert sorted(p.name for p in tmp_path.rglob("*")) == antes == []

def test_r4_sin_banco_distingue_las_dos_causas(tmp_path):
    peligroso = tmp_path / "con-artefactos"
    (peligroso / "artifacts" / "ground-truth").mkdir(parents=True)
    aislado = tmp_path / "almacen-vacio"
    aislado.mkdir()

    con_guardia = val.r4_sin_banco(
        raíz=peligroso,
        rechazo=val.rechazo_del_banco(AlmacenamientoEnArchivos(peligroso)),
    )
    sin_guardia = val.r4_sin_banco(raíz=aislado, rechazo=None)
    for r in (con_guardia, sin_guardia):
        assert not r.pasa and (r.n, r.de) == (0, 3)
        assert r.evaluable is False
        assert r.detalle["ejecutable"] is False
        assert r.detalle["clausulas_evaluadas"] == "0/3"
    assert "se niega a correr" in con_guardia.detalle["lo_que_contesto_el_guardia"]
    assert "no se opone" in sin_guardia.detalle["lo_que_contesto_el_guardia"]
    assert "es la población" in sin_guardia.detalle["lo_que_contesto_el_guardia"]

LIBRO_COMPLETO = {
    "planificador": {"n_llamadas": 1, "tokens_in": 900, "tokens_out": 60},
    "embedding_texto": {"n_llamadas": 1, "tokens_in": 120, "tokens_out": 0},
    "generador": {"n_llamadas": 1, "tokens_in": 1400, "tokens_out": 320},
    "embedding_imagen": {"n_llamadas": 2, "n_items": 2, "tokens_in": 0, "tokens_out": 0},
    "indice": {"n_llamadas": 1, "n_items": 20, "tokens_in": 0, "tokens_out": 0},
}

def _corrida(*, libro_mayor=LIBRO_COMPLETO, sin_poblar=(), degradaciones=(), **kwargs):
    base = corrida_de_prueba(degradaciones=degradaciones, **kwargs)
    contenido = json.loads(json.dumps(base.contenido))
    contenido["estado"]["claves_pobladas"] = [
        e for e in contenido["estado"]["claves_pobladas"] if e not in sin_poblar
    ]
    contenido["libro_mayor"] = json.loads(json.dumps(dict(libro_mayor)))
    return Corrida(contenido)

def _sin(rol):
    return {k: v for k, v in LIBRO_COMPLETO.items() if k != rol}

def test_r9_pasa_cuando_cada_etapa_tarifada_dejo_sus_tokens():
    r = val.r9_trazas_aunque_sea_codigo_propio(
        [
            _corrida(consulta_id="c01-visual", top5=["cma:1"], plan=plan_de_prueba()),
            _corrida(consulta_id="c02-visual", top5=["cma:2"], libro_mayor=_sin("planificador")),
        ]
    )
    assert r.pasa and (r.n, r.de) == (5, 5)
    assert r.detalle["por_rol"] == {
        "embedding_texto": "2/2",
        "generador": "2/2",
        "planificador": "1/1",
    }
    assert "2 corridas" in r.poblacion and "5 pares" in r.poblacion

def test_r9_acusa_la_etapa_que_poblo_su_clave_y_no_dejo_span():
    r = val.r9_trazas_aunque_sea_codigo_propio(
        [_corrida(consulta_id="c01-visual", top5=["cma:1"], libro_mayor=_sin("generador"))]
    )
    assert not r.pasa and (r.n, r.de) == (1, 2)
    (alarma,) = r.fallas
    assert (alarma["etapa"], alarma["rol"]) == ("generador", "generador")
    assert alarma["rol_en_el_libro_mayor"] is False and alarma["tokens"] == 0
    assert "sin pasar por el puerto" in alarma["problema"]
    assert r.detalle["por_rol"]["generador"] == "0/1"

def test_r9_un_plan_reusado_no_es_una_alarma_y_la_exclusion_va_dicha():
    r = val.r9_trazas_aunque_sea_codigo_propio(
        [
            _corrida(
                consulta_id="c01-visual",
                top5=["cma:1"],
                plan=plan_de_prueba(reusado=True),
                libro_mayor=_sin("planificador"),
            )
        ]
    )
    assert r.pasa and (r.n, r.de) == (2, 2)
    assert r.detalle["n_planes_reusados_excluidos"] == 1
    assert "1 plan(es) reusado(s)" in r.poblacion
    assert "planificador" not in r.detalle["por_rol"]

def test_r9_una_etapa_que_no_poblo_su_clave_queda_fuera_del_denominador():
    r = val.r9_trazas_aunque_sea_codigo_propio(
        [
            _corrida(
                consulta_id="c01-visual",
                top5=["cma:1"],
                libro_mayor=_sin("generador"),
                sin_poblar=("generador",),
                degradaciones=[
                    {"etapa": "generador", "planificada": True, "motivo": "excepcion"}
                ],
            )
        ]
    )
    assert r.pasa and (r.n, r.de) == (1, 1)
    assert r.fallas == ()

def test_r9_los_roles_locales_van_declarados_y_no_son_alarma():
    r = val.r9_trazas_aunque_sea_codigo_propio(
        [_corrida(consulta_id="c01-visual", top5=["cma:1"])]
    )
    assert r.pasa
    assert sorted(r.detalle["roles_sin_tokens_por_diseno"]) == ["embedding_imagen", "indice"]
    assert "['embedding_imagen', 'indice'] (locales)" in r.poblacion
    assert "CLIP corre **local**" in r.detalle["roles_sin_tokens_por_diseno"]["embedding_imagen"]

def test_r9_sin_corridas_no_pasa_vacuamente():
    r = val.r9_trazas_aunque_sea_codigo_propio([])
    assert not r.pasa and (r.n, r.de) == (0, 0)
    assert r.fallas == ()
    assert "0 pares" in r.poblacion

def _almacen_con(raíz, corridas):
    almacén = AlmacenamientoEnArchivos(raíz)
    de = AlmacenDeCorridas(almacén, "artifacts/corridas")
    for c in corridas:
        (raíz / "artifacts" / "corridas" / c.corrida_id).mkdir(parents=True, exist_ok=True)
        de.escribir(c.corrida_id, c.contenido)
    return almacén

def test_r9_desde_el_entrypoint_pasa_sobre_corridas_sanas(tmp_path, monkeypatch, capsys):
    _almacen_con(
        tmp_path,
        [
            _corrida(consulta_id="c01-visual", top5=["cma:1"], plan=plan_de_prueba()),
            _corrida(consulta_id="c02-visual", top5=["cma:2"], libro_mayor=_sin("planificador")),
        ],
    )
    monkeypatch.setattr(cli, "RAÍZ_DEL_REPO", tmp_path)
    assert cli.main(["r9"]) == 0
    salida = capsys.readouterr().out
    assert "✓  el libro mayor completo por rol" in salida
    assert "corridas   : 2 en artifacts/corridas" in salida
    assert "5 pares (corrida, etapa con modelo tarifado) sobre 2 corridas" in salida

def test_r9_desde_el_entrypoint_falla_con_el_defecto_puesto(tmp_path, monkeypatch, capsys):
    _almacen_con(
        tmp_path,
        [_corrida(consulta_id="c01-visual", top5=["cma:1"], libro_mayor=_sin("generador"))],
    )
    monkeypatch.setattr(cli, "RAÍZ_DEL_REPO", tmp_path)
    assert cli.main(["r9"]) == 1
    salida = capsys.readouterr().out
    assert "✗  el libro mayor completo por rol" in salida
    assert "'etapa': 'generador'" in salida
    assert "1/2" in salida

def test_r9_sin_corridas_en_disco_dice_como_producirlas(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "RAÍZ_DEL_REPO", tmp_path)
    assert cli.main(["r9"]) == 1
    error = capsys.readouterr().err
    assert "no hay ninguna corrida" in error
    assert "tp3-responder" in error

def test_los_dos_subcomandos_estan_en_el_parser_y_en_la_ayuda(capsys):
    for nombre in ("r4", "r9"):
        assert cli._argumentos([nombre]).comando == nombre
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    ayuda = capsys.readouterr().out
    assert "r9" in ayuda and "r4" in ayuda
