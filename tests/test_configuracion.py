from __future__ import annotations

import pytest

from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración

pytestmark = pytest.mark.nivel1

def test_la_configuracion_default_carga_y_valida():
    assert Configuración.cargar()["ingesta.museos"] == ("aic", "cma")

def test_una_clave_desconocida_es_un_error(tmp_path):
    archivo = tmp_path / "typo.toml"
    archivo.write_text("[ingesta]\nlimite_m = 20\n", encoding="utf-8")
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(archivo=archivo)

def test_una_clave_desconocida_por_flag_tambien_es_un_error():
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(overrides={"ingesta.limite_eme": 20})

def test_una_capa_especializa_sin_inventar(tmp_path):
    archivo = tmp_path / "solo-cma.toml"
    archivo.write_text('[ingesta]\nmuseos = ["cma"]\n', encoding="utf-8")
    cfg = Configuración.cargar(archivo=archivo)
    assert cfg["ingesta.museos"] == ("cma",)
    assert cfg["ingesta.tipos"] == ("Painting",)

def test_la_configuracion_es_inmutable():
    cfg = Configuración.cargar()
    with pytest.raises(TypeError):
        cfg._árbol["ingesta"] = {}
    assert isinstance(cfg["ingesta.museos"], tuple)

def test_las_capas_quedan_registradas_como_procedencia():
    cfg = Configuración.cargar(overrides={"ingesta.limite_n": 20})
    assert any("default.toml" in c for c in cfg.capas)
    assert any("--set ingesta.limite_n" in c for c in cfg.capas)

def test_el_config_id_cambia_con_la_configuracion_resuelta():
    a = Configuración.cargar()
    b = Configuración.cargar(overrides={"ingesta.limite_n": 20})
    assert a.config_id != b.config_id
    assert a.config_id == Configuración.cargar().config_id

def test_los_flags_se_interpretan_con_tipo():
    cfg = Configuración.cargar(overrides={"ingesta.limite_n": "20"})
    assert cfg["ingesta.limite_n"] == 20

def test_la_tabla_de_licencias_vive_en_la_configuracion():
    tabla = Configuración.cargar()["ingesta.tabla_de_licencias"]
    assert tabla["aic"]["description"] == "CC-BY-4.0"
    assert tabla["cma"]["description"] == "CC0"

def test_todo_el_resto_esta_enumerado_y_no_resuelto_por_un_default():
    cfg = Configuración.cargar()
    for museo in cfg["ingesta.museos"]:
        declaradas = set(cfg[f"ingesta.tabla_de_licencias.{museo}"])
        usadas = set(cfg[f"ingesta.campos_de_texto.{museo}"])
        assert usadas <= declaradas

def test_una_section_configurada_sin_licencia_falla_al_cargar(tmp_path):
    archivo = tmp_path / "campo-nuevo.toml"
    archivo.write_text(
        "[ingesta.campos_de_texto.cma]\n"
        'wall_description = { origen = "wall_description", render = "texto_plano" }\n',
        encoding="utf-8",
    )
    with pytest.raises(ErrorDeConfiguración) as e:
        Configuración.cargar(archivo=archivo)
    assert "error duro" in str(e.value) and "wall_description" in str(e.value)

def test_una_tabla_abierta_acepta_una_entrada_nueva(tmp_path):
    archivo = tmp_path / "tercer-museo.toml"
    archivo.write_text(
        """
[ingesta]
museos = ["aic", "cma", "met"]
[ingesta.media_license]
met = "CC0"
[ingesta.tabla_de_licencias.met]
description = "CC0"
[ingesta.campos_de_texto.met]
description = { origen = "description", render = "html_a_texto" }
[ingesta.campos_canonicos.met]
title = "title"
source_url = "objectURL"
[ingesta.urls.met]
campo_de_ficha = "objectURL"
""",
        encoding="utf-8",
    )
    cfg = Configuración.cargar(archivo=archivo)
    assert cfg["ingesta.museos"] == ("aic", "cma", "met")
    assert cfg["ingesta.tabla_de_licencias.met.description"] == "CC0"

def test_un_typo_fuera_de_una_tabla_abierta_sigue_siendo_un_error(tmp_path):
    archivo = tmp_path / "typo.toml"
    archivo.write_text("[http]\nreintetos = 9\n", encoding="utf-8")
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(archivo=archivo)

def test_un_museo_sin_media_license_falla_al_cargar(tmp_path):
    archivo = tmp_path / "sin-media.toml"
    archivo.write_text('[ingesta]\nmuseos = ["aic", "cma", "met"]\n', encoding="utf-8")
    with pytest.raises(ErrorDeConfiguración, match="met"):
        Configuración.cargar(archivo=archivo)

def test_un_criterio_de_culture_inexistente_falla_al_cargar(tmp_path):
    archivo = tmp_path / "culture.toml"
    archivo.write_text('[normalizacion.culture]\ncriterio = "inventado"\n', encoding="utf-8")
    with pytest.raises(ErrorDeConfiguración, match="punto de variación"):
        Configuración.cargar(archivo=archivo)

def test_k1_el_base_url_del_proveedor_esta_declarado_y_es_el_del_proxy():
    base_url = Configuración.cargar()["proveedor.base_url"]
    assert base_url.startswith("https://")
    assert "api.openai.com" not in base_url

def test_k1_un_base_url_vacio_falla_al_cargar_y_no_cae_al_default_del_sdk(tmp_path):
    archivo = tmp_path / "sin-host.toml"
    archivo.write_text('[proveedor]\nbase_url = ""\n', encoding="utf-8")
    with pytest.raises(ErrorDeConfiguración, match="base_url"):
        Configuración.cargar(archivo=archivo)

def test_k2_el_modelo_declarado_lleva_el_prefijo_del_proveedor():
    cfg = Configuración.cargar()
    assert cfg["embebido_texto.modelo"] == "openai/text-embedding-3-small"
    assert cfg["embebido_texto.dim"] == 1536

def test_el_precio_por_token_esta_declarado_para_el_modelo_que_se_usa():
    cfg = Configuración.cargar()
    precios = cfg["proveedor.precio_usd_por_millon_de_tokens"]
    assert precios[cfg["embebido_texto.modelo"]] == 0.02

def test_la_tabla_de_precios_es_abierta_porque_sus_claves_son_modelos(tmp_path):
    archivo = tmp_path / "otro-modelo.toml"
    archivo.write_text(
        '[proveedor.precio_usd_por_millon_de_tokens]\n'
        '"openai/text-embedding-3-large" = 0.13\n',
        encoding="utf-8",
    )
    cfg = Configuración.cargar(archivo=archivo)
    precios = cfg["proveedor.precio_usd_por_millon_de_tokens"]
    assert precios["openai/text-embedding-3-large"] == 0.13
    assert precios["openai/text-embedding-3-small"] == 0.02

def test_la_configuracion_resuelta_se_puede_serializar_entera():
    import json

    json.dumps(Configuración.cargar().a_dict(), ensure_ascii=False)

def test_un_override_puede_agregar_una_entrada_a_una_tabla_abierta():
    cfg = Configuración.cargar(
        overrides={"consulta.plan_de_falla.generador": "excepcion"}
    )
    assert dict(cfg["consulta.plan_de_falla"]) == {"generador": "excepcion"}

def test_un_override_sobre_una_tabla_cerrada_sigue_siendo_un_error():
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(overrides={"consulta.alfa_inventado": "0.5"})
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(overrides={"bindings.generador.temperatura": "0.5"})

def test_el_mensaje_del_override_desconocido_nombra_la_regla_de_las_tablas_abiertas():
    with pytest.raises(ErrorDeConfiguración) as e:
        Configuración.cargar(overrides={"consulta.inventada": "x"})
    assert "TABLAS_ABIERTAS" in str(e.value)

def test_m5_1_el_layout_no_mueve_el_config_id():
    base = Configuración.cargar()
    movido = Configuración.cargar(
        overrides={"rutas.artefactos_corridas": "/tmp/otro-lugar"}
    )
    assert movido["rutas.artefactos_corridas"] == "/tmp/otro-lugar"
    assert movido.config_id == base.config_id

def test_m5_1_los_parametros_de_maquina_y_el_ruteo_no_mueven_el_config_id():
    base = Configuración.cargar()
    for clave, valor in (
        ("embebido.threads", 2),
        ("indice.batch_de_upsert", 1000),
        ("embebido_texto.batch", 64),
        ("proveedor.base_url", "https://otro-proxy.example/"),
        ("http.reintentos", 9),
        ("proveedor.precio_usd_por_millon_de_tokens.openai/text-embedding-3-small", 0.03),
    ):
        otro = Configuración.cargar(overrides={clave: valor})
        assert otro.config_id == base.config_id, f"{clave} movió el `config_id`"

def test_m5_1_lo_que_si_determina_el_contenido_sigue_moviendo_el_config_id():
    base = Configuración.cargar()
    for clave, valor in (
        ("consulta.alfa", 0.75),
        ("consulta.n_candidatos", 50),
        ("evidencia.orden", "por_seccion_declarada"),
        ("evidencia.n_fragmentos_por_obra", 5),
        ("bindings.generador.version_del_prompt", "9"),
        ("consulta.top_k", 3),
        ("salida.formato", "texto"),
        ("trazas.proveedor", "nulo"),
        ("retencion.pedido", True),
        ("fragmentacion.tamano_objetivo", 2000),
        ("split.semilla", 99),
        ("embebido.checkpoint", "openai/clip-vit-large-patch14"),
    ):
        otro = Configuración.cargar(overrides={clave: valor})
        assert otro.config_id != base.config_id, f"{clave} NO movió el `config_id`"

def test_m5_1_el_bloque_se_declara_y_sale_con_sus_valores():
    from tp3_rag.nucleo.configuracion import NO_DETERMINAN_EL_CONFIG_ID

    cfg = Configuración.cargar()
    declarado = cfg.no_determinantes()
    assert set(declarado) == set(NO_DETERMINAN_EL_CONFIG_ID)
    assert declarado["proveedor.base_url"] == cfg["proveedor.base_url"]
    assert declarado["rutas"]["artefactos_corridas"] == cfg["rutas.artefactos_corridas"]

def test_m5_1_la_configuracion_que_viaja_en_el_artefacto_sigue_entera():
    cfg = Configuración.cargar()
    entera, determinante = cfg.a_dict(), cfg.a_dict_determinante()
    assert "rutas" in entera and "rutas" not in determinante
    assert "base_url" in entera["proveedor"]
    assert "proveedor" not in determinante
    assert set(entera) - set(determinante) == {
        "rutas",
        "http",
        "medicion",
        "proveedor",
        "fusion",
        "recuperacion_textual",
    }
    assert "proveedor" in determinante["trazas"]
    assert "proyecto" not in determinante["trazas"]
    assert set(determinante["bindings"]) == {"generador"}

def test_m5_1_el_nombre_del_proyecto_de_trazas_no_mueve_el_config_id():
    base = Configuración.cargar()
    renombrado = Configuración.cargar(overrides={"trazas.proyecto": "tp3-rag-g2-m8"})
    assert renombrado["trazas.proyecto"] == "tp3-rag-g2-m8"
    assert renombrado.config_id == base.config_id

CAPA_POR_FILA = {"C0": "c0.toml", "C1": None, "C2": "c2.toml", "C3": "c3.toml", "C4": "c4.toml"}

def _cfg(fila: str, **overrides):
    from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

    capa = CAPA_POR_FILA[fila]
    return Configuración.cargar(
        archivo=(RAÍZ_DEL_REPO / "config" / capa) if capa else None,
        overrides=overrides or None,
    )

def test_m9_3_el_binding_del_reranker_no_mueve_el_config_id_de_c1_a_c3():
    for fila in ("C0", "C1", "C2", "C3"):
        base = _cfg(fila)
        movido = _cfg(fila, **{"bindings.reranker.version_del_prompt": 7})
        assert movido["bindings.reranker.version_del_prompt"] == 7
        assert movido.config_id == base.config_id, f"{fila} se movió y no tiene reranker"
    assert _cfg("C4", **{"bindings.reranker.version_del_prompt": "7"}).config_id != _cfg(
        "C4"
    ).config_id

def test_m9_3_cada_eje_del_barrido_mueve_solo_las_filas_que_lo_leen():
    ejes = {
        "fusion.estrategia": ("concatenacion", {"C3", "C4"}),
        "fusion.k": (10, {"C3", "C4"}),
        "recuperacion_textual.agregacion": ("suma_de_los_2_mejores", {"C3", "C4"}),
        "recuperacion_textual.n_fragmentos_recuperados": (20, {"C3", "C4"}),
        "bindings.planificador.version_del_prompt": ("9", {"C2", "C3", "C4"}),
        "bindings.generador.version_del_prompt": ("9", {"C1", "C2", "C3", "C4"}),
        "consulta.alfa": (0.75, {"C0", "C1", "C2", "C3", "C4"}),
        "consulta.n_candidatos": (50, {"C0", "C1", "C2", "C3", "C4"}),
    }
    for clave, (valor, esperadas) in ejes.items():
        movidas = {f for f in CAPA_POR_FILA if _cfg(f, **{clave: valor}).config_id != _cfg(f).config_id}
        assert movidas == esperadas, f"{clave} movió {sorted(movidas)}"

def test_m9_3_alfa_congelado_no_mueve_el_config_id_y_sigue_declarado():
    for fila in CAPA_POR_FILA:
        base = _cfg(fila)
        congelado = _cfg(fila, **{"consulta.alfa_congelado": True})
        assert congelado["consulta.alfa_congelado"] is True
        assert congelado.config_id == base.config_id, f"{fila} se movió por una declaración"
    assert _cfg("C1", **{"consulta.alfa": 0.25}).config_id != _cfg("C1").config_id
    assert _cfg("C1")["consulta.alfa_congelado"] in (True, False)
    assert "consulta.alfa_congelado" in _cfg("C1").no_determinantes()

def test_m9_3_el_alcance_se_declara_y_crece_con_la_cadena():
    alcances = {f: set(_cfg(f).alcance_del_config_id()) for f in CAPA_POR_FILA}
    assert alcances["C0"] < alcances["C1"] < alcances["C2"] < alcances["C3"] < alcances["C4"]
    assert not {r for r in alcances["C0"] if r.startswith("bindings")}
    for fila, rutas in alcances.items():
        for r in rutas:
            assert not any(r != d and r.startswith(f"{d}.") for d in rutas), f"{fila}: {r}"

def _hojas(árbol, prefijo=""):
    for clave, valor in árbol.items():
        if isinstance(valor, dict) and valor:
            yield from _hojas(valor, prefijo=f"{prefijo}{clave}.")
        else:
            yield f"{prefijo}{clave}"

def test_m9_3_el_arbol_entero_esta_declarado_en_algun_lado():
    from tp3_rag.nucleo.configuracion import NO_DETERMINAN_EL_CONFIG_ID, _cubre

    cfg = Configuración.cargar()
    declarado = tuple(NO_DETERMINAN_EL_CONFIG_ID)
    for fila in CAPA_POR_FILA:
        declarado += _cfg(fila).alcance_del_config_id()
    sin_hogar = sorted(h for h in _hojas(cfg.a_dict()) if not _cubre(declarado, h))
    assert not sin_hogar, (
        f"{sin_hogar} no está ni en el alcance de ninguna fila ni en el bloque: "
        f"o la lee una etapa y esa etapa tiene que declararla, o no determina lo que la "
        f"corrida produce y va al bloque **con su razón**."
    )

def test_m9_3_una_etapa_declara_claves_que_existen():
    from tp3_rag.modulo_b.armador import CLASE_DE_ETAPA

    cfg = Configuración.cargar()
    for nombre, clase in CLASE_DE_ETAPA.items():
        assert clase.CLAVES_DE_CONFIGURACIÓN, f"`{nombre}` no declara ninguna clave"
        for ruta in clase.CLAVES_DE_CONFIGURACIÓN:
            assert cfg.get(ruta, None) is not None, f"`{nombre}` declara `{ruta}`, que no existe"

def test_m9_3_el_detector_caza_la_clave_que_una_etapa_lee_sin_declarar():
    from tp3_rag.nucleo.configuracion import fuera_del_alcance

    assert not fuera_del_alcance(
        ["consulta.alfa", "bindings.generador.modelo", "rutas.chroma", "http.reintentos"],
        alcance=_cfg("C1").alcance_del_config_id(),
    )
    assert fuera_del_alcance(
        ["fusion.estrategia", "consulta.alfa"],
        alcance=_cfg("C1").alcance_del_config_id(),
    ) == ("fusion.estrategia",)
    assert fuera_del_alcance(["bindings"], alcance=("bindings.generador",)) == ("bindings",)

def test_m9_3_el_armador_rechaza_la_cadena_cuyo_alcance_no_alcanza(monkeypatch):
    from tp3_rag.modulo_b.armador import ArmadorDePipeline, CadenaImposible, Fusionador

    monkeypatch.setattr(Fusionador, "CLAVES_DE_CONFIGURACIÓN", ())
    with pytest.raises(CadenaImposible, match="fuera del alcance declarado"):
        ArmadorDePipeline(
            _cfg("C3"),
            puerto_de_imagen=object(),
            indice=object(),
            puerto_de_llm=object(),
            embebedor_de_texto=object(),
            almacen_de_planes=object(),
            bytes_de_la_imagen=lambda _: b"",
        ).armar()

def test_m9_3_una_cadena_con_una_etapa_inventada_no_tiene_config_id():
    from tp3_rag.modulo_b.armador import CadenaImposible

    cfg = Configuración.cargar(
        overrides={"consulta.etapas": ["recuperador_visual", "reranqueador", "renderizador"]}
    )
    with pytest.raises(CadenaImposible, match="reranqueador"):
        cfg.config_id
