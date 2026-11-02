from __future__ import annotations

import pytest

from tp3_rag.dominio.binding import BindingDelGenerador
from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.explicacion import ViolaciónDelContratoDeSalida
from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.modulo_b.etapa import Candidata, EstadoDeConsulta
from tp3_rag.modulo_b.generador import (
    PROMPT_ID,
    SISTEMA,
    VERSIÓN_DEL_PROMPT,
    Generador,
    armar_mensajes,
)
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.puertos.llm import LLMGrabado

pytestmark = pytest.mark.nivel2

TEXTO = "A rural landscape painted near Arkville."

def _fragmento(artwork_id="cma:1", section="description", licencia="CC-BY-4.0"):
    return Fragmento(
        artwork_id=artwork_id, museum="cma", section=section, chunk_index=0,
        texto=TEXTO, source_url=f"https://x/{artwork_id}", license=licencia,
        title="Arkville Landscape", artist="Alexander H. Wyant",
    )

def _estado(obras=("cma:1", "cma:2")):
    entrada = EntradaDeConsulta.ad_hoc(
        datos_de_imagen=b"\x89PNGbytes", pedido="versiones más cálidas"
    )
    return (
        EstadoDeConsulta(entrada=entrada)
        .con(
            fuentes_de_ranking={
                "visual": tuple(
                    Candidata(
                        o, i + 1, 0.9,
                        {"title": f"T{o}", "artist": "A", "medium": "oil", "museum": "cma"},
                    )
                    for i, o in enumerate(obras)
                )
            }
        )
        .proyectar_top5(list(obras), por="recuperador_visual")
        .con(evidencia={o: (_fragmento(o),) for o in obras})
    )

def _binding(puerto, cfg=None):
    return BindingDelGenerador.desde_configuración(cfg or Configuración.cargar(), puerto=puerto)

def _respuesta(obras, *, tipo="dato", refs=None, extra=None):
    salida = []
    for o in obras:
        afs = [{
            "texto": "Es un paisaje rural al óleo.",
            "tipo": tipo,
            "refs": refs if refs is not None else [f"{o}::description::0"],
        }]
        if extra:
            afs.extend(extra)
        salida.append({"artwork_id": o, "afirmaciones": afs})
    return {"obras": salida}

def test_el_prompt_es_parte_de_la_identidad_del_binding():
    cfg = Configuración.cargar()
    assert cfg["bindings.generador.prompt_id"] == PROMPT_ID
    assert str(cfg["bindings.generador.version_del_prompt"]) == VERSIÓN_DEL_PROMPT
    desalineado = Configuración.cargar(
        overrides={"bindings.generador.version_del_prompt": "99"}
    )
    with pytest.raises(ViolaciónDelContratoDeSalida, match="identidad del binding"):
        Generador(binding=_binding(LLMGrabado({}), desalineado))

def test_el_prompt_prohibe_el_tercer_tipo_que_la_v1_dejaba_pasar():
    assert VERSIÓN_DEL_PROMPT == "2"
    assert '"abstencion" NO es un tipo' in SISTEMA
    assert "EXACTAMENTE DOS valores" in SISTEMA

def test_el_prompt_lleva_las_cinco_reglas_de_docs_03_6():
    for fragmento in (
        "ESPAÑOL",
        "fragment_id",
        "abstencion",
        "observacion",
        "DEBE llevar en `refs`",
    ):
        assert fragmento in SISTEMA

def test_el_mensaje_lleva_los_fragment_id_literales_y_el_texto_en_ingles():
    mensajes = armar_mensajes(_estado())
    usuario = [m for m in mensajes if m["rol"] == "usuario"][0]["texto"]
    assert "cma:1::description::0" in usuario
    assert TEXTO in usuario
    assert "versiones más cálidas" in usuario

def test_el_mensaje_no_lleva_media_license_ni_el_score():
    usuario = [m for m in armar_mensajes(_estado()) if m["rol"] == "usuario"][0]["texto"]
    assert "media_license" not in usuario
    assert "score" not in usuario.lower()

def test_una_sola_imagen_la_del_usuario():
    doble = LLMGrabado({"generador": [_respuesta(["cma:1", "cma:2"])]})
    Generador(binding=_binding(doble)).correr(_estado())
    assert doble.llamadas[0]["n_imagenes"] == 1

def test_la_explicacion_se_parsea_con_sa1_y_sa2_en_los_tipos():
    doble = LLMGrabado({"generador": [_respuesta(["cma:1", "cma:2"])]})
    nuevo = Generador(binding=_binding(doble)).correr(_estado())
    assert [e.artwork_id for e in nuevo.explicacion] == ["cma:1", "cma:2"]
    assert nuevo.explicacion[0].afirmaciones[0].tipo == "dato"

def test_una_obra_del_top5_que_el_modelo_no_explico_sale_con_abstencion():
    doble = LLMGrabado({"generador": [_respuesta(["cma:1"])]})
    nuevo = Generador(binding=_binding(doble)).correr(_estado())
    assert len(nuevo.explicacion) == 2
    assert nuevo.explicacion[1].afirmaciones == ()
    assert nuevo.explicacion[1].abstencion

def test_una_obra_explicada_fuera_del_top5_es_el_contrato_roto():
    doble = LLMGrabado({"generador": [_respuesta(["cma:1", "cma:99"])] * 3})
    with pytest.raises(ViolaciónDelContratoDeSalida, match="no están en el top-5"):
        Generador(binding=_binding(doble)).correr(_estado())

def test_el_tipo_fuera_del_enum_se_reintenta_y_despues_falla_la_etapa():
    mala = {"obras": [{"artwork_id": "cma:1", "afirmaciones": [
        {"texto": "x", "tipo": "abstencion", "refs": []}]}]}
    buena = _respuesta(["cma:1", "cma:2"])
    doble = LLMGrabado({"generador": [mala, buena]})
    generador = Generador(binding=_binding(doble))
    nuevo = generador.correr(_estado())
    assert generador.n_intentos == 2
    assert nuevo.explicacion
    doble = LLMGrabado({"generador": [mala, mala, mala]})
    with pytest.raises(ViolaciónDelContratoDeSalida):
        Generador(binding=_binding(doble)).correr(_estado())

def test_un_json_sin_la_clave_obras_falla_la_etapa():
    doble = LLMGrabado({"generador": [{"explicaciones": []}] * 3})
    with pytest.raises(ViolaciónDelContratoDeSalida, match="sin la clave `obras`"):
        Generador(binding=_binding(doble)).correr(_estado())

def test_el_generador_degrada_y_su_politica_es_la_de_9_3():
    assert Generador.DEGRADA is True
    assert "sin explicación" in Generador.POLÍTICA
    assert Generador.PRECONDICIONES == ("top5", "evidencia")

def test_el_generador_no_tiene_el_puerto_sino_el_binding():
    import inspect

    parámetros = set(inspect.signature(Generador.__init__).parameters) - {"self"}
    assert parámetros == {"binding"}
