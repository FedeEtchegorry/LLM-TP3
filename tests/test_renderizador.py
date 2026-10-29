from __future__ import annotations

import pytest

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.explicacion import Afirmación, Explicación
from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.dominio.salida import FormatoDeSalida, ObligaciónIncumplida
from tp3_rag.modulo_b.etapa import Candidata, EstadoDeConsulta
from tp3_rag.modulo_b.renderizador import RenderizadorDeSalida
from tp3_rag.nucleo.configuracion import Configuración

pytestmark = pytest.mark.nivel2

TEXTO_CCBY = "A rural landscape painted near Arkville in the Catskills."
TEXTO_CC0 = "Purchased from the artist in 1927."

def _fragmento(section="description", licencia="CC-BY-4.0", texto=TEXTO_CCBY, **cambios):
    base = dict(
        artwork_id="aic:109240",
        museum="aic",
        section=section,
        chunk_index=0,
        texto=texto,
        source_url="https://www.artic.edu/artworks/109240",
        license=licencia,
        title="Arkville Landscape",
        artist="Alexander H. Wyant",
    )
    base.update(cambios)
    return Fragmento(**base)

@pytest.fixture(scope="module")
def formato() -> FormatoDeSalida:
    return FormatoDeSalida.desde_configuración(Configuración.cargar())

def _estado(explicaciones, fragmentos, *, metadata=None):
    entrada = EntradaDeConsulta.ad_hoc(datos_de_imagen=b"\x89PNGxx", pedido="un pedido")
    meta = metadata or {
        "title": "Arkville Landscape",
        "artist": "Alexander H. Wyant",
        "date_display": "c. 1890",
        "medium": "oil on canvas",
        "style": "",
        "culture": "America",
        "source_url": "https://www.artic.edu/artworks/109240",
    }
    return (
        EstadoDeConsulta(entrada=entrada)
        .con(fuentes_de_ranking={"visual": (Candidata("aic:109240", 1, 0.91, meta),)})
        .proyectar_top5(["aic:109240"], por="recuperador_visual")
        .con(evidencia={"aic:109240": tuple(fragmentos)})
        .con(explicacion=tuple(explicaciones))
    )

def _render(estado, formato):
    return RenderizadorDeSalida(formato=formato).correr(estado)

def test_sa2_el_n_lo_asigna_el_renderizador_en_orden_de_primera_cita(formato):
    desc, prov = _fragmento(), _fragmento("provenance", "CC0", TEXTO_CC0)
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(
            Afirmación("Se compró en 1927.", "dato", (prov.fragment_id,)),
            Afirmación("Es un paisaje rural.", "dato", (desc.fragment_id,)),
        ),
    )
    salida = _render(_estado([explicación], [desc, prov]), formato).salida
    refs = {r.fragment_id: r.n for r in salida.obras[0].referencias}
    assert refs[prov.fragment_id] == 1
    assert refs[desc.fragment_id] == 2
    assert salida.obras[0].lineas[0].endswith("[1]")
    assert salida.obras[0].lineas[1].endswith("[2]")

def test_solo_se_numeran_los_fragmentos_efectivamente_citados(formato):
    desc, prov = _fragmento(), _fragmento("provenance", "CC0", TEXTO_CC0)
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    salida = _render(_estado([explicación], [desc, prov]), formato).salida
    assert [r.fragment_id for r in salida.obras[0].referencias] == [desc.fragment_id]

def test_p4_un_fragmento_cc_by_se_renderiza_con_atribucion(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    estado = _render(_estado([explicación], [desc]), formato)
    salida = estado.salida
    ref = salida.obras[0].referencias[0]
    assert ref.atribucion
    assert "CC BY 4.0" in ref.atribucion
    assert "Alexander H. Wyant" in ref.atribucion
    assert "Art Institute of Chicago" in ref.atribucion
    assert ref.atribucion in salida.render()
    assert estado.fallida is None

def test_un_fragmento_cc0_no_lleva_atribucion_y_eso_es_none_no_vacio(formato):
    prov = _fragmento("provenance", "CC0", TEXTO_CC0)
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Se compró en 1927.", "dato", (prov.fragment_id,)),),
    )
    salida = _render(_estado([explicación], [prov]), formato).salida
    assert salida.obras[0].referencias[0].atribucion is None

def test_sa3_falla_si_la_atribucion_no_esta_en_la_superficie(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    roto = FormatoDeSalida(
        formato="markdown",
        atribucion="en_bloque_final",
        campos_de_la_ficha=formato.campos_de_la_ficha,
        nombre_del_museo=dict(formato.nombre_del_museo),
        nombre_de_la_licencia=dict(formato.nombre_de_la_licencia),
        exigen_atribucion=dict(formato.exigen_atribucion),
    )
    estado = _render(_estado([explicación], [desc]), roto)
    assert estado.fallida is None
    sin_declarar = FormatoDeSalida(
        formato="markdown",
        atribucion="al_pie_de_la_referencia",
        campos_de_la_ficha=formato.campos_de_la_ficha,
        nombre_del_museo=dict(formato.nombre_del_museo),
        nombre_de_la_licencia=dict(formato.nombre_de_la_licencia),
        exigen_atribucion={"CC0": False},
    )
    con_falla = _render(_estado([explicación], [desc]), sin_declarar)
    assert con_falla.fallida == {
        "etapa": "renderizador",
        "obligacion": "licencia no declarada",
    }
    assert con_falla.salida is None

def test_una_licencia_sin_entrada_declarada_no_es_un_false(formato):
    with pytest.raises(ObligaciónIncumplida, match="no tiene entrada en"):
        formato.exige_atribucion("CC-BY-NC-3.0")

def test_sa3_falla_si_la_cita_no_es_byte_a_byte(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    estado = _render(_estado([explicación], [desc]), formato)
    salida = estado.salida
    otro = {desc.fragment_id: _fragmento(texto="otro texto")}
    with pytest.raises(ObligaciónIncumplida, match="no textual"):
        salida.verificar_obligaciones(otro)

def test_sa4_el_style_vacio_no_aparece_en_la_ficha(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    salida = _render(_estado([explicación], [desc]), formato).salida
    assert "style" not in salida.obras[0].ficha
    assert "medium" in salida.obras[0].ficha

def test_sa5_la_ref_colgada_degrada_cuenta_y_no_falla(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(
            Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),
            Afirmación("Se expuso en 1890.", "dato", ("aic:109240::exhibitions::0",)),
        ),
    )
    estado = _render(_estado([explicación], [desc]), formato)
    assert estado.fallida is None
    assert len(estado.afirmaciones_descartadas) == 1
    assert [d.etapa for d in estado.degradaciones] == ["renderizador"]
    assert len(estado.salida.obras[0].lineas) == 1

def test_p7_sin_explicacion_la_obra_sale_igual_con_ficha_y_link(formato):
    desc = _fragmento()
    estado = EstadoDeConsulta(
        entrada=EntradaDeConsulta.ad_hoc(datos_de_imagen=b"\x89PNGxx", pedido="p")
    )
    estado = (
        estado.con(
            fuentes_de_ranking={
                "visual": (
                    Candidata(
                        "aic:109240", 1, 0.9,
                        {"title": "T", "source_url": "https://www.artic.edu/artworks/1"},
                    ),
                )
            }
        )
        .proyectar_top5(["aic:109240"], por="recuperador_visual")
        .con(evidencia={"aic:109240": (desc,)})
        .degradar("generador", "boom", "sin explicación")
    )
    salida = _render(estado, formato).salida
    assert len(salida.obras) == 1
    assert salida.obras[0].lineas == ()
    assert salida.obras[0].abstencion
    assert "artic.edu" in salida.render()

def test_el_render_es_uno_solo_para_las_dos_superficies_de_p4(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    salida = _render(_estado([explicación], [desc]), formato).salida
    assert salida.render() == salida.render()
    assert salida.a_json()["render"] == salida.render()

def test_el_idioma_se_declara_en_vez_de_inferirse(formato):
    desc = _fragmento()
    explicación = Explicación(
        artwork_id="aic:109240",
        afirmaciones=(Afirmación("Es un paisaje.", "dato", (desc.fragment_id,)),),
    )
    salida = _render(_estado([explicación], [desc]), formato).salida
    assert salida.idioma_de_la_explicacion == "es"
    assert salida.idioma_de_las_citas == "en"
    assert TEXTO_CCBY in salida.render()

def test_el_formato_es_punto_de_variacion_y_no_toca_otra_pieza(formato):
    assert "sigue abierto" in formato.a_json()["nota"]
    with pytest.raises(Exception, match="formato"):
        FormatoDeSalida(formato="pdf")
    with pytest.raises(Exception, match="atribucion"):
        FormatoDeSalida(atribucion="en_el_pie_de_pagina")
