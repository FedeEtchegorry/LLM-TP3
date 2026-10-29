from __future__ import annotations

import pytest

from tp3_rag.dominio.explicacion import (
    Afirmación,
    Explicación,
    ViolaciónDelContratoDeSalida,
)

pytestmark = pytest.mark.nivel1

REF = "cma:109240::description::0"

def test_sa1_un_dato_sin_refs_no_se_construye():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="regla de honestidad"):
        Afirmación(texto="Es un óleo de 1890.", tipo="dato", refs=())

def test_sa1_una_observacion_con_refs_tampoco():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="otra dirección"):
        Afirmación(texto="La paleta es fría.", tipo="observacion", refs=(REF,))

def test_sa1_un_tercer_tipo_no_existe():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="no es uno de"):
        Afirmación(texto="No puedo decir.", tipo="abstencion")

@pytest.mark.parametrize("ref", ["1", "[1]", " [ 7 ] ", "12"])
def test_sa2_una_ref_numerica_es_el_contrato_roto(ref):
    with pytest.raises(ViolaciónDelContratoDeSalida, match="es un \\*\\*número\\*\\*|es un número"):
        Afirmación(texto="Es un óleo.", tipo="dato", refs=(ref,))

def test_una_ref_que_no_tiene_forma_de_fragment_id_se_rechaza():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="forma de un `fragment_id`"):
        Afirmación(texto="Es un óleo.", tipo="dato", refs=("cma:109240",))

def test_las_refs_son_un_conjunto():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="repite"):
        Afirmación(texto="Es un óleo.", tipo="dato", refs=(REF, REF))

def test_refs_como_string_se_rechaza_en_vez_de_iterarse():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="como string"):
        Afirmación.desde_json({"texto": "x", "tipo": "dato", "refs": REF})

def test_no_hay_default_para_el_tipo():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="tipo"):
        Afirmación.desde_json({"texto": "x", "refs": []})

def test_una_explicacion_vacia_es_el_hueco_que_sa5_prohibe():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="hueco silencioso"):
        Explicación(artwork_id="cma:1")

def test_las_refs_salen_en_orden_de_lectura_y_sin_repetir():
    a, b = "cma:1::description::0", "cma:1::provenance::0"
    e = Explicación(
        artwork_id="cma:1",
        afirmaciones=(
            Afirmación("Segundo.", "dato", (b,)),
            Afirmación("Primero.", "dato", (a, b)),
        ),
    )
    assert e.refs == (b, a)

def test_sa5_se_cae_la_afirmacion_entera_y_no_solo_la_ref():
    bueno, colgado = "cma:1::description::0", "cma:1::exhibitions::0"
    e = Explicación(
        artwork_id="cma:1",
        afirmaciones=(
            Afirmación("Con respaldo.", "dato", (bueno,)),
            Afirmación("Sin respaldo.", "dato", (bueno, colgado)),
            Afirmación("Observación propia.", "observacion"),
        ),
    )
    saneada, descartes = e.sin_refs_colgadas([bueno], texto_de_abstención="no alcanza")
    assert [a.texto for a in saneada.afirmaciones] == ["Con respaldo.", "Observación propia."]
    assert len(descartes) == 1
    assert descartes[0]["refs_colgadas"] == [colgado]
    assert all(a.refs != (bueno,) or a.texto == "Con respaldo." for a in saneada.afirmaciones)

def test_sa5_si_se_cae_la_ultima_la_obra_sale_con_abstencion():
    e = Explicación(
        artwork_id="cma:1",
        afirmaciones=(Afirmación("Dato.", "dato", ("cma:1::exhibitions::0",)),),
    )
    saneada, descartes = e.sin_refs_colgadas([], texto_de_abstención="la evidencia no alcanza")
    assert saneada.afirmaciones == ()
    assert saneada.abstencion == "la evidencia no alcanza"
    assert len(descartes) == 1

def test_sa5_no_lanza_porque_eso_rompia_p7_y_volvia_circular_a_p3a():
    e = Explicación(
        artwork_id="cma:1", afirmaciones=(Afirmación("D.", "dato", ("cma:1::x::0",)),)
    )
    saneada, _ = e.sin_refs_colgadas([], texto_de_abstención="x")
    assert saneada.artwork_id == "cma:1"

def test_una_explicacion_sin_artwork_id_no_se_puede_atribuir():
    with pytest.raises(ViolaciónDelContratoDeSalida, match="artwork_id"):
        Explicación.desde_json({"afirmaciones": []})

def test_el_json_ida_y_vuelta_conserva_la_forma():
    e = Explicación(
        artwork_id="cma:1",
        afirmaciones=(Afirmación("D.", "dato", (REF,)), Afirmación("O.", "observacion")),
        abstencion=None,
    )
    assert Explicación.desde_json(e.a_json()) == e
    assert e.n_datos == 1
