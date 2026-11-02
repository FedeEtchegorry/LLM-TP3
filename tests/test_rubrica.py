from __future__ import annotations

import pytest

from tp3_rag.dominio import rubrica as drubrica
from tp3_rag.dominio.juicios import GRADOS
from tp3_rag.dominio.rubrica import Rúbrica
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConfiguración

from conftest import rubrica_de_prueba

pytestmark = pytest.mark.nivel1

def test_la_identidad_es_la_designacion_declarada_y_no_un_hash_del_texto():
    r = rubrica_de_prueba(designacion="piloto")
    otra = Rúbrica.desde_texto(r.texto.replace("no cumple", "no cumple nada"))

    assert r.rubrica_id == otra.rubrica_id == "piloto"
    assert r.hash_del_texto != otra.hash_del_texto

def test_un_rubrica_id_con_forma_de_hash_se_rechaza():
    with pytest.raises(ErrorDeConfiguración, match="hash_corto"):
        drubrica.plantilla(designacion="a1b2c3d4e5f60718")

def test_el_nombre_y_el_contenido_no_pueden_discrepar(almacen_temporal, rutas_de_prueba):
    almacen_temporal.escribir_texto(
        Rúbrica.ruta_de(rutas_de_prueba.rubricas, "v2"), rubrica_de_prueba().texto
    )
    with pytest.raises(ErrorDeArtefacto, match="declara adentro"):
        Rúbrica.leer(almacen_temporal, rutas_de_prueba.rubricas, "v2")

def test_una_rubrica_no_se_puede_refinar_a_si_misma():
    with pytest.raises(ErrorDeArtefacto, match="apunta a sí misma"):
        rubrica_de_prueba(designacion="v2", refina="v2")

def test_las_cinco_secciones_son_obligatorias():
    completa = rubrica_de_prueba()
    assert sorted(completa.secciones()) == [
        "desempate", "ejemplos", "grado_0", "grado_1", "grado_2"
    ]

@pytest.mark.parametrize(
    "encabezado,falta",
    [
        ("## Desempate", "desempate"),
        ("## Ejemplos", "ejemplos"),
        ("## Grado 0 — no cumple", "grado_0"),
    ],
)
def test_sin_una_seccion_la_rubrica_se_rechaza(encabezado, falta):
    texto = rubrica_de_prueba().texto.replace(encabezado, "## Otra cosa")
    with pytest.raises(ErrorDeArtefacto, match=falta):
        Rúbrica.desde_texto(texto)

def test_una_seccion_vacia_tambien_se_rechaza():
    texto = rubrica_de_prueba().texto.replace("entre 2 y 1 gana 1.", "")
    with pytest.raises(ErrorDeArtefacto, match="vacías"):
        Rúbrica.desde_texto(texto)

def test_los_tres_grados_tienen_definicion():
    r = rubrica_de_prueba()
    for grado in GRADOS:
        assert r.definicion_de(grado).strip()

def test_la_plantilla_que_el_codigo_emite_es_plantilla_y_no_es_usable():
    r = Rúbrica.desde_texto(drubrica.plantilla())

    assert r.estado == "plantilla"
    assert not r.es_usable_para_anotar
    assert r.texto.count(drubrica.MARCA) >= 3

def test_una_rubrica_completa_con_huecos_se_rechaza():
    texto = drubrica.plantilla().replace('estado     = "plantilla"', 'estado = "completa"', 1)
    with pytest.raises(ErrorDeArtefacto, match=drubrica.MARCA):
        Rúbrica.desde_texto(texto)

def test_una_plantilla_sin_huecos_tambien_se_rechaza():
    texto = rubrica_de_prueba().texto.replace('estado = "completa"', 'estado = "plantilla"')
    with pytest.raises(ErrorDeArtefacto, match="declarala `completa`"):
        Rúbrica.desde_texto(texto)

def test_una_clave_de_mas_en_el_front_matter_se_rechaza():
    texto = rubrica_de_prueba().texto.replace(
        'estado = "completa"', 'estado = "completa"\nnotas_sueltas = "algo importante"'
    )
    with pytest.raises(ErrorDeArtefacto, match="no se guardan"):
        Rúbrica.desde_texto(texto)

def test_sin_front_matter_no_hay_identidad_que_citar():
    with pytest.raises(ErrorDeArtefacto, match=r"\+\+\+"):
        Rúbrica.desde_texto("# una rúbrica sin front matter\n\n## Grado 2\n\nalgo\n")

def test_la_cita_lleva_el_hash_y_dice_que_es_procedencia(rutas_de_prueba):
    cita = rubrica_de_prueba().cita(rutas_de_prueba.rubricas)

    assert cita["rubrica_id"] == "piloto"
    assert cita["estado"] == "completa"
    assert cita["hash_del_texto"]
    assert cita["archivo"].endswith("rubricas/piloto.md")
    assert "procedencia, no identidad" in cita["nota"]
