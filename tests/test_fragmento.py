from __future__ import annotations

import pytest

from tp3_rag.dominio.catalogo import CampoDeTexto, Imagen, RegistroCanónico
from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.nucleo.errores import ErrorDeConstrucción

pytestmark = pytest.mark.nivel1

DESCRIPTION = CampoDeTexto(
    section="description",
    texto="Corot painted this view of a pond near his home at Ville-d'Avray.",
    source_url="https://clevelandart.org/art/1944.80",
    license="CC-BY-4.0",
)
PROVENANCE = CampoDeTexto(
    section="provenance",
    texto="Sold in 1923 to X; inherited by Y.",
    source_url="https://clevelandart.org/art/1944.80",
    license="CC0",
)

def _registro(**cambios) -> RegistroCanónico:
    base = dict(
        artwork_id="cma:124078",
        museum="cma",
        type="Painting",
        title="Pond at Ville-d'Avray",
        artist="Jean Baptiste Camille Corot",
        date_display="late 1860s",
        year=1865,
        medium="oil on fabric",
        style=None,
        culture_raw=("France, 19th century",),
        source_url="https://clevelandart.org/art/1944.80",
        imagen=Imagen("https://x/1.jpg", "sha256:ab"),
        media_license="CC0",
        campos_de_texto=(DESCRIPTION, PROVENANCE),
        procedencia={},
    )
    base.update(cambios)
    return RegistroCanónico(**base)

def test_fr1_la_licencia_sale_del_campo_y_no_del_registro():
    registro = _registro()
    fragmento = Fragmento.desde_campo(
        registro, DESCRIPTION, texto=DESCRIPTION.texto, chunk_index=0
    )
    assert fragmento.license == "CC-BY-4.0"
    assert registro.media_license == "CC0"

def test_fr1_dos_secciones_de_la_misma_obra_pueden_tener_licencias_distintas():
    registro = _registro()
    description = Fragmento.desde_campo(
        registro, DESCRIPTION, texto=DESCRIPTION.texto, chunk_index=0
    )
    provenance = Fragmento.desde_campo(
        registro, PROVENANCE, texto=PROVENANCE.texto, chunk_index=0
    )
    assert (description.license, provenance.license) == ("CC-BY-4.0", "CC0")

def test_fr1_el_source_url_tambien_sale_del_campo():
    registro = _registro(source_url="https://OTRA-URL-DEL-REGISTRO")
    fragmento = Fragmento.desde_campo(
        registro, DESCRIPTION, texto=DESCRIPTION.texto, chunk_index=0
    )
    assert fragmento.source_url == DESCRIPTION.source_url

def test_fr2_un_trozo_reescrito_es_error_duro():
    with pytest.raises(ErrorDeConstrucción, match="la verificación de subcadena"):
        Fragmento.desde_campo(
            _registro(),
            DESCRIPTION,
            texto="Corot  painted  this  view",
            chunk_index=0,
        )

def test_fr2_un_prefijo_literal_pasa():
    fragmento = Fragmento.desde_campo(
        _registro(), DESCRIPTION, texto="Corot painted this view", chunk_index=0
    )
    assert fragmento.texto in DESCRIPTION.texto

def test_fr2_ancla_al_campo_normalizado_y_no_al_crudo():
    normalizado = CampoDeTexto("description", "Corot painted this.", "https://x", "CC0")
    with pytest.raises(ErrorDeConstrucción, match="la verificación de subcadena"):
        Fragmento.desde_campo(
            _registro(), normalizado, texto="<p>Corot painted this.</p>", chunk_index=0
        )

def test_un_fragmento_vacio_no_es_un_fragmento():
    with pytest.raises(ErrorDeConstrucción):
        Fragmento.desde_campo(_registro(), DESCRIPTION, texto="", chunk_index=0)

def test_fr3_el_id_es_determinista_y_tiene_la_forma_de_6_1():
    fragmento = Fragmento.desde_campo(
        _registro(), DESCRIPTION, texto="Corot painted", chunk_index=2
    )
    assert fragmento.fragment_id == "cma:124078::description::2"

def test_fr3_el_mismo_id_puede_llevar_textos_distintos_y_es_el_hallazgo_h8():
    registro = _registro()
    corto = Fragmento.desde_campo(registro, DESCRIPTION, texto="Corot", chunk_index=0)
    largo = Fragmento.desde_campo(
        registro, DESCRIPTION, texto="Corot painted this view", chunk_index=0
    )
    assert corto.fragment_id == largo.fragment_id
    assert corto.clave_de_valor() != largo.clave_de_valor()
    assert corto != largo

def test_dos_fragmentos_con_el_mismo_valor_son_iguales():
    a = Fragmento.desde_campo(_registro(), DESCRIPTION, texto="Corot", chunk_index=0)
    b = Fragmento.desde_campo(_registro(), DESCRIPTION, texto="Corot", chunk_index=0)
    assert a == b
    assert len({a, b}) == 1

def test_la_metadata_lleva_artwork_id_y_no_lleva_media_license():
    metadata = Fragmento.desde_campo(
        _registro(), DESCRIPTION, texto="Corot", chunk_index=0
    ).a_metadata()
    assert metadata["artwork_id"] == "cma:124078"
    assert "media_license" not in metadata
    assert set(metadata) == {
        "artwork_id",
        "museum",
        "section",
        "title",
        "artist",
        "source_url",
        "license",
        "chunk_index",
    }

def test_title_y_artist_viajan_desnormalizados():
    fragmento = Fragmento.desde_campo(
        _registro(), DESCRIPTION, texto="Corot", chunk_index=0
    )
    assert fragmento.title == "Pond at Ville-d'Avray"
    assert fragmento.artist == "Jean Baptiste Camille Corot"

CONCATENACIÓN = f"{DESCRIPTION.texto}\n\n{PROVENANCE.texto}"

def test_un_fragmento_de_un_campo_deriva_los_dos_campos_nuevos():
    f = Fragmento.desde_campo(_registro(), DESCRIPTION, texto="Corot", chunk_index=0)
    assert (f.campos, f.licencias) == (None, None)
    assert f.campos_abarcados == ("description",)
    assert f.licencias_abarcadas == ("CC-BY-4.0",)
    assert not f.mezcla_licencias

def test_la_metadata_de_un_fragmento_de_un_campo_es_la_de_siempre():
    metadata = Fragmento.desde_campo(
        _registro(), DESCRIPTION, texto="Corot", chunk_index=0
    ).a_metadata()
    assert "campos" not in metadata and "licencias" not in metadata

def test_un_chunk_que_cruza_campos_declara_los_campos_y_la_union():
    f = Fragmento.desde_concatenacion(
        _registro(),
        [DESCRIPTION, PROVENANCE],
        texto=CONCATENACIÓN,
        chunk_index=0,
        concatenacion=CONCATENACIÓN,
    )
    assert f.campos_abarcados == ("description", "provenance")
    assert f.licencias_abarcadas == ("CC-BY-4.0", "CC0")
    assert f.mezcla_licencias
    assert f.combinacion_de_licencias == "CC-BY-4.0 + CC0"
    assert f.a_metadata()["licencias"] == ["CC-BY-4.0", "CC0"]

def test_el_license_singular_de_un_chunk_mezclado_es_el_mas_restrictivo():
    f = Fragmento.desde_concatenacion(
        _registro(),
        [PROVENANCE, DESCRIPTION],
        texto=CONCATENACIÓN,
        chunk_index=0,
        concatenacion=CONCATENACIÓN,
    )
    assert f.section == "provenance"
    assert f.license == "CC-BY-4.0"

def test_una_licencia_sin_orden_de_restriccion_declarado_es_error_duro():
    raro = CampoDeTexto("provenance", "x", DESCRIPTION.source_url, "CC-BY-NC-3.0")
    with pytest.raises(ErrorDeConstrucción, match="orden de restricción"):
        Fragmento.desde_concatenacion(
            _registro(),
            [DESCRIPTION, raro],
            texto="Corot",
            chunk_index=0,
            concatenacion="Corot",
        )

def test_fr2_del_brazo_ciego_ancla_a_la_concatenacion():
    with pytest.raises(ErrorDeConstrucción, match="la verificación de subcadena"):
        Fragmento.desde_concatenacion(
            _registro(),
            [DESCRIPTION],
            texto="un texto que nadie escribió",
            chunk_index=0,
            concatenacion=CONCATENACIÓN,
        )

def test_un_chunk_que_mezclara_dos_fichas_es_error_duro():
    otra_ficha = CampoDeTexto("provenance", "x", "https://otro-museo/9", "CC0")
    with pytest.raises(ErrorDeConstrucción, match="source_url"):
        Fragmento.desde_concatenacion(
            _registro(),
            [DESCRIPTION, otra_ficha],
            texto="Corot",
            chunk_index=0,
            concatenacion="Corot",
        )

def test_el_singular_y_la_union_no_pueden_discrepar():
    with pytest.raises(ErrorDeConstrucción, match="unión"):
        Fragmento(
            artwork_id="cma:1",
            museum="cma",
            section="description",
            chunk_index=0,
            texto="x",
            source_url="https://f",
            license="CC0",
            title=None,
            artist=None,
            campos=("description", "provenance"),
            licencias=("CC-BY-4.0",),
        )
