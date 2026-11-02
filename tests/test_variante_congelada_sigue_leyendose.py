from __future__ import annotations

import json
from pathlib import Path

import pytest

from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.modulo_a.fragmentador import EstrategiaDeChunking, Fragmentador
from tp3_rag.modulo_b.recolector_de_evidencia import _fragmento_desde_chroma
from tp3_rag.modulo_b.renderizador import _atribucion
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.dominio.salida import FormatoDeSalida, ObraCitada, Referencia, SalidaCitada

pytestmark = pytest.mark.nivel1

CONGELADA = "c3274567371ff42b"
RECETA_DE_LA_CONGELADA = RAÍZ_DEL_REPO / "artifacts" / "indices" / f"{CONGELADA}.json"

METADATA_DE_LA_CONGELADA = {
    "artwork_id": "aic:16568",
    "museum": "aic",
    "section": "description",
    "title": "The Bedroom",
    "artist": "Vincent van Gogh",
    "source_url": "https://www.artic.edu/artworks/16568",
    "license": "CC-BY-4.0",
    "chunk_index": 0,
}

@pytest.mark.skipif(
    not RECETA_DE_LA_CONGELADA.exists(),
    reason="el RegistroDeÍndices de la variante congelada no está en este working tree",
)
def test_la_estrategia_de_la_congelada_se_reproduce_byte_a_byte_desde_la_configuracion():
    registrada = json.loads(RECETA_DE_LA_CONGELADA.read_text())["receta"][
        "estrategia_de_chunking"
    ]
    de_hoy = EstrategiaDeChunking.desde_configuración(Configuración.cargar()).a_json()
    assert de_hoy == registrada

@pytest.mark.skipif(
    not RECETA_DE_LA_CONGELADA.exists(),
    reason="el RegistroDeÍndices de la variante congelada no está en este working tree",
)
def test_el_variante_id_de_la_congelada_se_re_deriva_desde_su_propia_receta():
    from tp3_rag.nucleo import identidad

    receta = json.loads(RECETA_DE_LA_CONGELADA.read_text())["receta"]
    re_derivado = identidad.variante_id(
        catalogo_id=receta["catalogo_id"],
        split_id=receta["split_id"],
        criterio_de_inclusion=receta["criterio_de_inclusion"],
        checkpoint_de_clip=receta["checkpoint_de_clip"],
        version_del_preprocesado=receta["version_del_preprocesado"],
        estrategia_de_chunking=receta["estrategia_de_chunking"],
        campos_de_knowledge=receta["campos_de_knowledge"],
        modelo_de_embedding_de_texto=receta["modelo_de_embedding_de_texto"],
        criterio_de_culture=receta["criterio_de_culture"],
    )
    assert re_derivado == CONGELADA

def test_el_brazo_ciego_es_otra_variante_y_no_la_misma_con_otro_contenido():
    from tp3_rag.nucleo import identidad

    def vid(estrategia: str) -> str:
        cfg = Configuración.cargar(overrides={"fragmentacion.estrategia": estrategia})
        return identidad.variante_id(
            catalogo_id="c",
            split_id="s",
            criterio_de_inclusion={"description_no_vacia": True},
            checkpoint_de_clip="k",
            version_del_preprocesado="1",
            estrategia_de_chunking=EstrategiaDeChunking.desde_configuración(cfg).a_json(),
            campos_de_knowledge=cfg["fragmentacion.campos_de_knowledge"],
            modelo_de_embedding_de_texto="m",
            criterio_de_culture="crudo",
        )

    assert vid("por_campo_semantico") != vid("ciego")

def test_la_metadata_singular_de_la_congelada_se_reconstruye_sin_las_claves_nuevas():
    f = _fragmento_desde_chroma(
        "aic:16568::description::0", "There is my bedroom.", METADATA_DE_LA_CONGELADA
    )
    assert (f.campos, f.licencias) == (None, None)
    assert f.campos_abarcados == ("description",)
    assert f.licencias_abarcadas == ("CC-BY-4.0",)
    assert f.license == "CC-BY-4.0"
    assert f.section == "description"
    assert not f.mezcla_licencias
    assert f.a_metadata() == METADATA_DE_LA_CONGELADA

def test_el_aplanado_de_chroma_se_deshace_acá_y_no_antes():
    f = _fragmento_desde_chroma(
        "aic:1::description::0",
        "x",
        {**METADATA_DE_LA_CONGELADA, "campos": "description | provenance",
         "licencias": "CC-BY-4.0 | CC0"},
    )
    assert f.campos_abarcados == ("description", "provenance")
    assert f.licencias_abarcadas == ("CC-BY-4.0", "CC0")
    assert f.mezcla_licencias

def test_una_clave_nueva_vacia_se_lee_como_ausente():
    f = _fragmento_desde_chroma(
        "aic:1::description::0", "x", {**METADATA_DE_LA_CONGELADA, "campos": "", "licencias": ""}
    )
    assert (f.campos, f.licencias) == (None, None)

def _formato() -> FormatoDeSalida:
    return FormatoDeSalida.desde_configuración(Configuración.cargar())

def test_la_atribucion_de_un_fragmento_de_una_licencia_es_la_de_siempre():
    f = _fragmento_desde_chroma("aic:16568::description::0", "x", METADATA_DE_LA_CONGELADA)
    assert _atribucion(f, _formato()) == (
        "“The Bedroom”, Vincent van Gogh. Art Institute of Chicago. "
        "Texto bajo CC BY 4.0 — https://www.artic.edu/artworks/16568"
    )

def test_un_fragmento_cc0_sigue_sin_atribucion():
    f = _fragmento_desde_chroma(
        "cma:1::provenance::0", "x", {**METADATA_DE_LA_CONGELADA, "license": "CC0"}
    )
    assert _atribucion(f, _formato()) is None

def test_la_atribucion_de_un_chunk_mezclado_nombra_LAS_DOS_licencias():
    f = _fragmento_desde_chroma(
        "aic:16568::description::0",
        "x",
        {**METADATA_DE_LA_CONGELADA, "campos": "description | provenance",
         "licencias": "CC-BY-4.0 | CC0"},
    )
    assert _atribucion(f, _formato()) == (
        "“The Bedroom”, Vincent van Gogh. Art Institute of Chicago. "
        "Texto bajo CC BY 4.0 + CC0 1.0 — https://www.artic.edu/artworks/16568"
    )

def test_sa3_exige_atribucion_cuando_ALGUNA_licencia_de_la_union_la_exige():
    formato = _formato()
    texto_citado = "Sold in 1923. There is my bedroom."
    ref = Referencia(
        n=1,
        fragment_id="aic:16568::provenance::0",
        texto_citado=texto_citado,
        title="The Bedroom",
        artist="Vincent van Gogh",
        source_url="https://www.artic.edu/artworks/16568",
        license="CC-BY-4.0",
        museum="aic",
        atribucion=None,
        campos=("provenance", "description"),
        licencias=("CC0", "CC-BY-4.0"),
    )
    salida = SalidaCitada(
        consulta_id="c01",
        obras=(
            ObraCitada(
                artwork_id="aic:16568",
                rango=1,
                score=None,
                ficha={},
                source_url="https://www.artic.edu/artworks/16568",
                lineas=("Una afirmación. [1]",),
                abstencion=None,
                referencias=(ref,),
            ),
        ),
        formato=formato,
    )
    fragmento = _fragmento_desde_chroma(
        "aic:16568::provenance::0",
        texto_citado,
        {**METADATA_DE_LA_CONGELADA, "section": "provenance",
         "campos": "provenance | description", "licencias": "CC0 | CC-BY-4.0"},
    )
    from tp3_rag.dominio.salida import ObligaciónIncumplida

    with pytest.raises(ObligaciónIncumplida, match="atribución faltante"):
        salida.verificar_obligaciones({ref.fragment_id: fragmento})

@pytest.mark.nivel3
@pytest.mark.skipif(
    not (RAÍZ_DEL_REPO / "data" / "catalogo").exists()
    or not RECETA_DE_LA_CONGELADA.exists(),
    reason="necesita el catálogo en disco y el registro de la variante congelada",
)
def test_re_fragmentar_la_congelada_reproduce_sus_6232_fragmentos_exactos():
    from tp3_rag.dominio.catalogo import CatálogoCanónico
    from tp3_rag.dominio.split import ManifiestoDeSplit
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    registrado = json.loads(RECETA_DE_LA_CONGELADA.read_text())
    almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
    cfg = Configuración.cargar()
    catálogo = CatálogoCanónico.cargar(
        almacén, Path(cfg["rutas.catalogo"]) / registrado["receta"]["catalogo_id"]
    )
    manifiesto = ManifiestoDeSplit.leer(
        almacén, cfg["rutas.artefactos_splits"], registrado["receta"]["split_id"]
    )
    corpus = [r for r in catálogo.registros if r.artwork_id in set(manifiesto.ids_corpus)]
    por_obra = Fragmentador(cfg).fragmentar_muchos(corpus)
    assert sum(len(v) for v in por_obra.values()) == registrado["conteos"]["n_fragmentos"]
