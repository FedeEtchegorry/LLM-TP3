"""El filtro por metadata en pre-retrieval: qué reconoce, qué restringe y qué no mueve."""
from __future__ import annotations

import re

import pytest

from tp3_rag.modulo_b import filtro_de_metadata as fm
from tp3_rag.modulo_b.etapa import EstadoDeConsulta
from tp3_rag.modulo_b.recuperador_textual import RecuperadorTextual
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE, ÍndiceVectorialEnMemoria

from conftest import plan_de_prueba

pytestmark = pytest.mark.nivel1

# Los config_id con los que se corrieron las filas publicadas. Si el filtro los moviera,
# esas corridas quedarían huérfanas.
IDS_PUBLICADOS = {
    None: "4443862ae45e62d6",
    "c0.toml": "44bd01270964aec2",
    "c2.toml": "8c128ba44b0f67cc",
    "c3.toml": "6b1e25abd3e63952",
    "c4.toml": "88368002659b48fd",
}


# ---------------------------------------------------------------- el pedido

@pytest.mark.parametrize(
    "pedido, esperado",
    [
        ("el mismo paisaje pero pintado en Japón", ("Japan",)),
        ("escenas de agua en la pintura china", ("China",)),
        ("interiores arquitectónicos en la pintura italiana", ("Italy",)),
        ("composiciones geométricas en el arte de la India", ("India",)),
        ("retratos de mujeres en la pintura francesa", ("France",)),
        ("escenas con jinetes en el arte mogol", ("Mughal",)),
        ("retratos de la pintura holandesa", ("Netherlands", "Holland")),
    ],
)
def test_reconoce_la_cultura_que_el_pedido_nombra(pedido, esperado):
    assert fm.terminos_pedidos(pedido) == esperado


@pytest.mark.parametrize(
    "pedido",
    ["con más vida", "versiones más cálidas de este paisaje", "un paisaje en tinta china", ""],
)
def test_sin_cultura_nombrada_no_hay_terminos(pedido):
    assert fm.terminos_pedidos(pedido) == ()


def test_en_el_set_de_evaluacion_filtra_los_diez_pedidos_culturales_y_ninguno_mas():
    texto = (RAÍZ_DEL_REPO / "consultas-g2.toml").read_text(encoding="utf-8")
    consultas = re.findall(
        r'consulta_id\s*=\s*"([^"]+)"\s*\ncategoria\s*=\s*"([^"]*)"\s*\npedido\s*=\s*"([^"]*)"', texto
    )
    assert len(consultas) == 30
    con_filtro = {cid for cid, _, pedido in consultas if fm.terminos_pedidos(pedido)}
    culturales = {cid for cid, categoria, _ in consultas if categoria == "conceptual_cultural"}
    assert con_filtro == culturales and len(culturales) == 10


def test_un_termino_cubre_los_valores_que_lo_contienen_como_palabra_entera():
    vocabulario = ["India", "Mughal India", "Northern India", "China", "Indochina", "Tibet", "Central Tibet", ""]
    assert fm.valores_que_cumplen(("India",), vocabulario) == ("India", "Mughal India", "Northern India")
    assert fm.valores_que_cumplen(("Mughal",), vocabulario) == ("Mughal India",)
    assert fm.valores_que_cumplen(("China",), vocabulario) == ("China",)
    assert fm.valores_que_cumplen(("Tibet",), vocabulario) == ("Central Tibet", "Tibet")
    assert fm.valores_que_cumplen(("Korea",), vocabulario) == ()


# ---------------------------------------------------------------- la búsqueda

def _indice(n_chinas: int = 25, n_francesas: int = 25) -> ÍndiceVectorialEnMemoria:
    """Dos culturas mezcladas: sin filtro, la búsqueda devuelve obras de las dos."""
    indice = ÍndiceVectorialEnMemoria(variante_id="v")
    fijo = EmbeddingTextoFijo()
    obras = [(f"cma:{i}", "China") for i in range(n_chinas)] + [
        (f"aic:{i}", "France") for i in range(n_francesas)
    ]
    indice.upsert(
        ARTWORKS,
        ids=[o for o, _ in obras],
        vectores=[[1.0, 0.0, 0.0, 0.0] for _ in obras],
        metadatas=[
            {"museum": o.split(":")[0], "title": f"obra {o}", "culture": c, "source_url": f"https://museo/{o}"}
            for o, c in obras
        ],
    )
    textos = {"China": "ink scroll with mountains and mist", "France": "water scene with boats on a river"}
    indice.upsert(
        KNOWLEDGE,
        ids=[f"{o}::description::0" for o, _ in obras],
        vectores=[fijo.embeber_uno(f"{textos[c]} {o}") for o, c in obras],
        metadatas=[
            {"artwork_id": o, "museum": o.split(":")[0], "section": "description", "title": "obra",
             "source_url": "https://museo/x", "license": "CC0", "chunk_index": 0}
            for o, _ in obras
        ],
        documentos=[textos[c] for _, c in obras],
    )
    return indice


def _estado(pedido: str) -> EstadoDeConsulta:
    from tp3_rag.dominio.consulta import EntradaDeConsulta
    from tp3_rag.dominio.plan import PlanDeConsulta

    entrada = EntradaDeConsulta(
        consulta_id="c02-conceptual",
        imagen_hash=f"sha256:{0:064d}",
        pedido=pedido,
        procedencia="set",
        datos_de_imagen=b"jpeg",
        categoria="conceptual_cultural",
    )
    plan = PlanDeConsulta.desde_artefacto(
        plan_de_prueba(consulta_id="c02-conceptual", conceptual="water scene with boats on a river")
    )
    return EstadoDeConsulta(entrada=entrada).con(plan=plan)


def _rama(indice, **kwargs) -> RecuperadorTextual:
    return RecuperadorTextual(
        indice=indice,
        embebedor_de_texto=EmbeddingTextoFijo(),
        n_fragmentos=kwargs.pop("n_fragmentos", 10),
        agregacion="mejor_fragmento",
        **kwargs,
    )


def _culturas(estado, indice) -> set[str]:
    ids = [c.artwork_id for c in estado.fuentes_de_ranking["texto"]]
    return {m["culture"] for m in indice.metadatas(ARTWORKS, ids).values()}


def test_sin_el_filtro_la_busqueda_trae_lo_mas_parecido_sea_de_donde_sea():
    indice = _indice()
    estado = _rama(indice, n_fragmentos=50).correr(_estado("escenas de agua en la pintura china"))
    assert _culturas(estado, indice) == {"China", "France"}
    assert "filtro_por_metadata" not in estado.sellos


def test_con_el_filtro_solo_son_elegibles_las_obras_de_la_cultura_pedida():
    indice = _indice()
    estado = _rama(indice, n_fragmentos=50, filtro_por_metadata=True).correr(
        _estado("escenas de agua en la pintura china")
    )
    assert _culturas(estado, indice) == {"China"}
    assert len(estado.fuentes_de_ranking["texto"]) == 25
    registro = estado.sellos["filtro_por_metadata"]
    assert registro["aplicado"] is True
    assert registro["terminos"] == ["China"] and registro["valores"] == ["China"]
    assert registro["n_obras"] == 25


def test_si_el_pedido_no_nombra_una_cultura_el_filtro_no_cambia_nada():
    indice = _indice()
    pedido = "la misma escena de agua pero con luz dorada"
    sin = _rama(indice).correr(_estado(pedido))
    con = _rama(indice, filtro_por_metadata=True).correr(_estado(pedido))
    assert [c.artwork_id for c in con.fuentes_de_ranking["texto"]] == [
        c.artwork_id for c in sin.fuentes_de_ranking["texto"]
    ]
    assert con.sellos["filtro_por_metadata"]["aplicado"] is False


def test_con_pocas_obras_de_esa_cultura_no_se_filtra_y_queda_dicho():
    indice = _indice(n_chinas=fm.MÍNIMO_DE_OBRAS - 1)
    estado = _rama(indice, filtro_por_metadata=True).correr(
        _estado("escenas de agua en la pintura china")
    )
    registro = estado.sellos["filtro_por_metadata"]
    assert registro["aplicado"] is False and registro["n_obras"] == fm.MÍNIMO_DE_OBRAS - 1
    assert "menos de" in registro["motivo"]
    sin = _rama(indice).correr(_estado("escenas de agua en la pintura china"))
    assert [c.artwork_id for c in estado.fuentes_de_ranking["texto"]] == [
        c.artwork_id for c in sin.fuentes_de_ranking["texto"]
    ]


def test_una_cultura_que_no_esta_en_el_indice_no_filtra():
    indice = _indice()
    estado = _rama(indice, filtro_por_metadata=True).correr(_estado("retratos en la pintura coreana"))
    registro = estado.sellos["filtro_por_metadata"]
    assert registro["aplicado"] is False and registro["terminos"] == ["Korea"]


def test_el_valor_del_filtro_tiene_que_ser_booleano():
    with pytest.raises(ErrorDeConfiguración, match="filtro_por_metadata"):
        _rama(_indice(), filtro_por_metadata="si")


# ---------------------------------------------------------------- la identidad

def test_el_filtro_no_mueve_el_config_id_de_ninguna_fila_publicada():
    for capa, publicado in IDS_PUBLICADOS.items():
        cfg = Configuración.cargar(archivo=(RAÍZ_DEL_REPO / "config" / capa) if capa else None)
        assert cfg.config_id == publicado, f"{capa or 'default.toml'} cambió de identidad"
        assert cfg.get("recuperacion_textual.filtro_por_metadata") is None


def test_activar_el_filtro_mueve_solo_las_filas_que_tienen_rama_de_texto():
    clave = {"recuperacion_textual.filtro_por_metadata": True}
    for capa, publicado in IDS_PUBLICADOS.items():
        archivo = (RAÍZ_DEL_REPO / "config" / capa) if capa else None
        con = Configuración.cargar(archivo=archivo, overrides=clave)
        tiene_rama = "recuperador_textual" in con["consulta.etapas"]
        assert (con.config_id != publicado) == tiene_rama, capa


def test_las_capas_del_experimento_cargan_y_activan_el_filtro():
    for nombre in ("c3-filtro-metadata.toml", "c4-filtro-metadata.toml"):
        cfg = Configuración.cargar(archivo=RAÍZ_DEL_REPO / "config" / "analisis" / nombre)
        assert cfg["recuperacion_textual.filtro_por_metadata"] is True
        assert cfg["fusion.estrategia"] == "cascada"
        assert cfg.config_id not in IDS_PUBLICADOS.values()


def test_una_clave_inventada_sigue_siendo_un_error():
    with pytest.raises(ErrorDeConfiguración, match="clave desconocida"):
        Configuración.cargar(overrides={"recuperacion_textual.otro_filtro": True})


def test_el_armador_construye_la_cadena_con_el_filtro_activado(tmp_path):
    from tp3_rag.modulo_b.armador import ArmadorDePipeline
    from tp3_rag.dominio.plan import AlmacenDePlanes
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
    from tp3_rag.puertos.embedding_imagen import EmbeddingImagenFijo
    from tp3_rag.puertos.llm import LLMGrabado

    for nombre, esperado in (("c4-filtro-metadata.toml", True), (None, False)):
        cfg = Configuración.cargar(
            archivo=(RAÍZ_DEL_REPO / "config" / "analisis" / nombre) if nombre else RAÍZ_DEL_REPO / "config" / "c4.toml"
        )
        pipeline = ArmadorDePipeline(
            cfg,
            puerto_de_imagen=EmbeddingImagenFijo(),
            indice=ÍndiceVectorialEnMemoria(variante_id="v"),
            puerto_de_llm=LLMGrabado({}),
            embebedor_de_texto=EmbeddingTextoFijo(),
            almacen_de_planes=AlmacenDePlanes(AlmacenamientoEnArchivos(tmp_path), "planes"),
            bytes_de_la_imagen=lambda artwork_id: b"img",
        ).armar()
        rama = next(e for e in pipeline.etapas if e.NOMBRE == "recuperador_textual")
        assert rama.filtro_por_metadata is esperado
        assert rama.a_json()["filtro_por_metadata"] is esperado
