from __future__ import annotations

import json
from pathlib import Path

import pytest

from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.puertos.cliente_http import ClienteHTTP

FIXTURES = Path(__file__).parent / "fixtures"
N_FIXTURE = 8

@pytest.fixture(scope="session")
def cfg() -> Configuración:
    return Configuración.cargar(overrides={"ingesta.limite_n": N_FIXTURE})

@pytest.fixture(scope="session")
def credenciales_de_test() -> Credenciales:
    return Credenciales(contacto_aic="tests@itba.local")

@pytest.fixture
def cliente_sin_red(cfg, credenciales_de_test) -> ClienteHTTP:
    cliente = ClienteHTTP.desde_configuración(cfg, credenciales_de_test, red_habilitada=False)
    cliente.cache_dir = FIXTURES / "http-cache"
    return cliente

SPLIT_DE_PRUEBA = "0123456789abcdef"
VARIANTE_DE_PRUEBA = "fedcba9876543210"
CONFIG_DE_PRUEBA = "aaaabbbbccccdddd"

@pytest.fixture
def almacen_temporal(tmp_path):
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    return AlmacenamientoEnArchivos(tmp_path)

@pytest.fixture
def rutas_de_prueba(cfg):
    from tp3_rag.dominio import ground_truth as gt

    return gt.rutas(cfg)

def manifiesto_de_prueba(*, ids_corpus=None, ids_held_out=None, split_id=SPLIT_DE_PRUEBA):
    from tp3_rag.dominio.split import ManifiestoDeSplit

    corpus = list(ids_corpus or [f"cma:{n}" for n in range(100, 160)])
    held = list(ids_held_out or [f"aic:{n}" for n in range(900, 915)])
    manifiesto = ManifiestoDeSplit(
        {
            "split_id": split_id,
            "criterio_de_inclusion": {"description_no_vacia": True, "imagen_real": True},
            "semilla": 7369,
            "proporcion_held_out": 0.1,
            "definicion_de_estrato": {"ejes": ["museo", "periodo"]},
            "conteos": {
                "corpus_indexado": len(corpus),
                "held_out": len(held),
                "elegibles_con_texto_e_imagen": len(corpus) + len(held),
            },
            "corpus_indexado": {"ids": corpus},
            "held_out": {"ids": held},
            "procedencia": {"catalogo_id": "cafecafecafecafe"},
        }
    )
    manifiesto.validar()
    return manifiesto

def set_de_prueba(*, split_id=SPLIT_DE_PRUEBA, autoria="grupo-de-prueba", n_imagenes=10):
    from tp3_rag.dominio.consultas import CATEGORÍAS, CATEGORÍA_CULTURAL, SetDeConsultas
    from tp3_rag.nucleo import identidad

    imágenes = []
    for n in range(1, n_imagenes + 1):
        artwork_id = f"aic:{899 + n}"
        consultas = []
        for categoría in CATEGORÍAS:
            sufijo = categoría.split("_")[0]
            consulta = {
                "consulta_id": f"c{n:02d}-{sufijo}",
                "categoria": categoría,
                "pedido": f"pedido {sufijo} de la obra {n}",
                "es_ejemplo": False,
            }
            if categoría == CATEGORÍA_CULTURAL:
                consulta["cultura_pedida"] = {
                    "nombre": "la India",
                    "etiquetas": ["India", "Mughal India"],
                    "n": 7,
                    "n_por_etiqueta": {"India": 5, "Mughal India": 4},
                    "poblacion": "corpus de prueba",
                    "n_poblacion": 60,
                    "criterio_de_culture": "primer_componente_por_coma",
                }
            consultas.append(consulta)
        imágenes.append(
            {
                "artwork_id": artwork_id,
                "imagen_hash": f"sha256:{n:064d}",
                "archivo": f"artifacts/ground-truth/consultas/imagenes/{artwork_id}.jpg",
                "atributos": ["paisaje", "paleta fría"],
                "es_ejemplo": False,
                "consultas": consultas,
                "procedencia": {"title": f"obra {n}", "artist": "alguien"},
            }
        )
    parcial = SetDeConsultas(
        {
            "identidad": identidad.identidad_del_set_de_consultas(
                split_id=split_id, autoria=autoria
            ),
            "split_id": split_id,
            "autoria": autoria,
            "estado": "completo",
            "criterio_de_culture": "primer_componente_por_coma",
            "revision": {"n_imagenes_revisadas": 30},
            "imagenes": imágenes,
            "conteos": {},
        }
    )
    consultas = parcial.consultas()
    set_ = SetDeConsultas(
        {
            **parcial.contenido,
            "conteos": {
                "n_imagenes": len(imágenes),
                "n_consultas": len(consultas),
                "n_pedidos_culturales": sum(1 for c in consultas if c.cultura_pedida),
                "por_categoria": parcial.por_categoria(),
            },
        }
    )
    set_.validar()
    return set_

def plan_de_prueba(
    *,
    plan_id: str = "0f1e2d3c4b5a6978",
    consulta_id: str = "c01-visual",
    visual: str = "warm sunlit rural landscape",
    conceptual: str = "nineteenth-century pastoral landscape painting",
    pedido: str = "versiones más cálidas de este paisaje",
    procedencia: str = "set",
    reusado: bool = False,
):
    from tp3_rag.dominio.plan import PlanDeConsulta

    plan = PlanDeConsulta(
        plan_id=plan_id,
        consulta_id=consulta_id,
        imagen_hash=f"sha256:{0:064d}",
        pedido=pedido,
        consulta_visual=visual,
        consulta_conceptual=conceptual,
        binding={"rol": "planificador", "modelo": "openai/gpt-4.1-mini"},
        uso={"tokens_in": 900, "tokens_out": 60},
        procedencia=procedencia,
        reusado=reusado,
    )
    return plan.a_json()

def corrida_de_prueba(
    *,
    consulta_id: str,
    top5,
    configuracion: str = "C1",
    ordinal: int = 1,
    degradaciones=(),
    fallida=None,
    config_id: str = CONFIG_DE_PRUEBA,
    split_id: str = SPLIT_DE_PRUEBA,
    variante_id: str = VARIANTE_DE_PRUEBA,
    sin_ficha=False,
    plan=None,
    alfa: float = 0.5,
    alfa_congelado: bool = False,
    procedencia: str = "set",
    ficha_extra=None,
):
    from tp3_rag.dominio.corrida import Corrida
    from tp3_rag.nucleo import identidad

    commit = "0" * 40
    obras = [
        {
            "artwork_id": artwork_id,
            "rango": n,
            "score": 0.9 - n / 100,
            "source_url": f"https://museo/{artwork_id}",
            "ficha": {
                "title": f"obra {artwork_id}",
                "artist": "alguien",
                "culture": "France",
                "year": 1890 + sum(ord(c) for c in artwork_id) % 100,
                **(ficha_extra or {}),
            },
        }
        for n, artwork_id in enumerate(top5, 1)
    ]
    return Corrida(
        {
            "artefacto": "Corrida",
            "version_de_la_corrida": "1",
            "corrida_id": identidad.corrida_id(
                config_id=config_id, consulta_id=consulta_id, commit=commit, ordinal=ordinal
            ),
            "config_id": config_id,
            "consulta_id": consulta_id,
            "ordinal": ordinal,
            "pipeline": {
                "configuracion": configuracion,
                "cadena": (["planificador"] if plan else [])
                + ["recuperador_visual", "recolector_de_evidencia",
                   "generador", "renderizador"],
                "etapas": [{"etapa": "recuperador_visual", "alfa": alfa, "top_k": 5}],
            },
            "configuracion_resuelta": {
                "consulta": {"alfa": alfa, "alfa_congelado": alfa_congelado}
            },
            "sellos": {
                "commit": commit,
                "config_id": config_id,
                "split_id": split_id,
                "variante_id": variante_id,
            },
            "estado": {
                "entrada": {"procedencia": procedencia, "consulta_id": consulta_id},
                "top5": list(top5),
                "plan": plan,
                "degradaciones": [dict(d) for d in degradaciones],
                "fallida": fallida,
                "salida": {"obras": [] if sin_ficha else obras},
                "claves_pobladas": (["planificador"] if plan else [])
                + ["recuperador_visual", "recolector_de_evidencia",
                   "generador", "renderizador"],
                "sellos": {"commit": commit},
            },
        }
    )

def corridas_de_prueba(set_, **kwargs):
    salida = []
    for n, consulta in enumerate(set_.consultas()):
        top5 = [f"cma:{100 + (n * 5 + k) % 60}" for k in range(5)]
        salida.append(
            corrida_de_prueba(consulta_id=consulta.consulta_id, top5=top5, **kwargs)
        )
    return salida

CONFIG_IDS_DE_PRUEBA: dict[str, str] = {
    "C0": "c0c0c0c0c0c0c0c0",
    "C1": CONFIG_DE_PRUEBA,
    "C2": "c2c2c2c2c2c2c2c2",
    "C3": "c3c3c3c3c3c3c3c3",
    "C4": "c4c4c4c4c4c4c4c4",
}

def top5_de_prueba(fila: str, n: int) -> list[str]:
    f = list(CONFIG_IDS_DE_PRUEBA).index(fila)
    compartidas = [f"cma:{100 + (n * 3 + k) % 40}" for k in range(3)]
    propias = [f"cma:{140 + (f * 4 + k + n) % 20}" for k in range(2)]
    return compartidas + propias

def corridas_de_las_cinco_filas(set_, *, filas=None, alfa_congelado=True, **kwargs):
    filas = list(filas or CONFIG_IDS_DE_PRUEBA)
    salida = []
    for fila in filas:
        for n, consulta in enumerate(set_.consultas()):
            salida.append(
                corrida_de_prueba(
                    consulta_id=consulta.consulta_id,
                    top5=top5_de_prueba(fila, n),
                    configuracion=fila,
                    config_id=CONFIG_IDS_DE_PRUEBA[fila],
                    alfa=1.0 if fila == "C0" else 0.5,
                    alfa_congelado=alfa_congelado,
                    **kwargs,
                )
            )
    return salida

def registro_de_variante_de_prueba(
    *, variante_id=VARIANTE_DE_PRUEBA, split_id=SPLIT_DE_PRUEBA
):
    from tp3_rag.dominio.variante import INGREDIENTES_DEL_ID, RegistroDeÍndices

    receta = {k: f"valor-{k}" for k in INGREDIENTES_DEL_ID}
    receta["split_id"] = split_id
    registro = RegistroDeÍndices(
        {
            "variante_id": variante_id,
            "receta": receta,
            "conteos": {"n_indexadas": 60, "n_fragmentos": 140},
            "colecciones": {"artworks": "x", "knowledge": "y"},
            "procedencia": {"embebido": {"catalogo_id": "cafecafecafecafe", "ordinal": 1}},
            "invariantes_de_9_6": {},
        }
    )
    registro.validar()
    return registro

def rubrica_de_prueba(*, designacion="piloto", estado="completa", refina=None):
    from tp3_rag.dominio.rubrica import Rúbrica

    front = [
        "+++",
        f'rubrica_id = "{designacion}"',
        f'estado = "{estado}"',
        *([f'refina = "{refina}"'] if refina else []),
        "+++",
        "",
    ]
    cuerpo = []
    for grado, qué in ((2, "cumple claramente"), (1, "cumple parcialmente"), (0, "no cumple")):
        cuerpo += [f"## Grado {grado} — {qué}", "", f"definición operativa de {grado}.", ""]
    cuerpo += [
        "## Ejemplos", "", "| grado | por qué |", "|---|---|", "| 2 | porque sí |", "",
        "## Desempate", "", "entre 2 y 1 gana 1.", "",
    ]
    return Rúbrica.desde_texto("\n".join(front + cuerpo))

@pytest.fixture(scope="session")
def crudos_aic() -> list[dict]:
    return json.loads((FIXTURES / "crudos-aic.json").read_text(encoding="utf-8"))

@pytest.fixture(scope="session")
def crudos_cma() -> list[dict]:
    return json.loads((FIXTURES / "crudos-cma.json").read_text(encoding="utf-8"))

@pytest.fixture(scope="session")
def corot(crudos_cma) -> dict:
    for c in crudos_cma:
        if c["id"] == 124078:
            return c
    pytest.fail("falta cma:124078 en el fixture; corré tests/fixtures/construir_fixtures.py")

@pytest.fixture(scope="session")
def aic_sin_description(crudos_aic) -> dict:
    for c in crudos_aic:
        if not c.get("description"):
            return c
    pytest.fail("ningún AIC del fixture viene sin `description`")
