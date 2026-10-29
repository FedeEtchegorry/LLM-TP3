# TP3 — RAG multimodal sobre pinturas de AIC y CMA

Sistema de recuperación aumentada por generación (RAG) **multimodal** sobre un
corpus de pinturas de dominio público del Art Institute of Chicago (AIC) y el
Cleveland Museum of Art (CMA).

La entrada es **una imagen más un pedido en lenguaje natural** ("algo parecido
pero más oscuro", "otra obra de la misma época"); la salida son **cinco obras con
una explicación citada**, donde cada afirmación sobre una obra lleva una
referencia a un fragmento de texto del museo, con su licencia y su atribución.

## Cómo está armado

Dos colecciones vectoriales sobre el mismo corpus:

- `artworks` — un vector CLIP por obra, para la rama visual.
- `knowledge` — fragmentos del texto de museo embebidos con
  `text-embedding-3-small`, para la rama conceptual.

El pipeline de consulta es una **cadena de etapas seleccionada por
configuración**: pre-retrieval con LLM, recuperación visual, recuperación
textual, fusión por rangos, recolección de evidencia, reranking multimodal,
generación y renderizado. Cada etapa es omitible, y las cinco combinaciones que
se comparan son las filas de la tabla de ablación:

| Fila | Qué agrega |
|---|---|
| `C0` | solo imagen a CLIP, sin usar el pedido (piso de control) |
| `C1` | imagen + pedido crudo al text encoder, evidencia y explicación citada |
| `C2` | `C1` + pre-retrieval con LLM (reescritura y expansión de la consulta) |
| `C3` | `C2` + rama de texto sobre `knowledge` y fusión por rangos |
| `C4` | `C3` + reranking multimodal con LLM |

Cada fila vive en `config/c0.toml` … `config/c4.toml`, como una capa sobre
`config/default.toml`. La arquitectura es de puertos y adaptadores: el código no
conoce proveedores, conoce puertos (catálogo de museo, embedding de texto,
embedding de imagen, índice vectorial, LLM, almacenamiento, HTTP, trazas).

## Instalación

Requiere **Python 3.10** y [Poetry](https://python-poetry.org/).

```bash
poetry install                       # núcleo + dependencias de test
poetry install --with modelos        # torch/CLIP, chromadb, openai, langchain
poetry install --with figuras        # solo si vas a generar la figura UMAP
```

`torch` se instala desde el índice CPU de PyTorch: el modelo de imagen corre en
CPU y el paquete de PyPI arrastra dependencias de CUDA que no se usan.

### Variables de entorno

Van en un archivo `.env` en la raíz del repo (o en el entorno). **No se
versionan.**

| Variable | Para qué | ¿Obligatoria? |
|---|---|---|
| `TP3_CONTACTO_AIC` | contacto del header `AIC-User-Agent`, que AIC exige en API e imágenes | sí, para la ingesta |
| `LLM_API_KEY` | key del proxy de modelos (se acepta `OPENAI_API_KEY` como alias) | sí, desde el indexado de texto |
| `LANGSMITH_API_KEY` | trazas | solo si `trazas.proveedor = "langsmith"` |

El host del proveedor **no** es un secreto: sale de `proveedor.base_url` en
`config/default.toml`. La key es de un proxy, así que sin ese host la llamada
termina en un 401 contra otro proveedor.

## Cómo se regeneran los artefactos

Los artefactos (catálogo, imágenes, caché de embeddings, índices Chroma,
corridas, planes) **no están versionados**: se regeneran con estos comandos, en
este orden. Cada paso es idempotente y reanudable: lo que ya está hecho se lee en
vez de recalcularse.

```bash
# 1 · Ingesta — recorre los dos museos, normaliza y baja las imágenes.
#     Produce el catálogo canónico y el reporte de ingesta.
poetry run tp3-ingesta                        # universo completo
poetry run tp3-ingesta --limite-n 20          # corrida corta, para probar

# 2 · Split — parte el corpus elegible en corpus indexado / held-out y CONGELA
#     el manifiesto. Correrlo dos veces no mueve nada.
poetry run tp3-split

# 3 · Embebido de imágenes — CLIP sobre el universo con imagen, escribiendo el
#     caché. Es la etapa más larga en CPU, y es reanudable.
poetry run tp3-embeber

# 4 · Indexado — fragmenta el texto, embebe, construye las dos colecciones y
#     registra la variante con su receta.
poetry run tp3-indexar
poetry run tp3-indexar --hasta fragmentos     # sin tocar la red ni el LLM
poetry run tp3-indexar --mostrar              # leer el registro de índices

# 5 · Set de consultas — las consultas de evaluación. Se autoran una vez y se
#     congelan: de ahí en más se leen.
poetry run tp3-consultas plantilla            # emite el TOML para completar
poetry run tp3-consultas autorar --declaracion <archivo.toml>

# 6 · Consulta — responde una consulta del set o una ad-hoc.
poetry run tp3-responder set --config config/c1.toml
poetry run tp3-responder adhoc --imagen <ruta.jpg> --pedido "algo más oscuro"
poetry run tp3-responder mostrar --corrida <corrida_id>

# 7 · Validadores — los criterios de entrega, corridos como aserciones sobre los
#     artefactos. No construyen el pipeline ni piden la key: leen de disco.
poetry run tp3-validar criterios

# 8 · Anotación y métricas.
poetry run tp3-anotar rubrica                 # plantilla de la rúbrica
poetry run tp3-anotar pool --estado piloto    # deriva el pool a anotar
poetry run tp3-anotar exportar                # planilla para los anotadores
poetry run tp3-anotar importar                # congela los juicios
poetry run tp3-anotar kappa --anotador a --anotador b
poetry run tp3-metricas qs
```

Todos los entrypoints aceptan `--config <archivo.toml>` para elegir la capa de
configuración y `--set clave=valor` para pisar una clave suelta. Una clave
desconocida es un error: `config/default.toml` es el esquema.

## Tests

```bash
poetry run pytest                     # los 917
poetry run pytest -m nivel1           # unitarios, sin red ni API
poetry run pytest -m nivel2           # validadores de contrato
poetry run pytest -m nivel3           # sanity y regresión, necesitan índice
```

Los tests unitarios **no tocan la red**: el caché HTTP de `tests/fixtures/` es el
fixture, y un fallo de caché es un fallo de test en vez de una llamada. Por eso
`tests/fixtures/` está versionado aunque pese unos megas.

## Estructura

```
src/tp3_rag/
  nucleo/        configuración, credenciales, identidad, procedencia, observabilidad
  puertos/       los puertos y sus adaptadores (museos, embeddings, índice, LLM, …)
  dominio/       los tipos del dominio: catálogo, split, fragmentos, consulta, corrida
  modulo_a/      ingesta, normalización, selección de corpus, split, fragmentado, índice
  modulo_b/      consulta: armador, ejecutor, etapas, fusión, generación, renderizado
  modulo_c/      evaluación: set de consultas, pool, planilla, acuerdo, métricas, validadores
  cli/           un entrypoint por etapa
config/          default.toml (el esquema) + una capa por fila de la ablación
tests/           917 tests, con sus fixtures
```

## Licencias del corpus

El corpus es de dominio público, pero **la licencia es por campo de texto, no por
obra**: la `description` de AIC es CC-BY-4.0 y exige atribución, mientras otros
campos son CC0. Esa licencia viaja desde la ingesta hasta la cita renderizada, y
el renderizador **falla** en vez de degradar si una cita que exige atribución no
la lleva: una respuesta sin explicación es un resultado pobre, pero una cita sin
atribución es una infracción.

---

Trabajo práctico 3 — 73.69 Procesamiento de Lenguaje Natural con LLMs, ITBA,
2026. Grupo 2.
