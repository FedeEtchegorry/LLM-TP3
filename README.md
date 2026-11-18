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
textual, una cascada en la que el texto filtra las obras elegibles y CLIP las
ordena, reranking multimodal, recolección de evidencia, generación y
renderizado. Cada etapa es omitible, y las cinco combinaciones que
se comparan son las filas de la tabla de ablación:

| Fila | Qué agrega |
|---|---|
| `C0` | solo imagen a CLIP, sin usar el pedido (piso de control) |
| `C1` | imagen + pedido crudo al text encoder, evidencia y explicación citada |
| `C2` | `C1` + pre-retrieval con LLM (reescritura y expansión de la consulta) |
| `C3` | `C2` + rama de texto sobre `knowledge` en cascada: el texto filtra, CLIP ordena |
| `C4` | `C3` + reranking multimodal con LLM |

Cada fila vive en `config/c0.toml` … `config/c4.toml`, como una capa sobre
`config/default.toml`. Las variantes del Ejercicio 4 están en `config/analisis/`:
filtro por metadata en la rama de texto (`c3-filtro-metadata.toml`,
`c4-filtro-metadata.toml`, la arquitectura final medida), cantidad de candidatas del
reranker (`*-rerank40.toml`), modelo del reranker (`c4-luna.toml`, `c4-terra.toml`) y
prompt v2 del reranker (`c4-reranker-v2.toml`, `c4-luna-v2.toml`). La arquitectura es de puertos y adaptadores: el código no
conoce proveedores, conoce puertos (catálogo de museo, embedding de texto,
embedding de imagen, índice vectorial, LLM, almacenamiento, HTTP, trazas).

## Instalación

Requiere **Python 3.10** y [Poetry](https://python-poetry.org/).

```bash
poetry install                       # núcleo + dependencias de test
poetry install --with modelos        # torch/CLIP, chromadb, openai, langchain
poetry install --with figuras        # solo si vas a generar la figura UMAP
poetry install --with modelos,ui     # interfaz local, se corre con: poetry run python ui/app.py
```

`torch` se instala desde el índice CPU de PyTorch: el modelo de imagen corre en
CPU y el paquete de PyPI arrastra dependencias de CUDA que no se usan.

### Variables de entorno

Van en un archivo `.env` en la raíz del repo (o en el entorno). **No se
versionan.**

| Variable | Para qué | ¿Obligatoria? |
|---|---|---|
| `TP3_CONTACTO_AIC` | contacto del header `AIC-User-Agent`, que AIC exige en API e imágenes | sí, para la ingesta |
| `LLM_API_KEY` | key de OpenAI, que va a `api.openai.com` (se acepta `OPENAI_API_KEY` como alias) | sí, desde el indexado de texto |
| `LANGSMITH_API_KEY` | trazas | solo si `trazas.proveedor = "langsmith"` |

El host del proveedor **no** es un secreto: sale de `proveedor.base_url` en
`config/default.toml`. La key es de un proxy, así que sin ese host la llamada
termina en un 401 contra otro proveedor.

## Cómo se regeneran los artefactos

Los artefactos que costaron llamadas a la API (corridas, planes, juicios y
métricas) **están versionados** en `artifacts/`: con ellos todos los análisis y las
figuras se reproducen sin key (ver `ARTEFACTOS.md` y la sección siguiente). Lo que no
está en el repo es lo pesado y regenerable sin gasto: catálogo, imágenes, caché de
embeddings e índices Chroma (`data/`, ignorado por git).

Los comandos de abajo rearman todo desde cero, en este orden. Cada paso es
idempotente y reanudable: lo que ya está hecho se lee en vez de recalcularse.

**La condición, dicha de frente: los pasos que llaman al proveedor necesitan una
key del proxy con presupuesto disponible.** El ambiente del curso es compartido y
tiene un tope por grupo; el tope del grupo que escribió esto **está agotado**, así
que con esa key estos comandos no corren hoy — y con una key igualmente topeada
tampoco. El gasto medido del proyecto completo fue de **~7.313 requests al
proxy**, contados del libro mayor que cada artefacto emite: `reranker` 3.920,
`embedding_texto` 1.896, `generador` 1.043, `juez` ~391 y `planificador` 63. Es
una **cota inferior**: la verdad de facturación es el dashboard del proxy, y las
dos cifras no se promedian.

Los pasos **1, 2, 3 y 7** no llaman al proveedor y corren sin key (el embebido de
imagen es CLIP local). Los pasos **4, 5, 6 y 8** sí, y los que gastan lo avisan
antes de gastar: el del juez no corre el pool completo sin `--confirmar-gasto`.

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
poetry run tp3-responder set                                  # C1, la configuración por defecto
poetry run tp3-responder --config config/c4.toml set          # otra fila: --config va ANTES del subcomando
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
poetry run tp3-anotar juez --limite 5         # LLM-judge sobre el pool · GASTA
poetry run tp3-anotar juez-kappa              # los tres κ del juez
poetry run tp3-metricas qs
poetry run tp3-metricas tabla                 # la tabla C0–C4 completa
poetry run tp3-metricas costo                 # el gasto medido, por rol y por etapa
```

Todos los entrypoints aceptan `--config <archivo.toml>` para elegir la capa de
configuración y `--set clave=valor` para pisar una clave suelta; las dos opciones
van antes del subcomando. `config/` trae `c0`, `c2`, `c3` y `c4`: C1 es `default.toml`. Una clave
desconocida es un error: `config/default.toml` es el esquema.

## Análisis y resultados (Ejercicio 4)

Los scripts de `analisis/` leen `artifacts/corridas/` y los juicios
(`artifacts/ground-truth/` más los `artifacts/analisis/juicios-*.csv` hechos a mano) y
escriben tablas, JSON y figuras en `artifacts/analisis/`. **No gastan API.** El detalle
de cada uno está en `analisis/README.md`.

```bash
python analisis/cascada.py        # resultados.json, embudo, filtro, candidatas del reranker, costo
python analisis/ablacion.py       # tabla C0–C4: nDCG@5, P@5, MRR, QS, costo y latencia
python analisis/mixtas.py         # set de 10 consultas mixtas (rúbrica propia)
python analisis/ruido.py          # ruido entre repeticiones de C3 y C4
python analisis/confianza.py      # aviso por confianza: precisión y cobertura según el umbral
python analisis/figuras_deck.py   # figuras de la presentación (modelos, prompt, filtro, aviso, prefiltro…)
```

Los `analisis/correr-*.ps1` son los scripts que corrieron los experimentos que sí
gastan (con tope de USD y una consulta de prueba antes de cada set).

### Resultados principales

Set de 30 consultas (10 visuales, 10 conceptuales, 10 ambiguas), 952 juicios de
relevancia de 0 a 2, todos los pares del top-5 juzgados. Una mejora cuenta solo si el
IC 95 % de la diferencia pareada no cruza el 0.

| Configuración | nDCG@5 | P@5 estricta | QS | USD por consulta | Latencia media |
|---|---|---|---|---|---|
| C0 · solo imagen | 0,363 | 0,213 | 0,000 | 0 | 0,6 s |
| C1 · + pedido crudo | 0,350 | 0,200 | 0,419 | 0,0029 | 11 s |
| C2 · + reescritura | 0,469 | 0,287 | 0,426 | 0,0038 | 11 s |
| C3 · + cascada con filtro por metadata | 0,626 | 0,487 | 0,826 | 0,0038 | 7,5 s |
| C4 · + reranker (gpt-4.1-mini, prompt v1) | 0,749 | 0,607 | 0,897 | 0,0282 | 35,5 s |

Variantes del reranker sobre C4 (mismo filtro y 20 candidatas):

| Reranker | nDCG@5 | Δ contra C4 (IC 95 %) | USD por consulta |
|---|---|---|---|
| gpt-4.1-mini · prompt v2 | 0,832 | +0,083 (0,029 a 0,139) | 0,0313 |
| gpt-6-luna · prompt v1 | 0,832 | +0,083 (0,023 a 0,148) | 0,0096 |
| gpt-6-luna · prompt v2 | 0,827 | +0,078 (0,025 a 0,132) | 0,0110 |
| gpt-6.1-terra · prompt v1 | 0,791 | +0,042 (−0,007 a 0,096) | 0,0585 |

- El pedido pasa a definir el resultado: QS de 0 a 0,90, con el nDCG@5 duplicado.
- La cascada con filtro por metadata es la mayor mejora y no usa LLM; actúa cuando el
  pedido nombra una restricción (conceptuales 0,28 → 0,92, también en las mixtas).
- Las visuales y ambiguas solo mejoran con un mejor juicio del reranker. Luna, el
  prompt v2 y su combinación llegan al mismo nDCG@5 (≈0,83): los efectos no se suman
  porque corrigen el mismo problema.
- El reranker es el 86 % del costo y lleva la consulta a 35 s (mini) o ~80 s (luna).
- El aviso por confianza (puntaje medio del reranker < 7) detecta las 9 listas malas
  del set, con 7 falsas alarmas.

Los informes completos están en `artifacts/informes/` (`ejercicio-3.md`,
`ejercicio-4.md`, `presentacion-final.md`).

## Interfaz

```bash
poetry install --with modelos,ui
poetry run python ui/app.py
```

Responde consultas ad hoc con cualquier configuración, compara C0 a C4 sobre la misma
consulta (`ui/comparar.py`), muestra el aviso por confianza y la sugerencia del
prefiltro para pedidos vagos, y tiene una pestaña para juzgar pares a mano
(`ui/clasificador.py`).

## Tests

```bash
poetry run pytest                     # los 1.139
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
  modulo_c/      evaluación: set de consultas, pool, planilla, acuerdo, juez, métricas, validadores
  cli/           un entrypoint por etapa
config/          default.toml (el esquema) + una capa por fila de la ablación
config/analisis/ las variantes del Ejercicio 4
analisis/        scripts de análisis y figuras (no gastan) y scripts de corrida (.ps1)
ui/              interfaz local
artifacts/       corridas, planes, juicios, resultados, figuras e informes
docs/            consignas y rúbrica del set mixto
tests/           1.139 tests, con sus fixtures
```

## Licencias del corpus

El corpus es de dominio público, pero **la licencia es por campo de texto, no por
obra**: la `description` de AIC es CC-BY-4.0 y exige atribución, mientras otros
campos son CC0. Esa licencia viaja desde la ingesta hasta la cita renderizada, y
el renderizador **falla** en vez de degradar si una cita que exige atribución no
la lleva: una respuesta sin explicación es un resultado pobre, pero una cita sin
atribución es una infracción.

---

Trabajo práctico 3 — 73.69 Large Language Models, ITBA, 2026. Grupo 2: Matías
Sapino, Federico Etchegorry y Julieta Techenski.
