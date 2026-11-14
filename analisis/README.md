# Análisis de la cascada

Rama `analisis`. Estudia la cascada (el texto filtra, CLIP ordena, el reranker decide el
top-5) para decidir tres cosas: qué tan estricto tiene que ser el filtro, cuántas
candidatas tiene que ver el reranker y dónde se pierden las obras relevantes.

## Correr el análisis (no gasta nada)

```bash
python analisis/cascada.py
```

Lee `artifacts/corridas/` y `artifacts/ground-truth/`, y escribe en `artifacts/analisis/`:
`resultados.json`, `pares-sin-juzgar.json` y las figuras `embudo.png`, `filtro.png`,
`reranker.png` y `costo.png` (las figuras necesitan `matplotlib`: `poetry install --with figuras`).

Detecta solo las corridas en cascada y las agrupa por experimento
(`C4·filtro100·rerank20`, `C4·filtro100·rerank40`, …), así que al sumar corridas nuevas
alcanza con volver a correrlo.

## Experimentos nuevos (gastan API)

Necesitan `data/` (ver `ARTEFACTOS.md`) y `LLM_API_KEY` en `.env`. `--config` va antes
del subcomando.

| Config | Qué cambia | Pregunta | Llamadas LLM por consulta |
|---|---|---|---|
| `config/analisis/c4-rerank40.toml` | el reranker ve 40 candidatas | ¿la curva del reranker sigue subiendo después de 20? | ~42 |
| `config/analisis/c4-filtro20.toml` | el filtro deja pasar 20 fragmentos | ¿un filtro estricto mejora los pedidos conceptuales? ¿cuánto cuesta en los visuales? | ~22 |
| `config/analisis/c3-filtro20.toml` | el mismo filtro, sin reranker | separa el efecto del filtro del efecto del reranker | 2 |

```bash
V=c3274567371ff42b   # la variante de índice congelada
poetry run tp3-responder --config config/analisis/c3-filtro20.toml  --variante-id $V set
poetry run tp3-responder --config config/analisis/c4-filtro20.toml  --variante-id $V set
poetry run tp3-responder --config config/analisis/c4-rerank40.toml  --variante-id $V set
python analisis/cascada.py
```

Para probar antes de gastar las 30 consultas: agregar `--consulta c01-conceptual` después de `set`.

## Juzgar los pares nuevos

Después de correr, `artifacts/analisis/pares-sin-juzgar.json` lista los pares
(consulta, obra) del top-5 que nadie juzgó, en el formato de `tp3-anotar juez --solo-pares`.
Hasta que se juzguen, cuentan como no relevantes y las cifras de los experimentos nuevos
son cotas inferiores (columna `s/j %`).

Una vez juzgados, el análisis los suma con `--juicios-extra <archivo.csv>` (repetible).
Acepta cualquier CSV con las columnas `consulta_id`, `artwork_id` y `grado`, incluidos
los aportes de anotador y los del juez. El consenso original nunca se pisa.

## Cómo leer los resultados

- **Barridos simulados.** El del filtro recorta por posición en la rama de texto y el del
  reranker usa los puntajes ya guardados. No requieren corridas, pero el del filtro deja
  muchos pares sin juzgar en los tamaños chicos.
- **IC95.** Intervalo de confianza por bootstrap sobre las 30 consultas. Si los intervalos
  de dos configuraciones se pisan, mirar la diferencia pareada, que es más sensible.
- **Política adaptativa.** Elige la mejor configuración por tipo de pedido. Es un techo
  optimista: usa la categoría del set de evaluación y elige sobre las mismas consultas
  con las que mide.
- **USD.** Precio de lista sobre los tokens medidos. No incluye la llamada del
  planificador cuando el plan se reusa (unos USD 0,001 por consulta).

## Juzgar a mano

```bash
python analisis/planilla.py
```

Escribe `artifacts/analisis/planilla-juicios.html` con los pares de `pares-sin-juzgar.json`:
la imagen original, el pedido y cada obra, sin decir qué configuración la recuperó. Se abre
en el navegador desde esa carpeta, guarda el avance y exporta un CSV por anotador que
`cascada.py --juicios-extra` lee directamente.

## Filtro por metadata en pre-retrieval

Si el pedido nombra una cultura o un país, la búsqueda de texto se restringe con un `where`
de ChromaDB a las obras con esa `culture`, antes de la búsqueda por similitud (Clase 7.2:
metadata filtering). El detalle y sus límites están en `src/tp3_rag/modulo_b/filtro_de_metadata.py`.

Se activa con `recuperacion_textual.filtro_por_metadata = true`. La clave es opcional: como no
está en `default.toml`, las filas ya corridas conservan su `config_id`.

```bash
poetry run tp3-responder --config config/analisis/c3-filtro-metadata.toml --variante-id c3274567371ff42b set
poetry run tp3-responder --config config/analisis/c4-filtro-metadata.toml --variante-id c3274567371ff42b set
python analisis/cascada.py
```

Solo cambia el resultado de las 10 consultas culturales; en las otras 20 el filtro no se
aplica. Cada corrida registra qué filtro usó en `estado.sellos.filtro_por_metadata`.
