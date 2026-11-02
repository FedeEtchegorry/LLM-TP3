# Los artefactos — cómo rearmar el proyecto sin gastar una request

Esta rama es `main` **más los datos y los resultados**. El código, los tests y la configuración son
los mismos del commit que se entregó; lo que agrega es `artifacts/`, el set de consultas congelado
y la configuración de Poetry.

> 🔴 **Por qué existe.** Producir estos artefactos costó **~7.313 requests** al proxy de modelos del
> curso, y **el tope de USD 10 del grupo está agotado**. No se pueden volver a generar. Sin ellos se
> puede correr el sistema, pero **no se puede reproducir ninguna de las mediciones que el informe
> publica**.

---

## Qué hay

| Ruta | Qué es | Se puede regenerar |
|---|---|---|
| **`artifacts/corridas/`** | **1.151 corridas** — cada una es una ejecución completa de la cadena sobre una consulta, con su evidencia, su salida citada y su libro mayor de tokens por rol | 🔴 **no: cuesta proxy** |
| **`artifacts/planes/`** | **62 planes** del pre-retrieval con LLM (reescritura y expansión del pedido) | 🔴 **no: cuesta proxy** |
| **`artifacts/ground-truth/`** | el pool de evaluación, **6 aportes de anotación**, el consenso de **379 juicios** (grados `0`: 166 · `1`: 97 · `2`: 116), el κ de Cohen, la rúbrica `v1` y su decisión, y la revisión de **25 explicaciones** | 🔴 **no: es trabajo de anotación** |
| **`artifacts/ground-truth/pool/…-secundario-incremental.*`** | el pool del estudio secundario, con **31 juicios** más que los 379 | 🔴 **no** |
| **`artifacts/informes/`** | los informes de los Ejercicios 3 y 4, la presentación, y las tablas que citan (la tabla de ablación, κ del juez, retrieval secundario) | — son los entregables |
| **`artifacts/indices/`** | las **3 recetas de variante de índice** registradas, con su identidad: `c3274567371ff42b` (la congelada, 6.232 fragmentos) · `a02f862565637f06` (campos ampliados, 12.772) · `d8da543eb94034ca` (chunking ciego, 2.775) | ✅ sí, **sin key** |
| **`artifacts/splits/`** · **`artifacts/ingesta/`** · **`artifacts/embebido/`** | el manifiesto de split **congelado**, y los reportes de ingesta y de embebido | ✅ sí, **sin key** |
| **`consultas-g2.toml`** | el `SetDeConsultas`: 10 imágenes del held-out × 3 pedidos. **Congelado**: es el instrumento de medición | ✅ está acá |

### Lo que no está en el repo, y de eso sólo una parte hace falta bajar

`data/` son **3,8 GB** y no entra en un repo de git. **Pero casi todo se regenera gratis**, así que
lo único que hay que bajar son **192 MB**:

| Ruta | Peso | Cómo se obtiene |
|---|---:|---|
| `data/imagenes/` | 1,7 G | ✅ **`tp3-ingesta`** las baja de los museos. Sin key |
| `data/http-cache/` | 1,8 G | ✅ **es** el caché de esas bajadas: se llena solo |
| `data/catalogo/` | 53 M | ✅ **`tp3-ingesta`** lo deriva del caché. Sin key |
| `data/embeddings-imagen/` | 24 M | ✅ **`tp3-embeber`**: CLIP corre **local**, sin key. Sólo CPU |
| `data/tokenizadores/` | 1,7 M | ✅ se descarga solo |
| 🔴 **`data/chroma/`** | **303 M** | 🔴 **hay que bajarlo: está publicado como archivo adjunto de la *release* `datos-v1`** |

🔴 **Por qué `chroma/` es la excepción.** Tiene dos colecciones: `artworks`, que son embeddings de
imagen con CLIP **local**, y **`knowledge`, que son embeddings de TEXTO con `text-embedding-3-small`
— y eso pasa por el proxy.** Rearmarlo cuesta plata, y el tope del grupo está agotado. Es la única
pieza de `data/` que no se puede reconstruir gratis.

```bash
# bajarlo y descomprimirlo en su lugar
gh release download datos-v1 --pattern chroma.tar.gz
tar -xzf chroma.tar.gz -C data/
```

**192 MB comprimido**, y trae las tres variantes de índice. `sha256`:
`2a62ce42a276a5d05350dc1c7c8981e022411edd7f80a55f61c59ead18793658`.

⚠️ **Con `chroma/` bajado y el resto regenerado, el sistema responde consultas nuevas sin gastar una
request** — el único paso que llamaría al proveedor es volver a construir el índice de texto, y
justamente eso es lo que la *release* evita.

---

## Cómo arrancar

```bash
poetry install
```

Las variables van en un `.env` en la raíz —el `README` las lista— y **no hay `.env.example`**: la
única obligatoria para lo que gasta es `LLM_API_KEY`. **Sin ella, lo de abajo igual corre.**

**Sin key del proxy** corren los pasos **1, 2, 3 y 7** del `README` —ingesta, split, embebido de
imágenes con CLIP, y los validadores de los criterios de entrega—. Eso reconstruye todo `data/`:

```bash
poetry run tp3-ingesta      # los dos museos, normalizados; baja las imágenes
poetry run tp3-split        # congela el manifiesto; correrlo dos veces no mueve nada
poetry run tp3-embeber      # CLIP sobre el universo con imagen (lo más largo, y reanudable)
poetry run tp3-indexar      # fragmenta, embebe el texto y arma las dos colecciones
```

⚠️ **El paso 4 (`tp3-indexar`) embebe texto y eso sí llama al proveedor.** Con
`--hasta fragmentos` llega hasta la fragmentación **sin tocar la red**.

**Para leer los resultados no hace falta ninguna key.** Estos tres están **corridos y verificados**
sobre los artefactos de esta rama, sin `LLM_API_KEY` en el entorno y **sin escribir nada**:

```bash
poetry run tp3-indexar --mostrar          # las tres variantes registradas y sus recetas
poetry run tp3-consultas mostrar          # el set de evaluación congelado

poetry run tp3-metricas --variante-id c3274567371ff42b \
         tabla --pool 5b6cba32126fed61-definitivo
```

⚠️ **Casi nada tiene valor por omisión, y es a propósito.** Hay **3 variantes de índice** y **2
pools congelados**, así que *cuál se mide* es una decisión declarada y no un `mtime`. Si falta un
argumento, el error te dice cuáles hay y cómo se llaman.

🔴 **Dos avisos sobre los otros dos comandos:**

- **`tp3-validar criterios` necesita `TP3_CONTACTO_AIC` y red** —no una key del proveedor—: uno de
  los criterios resuelve las URLs de las fichas en los museos, y sin ese contacto en el header AIC
  devuelve 403.
- 🔴 **`tp3-anotar kappa` ESCRIBE: regenera el consenso y el κ, que son artefactos sellados.** Los
  valores no cambian, pero **el archivo sí**: el inventario de aportes se actualiza y la prosa del
  encabezado se reescribe, porque el código publicado tiene los identificadores internos
  reemplazados por prosa. **Si lo corrés, vas a ver los dos archivos modificados en `git status`** —
  y lo correcto es revertirlos, no commitearlos.

---

## Dos cosas que conviene saber antes de tocar algo

**1. Varios artefactos están congelados a propósito, y editarlos rompe la comparación.** El
manifiesto de split, el set de consultas, el pool, los aportes de anotación y la rúbrica `v1` tienen
su identidad **declarada, no derivada**: se escriben una vez. Corregir uno después de congelarlo no
lo edita — **autora otro**, y los juicios del anterior quedan sin con qué compararse.

**2. Las corridas tienen la `entrada` vacía, y no es un bug.** La política de retención del proyecto
vacía los campos de entrada al persistir. Los pedidos que se ven en el texto generado son **los 30
del set de evaluación**, que es nuestro propio held-out.

---

## Las tres rúbricas conservan su texto original, y es una decisión

🔴 **`ground-truth/rubricas/v1.md`, `v1-decision-de-a5.toml` y `piloto.md` son el único lugar de este
repo que cita documentos internos** —31 identificadores y 26 referencias—. **No es un olvido.**

Cada juicio de anotación lleva el **hash del texto de la rúbrica con la que se juzgó**, y ese hash
está escrito adentro de **15 artefactos**: el consenso, el κ, la decisión de `A5`, los **cuatro
aportes de anotación** y los tres archivos del juez. 🔑 **El hash existe exactamente para eso: atar
cada juicio al instrumento con el que se emitió.**

Reescribir el texto de una rúbrica cambia su hash, y entonces habría que reescribir los 15 —o sea
**el registro de anotación entero**—, afirmando que los anotadores leyeron una rúbrica **que no
existía cuando anotaron**. ⚠️ **Por eso se conservan tal cual, con sus referencias adentro**, y esta
sección es la excepción declarada.

## Una advertencia sobre las cifras

Toda cifra de este proyecto **va con la población sobre la que se midió**, y dos números correctos
pueden no ser comparables. Tres casos concretos que están en el informe:

- La suite da **1.121** tests en la rama de trabajo y **1.119** en el repo publicado: son **dos
  árboles**, porque el publicado no lleva `artifacts/`.
- El `nDCG` **no es estable ante extensiones del pool**: el ideal se computa sobre la población del
  ground truth, así que agregar juicios mueve el denominador. La fila `C3` da **0,5348** sobre los
  379 juicios y **0,5308** sobre el ground truth extendido, **y las dos son correctas**.
- Las métricas de ranking se publican sobre **dos poblaciones** —la completa y la restringida a los
  pares donde los dos anotadores coincidieron— y **van juntas**, porque en el estudio secundario el
  orden de las filas **cambia** entre una y otra.
