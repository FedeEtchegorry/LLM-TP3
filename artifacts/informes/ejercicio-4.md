# Ejercicio 4 — La iteración: de `v1` a `v2`

**Grupo 2 · 73.69 Large Language Models · ITBA, 2026**

El sistema recibe **una imagen y un pedido en lenguaje natural** y devuelve **5 pinturas reales**
del Art Institute of Chicago (AIC) y del Cleveland Museum of Art (CMA), cada una con una
explicación en español citada a fragmentos de texto de los museos. La propiedad que lo define es
falsable: *la misma imagen con pedidos distintos tiene que devolver obras distintas.*

**`v1` = `C1`** (RAG base completo y honesto: rama visual + evidencia + generación citada) ·
**`v2` = `C4`** (pre-retrieval con LLM + rama de texto + fusión RRF + reranking multimodal).
Lo fija el enunciado y `la evaluación § Las configuraciones` define las cinco por **inclusión incremental**.

**Poblaciones, porque las cifras no son intercambiables.** Universo elegible **5.915** pinturas de
licencia libre · universo con imagen **5.751** · **corpus indexado 2.605** en **6.232 fragmentos**
(variante congelada `c3274567371ff42b`) · held-out **289** · **30 consultas** (10 imágenes × 3
pedidos) · pool congelado `5b6cba32126fed61-definitivo` con **379 ítems** sobre 202 obras ·
rúbrica `v1` · y un **segundo cuerpo de anotación**, presupuestado aparte: **25 explicaciones
(5 consultas × 5 obras) y 51 afirmaciones** para la regla de honestidad (§5.5).

> ### 📐 El formato lo elegimos nosotros, y es una decisión sobre qué entra
>
> el enunciado fija **cuatro cosas** para el Ejercicio 4 —limitación, modificación con su
> justificación, configuración antes y después, resultados comparados— y **no fija formato ni
> extensión** (a diferencia del Ejercicio 3, que pide ≤ 2 hojas). Entonces: **cuatro secciones con
> los títulos del enunciado**, y tres decisiones más.
>
> **(1) Cada cifra lleva su población y su artefacto en la misma línea.** Este proyecto registra
> trece correcciones de cifras publicadas sin su población (*modo de falla 3*), y el
> formato es la mitigación: si una cifra no se puede citar, no entra.
> **(2) Las limitaciones van en el cuerpo (§6), no en una nota al pie.** el acuerdo entre anotadores compromete
> *reportar el acuerdo pase lo que pase*, y la anotación por IA eligió la salida *limitación declarada*.
> Una limitación al pie es una limitación que nadie lee.
> **(3) Donde el resultado depende de una elección metodológica, van las dos lecturas.** No se
> elige la que cierra mejor. Pasa **tres veces** —§1.2, §4.2 y §5.3b— y las tres están publicadas
> enteras. ⚠️ **En la tercera las dos lecturas no se reconcilian, y el no-resultado es lo que se
> publica.**

---

## 1. El problema o limitación encontrado en `v1`

### 1.1 El mecanismo, y por qué era predecible

`v1` embebe **el pedido crudo con el text encoder de CLIP** y lo mezcla con el embedding de la
imagen: `e_q = normalize(α·e_img + (1−α)·e_txt)`, con `α = 0,5`.

CLIP está entrenado con **captions descriptivos**, no con instrucciones de transformación.
*"versiones más cálidas"* **no es la descripción de ninguna imagen**: es una instrucción relativa a
una referencia que el text encoder no ve. El vector resultante no apunta a nada útil, y el pedido
—que debería **definir cuál es el resultado correcto**— termina actuando como ruido sobre el
vector de la imagen.

**Y hay un segundo mecanismo, que medimos antes de tener la métrica con la que confirmarlo.** La
figura UMAP del 2026-10-18 sobre las 2.605 indexadas midió que **el formato físico separa el
espacio de CLIP más que la cultura**: κ del vecino **0,663** para *folio de manuscrito vs. pintura
de caballete* (n = 2.605, 2 clases, k = 10, pureza 0,894 sobre un azar de 0,685) contra **0,459**
para `culture` (n = 2.594, 9 clases, pureza 0,550 sobre un azar de 0,168) —
`artifacts/informes/umap-procedencia.json`, figura `umap-05-separacion.png`. Se sostiene **dentro**
de cada cultura con masa de los dos formatos (China 0,54 · otras 0,52 · Mughal India 0,49;
`umap-espacio-clip.ipynb`), ⚠️ **y no se sostiene en Northern India (0,02)**, que es el control que
va dicho igual.

La consecuencia operativa es un **filtro que nadie puso**: en la rama visual, quien sube una
pintura de caballete recibe pinturas de caballete **para cualquier pedido**. Y en el corpus la
cultura y el formato están casi perfectamente confundidos. Entonces **un pedido cultural cuya
cultura vive en folios no lo puede resolver la rama visual**. De ahí salió una predicción escrita
el 18/10, antes de que `nDCG@5` existiera: *el delta de agregar la rama de texto tiene que
concentrarse en los pedidos culturales.* Se confirma en §4.3.

### 1.2 🔴 Confirmado con número — y el criterio se cumple en una población y no en la otra

la documentación del cuarto ejercicio pide la limitación de `v1` **confirmada con número**: `nDCG@5` de `C1` en pedidos
*ambiguos* contra las otras dos categorías. **El resultado depende de qué consenso se use, y se
publican los dos.** Fuente: `artifacts/informes/tabla-a2-m13.json`, claves
`por_categoria_completa` / `por_categoria_restringida`, métrica `ndcg_at_5_sobre_todas`.

| `nDCG@5(C1)` | `visual_cromatico` | `conceptual_cultural` | `ambiguo` | ¿el peor es `ambiguo`? |
|---|---:|---:|---:|---|
| **(a) consenso completo — 379 pares** | 0,435 | **0,318** | 0,364 | ✗ **no** |
| **(b) solo acuerdo — 246 pares** | 0,397 | 0,407 | **0,292** | ✓ **sí** |

**La frase publicable es la que lleva las dos:** *en pedidos ambiguos `C1` es el peor de los tres
**solo cuando se saca el tercio disputado**; sobre el consenso completo el peor es el conceptual.*

🔴 **Publicar solo (b) sería publicar el criterio dado vuelta.** Es la dirección cómoda que este
repositorio documenta una docena de veces, y no se toma.

⚠️ **Y la otra mitad de la documentación del cuarto ejercicio es NO COMPUTABLE por construcción, no baja.** No existe un *"`QS` de
`C1` en pedidos ambiguos"*: `QS = 1 − media(Jaccard(top5_i, top5_j))` se define sobre **los 3
pedidos de una misma imagen** (`la evaluación §B`) y los 3 son **uno por categoría** — un `QS` por
categoría tendría **un** top-5 por imagen, y Jaccard necesita dos. **Ausente, no bajo.**

Lo que sí existe es `QS` **por configuración**, sobre las 10 imágenes, y la escalera es la de la
modificación (`tabla-a2-m13.json`, `columnas_sin_ground_truth.*.qs`):

| | `C0` | `C1` = v1 | `C2` | `C3` | `C4` = v2 |
|---|---:|---:|---:|---:|---:|
| **`QS`** | **0,000** exacto | 0,419 | 0,426 | 0,727 | **0,900** |

**`QS(C0) = 0,000` exacto es la demostración de que el criterio es falsable**, no una métrica que
salió bien: con `α = 1` la combinación cortocircuita y devuelve `e_img` sin tocar, así que el
vector de consulta **no depende del pedido en ningún bit** y las tres consultas de una imagen dan
el mismo top-5. Es `los criterios de entrega el criterio de que el pedido cambia el resultado` verificado por construcción.
⚠️ **El `0,419` de `QS(C1)` no tiene nada que ver con el `κ = 0,419` del piloto.** Coinciden al
tercer decimal y son dos mediciones de cosas distintas; no se citan juntas.

### 1.3 Y una limitación del instrumento, medida cuando corregirla costaba 20 juicios

La ronda piloto mostró que la rúbrica **admite dos lecturas del mismo par**: `cma:124080` da
**0** bajo la lectura de encuadre y **2** bajo la de intimidad, con el mismo texto delante
(`el registro de decisiones `). Es un defecto **del texto de la rúbrica**, no del anotador. Entra como resultado
porque muy pocos trabajos miden su propio instrumento.

---

## 2. La modificación propuesta y su justificación

### 2.1 Qué se agrega, y por qué cada pieza

| | Qué agrega | Por qué, y contra qué mecanismo de §1 |
|---|---|---|
| **`C2`** | **Pre-retrieval con un LLM multimodal**: *query rewriting* + *query expansion* en **una sola llamada** | El LLM **ve la imagen y el pedido juntos**, así que puede resolver *"más cálidas **respecto de este gris**"* y emitir una **descripción absoluta** que CLIP sí entiende. Ataca el mecanismo de §1.1 en su raíz: deja de embeber una instrucción relativa |
| **`C3`** | **Rama de texto sobre `knowledge` + fusión RRF** | Un pedido cultural **cruza formatos físicos** y la rama visual no (§1.1). El texto del museo sí nombra la cultura. Es la pieza que la predicción de la UMAP señalaba |
| **`C4`** | **Reranking con LLM multimodal**, un candidato por llamada, score 0–10 con justificación | El retrieval ordena por similitud; **cumplir un pedido no es ser similar**. El reranker es el único componente que juzga *el pedido* con la imagen candidata a la vista |

**Las cuatro técnicas de las técnicas avanzadas aisladas por medición, cada una con su artefacto verificable** — y la evidencia es el
artefacto, no el conteo de trazas:

| Técnica | Etapa | Artefacto que la prueba | Instancia en disco |
|---|---|---|---|
| **Query rewriting** | pre-retrieval | `PlanDeConsulta`, clave `pl3.distintas_del_pedido_crudo` | `artifacts/planes/0911f8725c7e813d.json`: `pedido` *"más cercano"* → `consulta_visual` de 14 palabras |
| **Query expansion** | pre-retrieval | el mismo plan, clave `pl3.distintas_entre_si` — **dos** consultas de una llamada | mismo archivo: `consulta_visual` ≠ `consulta_conceptual`, las tres distintas |
| **Fusión RRF** | post-retrieval | `corrida.json`, `estado.ranking_fusionado` con `rangos_de_origen` **por rama** | `artifacts/corridas/e3821d8533de06a3-1/corrida.json`: `cma:155097` entra con `{texto: 12, visual: 7}`, `n_ramas: 2` |
| **Reranking** | post-retrieval | el mismo `corrida.json`, `estado.rerank` con `score_0_10` + `justificacion` + `rango_fusionado` | mismo archivo: `cma:149126`, `score_0_10: 9`, promovido del puesto 2 al 1 |

🔑 **Las cuatro se pueden mostrar sobre una sola consulta de punta a punta**: el plan
`artifacts/planes/d408abc4d01bdd73.json` (pedido *"escenas de agua en la pintura china"*) es el
que consume esa corrida, y la `justificacion` del reranker **le devuelve el pedido reescrito**.

⚠️ **las técnicas avanzadas aisladas por medición se cumple con cuatro y los deltas aíslan de a dos**: `C2−C1` (rewriting + expansion,
conjuntas), `C3−C2` (rama de texto + RRF, conjuntas), `C4−C3` (reranking, **aislado**). No se
reclama aislamiento individual de las cuatro. **Y el filtrado por metadata no se presenta como
quinta técnica** (el registro de decisiones, decidido el 28/10): el sistema usa `where` como **join** para traer
los fragmentos de una obra por `artwork_id`, no para angostar el espacio de búsqueda.

### 2.2 La justificación de la fusión dejó de ser un argumento y pasó a ser un número

la fusión recíproca de rangos con k = 60 eligió **RRF con `k = 60`** sobre la suma de scores normalizados con un argumento:
*un coseno de CLIP y uno de `text-embedding-3-small` no son calibrables entre sí*, así que se
fusiona por **rango** y el problema desaparece. El barrido lo midió:

| Estrategia de fusión | `QS` | Cobertura cultural | Churn vs. RRF |
|---|---:|---:|---:|
| **`rrf` (k = 60)** | **0,727** | 26/50 | — |
| `suma_de_scores_normalizados` | 0,682 | 26/50 | 1,43/5 |
| `concatenacion` | 0,649 | 23/50 | 2,07/5 |

**Es exactamente el orden que el argumento predijo:** la única estrategia que **suma dos escalas**
queda en el medio, y la que **ignora el acuerdo entre ramas** es la peor.
⚠️ **Y las tres cifras no tienen la misma procedencia, así que van etiquetadas distinto.** El
**0,727** lo emite `artifacts/informes/tabla-a2-m13.json` (fila `C3`). El **0,682** y el **0,649**
viven en el bloque `[fusion]` de `config/default.toml`: los tres brazos
corrieron y están en disco (`config_id` `ce0910b50e3569e9`, `f954237a74462e66`, `b2f6389c73f7da2e`,
30 corridas cada uno), pero **la comparación nunca se escribió a un artefacto**. Son medidas, no
emitidas.

El mismo criterio aplica a **`α = 0,5`**, y la elección tiene una propiedad que conviene contar
porque **demuestra por qué la vara tiene que declararse antes del barrido**:

| `α` | `QS` | Cobertura cultural | Anclaje a la imagen (top-1) |
|---:|---:|---:|---|
| 0 | **0,821** | 32/50 | 🔴 **0/20** — y las 20 devuelven **la misma obra** |
| 0,25 | 0,753 | 28/50 | 19/20 |
| **0,5** ← congelado | **0,727** | 26/50 | **20/20** |
| 0,75 | 0,675 | 27/50 | 20/20 |
| 1 | 0,626 | 22/50 | 20/20 |

🔑 **`QS` es monótona decreciente en `α`, así que maximizarla sola empuja a `α = 0`** — que da la
`QS` más alta de las cinco **con anclaje 0/20 y las 20 obras devolviendo la misma**. **Es `C0` por
el otro lado:** `C0` ignora el pedido, `α = 0` ignora la imagen. **Es el ejemplo más limpio de que
*`QS` alta sola no significa nada*** (`la evaluación §B`: un sistema aleatorio tiene `QS ≈ 1`), y la razón
por la que el criterio —*la `QS` más alta entre los `α` que conservan el anclaje intacto, top-1
20/20*— **se declaró antes de mirar el barrido**. Con el umbral en top-5 en vez de top-1 habría
ganado `α = 0,25`, con **+0,026 de `QS`**; moverlo después habría sido **elegir la vara después de
ver a qué `α` favorece**, y no se hizo.
⚠️ **Misma salvedad de procedencia que arriba:** esta tabla vive en `config/default.toml` y **no**
en un artefacto. Los cinco brazos corrieron y están en disco (30 corridas cada uno).

### 2.3 🔑 La alternativa que no tomamos: **cascada** (un criterio filtra, el otro ordena)

*Escrito el 2026-11-03, a partir de una recomendación recibida en clase.* Nos plantearon una
objeción que vale la pena contestar entera, porque es la primera que a cualquiera se le ocurre:
**¿cómo se compara un resultado de texto con uno de imagen para decir cuál es mejor?** Y la salida
sugerida fue una **cascada**: que un criterio *filtre* y el otro *ordene*, sin sumar puntajes.

**La objeción no aplica a este sistema, y el motivo es estructural: lo único que rankeamos son
obras.** Cada candidata es una obra que tiene imagen *y* texto, y el anotador juzga el par
`(consulta, obra)` con **una sola** rúbrica 0/1/2. No existen en el sistema dos tipos de ítem que
haya que poner en una misma lista. Conviene separar tres cosas que la pregunta junta:

1. **Qué se rankea** → obras. Un solo tipo de ítem, de punta a punta.
2. **Cómo cada rama propone candidatas** → la rama de texto busca **fragmentos** en `knowledge`,
   pero cada fragmento lleva `artwork_id` y se agrega a nivel obra (`mejor_fragmento`). Las dos
   ramas terminan en **un ranking del mismo universo**.
3. **Cómo se combinan los dos rankings** → **RRF, que usa solo rangos**. `fusionador.py` no lee
   `score` en ninguna línea, y hay un test que lo prueba por la vía que importa: *permutar los
   scores dentro de una rama sin cambiar sus rangos no mueve el ranking fusionado.*

🔑 **O sea que RRF está en el diseño precisamente por esa objeción** (la fusión recíproca de rangos con k = 60): un coseno de
CLIP y uno de `text-embedding-3-small` no son calibrables entre sí, así que se fusiona por rango y
el problema desaparece. La §2.2 lo midió: el único brazo que **sí** suma dos escalas
(`suma_de_scores_normalizados`) quedó en el medio.

> **Dónde sí mezclamos modalidades, que es la versión honesta de la objeción:** no en la fusión,
> sino **adentro de la rama visual**, en `e_q = normalize(α·e_img + (1−α)·e_txt)`. Ahí se suma un
> vector de imagen con uno de texto y el *modality gap* es real. La mitigación está en §3.2
> (normalizar antes y después, técnica documentada de la bibliografía de la Clase 9) y el control
> empírico es la tabla de `α` de §2.2: con `α = 0` el anclaje se cae a **0/20**.

**Y la cascada se puede evaluar sin gastar una llamada**, porque las corridas persisten las dos
ramas completas con sus rangos (`estado.fuentes_de_ranking`). Medido sobre **las 30 corridas
publicadas de `C3`**:

| | |
|---|---:|
| candidatas de la rama visual | 20,0 (fijo, `N`) |
| candidatas de la rama de texto | 49,1 |
| **unión** (lo que RRF considera) | **65,1** |
| 🔴 **intersección visual ∩ texto** (lo que una cascada dejaría elegible) | **3,97** — mín. **0** |
| consultas donde la intersección **no llega a 5** | **19 de 30** |

Y la procedencia del top-5 que efectivamente se publica (150 ítems por fila):

| | solo visual | solo texto | ambas ramas |
|---|---:|---:|---:|
| `C3` | 17,3 % | 18,7 % | 64,0 % |
| **`C4` = v2** | **23,3 %** | 38,7 % | 38,0 % |

**Tres razones para no cambiar, en orden de peso:**

1. 🔴 **La cascada deja afuera el 23,3 % del top-5 de `v2`.** Si la rama de texto decide quién es
   elegible, las obras que **solo** propuso la visual dejan de existir. Y el riesgo lo nombra la
   propia recomendación —*«si el filtro deja afuera la obra correcta, CLIP no la puede
   recuperar»*—: con una intersección media de **3,97**, en **19 de 30 consultas la cascada no
   llegaría a juntar 5 candidatas**. No es un riesgo teórico, es el régimen normal de este corpus.
2. 🔴 **Deshace el mecanismo que `C3` vino a arreglar.** §1.1 midió que el espacio de CLIP está
   organizado por **formato físico** (κ = 0,663) más que por **cultura** (κ = 0,459), y en el
   corpus las dos cosas están casi perfectamente confundidas (0 óleos indios, 0 folios franceses,
   Japón 96 % folio). Si CLIP queda como **único ordenador final**, ordena por formato — que es
   exactamente *el filtro que nadie puso*. Lo que se perdería tiene número: el `Δ C3−C2` de
   cobertura cultural, **+2,00 de 5**.
3. **Contradice el entregable aprobado.** la entrega aprobada describe la etapa 4 como *"las obras
   que aparecen en ambas ramas **suben** en el ranking, y las que aparecen en una sola **se
   conservan**, sin duplicados"*. Eso es **unión con premio al acuerdo**, que es RRF. Una cascada
   es intersección: lo que aparece en una sola rama **muere**. Y rompería la inclusión incremental
   de §3.1 —`C3` dejaría de ser `C2` más una etapa— con la que el enunciado pide *"cambios
   incrementales"*.

⚠️ **Lo que la cascada sí tiene de bueno, y dónde ya está en el sistema.** Su variante *(a)* —el
pedido conceptual se traduce en un **filtro por metadata** y lo visual ordena— es *metadata
filtering* de la Clase 7.2, y es una técnica legítima. En este sistema el `where` existe pero como
**join** (traer los fragmentos de una obra por `artwork_id`, §3.1), no como filtro del espacio de
búsqueda, y por eso el registro de decisiones decidió **no presentarla como quinta técnica**. Convertirla en
filtro duro sobre `culture` choca además con un problema abierto y medido: la normalización de
`culture` mezcla lugar y período y no une sinónimos (`Holland`/`Netherlands`), así que un filtro
duro sobre ese campo **sería frágil justo en los pedidos culturales**, que son los que debería
ayudar.

📌 **Qué queda sin medir, y se dice:** no corrimos la cascada de punta a punta. **No es una
omisión de criterio: el tope del proxy está agotado** (§5.4), y la variante *(b)* además exige
puntuar con CLIP las ~49 candidatas del texto, de las cuales hoy solo ~4 tienen rango visual. Lo
que está medido es **la estructura del corpus de candidatas**, que es lo que decide si la cascada
puede funcionar — y dice que no.

---

## 3. La configuración antes y después

### 3.1 La cadena de etapas — es lo único que cambia entre las cinco filas

`config/default.toml` + `config/c0.toml … c4.toml`. **Inclusión incremental estricta**: cada
cadena es un prefijo de la siguiente con una etapa más.

| | `planificador` | `recuperador_visual` | `recuperador_textual` | `fusionador` | `reranker` | `recolector_de_evidencia` | `generador` | `renderizador` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `C0` | | ✓ | | | | | | ✓ |
| **`C1` = v1** | | ✓ | | | | ✓ | ✓ | ✓ |
| `C2` | ✓ | ✓ | | | | ✓ | ✓ | ✓ |
| `C3` | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| **`C4` = v2** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**v1 → v2 son cuatro etapas nuevas: `planificador`, `recuperador_textual`, `fusionador`,
`reranker`.** Nada se saca.

### 3.2 Lo que NO cambió, y es la mitad del diseño experimental

| Parámetro | Valor, congelado para las cinco filas |
|---|---|
| **Índice** | variante `c3274567371ff42b` — 2.605 obras, 6.232 fragmentos, chunking por campo semántico, campos `description` · `did_you_know` · `provenance` |
| **`α`** | **0,5**, `alfa_congelado = true`, el mismo en `C1`–`C4`. `C0` es `α = 1` **por construcción** (ignora el pedido) |
| **`N` candidatos / `top_k`** | 20 / 5 |
| **Fusión** | `rrf`, `k = 60` |
| **Evidencia** | `n_fragmentos_por_obra = 3`; recuperación textual 50 fragmentos, agregación `mejor_fragmento` |
| **Modelos** | `planificador` · `reranker` · `generador`: **`openai/gpt-4.1-mini`** · **juez: `openai/gpt-4o`**, *otro modelo a propósito* (el registro de decisiones, mitigación 3 de la circularidad) · texto: `openai/text-embedding-3-small` (1536 d) · imagen: **`openai/clip-vit-base-patch32`** local |
| **Muestreo** | `seed = 7369` en los cuatro roles; `temperature` 0,1 planificador · **0,0** reranker · 0,2 generador · 0,0 juez |
| **Pool y rúbrica** | `5b6cba32126fed61-definitivo` (379 ítems) y rúbrica `v1` — **sellados** |

🔑 **Por qué `α` congelado no es prolijidad:** si `C1` corriera con un `α` distinto de `C2`,
entonces `C2 − C1` mezclaría **dos** cambios —el pre-retrieval **y** el valor de `α`— y el delta
perdería su interpretación. El titular del ejercicio se cae con él.
📌 **Y el tag `v1` del repositorio no es la corrida que produce la fila `C1`.** el pipeline base completo y andando pide que `v1`
**corra las 30/30 y sea anterior** a `C2`–`C4`; el barrido de `α` no puede cerrar antes de que
`C3` y `C4` existan. El tag corre con un `α` provisorio y es un **hito de orden**; la fila `C1` se
computa después de congelar. el pipeline base completo y andando no se rompe — lo que faltaba era decirlo.

### 3.3 Las identidades, para que cada fila sea recomputable

| | `C0` | `C1` | `C2` | `C3` | `C4` |
|---|---|---|---|---|---|
| `config_id` | `44bd01270964aec2` | `4443862ae45e62d6` | `8c128ba44b0f67cc` | `ce0910b50e3569e9` | `e30ca8c5003c614c` |

📌 **La etiqueta describe, el `config_id` identifica.** Un brazo de barrido puede llevar la
etiqueta `C3` y **no** ser la fila `C3` publicada; pasa, y en §5.4 cuesta una cifra.

---

## 4. Los resultados comparados

### 4.1 La tabla, sobre el consenso completo (a) — 379 pares, política `cero`

`artifacts/informes/tabla-a2-m13.json`, clave `tabla_a_completa` · 30 consultas × 5 obras = 150
ítems por fila · `% sin juzgar = 0,0` en las cinco **por construcción** (el pool *es* la unión de
sus top-5; es una tautología, no una validación).

📌 **Regla de redondeo, declarada porque hay un empate exacto.** El artefacto emite **4 decimales**
y las tablas de este informe publican **3, redondeando el 5 hacia arriba**. Importa en una celda:
`nDCG@5(C2)` de la población (b) vale **0,4865** exacto y se publica **0,487**. Los **deltas se
calculan sobre los 4 decimales del artefacto**, no sobre los 3 publicados.

| Config | `nDCG@5` | `(def)` | `P@5` | `P@5` estricta | `MRR` | `QS` | p50 s | p95 s | llam. LLM / consulta | tokens in / consulta | USD / consulta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0` solo visual | 0,386 | 0,399 | 0,460 | 0,213 | 0,605 | **0,000** | 0,43 | 2,19 | 0,0 | 0 | **0,0000** |
| **`C1` = v1** | 0,372 | 0,385 | 0,487 | 0,200 | 0,592 | 0,419 | 10,58 | 15,29 | 1,03 | 3.722 | **0,0029** |
| `C2` + pre-retrieval | 0,498 | 0,515 | 0,607 | 0,287 | 0,684 | 0,426 | 9,57 | 19,78 | 2,0 | 5.942 | **0,0038** |
| `C3` + texto + RRF | 0,535 | 0,553 | 0,593 | 0,320 | 0,758 | 0,727 | 10,94 | 21,37 | 2,0 | 5.977 | **0,0038** |
| **`C4` = v2** | **0,743** | **0,769** | **0,733** | **0,560** | **0,867** | **0,900** | 49,44 | 65,34 | 22,0 | 61.733 | **0,0277** |

**El titular v1 → v2: `nDCG@5` 0,372 → 0,743, un delta de +0,371 — se duplica.** `P@5` estricta
—la fracción del top-5 con grado 2, la vara dura— va **0,200 → 0,560**, que es **2,8×**.
Y `QS` 0,419 → 0,900: el sistema pasa de cambiar poco el resultado según el pedido a cambiarlo
casi todo, **sin** perder precisión, que es la única combinación que `la evaluación §B` declara buena.

**El costo de v2 está medido y no es chico:** p95 **15,29 s → 65,34 s** (4,3×), tokens de entrada
por consulta **3.722 → 61.733** (17×) y **USD/consulta 0,0029 → 0,0277, que es 9,6×**, porque `C4`
paga 20 llamadas de reranking con imagen. **El `nDCG@5` se duplica y el precio se multiplica por
diez**, y las dos cifras van juntas: es el trade-off que la fila `C4` compra.

⚠️ **Una columna con dos promedios, y dos correcciones del 2026-11-03:**
**(1) `nDCG@5` publica dos promedios.** `c05-ambiguo` salió **15/15 en grado 0** ⇒ `IDCG = 0` ⇒
`nDCG` **ausente, no 0**. La columna principal divide por las 30 y cuenta 0 la indefinida —que es
lo que la decisión del acuerdo entre anotadores declara para esa consulta, *«no aporta a nDCG@5 y sí al denominador»*—;
`(def)` divide por las 29 definidas. **Se publican las dos en vez de elegir una en silencio.**

> ### 🔴 **(2) `USD/consulta` dejó de ser ausente, y no hizo falta re-correr nada**
> *2026-11-03. La ausencia tenía una condición escrita y se cumplió.*
>
> Esta columna decía **ausente** en las cinco filas porque `[proveedor]` declaraba el precio de
> `text-embedding-3-small` y de ningún modelo de chat, y *un modelo sin precio da ausente, no cero*
>. La misma regla dejaba dicho el remedio: *«el día que alguien lea el precio de
> `gpt-4.1-mini` **con su fecha y su URL**, se anota ahí y esta tabla publica USD sin tocar nada
> más»*. Se leyó el **2026-11-03** en `https://developers.openai.com/api/docs/pricing`:
> `gpt-4.1-mini` **0,40 / 1,60** y `gpt-4o` **2,50 / 10,00** por millón.
>
> **Costó cero llamadas al proxy**, que es el punto: los tokens ya estaban medidos del `usage` de
> la API en los libros mayores de las **1.151 corridas**. Y no podía ser de otro modo —el precio
> **no determina el `config_id`**—, así que declararlo **no huerfanó una sola corrida**.
> Regenerar `tabla-a2-m13.json` movió **exactamente 5 valores**: los cinco `usd_por_consulta`.
> Ningún `nDCG`, ningún `P@5`.
>
> ⚠️ **Lo que sigue siendo verdad, y hay que decirlo igual:** los **tokens son medidos** y el
> **USD es derivado de un precio declarado**. Y es el precio de **lista de OpenAI**, no la tarifa
> del proxy de ITBA, que nadie puede leer (`/key/info` bloqueado para virtual keys). La
> reconciliación contra el dashboard está en **§5.4** y **las dos cifras no se promedian**.
> ⚠️ **Y sigue sin tomarse de LangSmith**: con el nombre del modelo en la metadata, LangSmith
> computaría el costo de su propia tabla de precios y tendríamos **dos cifras del mismo número, la
> segunda actualizándose sola**.

> ### 🔴 **(3) Las columnas de llamadas y tokens estaban sub-reportadas en `C2`, `C3` y `C4`**
> *Encontrado el 2026-11-03 al derivar el USD, y es un error nuestro.*
>
> Esta tabla publicaba **`C2` y `C3` con 1,0 llamadas/consulta** cuando `los criterios de entrega el criterio de costo` compromete
> **2**, y `C4` con **21,0** cuando compromete **2 + N = 22**. La causa es el reuso del plan al barrer el peso de fusión: **un plan
> reusado no paga la llamada del planificador** —cinco brazos de un barrido comparten un plan—,
> así que el libro mayor de `C2` tiene **cero** llamadas de planificador y la columna, que leía
> solo el libro mayor, contaba solo el generador. **Los 30 planes de cada fila están reusados.**
>
> 🔑 **El caché del plan es un artefacto de nuestra ablación, no de producción: una consulta de un
> usuario paga su planificador siempre.** Entonces la columna correcta es la **nominal**, que suma
> un plan por consulta desde `estado.plan.uso` (y resta el planificador del libro mayor, que si no
> se contaría dos veces en la corrida que sí lo creó).
>
> | | llam./cons. | | tokens in/cons. | |
> |---|---:|---:|---:|---:|
> | | **publicado antes** | **corregido** | **publicado antes** | **corregido** |
> | `C2` | 1,0 | **2,0** | 3.604 | **5.942** (+65 %) |
> | `C3` | 1,0 | **2,0** | 3.639 | **5.977** (+64 %) |
> | `C4` | 21,0 | **22,0** | 59.395 | **61.733** (+3,9 %) |
>
> ✅ **Corregida, la columna cierra exacto contra el criterio de costo:** `C0 = 0 · C1 = 1 · C2 = C3 = 2 ·
> C4 = 2 + N`. **El validador del criterio de costo siempre contó bien** —lee el *estado*, no el libro mayor— así
> que el compromiso nunca estuvo incumplido: lo que estaba mal era **esta tabla**. Es otra
> vez: dos lecturas del mismo número, *pagada* y *nominal*, y la que se publica tiene que decir
> cuál es. **Las dos se emiten** (`tp3-metricas costo`).
> ⚠️ `C1` no se mueve: no tiene planificador en la cadena. Su **1,03** es el reintento.

**Groundedness automática** (150 obras por fila, misma clave del artefacto): **0 referencias
colgadas en las cinco filas**, y de las que existen, **100,0 % cita byte a byte** en `C1`–`C4`
(163 · 162 · 170 · 166 referencias). En `C0` es **ausente**: no hay generador, así que no hay
referencias. La **abstención** baja monótonamente con la cadena —**150 → 90 → 88 → 61 → 55**
obras—: cada etapa nueva le pone mejor evidencia delante al generador.
⚠️ **Esto es el conteo de referencias colgadas y es la mitad automática.** *Que el fragmento exista y se cite byte a byte* **no
es lo mismo que** *que el fragmento diga lo que la afirmación dice que dice*. La mitad fina
—la revisión humana y la cita que respalda la afirmación— se mide **afirmación por afirmación** y está en el **§5.5**, sobre 25
explicaciones y 51 afirmaciones. **Las dos cifras conviven y no se promedian.**

### 4.2 🔴 El orden de la tabla CAMBIA según el consenso, y se publican los dos

Los 379 pares del consenso son **246 de acuerdo + 133 resueltos por un tercer anotador**. La
población **(b)** restringe a los 246 donde los dos primeros ya coincidían.
Clave `tabla_b_restringida_condensada` / `_cero` y `sensibilidad_td_1`.

| | Orden por `nDCG@5` | `C4` − 2.º |
|---|---|---:|
| **(a) completa — 379 pares** | **`C4` > `C3` > `C2` > `C0` > `C1`** | **+0,208** |
| **(b) restringida — 246 pares** | **`C4` > `C2` > `C3` > `C1` > `C0`** | **+0,149** |

| Config | (a) | (b) `condensada` | (b) `cero` | `P@5` (b) | estricta (b) | % del top-5 sin juzgar en (b) |
|---|---:|---:|---:|---:|---:|---:|
| `C0` | 0,386 | 0,333 | 0,293 | 0,327 | 0,120 | **34,7 %** |
| `C1` | 0,372 | 0,365 | 0,327 | 0,353 | 0,127 | 36,7 % |
| `C2` | 0,498 | 0,487 | 0,412 | 0,447 | 0,193 | **40,7 %** |
| `C3` | 0,535 | 0,464 | 0,404 | 0,427 | 0,213 | 39,3 % |
| `C4` | **0,743** | **0,636** | **0,560** | **0,493** | **0,407** | 38,7 % |

**Dos permutas** —`C2↔C3` y `C0↔C1`— **y el orden cambia igual con las dos políticas** para el
ítem excluido (`condensada` y TREC-`cero`), **así que la conclusión no depende de esa elección**.
🔴 **El `C3 − C2` global cambia de signo: +0,037 → −0,022** (`condensada`; con `cero` es −0,008,
también negativo).

**Qué se dice y qué no:**

- ✅ **`C4` primero en las dos poblaciones**, por +0,208 en **(a)** y +0,149 en **(b)** sobre el
  segundo —que no es la misma fila en las dos—. **El titular v1 → v2 se sostiene**, y en (b)
  `C4 − C1` sigue siendo **+0,270**.
- ✅ **El desglose por categoría sobrevive** (§4.3). **Es más robusto que el agregado.**
- 🔴 **Nunca más *"`C3` le gana a `C2`"*.** Sobre el consenso completo sí; sobre los 246 de
  acuerdo, no. Entre el **34,7 %** y el **40,7 %** del top-5 de cada fila **depende del desempate**.

🔑 **Por qué existe la sensibilidad, contado y no escondido.** El desempate lo hizo un **tercer
anotador** que coincide con `ia-claude` en **95 de 133 pares (71 %)**, con `ia-codex` en **29
(22 %)** y con ninguno en **9 (7 %)** — y en `conceptual_cultural` es **27 de 28, el 96 %**
(` el desempate`). O sea que **el consenso del tercio disputado es, de hecho, el juicio de uno de
los dos anotadores**. **No hay un tercero neutral disponible** —el proxy de la cátedra sirve solo
modelos de OpenAI y el otro anotador es de otra familia—, así que **una sensibilidad es lo único
honesto** que se puede decir de ese tercio. Los 133 disputados son **35,1 % de los 379**, que es
exactamente la tasa que el piloto proyectó (7 de 20 = 35 %).

### 4.3 🔑 El resultado más fuerte: el delta por categoría de pedido confirmado con `nDCG@5`, y es ROBUSTO

la evaluación hipotetizaba, desde el diseño del set de consultas, que **la rama de texto aporta casi
todo en los pedidos conceptuales y casi nada en los cromáticos**. Es el objetivo el delta por categoría de pedido de
los criterios de entrega, y deliberadamente **no** es un criterio: que no aportara habría sido un resultado
válido. **Aportó, con el signo correcto, y en las dos poblaciones.**

**`C3 − C2`** (`ndcg_at_5_sobre_todas`, 10 consultas por celda). ⚠️ **Y la (b) de esta tabla es la
política `condensada`** —lo emite `por_categoria_restringida` y es su única política—, a diferencia
del `§4.2`, que publica las dos por separado:

| | (a) 379 pares | (b) 246 pares, `condensada` |
|---|---:|---:|
| **`conceptual_cultural`** | **+0,262** | **+0,161** |
| **`visual_cromatico`** | **−0,089** | **−0,211** |
| `ambiguo` | −0,062 | −0,017 |

**Positivo y grande en los conceptuales, negativo en los cromáticos, en las DOS poblaciones.** El
agregado se da vuelta entre (a) y (b); **el desglose no**. Por eso el titular del ejercicio es
éste y no el orden global.

🔑 **Y cierra un arco de dos semanas.** La figura UMAP del **18/10** midió que el formato físico
separa el espacio CLIP más que la cultura (§1.1) y de ahí se **escribió la predicción**: un pedido
cultural cruza formatos, la rama visual no, así que el delta `C3−C2` **tenía que concentrarse en
los culturales**. **La predicción se escribió antes de que la métrica existiera y se confirmó con
`nDCG@5` el 1/11.** No es una interpretación post-hoc: está fechada en los dos extremos.

**La tabla completa por categoría**, para que el arco se pueda auditar y no solo creer:

| | | `visual_cromatico` | `conceptual_cultural` | `ambiguo` |
|---|---|---:|---:|---:|
| **(a) 379** | `C1` = v1 | 0,435 | 0,318 | 0,364 |
| | `C2` | 0,577 | 0,390 | 0,527 |
| | `C3` | 0,488 | **0,652** | 0,465 |
| | **`C4` = v2** | **0,694** | **0,961** | **0,575** |
| **(b) 246** | `C1` = v1 | 0,397 | 0,407 | 0,292 |
| | `C2` | 0,616 | 0,475 | 0,369 |
| | `C3` | 0,405 | **0,636** | 0,351 |
| | **`C4` = v2** | **0,536** | **0,835** | **0,536** |

**`C4` en los conceptuales llega a 0,961 sobre (a)**, que es el número más alto de todo el estudio,
y **`C4 − C1` en esa categoría es +0,643 (a) / +0,428 (b)**. La historia del ejercicio está en esta
tabla: **el pedido conceptual era el peor para v1 y es el mejor para v2.**

📌 **Y el reparto de los 30 pedidos no fue arbitrario:** cinco pedidos culturales que **cruzan**
formato físico y cinco que se quedan adentro. Sin esa mezcla, un delta `C3−C2` sería atribuible al
formato y no a la cultura, y esta sección no podría distinguirlos.

**Como evidencia de mecanismo, no solo de efecto:** la cobertura cultural del top-5 sobre los 10
pedidos culturales va **0,60 → 2,60 → 4,50 de 5** en `C2 → C3 → C4`, con **`Δ C3−C2 = +2,00`**.
⚠️ **Con dos salvedades que van dichas:** la métrica cuenta contra el conjunto declarado en
`cultura_pedida`, que en una consulta es angosto; y un reconteo a mano del 30/10 dio **4,40** en
lugar de 4,50 para `C4` —una obra en 50—. **Se publica el 4,50 que emite el instrumento, con la
discrepancia al lado.**

### 4.4 El piso de ruido va con su población, y **no es el mismo para las cinco filas**

`tabla-a2-m13.json`, clave `piso_de_ruido_m9_1`. Dos corridas de la misma configuración, con
`temperature = 0,0` y `seed` fijo, no devuelven necesariamente el mismo top-5 — el reranker no es
determinista. **Lo que está medido, y lo que no, se etiqueta distinto:**

| Fila | Piso | Etiqueta |
|---|---|---|
| `C3` | **0,000 exacto** — top-5 idéntico **30/30**, churn 0,000 | ✅ **medido**, y en el `config_id` publicado |
| `C0` · `C1` · `C2` | — | ⚠️ **acotado por inclusión**: sus cadenas son **prefijos** de la de `C3`. **Argumento, no medición** |
| `C4` | **sin medir** en `e30ca8c5003c614c` | 🔴 La banda de **±0,8 obras de 5** y el piso de `QS` **0,249** salen de `7f257a8b9a209198` — **otro `config_id` con la misma etiqueta `C4`** |

🔑 **La banda de ±0,8 obras aplica a `C4` y a ninguna otra fila**, porque la fuente de no
determinismo que el proyecto midió es **el reranker** y el reranker solo está en `C4`. Pero está
medida en un `config_id` que no es el publicado, **y eso va dicho**: durante diez días se citó como
si fuera el de la fila. **Consecuencia práctica:** el `C4 − C3` de +0,208 **de la población (a)
—los 379 pares del consenso completo, política `cero`—** está muy por encima de la banda y se
sostiene; una diferencia de ±0,05 entre dos filas **no se distingue del ruido** con 10
consultas por celda, y así hay que leer el desglose del §4.3.

---

## 5. Seis resultados más del proyecto, fuera de la tabla `C0`–`C4`

### 5.1 🔑 El chunking que la entrega aprobada pedía tiene un costo de licencia, y está medido

Nos apartamos de los *"fragmentos de unos 500 tokens con solapamiento"* de la entrega aprobada y
fragmentamos **por campo semántico** (desvío, el registro de decisiones, **sin consultar** al 1/11). El
brazo ciego **se construyó igual y corrió**: variante `d8da543eb94034ca`, **2.775 fragmentos**,
1,07 por obra, 30 corridas sobre `C3` (`artifacts/indices/d8da543eb94034ca.json`).

**El chunking ciego fuerza atribución CC-BY sobre texto CC0 en el 21,2 % de las citas que publica
— 22 de 104 —**, porque en AIC `description` es **CC-BY-4.0** y el resto es CC0, y un chunk de 500
tokens los concatena: **442 de sus 2.775 fragmentos mezclan licencias**. El brazo por campo **no
puede producir ni una**: un fragmento de un campo tiene **una** licencia por construcción.
⚠️ **Y el `0` del brazo por campo NO es una medición, es una propiedad estructural.** Decirlo
importa: una cifra que sale de la definición no es evidencia del mismo tipo que una medida.

⚠️ **Las cifras van sobre los 103 pares `(consulta, obra)` comunes, y eso no es un detalle:** el
eje **también mueve el top-5**, así que 47 de 150 obras son distintas entre los dos brazos y **un
delta sobre poblaciones distintas no es un delta**. Las cifras sobre los 150-vs-150 no se publican.
Sobre la misma evidencia, el ciego **cita menos veces pero no menos obras**: **103/103** obras con
≥ 1 referencia contra 102/103, y **1,126 → 1,010** referencias por obra.
⚠️ **Y la mitad honesta: nada de esto dice que el ciego *recupere* peor.** Eso sería `nDCG@5`, y los
fragmentos de ese brazo **no tenían juicio por construcción** —el pool cuelga de la otra variante—:
**29 de 150** ítems de su top-5 quedaban sin juzgar. 🔑 **Esos 29 ya se cerraron con pooling
incremental y la pregunta está medida en el §5.3b — y la respuesta sigue siendo que no se puede
afirmar**, por otro motivo: el `Δ` contra `C3` **cambia de signo** según la población del ground
truth. **El desvío llega mitigado con números; no deja de ser un desvío por estar medido.**

### 5.2 🔑 `provenance` es el 39 % del índice congelado y NUNCA se cita

**2.408 de los 6.232 fragmentos** de la variante congelada son `provenance`
(`artifacts/indices/c3274567371ff42b.json`). Y en los dos brazos del estudio secundario: **128
ofrecidos al generador → 0 citados**, y **29 → 0**. el relevamiento lo había anticipado por argumento
—*"sirve para citar pero casi no ayuda a buscar"*— y **la medición dice que tampoco se cita**.

Es un **resultado negativo**, de los que más cuesta publicar y más valen: **el 39 % del índice no
aporta a la salida**, y eso es lo primero que hay que saber antes de pagar una estructura ANN sobre
él. ⚠️ **Y no se tocó**: cambiar los campos mueve el `variante_id`, y el pool congelado y la tabla
entera cuelgan de él. **Entra como resultado, no como cambio.**

### 5.3 🔑 El presupuesto de evidencia es de suma cero — agregar campos no agrega evidencia, la desplaza

`evidencia.n_fragmentos_por_obra = 3` es un **tope fijo**. Entonces sumar campos a `knowledge` no
amplía la evidencia: **compite por los mismos tres lugares**. Medido: `description` **142 → 136**
obras, `provenance` **128 → 29**, `did_you_know` **72 → 44**, y la **abstención sube 61 → 76 obras,
un +25 %** — con menos `description` delante, el generador se abstiene más.

**la evaluación escribía el eje como *"¿ayudan o meten ruido?"* y la respuesta medida es una tercera:
desplazan.** Reencuadra el pendiente entero: la pregunta no era *"¿agregamos campos?"* sino
**"¿qué campo sacamos para meterlos?"**.
⚠️ **Con su salvedad, porque el desplazamiento no es universal:** vale **donde el índice ofrece más
de 3 fragmentos por obra**, que es la variante ampliada (4,90 > 3). En el eje del brazo ciego **no
ocurre** — ese índice ofrece 2,39 y 1,07 por obra, por debajo del tope.

### 5.3b 🔴 Los mismos dos ejes del lado del *retrieval* — y el resultado es un NO-resultado

`§5.1` y `§5.3` publican los dos ejes del estudio secundario **del lado de la generación**
—licencias mezcladas, citas, desplazamiento de evidencia—. Ésta publica **los mismos dos ejes del
lado del *retrieval***, que es la mitad que faltaba. ⚠️ **Las dos mitades NO miden lo mismo y no se
promedian ni se suman:** aquello es groundedness y atribución, esto es calidad de ranking, y tienen
`n` y ground truth distintos.

**Cómo se midió, porque el instrumento es nuevo.** Los dos brazos corren sobre **otra variante de
índice**, así que sus top-5 traen pares que el pool congelado nunca juzgó. Se derivó un pool por
**pooling incremental** (`la evaluación § El pool se congela`, punto 4): identidad
`5b6cba32126fed61-secundario-incremental`, **197 pares distintos** sobre 30 consultas y 133 obras,
de los cuales **166 reusan el grado del consenso definitivo** y **31 son nuevos**, anotados por
`ia-incremental`. ⚠️ **El pool definitivo no se toca** — *"congelar" es derivar otro pool, no mutar
el mismo* (la evaluación, precisión del 12/10), y la tabla del §4.1 sigue colgando del suyo.
Fuente de todo lo que sigue: `artifacts/ground-truth/retrieval-secundario-m14b.json` y
`artifacts/ground-truth/pool/5b6cba32126fed61-secundario-incremental.json`. Métrica
`ndcg_at_5_sobre_todas` —la columna principal del §4.1: divide por las 30 y cuenta 0 la
indefinida— y `(def)`, que divide por las definidas, **las dos como en el §4.1**.

📌 **Y esta sección publica 4 decimales donde las tablas del §4 publican 3.** Es una excepción
declarada y tiene motivo: **dos de los tres deltas viven en la cuarta cifra**, y redondear a 3 los
borraría — lo cual es, precisamente, el resultado.

#### 🔴 La tabla, y el `Δ` cambia de signo entre las dos poblaciones de ground truth

Las dos poblaciones, con su composición, porque **no son las del §4.2**:
**(a) completa** = los **379** pares del consenso definitivo **+ los 31 nuevos = 410 pares con
grado** (246 de acuerdo · 133 de desempate · 31 de anotador único). Las tres filas tienen **0
ítems del top-5 sin grado**, así que `cero` y `condensada` coinciden y hay **una sola fila**.
**(b) restringida** = la restricción del desempate aplicada a esa población —saca los 133 que resolvió
el tercer anotador— = **277 pares** (246 + 31). Ítems del top-5 sin grado: **59 de 150** en `C3` y
en el ampliado, **55 de 150** en el ciego.

| Población del ground truth | `C3` | `C3`-ampliado | `C3`-ciego | `Δ ciego − C3` |
|---|---:|---:|---:|---:|
| **(a) completa — 410 pares** | 0,5308 | **0,5415** | 0,4903 | **−0,0405** |
| **(b) restringida — 277 pares, TREC-`cero`** | 0,3860 | 0,3898 | 0,3895 | **+0,0035** |
| **(b) restringida — 277 pares, `condensada`** | 0,4422 | 0,4468 | **0,4472** | **+0,0050** |

**Y el otro promedio, para que el no-resultado no dependa de la política de promediado:**

| Población | `C3` `(def)` | ampliado `(def)` | ciego `(def)` | `Δ` `(def)` |
|---|---:|---:|---:|---:|
| **(a) completa** — 29 de 30 definidas | 0,5491 | **0,5602** | 0,5072 | **−0,0419** |
| **(b) `cero`** — 26 de 30 definidas | 0,4454 | 0,4497 | 0,4494 | **+0,0040** |
| **(b) `condensada`** — 26 de 30 definidas | 0,5102 | 0,5155 | **0,5160** | **+0,0058** |

🔴 **El `Δ ciego − C3` CAMBIA DE SIGNO entre las dos poblaciones —con los dos promedios—, así que
la única afirmación publicable es que el eje de chunking NO mueve el retrieval de forma
concluyente.** *No* que el chunking ciego lo empeore. **Las tres filas van juntas, con sus dos
políticas, y la conclusión va en esa forma.** Es la sensibilidad del orden a la población un nivel más abajo: la misma sensibilidad
del desempate que da vuelta el orden de `C0`–`C4` en el §4.2 lo da vuelta acá, y la afirmación cómoda
—*"el chunking que la entrega aprobada pedía también recupera peor"*— **es justamente la que el
dato no sostiene**.

🔴 **Y el confundidor va acá, pegado a la tabla, porque sin él la tabla miente.** **La población
restringida no restringe igual a las tres filas.** La restricción del desempate saca los pares que
resolvió el tercer anotador, y **los 31 juicios nuevos no pasaron nunca por el filtro de
acuerdo/desempate** —no hay con quién acordar cuando el anotador es uno—, así que **sobreviven a la
restricción por construcción**. El brazo ciego tiene **29 de sus 150** ítems ahí y **`C3`
ninguno** (clave `composicion_del_ground_truth_por_fila`; el ampliado, 3). Consecuencia directa: en
la población (b) el ciego **conserva más ítems con grado que `C3`** —**36,67 %** sin grado contra
**39,33 %**— **por el protocolo y no por el sistema**. 🔴 **Entonces la fila restringida favorece
estructuralmente al brazo que trajo ítems nuevos, y eso explica el cambio de signo mejor que
cualquier hipótesis sobre el chunking.** Es el hallazgo de método del hito, y vale más que la
tabla: **la sensibilidad del desempate deja de ser neutral en cuanto una fila tiene juicios de un solo
anotador.**

⚠️ **Dos cosas que esta tabla NO dice, y las dos son tentadoras:**

1. 🔴 **Que *«el ciego gana»* en la población (b).** Entre *ampliado* y *ciego* el orden **se da
   vuelta entre las dos políticas del mismo ground truth** —0,3898 vs 0,3895 y 0,4468 vs
   **0,4472**—: **3 y 4 diezmilésimas. Eso es ruido, y se dice que es ruido**, no un segundo
   resultado. ⚠️ **Y el piso de ruido de los dos brazos está SIN MEDIR:** el `0,000` exacto del
   §4.4 es de `ce0910b50e3569e9` y **ninguno de los dos brazos tiene dos ordinales**. Lo que hay es
   el argumento de inclusión de cadenas —la cadena es la de `C3` sobre otra variante— y la variante
   **sí** cambia lo que el índice devuelve, que es el eje que se está midiendo. `medido` ≠
   `acotado por inclusión`, igual que en el §4.4.
2. 🔴 **Que el eje de campos de `knowledge` movió el `nDCG`.** El `+0,0107` de *ampliado* − `C3`
   sobre (a) sale de **3 ítems nuevos sobre 150**: las dos filas **comparten 147 pares**, así que
   el delta está dominado por lo que no cambió y **no es interpretable con `n = 3`**.

🔑 **Pero el no-movimiento del eje de campos SÍ es un resultado, y está medido donde el `nDCG` no
llega: en el churn del top-5 contra `C3`** (clave `churn_del_top5_contra_c3`; 30 consultas, 5 obras
por consulta, 10 consultas por categoría).

| Churn del top-5 contra `C3` | medio, de 5 | top-5 idéntico | `visual_cromatico` | `conceptual_cultural` | `ambiguo` |
|---|---:|---:|---:|---:|---:|
| **`C3`-ampliado** · eje de campos | **0,267** | **24 de 30** | 0,00 | 0,60 | 0,20 |
| **`C3`-ciego** · eje de chunking | **1,567** | **4 de 30** | 1,50 | 1,80 | 1,40 |

**El eje de chunking mueve 5,9× más el top-5 que el eje de campos.** Y el 0,267 es **una
afirmación y no un empate**, porque el piso de ruido de `C3` está **medido en 0,000 exacto en su
`config_id` publicado** (§4.4): el eje de campos **sí** mueve algo —**8 ítems en 30 consultas**—
**pero casi nada**. ✅ **Es consistente con el §5.2:** `provenance` es el 39 % del índice congelado
y nunca se cita, así que agregar `exhibitions` y `citations` **agranda el índice sin cambiar lo que
se recupera**. Y del lado del chunking **confirma desde el retrieval lo que el §5.1 contó desde el
pool**: 29 ítems sin juzgar contra 3. 🔑 **Las dos mitades del estudio secundario son asimétricas y
así se publican:** el eje de chunking tiene tamaño de efecto y el de campos no.

#### 🔴 Y una decisión de método que este informe no decía en ninguna parte: el `IDCG` se mueve con el pool

**El `IDCG` de `nDCG@5` se computa sobre la población del ground truth, así que extender el pool
MUEVE el baseline.** Por eso las tres filas se miden contra **la misma** población y no cada una
contra su propio pool: si los brazos se midieran contra sus **197** pares y `C3` contra sus
**379**, el ideal por consulta del primero saldría de **~6,6** pares juzgados y el del segundo de
**~12,6**, y **los brazos publicarían un `nDCG` más alto por el recorte del pool y no por el
sistema**. Es el modo de falla de población del proyecto metido adentro del denominador de la
métrica titular.

**Y el control lo prueba, con la cifra incómoda** (clave `c3_publicada_vs_recomputada`):

| `nDCG@5` de `C3`, población (a) | Valor | Base |
|---|---:|---|
| **Publicado en el §4.1** | **0,535** (0,5348) | los **379** pares del consenso definitivo · `tabla-a2-m13.json` |
| **Recomputado acá** | **0,5308** | los **410** = 379 + los 31 nuevos · `retrieval-secundario-m14b.json` |

**El top-5 de `C3` no cambia entre las dos:** lo que cambia es que **algunos de los 31 juicios
nuevos entraron a ideales de `C3`**. Las tres consecuencias, dichas por separado porque no son la
misma:

- 🔴 **`nDCG` no es estable ante extensiones del pool.** El pooling incremental **compra cobertura
  y mueve el denominador ideal.** Es la regla de que toda cifra lleva su base, adentro de la métrica
  titular, y es la primera vez que este informe lo dice.
- ✅ **No invalida la tabla `C0`–`C4` del §4.1:** sus **cinco** filas se midieron contra **el
  mismo** ground truth de 379, así que la comparación entre ellas se sostiene entera.
- 🔴 **Sí prohíbe comparar un `nDCG` de esa tabla contra uno calculado sobre un ground truth
  extendido.** Por eso esta sección usa **0,5308** y **no** el 0,535 publicado. ⚠️ **Y el 0,535 del
  §4.1 sigue siendo la cifra correcta de su tabla: no se toca.** Las dos conviven **porque tienen
  bases distintas**, y eso es lo que hay que explicar, no corregir.

#### Las dos limitaciones, que acá tampoco van al pie

1. 🔴 **Los 31 juicios nuevos los anotó UN solo anotador y no tienen κ propio.** Textual: *las
   métricas de retrieval de los dos brazos secundarios descansan sobre **166 juicios del ground
   truth principal** —con su κ medido y su limitación declarada— más **31 juicios de un solo
   anotador (`ia-incremental`), sin κ propio***. Su reparto por grado es **11 · 16 · 4** en los
   grados 0 · 1 · 2. ⚠️ **Por eso esta sección NO es comparable con la tabla `C0`–`C4` en pie de
   igualdad, y va en su propia sección:** el ground truth principal lo produjeron dos anotadores de
   proveedores distintos más un desempate (la anotación por IA) y **un κ no existe sobre un anotador**.
   Pegarle el κ del acuerdo entre anotadores a estas filas sería **prestarle a un número la credencial de otro**.
   📌 **Y son 31, no 32 — la diferencia es una población, no un error de conteo.** *Por brazo*,
   sumando los dos top-5 de 150, son **268 reusados + 32 nuevos = 300**. *Por par distinto*, que es
   lo que cuesta anotación, son **166 + 31 = 197**: los dos brazos **comparten 103 pares**, y uno
   —`c03-conceptual · cma:146167`— **lo contaba dos veces la suma por brazo**. **Un juicio es sobre
   el par, no sobre el brazo que lo trajo** (la evaluación). ⚠️ **Los `%` sin juzgar por brazo no
   cambian** (2,00 % y 19,33 %): se cuentan sobre **sus propios 150**.
2. **El `% de ítems del top-5 sin juzgar = 0` de los dos brazos es una tautología, y se dice que lo
   es.** Es la tautología del pool apareciendo por **segunda vez** —la primera son las cinco filas del §4.1—: este
   pool **se derivó para cubrir a estos dos brazos**, así que todo par de su top-5 está adentro
   **por definición**. **Ese 0 no dice que la comparación se sostenga: dice que la pregunta no
   aplica.** ⚠️ Y el `0` de la fila `C3` contra el pool **definitivo** es la misma tautología por el
   otro lado, porque `C3` formó ese pool: **las tres filas publican 0 y ninguno de los tres ceros
   es evidencia.** 🔑 **La cifra que sí mide algo es la otra:** contra el pool **definitivo**, el
   ampliado tenía **3 de 150 = 2,00 %** sin juzgar y el ciego **29 de 150 = 19,33 %** — y ése era
   el hallazgo (§5.1), no esto.

### 5.4 🔴 El tope del proxy: USD 10 por grupo, agotado — y es material de producción, no una excusa

El acceso no es a `api.openai.com`: es un **proxy LiteLLM de ITBA** con una **virtual key por
grupo**. Al 1/11 devuelve `400 · budget_exceeded · Current cost: 10.0654 · Max budget: 10.0`
(la decisión de presupuesto, verificado con una llamada mínima). **El grupo decidió no pedir que lo suban**, por
coherencia con el principio de gastar lo menos posible: lo único que el tope impide —completar el
juez, 200 pares de 379— **mejora la `n` y no habilita ningún criterio**.

**El gasto está medido y desglosado por rol** — **~7.313 requests**, contadas del libro mayor de
cada artefacto y publicadas como **cota inferior**, porque la verdad de facturación la dice el
dashboard y **las dos no se promedian**:

| `reranker` | `embedding_texto` | `generador` | `juez` | `planificador` |
|---:|---:|---:|---:|---:|
| 3.920 | 1.896 | 1.043 | ~391 | 63 |

🔑 **Y la causa del sobregiro es un defecto del modelo de costo, no del gasto, que es justamente lo
que lo vuelve un resultado de puesta en producción.** El proyecto costeaba a **USD 0,0010 por
request** —la cifra que mostraba el dashboard— y **el proxy factura por token**. El rol más caro es
el juez, que manda **la rúbrica entera (26.345 caracteres) más dos imágenes en cada llamada**:
~10.000 tokens de entrada, que a precio de lista de `gpt-4o` son **~USD 4,4 solo de entrada**. **La
sesión no se pasó de llamadas (391 ≤ ~400): se pasó de dólares.** *El precio por request no es el
precio.*

**La lección operativa, que es la que vale fuera de este TP:** en un ambiente compartido con tope
duro, el presupuesto hay que modelarlo **por token y por rol**, y el rol que repite el prompt largo
N veces por consulta es el que lo consume. ⚠️ **Y tiene una consecuencia declarada en el
entregable:** el repositorio publicado no lleva los artefactos, y lo que compensaba eso era que el
`README` dijera cómo regenerarlos. **Con el tope agotado esos comandos no corren**, así que la
promesa va **con su condición al lado**.

> ### 🔑 El modelo por token se cerró el 2026-11-03, y el proxy factura un 19 % por debajo de lista
> *La acción 5 de la lista de pendientes —«modelar el costo por token y por rol»— ejecutada, con
> **cero llamadas al proxy**: los tokens ya estaban en los artefactos.*
>
> Derivar el gasto de **todos** los artefactos en disco a precio de lista de OpenAI (leído el
> 2026-11-03, §4.1) da **USD 12,43** sobre las mismas **7.313 requests** que esta sección cuenta.
> El dashboard facturó **USD 10,0654**. ⚠️ **La brecha va con su base, porque las dos lecturas
> del mismo hueco dan números distintos:** el proxy facturó un **19 % menos que el precio de
> lista** (10,0654 / 12,4292 = 0,810), que es lo mismo que decir que **lista sobreestima la
> factura en un 23 %** (12,4292 / 10,0654 = 1,235). **Se publica la primera** —base: lista—
> porque la hipótesis de abajo es un **descuento** sobre el precio de lista.
>
> | rol | modelo | llamadas | tokens in | tokens out | USD derivado |
> |---|---|---:|---:|---:|---:|
> | **`juez`** | `gpt-4o` | 391 | 1.867.710 | 10.524 | **4,77** |
> | **`reranker`** | `gpt-4.1-mini` | 3.920 | 11.009.329 | 194.322 | **4,71** |
> | `generador` | `gpt-4.1-mini` | 1.043 | 3.689.802 | 832.772 | 2,81 |
> | `planificador` | `gpt-4.1-mini` | 63 | 129.292 | 3.837 | 0,06 |
> | `embedding_texto` | `text-embedding-3-small` | 1.896 | 3.692.620 | 0 | 0,07 |
> | **total** | | **7.313** | | | **12,43** |
>
> ✅ **Y confirma la atribución que esta sección hizo a mano.** Arriba dice que el juez manda
> *«~10.000 tokens de entrada por llamada, que a precio de lista son ~USD 4,4 solo de entrada»*.
> La derivación da **4,67 de entrada** sobre los 391 intentos: **la estimación se sostiene**, y
> ahora la emite el pipeline en vez de una cuenta al margen.
>
> 🔑 **Ese 19 % tiene una hipótesis con nombre, y se publica como hipótesis.** El precio de entrada
> **cacheada** de `gpt-4.1-mini` es **0,10** —el 25 % de 0,40— y el reranker manda **el mismo
> prompt y la misma imagen del usuario 20 veces por consulta**, que es el caso de libro del caché
> de prompts. ⚠️ **No la podemos verificar con nuestros artefactos**: el libro mayor guarda
> `prompt_tokens` y **no** `cached_tokens` (`puertos/llm.py`). Queda escrita **con el dato que la
> cerraría al lado**, que es lo que corresponde a una causa no medida. La otra explicación posible
> —que el proxy aplique una tarifa propia— es indistinguible con lo que tenemos.
>
> ⚠️ **Las dos cifras del gasto total conviven y NO se promedian**. La de **facturación**
> es la del dashboard: **10,0654**, y es la que manda para decir cuánto se gastó. La **derivada**
> es la única que se puede **atribuir por rol y por configuración**, que es lo que la tabla de tokens, costo y la figura publica y
> el dashboard no sabe hacer. Cada una con su base.

### 5.5 🔑 La regla de honestidad dejó de ser una promesa: `la revisión humana(c)` está medido

el criterio de honestidad compromete **dos** cosas y hasta el 1/11 solo una tenía número. el conteo de referencias colgadas —*0
referencias a un fragmento inexistente o no recuperado*— es automática y da **0 colgadas** (§4.1).
La mitad fina —**que todo dato de museo vaya con su cita y toda observación visual vaya marcada
como propia del modelo**— exige revisar las explicaciones **afirmación por afirmación**, y eso
ahora existe: **`artifacts/ground-truth/revision-explicaciones/m13b-c1-5x5.json`** (clase 3, las 25
explicaciones **copiadas por valor** con su veredicto; **0 llamadas al proxy** — se hizo leyendo
artefactos ya persistidos).

**La muestra, y el criterio se declaró antes de leer una sola afirmación.** 25 explicaciones =
**5 consultas × 5 obras** —el tamaño lo fija el criterio de honestidad y **no se movió**, porque moverlo
habría sido elegir la vara después de ver el resultado—, sobre **`C1` = v1**: **2
`conceptual_cultural` + 2 `visual_cromatico` + 1 `ambiguo`**, **5 imágenes de consulta distintas**,
**25 obras todas distintas**, los dos museos (`cma` 18 · `aic` 7) y abstenciones en los dos extremos
(dos consultas con 5/5 obras abstenidas, dos con 0/5, una con 2/5).
🔑 **Por qué `C1` no sesga lo que el criterio de honestidad mide:** el `binding` del generador es **byte-idéntico en
`C1`–`C4`** —mismo modelo, mismo `prompt_id`, `version_del_prompt = 2`, `seed = 7369`— y la
evidencia también (3 fragmentos por obra, mismo orden). **El instrumento que el criterio de honestidad mide no cambia
con el brazo**; elegir `C1` solo cambia *qué obras se explican*. Y las **148 observaciones** del
antecedente son exactamente las de `C1`, así que la muestra lo re-verifica **sobre su propia
población** y no sobre una parecida.
⚠️ **Con una contaminación que declaró quien anotó, y nadie habría detectado desde afuera:** al
inferir el esquema de los artefactos leyó antes las afirmaciones de `c01`, **y por eso `c01` quedó
fuera de la muestra**.

**Los tres conteos van por separado, porque son tres fallas distintas** (clave `conteos`; la unidad
es **la afirmación**, no la corrida, por el descarte contado de afirmaciones):

| Eje | `n` | Resultado |
|---|---:|---|
| **1 · ¿está bien clasificada?** | **51** afirmaciones (27 `dato` + 24 `observacion`) | **39 bien · 12 mal** |
| **2 · ¿la cita se sostiene contra el fragmento?** | **27** afirmaciones `dato`, sobre **25** fragmentos citados distintos | **20 sostiene · 4 parcial · 3 NO sostiene** |
| **3 · ¿la observación es defendible mirando la imagen?** | **16** de las 24 observaciones hacen una aserción visual verificable | **12 defendibles · 2 discutibles · 2 FALSAS a la vista** |

🔴 **Las 12 mal clasificadas se desglosan, y el desglose no es opcional:**

| Falla | `n` | Qué significa |
|---|---:|---|
| **Observación disfrazada de `dato` citado** | **1** | Es **la falla que el criterio de honestidad existe para medir** |
| **Dato de museo presentado como observación propia** | **6** → 🔴 **2 forzadas por el contrato y 4 no** | Pierde la verificabilidad, que es la mitad del producto |
| **Conocimiento general del modelo presentado como observación propia** | **5** | **No es ninguna de las dos cosas que la regla de que un dato lleva referencia tiene** |

**Por qué la cifra de 6 nunca va sola.** La `ficha` (`artist`, `culture`, `year`, `medium`) **se
renderiza pero no es un fragmento de `knowledge`**, así que no tiene `fragment_id`; y por la regla de que un dato lleva referencia
(`tipo = dato ⟺ refs ≠ ∅`) una afirmación que descansa en ella **no puede ser `dato`** aunque el
hecho lo publique el museo. **2 de las 6** están forzadas por esa frontera (la nacionalidad del
artista en `cma:163456` y `aic:120163`): en esa fracción **la revisión humana no mide al modelo, mide el
esquema de evidencia**. 🔴 **Se publica *"6, de las cuales 2 forzadas por el contrato y 4 no"*, y
nunca la cifra sola** — lo contrario le cargaría al generador un defecto del contrato de datos.
**Las otras 4 no tienen excusa** (`cma:111691`, `cma:169341`, `aic:110663`, `aic:495`): en tres de
ellas el texto es **traducción del fragmento que la afirmación `dato` vecina ya cita con `[1]`**.
🔑 **Es la falla en la dirección que se mira menos: no inventa, desacredita.** Un dato verificable
vuelve al usuario como opinión del modelo.

#### 🔑 El hallazgo conceptual: la regla de que un dato lleva referencia tiene dos tipos y el generador produce TRES clases

**5 de las 24 observaciones** —las cinco de `c09-conceptual`— no son observaciones de la imagen ni
datos de museo: son **conocimiento general del modelo** (*"el dragón es un motivo clásico en la
pintura china"*). la regla de que un dato lleva referencia define `observacion` como afirmación del modelo **sobre la imagen** —paleta,
pincelada, composición—, así que tiparlas así le dice al usuario **"el modelo miró la obra y vio
esto"**, que es falso. Y como el bicondicional **`dato ⟺ refs ≠ ∅`** no deja otra salida —sin cita
no hay `dato`—, **todo lo que no tiene fragmento cae en `observacion`**.

🔴 **La consecuencia es sobre nuestra propia métrica:** la *separación observación / dato* de
`la evaluación §D` —*"% de observaciones visuales correctamente presentadas como propias"*— **corta
binario sobre una población de tres clases**. El porcentaje sale alto **sin distinguir la
observación visual de la enciclopedia**, y el error va **en la dirección peor**: el conocimiento
general se presenta con la autoridad de algo que el modelo vio. La regla de honestidad del proyecto
separa *lo que el museo dice* de *lo que el modelo ve*, **y hay una tercera cosa —lo que el modelo
sabe— que no tiene dónde ir.**
⚠️ **No se agrega un tercer tipo ahora**: movería el prompt, el `config_id` y las 1.151 corridas en
disco. **Entra como resultado declarado, no como cambio** — el mismo criterio que `provenance` en
§5.2.
📌 **Y la forma se cumple 51/51**, que es parte del resultado: **0** `dato` con `refs` vacío, **0**
`observacion` con `refs`, **0** afirmaciones descartadas por el descarte contado de afirmaciones y ninguna mal formada (la afirmación mal formada registrada y contada,
reportada aparte). **Que la forma se cumpla 51/51 y la clasificación falle 12/51 es
exactamente el hallazgo:** un invariante de forma no puede ver si el tipo es el correcto.

#### 🔴 El caso que vale por la tabla: falla los tres ejes a la vez — y degradó la respuesta

`c03-visual` (*"el mismo interior pero en tonos fríos"*) / **`aic:43211`** — *Interior of the Oude
Kerk, Delft*, Emanuel de Witte, c. 1680 (AIC). Clave `casos_para_el_informe`, prioridad 1.

| Eje | Veredicto |
|---|---|
| **1 · clasificación** | **Falla.** `[dato] [1]` afirma *"...decoración mínima y **tonos cálidos** propios del óleo sobre panel del siglo XVII"*. **"Tonos cálidos" es una observación de paleta** —la definición misma de `observacion`— **tipada `dato` y firmada con `[1]`** |
| **2 · la cita se sostiene** | **No sostiene.** El fragmento respalda *decoración mínima* (*"are the only decoration"*) y *luz y sombra* (*"light, shade, and volume"*), y **no menciona tono ni color en ningún lado** |
| **3 · observación defendible** | **Falsa a la vista.** Dos renglones abajo, la `[observacion]` dice *"paleta cálida, con predominancia de tonos marrones y amarillentos"* sobre **un interior de piedra gris-blanca con sombras azul-grisáceas**; los cálidos (candelabro dorado, púlpito de madera) son **acentos minoritarios** |

🔑 **Por qué es el mejor caso del estudio:** la **misma** afirmación de paleta aparece **dos veces en
la misma explicación**, una mal tipada como `dato` citado y otra **bien** tipada como `observacion`.
**El generador sabe cuál es el tipo correcto y no lo aplica de forma estable** — no es que no
distinga los tipos, es que no los respeta.
🔴 **Y la falla degradó la respuesta, no solo su justificación.** El pedido pedía **tonos fríos**,
`aic:43211` era **la obra más fría de su propio top-5**, y el sistema **se abstuvo sobre ella**
—*"no hay información en los fragmentos que indique que esta obra utiliza tonos fríos"*— **apoyado
en la observación falsa**. La honestidad de la explicación no es cosmética: acá **le costó al
sistema la mejor obra que había recuperado.**

#### Las 4 fallas del eje 3 son 4 de 4 sobre paleta, y las 4 en `visual_cromatico`

Las 2 falsas **y** las 2 discutibles (clave `hallazgos_de_diseno`, `RE-6`). Y el mecanismo se delata
en el *hedge*: *"la paleta **probablemente** incluye tonos neutros y terrosos **propios de la
arquitectura antigua**"* (`cma:148861`) — **la observación se infiere del asunto del fragmento, no
del píxel**. Acierta cuando el asunto y la imagen coinciden y falla cuando no. La segunda falsa es
la más verificable de las dos: `c10-visual` / `cma:97169` afirma **"un fondo azul brillante"** sobre
una obra de fondo pardo-negro con follaje en sombra, y **vuelve a invertir la respuesta al pedido**
—el usuario pidió *una paleta más oscura*, la obra **es** oscura y la explicación le dice que no.
🔑 **Esto cierra desde un tercer ángulo el `κ = 0,249` cromático del §6.2:** el acuerdo entre
anotadores es el más bajo en `visual_cromatico` (0,249), el juez lo mide 0,450 ahí, y **las cuatro
fallas de observación visual están todas en esa categoría**. *La categoría que depende de mirar es
exactamente donde el sistema no mira.*

#### El eje 2 falla por la razón que el delta por categoría de pedido ya anticipaba

**Las 3 citas que no sostienen tienen la misma forma: la `description` del museo no describe lo
representado** (`RE-4`). `cma:152831` (*Dragon amid Clouds*) cita un fragmento que es **la biografía
de Min Zhen** y no menciona dragón ni nubes —**la afirmación es verdadera** y la imagen la confirma,
pero un usuario que siga `[1]` para verificarla **llega a un texto sobre la orfandad del artista**—;
`aic:43211` es el caso de arriba; y `cma:148862` (*The Grotto of Posillipo*) cita un fragmento
**byte a byte idéntico** (1.015 caracteres) al de `cma:148861`, que describe **otra obra** —la
Columnata de San Pedro—: CMA publica **un único texto curatorial para el par de óvalos**.
🔴 **Separa limpiamente el conteo de referencias colgadas de la cita que respalda la afirmación:** en los tres casos **el fragmento existe y fue
recuperado** —las 27 citas de la muestra dan **0 colgadas**— **y no dice lo que la afirmación
dice que dice**. Un umbral único habría perdido además los **4 parciales**, que son otra falla: la
cláusula que **contesta el pedido** se suelda adentro de un `dato` citado (`aic:244013`:
*"...**sin relación con la pintura china**"*, que no está en el fragmento). **la regla de que un dato lleva referencia no tiene lugar
para una afirmación sobre la *ausencia* de evidencia**, así que la misma frase viaja como `dato`
citado en una obra y como `observacion` sin cita en las otras cuatro de la misma consulta.
⚠️ **Y el caso de `cma:148862` es falla del dato del museo, no del modelo.** la cita que respalda la afirmación la cuenta
igual, y debe: desde el lado del usuario, seguir `[1]` lleva a la ficha de otra pintura. 🔑 **Es
detectable automáticamente** —dos `description` idénticas con distinto `artwork_id` es una consulta
sobre `knowledge`, no una revisión a mano— y es **el cuarto defecto de datos de museo que una sesión
de anotación encuentra mirando las obras de a una** (los otros tres están en el anexo §B).
⚠️ **Y las dos cifras no se confundan: el «cuarto» cuenta defectos ENCONTRADOS A MANO, y el 14,2 %
cuenta la POBLACIÓN afectada.** Las 370 obras no son cuatro casos: son lo que el barrido automático
encuentra cuando se lo corre, y es la razón por la que conviene correrlo.

#### 🔴 Y la causa de esas dos citas es estructural: el 14,2 % del corpus tiene una `description` que no es de esa obra

**Los museos publican UNA `description` por álbum y la cuelgan de cada hoja.** Medido el **1/11**
sobre el índice congelado `c3274567371ff42b` —agrupar los **2.625** fragmentos con
`section = "description"` **por su texto** y quedarse con los grupos que tienen más de un
`artwork_id`; **0 llamadas al proxy**—: **120 textos** compartidos por **370 de las 2.605 obras
indexadas = 14,2 %** (`cma` 333 · `aic` 37), y el grupo más grande son **25 hojas**
(`cma:163105`…`163155`: *"This extraordinary album has 50 paintings on a variety of religious
subjects…"*). ⚠️ **No es un bug de nuestra ingesta, y eso lo hace peor, no mejor:** los textos son
coherentes **como descripción del álbum**, y los **120** grupos agrupan obras de **un solo museo**
cada uno, en `artwork_id` vecinos —las hojas de un mismo objeto—. **El dato es correcto en el origen
y está mal atribuido en el destino**, que es la clase de defecto que **ninguna garantía de
procedencia cubre** (`el relevamiento § Riesgos de datos conocidos`).
**Y la exposición se mide en la salida, con su base porque la etiqueta no es la fila (§3.3):** las
referencias que apuntan a uno de esos textos son **30 de 409 = 7,3 %** sobre **13 obras** en el
brazo de barrido etiquetado `C4` (`7f257a8b9a209198`, 61 corridas) y **12 de 191 = 6,3 %** sobre
**9 obras** en la fila `C4` publicada (`e30ca8c5003c614c`, 30 corridas).

🔑 **Los dos casos del eje 2 que no son del modelo son exactamente los dos con `description`
compartida.** El fragmento que `cma:152831` cita **no es una biografía que por casualidad ocupe el
lugar de la descripción: es la descripción del álbum**, compartida byte a byte por **8 hojas**
(`cma:152821`…`cma:152833`), y lo dice su última oración —*"This **album** demonstrates Min's
versatility and mature style…"*—; la de `cma:148862` es el par de óvalos, **grupo de 2**. **Las
otras cuatro obras que este §5.5 publica tienen descripción propia** (`aic:43211`, `cma:97169`,
`cma:169341`, `aic:244013`).
⚠️ **2 de 2 contra 0 de 4 no es una prueba estadística y no se presenta así.** Lo que establece el
mecanismo es **la construcción** —el texto citado *es* el del álbum y se verifica leyéndolo contra
su `artwork_id`, sin estadística de por medio—; el **2 de 2** sobre un `n` de 6 **corrobora**.

🔴 **Entonces el `20 de 27` mide dos cosas sumadas, y hay que decirlo donde está la cifra.** De las
**7** afirmaciones que no llegan a *sostiene*, **2 son defecto del dato** —`cma:152831` y
`cma:148862`— y **5 son del modelo o del contrato** (`aic:43211` y los 4 parciales). *El modelo cita
mal* y *el modelo cita bien un fragmento que está mal atribuido* **son dos defectos distintos con
dos arreglos distintos**: el primero vive en el prompt y en el contrato de evidencia, el segundo en
la ingesta, **y ninguno de los dos arregla al otro**. ⚠️ **La cifra no se recalcula:**
`revision-explicaciones/m13b-c1-5x5.json` es **clase 3**, y un artefacto sellado le pone un límite
al refactor. **Se declara qué mide, no se cambia el número.**

📌 **Y es la tercera vez en este §5.5 que una cifra correcta mide más de una cosa, así que conviene
leerlas juntas y no como tres notas sueltas:** la *separación observación / dato* de `la evaluación §D`
corta **binario sobre tres clases**; el **6** de *dato presentado como observación* lleva **2
forzadas por el contrato y 4 no**; y el **`20 de 27`** suma **dos defectos con dos dueños
distintos**. 🔑 **Ninguna de las tres es un error de medición: son tres cifras bien medidas cuya
unidad no es homogénea** — y en las tres el arreglo es el mismo, **declarar la composición en vez de
promediarla**.

**El camino de mitigación va con su costo, porque la decisión solo es honesta con el costo al lado.**
El defecto es **detectable sin LLM y antes de publicar** —es la consulta de arriba sobre
`knowledge`, no una revisión a mano—, así que caben dos arreglos: **degradar la cita** (presentarla
como *descripción del álbum* y no *de la obra*, que es lo que el texto efectivamente dice) **o
excluir esos fragmentos del índice**. 🔴 **Ninguno se hace ahora, y por el mismo motivo que en §5.2
—con dos mecanismos distintos:** excluirlos mueve el `variante_id`, del que cuelgan el pool
congelado y los **410 juicios**; degradar la cita mueve el renderizador y con él el `config_id`, del
que cuelgan las **1.151 corridas**. **`c3274567371ff42b` está congelado y cualquiera de los dos
arreglos invalidaría lo que este informe mide.** ⚠️ **Y excluirlas no es gratis: cuesta el
14,2 % de las obras citables.** *Lo detectamos y elegimos declararlo* es una posición defendible;
*no lo vimos* no lo es.

#### 🔑 Y la frontera entre *honesto* y *correcto*, que es lo más informativo de la revisión

`c06-ambiguo` / `cma:169341` (*Portrait of Maharaja Pratap Singh*). La `[observacion]` sin cita dice
*"el perfil y **el rizo distintivo en la nuca**"*. **Como observación es defendible y verdadera** —se
verificó contra la imagen—. **Como tipo está mal**: es traducción del fragmento (*"because of his
profile and the distinctive curl of hair at the nape of his neck"*) que la afirmación `dato` vecina
**ya cita con `[1]`**. **Las dos lecturas son correctas y no se cancelan: mirando solo el eje 3 el
caso pasa; mirando solo el eje 1 falla.** Es exactamente por qué los tres conteos van separados —y
por qué **una groundedness en una sola cifra habría hecho desaparecer este caso en cualquiera de las
dos direcciones**.
📌 **Y el espejo del mismo par:** *"el artista es estadounidense"* sobre `aic:244013` es **honesto**
—el museo ancla a Thomas Birch en Philadelphia— **y discutible como hecho**: Birch nació en
Inglaterra. *Honesto no implica correcto*, y los dos ejes lo muestran por separado.

#### 🔴 El mismo material refuta un antecedente propio: la medición anterior no era `0`

la medición anterior publicaba *"**0 de 148** observaciones persistidas mencionan la imagen de
consulta"*, y sobre esa base la cadena de la licencia calificaba la segunda cláusula de la regla de que un dato lleva referencia —*"una
`observacion` es sobre una obra recuperada"*— de **guardia preventiva**. **En la misma población, el
mismo `config_id` y los mismos artefactos: son tres consultas.**

| Consulta · obra | Texto persistido | Tipo |
|---|---|---|
| `c05-visual` · `aic:61151` | *"La paleta de la obra parece más clara y luminosa **en comparación con la imagen original**"* — y la `abstencion` de la misma obra lo repite | `observacion` |
| `c08-visual` · `cma:314439` | *"Los colores en **la imagen original** son vivos y saturados..."* ⚠️ **admite una lectura benigna** (*"la pintura original"* frente a una reproducción) y va declarada: es el caso probable, no el seguro | `observacion` |
| `c09-visual` · `cma:147045` | **Describe el contenido de la imagen del usuario**: *"...diferente al motivo de **bambú y flores de la imagen proporcionada**"* — 🔴 y va tipado **`dato` con su `ref`**, o sea **presentado al usuario como respaldado por el museo** | `dato` |

**Sobre la definición que la medición anterior usó —observaciones— el `0` es `2`. Y sobre los tres campos de texto
libre que el generador persiste es `5`** en `C1` (2 `observacion` + 2 `dato` + 1 `abstencion`,
clave `alcance_que_M4_3_no_cubria`): **el predicado de la regla de que un dato lleva referencia, escrito sobre `observacion`, no
alcanzaría a los `dato` ni a la `abstencion`.**
🔴 **Entonces la cláusula 2 de la regla de que un dato lleva referencia es un ARREGLO, no una guardia:** cierra un camino que el
generador **ya está tomando**. El predicado sigue siendo la decisión correcta; lo que cambia es que
**no es preventivo**, y que **tiene que ser un validador que corra** y no un invariante que se
cumple por disciplina.

> ### 🔴 Y la magnitud es de `C1`, que es la configuración MENOS afectada: `C4` da el triple
>
> la medición anterior midió `C1` —y esta sección lo corrigió sobre esa misma población—, **pero `C1` no es la
> configuración que este informe publica.** Medido sobre los tres `config_id`, contando las
> observaciones persistidas que refieren a la imagen de consulta:
>
> | configuración | `config_id` | observaciones | 🔴 **refieren a la imagen de consulta** |
> |---|---|---:|---:|
> | **C1** | `4443862ae45e62d6` | 148 | **2** (más 2 `dato` con su `ref`) |
> | **C3** | `7eb0c12e63b95947` | 148 | **0** |
> | 🔴 **C4** — la que se publica | `4d35057811cf8e2f` | 144 | 🔴 **6** |
>
> ⚠️ **Y una de las seis tiene la imagen del usuario como SUJETO gramatical**, que no es una
> comparación sino una descripción: *"**La imagen que subió la persona** muestra un motivo similar de
> bambú y ramas, pero en blanco y negro"* (`c09-visual`). 🔑 **Hay una explicación mecánica que vale
> como hipótesis y no como resultado medido:** `C4` agrega reranking multimodal, así que la imagen de
> consulta está **más presente** en el contexto del generador. ⚠️ **Si es eso, la exposición crece
> con la calidad del sistema**, que es la dirección más incómoda posible.
>
> 🔴 **Y de acá sale la precisión que vuelve el predicado implementable, porque como está escrito no
> lo es.** *"Una `observacion` es sobre una obra recuperada"* suena bien y **rechazaría el
> producto**: `c05`, `c08` y `c09` **son consultas por imagen**, y *"el mismo motivo"* es
> literalmente lo que preguntan. **La forma que un validador puede exigir es que la imagen de
> consulta no sea el SUJETO de la afirmación —no que no se mencione— y alcanza a los dos tipos, no
> sólo a `observacion`.** Eso es el trabajo futuro, y está sin implementar: el código quedó congelado
> con el commit que se entrega.
🔑 **Y la causa del `0` vale más que la corrección, porque es nuestro modo de falla dominante en
estado puro: el `0` salió de buscar un patrón** (*"menciona la imagen"*) **y el texto usa otras tres
formulaciones** —*en comparación con*, *similar a*, *el mismo que*— (el hallazgo de las formulaciones comparativas). **Un
conteo por patrón no es un conteo por propiedad**, y el reporte original decía *"verificado"*.
✅ **Sobre las 25 de la muestra el antecedente se confirma:** ninguna de las 24 observaciones
menciona la imagen de consulta. **Lo refutado es el `0` sobre las 148**, que es la población en la
que se había afirmado.

#### Y las dos verificaciones de la muestra que cierran el criterio de atribución y el criterio de idioma donde estaban escritas

`los criterios de entrega el criterio de atribución` pide la atribución CC-BY verificada **sobre la muestra del criterio de honestidad** y el criterio de idioma compromete
textualmente **25/25**. Las dos existían solo como verificación automática sobre poblaciones más
amplias (32/32 y 314/314, §F del anexo); ahora están **sobre la población que el criterio nombra**:
**el criterio de atribución 7/7** —de los 25 fragmentos citados, **7 tienen `license = CC-BY-4.0`** (son `description`
de AIC) y los 7 llevan título, artista, museo, *«Texto bajo CC BY 4.0»* y URL; **0 sin
atribución**, y los 18 CC0 no la exigen— y **el criterio de idioma 25/25** en español con la cita **textual en inglés
sin traducir**, byte a byte contra el fragmento. **Ninguna cita traducida**, que habría sido un
hallazgo.
⚠️ **Y la salvedad que arrastra todo este §5.5, dicha donde se ve:** el criterio de honestidad dice *"muestra
**humana** fija de 25 explicaciones"* **y la anotó IA** (la anotación por IA, igual que los 379 juicios de
relevancia). Lo que hay es **un modelo juzgando si otro modelo citó bien** — con la diferencia, que
vale, de que acá el juicio es **contra un fragmento en disco y una imagen en disco**, no contra una
convención de la rúbrica. El veredicto completo del criterio de honestidad está en el §F del anexo.

---

## 6. 🔴 Cómo lo medimos, y qué no podemos afirmar

**Esta sección está en el cuerpo a propósito.** el acuerdo entre anotadores compromete *reportar el acuerdo
pase lo que pase*, y la anotación por IA eligió entre las dos salidas que el acuerdo entre anotadores ofrece —refinar la rúbrica
y re-anotar, o **reportarlo como limitación explícita**— la segunda. El texto completo de esa
decisión está en `artifacts/ground-truth/rubricas/v1-decision-de-a5.toml`. Lo que sigue **no lo
contradice ni lo afloja**: lo resume y le agrega lo que se midió después.

### 6.1 Quién anotó, y qué deja de ser afirmable

🔴 **Ningún humano anotó.** Los 379 juicios × 2 los produjeron **dos anotadores de IA de familias
de modelo distintas**, ninguno el modelo del pipeline, y el desempate un tercero
(`el registro de decisiones `, la anotación por IA, el consenso). Las consecuencias se declaran, no se dejan caer:

- **La mitigación 1 de la circularidad de la evaluación queda formalmente ROTA.** Decía *"la tabla
  C0–C4 se calcula únicamente con anotación humana"*. Lo que la reemplaza es **separar
  proveedores**, y 🔴 **la configuración alcanzable es asimétrica**: un anotador es independiente
  del proveedor del pipeline y **el otro comparte proveedor con él**. Lo que compra es que el κ
  cruza **al menos una frontera de proveedor**; lo que **no** compra es un ground truth
  independiente del proveedor evaluado.
- **No se puede afirmar acuerdo inter-humano.** No existe y no va a existir. El κ es
  **inter-modelo**, y se publica con esa etiqueta.
- **No se puede presentar el κ como validación del juez LLM.** Validaba al juez *contra humanos*;
  ahora compara dos clases de IA.
- **No se puede afirmar que el ground truth sea independiente del sistema evaluado.** Son todos
  modelos de lenguaje leyendo la misma rúbrica. Lo que se afirma es más débil y es verdad.
- ✅ **Lo que el acuerdo entre anotadores sí cumple, y es lo que pide:** *se reporta el acuerdo, pase lo que pase.*
  **Y la tabla del §4 no se apoya en el juez LLM en ningún punto** — eso sería un modo de falla no
  aceptable de los modos de falla del proyecto, y no se comete.

### 6.2 El κ, estratificado porque el instrumento se pre-juzga a sí mismo

**κ de Cohen = 0,475** sobre los 379 pares (po 0,649 · pe 0,332), por debajo del 0,6 que el acuerdo entre anotadores fija.
**Se publica estratificado**, porque cruzando los 30 pedidos contra el texto de la rúbrica **8
aparecen textuales** —y en `c09-visual` el ejemplo **no es análogo: es el caso mismo**—, contra el
📌 que la rúbrica se puso ella sola: *"ningún ejemplo usa un pedido del set, a propósito: si la
rúbrica pre-juzga un par, el κ mide la rúbrica"*.

| Estrato | Consultas | `n` | `κ` | |
|---|---:|---:|---:|---|
| todos | 30 | 379 | 0,475 | ✗ |
| 🔴 pre-juzgado | 8 | 110 (29 %) | **0,570** | **alto por construcción** |
| ✅ **limpio — autoritativo** | **22** | **269 (71 %)** | **0,435** | ✗ |

**El estrato pre-juzgado da +0,135 más alto que el limpio**, lo que confirma la contaminación **con
una medición en vez de con un argumento**. Sin estratificar, el κ publicado habría sido 0,475.

**Y el resultado no es el κ agregado: es el κ por categoría** (estrato limpio):
`conceptual_cultural` **0,810** (n = 83) · `visual_cromatico` **0,249** (n = 86) · `ambiguo`
**0,228** (n = 100). Un **factor 3,5** entre categorías no es ruido, y la causa está medida: el
piloto fueron **20 juicios sobre 4 consultas** y las reglas se afinaron sobre lo que ahí apareció
—los pedidos culturales—. Para los otros dos tipos la rúbrica fija **el criterio y no el umbral**, y
los dos anotadores lo completaron distinto: uno declaró **siete umbrales medidos** (luminancia
media, croma en CIELAB, *"con color" = pigmento aplicado*) y el otro declaró **no haber fijado
ninguno** y haber llevado lo incierto a 1. Se ve en la distribución: **153 unos contra 104**.
🔑 **Y el `0,249` cromático tiene un tercer ángulo, medido después y con otro instrumento:** las **4**
fallas del eje 3 de la revisión de explicaciones (§5.5) son **4 de 4 sobre paleta y todas en
`visual_cromatico`**. Tres mediciones independientes —el acuerdo entre anotadores, el juez (§6.3) y
la revisión de honestidad— **señalan la misma categoría**, y por caminos que no se tocan.
**Y la causa del 0,228 en `ambiguo` está contada:** de las 10 lecturas derivadas con la escalera de
cuatro escalones, **3 convergen y 7 divergen, 2 de ellas invertidas** —cada anotador conservó la
lectura que el otro descartó—. El cuarto escalón es un corte alfabético que existe para garantizar
convergencia y **no fija el vocabulario del eje**: un anotador llegó a él dos veces y el otro
ninguna. **Queda declarado roto como mecanismo de convergencia.**

**Dos cosas más que no se pueden concluir, y van acá:**
**(1)** La fila `nDCG@5` del tercio `ambiguo` **no se puede leer como medida de relevancia**: sobre
esos ítems mide **acuerdo con una convención declarada**, porque los pedidos son comparativos sin
eje objetivo **por construcción** — son los **128 de 379 ítems (33,8 %)** que estresan el
pre-retrieval a propósito.
**(2)** El κ de este pool **no se puede comparar con el 0,419 del piloto**: aquel eran **dos
pasadas de la misma sesión** sobre 20 pares, o sea **auto-consistencia**.
**(3)** Una consulta —`c05-ambiguo`— **salió con varianza cero**, 15 de 15 en grado 0: **no aporta
a `nDCG@5` y sí al denominador** (§4.1).

### 6.3 🔴 El κ del juez INVIERTE el patrón humano, y eso ACOTA el hallazgo anterior

Un **cuarto modelo** (`openai/gpt-4o`, otro que el pipeline) leyó **la misma rúbrica** sobre el
mismo pool. Es la única prueba independiente que el argumento de §6.2 podía tener hoy — *si el
patrón por categoría se reproduce, el argumento se sostiene; si se invierte, lo que el 0,810 medía
no era la precisión del instrumento.* **Se invirtió.**
`artifacts/informes/kappa-del-juez-m13.json`, **179 de 379 pares** (los que el juez alcanzó a
juzgar antes del tope; los otros 200 quedaron sin grado, §5.4).

| | `ambiguo` | `visual_cromatico` | `conceptual_cultural` |
|---|---:|---:|---:|
| **Juez vs. consenso** (n = 67 · 49 · 63) | **0,505** | 0,450 | **0,312** |
| **Humanos**, estrato limpio (n = 100 · 86 · 83) | 0,228 | 0,249 | **0,810** |

**Donde los dos anotadores más acordaron, el juez menos acuerda.** No queda refutado —son
poblaciones distintas, 63 contra 83 ítems— 🔴 **pero la frase *"la rúbrica es precisa exactamente
donde se la refinó"* se publica ACOTADA: el 0,810 parece medir algo propio de esos dos anotadores,
o del estrato pre-juzgado, más que la precisión del instrumento.** Es la corrección más incómoda
del trabajo y va en el cuerpo igual. **Lo que la salva como hallazgo es que ahora tiene dos
mediciones y no una, y la discrepancia *es* el dato:** *el acuerdo alto en los culturales no viaja
a un lector nuevo de la rúbrica.*

### 6.4 🔑 Pero el efecto proveedor salió AL REVÉS, y eso es un resultado

Si el 71 % / 22 % del desempate (§4.2) se explicara por **proveedor compartido**, el juez —que es
de OpenAI— tendría que concordar más con el anotador de OpenAI. Medido:

| | κ (n = 179) |
|---|---:|
| juez vs. consenso | 0,414 |
| juez vs. **`ia-claude`** (otra familia) | **0,399** |
| juez vs. **`ia-codex`** (familia de OpenAI, la del pipeline) | **0,252** |

**El juez de OpenAI concuerda MÁS con el anotador de la otra familia**, por Δ = +0,148. **Entonces
el 71 % / 22 % del desempate no se explica por proveedor compartido:** un cuarto modelo, de la
familia del anotador minoritario, **reproduce la convergencia hacia el otro**. La lectura se
corrige — de *"los proveedores se agrupan"* a **"las lecturas de `ia-claude` son las que un lector
fresco de la rúbrica reproduce"**. Y hay control de coherencia: **κ(juez, consenso) 0,414 ≈
κ(juez, `ia-claude`) 0,399 ≫ κ(juez, `ia-codex`) 0,252**, que corrobora independientemente que el
consenso del tercio disputado *es* el juicio de `ia-claude`.
⚠️ **Y no es uniforme: por categoría se invierte.** En `ambiguo` el juez concuerda algo más con
`ia-codex` (0,405 vs 0,396) y en `conceptual_cultural` concuerda **0,722 con el tercero** y **0,123
con `ia-codex`**. **La asimetría agregada la manda `conceptual`**, y eso va dicho, no promediado.

### 6.5 ✅ Y la circularidad quedó demostrada en vez de asumida

la evaluación temía un escenario concreto: *κ alto en `C0` y bajo en `C4`, con el juez puntuando más
alto que los anotadores* — o sea un juez que infla los ítems que vienen de la configuración que
comparte modelo con él. **Medido, estratificado por configuración de origen** (clave
`por_configuracion_de_origen`):

| `C0` | `C1` | `C2` | `C3` | `C4` |
|---:|---:|---:|---:|---:|
| **0,468** (n=78) | 0,381 (78) | 0,362 (78) | 0,409 (79) | **0,430** (78) |

**Rango 0,362–0,468, con `C4` en el medio-alto: no hay favoritismo hacia `C4`.**
🔑 **Y el juez es más DURO que el consenso, no más blando** —**44 pares más bajo contra 21 más
alto**, 114 iguales— **y evita el grado intermedio**: usa el `1` en **7,3 %** de los casos (13 de
179) contra **25,1 %** del consenso (45 de 179). **El desacuerdo dominante es *el juez no usa el
medio*, no *el juez infla `C4`*.** Es la mitigación 2 de la evaluación verificada con número, y el κ
global **no lo habría detectado**: se mide sobre el pool agregado, así que un juez que coincide en
promedio puede estar inflando una fila y compensando con otra.
⚠️ **Los estratos no son disjuntos**: un ítem del pool tiene hasta **cinco** configuraciones de
origen, así que un par entra a varios estratos y la suma de las `n` es mayor que 379. Elegir un
origen por ítem sería **perder el dato en vez de estratificarlo**.

### 6.6 Las limitaciones restantes, en una lista

| | Qué |
|---|---|
| **Potencia** | **10 consultas por celda** en el desglose por categoría. Los tamaños de muestra son **parámetros del presupuesto, no hallazgos**: diferencias de ±0,05 no se distinguen del ruido, y así hay que decirlo si lo preguntan |
| **Un anotador vio metadata** | Uno de los dos **declaró** haber visto accidentalmente metadata de una candidata por consulta —sin imagen— antes de derivar sus fichas. **No se descarta nada**: el efecto esperable sería convergencia y lo medido es lo contrario (7 de 10 fichas divergen). Se declara como limitación del estrato `ambiguo`, que ya es el peor. 🔑 **Y lo que queda escrito es la conducta: declaró una contaminación que nadie habría detectado desde afuera** |
| **La limitación contamina a quien la lee** | El texto de la decisión del acuerdo entre anotadores lleva el κ por categoría y la asimetría de métodos entre anotadores. Se detectó que el comando lo copiaba al briefing del anotador siguiente. **Va en el artefacto y en el informe, no en el briefing** |
| **Desvíos sin consultar** | el desvío de chunking llega mitigado con números (§5.1) pero **no consultado**. el desvío de `style` (`style` fuera del output) y **la magnitud del desvío del recorte del corpus** (se avisó el recorte de tipo con ~5.900 obras en mente y hoy el corpus es 2.605) tampoco. **Ninguno se presenta como aprobado** |
| **Un tercero al correr** | *"Toda obra devuelta existe y es trazable"* depende de los servidores de los museos: los mismos artefactos dieron **148/150** y después **150/150**, y un `source_url` devolvió un lazo de **30 redirecciones** |
| 🔴 **La retención se verifica sobre la entrada y nada depura la salida** | ✅ **el registro de decisiones se cumple y la frase de privacidad del informe del Ejercicio 3 se sostiene:** de las **1.151 corridas en disco** todas menos dos son del **set de evaluación**, que no se persiste nada de un tercero **excepciona explícitamente** —*"no son dato de un tercero: son obras de nuestro propio held-out, versionadas porque son el instrumento de medición"*—, y las **2 ad-hoc tienen `entrada` vacía**, verificadas una por una. ⚠️ **Lo que queda es el mecanismo, y entra como riesgo declarado:** la política se implementa **vaciando campos de `entrada`** (`entrada.pedido = null`, `retencion.imagen = false`) y **nada depura el texto generado**, que viaja en el mismo artefacto. Para el set da igual; **para una consulta real de un usuario el texto generado podría describir su imagen y el vaciado no lo atraparía** — y §5.5 mide que **el generador ya lo hace** (3 consultas, 5 textos en `C1`). **La guardia de la regla de que un dato lleva referencia tiene que ser un validador que corra** |
| ⚠️ **Dos métricas de `la evaluación §D` no son computables como están escritas** | La **separación observación / dato** corta **binario sobre tres clases** (§5.5), así que su porcentaje no distingue la observación visual del conocimiento general del modelo. Y la **abstención correcta** no se puede computar leyendo el campo `abstencion`: en **2 de las 25** explicaciones de la muestra (`cma:97169` y `cma:150082`, las dos de `c10-visual`) el veredicto negativo viaja **en el texto de una observación** y el campo que existe para registrarlo está en `null`. **El campo no es índice confiable de *"el modelo dijo que no"*** |

---

## 7. Conclusiones

1. **La iteración funcionó y el titular es robusto.** `nDCG@5` **0,372 → 0,743** y `P@5` estricta
   **0,200 → 0,560** sobre el consenso completo —los **379** pares—; `C4` queda primero **en las dos
   poblaciones**, por **+0,208** sobre esos 379 y **+0,149** sobre los **246 de acuerdo**.
2. 🔑 **El resultado más fuerte no es el agregado: es el delta por categoría de pedido por categoría.** `C3 − C2` da
   **+0,262 (379 pares) / +0,161 (246)** en los conceptuales y **−0,089 (379) / −0,211 (246)** en
   los cromáticos, **con el signo correcto en las dos poblaciones** —el primer número de cada par
   es siempre la completa—. Y el mecanismo estaba **medido y la predicción escrita dos
   semanas antes** de que existiera la métrica que la confirma.
3. 🔴 **El orden de `C0`–`C4` cambia según el consenso, y eso se publica.** Lo que sobrevive es
   `C4` primero y el desglose por categoría. Lo que **no se puede decir nunca más** es *"`C3` le
   gana a `C2`"*.
4. **El sistema no es un Q&A y tiene con qué demostrarlo:** `QS(C0) = 0,000` **exacto** es el
   control que vuelve falsable la premisa, y `QS(C4) = 0,900` con `nDCG` alto es la única
   combinación que la evaluación declara buena.
5. **Tres resultados negativos que valen más que un número redondo:** el 39 % del índice nunca se
   cita, el presupuesto de evidencia es de suma cero, y el chunking que la entrega aprobada pedía
   tiene un costo de licencia del 21,2 %.
6. 🔑 **Y la regla de honestidad dejó de ser una promesa: `la revisión humana(c)` tiene número.** Sobre 25
   explicaciones y **51 afirmaciones**: clasificación **39/51**, la cita se sostiene en **20 de
   27**, la observación es defendible en **12 de 16**. **Pero el resultado no es el porcentaje: es
   que la regla de que un dato lleva referencia tiene dos tipos y el generador produce tres clases** —dato de museo, observación de la
   imagen y **conocimiento general del modelo**, que no tiene dónde ir—, así que la métrica de
   `la evaluación §D` **corta binario sobre una población de tres clases**. Y **una obra falla los tres
   ejes a la vez y la falla le costó al sistema la mejor obra de su propio top-5**.
7. 🔴 **Y una limitación metodológica que no se esconde: el ground truth lo produjo IA.** El κ es
   inter-modelo, el autoritativo es **0,435** sobre 269 ítems, y la mitigación de circularidad que
   pedía juicios humanos **queda formalmente rota**. Lo que la reemplaza está medido, incluido el
   hallazgo de que **el efecto proveedor salió al revés** y que **el κ del juez invierte el patrón
   humano**, lo que acota nuestro propio hallazgo anterior.

### Trabajo futuro, en orden de lo que más cambiaría el resultado

| | Qué, y por qué es lo siguiente |
|---|---|
| **1** | **Un umbral numérico declarado para los pedidos cromáticos.** Es el hallazgo más accionable: no pide reescribir la rúbrica, pide **una cifra** donde hoy hay un criterio cualitativo. Es la causa medida del `κ = 0,249` — y **tres instrumentos distintos señalan esa categoría**: el acuerdo entre anotadores, el juez, y las **4 de 4** fallas de observación visual del §5.5 |
| **2** | **Reemplazar el cuarto escalón de la escalera de pedidos ambiguos.** Está **declarado roto**: el corte alfabético depende de qué sustantivos se nominan, y la rúbrica no restringe cuáles |
| **3** | **Un filtro de formato físico en la rama visual.** Es el único mecanismo *causal* que el proyecto midió (κ 0,66 > 0,46) y hoy actúa como un filtro que nadie puso |
| **4** | **Sacar `provenance` del índice y medir qué entra en su lugar.** El 39 % del espacio no se cita; el presupuesto de evidencia es de suma cero, así que la pregunta correcta es qué campo lo reemplaza |
| **5** | **Completar el juez (200 pares de 379).** Mejora la `n` del §6.3–6.5; no habilita ningún criterio nuevo. ✅ **La otra mitad de esta fila —*modelar el costo por token y por rol*— está hecha** (§4.1 y §5.4, 2026-11-03), y costó **cero llamadas al proxy**: los tokens ya estaban en los artefactos. Lo que queda es lo que **sí** necesita presupuesto |
| **6** | **Un tercer tipo de afirmación en la regla de que un dato lleva referencia, y la cláusula 2 como validador que corre.** Es el único de los seis que **no cambia una métrica: cambia una promesa.** Hoy el conocimiento general del modelo se publica con la etiqueta de *"el modelo miró y vio esto"* (§5.5), y la cláusula que impide una observación sobre la imagen del usuario **es un arreglo que nadie ejecuta**. Los dos son baratos —un tipo en el esquema y un predicado— **y los dos mueven el `config_id`**, así que van después de la entrega, no antes. 📌 **Y de paso, el detector más barato del proyecto:** dos `description` idénticas con distinto `artwork_id` es una consulta sobre `knowledge` |

---

**Repositorio entregado — ‹HASH DEL COMMIT: PENDIENTE›**
⚠️ *Lugar reservado a propósito. La exportación a la rama `main` es otra pasada, y el hash se pega
recién cuando ese commit exista; el criterio de la entrega en fecha y en formato lo pide junto con la presentación el 18/11 18:59.*

*Fuentes: AIC y CMA, obras de dominio público / CC0. `description` de AIC es **CC BY 4.0**; este
informe no reproduce fragmentos de esos textos. Todas las cifras las emitió nuestro pipeline entre
el 14/10 y el 1/11 de 2026 y cada una lleva su población y su artefacto; donde la cifra no sale de
un artefacto, está dicho. El ground truth lo produjeron anotadores de IA (la anotación por IA) y el
acuerdo entre ellos se reporta en el §6.*

---
---

# Anexo — fuera del entregable

> **No es parte del informe.** Es lo que le permite al grupo cambiar la selección sin rehacerlo, y
> lo que la próxima sesión necesita para no repetir una verificación. **La selección final es del
> grupo; acá se propone.**

## A. Qué entró y con qué criterio

El criterio fue uno y es el mismo del Ejercicio 3: **un resultado entra si lo podemos sostener con
una cifra propia, medida, y citar el artefacto que la emite.** Lo que solo podemos afirmar, no
entra. Como el enunciado **no limita la extensión** del Ejercicio 4 (a diferencia del Ej. 3), la
restricción no fueron las líneas sino la **trazabilidad**, y mordió cuatro veces (sección C).

📌 **Y una entrada tardía, que se declara porque cambia un veredicto:** el **§5.5** (`la revisión humana(c)`) se
produjo **en paralelo con este informe** y entró después. La versión anterior declaraba el criterio de honestidad *"no se
cumple: la revisión humana y la cita que respalda la afirmación no tienen número"*; **ahora lo tienen**, así que el §5.5, la conclusión 6,
dos filas del §6.6, la fila el criterio de honestidad del §F y el punto 5 del §E **se reescribieron sobre el artefacto**
(`revision-explicaciones/m13b-c1-5x5.json`) y no sobre el reporte. **El veredicto nuevo no es
*"cumple"*: es *"medido, y el resultado hay que leerlo con dos salvedades"*.**

**Jerarquía de titulares, y es una decisión:** el titular es **el delta por categoría de pedido por categoría** (§4.3) y **no**
el orden global de `C0`–`C4`. Razón: el desglose **sobrevive** la sensibilidad del tercio disputado
y el agregado **no** —el `C3−C2` global cambia de signo—. Un informe que titulara con el orden
quedaría apoyado en la cifra más frágil que el estudio produjo.

## B. Qué quedó afuera, y por qué

| Tema | Por qué no entró |
|---|---|
| **El κ ponderado como rescate de la escala** (0,437 lineal · 0,460 cuadrático contra 0,419 sin pesos) | Es un **resultado negativo medido** y vale, pero es sobre el **piloto** (20 pares), no sobre el pool. Entraría en §6.2 en una oración si el grupo quiere cerrar de antemano la propuesta *"usen κ ponderado"* en el oral — **y conviene llevarlo al oral igual**, porque es la objeción más obvia |
| **el hallazgo de `style`: `style_title` de AIC es un campo de cultura disfrazado** (separan `Himalayan` 0,86 · `Chinese` 0,75 · `Japanese` 0,71; `Renaissance` 0,24 y `19th century` 0,06 no separan nada) | Es excelente material y **refuerza el desvío de `style`**, pero `style` **no es parte de la iteración v1→v2**: no cambió entre las dos versiones. Va al oral si preguntan por qué `style` salió del output |
| **Los tres defectos de ingesta que los tres anotadores encontraron por separado** (`cma:160723`: la imagen es la seda de montaje del rollo, no la pintura; `cma:136296`: thumbnail de 900 × 46 px) | 🔑 **Es la mejor evidencia que el proyecto tiene de que la cobertura de imagen no es binaria**, y **tres hallazgos independientes del mismo defecto** es más fuerte que cualquier argumento. Quedó afuera porque es material del relevamiento y no de la iteración. **Primer candidato a entrar**, en §6.6, ~40 palabras. 📌 **Y ya son cuatro:** el §5.5 publica el cuarto (`cma:148862`, dos `description` idénticas que describen otra obra) porque ese sí llega hasta una cita que el sistema publica — **y los cuatro los encontró mirar las obras de a una, no un validador** |
| **El reparto 5/5 de pedidos culturales que cruzan formato** | Entró, pero como 📌 de dos líneas en §4.3. Merece más: es la decisión de diseño que **hace interpretable** el delta `C3−C2`. Si el grupo quiere subirlo, va como bloque propio en §2 |
| **La degradación elegante corrida** (301 corridas de inyección, 6 etapas, ≥ 30 cada una) | Está en el informe del **Ejercicio 3** y duplicarlo gasta espacio en los dos. Va al oral en el bloque de producción |
| **El desglose de latencia por etapa** | la latencia por etapa no tiene umbral —no hay baseline publicado ni requisito de la cátedra— así que el p50/p95 por fila alcanza. El por-etapa es material de oral si preguntan de dónde salen los 65 s |

## C. 🔴 Cifras que NO publiqué porque no pude citar el artefacto

**Las cuatro son correctas en el registro del proyecto; lo que falta es el artefacto que las
emita.** Van al orquestador.

1. **`referencias 116 → 104` y `abstenciones 41 → 44 (+7 %)`** del brazo ciego sobre los 103 pares
   comunes. Solo existen en el registro de hallazgos, **y la abstención está explícitamente
   calculada a mano** —*"no tiene versión por pares comunes en ninguna salida"*—. **Publiqué las
   que sí tienen artefacto**: `1,126 → 1,010` referencias por obra y `103/103` vs `102/103`.
2. **`did_you_know 72 → 44` obras** del presupuesto de suma cero. Solo. Lo
   publiqué porque las otras dos cifras del mismo trío (`142→136`, `128→29`) **sí** están en
   `artifacts/informes/ejercicio-3.md`, y romper el trío habría sido peor — **pero la tercera no
   tiene artefacto y queda dicho acá.**
3. **Los fragmentos citados por campo** (`description` 177 · `did_you_know` 21 · `citations` 4 ·
   `exhibitions` 1 · `provenance` 0). Solo. Publiqué el `128 → 0` y `29 → 0`, que
   sí están en `ejercicio-3.md`.
4. ✅ **Resuelta el 1/11, y la dejo dicha porque muestra la forma del arreglo.** El **churn del eje
   de chunking** (1,50 · 1,80 · 1,40 de 5) y el **19,33 %** sin juzgar **ya los emite un
   artefacto** —`retrieval-secundario-m14b.json`, claves `churn_del_top5_contra_c3` e
   `items_del_top5_sin_juzgar`— y el §5.3b los publica citándolo. **No hacía falta reescribir la
   prosa: hacía falta que el pipeline emitiera la cifra.**

**Y dos cifras que publiqué con la etiqueta cambiada a propósito:** las `QS` de `suma` (0,682) y
`concatenacion` (0,649) del §2.2, y la tabla de `α` — **medidas, con sus corridas en disco, pero
nunca escritas a un artefacto**. El patrón es consistente y conviene nombrarlo: **los conteos de
construcción de índice son artefactos estructurados; los deltas entre brazos existen solo como
prosa.** Cualquier cifra que exija unir las corridas de dos brazos está sin emitir.

**Y tres cosas del §5.5 que salen de y no de un artefacto, dichas con esa
etiqueta:** las **tres formulaciones** que explican el `0` de la medición anterior (*en comparación con*, *similar
a*, *el mismo que*, el hallazgo de las formulaciones comparativas); **que las 2 corridas ad-hoc tienen `entrada` vacía**, verificada una
por una (el hallazgo de las corridas ad-hoc); y que **las 1.151 corridas son todas del set menos esas dos** —esta última sí
tiene hermana publicada, en `ejercicio-3.md`—. **Los tres conteos de los ejes, el caso de
`aic:43211` y los seis hallazgos de diseño sí los emite el artefacto**, que es de donde salen todas
las cifras del §5.5.

## D. 🔴 Hallazgos sobre el material — van al orquestador

1. 🔴 **La cuenta de incidentes del encargo no coincide, y es del mismo tipo que
   los incidentes que cuenta.** Esta revisión afirma **catorce** dos veces (*"la
   dirección cómoda que este proyecto documenta catorce veces"* y *"catorce incidentes de cifras
   mal publicadas"*). **En el máximo afirmado es trece** —la medición del piso de ruido dice
   *"decimotercera vez que una cifra se corrige al medirla"* y una fila del registro de hallazgos *"las trece veces
   anteriores"*—, y el `catorce` aparece tres veces en el archivo contando **otra cosa** (los 14
   hallazgos de la auditoría de Fase 3, y cuántas cosas reportaron dos sesiones).
   **Y el contador es internamente inconsistente:** otra fila del registro de hallazgos dice *"doce veces"* y la fila siguiente,
   la medición del piso de ruido, dice *"decimotercera"*; antes, una línea dice *"diez veces"* y otra cita a una sesión
   diciendo *"nueve"* con un *"y tenía razón"* al lado. **Cuentan tres series distintas** —*error
   encontrado al abrir un archivo* · *cifra corregida al medirla* · *la dirección cómoda*— que
   nunca se declararon como una. **Por eso no publiqué ninguna cuenta en el informe ni en la
   presentación: publiqué el patrón.** 🔑 **Es el modo de falla 3 sobre el contador del modo de
   falla 3, y la forma del arreglo ya está escrita en el repo: la cifra la emite el artefacto.**
   Si el grupo quiere un número en el oral, hay que **definir la serie** y contarla una vez.
2. ⚠️ **Dos artefactos dicen *"llamadas por consulta de `C4`"* y dan 21 y 33. No se contradicen, y
   conviene que no se promedien.** `tabla-a2-m13.json` publica **21,0** —son **llamadas al LLM**:
   20 de reranking + 1 del generador—. `ejercicio-3.md` publica **33**, que es **todo el libro
   mayor**: esas 21 más 2 `embedding_texto`, 2 `embedding_imagen` (CLIP local) y 8 lecturas al
   índice. **21 + 2 + 2 + 8 = 33 exacto**, verificado contra las 30 corridas de
   `e30ca8c5003c614c`. Las dos cifras son correctas con definiciones distintas de *"llamada"*.
   **Recomendación: que el que publica diga `llamadas al LLM` o `entradas del libro mayor`**, nunca
   *"llamadas"* pelado.
3. ⚠️ **el hallazgo de la figura UMAP cita tres κ que no están en `umap-procedencia.json`.** Los *"China 0,54 ·
   otras 0,52 · Mughal India 0,49"* del formato **dentro** de cada cultura **no existen** en ese
   artefacto —no hay una clave `formato_dentro_de_cada_cultura`; sus `controles` son
   `tema_dentro_de_cada_formato` y `tema_dentro_de_cada_cultura`—. **Sí están en
   `umap-espacio-clip.ipynb`**, con un cuarto valor que el hallazgo de la figura UMAP no cita: **no se sostiene en Northern
   India (0,02)**. Cité el notebook y publiqué el control negativo.
4. 🟡 **`la evaluación § Acuerdo entre anotadores` conserva un 📌 vencido.** Su último recuadro dice que
   el κ del acuerdo entre anotadores se calcula *"sobre el ~20 % del pool definitivo (**~108** juicios duplicados)"*.
   El pool congelado publica **76**, y el mismo documento lo corrige **doce líneas más arriba**.
   Es la misma deuda del relevamiento de cifras en un párrafo que la pasada de deuda no tocó. (No lo edité: `docs/`
   está fuera de mi alcance.)
5. 🟡 **`el plan de construcción:34,193`** sigue con `~45 min` y `~108`, según el propio el relevamiento de cifras. No verificado
   por mí — está fuera de mi alcance de lectura obligatoria y lo anoto porque el relevamiento de cifras lo deja
   abierto.
6. 🟡 **El informe del Ejercicio 3 publica `4,50` de cobertura cultural para `C4` y su propio anexo
   reporta `4,40` al recontarlo a mano.** Sigue sin cerrar. Publiqué **4,50 con la discrepancia al
   lado**, igual que él. **Una obra en 50, antes de una entrega, conviene mirarla.**
7. 🔴 Dos renglones del registro interno afirman el `0 de 148` que el §5.5 refuta, y uno de ellos saca una
   conclusión de diseño de ahí.** el diseño de la arquitectura (la fila de la regla de que un dato lleva referencia y el resumen de `§6.1`) dice
   *"hoy no se viola —verificado: **0 de 148** observaciones persistidas mencionan la imagen de
   consulta—, **y por eso entra como guardia y no como arreglo**"*, y la fila **la medición anterior de
   ** repite el `0` con un *"verificado por el orquestador: no es así hoy"*. **Son
   tres consultas** (§5.5). ⚠️ **el hallazgo de las formulaciones comparativas ya lo corrige más abajo en el mismo archivo**,
   así que la fila la medición anterior quedó **internamente contradicha**; el diseño de la arquitectura no. **No los edité:
   ese registro queda sin corregir.** 🔑 **Y la corrección que importa no es el número: es que la
   cláusula pasa de *guardia* a *arreglo*,** y eso cambia qué hay que construir.
8. ⚠️ **El artefacto de la revisión afirma que la política de retención NO se cumple, y la
   verificación de población dice que sí. Publiqué la segunda, y declaro las dos.** El artefacto
   (`hallazgo_de_retencion`) escribe que *"vaciar `entrada.pedido` y dejarlo escrito 56 veces en la
   salida **no cumple** «no se persiste nada de un tercero»"*. 🔑 **el registro de decisiones excepciona
   explícitamente las 30 consultas del set** —*"no son dato de un tercero: son el instrumento de
   medición"*— **y las 1.151 corridas en disco son todas del set menos dos, cuyas `entrada` están
   vacías** (el hallazgo de las corridas ad-hoc, verificadas una por una). **Entonces que no se persiste nada de un tercero se cumple y la frase de
   privacidad del informe del Ejercicio 3 se sostiene.** Lo que publiqué es **el mecanismo como
   riesgo declarado** (§6.6), que es la parte real. 📌 **Y la forma del casi-error vale guardarla:
   el artefacto midió el texto correcto y le puso la población equivocada**, que es el modo de falla
   que este anexo describe en el §A — esta vez **en la dirección incómoda**, que es la que nadie
   revisa.
9. 🟡 **`la evaluación §D` mide una regla que el contrato de datos no puede cumplir enteramente.** La
   *separación observación / dato* supone **dos** clases y el generador produce **tres**, y la
   `ficha` del museo **no es citable**, así que parte de la revisión humana mide el esquema de evidencia y no al
   modelo (§5.5). **Es una deuda de la métrica, no del informe**, y va al orquestador con la fila
   el criterio de honestidad del §F al lado.

## E. Afirmaciones que me hubiera gustado hacer y no pude sostener

1. **"El delta `C4−C3` está por encima del piso de ruido de `C4`".** Lo digo más débil en §4.4,
   porque **el piso de `C4` no está medido en el `config_id` publicado**. Lo que sí puedo afirmar
   es que +0,208 —el `C4 − C3` de la población **(a)**, los **379** pares del consenso completo con
   política `cero`— está muy por encima de la banda de ±0,8 obras **medida en otro `config_id` con
   la misma etiqueta**, y que el piso de `C3` es **0,000 exacto** en su `config_id` real.
2. **"La rúbrica es precisa donde se la refinó".** Era la interpretación que el proyecto había
   escrito, y **un cuarto modelo leyendo la misma rúbrica no la reproduce** (§6.3). Publiqué la
   versión acotada.
3. **"El chunking ciego recupera peor".** No se puede, **y ahora el motivo es mejor**: ya no es que
   falte el `nDCG@5` —el pooling incremental lo midió (§5.3b)— sino que **el `Δ` contra `C3` cambia
   de signo entre las dos poblaciones del ground truth**, y el confundidor de la restringida
   explica el cambio. **Lo publicable es un no-resultado**: el eje de chunking **no mueve el
   retrieval de forma concluyente**. Lo que el informe sí dice es que **atribuye** peor —el 21,2 %
   del §5.1— y que **mueve mucho el top-5** —churn 1,567 de 5—, que son las dos cosas medidas.
4. ✅ **"El costo por consulta es X USD".** ⚠️ **Resuelto el 2026-11-03, y la frase anterior
   describe bien por qué podía resolverse sin gastar nada.** Decía: *"el dólar sale ausente porque
   el precio de los modelos de chat no está declarado… **es una tarea de una persona con un
   navegador**: anotar el precio con su fecha y su URL"*. Es exactamente lo que se hizo
   (`developers.openai.com/api/docs/pricing`, 2026-11-03), y la columna salió **sin re-correr una
   sola consulta**: los tokens ya estaban medidos. **`C1` 0,0029 → `C4` 0,0277 USD/consulta, 9,6×**
   (§4.1), y el total derivado reconcilia con el dashboard —el proxy factura un **19 % por debajo
   del precio de lista**, con lista como base (§5.4)—.
   **Lo que sigue siendo cierto:** el dólar es **derivado de un precio declarado** —de lista, no
   del proxy— y nunca se toma de LangSmith. *Un modelo sin precio sigue dando ausente, no cero.*
   🔑 **Y la ausencia dejó un segundo hallazgo que nadie buscaba:** derivar el USD obligó a mirar
   el libro mayor rol por rol, y ahí apareció que **`C2`–`C4` publicaban una llamada de menos por
   consulta** (el reuso del plan al barrer el peso de fusión, §4.1). La columna que faltaba estaba tapando un error en la que sí estaba.
5. **"La regla de honestidad se cumple".** Ahora **tiene número** (§5.5) y lo que el número dice es
   más incómodo que la ausencia: **12 de 51** afirmaciones mal clasificadas, **7 de 27** citas que
   no se sostienen del todo, **4 de 16** observaciones no defendibles. Lo que **no** puedo afirmar
   es que el resultado **escale a las 1.151 corridas**: la muestra son **25 explicaciones de un
   brazo**, el tamaño lo fijó el criterio de honestidad antes de medir y **agrandarlo ahora sería mover la
   vara después de ver el resultado**. Lo que sí viaja es el argumento de instrumento —el `binding`
   del generador es byte-idéntico en `C1`–`C4`—, que es **un argumento, no una medición**.
6. **"Y la anotó un humano".** No: la anotó IA, como los 379 juicios (la anotación por IA), y
   el criterio de honestidad dice *"muestra humana"*. **La letra del criterio de honestidad falla en quién anota, igual que la
   de la documentación del cuarto ejercicio y el acuerdo entre anotadores.**

## F. 🔴 Criterios de entrega que creo que NO se cumplen

| Criterio | Mi lectura |
|---|---|
| la documentación del cuarto ejercicio | 🔴 **Se cumple a medias, y la mitad que falla hay que decirla en voz alta.** De las dos cifras que la documentación del cuarto ejercicio nombra para confirmar la limitación de v1, **una se cumple solo en una de las dos poblaciones** (§1.2) y **la otra no es computable por construcción** (`QS` por categoría no existe). El criterio pide *"la limitación confirmada con número"*; lo que tenemos es **una confirmación condicionada a la población y una ausencia estructural**. El informe lo publica así. ⚠️ **Y la segunda mitad de la documentación del cuarto ejercicio —*"la comparación apoyada únicamente en anotación humana"*— NO se cumple**, porque **ningún humano anotó**. Se cumple su *espíritu* —la tabla no se apoya en el juez LLM— pero no su letra |
| el acuerdo entre anotadores | ✅ **Se cumple como está escrito** —*se reporta el acuerdo pase lo que pase*—, con la salida *limitación declarada* ejercida y el texto publicado. ⚠️ **Pero dos de sus supuestos cayeron:** el κ es inter-modelo y no inter-anotador, y la decisión del acuerdo entre anotadores **la escribió el orquestador bajo delegación del grupo; no la leyeron ni la firmaron los integrantes**, y eso está declarado en el propio artefacto. Y el **κ juez-vs-anotadores estratificado por configuración** que el acuerdo entre anotadores también pide está sobre **179 de 379 pares**, no sobre el pool entero |
| el criterio de la entrega en fecha y en formato | ⚠️ **Está en riesgo por una dependencia externa, no por el contenido.** El entregable es **el hash de un commit** y los artefactos **no van al remoto** (el repo publicado); lo que compensa eso es la promesa de regeneración del `README`, **y con el tope del proxy agotado esos comandos no corren**. No falla hoy, pero la afirmación del `README` tiene que llevar su condición |
| el dataset documentado y reproducible | ⚠️ **Se cumple en `diseno-sdd` y se degrada en el repo entregado**, por el repo publicado. Las cifras existen y las emite el pipeline; lo que no existe en `main` es el artefacto que las versiona |
| el criterio de honestidad | ⚠️ **Medido — y el veredicto cambió el 1/11, después de que el resto del informe estuviera escrito.** el conteo de referencias colgadas se cumple con umbral duro (**0 referencias colgadas**: 0/164 en las cinco filas, 0/27 en la muestra). **la revisión humana y la cita que respalda la afirmación ya tienen número** (§5.5, artefacto `revision-explicaciones/m13b-c1-5x5.json`): clasificación **39/51**, la cita se sostiene **20/27** (+4 parciales, 3 que no), la observación es defendible **12/16** (2 discutibles, 2 falsas). 🔴 **Las cifras van con dos salvedades que no son opcionales:** de las 6 fallas de tipo **2 están forzadas por el contrato** —la `ficha` no es citable—, así que **esa fracción mide el esquema de evidencia y no al modelo**; y el `%` de *separación observación / dato* de `la evaluación §D` **corta binario sobre tres clases**. ⚠️ **Y la letra falla en quién anota:** el criterio de honestidad dice *"muestra **humana**"* y la anotó IA (la anotación por IA), igual que los 379 juicios. **Entonces: el criterio pasó de *sin medir* a *medido con el resultado publicado y dos salvedades*, no a *cumplido*** |
| el objetivo de la interfaz | ✅ **No se cumple y está bien que no se cumpla.** La UI murió por su disparador: el corte de la interfaz pedía la ronda piloto anotada por **al menos dos personas** y se estableció que no lo está. Es un extra, el disparador era objetivo, y **se cerró sin reabrir la discusión** — que era exactamente para lo que el disparador existía |
| el valor del juez | **No se cumple, y es un objetivo, no un criterio.** `κ` juez-consenso = **0,414 < 0,6**. la evaluación lo dice: *un κ bajo honestamente reportado es un hallazgo* |

**Lo que sí cumplimos sin asterisco:** el pipeline base completo y andando (tag `v1`, 30/30, **self-retrieval 20/20 con `α = 1`**),
la tabla de ablación (5 filas × 7 columnas + desglose por las tres categorías + `%` sin juzgar), las técnicas avanzadas aisladas por medición (cuatro
técnicas, cada una con su artefacto, con los deltas que las aíslan de a dos), la tabla de tokens, costo y la figura (LangSmith
integrado, tokens por etapa, UMAP sobre las indexadas), el pre y postprocesamiento medido, el criterio de que el pedido cambia el resultado (**`QS(C0) = 0,000` exacto**),
el criterio de trazabilidad (**150/150**), el conteo de referencias colgadas (**0/164** referencias colgadas, y **0/27** sobre la muestra del §5.5),
el criterio de atribución (**32/32** — y **7/7** sobre la muestra del criterio de honestidad, que es la verificación que el criterio nombra),
el criterio de costo, el criterio de respuesta garantizada y el criterio de idioma (**314/314** — y **25/25** sobre la muestra, que es la cifra que el criterio de idioma compromete
textualmente, con la cita **en inglés sin traducir** en las 25).

## G. Qué del material no cerró al escribirlo

1. **El `0,419` que aparece dos veces.** `QS(C1) = 0,419` y el `κ` del piloto `= 0,419`. Son dos
   mediciones sin relación y coinciden al tercer decimal. **Lo marqué en §1.2 para que nadie las
   junte en el oral**, pero conviene que alguien más lo verifique: es el tipo de coincidencia que
   produce una pregunta incómoda.
2. **el criterio de costo nominal vs. pagado, en las dos direcciones.** `C1` pagó **31** llamadas sobre 30
   nominales (un reintento) y `C2`–`C4` pagaron **una menos** porque se reusaron los 30 planes. O
   sea que `2 + N` es **cota superior por un lado e inferior por el otro**. Publiqué las **pagadas,
   que es lo que emite el libro mayor**, y no la nominal.
3. **El juez emitió 379 juicios y 200 salieron sin grado.** El artefacto lo declara
   (`n_sin_grado: 200`), y es el tope del proxy. Pero significa que **el κ del juez está sobre 179
   pares elegidos por dónde se cortó el presupuesto**, no por un criterio metodológico. No creo que
   sesgue —el orden de juzgado es el del pool— **pero no está verificado**, y preferiría que lo
   estuviera antes del oral.
4. **El estrato limpio del κ tiene 6 consultas conceptuales de 22**, y es justo la categoría del
   `0,810`. Con `n = 83` ítems el κ es sólido, pero **seis consultas es poco** para una afirmación
   sobre una categoría. §6.3 ya lo acota por otra vía; esto lo acota por una segunda.
5. **El `0` de la medición anterior tiene dos conteos y los reconcilié antes de publicar.** El artefacto del §5.5
   reporta **2 de 148** *sobre observaciones*, que es la definición que la medición anterior usó, y **5 textos en
   `C1`** sobre los tres campos que el generador persiste (`observacion`, `dato`, `abstencion`);
   el hallazgo de las formulaciones comparativas reporta **tres consultas**. **No se contradicen** —2 observaciones en
   `c05-visual` y `c08-visual`, más un `dato` en `c09-visual`— y publiqué **las tres consultas con
   su tipo al lado**, que es la forma que no se puede promediar mal. ⚠️ **Lo que no cerró:** el texto
   que el hallazgo de las formulaciones comparativas cita para `c09-visual` (*"el mismo que el de la imagen subida"*) **no es el mismo
   string** que el artefacto adjunta para esa consulta (`cma:147045`, *"...de la imagen
   proporcionada"*). **Cité el del artefacto**, que es el que puedo mostrar; puede haber **dos**
   textos en esa consulta y nadie los contó por separado.
6. **La muestra de 25 no está estratificada como el pool de 379, y es a propósito.** Las 5 consultas
   se eligieron **sobre metadatos** —categoría, museo, licencia, conteo de abstenciones— y no sobre
   el estrato pre-juzgado de la rúbrica, porque **la rúbrica no entra en el prompt del generador**
   y la estratificación del κ no aplica. **Es correcto y conviene que alguien lo confirme en voz alta antes del
   oral**, porque es la pregunta natural de quien escuchó el §6.2.
