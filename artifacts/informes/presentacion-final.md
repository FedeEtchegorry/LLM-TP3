# Presentación final — ~30 minutos

**Grupo 2 · 73.69 Large Language Models · ITBA, 2026**
*Recuperación de obras condicionada a un pedido sobre una imagen de referencia.*

> ### 📐 El formato lo elegimos nosotros. Esto es el guion, no la maqueta
>
> La cátedra **no fija formato** para la presentación: pide que contenga **seis cosas**
> (`el enunciado § Presentación oral final`) — el problema, los desafíos, las decisiones de
> implementación **bien justificadas**, los resultados **en ambas versiones**, limitaciones y
> trabajo futuro, y conclusiones. **Las seis están mapeadas abajo, en la columna *«cubre»*.**
>
> **Qué es este archivo y qué no.** Es el **guion**: 18 diapositivas, cada una con su presupuesto
> de minutos, **qué se muestra**, **qué se dice** y 🔗 **el artefacto del que sale cada cifra**. No
> es el `.pptx` — la maqueta la arma quien presente, y la sección **§Z** dice exactamente qué
> necesita para armarla (seis figuras, ya en disco, más cuatro tablas que se copian de
> `ejercicio-4.md`). **Lo defendible es el guion; el resto es tipografía.**
>
> **Cinco decisiones de contenido, porque el reparto de minutos *es* el argumento:**
> **(1)** Los resultados se llevan **7,5 minutos** y *«cómo lo medimos y qué no podemos afirmar»* se
> lleva **7**. Es casi mitad y mitad a propósito: en la rúbrica de 6 conceptos, **Métricas** y
> **Evaluación** son dos conceptos separados, y el segundo se subestima siempre.
> **(2)** El titular es **el delta por categoría de pedido por categoría**, no el orden global de `C0`–`C4`. El orden global
> **cambia** según el consenso (D-9) y el desglose **no**: se titula con lo robusto.
> **(3)** Hay una diapositiva que nadie espera — **D-14, *«qué nos salió mal y cómo lo supimos»***.
> Es método, y es lo que vuelve creíble todo lo anterior.
> **(4)** Y **D-16b** entró último, porque la medición entró última: la **regla de honestidad**
> —*dato del museo con su cita, observación del modelo marcada como propia*— es lo que la
> diapositiva D-1 promete, y **se mide, no se promete**. El minuto y medio sale de **arquitectura**,
> siguiendo el criterio que este archivo ya tenía escrito: *se recorta arquitectura, no evaluación.*
> **(5)** 🔴 **Y la mitad de *retrieval* del estudio secundario NO entra como diapositiva, a
> propósito.** Está medida y publicada —`ejercicio-4.md §5.3b`— y su resultado es un **no-resultado**:
> el `Δ` del eje de chunking **cambia de signo** entre las dos poblaciones del ground truth, con un
> confundidor de protocolo que lo explica. **Contarlo bien son dos minutos** —tabla, cambio de
> signo, confundidor— **y los dos minutos solo salen de D-14 o del bloque 4, que son justamente lo
> que este archivo declara que no se recorta.** Entra, en cambio, **como respuesta preparada** en
> §Z, que es donde cae la pregunta si alguien la hace. 🔑 **Y es coherente con la decisión (2):** se
> titula con lo robusto, y un no-resultado **no se titula; se contesta.**

**Reparto, 30 minutos:**

| Bloque | Diapositivas | Min | Cubre |
|---|---|---:|---|
| **1 · El problema, y por qué no es un Q&A** | D-1 … D-3 | **4** | *el problema* |
| **2 · La arquitectura, por sus decisiones** | D-4 … D-7 | **5** | *decisiones justificadas* |
| **3 · 🔑 El resultado** | D-8 … D-11 | **7,5** | *resultados en ambas versiones* |
| **4 · 🔴 Cómo lo medimos, y qué no podemos afirmar** | D-12 … D-14 | **7** | *desafíos* · *limitaciones* |
| **5 · Producción y Responsible AI** | D-15 · D-16 · **D-16b** | **5,5** | *trabajo futuro* · *limitaciones* |
| **6 · Cierre** | D-17 | **1** | *conclusiones* |

---

## Bloque 1 — El problema, y por qué no es un Q&A · 4 min

### D-1 · La entrada son **dos** cosas, no una · 1,5 min

**Se muestra:** una imagen del held-out y **tres** pedidos al lado, uno por categoría
(*"versiones más cálidas"* · *"el mismo tema pero en la cultura japonesa"* · *"con más vida"*), con
**tres top-5 distintos** debajo. Nada de texto.

**Se dice:** el usuario aporta **una imagen y un pedido**, y el pedido **no filtra** los
resultados: **define cuál es el resultado correcto**. La salida son 5 pinturas **reales** de dos
museos, cada una con explicación en español **citada a un fragmento verificable** — porque el
output no es terminal: el usuario **sigue el link a la ficha del museo**. 🔑 **Y esa promesa la
medimos afirmación por afirmación: D-16b.** Está materializado en un
solo hiperparámetro: `e_q = normalize(α·e_img + (1−α)·e_txt)`, donde **`α = 1` degenera en un
buscador visual** y `α = 0` en text-to-image.

### D-2 · Por qué no es un Q&A, y es una respuesta de tres frases · 1 min

**Se muestra:** tres líneas grandes.

**Se dice:** **(1)** La respuesta correcta **no está escrita en ningún documento** — es un ítem de
un catálogo, y puede no estar descrita por ningún fragmento. **(2)** La entrada no es una pregunta
y la salida no es un párrafo: **es un ranking**. **(3)** Hay **dos espacios de embedding
incompatibles que hay que fusionar** — un coseno de CLIP y uno de `text-embedding-3-small` **no son
calibrables entre sí**, y eso es un problema de diseño que un Q&A no tiene. *(Volvemos a esta
tercera en D-6, con número.)*

### D-3 · 🔑 El criterio es falsable, y acá está la demostración · 1,5 min

**Se muestra:** `QS = 1 − media(Jaccard(top5_i, top5_j))` arriba, y debajo una sola fila:
**`QS(C0) = 0,000` exacto**.

**Se dice:** *"la misma imagen con pedidos distintos devuelve obras distintas"* suena a marketing,
así que lo convertimos en una métrica con **un valor que significa el fracaso**: `QS = 0` es
literalmente *"el sistema ignora al usuario"*. Un buscador por similitud visual tiene `QS = 0`
**por construcción**, y lo construimos: es el brazo `C0`, con `α = 1`, donde la combinación
cortocircuita y devuelve el embedding de la imagen **sin tocar**, así que el vector de consulta **no
depende del pedido en ningún bit**. **El `0,000` es exacto, no redondeado.** Es el control que
vuelve falsable la premisa — y al final de la charla vamos a ver que el sistema completo llega a
**0,900**.
🔗 `artifacts/informes/tabla-a2-m13.json` → `columnas_sin_ground_truth.C0.qs` · `config/c0.toml`

---

## Bloque 2 — La arquitectura, por sus decisiones · 5 min

### D-4 · El pipeline, y la limitación de v1 en el mismo diagrama · 1 min

**Se muestra:** el flujo de 8 etapas, con las **cuatro** de v2 resaltadas y las cuatro de v1 en
gris. Al pie: **v1 = `C1`** · **v2 = `C4`**.

**Se dice:** el enunciado exige tener el **RAG base completo antes de agregar nada avanzado**, así
que la ablación son **cinco configuraciones por inclusión incremental**: cada cadena de etapas es un
**prefijo** de la siguiente con una etapa más. v1 es `C1` —un baseline **justo**, no trucado:
ganarle a `C0` no demostraría nada—. **Y la limitación de v1 está en el diagrama:** v1 embebe el
**pedido crudo** con el text encoder de CLIP, que está entrenado con **captions descriptivos**.
*"Versiones más cálidas"* **no es la descripción de ninguna imagen**: es una instrucción relativa a
una referencia que el text encoder no ve. v2 agrega cuatro etapas y **no saca ninguna**.

### D-5 · 🔑 El mecanismo que medimos **antes** de tener la métrica · 2 min

**Se muestra:** `umap-05-separacion.png`, y encima dos κ grandes: **formato 0,66** vs. **cultura
0,46**.

**Se dice:** esta es la figura del extra de UMAP, y encontró algo que **no estábamos buscando**.
Sobre las 2.605 obras indexadas, **el formato físico separa el espacio de CLIP más que la cultura**:
κ del vecino **0,663** para *folio de manuscrito vs. pintura de caballete* contra **0,459** para
`culture`. Se sostiene **dentro** de cada cultura con masa de los dos formatos — y **no se sostiene
en Northern India, 0,02**, que lo decimos igual.
**La consecuencia es un filtro que nadie puso:** en la rama visual, quien sube una pintura de
caballete recibe pinturas de caballete **para cualquier pedido**. Y en el corpus la cultura y el
formato están casi perfectamente confundidos.
🔑 **De acá salió una predicción, y la escribimos el 18 de octubre:** si un pedido cultural cruza
formatos y la rama visual no, entonces **el beneficio de agregar la rama de texto tiene que
concentrarse en los pedidos culturales**. En ese momento `nDCG@5` **no existía todavía**. La
confirmación es la diapositiva **D-10**.
🔗 `artifacts/informes/umap-procedencia.json` (κ, con `n` y azar) · `umap-espacio-clip.ipynb` (los
controles por cultura)

### D-6 · Las cuatro técnicas, cada una con su artefacto · 1 min

**Se muestra:** una tabla de cuatro filas: técnica → etapa → **la clave del artefacto que la
prueba**.

**Se dice:** el enunciado pide **una** técnica avanzada; entregamos **cuatro**, y la evidencia es
el artefacto, no el conteo de trazas.
**Query rewriting** y **query expansion** salen de **una sola llamada** y las prueba el mismo
`PlanDeConsulta`, con **un booleano para cada una**: `distintas_del_pedido_crudo` es rewriting,
`distintas_entre_si` es expansion. **La fusión RRF** la prueba el `ranking_fusionado` **con los
rangos de origen por rama**. **El reranking**, el `score_0_10` con su justificación y el rango
previo.
🔑 **Y las cuatro se pueden ver sobre una sola consulta de punta a punta** — el pedido *"escenas de
agua en la pintura china"*, cuyo plan es el que esa corrida consume, y donde **la justificación del
reranker le devuelve el pedido reescrito**.
⚠️ **Y una técnica que decidimos NO presentar como quinta:** el filtrado por metadata. Usamos
`where`, pero como **join** —traer los fragmentos de una obra por su id— no para angostar el espacio
de búsqueda, que es lo que la técnica significa. **Cuatro sólidas valen más que cinco discutibles.**
🔗 `artifacts/planes/d408abc4d01bdd73.json` + `artifacts/corridas/e3821d8533de06a3-1/corrida.json`

### D-7 · 🔑 Una decisión que dejó de ser un argumento y pasó a ser un número · 1 min

**Se muestra:** tres filas. **`rrf` 0,727** · `suma de scores` 0,682 · `concatenacion` 0,649.

**Se dice:** elegimos **RRF con k = 60** con un argumento: los scores de las dos ramas **no son
calibrables entre sí**, así que se fusiona por **rango** y el problema desaparece. Corrimos los tres
brazos. **El orden medido es exactamente el que el argumento predijo:** la única estrategia que
**suma dos escalas** queda en el medio, y la que **ignora el acuerdo entre ramas** es la peor.
⚠️ **Y una nota de honestidad sobre las propias cifras:** el **0,727** lo emite el artefacto de la
tabla; los otros dos **están medidos y sus corridas están en disco, pero la comparación nunca se
escribió a un artefacto**. Son medidas, no emitidas, y lo decimos distinto.
🔗 `tabla-a2-m13.json` (0,727) · `config/default.toml [fusion]` (los tres)

---

## Bloque 3 — 🔑 El resultado · 7,5 min

### D-8 · La tabla: v1 → v2 · 2 min

**Se muestra:** las 5 filas × 7 columnas sobre el consenso completo. Resaltar la columna
`P@5 estricta` y la fila `C4`.

**Se dice:** sobre los 379 pares anotados del pool congelado, 30 consultas × 5 obras por fila.
**`nDCG@5` va 0,372 → 0,743: se duplica.** Y en la vara dura —`P@5` **estricta**, la fracción del
top-5 con **grado 2**— va **0,200 → 0,560**, que es **2,8×**. **`QS` 0,419 → 0,900**: el sistema
pasa de cambiar poco el resultado según el pedido a cambiarlo casi todo **sin perder precisión**,
que es la **única** combinación que nuestro plan de métricas declara buena — `QS` alta sola no
significa nada, un sistema **aleatorio** tiene `QS ≈ 1`.
**Y el costo de v2 está medido y no es chico:** p95 **15,3 s → 65,3 s** y tokens de entrada por
consulta **3.722 → 59.395**, porque `C4` paga **20 llamadas de reranking con imagen** por consulta.
**La columna de dólares dice *ausente*, no cero** — y eso es una decisión, no un olvido: D-15.
🔗 `tabla-a2-m13.json` → `tabla_a_completa`, `columnas_sin_ground_truth`

### D-9 · 🔴 El orden de la tabla **CAMBIA**, y publicamos los dos · 2 min

**Se muestra:** las dos secuencias, una debajo de la otra:
**(a) `C4` > `C3` > `C2` > `C0` > `C1`** · **(b) `C4` > `C2` > `C3` > `C1` > `C0`**, con las dos
permutas marcadas.

**Se dice:** los 379 pares del consenso son **246 de acuerdo entre los dos anotadores + 133 que
resolvió un tercero**. Si restringimos a los 246, **el orden cambia: dos permutas**, y 🔴 **el
`C3 − C2` global cambia de signo, de +0,037 a −0,022.**
**No elegimos una.** Entre el **34,7 %** y el **40,7 %** del top-5 de cada fila **depende del
desempate**, así que una sola tabla sería una tabla con una incertidumbre escondida adentro.
**Qué sobrevive:** ✅ **`C4` primero en las dos poblaciones**, por +0,208 sobre los 379 de (a) y
por +0,149 sobre los 246 de (b), en cada caso sobre el segundo.
**El titular v1 → v2 se sostiene.** ✅ Y el desglose por categoría sobrevive — la próxima.
🔴 **Qué no podemos decir nunca más:** *"`C3` le gana a `C2`"*.
**Y el orden cambia igual con las dos políticas** para el ítem excluido, así que **la conclusión no
depende de esa elección**.
🔗 `tabla-a2-m13.json` → `sensibilidad_td_1`, `tabla_b_restringida_condensada`

### D-10 · 🔑 El titular: el delta por categoría de pedido confirmado con `nDCG`, y el arco se cierra · 2,5 min

**Se muestra:** cuatro números grandes, la tabla de dos filas × dos poblaciones —**(a) los 379
pares del consenso completo** y **(b) los 246 de acuerdo**, en ese orden—:
`conceptual` **+0,262 / +0,161** · `cromático` **−0,089 / −0,211**. Al pie, la línea de tiempo:
**18/10 figura UMAP → predicción escrita → 1/11 confirmada con nDCG**.

**Se dice:** esta es la diapositiva del trabajo. Nuestro plan de evaluación hipotetizaba, **desde el
diseño del set de consultas**, que *la rama de texto aporta casi todo en los pedidos conceptuales y
casi nada en los cromáticos*. Lo dejamos como **objetivo y no como criterio** a propósito: **que no
aportara habría sido un resultado válido**, y convertirlo en criterio nos habría dado incentivo a
esconderlo.
**Aportó, con el signo correcto, y en las dos poblaciones:** `C3 − C2` da **+0,262** sobre los 379
y **+0,161** sobre los 246 en los conceptuales, y **−0,089** y **−0,211** en los cromáticos, en ese
mismo orden de poblaciones. **Positivo y grande donde se predijo, negativo donde se predijo, en los
dos ground truths.**
🔑 **Y esto es lo que cierra el arco de D-5:** la figura UMAP midió el mecanismo —el formato separa
el espacio CLIP más que la cultura— y de ahí **predijimos exactamente este delta, dos semanas antes
de que la métrica existiera**. No es una interpretación posterior: **está fechada en los dos
extremos**.
🔑 **Y por eso el titular es éste y no el orden global:** el agregado se da vuelta entre las dos
poblaciones; **el desglose no**. Titulamos con lo robusto.
📌 Y el reparto de los 30 pedidos no era arbitrario: **cinco culturales que cruzan formato físico y
cinco que se quedan adentro**. Sin esa mezcla, el delta sería atribuible al formato y no a la
cultura.
🔗 `tabla-a2-m13.json` → `por_categoria_completa` / `por_categoria_restringida`

### D-11 · El pedido conceptual era el peor para v1 y es el mejor para v2 · 1 min

**Se muestra:** la matriz 5 × 3 por categoría sobre el consenso completo, con **`C1` conceptual
0,318** y **`C4` conceptual 0,961** resaltados en los dos extremos.

**Se dice:** si hay una sola tabla para llevarse, es ésta. **`C4` en los conceptuales llega a
0,961**, el número más alto de todo el estudio, y **`C4 − C1` en esa categoría es +0,643**. La
categoría que v1 resolvía peor es la que v2 resuelve mejor, **y sabemos por qué** (D-5).
**Como evidencia de mecanismo y no solo de efecto:** la cobertura cultural del top-5 sobre los 10
pedidos culturales va **0,60 → 2,60 → 4,50 de 5** en `C2 → C3 → C4`. ⚠️ **Con una salvedad que
decimos nosotros:** un reconteo a mano nos dio **4,40** en lugar de 4,50 para `C4` — **una obra en
50**. Publicamos la que emite el instrumento, **con la discrepancia al lado**.

---

## Bloque 4 — 🔴 Cómo lo medimos, y qué no podemos afirmar · 7 min

> 🔑 **Nota para quien presente: esto no es el apéndice, es el bloque que más puntúa.** La rúbrica
> tiene **Métricas** y **Evaluación** como **dos** conceptos. Y si una de las dos preguntas del
> oral va a ser *"¿y cómo saben que esto es verdad?"*, conviene haberla contestado antes.
> **No apurar este bloque para llegar al final.**

### D-12 · 🔴 Quién anotó, y qué deja de ser afirmable · 2,5 min

**Se muestra:** tres líneas: **«ningún humano anotó»** · **κ = 0,435** (autoritativo) ·
**«la mitigación 1 queda formalmente rota»**.

**Se dice:** el ground truth son **379 juicios × 2 anotadores**, y los produjo **IA**: el grupo
decidió que **ningún humano anota**. Las consecuencias las declaramos nosotros, no esperamos la
pregunta.
**(1)** La mitigación de circularidad que decía *"la tabla C0–C4 se calcula únicamente con
anotación humana"* **queda formalmente rota**. Lo que la reemplaza es **separar proveedores**, y
🔴 **la configuración alcanzable es asimétrica**: un anotador es independiente del proveedor del
pipeline y **el otro comparte proveedor con él** — porque el proxy de la cátedra sirve **solo**
modelos de OpenAI. **Lo que eso compra** es que el acuerdo cruza **al menos una frontera de
proveedor**; **lo que no compra** es un ground truth independiente del proveedor evaluado.
**(2)** El κ pasa de **inter-anotador a inter-modelo**, y se publica con esa etiqueta.
**(3) El κ es 0,475, por debajo del 0,6**, y lo reportamos **estratificado**, porque descubrimos que
**la rúbrica se pre-juzga a sí misma**: cruzando los 30 pedidos contra su propio texto, **8 aparecen
textuales** —y en uno el ejemplo **no es análogo, es el caso mismo**— contra una regla que la
rúbrica se había puesto ella sola. Son **110 de 379 ítems, el 29 %**. **El estrato contaminado da
+0,135 más alto**, lo que confirma la contaminación **con una medición y no con un argumento**.
**El κ autoritativo es 0,435, sobre 269 ítems y 22 consultas.**
**(4) Y el resultado no es el κ agregado: es el κ por categoría** — conceptual **0,810**, visual
**0,249**, ambiguo **0,228**. Un **factor 3,5**, y la causa está medida: el piloto fueron **20
juicios sobre 4 consultas** y las reglas se afinaron sobre lo que ahí apareció. Para los otros dos
tipos la rúbrica fija **el criterio y no el umbral**: un anotador declaró **siete umbrales medidos**
en CIELAB y el otro declaró **ninguno**. Se ve en la distribución: **153 unos contra 104**.
**Lo que no podemos afirmar, y lo decimos:** acuerdo inter-humano (no existe y no va a existir), que
el ground truth sea independiente del sistema evaluado, y que el κ valide al juez LLM.
🔗 `artifacts/ground-truth/rubricas/v1-decision-de-a5.toml` (la limitación, escrita y decidida)

### D-13 · 🔴 Un cuarto modelo puso a prueba nuestro mejor hallazgo, y lo acotó · 2,5 min

**Se muestra:** dos filas enfrentadas.
**Juez:** ambiguo **0,505** · visual 0,450 · conceptual **0,312**.
**Humanos:** conceptual **0,810** · visual 0,249 · ambiguo 0,228.
Una flecha: **se invierte**.

**Se dice:** de D-12 habíamos concluido algo elegante: *"la rúbrica no es ambigua — es precisa
exactamente donde se la refinó, y está ausente donde no"*. **Era la interpretación de la que más
seguros estábamos, y la pusimos a prueba:** un **cuarto modelo**, de otra familia que el pipeline,
leyó **la misma rúbrica** sobre el mismo pool. Si el patrón por categoría se reproducía, el
argumento se sostenía; **si se invertía, lo que el 0,810 medía no era la precisión del
instrumento.** **Se invirtió.** Donde los dos anotadores **más** acordaron, el juez **menos**
acuerda.
🔴 **Así que la frase se publica ACOTADA**, y la acotamos nosotros: el 0,810 parece medir algo
propio de esos dos anotadores, o del estrato pre-juzgado, **más que la precisión de la rúbrica**.
No queda refutado —son poblaciones distintas— **pero ahora tiene dos mediciones y no una, y la
discrepancia *es* el dato**: *el acuerdo alto en los culturales no viaja a un lector nuevo.*
🔑 **Y el mismo experimento nos dio un resultado al revés de lo que esperábamos, que es mejor.**
Diseñamos todo esto para medir el **efecto proveedor**. Medido: **κ(juez, anotador de la otra
familia) = 0,399** contra **κ(juez, anotador de su propia familia) = 0,252**. **El juez de OpenAI
concuerda MÁS con el anotador que no es de OpenAI.** Entonces el sesgo del desempate que vimos en
D-9 **no se explica por proveedor compartido**: es que **las lecturas de un anotador son las que un
lector fresco de la rúbrica reproduce**.
✅ **Y la circularidad quedó demostrada en vez de asumida.** El miedo escrito era *κ alto en `C0` y
bajo en `C4`*, o sea un juez que infla la configuración que comparte modelo con él. Medido,
estratificado por configuración de origen: **0,362 – 0,468, con `C4` en el medio-alto. No hay
favoritismo hacia `C4`.** Y el juez es **más duro** que el consenso, no más blando —**44 pares más
bajo contra 21 más alto**— y **evita el medio**: usa el grado 1 en **7,3 %** de los casos contra
**25,1 %**. **El desacuerdo dominante es *el juez no usa el medio*, no *el juez infla `C4`*.**
⚠️ **Y el κ del juez está sobre 179 de 379 pares**, porque se nos agotó el presupuesto — D-15.
🔗 `artifacts/informes/kappa-del-juez-m13.json`

### D-14 · 🔑 Qué nos salió mal, y cómo lo supimos · 2 min

**Se muestra:** cuatro ejemplos reales, con la cifra publicada y la correcta al lado. Al pie, una
línea sola: **«ninguna la encontró releer el párrafo. Todas, abrir el archivo.»**

**Se dice:** esta diapositiva no la pide el enunciado y la ponemos igual, porque es la parte del
trabajo que no se ve en una tabla.
**Nuestro modo de falla dominante tiene nombre y lo contamos:** *publicar una cifra correcta sobre
una población que no es la que la decisión necesita.* Nos pasó **repetidamente**, siempre en la
misma dirección —**la cómoda**— y **siempre lo encontró alguien abriendo el archivo del que salía la
cifra, nunca releyendo el párrafo que la contenía.** Tres ejemplos:
- *"Cobertura del 59,8 %"* era **del universo**; sobre el corpus es **84,8 %**. Faltaba la población.
- *"El pool son ~540 juicios"* eran **379**, y ese no fue un error: **fue un resultado** — las cinco
  configuraciones **se solapan el 49 %**, mucho más de lo que el diseño supuso.
- El piso de ruido que citábamos para `C4` está medido **en otro `config_id` con la misma
  etiqueta**. *La etiqueta describe; el `config_id` identifica.*
- 🔴 **Y la última es de anteayer y es la más pura:** habíamos *"verificado"* que **0 de 148**
  explicaciones mencionan la imagen que sube el usuario, y de ese `0` concluimos que la guardia de
  privacidad del contrato era **preventiva**. **Son tres consultas** — y una de ellas presenta la
  comparación **como un dato respaldado por el museo**. 🔑 **La causa no es descuido: el `0` salió de
  buscar una frase** (*"menciona la imagen"*) **y el texto usa otras tres** —*en comparación con*,
  *similar a*, *el mismo que*—. **Un conteo por patrón no es un conteo por propiedad**, y lo
  encontró **leer los 148 textos de a uno**, no un `grep` mejor.
🔑 **Y la lección que se asentó no fue *"releer con más cuidado"*, que no funciona. Fue: *la cifra
la emite el pipeline, no la transcribe una persona.*** Por eso el informe de hoy lleva **la
población y el artefacto al lado de cada número**, y por eso cuando una cifra **no** sale de un
artefacto —como dos de las tres de D-7— **está dicho**.
⚠️ **Y la última corrección fue al revés, que es la que más nos gusta:** habíamos escrito que el
presupuesto agotado **bloqueaba la entrega**, y es falso — ningún entregable llama al proxy.
**Sobrestimar un riesgo distorsiona la prioridad igual que sobrestimar un avance, y es más difícil
de ver porque suena prudente.**

---

## Bloque 5 — Producción y Responsible AI · 5,5 min

### D-15 · 🔴 El tope que nos agotamos, y por qué es un resultado y no una excusa · 2 min

**Se muestra:** `400 · budget_exceeded · Current cost: 10.0654 · Max budget: 10.0`, y debajo el
desglose por rol.

**Se dice:** el acceso no es a OpenAI directo: es un **proxy LiteLLM de ITBA** con una **virtual
key por grupo** y un tope de **USD 10**. **Lo agotamos**, y la causa está medida y es más
interesante que el número: 🔑 **es un defecto del modelo de costo, no del gasto.** Costeábamos a
**USD 0,0010 por request** —la cifra que mostraba el dashboard— y **el proxy factura por token**.
Nuestro rol más caro es el juez, que manda **la rúbrica entera —26.345 caracteres— más dos imágenes
en cada llamada**: ~10.000 tokens de entrada, que a precio de lista son **~USD 4,4 solo de
entrada**. **No nos pasamos de llamadas —391 contra un presupuesto de ~400—: nos pasamos de
dólares.** *El precio por request no es el precio.*
**El gasto está medido y desglosado por rol: ~7.313 requests** — reranker 3.920 · embedding de
texto 1.896 · generador 1.043 · juez ~391 · planificador 63 — contadas del libro mayor de cada
artefacto y publicadas como **cota inferior**, porque la verdad de facturación la dice el dashboard
y **las dos no se promedian**.
**Y por eso la columna de dólares de D-8 dice *ausente*:** la configuración declara el precio de un
solo modelo, y **un modelo sin precio da ausente, nunca cero**. Tampoco lo tomamos del dashboard de
trazas, que lo calcularía con **su propia** tabla de precios: serían **dos cifras del mismo número,
y la segunda se actualiza sola**.
**La lección de producción, que es la que vale fuera de este TP:** en un ambiente compartido con
tope duro, el presupuesto se modela **por token y por rol**, y el rol que repite el prompt largo
**N veces por consulta** es el que se lo come. **Decidimos no pedir que nos suban el tope:** lo
único que impide es mejorar la `n` del juez, y **no habilita ningún criterio**.

### D-16 · Responsible AI: tres cosas medidas, una de ellas negativa · 2 min

**Se muestra:** tres bloques: **21,2 %** · **39 % → 0** · **suma cero**.

**Se dice:** tres resultados, y los tres son nuestros.
**(1) El chunking tiene un costo de licencia, y es del chunking que la entrega aprobada pedía.**
Nos apartamos de los *"500 tokens con solapamiento"* y fragmentamos **por campo semántico**;
construimos el otro brazo igual y lo corrimos. **El chunking ciego fuerza atribución CC-BY sobre
texto CC0 en el 21,2 % de las citas que publica — 22 de 104** — porque en AIC la `description` es
**CC BY 4.0** y el resto es CC0, y un chunk de 500 tokens los concatena. **El brazo por campo no
puede producir ni una**: un fragmento de un campo tiene **una** licencia por construcción.
⚠️ **Y ese cero NO es una medición, es una propiedad estructural** — decirlo importa, porque una
cifra que sale de la definición no es evidencia del mismo tipo que una medida. Y una cita sin
atribución **es una infracción, no un defecto de calidad**: el renderizador **falla en vez de
degradar**.
**(2) Un resultado negativo: `provenance` es el 39 % del índice congelado y NUNCA se cita.** 2.408
de 6.232 fragmentos, y en los dos brazos: **128 ofrecidos al generador → 0 citados**, y 29 → 0. Lo
habíamos anticipado por argumento; la medición dice **que tampoco se cita**. Es lo primero que hay
que saber antes de pagar una estructura de búsqueda sobre el 39 % del índice.
**(3) Y el presupuesto de evidencia es de suma cero.** El tope es **3 fragmentos por obra**, así que
agregar campos **no agrega evidencia: la desplaza** —`description` 142 → 136 obras, `provenance` 128
→ 29— **y la abstención sube un 25 %**. Nuestro propio plan preguntaba *"¿los campos ayudan o meten
ruido?"* y la respuesta medida **es una tercera: desplazan.** La pregunta correcta no era *"¿qué
agregamos?"* sino **"¿qué sacamos para meterlo?"**.
**Y una que es verificable en vez de prometida: del usuario no se persiste nada.** En los planes y
las corridas en disco **la imagen no se retiene nunca** —solo su hash—. El reverso son **tres
destinos externos, no uno**, y están declarados. ⚠️ **Y lo volvimos a verificar para esta entrega,
con una salvedad que decimos nosotros: se cumple, y el mecanismo igual es frágil.** La política se
implementa **vaciando campos de la entrada** y **nada depura el texto que el modelo genera**, que
viaja en el mismo artefacto. Para nuestras 1.151 corridas da igual —son del **set de evaluación**,
que la decisión excepciona explícitamente porque es el instrumento de medición, y las **2 ad-hoc
tienen la entrada vacía**—; **para una consulta real de un usuario, el texto generado podría
describir su imagen y el vaciado no lo atraparía.** Y la próxima diapositiva mide que **el generador
ya lo hace**.

### D-16b · 🔴 La regla de honestidad: medida, y al contrato le falta un tipo · 1,5 min

**Se muestra:** tres pares de números grandes — **39 / 51** · **20 / 27** · **12 / 16** — y debajo
una sola línea: **`dato del museo` · `observación del modelo` · ¿y lo que el modelo *sabe*?**

**Se dice:** la diapositiva D-1 prometió que cada explicación distingue **lo que el museo dice**
—con su cita— de **lo que el modelo ve** —marcado como propio—. **Eso no se promete: se mide**, y lo
medimos **afirmación por afirmación** sobre **25 explicaciones = 5 consultas × 5 obras**, el tamaño
que nuestro propio plan fijó y **no movimos**, con las tres categorías representadas. **51
afirmaciones**, y los conteos van **separados porque son tres fallas distintas**: la clasificación
da **39 bien / 12 mal** sobre las 51; la cita **se sostiene contra su fragmento en 20 de las 27
afirmaciones citadas** (4 parciales, **3 que no**); y la observación **es defendible mirando la
imagen en 12 de las 16 que hacen una aserción visual** (2 discutibles, **2 falsas a la vista**).
🔑 **El hallazgo no es el porcentaje: nuestro contrato tiene DOS tipos y el generador produce
TRES clases.** 5 de las 24 observaciones no son observaciones de la imagen ni datos del museo: son
**conocimiento general del modelo** —*"el dragón es un motivo clásico en la pintura china"*—. Y como
la regla de forma es **`dato ⟺ tiene cita`**, lo que no tiene cita **cae en `observación`**, que le
dice al usuario *"el modelo miró y vio esto"*. **Nuestra propia métrica de separación corta binario
sobre una población de tres clases.** No lo arreglamos ahora —movería el prompt, el `config_id` y
las 1.151 corridas—: **lo declaramos**, que es lo que hicimos con `provenance`.
🔴 **Y un caso que falla los tres ejes a la vez, donde la falla no fue cosmética.** `aic:43211`,
*Interior of the Oude Kerk*: un **`[dato]` citado** afirma **"tonos cálidos"** y el fragmento **no
menciona tono en ningún lado**; dos renglones abajo, la **`[observación]`** dice **"paleta cálida,
marrones"** sobre un interior de **piedra gris-blanca**. El pedido era **"tonos fríos"**, ésa era
**la obra más fría de su propio top-5**, y **el sistema se abstuvo sobre ella apoyado en la
observación falsa.** **La falla de honestidad le costó al sistema la mejor obra que había
recuperado.**
🔑 **Y las cuatro fallas de observación son 4 de 4 sobre paleta y las 4 en `visual_cromatico`** — la
misma categoría del **κ = 0,249** de D-12, ahora señalada por **un tercer instrumento que no toca a
los otros dos**: *la categoría que depende de mirar es exactamente donde el sistema no mira.*
⚠️ **Dos honestidades sobre nuestras propias cifras, y las dos van dichas.** **(1)** De las 6 fallas
de tipo, **2 están forzadas por el contrato**: la ficha del museo **no es citable**, así que cultura,
nacionalidad y año **no pueden** ir como `dato` aunque los publique el museo. **Decimos *«6, de las
cuales 2 forzadas y 4 no»*, nunca la cifra sola** — en esa fracción la métrica mide **nuestro
esquema de evidencia, no al modelo**. **(2)** Lo anotó **IA**, como el resto del ground truth, y
nuestro criterio decía *"muestra humana"*: **la letra falla en quién anota**, igual que en D-12.
🔗 `artifacts/ground-truth/revision-explicaciones/m13b-c1-5x5.json` · `ejercicio-4.md §5.5`

---

## Bloque 6 — Cierre · 1 min

### D-17 · Conclusiones · 1 min

**Se muestra:** seis líneas y el hash.

**Se dice:**
1. **La iteración funcionó:** sobre los **379 pares** del consenso completo, `nDCG@5` **0,372 →
   0,743** y `P@5` estricta **0,200 → 0,560**, con `C4` primero **en las dos poblaciones**.
2. 🔑 **El resultado más fuerte no es el agregado: es el desglose por categoría** — la rama de
   texto aporta **+0,262** sobre esos 379 pares y **+0,161** sobre los **246 de acuerdo** en los
   conceptuales, y **resta** en los cromáticos, **con el signo correcto en los dos ground truths**.
   Y el mecanismo lo habíamos medido, y la predicción escrita, **dos semanas antes** de tener la
   métrica.
3. **La premisa es falsable y la falsamos a propósito:** `QS(C0) = 0,000` exacto, `QS(C4) = 0,900`.
4. 🔴 **Y lo que no podemos afirmar está en el cuerpo del informe, no al pie:** el ground truth lo
   produjo IA, el κ es inter-modelo y vale **0,435**, y el hallazgo del que más seguros estábamos
   **lo acotamos nosotros** cuando un cuarto modelo no lo reprodujo.
5. 🔑 **Y la regla de honestidad tiene número, no promesa:** **39/51** · **20/27** · **12/16**, y el
   hallazgo es que **nuestro contrato tiene dos tipos y el generador produce tres clases**.
6. **Trabajo futuro, en el orden en que cambiaría el resultado:** un **umbral numérico** para los
   pedidos cromáticos —es la causa medida del κ = 0,249, y tres instrumentos la señalan—, reemplazar
   el **cuarto escalón** de la escalera de pedidos ambiguos, que está **declarado roto**, un
   **filtro de formato físico** en la rama visual, que es el único mecanismo causal que medimos, y
   **un tercer tipo de afirmación** con la guardia de privacidad **como validador que corre** — el
   único que no cambia una métrica: **cambia una promesa**.

**Repositorio — ‹HASH DEL COMMIT: PENDIENTE›**
⚠️ *Lugar reservado a propósito: la exportación a la rama `main` es otra pasada. **Quien arme la
maqueta tiene que pegar el hash acá antes de subir**, y es lo último que se hace.*

---
---

## §Z — Qué necesita quien arme la maqueta

### Figuras: las seis ya están en disco

| Diapositiva | Archivo |
|---|---|
| **D-5** (la que más importa) | `artifacts/informes/umap-05-separacion.png` |
| D-5, alternativa | `umap-02-culture.png` (el espacio coloreado por `culture`) |
| opcional, bloque 1 | `umap-01-tema.png` — los clusters por **tema**, que es el objetivo opcional de la figura UMAP cumplido |
| opcional, D-6 | `umap-06-hallazgos-no-pedidos.png` |
| **D-1** | hay que **armarla**: 1 imagen del held-out + los 3 top-5. Salen de las corridas de `e30ca8c5003c614c` (30 en disco). **Es la única figura que falta y es la primera que se ve** |
| **D-14** | tabla de **cuatro** filas, se tipea |
| **D-16b** | sin figura: **tres pares de números** y una línea. Se tipea. ⚠️ Si quien presente quiere **una** imagen, va `aic:43211` —*Interior of the Oude Kerk*— al lado de las dos afirmaciones: **el gris-blanco de la piedra contra el texto que dice «cálida» es el argumento entero** |

### Tablas: se copian de `ejercicio-4.md`, sin recalcular

| Diapositiva | De dónde |
|---|---|
| D-8 | `ejercicio-4.md §4.1` entera |
| D-9 | `§4.2`, las dos secuencias + la columna `% sin juzgar` |
| D-10 | `§4.3`, la tabla de `C3 − C2` |
| D-11 | `§4.3`, la matriz 5 × 3 de la población (a) |
| D-12 | `§6.2`, los dos estratos y el κ por categoría |
| D-13 | `§6.3` y `§6.4` |
| D-15 | `§5.4`, el desglose por rol |
| **D-16b** | `§5.5`, la tabla de los tres ejes y el caso de `aic:43211`. 🔴 **Los tres pares de números van juntos y sin promediar**, y el **6** de las fallas de tipo **no va sin su *«2 forzadas / 4 no»*** |

🔴 **Regla: no se recalcula ninguna cifra para la diapositiva.** Si una no está en
`ejercicio-4.md`, **no va** — es el modo de falla de D-14 y sería desafortunado cometerlo en la
diapositiva que lo cuenta.

### Preguntas que conviene tener contestadas, con dónde está la respuesta

| Pregunta probable | Respuesta corta | Dónde |
|---|---|---|
| *"¿Y cómo saben que el juez automático acierta?"* | **No lo usamos para sostener v1 → v2.** La tabla se calcula sin él; el juez es una **medición sobre el instrumento**. Y su κ es 0,414 < 0,6, reportado | `ejercicio-4.md §6.1`, `§6.3` |
| *"¿Por qué no usan un κ ponderado, si la escala es ordinal?"* | **Lo medimos: no rescata nada.** 0,437 lineal y 0,460 cuadrático contra 0,419, sobre el piloto. Sube el acuerdo observado y sube el esperado por azar casi igual | anexo `§B` |
| *"¿Por qué 30 consultas?"* | Son **10 por celda** en el desglose. Los tamaños son **presupuesto, no potencia estadística**, y así lo decimos: ±0,05 no se distingue del ruido | `§6.6` |
| 🔑 *"¿Y por qué `α = 0,5` y no el que maximiza `QS`?"* | **Porque `QS` es monótona decreciente en `α`:** maximizarla sola empuja a **`α = 0`**, que da la `QS` más alta de las cinco (**0,821**) **con anclaje 0/20 y las 20 obras devolviendo la misma**. **Es `C0` por el otro lado** — `C0` ignora el pedido, `α = 0` ignora la imagen. El criterio —*la `QS` más alta entre los `α` que conservan el anclaje top-1 20/20*— **se declaró antes de mirar el barrido**, y con el umbral en top-5 habría ganado `α = 0,25` por **+0,026 de `QS`**. **No se movió: eso sería elegir la vara después de ver a qué `α` favorece.** Es la mejor respuesta de oral que tenemos para *"`QS` alta sola no significa nada"* | `§2.2` |
| *"¿Por qué sacaron `style` del output?"* | CMA no lo publica y en AIC está en el 78 %: existiría en **1 de cada 4** obras. **Y medimos que no es un campo de estilo consistente**: lo que separa el espacio son `Himalayan` 0,86 y `Japanese` 0,71, que **no son estilos**; `Renaissance` 0,24 no separa nada. **Es un desvío no consultado** | anexo `§B` |
| *"¿El corpus no eran 10.000 obras?"* | **10.000 prometidas → ~5.900 al consultar → 2.605 hoy**, y el motivo de cada salto está medido. El recorte **de tipo** está aceptado; **la magnitud la informamos nosotros** | encuadre del informe |
| *"¿Por qué no filtran por metadata en los pedidos culturales?"* | **Porque de eso se encarga la rama de texto, y está medido**: cobertura cultural 0,60 → 2,60 → 4,50 de 5 | D-6, D-11 |
| 🔑 *"¿Cómo comparan un resultado de texto con uno de imagen?"* | **No los comparamos: lo único que rankeamos son obras**, y cada obra tiene imagen y texto. Las dos ramas producen **rankings del mismo universo** y se fusionan con **RRF, que usa solo rangos** — justamente porque un coseno de CLIP y uno de `text-embedding-3-small` no son calibrables. Donde **sí** mezclamos modalidades es adentro de CLIP (`α·e_img + (1−α)·e_txt`), y ahí se normaliza antes y después. ⚠️ **La alternativa en cascada la evaluamos y perdería**: la intersección de las dos ramas es **3,97 de 65,1**, en **19 de 30 consultas no junta 5 candidatas**, y dejaría afuera el **23,3 %** del top-5 de v2 | `§2.3` |
| *"¿Cuánto costó?"* | **Tokens y llamadas, medidos y por rol. Dólares: ausente**, porque el precio del modelo de chat no está declarado y el proxy no deja leer tarifas | D-15 |
| 🔑 *"¿Y cómo saben que las explicaciones no mienten?"* | **Dos mitades, y no se promedian.** La automática: **0 referencias colgadas** y **100 % de cita byte a byte** en las 150 obras de cada fila. La fina: **25 explicaciones y 51 afirmaciones revisadas una por una** — clasificación **39/51**, la cita se sostiene **20/27**, la observación es defendible **12/16**. **Y el resultado que importa es cualitativo:** la regla de que un dato lleva referencia tiene dos tipos y el generador produce **tres clases**. 🔴 **Y de las 7 afirmaciones que no llegan a *sostiene*, 2 son defecto del DATO y no del modelo: el 14,2 % del corpus (370 de 2.605 obras) comparte su `description` con otra**, porque el museo publica una descripción por álbum y la cuelga de cada hoja | D-16b, `ejercicio-4.md §5.5` |
| *"¿Y las 25 las anotó una persona?"* | **No: IA, igual que los 379 juicios.** Nuestro criterio decía *"muestra humana"*, así que **la letra no se cumple** y lo decimos. Lo que sí tiene distinto del κ: acá el juicio es **contra un fragmento en disco y una imagen en disco**, no contra una convención de la rúbrica | D-16b, `§F` del anexo |
| 🔑 *"Y el chunking ciego, ¿además **recupera** peor?"* | 🔴 **Lo medimos y la respuesta es un NO-resultado, que es lo que hay que contestar.** Extendimos el ground truth por **pooling incremental** (31 juicios nuevos) y el **`Δ` contra `C3` cambia de signo** según la población: **−0,0405** sobre el consenso completo y **+0,0035 / +0,0050** sobre el restringido. **Así que no podemos afirmar que empeore el retrieval, ni que no lo empeore.** 🔑 **Y el confundidor vale más que la tabla:** los 31 juicios nuevos **no pasaron por el filtro de acuerdo/desempate** —no hay con quién acordar cuando el anotador es uno— así que **sobreviven a la restricción por construcción**, y la fila restringida **favorece estructuralmente al brazo que trajo ítems nuevos**. ⚠️ **Lo que sí está medido es que el eje mueve el top-5: churn 1,567 de 5 contra 0,267 del eje de campos, 5,9×.** Y una decisión de método que sale de ahí: **el `IDCG` se computa sobre la población del ground truth, así que extender el pool mueve el baseline** — `C3` recomputada da **0,5308** contra el **0,535** de D-8, **y las dos son correctas con bases distintas** | `ejercicio-4.md §5.3b` |

### Ensayo: dónde se va a ir de tiempo, y qué se corta

El bloque 3 (7,5 min) es el que se estira, porque D-10 invita a explicar el mecanismo dos veces —una
en D-5 y otra ahí—. **Si hay que recortar, el orden es:** (1) D-16 pierde el tercer bloque (suma
cero), que es el más difícil de contar en 30 segundos; (2) D-2 se funde en D-1; (3) D-11 se muestra
sin leerla; (4) D-16b se queda **solo con los tres pares de números y el caso de `aic:43211`**, y
suelta el desglose de las 6 fallas de tipo.
⚠️ **Y una nota de presupuesto, porque este archivo ya se recortó una vez:** D-4, D-6 y D-11 bajaron
**medio minuto cada uno** para que entrara D-16b. **El criterio fue el de abajo**, aplicado a
nosotros mismos.
🔴 **Lo que no se recorta es el bloque 4**, ni D-14. Si el tiempo aprieta, **se recorta
arquitectura, no evaluación** — la arquitectura está en el informe y el oral no la mejora; la
honestidad metodológica solo existe si se dice en voz alta.
