# Ejercicio 3 — Responsible AI, Safety y puesta en producción

**Grupo 2 · 73.69 Large Language Models · ITBA, 2026**

El sistema recibe una imagen y un pedido en lenguaje natural y devuelve cinco pinturas reales del
Art Institute of Chicago (AIC) y del Cleveland Museum of Art (CMA), con una explicación citada a
fragmentos de texto de los museos. Este informe no enumera riesgos genéricos: reporta los que
**medimos sobre nuestro propio corpus**, y separa lo construido de la propuesta. **Poblaciones**,
porque las cifras no son intercambiables: universo elegible **5.915** pinturas de licencia libre
(3.950 CMA + 1.965 AIC), universo con imagen **5.751**, **corpus indexado 2.605** en **6.232
fragmentos**, held-out **289**, **30 consultas** de evaluación.

## 1. Responsible AI y Safety

**El sesgo de colección está cuantificado y no es el europeo que se espera** —en las 5.915
elegibles `China` (785) le gana a `France` (716), y el masivo es otro: **el 53,9 % de las 3.950
pinturas de CMA son anónimas**, con **0 fallas del parser de autoría**—. **Pero el sesgo que
medimos mejor es el que introdujimos nosotros.** El criterio de inclusión —`description` no vacía,
porque sin texto no hay cita— **no es culturalmente neutro**: la tasa de ingreso al corpus va del
**0 %** al **83,6 %** según la etiqueta de `culture`, y **54 de las 156 etiquetas del universo
desaparecen**. **Está mitigado, y está medido cuánto:** la cobertura cultural del top-5, sobre los
**10 pedidos culturales** de las 30 consultas, sube **0,60 → 2,60 → 4,50 de 5** al sumar la rama
de texto y el reranking, y se publica con el `n` de la cultura pedida al lado, para que una falla
de cobertura no se lea como falla del buscador.

**La atribución es una obligación, y su costo ya no es una proyección.** Casi todo es CC0, pero
`description` de AIC es **CC-BY 4.0**: son **638 fragmentos, exactamente las 638 obras de AIC del
corpus**, porque esa `description` es el criterio de entrada. La licencia viaja con cada fragmento
y el renderizador **falla en vez de degradar**: una cita sin atribución es una infracción, no un
resultado pobre. Y el chunking ciego de 500 tokens de la entrega aprobada —nos apartamos de él
fragmentando por campo semántico— se construyó igual como brazo de comparación y se corrió:
**fuerza atribución CC-BY sobre texto CC0 en el 21,2 % de las citas que publica** (22 de 104,
sobre las 103 obras que los dos brazos citan en común). ⚠️ El **0** del brazo por campo **no es
una medición**: un fragmento de un solo campo tiene una licencia por construcción.

**Privacidad: la única respuesta de este informe que es verificable en vez de prometida, porque
fuimos a verificarla.** Del usuario no se persiste nada, y está contado: en los **62 planes y las
1.151 corridas en disco la imagen no se retiene nunca** —solo su hash—, y el pedido se guarda solo
cuando la consulta sale de nuestro set de evaluación, que es instrumento de medición y no dato de
un tercero. El reverso son **tres destinos externos, no uno**: la imagen y el pedido viajan al
**proxy LiteLLM de la cátedra** —que tiene del otro lado las credenciales de OpenAI— y las trazas
van a **LangSmith**. Ese tercero casi rompió la promesa: trazar con la configuración por defecto
devolvió `inputs` y `outputs` **completos** leídos del servidor. Por eso la política se partió, y
queda escrita en cada corrida: las **1.149 del set se trazan enteras**; las **2 ad-hoc** trazan
estructura, latencia y tokens, **sin capturar entradas y con 5 campos censurados**. **La forzó una
medición, no la documentación.**

**Y quién anotó va en el cuerpo, no en una nota al pie: el ground truth lo produce IA.** El grupo
decidió que **ningún humano anota** los **379 juicios** del pool congelado (30 consultas, 202
obras). Las consecuencias se declaran: la mitigación de circularidad que pedía *"la tabla C0–C4 se
calcula solo con juicios humanos"* **queda formalmente rota**; el κ de Cohen pasa de
**inter-anotador a inter-modelo**; y los dos anotadores son de **familias de modelo distintas**,
porque el pipeline usa un solo modelo en planificador, reranker y generador. **Lo que no se puede
afirmar, y por eso se dice, es que el ground truth sea independiente del sistema evaluado.** El
único κ que hay —**0,419**, sobre 20 juicios del piloto— son **dos pasadas de la misma sesión**: el
número es correcto; su lectura como acuerdo entre dos, no.

## 2. Estrategias de puesta en producción

El enunciado pide **definir**, y al 3/11 va separado: ✅ **construido, corrido y medido** la
indexación, el caché, el versionado, la observabilidad, la degradación y el **costo por token y
por rol** —en USD desde que se declaró el precio con su fecha—; ⬜ **definición pura** el escalado
a 100k y la evaluación continua. Nada de lo ⬜ se presenta como construido.

**La indexación es un batch offline y el serving solo lee**, lo que resuelve el rate limit de 60
req/min de AIC: nunca se llama a las APIs de los museos en tiempo de consulta. El costo offline se
paga una vez y está medido —**69 minutos** para las 5.915 obras, **0 fallas**— y **el caché no
conoce el corpus**, así que cambiar el criterio de inclusión no vuelve a pagar embeddings.
Catálogo, split e índice guardan adentro su configuración, y una variante **no se registra si no
pasa cuatro invariantes** —entre ellos que el held-out no entre—.

**Degradación elegante: corrida, no prometida.** Cada etapa posterior al retrieval es omitible —si
falla el reranking se entrega el ranking fusionado, si falla la rama de texto queda la visual, si
no alcanza la evidencia el sistema se abstiene— y se verificó inyectando la falla en **las seis
etapas con fallback sobre las 30 consultas: 301 corridas de inyección, cada etapa con ≥ 30**, y
la degradación queda **anotada en el artefacto de la corrida**. El piso no se exagera: el mínimo
irreducible es **un embedding CLIP más una consulta al índice** — pero **no tiene red**, así que un
corte de conectividad degrada el sistema y no lo apaga.

**Costo y observabilidad: medidos, cada cifra con su base.** El embebido del corpus se pagó
**622.314 tokens en 49 llamadas** para los 6.232 fragmentos (**0,012446 USD**), y una consulta del
sistema completo cuesta **0,0277 USD** contra **0,0029** del base: el `nDCG@5` se duplica y el
precio se multiplica por **diez**. 🔑 **El gasto total modelado por token y por rol da USD 12,43 y
el proxy facturó 10,0654; dos roles son el 76 %.** Los tokens son **medidos** y el USD **derivado**
de un precio de lista; **las dos cifras del total no se promedian**, cada una con su base. La observabilidad
**está construida**: las trazas van a LangSmith y los tokens se persisten **también** en el
artefacto de la corrida. ⚠️ **Y queda un riesgo que es nuestro declarar y no
nuestro arreglar:** el criterio *"toda obra devuelta existe y es trazable"* **depende de un tercero
al correr** —los mismos artefactos dieron **148/150** y después **150/150**, y un `source_url` de
AIC devolvió un lazo de **30 redirecciones**—. El entregable es un hash de commit: puede fallar el
día de la corrección por algo que no pasa en nuestro código.

*Fuentes: AIC y CMA, obras de dominio público / CC0. `description` de AIC es CC-BY 4.0; este
informe no reproduce fragmentos de esos textos. Las cifras las emitió nuestro pipeline entre el
14 y el 30 de octubre de 2026, y cada una lleva su población.* · **1.251 palabras · 94 líneas**

---
---

# Anexo — fuera de las dos hojas

> **No es parte del entregable.** Es lo que le permite al grupo cambiar la selección sin
> rehacer el informe. **La selección final es del grupo; acá se propone.**
>
> 📌 **Esta versión es del 2026-10-30 y actualiza la del 18/10.** El cuerpo no se reescribió: se
> editó. Lo que cambió, y por qué, está en la sección **G**.
>
> ⚠️ **Con una precisión posterior, del 2026-11-01, en el ítem 8 de la sección D** —y lleva su
> propia fecha adentro para que se sepa cuándo se midió qué—: **la deuda que ese ítem declaraba se
> pagó** (los 29 ítems sin juzgar del brazo ciego se juzgaron por pooling incremental), así que el
> motivo que daba quedó falso. **El veredicto no cambió; el motivo sí.** 🔑 **Es el modo de
> obsolescencia que nadie busca: un documento envejece porque se cerró la deuda que citaba como
> limitación.**
>
> ⚠️ **Y una segunda precisión del 2026-11-01, en la fila de *atribución errónea* de la sección B**,
> también con su fecha adentro: decía que el riesgo estaba *cubierto por construcción* y **está
> medido que no lo está** —el **14,2 %** del corpus comparte su `description` con otra obra—. **La
> fila no se movió: sigue siendo material que quedó afuera por espacio, y lo que cambió es el
> motivo.** 🔑 **Las dos precisiones son la misma lección: una limitación dicha *"por construcción"*
> no lleva fecha, y por eso nadie la vuelve a mirar.**
>
> El resto del anexo sigue siendo del 30/10.

## A. Qué se eligió y con qué criterio

El material disponible son 17 temas (9 de Responsible AI + 8 de Producción), más seis hitos
medidos entre el 24 y el 30/10. En dos carillas son ~130 palabras por tema: un listado. El
criterio fue, y sigue siendo, **un tema entra si lo podemos sostener con una cifra propia, medida
y con su artefacto**; si solo lo podemos afirmar, va al anexo. Quedaron **4 bloques** en la
sección 1 y **3 párrafos** en la sección 2, y la sección 1 se lleva el ~56 % de las palabras
porque es donde tenemos números que nadie espera.

**El criterio de desempate entre lo nuevo y lo viejo fue uno solo: *reemplazar una proyección por
una medición no cuesta líneas y gana verdad*.** Por eso entró primero lo que **sustituye** texto
existente —el costo del chunking ciego, el costo del embebido, la degradación corrida, la
observabilidad— y después lo que **agrega bloque**, que fue solo uno: quién anota.

⚠️ **Y la restricción mordió de verdad: cinco cosas buenas y medidas quedaron afuera** (las cinco
primeras filas de la sección B). La que más me costó dejar es el filtro de formato físico, porque
es el **único mecanismo causal** que el proyecto midió. El límite no son las palabras sino **las
líneas**: el cuerpo son 94 y a 48 por carilla eso ya es 1,96.

📌 **Un cambio de forma, declarado:** la tabla de estado ✅/🟡/⬜ del 18/10 pasó a prosa. Ocupaba
**6 líneas para 25 palabras** y la separación que el enunciado pide sigue entera, en la primera
oración de la sección 2.

## B. Qué quedó afuera, y por qué — ordenado por qué tan fácil es reponerlo

| Tema | Por qué quedó afuera | Si entra, qué sale |
|---|---|---|
| 🔑 **El filtro de soporte que nadie puso.** Sobre las 2.605 indexadas, **el formato físico separa el espacio de CLIP más que la cultura** —κ del vecino **0,66** (folio de manuscrito vs. pintura de caballete) contra **0,46** de `culture`—, así que la rama visual devuelve el soporte que se le subió **para cualquier pedido** | **Es lo mejor que dejé afuera y es una decisión de línea, no de mérito.** Es el único mecanismo *causal* medido del proyecto: explica **por qué** la rama de texto sube la cobertura cultural, en vez de solo reportar que sube. Son ~45 palabras y 2 líneas, y a 94 líneas no entraban | **Primer candidato a entrar.** Va en el primer bloque, justo antes de *"Está mitigado"*; sale la enumeración del párrafo de ingesta (69 minutos / 0 fallas) |
| 🔑 **`provenance` es el 39 % del índice congelado y NUNCA se cita** — 2.408 de 6.232 fragmentos, 128 ofrecidos → **0** citados en un brazo y 29 → **0** en el otro | Es un resultado **negativo** y por eso vale, y encaja en producción (antes de pagar ANN conviene saber que el 39 % del índice no se usa). Pero el registro del proyecto ya lo asignó **al informe del Ejercicio 4**, y duplicarlo acá gasta líneas en los dos | Una oración al final de la sección 2; sale la mención del escalado a 100k de la prosa de estado |
| **El vocabulario esconde el sesgo que queremos reportar** — la India partida en **once etiquetas** (pedir `India` sola da 67 obras donde hay 619), `made for Abd al-Rahim the Khan Khanan` como valor de `culture`, **102** etiquetas sobre el corpus y 11 obras sin ninguna | Era un bloque propio el 18/10 y se comprimió a su consecuencia operativa (*la cultura pedida es un conjunto declarado, publicado con su `n`*). ⚠️ **Y su mejor ejemplo también se perdió**: un pedido que declara `Mughal India` recibe obras con `India` y `Northwestern India`, y **la métrica cuenta 1 de 5 donde un lector ve tres obras de la India** — o sea que el vocabulario partido **deprime una cifra que publicamos** | Entra como bloque propio de ~90 palabras; sale el bloque de privacidad entero, que es el único de ese tamaño |
| **Procedencia conflictiva** — `is_nazi_era_provenance = true` en **267 de las 3.950 pinturas de CMA (6,8 %)** y **205 de las 2.605 del corpus (7,9 %)**, y el campo **no está en el esquema canónico**, así que mostrarlo es propuesta | Estaba en el cuerpo el 18/10 y salió: es el único tema de la sección 1 que **no podemos presentar como función**, y el espacio lo ganó la política de trazado, que sí está implementado y contado. Agregarlo al esquema cuesta una línea del normalizador y una re-normalización barata | ~35 palabras al final del bloque de privacidad; es el tema que este refresco desplazó, así que reponerlo es revertir una decisión, no agregar una |
| **Alucinación y sourcing** (la regla *dato con fuente vs. observación propia*, y que **se mide**) | Sigue sin número: la muestra de 25 explicaciones no se anotó, y con la anotación por IA la va a anotar IA, lo que cambia qué se puede afirmar. Lo único medido hoy es el reverso, y es bueno: **0 de 148 observaciones persistidas hablan de la imagen del usuario**, y el contrato ahora declara que una observación es *sobre una obra recuperada*, o sea que es **verificable por un validador** y no por confianza | **Entra en cuanto exista el número**, al final del bloque de licencias; se paga con la oración del piso irreducible de la sección 2 |
| **El presupuesto de evidencia es de suma cero** — `n_fragmentos_por_obra = 3`, así que agregar campos **no agrega evidencia: la desplaza** (`description` 142 → 136 obras, `provenance` 128 → 29) y la abstención sube **+25 %** (61 → 76 obras) | Es un resultado fuerte y es nuestro, pero **publicarlo honesto cuesta la salvedad**: en el eje del brazo ciego el desplazamiento **no ocurre** (el índice ofrece 2,39 y 1,07 fragmentos por obra, por debajo del tope de 3), así que vale **donde muerde**, que es la variante ampliada (4,90 > 3). Esa salvedad son ~45 palabras y el informe de Ej. 3 no tiene dónde ponerlas | Entra en el párrafo de escalado, desplazando la oración de `provenance`; es el **segundo candidato** |
| **"Abierto" no es abierto**: 71 obras de AIC marcadas `is_public_domain = true` (3,6 % de su catálogo de pinturas) tienen imagen con **403 persistente** | Medido y contraintuitivo, pero es el único hallazgo fuerte que **no cambia ninguna decisión nuestra**: esas obras simplemente no entran al corpus | Es **una oración** al final del párrafo de licencias; entra sin desplazar nada si sobra espacio |
| **Guardrail de *contenido fuera de contexto*** (Clase 8): el usuario puede subir cualquier imagen, no una obra | Es el guardrail más relevante del material de la cátedra para nuestro caso, pero **el comportamiento sigue sin decidir** y decirlo en dos líneas sin propuesta concreta no aporta | Dos oraciones al final del bloque de privacidad, si se recorta la escalera de la sección F |
| **El brazo ciego no cita menos obras: cita menos veces** — **103/103** obras con ≥ 1 referencia contra 102/103 del brazo por campo, y 1,126 → 1,010 referencias por obra | Es el matiz que vuelve refutable la lectura *"el chunking ciego degrada la citabilidad"*, y por eso **va al informe del Ej. 4**, que es donde se compara. Acá el titular es el de licencias | Una oración al final del párrafo de licencias |
| **SynthID / contenido generado por IA** (Clase 9) | Material del propio deck de la cátedra, y encaja en dos direcciones (entrada generada por IA, salida trazable). Pero **no lo medimos ni lo implementamos**: serían 100 palabras de buena intención | Entra solo si el grupo acepta cambiar una medición por un tema del deck; no hay nada barato que sacar |
| **La pregunta abierta de la Clase 7.2** (*"¿qué atributos de seguridad se deberían considerar en RAG?"*) | Aparece dos veces en el material y vale citarla. Se decidió **no abrir con una cita**: el informe gana más empezando por una cifra que contradice lo esperado | Es una oración; entra en el párrafo de apertura sin desplazar nada |
| **Atribución errónea** (afirmar que una obra es de un artista que no la pintó) | 🔴 **Esta fila decía que el riesgo estaba *"cubierto por construcción: el retriever no puede devolver una obra que no esté en el índice, y la metadata sale del museo"*, y la conclusión es falsa — medido el 2026-11-01.** Las dos cláusulas son verdaderas y no alcanzan: para el **14,2 %** de las obras indexadas (**370 de 2.605**, 120 textos compartidos) la `description` que el museo publica **es del álbum entero y no de esa hoja**, así que la metadata **sale** del museo **y no es sobre esa obra** — y el **7,3 %** de las referencias que emite un brazo `C4` apunta a uno de esos textos. 🔑 **Un dato correcto en el origen y mal atribuido en el destino no está cubierto por nada.Queda afuera por espacio, no por estar cubierto:** lo mide el informe del Ejercicio 4 (`ejercicio-4.md §5.5`) y la versión canónica del hecho está en `el relevamiento § Riesgos de datos conocidos` | Una frase del bloque de alucinación, si ese entra |
| **Contenido sensible dentro del corpus** (desnudos, violencia) | No está medido, y abrir el tema sin cifra invita a una pregunta que no podemos contestar | — |
| **`style` como campo del output** | CMA no lo publica y en AIC está en el 78 % de sus pinturas. **Es un desvío no consultado**, y además ahora está medido que *no es un campo de estilo consistente*: lo que separa el espacio CLIP son `Himalayan` (0,86), `Chinese` (0,75) y `Japanese` (0,71) —que no son estilos— y `Renaissance` (0,24) o `19th century` (0,06) no separan nada | — |
| **UI y su superficie de atribución** | La UI **murió sin decidirse el 25/10**: el disparador del 4/11 pedía la ronda piloto anotada por dos personas y no lo está. La obligación de atribuir **no depende** de que haya UI | — |
| **Observabilidad en detalle** (tabla de tokens y costo por etapa) | ⚠️ **Esta fila decía que LangSmith era *"una herramienta que todavía no integramos"*, y es falso desde el 24/10.** Está integrada, la key se verificó en los dos sentidos y el cuerpo del informe ahora lo dice. Lo que queda afuera es **la tabla de tokens por etapa**, que la produce el hito de métricas; y hay una deuda conocida: los tokens **no caen en los campos nativos de LangSmith**, así que el dashboard del servicio muestra cero aunque el artefacto de corrida tenga los números | Entra como dos columnas cuando la tabla exista |
| **pgvector / Postgres, MLOps en la nube** | Es la elección correcta en un escenario productivo real y aparece en la Clase 8, pero el informe quedaría hablando de un sistema que no construimos | Entra en el bloque de escalado si el grupo prefiere ese enfoque al de ANN |
| **Los cuatro orígenes distintos del 403, el header en la allowlist del WAF** | Material excelente de ingeniería y de oral, pero es *implementación*, no estrategia de producción | — |
| **El límite honesto del caché** (*re-embeber es gratis, pero cambiar el criterio de inclusión invalida el split, el set de consultas y los **379** juicios del pool*) | Es honestidad que suma, y fue de lo último que se cortó por espacio | ~30 palabras al final del párrafo del caché |
| **La política de degradación etapa por etapa**, y que *"la escalera de degradación es la escalera de la ablación"* | La política ya corrió y el cuerpo lo dice con su cifra; desarrollarla etapa por etapa son ~80 palabras. Y la glosa es elegante y **falsa en general** (vale para el reranker, no para el planificador) | — |

## C. Tres swaps listos, si el grupo quiere otro énfasis

1. **Menos "quién anota", más teoría de la cátedra.** Sacar el cuarto bloque de la sección 1 y
   pasar sus ~115 palabras a la pregunta abierta de la Clase 7.2 unida a los guardrails de la
   Clase 8 y SynthID de la Clase 9. **Costo, y es alto: el informe dejaría de declarar que el
   ground truth lo produce IA**, que es la limitación metodológica más grande del trabajo. No lo
   recomiendo; se deja listo porque es la decisión del grupo.
2. **Más mecanismo, menos producción.** Reponer las dos primeras filas de la sección B —el filtro
   de formato y el 39 % de `provenance`— sacando de la sección 2 el párrafo de indexación y caché
   entero (~95 palabras). Costo: se pierde la mitad *"lo construimos"* del primer tema de
   producción, que es la que el criterio de respuesta garantizada señala como la que vuelve bueno este informe.
3. **Volver a la tabla de estado.** Si la maqueta final prefiere la tabla a la prosa, se repone
   tal cual y hay que recortar **6 líneas** en otro lado; la escalera de la sección F dice dónde.

## D. Afirmaciones que me hubiera gustado hacer y no pude sostener con el repo

1. **"Medimos la regla de honestidad y el resultado es X".** Sigue siendo la medición que más
   mejoraría este informe y sigue sin hacerse. **Y ahora tiene una arruga nueva:** con la anotación por IA la
   anota IA, así que el resultado va a medir *un modelo juzgando si otro modelo citó bien*.
2. ✅ **"El costo por consulta es X USD" — RESUELTO el 2026-11-03.** Este punto decía que faltaba
   el dólar *"por una razón concreta y barata de arreglar"* y que era **una tarea de una persona
   con un navegador**: anotar el precio del modelo de chat con su fecha y su URL. Era exacto. Se
   leyó en `developers.openai.com/api/docs/pricing` el 2026-11-03 y la columna salió **sin
   re-correr una sola consulta**: los tokens ya estaban en los libros mayores de las 1.151
   corridas, y el precio **no determina el `config_id`**, así que nada quedó huérfano.
   **`C1` 0,0029 → `C4` 0,0277 USD/consulta (9,6×)**, y el total derivado reconcilia con el
   dashboard: el proxy factura un **19 % por debajo del precio de lista**, con lista como base
   (`ejercicio-4.md §5.4`). ⚠️ **La distinción se mantiene**: tokens
   medidos, USD derivado de un precio **de lista** que no es la tarifa del proxy.
   🔑 **Y derivarlo destapó un error en una cifra que sí estaba publicada:** al mirar el libro
   mayor rol por rol apareció que `C2`–`C4` contaban **una llamada de menos por consulta**, porque
   un plan reusado no paga el planificador (el reuso del plan al barrer el peso de fusión). *La columna que faltaba tapaba un error en la
   que no faltaba.*
3. **"La distribución cultural del corpus es esta"**, agrupada por región. Las cifras por etiqueta
   del informe son exactas, pero cualquier agregado regional exige una tabla de sinónimos que no
   existe (el criterio de normalización de culture, abierto). Por eso el informe reporta etiquetas y la unión India —que ya
   está decidida— y **no** inventa un agrupamiento.
4. **"N obras del corpus tienen contenido sensible"**: nadie lo midió y ningún campo del museo lo
   declara.
5. **"El sistema detecta imágenes generadas por IA"**: no está implementado ni evaluado.
6. 🔴 **"El ground truth es independiente del sistema evaluado".** No se puede y no se va a poder:
   son todos modelos de lenguaje leyendo la misma rúbrica. Lo que el informe afirma es más débil y
   es verdad — dos familias distintas, y el acuerdo entre ellas medido cuando se anote.
7. ⚠️ **"El acuerdo inter-modelo es κ = X".** Al 2026-10-30 **los 379 juicios todavía no existen**:
   la cláusula del refinamiento de la rúbrica manda refinar la rúbrica antes de anotar, y en disco solo están los **20 del piloto**.
   El cuerpo está escrito en consecuencia —habla del κ del piloto, que sí existe— y **no** anticipa
   un número que nadie midió.
8. **"El chunking ciego recupera peor".** El informe dice que **atribuye peor**, que es lo medido, y
   **el veredicto se conserva: no se puede afirmar que *recupere* peor.** 🔴 **Lo que cambió es el
   motivo, y la precisión es del 2026-11-01.** Al 30/10 este ítem decía que los fragmentos de ese
   brazo *no tienen juicio por construcción* —**29 de 150** ítems de su top-5 sin juzgar—; **esos 29
   se juzgaron** por pooling incremental (31 juicios nuevos, de un solo anotador), así que
   el `nDCG@5` **existe** y la pregunta está medida. **Y lo que salió es un no-resultado:** el
   `Δ ciego − C3` **cambia de signo** entre las dos poblaciones del ground truth —**−0,0405** sobre
   la completa (410 pares) y **+0,0035** sobre la restringida (277 pares, TREC-`cero`)—, así que lo
   publicable es que el eje de chunking **no mueve el retrieval de forma concluyente**
   (`ejercicio-4.md §5.3b`). ⚠️ **Y va con su confundidor, que explica el cambio de signo mejor que
   cualquier hipótesis sobre el chunking:** los 31 juicios nuevos **no pasaron por
   acuerdo/desempate** —no hay con quién acordar cuando el anotador es uno— así que **sobreviven a
   la restricción por construcción**, y ahí el brazo ciego conserva **29** ítems con grado contra
   **ninguno** de `C3`. 🔑 **La conclusión no mejoró: se volvió más interesante.** Pasó de *"no lo
   sabemos porque no medimos"* a *"lo medimos, no es concluyente, y sabemos por qué"*.

## E. Hallazgos sobre el material — para

1. 🔴 **Quinta vez del error de población, en el alcance del proyecto.** El recuadro de `artista` dice *"`artist`
   cubre el **67,4 %** del corpus (AIC 92,8 %, CMA 59,2 %)"*. Esas tres cifras son de los **2.894
   elegibles con texto e imagen** (pre-split), no del corpus: sobre los 2.605 son **68,2 % /
   93,1 % / 60,1 %**. Y en el mismo recuadro *"las 2.131 obras sin artista"* es del **universo de
   CMA** (3.950) — una tercera población en el mismo párrafo. *(Del 18/10; se repite acá porque no
   verifiqué que esté aplicado.)*
2. ✅ **Resuelto: `101` vs `102` etiquetas de `culture`.** La discrepancia que este anexo reportó
   el 18/10 está cerrada a favor del artefacto: **son 102** sobre los 2.605 ids del split
   congelado, y el **101** del docstring del código es el defecto. El informe publica 102.
3. ✅ **Resuelto, y ya no "a medias": el `0,012 USD`.** Ya no es una proyección: el embebido se
   pagó el 24/10 y los **622.314 tokens en 49 llamadas están medidos** del `usage` de la API.
   El **USD sigue siendo derivado** —el precio es declarado— pero desde el 2026-11-03 **lo es para
   todos los roles y no solo para el embebido**, y el modelo por token reconcilia contra el
   dashboard del proxy: factura un **19 % por debajo de lista**, con lista como base. El informe
   mantiene la distinción explícita.
4. 🟡 **Las cifras de licencias del informe, reverificadas hoy contra el catálogo** y por si
   alguien las quiere asentar: sobre los 2.605 ids del split, los campos de `knowledge` dan **638 `description`
   CC-BY-4.0** (= las 638 obras de AIC, una por obra), 1.967 `description` CC0, 1.199
   `did_you_know` y 2.372 `provenance`; y **542 obras del corpus tienen más de una licencia entre
   sus campos de `knowledge`**, que es el 85 % de las 638 y la cifra que el informe usaba como
   proyección del chunking ciego. **Ahora esa proyección tiene su medición al lado** (442
   fragmentos mezclados de 2.775 en el brazo ciego construido).
5. 🔴 **el registro de decisiones quedó viejo en dos lugares por y la anotación por IA, y los dos dicen lo contrario de lo
   decidido.** `§ Cuellos de botella reales` abre con *"el único recurso verdaderamente escaso es
   **la anotación humana**: las ~540 anotaciones manuales dependen de tres personas"* —el recuadro
   de abajo corrige el ~540 a 379, pero **no** el *"manuales"* ni el *"tres personas"*—. Y
   `§ Riesgos abiertos` tiene dos filas vivas con la misma premisa: *"κ de Cohen < 0.6 **entre
   anotadores**"* y *"**la anotación humana** es el recurso más escaso del proyecto"*. Después de
   la anotación por IA, **ningún humano anota**.
6. 🟡 **El artefacto del κ del piloto lleva una cifra anterior a la medición del pool.** Su
   `nota_de_los_dos_kappa` dice que el κ del acuerdo entre anotadores se calcula sobre *"el ~20 % del pool definitivo
   (**~108** juicios duplicados)"*, y el pool congelado publica **76**. Es la misma deuda del
   `~540` en un artefacto, no en un documento.
7. 🟡 **`el registro de decisiones § Insumos para el Ejercicio 3` es anterior a seis hitos y manda mal en tres
   lugares.** Su tabla de Producción lista **Observabilidad** como *"LangSmith para trazas"* sin
   decir que está integrada; su fila de **Privacidad** pregunta *"¿se mandan a la API de
   OpenAI?"* cuando el destino real es el **proxy de la cátedra**; y **la política de trazado no aparece**, siendo
   el mejor insumo de privacidad que el proyecto produjo. Es la lista de la que salió este
   informe, así que conviene que no mande mal a la próxima sesión.
8. 🔴 **la anotación por IA deja un hueco de criterio y no lo resuelvo yo — ver la sección H.**
9. 🟡 **La cobertura cultural de C4 no me cerró al segundo decimal.** El titular publicado es
   **0,60 → 2,60 → 4,50 de 5**; recontándolo a mano sobre las 10 consultas culturales, los top-5
   de las corridas limpias y `culture` del catálogo cortado por coma, me da **0,60 → 2,60 →
   4,40**. C2 y C3 coinciden exactos. El informe publica **4,50**, que es lo que emite el
   instrumento; pero una diferencia de una obra en 50 antes de una entrega conviene mirarla.
10. ⚠️ **Antes de entregar, revisar el estado de consulta.** El informe se entrega el **17/11**,
    después de la instancia del **4/11**. El chunking sigue sin consultar, pero ahora
    **llega mitigado con números** —es el 21,2 % de citas con atribución forzada— y ése es
    exactamente el argumento que conviene llevar al 4/11. `style` sigue sin consultar y
    el informe no lo menciona.

## F. Conteo de palabras y ajuste a dos hojas

| | Palabras | Líneas |
|---|---|---|
| Cabecera (título, encuadre, poblaciones) | 117 | 11 |
| **Sección 1 — Responsible AI y Safety** (4 bloques) | **627** | 44 |
| **Sección 2 — Producción** (3 párrafos + la línea de estado) | **455** | 33 |
| Nota de fuentes al pie | 52 | 3 |
| **Informe completo**, hasta el separador | **1.251** | **94** |
| Anexo | no cuenta | no cuenta |

🔢 **El conteo es reproducible y la regla va escrita, porque la del 18/10 no se puede reproducir.**
Cuento **tokens separados por espacio** sobre el cuerpo con las marcas de Markdown quitadas
(backticks, asteriscos, guiones bajos, barras de tabla, almohadillas de título), tablas incluidas.
⚠️ **Con esa misma regla, la versión del 18/10 mide 1.135 y su propia nota decía 1.168**, así que
los dos números **no son comparables** y publico el que puedo recalcular. En la regla común:
**1.135 → 1.251, un 10 % más de palabras**.

**El objetivo del encargo era ~1.100 y esto queda un 13 % por encima**, y conviene decirlo en vez
de redondear para abajo. **Pero la restricción real no son las palabras: son las líneas**, y ahí
el informe **no creció**: 94 contra 92. Entraron siete resultados nuevos y la densidad subió,
porque salió una tabla que gastaba 6 líneas en 25 palabras y cinco temas se cortaron enteros.

**¿Entra en dos hojas? Sí, con el mismo margen chico que la versión auditada.** El cuerpo son
**7.023 caracteres** en 11 párrafos, sin tablas. A A4 con márgenes de 2,5 cm, cuerpo de 11 pt e
interlineado sencillo entran ~95 caracteres por línea; contando cada párrafo por separado (una
línea parcial por párrafo) y sumando los saltos entre bloques dan **94 líneas**: **1,96 carillas a
48 líneas por página y 2,09 a 45**. El 18/10 daba 92 líneas, 1,92 y 2,04 — **la misma maqueta**.
Traducido a instrucciones: **mantener 11 pt e interlineado sencillo y entra; a 12 pt o con
interlineado 1,15 se va a tres carillas.**

**Si al maquetar no entra**, el orden de corte —y cada paso es autosuficiente— es: **(1)** la
enumeración de la ingesta (*"69 minutos para las 5.915 obras, 0 fallas"*), −12 palabras y −1
línea, porque es el dato de producción menos interesante ahora que el costo del LLM está medido;
**(2)** la oración *"Nada de lo ⬜ se presenta como construido"* (−9 y −1 línea: es énfasis, la
separación ya está arriba); **(3)** el conteo de las 33 llamadas por consulta (−14 y −1); **(4)**
la oración del piso irreducible sin red (−35 y −2), que es la más citable de la sección 2 y por
eso va última. Los cuatro pasos juntos dan **−5 líneas → 89**, o sea **1,85 y 1,98**.

## G. Qué cambió respecto de la versión del 18/10

**Salió, porque dejó de ser verdad:**

- 🔴 *"Observabilidad en detalle … describir **una herramienta que todavía no integramos**"*
  (anexo) y *"La observabilidad **no está construida**"* (cuerpo). **Falso desde el 24/10**: la
  integración está hecha y la key se verificó **en los dos sentidos** —autentica y escribe—.
- *"El embebido de texto está **presupuestado** en 0,012 USD … **no un gasto medido**"*. Se pagó el
  24/10: **622.314 tokens en 49 llamadas**, y el USD sigue siendo derivado de un precio declarado.
- *"Degradación elegante —contratada, **todavía sin correr**"*. Corrió, en las seis etapas.
- *"Las **2 + N** llamadas por consulta están **diseñadas, no corridas**"*. Están medidas: 33.
- *"**La imagen se envía a la API de OpenAI**"* como único destino externo. **Son tres**, y el
  primero **no es `api.openai.com`** sino el proxy LiteLLM de la cátedra.
- *"Un chunk ciego de 500 **mezclaría** CC-BY con CC0 en 542 de 638 obras (85 %)"*. El condicional
  era correcto como proyección y se reemplazó por la medición del brazo construido.
- *"El límite honesto del caché … los **~540** juicios"* → **379**. *"**101** etiquetas de
  `culture`"* → **102**, con la discrepancia resuelta a favor del artefacto. *"Costo total del
  proyecto (~4.000 llamadas) … es **una cuenta, no una medición**"* → **6.717 llamadas ≈ 6,72
  USD**, con su base declarada.
- *"La UI está fuera del alcance **comprometido** (disparador del 4/11)"* → el pendiente **murió
  sin decidirse** el 25/10; ya no hay disparador que esperar.

**Entró, y qué desplazó:**

| Entró | Qué desplazó |
|---|---|
| 🔑 El chunking ciego **fuerza CC-BY sobre CC0 en el 21,2 %** de las citas que publica (22 de 104, sobre los 103 pares comunes) | El condicional de las 542 obras, en el mismo párrafo. **Neto ≈ 0 palabras** |
| 🔑 la política de trazado: la traza por defecto devolvía `inputs`/`outputs` completos; la política partida en dos, con **62 planes + 1.151 corridas** sin imagen, **1.149** trazadas enteras y **2 ad-hoc** con 5 campos censurados | **`is_nazi_era_provenance` entero** (~55 palabras), que era el otro medio bloque de ese párrafo |
| 🔴 **Quién anota** (la anotación por IA): bloque nuevo de ~115 palabras, con el κ **inter-anotador → inter-modelo** y el 0,419 publicado con lo que mide | **El bloque del vocabulario** como bloque propio (~110): se comprimió a su consecuencia operativa dentro del bloque del sesgo propio |
| La **cobertura cultural 0,60 → 2,60 → 4,50 de 5** (C2 → C3 → C4, 10 pedidos culturales) | Los dos ejemplos nominales del bloque del sesgo propio (`Rajasthan`, `Korea`) y la etiqueta `Japan` de la apertura |
| **Degradación corrida**: 301 corridas de inyección, 6 etapas, ≥ 30 cada una | La redacción condicional del mismo párrafo, más la frase de las métricas que se niegan a agregar |
| **Costo medido**: 622.314 tokens / 49 llamadas / 0,012446 USD · 33 llamadas por consulta · 6.717 ≈ 6,72 USD como **cota inferior** | La prosa condicional del mismo párrafo, el detalle de la segunda corrida del caché (64 / 5.687) y el tamaño en GB de la ingesta |
| El criterio que **depende de un tercero al correr**: 148/150 → 150/150 sobre los mismos artefactos, y un lazo de **30 redirecciones** | El resto del párrafo de escalado a 100k, que pasó a la prosa de estado |
| **La tabla de estado → prosa** | 6 líneas por 25 palabras; la separación ✅/🟡/⬜ quedó entera |

**Se evaluó y quedó afuera, con su razón** (las cinco primeras filas de la sección B): el filtro de
formato físico, el 39 % de `provenance`, el bloque del vocabulario, `is_nazi_era_provenance` y el
presupuesto de evidencia de suma cero. **Ninguno salió por ser débil: salieron por líneas**, y las
cinco filas dicen exactamente qué recortar para reponer cada uno.

### G.2 · Y qué cambió el **2026-11-03** — solo el párrafo de costo

Una sola edición del cuerpo, **+15 palabras y 0 líneas**, que es la restricción que manda (§F).

| Salió | Entró |
|---|---|
| *"🟡 **medido a medias** el costo —llamadas y tokens sí, dólares solo donde hay precio declarado—"* | ✅ el costo **por token y por rol**, en USD, en la lista de lo construido-corrido-y-medido |
| *"el acumulado al 27/10 son **6.717 llamadas ≈ 6,72 USD** sumando los libros mayores"* | **0,0277 USD/consulta** en el sistema completo contra **0,0029** del base, y el total **USD 12,43 derivado contra 10,0654 facturados** |

🔴 **Y la salida de esa cifra es, en sí misma, un resultado de producción que el informe
necesitaba.** El **6,72** salía de multiplicar 6.717 llamadas por **USD 0,0010 por request** —la
cifra del dashboard—, y `ejercicio-4.md §5.4` **refutó ese modelo de costo**: el proxy factura por
token, el grupo agotó el tope de USD 10 **sin pasarse de llamadas**, y *el precio por request no es
el precio*. Dejar el 6,72 en el informe de producción era publicar el número con el modelo que el
propio proyecto había desmentido. **El reemplazo no es una cifra más nueva: es la cifra con el
modelo correcto.**

## H. 🔴 El hueco de criterio que la anotación por IA me dejó — es del orquestador, no mío

la regla de los dos anotadores de familias distintas dice, textual, **"dos anotadores de *familias de modelo distintas*, y **ninguna puede ser
la del reranker**"**. Y en la justificación prescribe la implementación: *"son **sesiones de
agente** — **un Claude y un Codex/GPT**"*.

**Las dos mitades no cierran.** El reranker es `openai/gpt-4.1-mini`; un anotador *Codex/GPT*
**es de la familia de OpenAI**, o sea la del reranker. La cláusula se sostiene si *familia*
significa *modelo concreto* y se rompe si significa *proveedor* — y la propia justificación usa el
sentido de proveedor cuando dice que *"un anotador OpenAI sería el modelo evaluándose a sí
mismo"*.

**Qué hice con eso, y por qué:** escribí el cuerpo con la afirmación que es verdadera bajo
**cualquiera** de las dos lecturas —*"los dos anotadores son de familias de modelo distintas"* y
*"no se puede afirmar que el ground truth sea independiente del sistema evaluado"*— y **no** puse
*"ninguna es la familia del reranker"*, que es la que depende de la lectura. **No ablandé ni
endurecí la declaración: la acoté a lo que sobrevive.** La decisión de cuál lectura vale es del
orquestador, y conviene tomarla antes, porque cambia **qué sesión anota**.

**Un segundo hueco, menor y de calendario:** la anotación por IA decide el *cómo* y la cláusula del refinamiento de la rúbrica manda refinar la
rúbrica primero, así que **al 30/10 los 379 juicios no existen y el κ inter-modelo tampoco**. El
cuerpo está escrito en consecuencia —habla del κ **0,419 del piloto**, que sí existe, y describe
el régimen de anotación en futuro— pero si alguien lee el informe esperando un número de acuerdo
inter-modelo, no lo va a encontrar, y eso es correcto.
