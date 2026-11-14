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
