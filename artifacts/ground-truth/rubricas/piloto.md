+++
rubrica_id = "piloto"
estado     = "completa"      # plantilla | completa
# refina   = "piloto"      # si esta rúbrica REFINA otra (A5, κ < 0.6), decilo acá
escrita_por = "sesiones de IA — NO leída ni firmada por los integrantes (ver el recuadro)"
fecha      = "2026-10-25"        # AAAA-MM-DD
+++

# Rúbrica `piloto` — cómo se asigna un grado 0 / 1 / 2

> 🔴 **Este texto NO lo escribió ni lo leyó ninguna persona del grupo, y hay que
> saberlo antes de usarlo.** El borrador lo redactó la sesión de orquestación y lo
> corrigió en cinco lugares otra sesión de IA (ver *Qué cambió del borrador*, al
> final). El encabezado decía los nombres de los tres integrantes; **se corrigió
> el 2026-10-25 porque era falso**: el campo `escrita_por` significa *"quiénes la
> escribieron (es trabajo humano, §3.6)"*.
>
> **Las cinco correcciones son buenas y una es load-bearing** —la precedencia del
> metadato faltante, que evitó un 1 regalado en `cma:314448`—, así que el texto se
> conserva. Lo que se corrige es **la afirmación sobre quién lo hizo**, no el
> contenido.
>
> ⚠️ **Consecuencia, y es la que importa:** una rúbrica que ninguna persona leyó
> **no puede sostener un κ inter-anotador** (`A5`). Los 20 juicios que existen
> salieron de la misma sesión en dos pasadas, así que el κ que hay mide
> **consistencia de un lector consigo mismo**, no acuerdo entre dos. Está
> declarado así en `kappa-…json` y en `docs/07 D15`.
>
> **Por qué importa tanto un texto:** κ es siempre *κ sobre una rúbrica*. Dos
> anotadores que no leyeron la misma no produjeron juicios comparables, y si este
> texto cambia, **los juicios hechos con el anterior se descartan**. Eso es lo que
> el piloto de 20 juicios compra: pagar ese descarte cuando cuesta 20 y no 540.
>
> ⚠️ **Y desde la firma, este archivo no se toca.** Corregir una coma acá mueve el
> `hash_del_texto` que cada aporte cita, y entonces los juicios ya hechos no se
> pueden cruzar con los nuevos. Lo que se hace es **derivar otra rúbrica**, con
> otra designación y `refina = "piloto"` — que es la rama de `A5` si κ < 0.6, y es
> barata justamente porque acá cuesta 20 juicios.

## La pregunta que se contesta en cada juicio

Dada **la imagen original** de la consulta y **el pedido**, ¿cuánto cumple
**esta obra recuperada** el pedido *respecto de la imagen original*?

⚠️ **No es «se parece»**: `docs/05` dice que *si el usuario pide «versiones más
cálidas» de un paisaje gris, la respuesta correcta probablemente no comparte
estilo con el original*. Premiar el parecido sería premiar al sistema por ignorar
al usuario.

**Tres reglas de encuadre, y las tres importan más que las definiciones:**

1. **Cada par se juzga solo.** No se reparten notas entre las 5 obras de una
   consulta. Si las 5 son malas, las 5 son 0: *la menos mala no es un 2*.
2. **No se mira el orden.** La planilla no trae el puesto del ranking a
   propósito — *«esta salió primera»* ancla más fuerte que cualquier definición.
3. **No se consulta con la otra persona** mientras se anota. Un acuerdo
   conversado mide la conversación, no la rúbrica, y κ quedaría inflado.

## Grado 2 — Cumple claramente el pedido respecto de la imagen original

Se pone **2** cuando las dos cosas son ciertas:

- **(a)** La obra tiene **la propiedad que el pedido nombra**, y se ve o se lee
  en lo que la planilla muestra (la imagen de la obra, o su cultura / año /
  artista / museo).
- **(b)** Esa propiedad está **respecto de la imagen original**, no en absoluto.
  *«Más cálida»* significa más cálida **que la original**, no cálida a secas.

**Lo que no hace falta para un 2:** que la obra se parezca a la original, que
sea del mismo estilo, del mismo período o del mismo formato. Puede ser un folio
cuando la original era un óleo y seguir siendo un 2.

**Por categoría, qué es la propiedad:**

- **cromático / visual** — el cambio de paleta, luz o composición que el pedido
  pide. Es **comparativo**: hay que mirar las dos imágenes.
- **cultural** — **las dos mitades**: la obra es de la cultura pedida **y** tiene
  **el tema que el pedido nombra**. Solo si el pedido no nombra ninguno, el tema
  es el de la imagen original. Una mitad sola es un 1 (abajo).
  > 🔑 Los dos pedidos culturales no se juzgan igual, y confundirlos es un 1 donde
  > iba un 2. *«El mismo paisaje pero pintado en Japón»* **hereda** el tema de la
  > imagen; *«retratos conmemorativos en Italia»* **lo nombra**, y entonces lo que
  > hay que buscar es un retrato conmemorativo italiano, **se parezca o no a la
  > imagen original**.
- **ambiguo** — la obra cumple **la lectura que anotaste en `nota`** (ver
  *Desempate*).

## Grado 1 — Cumple parcialmente o es defendible

Se pone **1** en cualquiera de estos casos:

- **Media propiedad.** La obra es de la cultura pedida pero **no del tema** —el
  que el pedido nombra—, o tiene el tema pero no la cultura. Es el caso típico de
  los pedidos culturales.
- **La propiedad está, pero poco.** Es algo más cálida, no claramente más
  cálida. Un desplazamiento que habría que señalar con el dedo.
- **Es defendible y no obvio.** Otra persona razonable podría poner 2 o 0, y vos
  podés explicar por qué 1 en un renglón. **Escribilo en `nota`.**
- **No se puede juzgar con lo que hay.** La obra no trae el dato con el que se
  juzgaría —sin `culture`, sin fecha— y la imagen no alcanza para decidir.
  **No es 0**: 0 significa *«no cumple»*, y *«no se puede saber»* no es eso.
  Dejalo en 1 y decilo en `nota`. ⚠️ **Antes de usar esta salida, mirá la
  imagen**: muchas veces la obra se juzga sola aunque le falte el metadato.
  > 🔑 **Y esta salida vale solo si el dato que falta es el único que decidiría el
  > juicio.** Si la mitad que **sí** se ve ya falla, es **0** y no 1: una obra sin
  > `culture` que además **no** es un retrato conmemorativo no cumple *«retratos
  > conmemorativos en Italia»*, y el metadato faltante no la rescata. *«No se
  > puede saber»* es sobre lo que falta, nunca sobre lo que está a la vista.

## Grado 0 — No cumple

Se pone **0** cuando pasa cualquiera de estas:

- **La propiedad no está.** No es más cálida, no es de la cultura pedida, no
  tiene lo que el pedido nombra.
- 🔑 **La obra ignora el pedido aunque acompañe a la imagen.** Es parecida a la
  original —mismo tema, misma paleta, misma escena, incluso la misma serie, el
  mismo artista o **la obra original misma**— **y no cambió en la dirección que
  el pedido pide**. Esto es 0,
  no 1, y es el renglón más importante de la rúbrica: **es exactamente lo que el
  proyecto existe para medir.** Un sistema que devuelve «parecidas» sin leer el
  pedido tiene que puntuar 0 acá, o no estaríamos midiendo nada.
- **Va en la dirección contraria.** Se pidió más cálida y es más fría.

## Ejemplos

> 📌 **Los cuatro ejemplos usan consultas que NO están en el piloto, a
> propósito**: si la rúbrica pre-juzgara un par que después hay que anotar, el
> κ mediría la rúbrica y no a los anotadores. Y las obras están descritas en
> vez de nombradas, por la misma razón.

| grado | imagen original | pedido | obra recuperada | por qué |
|---|---|---|---|---|
| **2** | paisaje rural en verdes y grises apagados | *versiones más cálidas de este paisaje* | un campo al atardecer, con ocres y rojos dominando | tiene la propiedad pedida y es **claramente más cálida que la original**. Que sea de otro país y otro siglo no importa |
| **1** | retrato de mujer, fondo oscuro | *retratos de mujeres en la pintura francesa* | un paisaje francés del XIX | media propiedad: la cultura pedida está, **el tema no**. El caso canónico de 1 |
| **1** | marina con hielo, paleta fría | *escenas de agua en la pintura china* | una obra china con un río al fondo, el tema es una escena cortesana | la cultura está, el agua aparece pero **no es el asunto de la obra**. Defendible en las dos direcciones → 1 + `nota` |
| **0** | tinta monocroma, bambú | *el mismo motivo pero con color* | otra tinta monocroma de bambú, muy parecida | 🔑 **acompaña a la imagen e ignora el pedido.** Es el 0 más importante de todos: si esto puntuara 1, el sistema cobraría por no escuchar |

## Desempate

**La duda va al medio.** Entre **2 y 1**, poné **1**. Entre **1 y 0**, poné
**1**. El 2 y el 0 son los grados seguros; el 1 dice *«parcialmente o
defendible»*, así que **dudar ya es la definición del 1**.

> ⚠️ **Y siempre que uses el desempate, escribí por qué en `nota`.** Esas notas
> son **el producto del piloto**: es con ellas que se decide si la rúbrica se
> refina antes de gastar los ~540. Una nota de media línea vale más que el grado.

**Los cuatro casos que `docs/05` ya anticipa:**

- **La obra cumple el pedido pero es la misma escena que la original** → mirá si
  **cambió en la dirección pedida**. Si cambió, el parecido no la castiga: puede
  ser 2. Si no cambió, es **0** (ver el 🔑 del grado 0).
- **El pedido es ambiguo** (*«con más vida»*) → **elegí una lectura, escribila en
  `nota` en la fila de `fila` más baja de esa consulta, y usá la misma para las
  5.** Las 5 obras de una consulta vienen **juntas** en la planilla, así que
  conviene mirar las 5 antes de fijar la lectura; lo que no se puede es fijar
  una, juzgar dos y cambiarla. Cambiar de lectura entre obras de la misma
  consulta es lo único que vuelve el juicio no comparable. **Qué lectura elegís
  no está mal**; cambiarla a mitad, sí.
- **El pedido es cultural y la obra es de la cultura pedida pero no del tema
  —el que el pedido nombra—** → **1**. Está arriba y es el caso más frecuente.
- **La obra no tiene el dato con el que se juzgaría** → mirá la imagen primero;
  si aun así no se puede, **1** con su `nota`. Pero si la mitad visible ya falla,
  **0**: el dato que falta no rescata lo que se ve.

**Si un desacuerdo entre las dos personas hay que resolverlo**, lo hace el tercer
integrante, y **ese desempate es una anotación más** y no una corrección del
consenso (`docs/05`, precisión del 12/10).

---

## Qué cambió del borrador

Lo que el grupo corrigió al leerlo, para que el diff se pueda leer sin el diff.
**Ninguno de los cinco cambios se decidió mirando un par del piloto**: los 20
seguían sin abrir cuando se firmó.

| # | Dónde | Qué decía el borrador | Por qué se cambió |
|---|---|---|---|
| 1 | **Grado 2**, categoría *cultural* | *«la obra es de la cultura pedida y conserva el tema o el género de la original»* | Solo servía para los pedidos que **heredan** el tema (*«el mismo paisaje pero pintado en Japón»*). Cuatro de los cinco culturales del set **nombran el tema ellos mismos**, y con la redacción vieja un retrato conmemorativo italiano que no se pareciera a la imagen original puntuaba 1 cuando iba 2. Ahora el tema es **el que el pedido nombra**, y el de la imagen solo si el pedido no nombra ninguno |
| 2 | **Grado 1**, *«no se puede juzgar con lo que hay»* | la salida a 1 no tenía límite | Dejaba pasar a 1 una obra que **ya se ve** que no cumple, solo porque le falta un metadato. Agregada la precedencia: *«no se puede saber»* es sobre lo que falta, nunca sobre lo que está a la vista. Con el metadato ausente y la mitad visible fallando, es **0** |
| 3 | **Grado 0**, el 🔑 | *«la misma serie o el mismo artista»* | Faltaba el caso extremo: que la obra recuperada **sea la imagen original**. Es el 0 más claro de todos y no estaba nombrado |
| 4 | **Desempate**, pedido ambiguo | *«escribila en `nota` en la primera obra de esa consulta»* | *«La primera»* no estaba definido en una planilla mezclada. Ahora dice **`fila` más baja**, y agrega que conviene mirar las 5 antes de fijar la lectura: las 5 de una consulta vienen juntas |
| 5 | **Encabezado** | el banner decía que era un borrador para corregir | Firmado. Y agregada la regla que el borrador no decía: **desde la firma el archivo no se edita**, porque mueve el `hash_del_texto` que los aportes citan; lo que se hace es derivar otra rúbrica con `refina` |

**Lo que se dejó igual a propósito:** las tres reglas de encuadre, los cuatro
ejemplos y la regla *«la duda va al medio»*. Y en particular **no se acordó de
antemano la lectura de los pedidos ambiguos**: cada uno elige la suya y la
declara. Si las dos personas eligen lecturas distintas y κ se cae en la
categoría `ambiguo`, **eso es el hallazgo del piloto** — acordarlo antes lo
taparía, y un acuerdo conversado mide la conversación.

---

## Procedencia (no se edita)

- Identidad: `rubrica_id` = la **designación declarada** del front matter
  (`architecture.md §6.6`, clase 3). **No es un hash del texto, a propósito**: una
  coma corregida daría otra rúbrica y descartaría los ~540 juicios, y la clase 3
  existe para que ese precio lo decida una persona y no un diff.
- La escala (tres valores) es compartida; **lo que significa cada valor lo fija
  este archivo** (`domain-model.md §4.5`).
- Validar: `poetry run tp3-anotar rubrica`
