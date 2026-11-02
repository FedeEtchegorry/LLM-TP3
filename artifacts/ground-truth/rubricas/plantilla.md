+++
rubrica_id = "plantilla"
estado     = "plantilla"      # plantilla | completa  ← pasalo a `completa` al terminar
# refina   = "piloto"      # si esta rúbrica REFINA otra (el acuerdo entre anotadores, κ < 0.6), decilo acá
escrita_por = "REEMPLAZAR"       # quiénes la escribieron (es trabajo humano, el módulo de evaluación)
fecha      = "REEMPLAZAR"        # AAAA-MM-DD
+++

# Rúbrica `plantilla` — cómo se asigna un grado 0 / 1 / 2

> 🔴 **Este archivo es una plantilla y el texto lo escribe el grupo.**
> Reemplazá cada `REEMPLAZAR`. Mientras quede uno, `estado` tiene que seguir en
> `plantilla`, y con `estado = "plantilla"` **no se puede exportar una planilla**:
> gastar anotación humana bajo una rúbrica a medio escribir es exactamente lo que
> la ronda piloto existe para no hacer con los ~540 (la evaluación).
>
> **Por qué importa tanto un texto:** κ es siempre *κ sobre una rúbrica*
> (el módulo de evaluación). Dos anotadores que no leyeron la misma no produjeron
> juicios comparables, y si este texto cambia, **los juicios hechos con el anterior
> se descartan** (la evaluación). Eso es lo que el piloto de 20 juicios compra: pagar
> ese descarte cuando cuesta 20 y no 540.

## La pregunta que se contesta en cada juicio

Dada **la imagen original** de la consulta y **el pedido**, ¿cuánto cumple
**esta obra recuperada** el pedido *respecto de la imagen original*?

⚠️ **No es «se parece»**: la evaluación dice que *si el usuario pide «versiones más
cálidas» de un paisaje gris, la respuesta correcta probablemente no comparte
estilo con el original*. Premiar el parecido sería premiar al sistema por ignorar
al usuario.

## Grado 2 — Cumple claramente el pedido respecto de la imagen original

*(El renglón de arriba es el de `la evaluación § Escala` y **ya está decidido**: no
se cambia. Lo que falta es la definición operativa.)*

REEMPLAZAR: cuándo exactamente se pone 2. Escribilo de forma que dos
personas que lean esto solas pongan el mismo número. Si para decidirlo hace
falta mirar algo que la planilla no muestra, decilo acá.

## Grado 1 — Cumple parcialmente o es defendible

*(El renglón de arriba es el de `la evaluación § Escala` y **ya está decidido**: no
se cambia. Lo que falta es la definición operativa.)*

REEMPLAZAR: cuándo exactamente se pone 1. Escribilo de forma que dos
personas que lean esto solas pongan el mismo número. Si para decidirlo hace
falta mirar algo que la planilla no muestra, decilo acá.

## Grado 0 — No cumple

*(El renglón de arriba es el de `la evaluación § Escala` y **ya está decidido**: no
se cambia. Lo que falta es la definición operativa.)*

REEMPLAZAR: cuándo exactamente se pone 0. Escribilo de forma que dos
personas que lean esto solas pongan el mismo número. Si para decidirlo hace
falta mirar algo que la planilla no muestra, decilo acá.

## Ejemplos

*(Al menos uno por grado. Un ejemplo con el par concreto —imagen, pedido, obra—
vale más que una definición más larga: es lo que se mira cuando hay duda.)*

| grado | imagen original | pedido | obra recuperada | por qué |
|---|---|---|---|---|
| 2 | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR |
| 1 | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR |
| 0 | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR | REEMPLAZAR |

## Desempate

*(La regla para cuando el juicio queda entre dos grados. la evaluación fija quién
resuelve un desacuerdo **entre anotadores** —el tercer integrante, y ese desempate
es **una anotación más**, no una corrección del consenso—; lo que falta acá es la
regla para la duda de **una sola persona**.)*

REEMPLAZAR: entre 2 y 1, ¿cuál se elige? Entre 1 y 0, ¿cuál? Y qué se hace con
los casos que la evaluación ya nombra y que van a aparecer:

- la obra cumple el pedido pero **es la misma escena** que la original;
- el pedido es **ambiguo** (*«con más vida»*) y hay dos lecturas razonables;
- el pedido es **cultural** y la obra es de la cultura pedida pero no del tema;
- la obra **no tiene** el dato con el que se juzgaría (sin `culture`, sin fecha).

---

## Procedencia (no se edita)

- Identidad: `rubrica_id` = la **designación declarada** del front matter
  (la tabla de identidades, clase 3). **No es un hash del texto, a propósito**: una
  coma corregida daría otra rúbrica y descartaría los ~540 juicios, y la clase 3
  existe para que ese precio lo decida una persona y no un diff.
- La escala (tres valores) es compartida; **lo que significa cada valor lo fija
  este archivo** (el núcleo compartido).
- Validar: `poetry run tp3-anotar rubrica`
