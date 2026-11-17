# Rúbrica de las consultas mixtas

Se escribe **antes de correr** las 10 consultas de `consultas-g2-mixto.toml`, para que no se
ajuste mirando resultados. No forma parte del pool ni del κ del set de 30: es un instrumento
aparte y se declara así en el informe.

## Qué se juzga

Cada par (consulta mixta, obra recuperada) tiene tres componentes:

| | Componente | Cómo se verifica |
|---|---|---|
| A | **Cambio visual**: la obra tiene el cambio cromático pedido respecto de la imagen original («más cálido», «colores apagados»…). | Mirando la obra al lado de la imagen original. |
| B | **Restricción**: la obra es de la cultura, técnica, período o época que el pedido nombra. | **En los metadatos** (`culture`, `medium`, `year`), no a ojo. |
| C | **Adjetivo vago**: la obra transmite el rasgo («más calma», «más austero»). | A ojo. Es subjetivo: **no cuenta para el grado**, solo desempata y se anota aparte. |

## Grado

- **2**: cumple A **y** B.
- **1**: cumple exactamente una de las dos.
- **0**: no cumple ninguna.

`c06-mixto` es el control y no nombra restricción: se juzga con A y C. 2 si cumple las dos,
1 si cumple una, 0 si ninguna.

B es la parte dura y verificable. Si la obra no tiene `year` o `medium`, B se juzga
«no cumple» y se anota, no se adivina.

## Cómo se juzga

1. Dos personas juzgan por separado, sin saber qué configuración recuperó cada obra.
2. Una IA juzga después, con este mismo texto y sin ejemplos del set.
3. Se mide el acuerdo (κ) **sobre estos pares** y se publica aparte. Con ~50 pares por
   configuración es una estimación gruesa, y se dice así.
4. Los CSV con columnas `consulta_id`, `artwork_id`, `grado` entran al análisis con
   `--juicios-extra`.

## Límites que van al informe

- Son 10 consultas: los intervalos de confianza son anchos y no alcanzan para afirmar
  diferencias chicas entre configuraciones.
- La restricción «al óleo» es poco selectiva: la técnica óleo cubre el 42 % del índice
  (1.085 de 2.605 obras). En `c01`, `c04` y `c05` el filtro que pesa es el período.
- El control (`c06`) es una sola consulta: ilustra el contraste, no lo mide.
