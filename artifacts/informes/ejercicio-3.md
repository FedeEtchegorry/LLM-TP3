# Ejercicio 3 — Responsible AI, Safety y puesta en producción

**Grupo 2 · 73.69 Large Language Models · ITBA, 2026**

El sistema recibe una imagen y un pedido en lenguaje natural y devuelve cinco pinturas del Art
Institute of Chicago (AIC) y del Cleveland Museum of Art (CMA), cada una con una explicación
citada a textos de los museos. La recuperación es una cascada: si el pedido nombra una cultura,
un filtro por metadata restringe la búsqueda a esas obras; el pedido conceptual define cuáles son
elegibles, la similitud visual de CLIP las ordena y un reranker multimodal decide el top-5. El corpus indexado tiene 2.605 obras en 6.232 fragmentos, y la evaluación usa 30 consultas
sobre imágenes que no están en el índice. La consigna pide definir, no implementar: en cada punto
se separa lo que **ya está construido y medido** de lo que es **propuesta**.

## 1. Responsible AI y Safety

**Explicaciones inventadas.** Es el riesgo principal de un sistema que redacta. *Construido:* el
modelo no elige las obras (el top-5 sale del índice), cada afirmación de tipo dato debe citar un
fragmento recuperado, las citas a fragmentos inexistentes se descartan y, si la evidencia no
alcanza, el sistema se abstiene: lo hizo en 51 de las 150 obras devueltas por la configuración
final, 37 de ellas en pedidos de cambio visual, porque los textos de los museos casi no hablan de
color.
*Límite conocido:* las observaciones sin cita no tienen ese control. Encontramos una explicación
que atribuía una obra coreana a Japón a partir del texto de procedencia; la causa era que la ficha
de la obra no llegaba al generador, y se corrigió. *Propuesta:* medir Faithfulness sobre una
muestra de respuestas en producción y exigir cita también para las afirmaciones de origen y fecha.

**Prompt injection.** La imagen del usuario llega a tres llamadas a un LLM (pre-retrieval,
reranker y generación) y los textos de los museos entran al prompt del generador; cualquiera de
los dos puede traer instrucciones, incluso como texto oculto en la imagen. *Construido:* solo una
defensa estructural: la salida es un JSON con contrato y el modelo no puede agregar obras ni
fuentes que el índice no devolvió. *Propuesta:* guardrails de entrada (tipo y tamaño de la imagen,
detección de texto incrustado, límite de longitud del pedido), delimitar imagen y fragmentos como
datos y no como instrucciones, y un guardrail de salida que verifique el formato y el idioma.

**Contaminación de la base (data poisoning).** Si alguien pudiera insertar textos en el índice,
controlaría las citas. *Construido:* el índice se arma en un proceso offline que solo lee las APIs
oficiales de los dos museos, y una variante no se registra si no pasa cuatro invariantes, entre
ellas que las imágenes de evaluación no entren. *Propuesta:* toda fuente nueva se valida antes de
incorporarse y se versiona, para poder volver atrás.

**Sesgo de representación.** Está medido sobre nuestro corpus. El 53,9 % de las pinturas de CMA
son anónimas, y el sesgo más fuerte lo introdujimos nosotros: exigir que la obra tenga descripción
(sin texto no hay cita) deja una tasa de ingreso de entre 0 % y 83,6 % según la cultura, y 54 de
las 156 etiquetas culturales desaparecen. *Construido:* en los 10 pedidos culturales, las obras
de la cultura pedida en el top-5 pasan de 0,6 de 5 en la versión 1 a 4,1 con la cascada y a 5 de
5 con el filtro por metadata. Ese 5 de 5 vale por construcción (es un filtro duro) y solo para las
culturas que el filtro reconoce. *Propuesta:* publicar la distribución del corpus por cultura y
avisar "sin obras suficientes" cuando la cultura pedida no se reconoce o tiene menos de 20 obras:
hoy, en ese caso, la búsqueda corre sin restricción y el top-5 puede traer otras culturas.

**Derechos y atribución.** Casi todo el material es CC0, pero la descripción de AIC es CC-BY 4.0:
son 638 fragmentos. *Construido:* la licencia viaja con cada fragmento, cada obra se muestra con
museo y enlace, y el sistema falla en vez de publicar una cita sin atribución.

## 2. Estrategias de puesta en producción

**Despliegue.** *Construido:* la indexación es un proceso offline y el servicio solo lee un
ChromaDB persistente, así que nunca se llama a los museos al responder. *Propuesta:* una API que
recibe imagen y pedido, empaquetada en un contenedor, con el índice versionado.

**Costo y latencia.** *Medido:* los tokens se registran por etapa en cada corrida. La
configuración final cuesta **USD 0,028 por consulta** a precio de lista: 0,024 del reranker, 0,003
del generador y 0,001 del pre-retrieval. Sin reranker cuesta 0,004. El reranker concentra el 86 %
del costo y lleva la latencia media de 9 a 37 segundos, porque hace 20 llamadas en serie de 1,4 s
cada una. Mostrarle 40 candidatas duplica el costo (USD 0,052) y lleva la latencia a 65 segundos
sin una mejora robusta. *Propuesta:* lanzar las llamadas del reranker en paralelo (son
independientes) y cachear embeddings y planes de consultas repetidas.

**Monitoreo y privacidad.** *Construido:* las trazas van a LangSmith y los tokens quedan además en
el artefacto de cada corrida. De las consultas de usuarios no se guarda la imagen, solo su hash, y
se trazan sin capturar las entradas. La imagen y el pedido sí viajan al proveedor del modelo, y el
usuario tiene que saberlo. *Propuesta:* alertas sobre la tasa de abstención, la latencia y el
costo diario, y límites de uso por usuario.

**Cambio de distribución.** Evaluamos con imágenes de museo; las fotos de usuarios (encuadre, luz,
reflejos) son otra distribución y CLIP puede comportarse distinto. *Propuesta:* sumar fotos reales
al set de evaluación y vigilar la similitud del primer resultado como señal de corrimiento. Si
cambia el modelo de embeddings, se reindexa todo.

**Calidad continua.** *Construido:* el set de 30 consultas con sus juicios permite recalcular la
tabla de métricas sin llamar a ningún modelo, y cada etapa tiene una degradación definida y
probada inyectando fallas (si cae el reranker se entrega el orden de CLIP; si no hay evidencia, el
sistema se abstiene). *Propuesta:* correr esa evaluación antes de cada cambio como condición para
desplegar, y ampliar el set: 30 consultas alcanzan para ver tendencias, no diferencias chicas.

*Las cifras salen de las corridas guardadas en `artifacts/`. La latencia suma 1,9 s del
pre-retrieval, medidos en las corridas que generaron el plan: las finales lo reusaron. De los 634 juicios de relevancia, 494 son de
anotadores de IA y 140 están hechos a mano; el detalle y sus límites están en el Ejercicio 4.*
