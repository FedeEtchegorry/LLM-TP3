"""Prefiltro de pedidos vagos: corre antes de la cadena y solo sugiere."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from tp3_rag.nucleo.errores import ErrorDelProyecto

PROMPT_ID = "aclarador-pedido-vago"
# v1 y v2 eran de dos niveles (vago o no)
VERSIÓN_DEL_PROMPT = "3"
NIVELES = ("concreto", "medio", "vago")

SISTEMA = """\
Sos el primer filtro de un buscador multimodal de pinturas. Recibís la imagen que subió una \
persona y su pedido en español. Tu tarea es decir cuán concreto es el pedido y, solo si es \
vago, proponer una versión que se le parezca lo más posible.

Clasificá el pedido en uno de tres niveles:

- "concreto": nombra algo que se puede ver o verificar en una pintura: una cultura o un país, \
una época, una técnica o un soporte, un motivo o un objeto, un color o una paleta determinados. \
Ejemplos: "con un cielo de tormenta", "pintado en Egipto", "en técnica de acuarela", \
"de la época barroca".
- "medio": pide cambiar, en términos relativos, un atributo visible de la imagen (el color, la \
luz, el contraste, el fondo, el trazo, la cantidad o el tamaño de las figuras) sin decir hacia \
dónde. Ya dice qué mirar, así que se puede buscar tal cual. Ejemplos: "con más contraste", \
"con la luz más suave", "con más figuras".
- "vago": su única guía es una cualidad abstracta, un carácter, una emoción o un juicio de \
valor, y no nombra ningún atributo visible. Hay varias lecturas razonables y buscar sin elegir \
una da resultados al azar. Ejemplos: "más elegante", "más misterioso", "más moderno", \
"más dramático".

REGLAS, y son obligatorias:

1. En la duda entre dos niveles, elegí el más concreto.
2. Si el pedido nombra una cultura, un país, una época, una técnica, un motivo o un color \
determinado, es "concreto", aunque la imagen no coincida con eso. Nunca juzgues el pedido por \
lo que muestra la imagen.
3. Si el nivel es "vago", "sugerencia" es el pedido original con sus mismas palabras, más una \
cláusula corta que dice cómo se vería esa cualidad en una pintura. Ejemplo: "más moderno" se \
vuelve "más moderno, con formas más simples y colores planos". Es una sola oración y suena a \
algo que la persona escribiría.
4. La sugerencia conserva todo lo que el pedido ya decía. No agrega motivos, objetos, lugares ni \
momentos del día que la persona no mencionó. Mirá la imagen solo para elegir qué atributo visible \
(luz, color, trazo, composición) acompaña la cualidad, nunca para cambiar el tema.
5. Si el nivel es "concreto" o "medio", "sugerencia" repite el pedido tal cual.
6. "motivo" dice en una oración corta por qué elegiste ese nivel.
7. No inventes datos de la imagen: lo que no se ve en ella, no se afirma.

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma, sin texto alrededor:

{"nivel": "concreto" o "medio" o "vago", "motivo": "...", "sugerencia": "..."}\
"""


class AclaraciónInválida(ErrorDelProyecto):
    pass


@dataclass(frozen=True)
class Aclaración:
    nivel: str
    motivo: str
    sugerencia: str
    uso: Mapping[str, Any] = field(default_factory=dict)

    @property
    def vago(self) -> bool:
        return self.nivel == "vago"

    def a_json(self) -> dict:
        return {
            "nivel": self.nivel,
            "motivo": self.motivo,
            "sugerencia": self.sugerencia,
            "uso": dict(self.uso),
        }


def armar_mensajes(pedido: str) -> list[dict]:
    return [
        {"rol": "sistema", "texto": SISTEMA},
        {
            "rol": "usuario",
            "texto": f"Pedido de la persona: «{pedido}»\n\nLa imagen es la que subió la persona.",
        },
    ]


def _parsear(respuesta, pedido: str) -> tuple[str, str, str]:
    try:
        crudo = respuesta.json()
    except json.JSONDecodeError as e:
        raise AclaraciónInválida(f"el aclarador devolvió algo que no parsea ({e}).") from None
    if not isinstance(crudo, dict):
        raise AclaraciónInválida(f"el aclarador devolvió un {type(crudo).__name__} y no un objeto.")
    faltan = sorted({"nivel", "motivo", "sugerencia"} - set(crudo))
    if faltan:
        raise AclaraciónInválida(f"el aclarador devolvió un JSON sin {faltan}.")
    nivel = crudo["nivel"]
    if nivel not in NIVELES:
        raise AclaraciónInválida(f"`nivel` vino como {nivel!r} y tiene que ser uno de {list(NIVELES)}.")
    motivo = str(crudo["motivo"] or "").strip()
    sugerencia = str(crudo["sugerencia"] or "").strip()
    if not sugerencia:
        raise AclaraciónInválida("`sugerencia` vino vacía.")
    if nivel != "vago":
        sugerencia = pedido
    return nivel, motivo, sugerencia


def aclarar(binding, *, pedido: str, datos_de_imagen: bytes) -> Aclaración:
    mensajes = armar_mensajes(pedido)
    tokens_in = tokens_out = 0
    último: Exception | None = None
    for _ in range(int(binding.reintentos) + 1):
        respuesta = binding.invocar(mensajes, imagenes=[datos_de_imagen])
        tokens_in += int(respuesta.uso.get("tokens_in", 0) or 0)
        tokens_out += int(respuesta.uso.get("tokens_out", 0) or 0)
        try:
            nivel, motivo, sugerencia = _parsear(respuesta, pedido)
        except AclaraciónInválida as e:
            último = e
            continue
        return Aclaración(
            nivel=nivel,
            motivo=motivo,
            sugerencia=sugerencia,
            uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
        )
    raise AclaraciónInválida(f"sin una aclaración válida después de {int(binding.reintentos) + 1} intentos: {último}")
