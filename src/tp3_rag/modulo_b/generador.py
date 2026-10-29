from __future__ import annotations

import json
from typing import Mapping, Sequence

from tp3_rag.dominio.explicacion import Explicación, ViolaciónDelContratoDeSalida
from tp3_rag.modulo_b.etapa import Etapa, EstadoDeConsulta

PROMPT_ID = "generador-es-afirmaciones-tipadas"
VERSIÓN_DEL_PROMPT = "2"

SISTEMA = """\
Sos un asistente de un buscador de obras de arte. Recibís la imagen que subió una \
persona, su pedido en lenguaje natural, y cinco obras candidatas de un catálogo de museo, \
cada una con fragmentos de texto del museo identificados por un `fragment_id`.

Tu tarea: por cada obra, explicar en ESPAÑOL por qué la obra responde al pedido.

REGLAS, y son obligatorias:

0. El campo `tipo` de una afirmación admite EXACTAMENTE DOS valores: "dato" y
   "observacion". No hay un tercero. En particular, "abstencion" NO es un tipo de
   afirmación: la abstención es un campo de la OBRA (ver la regla 3 y el esquema).
1. Toda afirmación sobre la obra, el artista, la procedencia o la historia es de tipo
   "dato" y DEBE llevar en `refs` el o los `fragment_id` de los fragmentos que la
   respaldan. Un "dato" sin `refs` es inválido.
2. Las observaciones sobre lo que se ve —paleta, pincelada, composición, parecido con la
   imagen que subió la persona— son de tipo "observacion" y van SIN `refs`, porque no
   salen de ningún fragmento: son tuyas. Una "observacion" con `refs` es inválida.
3. Si los fragmentos no alcanzan para justificar por qué la obra cumple el pedido,
   escribí el campo `abstencion` DE LA OBRA —no una afirmación— en vez de inventar.
   Abstenerse es una respuesta correcta; inventar no. Una obra puede tener `abstencion`
   y además algunas afirmaciones, o solo `abstencion`.
4. En `refs` va el `fragment_id` LITERAL, tal como aparece en la evidencia. NUNCA un
   número, ni `[1]`, ni un índice. La numeración de las referencias la pone otra pieza.
5. Entre 2 y 4 afirmaciones por obra. En español, sin traducir ni citar textualmente el
   fragmento: el texto citado lo agrega otra pieza, vos solo lo referenciás.
6. No afirmes nada que no esté en los fragmentos de ESA obra. No uses conocimiento
   propio sobre el artista o el período: si no está en la evidencia, es una observación
   o es una abstención.

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma:

{
  "obras": [
    {
      "artwork_id": "<el artwork_id tal como se te dio>",
      "afirmaciones": [
        {"texto": "...", "tipo": "dato", "refs": ["<fragment_id>"]},
        {"texto": "...", "tipo": "observacion", "refs": []}
      ],
      "abstencion": "<solo si corresponde; si no, omitilo. NO es un valor de `tipo`>"
    }
  ]
}

Las cinco obras tienen que estar en `obras`, en el mismo orden en que se te dieron.\
"""

def _ficha(metadata: Mapping) -> str:
    campos = ("title", "artist", "date_display", "medium", "style", "culture", "museum")
    partes = [f"{c}: {metadata[c]}" for c in campos if metadata.get(c)]
    return " · ".join(partes)

def armar_mensajes(estado: EstadoDeConsulta) -> list[dict]:
    evidencia = dict(estado.evidencia or {})
    bloques: list[str] = [f"Pedido de la persona: «{estado.entrada.pedido}»", ""]
    for i, artwork_id in enumerate(estado.top5 or (), start=1):
        candidata = next(
            (
                c
                for c in estado.fuentes_de_ranking.get("visual", ())
                if c.artwork_id == artwork_id
            ),
            None,
        )
        bloques.append(f"### Obra {i} · artwork_id: {artwork_id}")
        if candidata is not None:
            bloques.append(_ficha(candidata.metadata))
        fragmentos = evidencia.get(artwork_id) or ()
        if not fragmentos:
            bloques.append(
                "(sin fragmentos de evidencia para esta obra — corresponde `abstencion`)"
            )
        for f in fragmentos:
            bloques.append(f"- fragment_id: {f.fragment_id}  [section: {f.section}]")
            bloques.append(f"  {f.texto}")
        bloques.append("")
    return [
        {"rol": "sistema", "texto": SISTEMA},
        {"rol": "usuario", "texto": "\n".join(bloques)},
    ]

class Generador(Etapa):
    NOMBRE = "generador"
    PRECONDICIONES = ("top5", "evidencia")
    CLAVES_DE_CONFIGURACIÓN = ("bindings.generador",)
    DEGRADA = True
    POLÍTICA = (
        "**salida sin explicación**: 5 obras con metadata, link y técnica. Degradación "
        "visible. Es lo que el criterio de respuesta garantizada compromete: *30/30 devuelven 5 obras*."
    )

    def __init__(self, *, binding) -> None:
        self.binding = binding
        if binding.prompt_id != PROMPT_ID or binding.version_del_prompt != VERSIÓN_DEL_PROMPT:
            raise ViolaciónDelContratoDeSalida(
                f"`[bindings.generador]` declara "
                f"`prompt_id = {binding.prompt_id!r}` v{binding.version_del_prompt} y este "
                f"módulo tiene {PROMPT_ID!r} v{VERSIÓN_DEL_PROMPT}. **El prompt es parte de "
                f"la identidad del binding**: si el texto y el id pueden discrepar, "
                f"dos corridas con el mismo `config_id` pudieron correr con prompts "
                f"distintos, y el `config_id` dejaría de reproducir una fila de la tabla."
            )
        self.ultima_respuesta = None
        self.n_intentos = 0

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        mensajes = armar_mensajes(estado)
        último: Exception | None = None
        for intento in range(1, int(self.binding.reintentos) + 2):
            respuesta = self.binding.invocar(
                mensajes, imagenes=[estado.entrada.datos_de_imagen]
            )
            self.ultima_respuesta = respuesta
            self.n_intentos = intento
            try:
                explicaciones = self._parsear(respuesta, tuple(estado.top5 or ()))
            except ViolaciónDelContratoDeSalida as e:
                último = e
                if intento > int(self.binding.reintentos):
                    raise
                continue
            return estado.con(explicacion=explicaciones)
        raise último  # pragma: no cover - el bucle siempre devuelve o levanta

    def _parsear(self, respuesta, obras: Sequence[str]) -> tuple[Explicación, ...]:
        try:
            crudo = respuesta.json()
        except json.JSONDecodeError as e:  # pragma: no cover - el puerto ya reintentó
            raise ViolaciónDelContratoDeSalida(
                f"el generador devolvió algo que no parsea ({e}). el contrato de los puertos ya reintentó hasta "
                f"`binding.reintentos`; acá es **falla de etapa** y la política de la tabla de degradación por etapa es "
                f"*salida sin explicación: 5 obras con metadata, link y técnica*."
            ) from None
        if not isinstance(crudo, dict) or "obras" not in crudo:
            raise ViolaciónDelContratoDeSalida(
                f"el generador devolvió un JSON sin la clave `obras`: "
                f"{list(crudo)[:5] if isinstance(crudo, dict) else type(crudo).__name__}. "
                f"El esquema está en el prompt del sistema y es parte del binding."
            )
        por_obra = {}
        for d in crudo["obras"]:
            explicación = Explicación.desde_json(d)
            por_obra[explicación.artwork_id] = explicación

        desconocidas = sorted(set(por_obra) - set(obras))
        if desconocidas:
            raise ViolaciónDelContratoDeSalida(
                f"el generador explicó obras que no están en el top-5: {desconocidas}. Es "
                f"la evidencia registrada en la corrida por el otro extremo —*sin esto, el generador puede citar evidencia "
                f"de una obra que no está en la salida*— y acá sería peor: una obra "
                f"explicada que el usuario no recibe."
            )
        salida: list[Explicación] = []
        for artwork_id in obras:
            explicación = por_obra.get(artwork_id)
            if explicación is None:
                explicación = Explicación(
                    artwork_id=artwork_id,
                    abstencion=(
                        "El modelo no produjo una explicación para esta obra. Se emite con "
                        "abstención en vez de omitirla: el criterio de respuesta garantizada compromete 5 obras, y una obra "
                        "sin afirmaciones y sin abstención es indistinguible de una obra "
                        "que nadie intentó explicar (el descarte contado de afirmaciones)."
                    ),
                )
            salida.append(explicación)
        return tuple(salida)

    def a_json(self) -> dict:
        efectivo = getattr(self.ultima_respuesta, "muestreo_efectivo", None)
        return {
            "etapa": self.NOMBRE,
            "binding": self.binding.a_json(),
            "muestreo_efectivo": dict(efectivo) if efectivo else None,
            "modelo_efectivo": getattr(self.ultima_respuesta, "modelo_efectivo", None),
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "idioma": "es",
            "nota": (
                "la regla de que un dato lleva referencia y que el generador emite ids y no números **no están en el prompt: están en el tipo**. Un `dato` sin "
                "`refs`, una `observacion` con `refs` y una `ref` numérica no se pueden "
                "construir (`dominio/explicacion.py`). Lo que el prompt pide es la calidad; "
                "lo que el tipo garantiza es la forma."
            ),
        }
