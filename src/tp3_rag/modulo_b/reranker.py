from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tp3_rag.modulo_b.etapa import Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto

PROMPT_ID = "reranker-score-0-10-un-candidato"
VERSIÓN_DEL_PROMPT = "1"

SCORE_MÍNIMO, SCORE_MÁXIMO = 0, 10

CAMPOS_DE_LA_FICHA = (
    "title",
    "artist",
    "date_display",
    "medium",
    "style",
    "culture",
    "museum",
)

SISTEMA = """\
Sos la etapa de reranking de un buscador multimodal de obras de arte. Recibís:

1. LA IMAGEN QUE SUBIÓ LA PERSONA (la primera imagen),
2. su PEDIDO en español, tal como lo escribió,
3. UNA obra candidata del catálogo: su IMAGEN (la segunda) y su ficha de museo.

Tu tarea: puntuar DE 0 A 10 cuánto cumple ESA candidata el pedido de la persona, tomando la \
primera imagen como punto de partida, y justificarlo en una o dos oraciones en español.

CÓMO PUNTUAR, y la escala es la misma para todos los candidatos:

  0-2  no cumple el pedido: ni lo que el pedido nombra ni lo que la imagen aporta.
  3-4  cumple de costado: se parece a la imagen pero ignora lo que el pedido pide cambiar,
       o cumple el pedido y no tiene nada que ver con la imagen.
  5-6  cumple a medias: cumple lo que el pedido pide o lo que la imagen aporta, no las dos.
  7-8  cumple: lo que el pedido nombra está, y lo que el pedido no cambia se mantiene.
  9-10 cumple de lleno y es un resultado que la persona querría ver primero.

REGLAS, y son obligatorias:

1. Juzgá contra EL PEDIDO, no contra el parecido visual con la primera imagen. El parecido
   cuenta solo en lo que el pedido NO pide cambiar.
2. Si el pedido nombra una cultura, un país, una región o un período, una candidata que NO
   es de eso NO puede pasar de 4, aunque se parezca mucho a la imagen.
3. Puntuá ESTA candidata sola, contra la escala de arriba. No la compares con otras
   candidatas: no las viste y no las vas a ver.
4. La justificación es en ESPAÑOL, concreta y breve: qué del pedido cumple y qué no. No
   repitas la ficha ni describas la obra entera.
5. No inventes datos de la obra: lo que no está en la ficha ni se ve en la imagen, no se
   afirma.

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma, sin texto alrededor:

{"score": <entero de 0 a 10>, "justificacion": "..."}\
"""

class RerankInválido(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class CandidataPuntuada:
    artwork_id: str
    score_0_10: int | None
    justificacion: str
    rango_fusionado: int
    estado: str = "puntuado"
    uso: Mapping[str, Any] = field(default_factory=dict)
    n_intentos: int = 1

    @property
    def puntuada(self) -> bool:
        return self.estado == "puntuado"

    def a_json(self) -> dict:
        return {
            "artwork_id": self.artwork_id,
            "score_0_10": self.score_0_10,
            "justificacion": self.justificacion,
            "rango_fusionado": self.rango_fusionado,
            "estado": self.estado,
            "uso": dict(self.uso),
            "n_intentos": self.n_intentos,
        }

def _ficha(metadata: Mapping[str, Any]) -> str:
    partes = [f"{c}: {metadata[c]}" for c in CAMPOS_DE_LA_FICHA if metadata.get(c)]
    return " · ".join(partes) or "(el catálogo no publica ficha para esta obra)"

def armar_mensajes(estado: EstadoDeConsulta, *, artwork_id: str, metadata) -> list[dict]:
    return [
        {"rol": "sistema", "texto": SISTEMA},
        {
            "rol": "usuario",
            "texto": (
                f"Pedido de la persona: «{estado.entrada.pedido}»\n\n"
                f"Candidata · artwork_id: {artwork_id}\n"
                f"{_ficha(metadata or {})}\n\n"
                f"La primera imagen es la que subió la persona; la segunda es la de esta "
                f"candidata."
            ),
        },
    ]

def ordenar(puntuadas: Sequence[CandidataPuntuada]) -> tuple[CandidataPuntuada, ...]:
    return tuple(
        sorted(
            puntuadas,
            key=lambda c: (
                -(c.score_0_10 if c.score_0_10 is not None else -1),
                c.rango_fusionado,
            ),
        )
    )

class Reranker(Etapa):
    NOMBRE = "reranker"
    PRECONDICIONES = ("ranking_fusionado",)
    CLAVES_DE_CONFIGURACIÓN = (
        "bindings.reranker",
        "consulta.n_candidatos",
        "consulta.top_k",
    )
    DEGRADA = True
    POLÍTICA = (
        "se entrega el top-5 del ranking fusionado sin reordenar, más la "
        "degradación anotada; el fusionado queda con su score intacto. Es el único "
        "caso en que degradar una etapa entrega exactamente la fila anterior de la "
        "ablación. Una candidata sin puntaje degrada igual: ordenar mezclando "
        "«puntuado» con «no pude puntuarlo» sería publicar dos escalas en una "
        "columna, y los puntajes ya pagados quedan en el artefacto."
    )

    def __init__(self, *, binding, bytes_de_la_imagen, n_candidatos: int, top_k: int = 5):
        self.binding = binding
        if (
            binding.prompt_id != PROMPT_ID
            or str(binding.version_del_prompt) != VERSIÓN_DEL_PROMPT
        ):
            raise RerankInválido(
                f"`[bindings.reranker]` declara `prompt_id = {binding.prompt_id!r}` "
                f"v{binding.version_del_prompt} y este módulo tiene {PROMPT_ID!r} "
                f"v{VERSIÓN_DEL_PROMPT}. El prompt es parte de la identidad del binding: "
                f"si el texto y el id pueden discrepar, dos corridas con el mismo "
                f"`config_id` pudieron correr con prompts distintos, y el `config_id` "
                f"dejaría de reproducir su fila de la tabla de ablación."
            )
        if bytes_de_la_imagen is None:
            raise ErrorDeConfiguración(
                "el `reranker` necesita leer la imagen de la candidata y no se le inyectó "
                "ningún lector. La llamada lleva la imagen que subió la persona más la "
                "imagen de la candidata y su ficha: son las dos imágenes sobre las que se "
                "midió el costo para elegir el modelo. Sin la de la candidata esto no es "
                "reranking multimodal, es reranking por ficha, que es otra técnica."
            )
        self.bytes_de_la_imagen = bytes_de_la_imagen
        self.n_candidatos = int(n_candidatos)
        if self.n_candidatos < 1:
            raise ErrorDeConfiguración(
                f"`consulta.n_candidatos = {n_candidatos}` tiene que ser >= 1: es cuántas "
                f"candidatas se puntúan, y de ahí sale el conteo de llamadas que se "
                f"publica."
            )
        self.top_k = int(top_k)
        if self.top_k < 1 or self.top_k > self.n_candidatos:
            raise ErrorDeConfiguración(
                f"`consulta.top_k = {top_k}` tiene que estar en [1, n_candidatos="
                f"{self.n_candidatos}]: el reranker puntúa N y se queda con las primeras, "
                f"así que pedir 5 de 3 entregaría 3 obras donde la consulta compromete 5."
            )
        self.ultima_respuesta = None

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        fusionado = tuple(estado.ranking_fusionado or ())
        if not fusionado:
            degradó_la_fusion = any(d.etapa == "fusionador" for d in estado.degradaciones)
            raise RerankInválido(
                "el `ranking_fusionado` no está y el reranker no tiene qué puntuar. "
                + (
                    "El `fusionador` degradó, así que esto es la cascada: sin ranking "
                    "fusionado no hay candidatas que puntuar, y lo que queda es el top-5 "
                    "de la rama visual, que es el piso. La degradación que esto produce "
                    "es espontánea porque nadie la inyectó, y eso saca la corrida de la "
                    "tabla de ablación: no es ninguna de las filas, es un punto que la "
                    "matriz no tiene."
                    if degradó_la_fusion
                    else "El `fusionador` no dejó degradación, así que esto no es un "
                    "reranking pobre sino un artefacto mal formado: una etapa que no "
                    "hizo nada y no registró nada. El armador verificó la cadena antes "
                    "de correr, así que la clave tendría que estar."
                )
            )
        candidatas = fusionado[: self.n_candidatos]
        metadata_por_obra = self._metadata_de(estado)

        puntuadas: list[CandidataPuntuada] = []
        for rango, fila in enumerate(candidatas, start=1):
            puntuadas.append(
                self._puntuar(
                    estado,
                    artwork_id=fila.artwork_id,
                    rango=rango,
                    metadata=metadata_por_obra.get(fila.artwork_id) or {},
                )
            )
        ordenadas = ordenar(puntuadas)
        nuevo = estado.con(rerank=tuple(c.a_json() for c in ordenadas))

        sin_puntaje = [c for c in ordenadas if not c.puntuada]
        if sin_puntaje:
            return nuevo.degradar(
                self.NOMBRE,
                causa=(
                    f"{len(sin_puntaje)} de {len(ordenadas)} candidatas quedaron sin "
                    f"puntaje después de agotar los reintentos del binding: "
                    f"{[c.artwork_id for c in sin_puntaje][:3]}. Ordenar mezclando "
                    f"«puntuado» con «no pude puntuarlo» sería publicar dos escalas en "
                    f"una columna, así que se entrega el top-5 del fusionado y los "
                    f"puntajes ya pagados quedan en `rerank`."
                ),
                politica=self.POLÍTICA,
            )
        return nuevo.proyectar_top5(
            [c.artwork_id for c in ordenadas[: self.top_k]], por=self.NOMBRE
        )

    def _metadata_de(self, estado: EstadoDeConsulta) -> dict[str, Mapping[str, Any]]:
        fichas: dict[str, Mapping[str, Any]] = {}
        for nombre in ("texto", *sorted(estado.fuentes_de_ranking), "visual"):
            for c in estado.fuentes_de_ranking.get(nombre, ()):
                if c.metadata:
                    fichas[c.artwork_id] = c.metadata
        return fichas

    def _puntuar(
        self, estado: EstadoDeConsulta, *, artwork_id: str, rango: int, metadata
    ) -> CandidataPuntuada:
        try:
            imagen_del_candidato = self.bytes_de_la_imagen(artwork_id)
        except Exception as e:
            return CandidataPuntuada(
                artwork_id=artwork_id,
                score_0_10=None,
                justificacion=(
                    f"sin puntaje: no se pudo leer la imagen del candidato "
                    f"({type(e).__name__}: {e}). La llamada no se hizo, así que no hay "
                    f"tokens que contar."
                ),
                rango_fusionado=rango,
                estado="sin_puntaje",
                n_intentos=0,
            )

        mensajes = armar_mensajes(estado, artwork_id=artwork_id, metadata=metadata)
        tokens_in = tokens_out = 0
        último: Exception | None = None
        intento = 0
        for intento in range(1, int(self.binding.reintentos) + 2):
            try:
                respuesta = self.binding.invocar(
                    mensajes,
                    imagenes=[estado.entrada.datos_de_imagen, imagen_del_candidato],
                )
            except Exception as e:
                último = e
                break
            self.ultima_respuesta = respuesta
            tokens_in += int(respuesta.uso.get("tokens_in", 0) or 0)
            tokens_out += int(respuesta.uso.get("tokens_out", 0) or 0)
            try:
                score, justificación = self._parsear(respuesta)
            except RerankInválido as e:
                último = e
                if intento > int(self.binding.reintentos):
                    break
                continue
            return CandidataPuntuada(
                artwork_id=artwork_id,
                score_0_10=score,
                justificacion=justificación,
                rango_fusionado=rango,
                uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
                n_intentos=intento,
            )
        return CandidataPuntuada(
            artwork_id=artwork_id,
            score_0_10=None,
            justificacion=(
                f"sin puntaje después de {intento} intento(s) de la etapa: "
                f"{type(último).__name__}: {último}"
            ),
            rango_fusionado=rango,
            estado="sin_puntaje",
            uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
            n_intentos=intento,
        )

    def _parsear(self, respuesta) -> tuple[int, str]:
        try:
            crudo = respuesta.json()
        except json.JSONDecodeError as e:  # pragma: no cover - el puerto ya reintentó
            raise RerankInválido(
                f"el reranker devolvió algo que no parsea ({e}). El puerto ya reintentó "
                f"hasta el límite del binding."
            ) from None
        if not isinstance(crudo, dict):
            raise RerankInválido(
                f"el reranker devolvió un JSON que no es un objeto "
                f"({type(crudo).__name__}). El esquema está en el prompt del sistema y es "
                f"parte del binding."
            )
        faltan = sorted({"score", "justificacion"} - set(crudo))
        if faltan:
            raise RerankInválido(
                f"el reranker devolvió un JSON sin {faltan} (vinieron {sorted(crudo)[:6]}). "
                f"Hacen falta los dos: un puntaje sin justificación no explica el "
                f"ranking, y una justificación sin puntaje no lo ordena."
            )
        crudo_score = crudo["score"]
        if isinstance(crudo_score, bool) or not isinstance(crudo_score, (int, float)):
            raise RerankInválido(
                f"`score` vino como {type(crudo_score).__name__} "
                f"({crudo_score!r}) y tiene que ser un número de 0 a 10. Un JSON válido "
                f"que no satisface el esquema se reintenta igual."
            )
        if float(crudo_score) != int(crudo_score):
            raise RerankInválido(
                f"`score = {crudo_score!r}` no es entero. La escala es de 0 a 10 entera: "
                f"un 7.5 no es más información, es otra escala."
            )
        score = int(crudo_score)
        if not SCORE_MÍNIMO <= score <= SCORE_MÁXIMO:
            raise RerankInválido(
                f"`score = {score}` está fuera de [{SCORE_MÍNIMO}, {SCORE_MÁXIMO}]. La "
                f"escala es la del prompt y es la misma para todos los candidatos: un "
                f"puntaje fuera de rango vuelve incomparables los N de esta consulta."
            )
        justificación = str(crudo["justificacion"] or "").strip()
        if not justificación:
            raise RerankInválido(
                "`justificacion` vino vacía. El puntaje y la justificación van los dos "
                "al estado y por lo tanto a la corrida: lo que explica el reranking es el "
                "par, no el número solo."
            )
        return score, justificación

    def a_json(self) -> dict:
        efectivo = getattr(self.ultima_respuesta, "muestreo_efectivo", None)
        return {
            "etapa": self.NOMBRE,
            "binding": self.binding.a_json(),
            "muestreo_efectivo": dict(efectivo) if efectivo else None,
            "modelo_efectivo": getattr(self.ultima_respuesta, "modelo_efectivo", None),
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "n_candidatos": self.n_candidatos,
            "top_k": self.top_k,
            "escala": [SCORE_MÍNIMO, SCORE_MÁXIMO],
            "imagenes_por_llamada": 2,
            "llamadas_al_llm_por_consulta": f"min(N={self.n_candidatos}, |fusionado|)",
            "desempate": "rango del ranking fusionado",
            "nota_de_p6": (
                "Las llamadas por consulta son dos más N, y la N sale de "
                "`consulta.n_candidatos` y no de una clave propia: dos hogares para el "
                "mismo dato serían dos valores que pueden discrepar justo en el número "
                "que se publica. El reranker puntúa min(N, |fusionado|), así que dos "
                "más N es una cota superior: se alcanza cuando el fusionado tiene al "
                "menos N filas, que con N candidatas visuales es verdad por "
                "construcción."
            ),
            "nota_de_un_candidato_por_llamada": (
                "Una candidata por llamada, y no es ineficiencia: meter cinco imágenes "
                "en un mismo mensaje reduciría las llamadas cinco veces, pero el modelo "
                "empieza a comparar las candidatas entre sí en vez de evaluar cada una "
                "contra el pedido. Y de eso depende el conteo de llamadas que se "
                "publica."
            ),
            "nota_del_pedido_crudo": (
                "Entra el pedido crudo y no la consulta reescrita: lo que se verifica es "
                "el cumplimiento de lo que la persona pidió. Es la única etapa con "
                "reranking que no consume el plan, y es lo que vuelve informativo el "
                "delta: si leyera la consulta reescrita estaría midiendo la reescritura "
                "otra vez."
            ),
            "nota_de_et3": (
                "Esta etapa agrega `rerank` y reordena el top-5; no pisa el "
                "`score_fusionado` ni los rangos de origen. Si los pisara, los dos "
                "rankings dejarían de ser legibles del mismo artefacto. El top-5 "
                "siempre es un subconjunto de lo que alguna etapa de ranking propuso, "
                "porque el reranker puntúa N y se queda con las primeras."
            ),
        }

def resumen_del_rerank(filas: Sequence[Mapping[str, Any]]) -> dict:
    puntuadas = [f for f in filas if f.get("estado") == "puntuado"]
    scores = [int(f["score_0_10"]) for f in puntuadas]
    return {
        "n_candidatos_puntuados": len(puntuadas),
        "n_sin_puntaje": len(filas) - len(puntuadas),
        "n_con_justificacion": sum(
            1 for f in puntuadas if str(f.get("justificacion") or "").strip()
        ),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_medio": round(sum(scores) / len(scores), 2) if scores else None,
        "scores_distintos": len(set(scores)),
        "tokens_in": sum(int((f.get("uso") or {}).get("tokens_in", 0) or 0) for f in filas),
        "tokens_out": sum(int((f.get("uso") or {}).get("tokens_out", 0) or 0) for f in filas),
    }

def movimiento_del_top5(
    ranking_fusionado: Sequence[Mapping[str, Any]],
    rerank: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> dict:
    antes = [f["artwork_id"] for f in ranking_fusionado][:top_k]
    después = [f["artwork_id"] for f in rerank][:top_k]
    return {
        "top5_fusionado": antes,
        "top5_reranqueado": después,
        "obras_que_entraron": [o for o in después if o not in antes],
        "obras_que_salieron": [o for o in antes if o not in después],
        "churn": len([o for o in después if o not in antes]),
        "mismo_orden": antes == después,
        "promovidas_desde_fuera_del_top5": [
            {"artwork_id": f["artwork_id"], "rango_fusionado": f.get("rango_fusionado")}
            for f in rerank[:top_k]
            if int(f.get("rango_fusionado") or 0) > top_k
        ],
    }
