from __future__ import annotations

import json

from tp3_rag.dominio.plan import PlanDeConsulta, PlanInválido
from tp3_rag.modulo_b.etapa import Etapa, EstadoDeConsulta
from tp3_rag.nucleo import identidad

PROMPT_ID = "planificador-pre-retrieval-dos-consultas"

PALABRAS_MÁXIMAS_VISUAL = 30

SISTEMA_V1 = """\
Sos la etapa de pre-retrieval de un buscador multimodal de obras de arte. Recibís la imagen \
que subió una persona y su pedido en español, y traducís ese pedido —que suele ser vago, \
relativo o ambiguo— en DOS consultas de búsqueda concretas PARA ESA OBRA EN PARTICULAR.

Las dos consultas alimentan dos buscadores distintos y por eso NO pueden ser la misma:

1. `consulta_visual` → va al TEXT ENCODER DE CLIP, que compara contra fotos de pinturas.
   Escribila como un CAPTION CORTO EN INGLÉS: lo que se vería en la obra buscada. Concreta y
   visual: motivo, colores, luz, composición, formato. MÁXIMO 25 PALABRAS. Nada de lenguaje
   curatorial, nada de nombres de museo, nada de fechas ni de metadatos.

2. `consulta_conceptual` → va a un buscador sobre TEXTO DE MUSEO EN INGLÉS (descripciones,
   fichas curatoriales, procedencias). Escribila EN INGLÉS con registro curatorial: tema,
   cultura, período, escuela, técnica, contexto. MÁXIMO 35 PALABRAS. Nada de describir
   píxeles ni colores.

REGLAS, y son obligatorias:

0. LAS DOS VAN EN INGLÉS, aunque el pedido esté en español. Los dos buscadores están
   entrenados sobre inglés y el corpus de museo está en inglés.
1. Las dos tienen que ser NO VACÍAS, DISTINTAS ENTRE SÍ y DISTINTAS del pedido crudo. No
   alcanza con traducir el pedido: traducirlo no es interpretarlo.
2. Son distintas POR SU REGISTRO, no por un sinónimo: una describe lo que se ve, la otra
   describe de qué trata la obra. Si una es la paráfrasis de la otra, están mal.
3. Mirá la imagen. El pedido suele ser RELATIVO a ella ("más cálido", "con más vida", "lo
   mismo pero en Japón"): las dos consultas describen LA OBRA BUSCADA, no la que subió.
   Tomá de la imagen lo que el pedido no cambia (el motivo, el formato, la técnica) y
   cambiá lo que el pedido pide cambiar.
4. Si el pedido nombra una cultura, un país o un período, ESO VA en `consulta_conceptual`
   con el término que usaría una ficha de museo en inglés.
5. No inventes el título, el autor ni la fecha de la obra de la imagen, y no los nombres.

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma, sin texto alrededor:

{"consulta_visual": "...", "consulta_conceptual": "..."}\
"""

SISTEMA_V2 = """\
Sos la etapa de pre-retrieval de un buscador multimodal de obras de arte. Recibís la imagen \
que subió una persona y su pedido en español, y traducís ese pedido —que suele ser vago, \
relativo o ambiguo— en DOS consultas de búsqueda concretas PARA LA OBRA QUE LA PERSONA \
BUSCA, que NO es la que subió.

ANTES DE ESCRIBIR, leé el pedido y fijate qué nombra explícitamente: ¿una cultura, un país, \
una región, un período, una técnica, un tema? Todo lo que el pedido nombre tiene que \
aparecer en tu respuesta. Lo que el pedido NO nombra, lo tomás de la imagen.

REGLA DE PRECEDENCIA, y es la más importante: SI EL PEDIDO Y LA IMAGEN SE CONTRADICEN, GANA \
EL PEDIDO. La imagen es el punto de partida, no el destino. Si la imagen es una escena de \
caza de ballenas europea y el pedido dice "en la pintura china", lo que se busca son \
ESCENAS DE AGUA CHINAS: la cultura la fija el pedido y el tema lo fija la imagen.

Las dos consultas alimentan dos buscadores distintos y por eso NO pueden ser la misma:

1. `consulta_visual` → va al TEXT ENCODER DE CLIP, que compara contra fotos de pinturas.
 Escribila como un CAPTION CORTO EN INGLÉS: lo que se vería en la obra buscada. Concreta y
 visual: motivo, colores, luz, composición, formato. MÁXIMO 25 PALABRAS. Nada de lenguaje
 curatorial, nada de nombres de museo, nada de fechas ni de metadatos.

2. `consulta_conceptual` → va a un buscador sobre TEXTO DE MUSEO EN INGLÉS (descripciones,
 fichas curatoriales, procedencias). Escribila EN INGLÉS con registro curatorial: tema,
 cultura, período, escuela, técnica, contexto. MÁXIMO 35 PALABRAS. Nada de describir
 píxeles ni colores.

REGLAS, y son obligatorias:

0. LAS DOS VAN EN INGLÉS, aunque el pedido esté en español. Los dos buscadores están
 entrenados sobre inglés y el corpus de museo está en inglés.
1. Las dos tienen que ser NO VACÍAS, DISTINTAS ENTRE SÍ y DISTINTAS del pedido crudo. No
 alcanza con traducir el pedido: traducirlo no es interpretarlo.
2. Son distintas POR SU REGISTRO, no por un sinónimo: una describe lo que se ve, la otra
 describe de qué trata la obra. Si una es la paráfrasis de la otra, están mal.
3. SI EL PEDIDO NOMBRA UNA CULTURA, UN PAÍS, UNA REGIÓN O UN PERÍODO, ESO VA EN LAS DOS
 CONSULTAS, con el término que usaría una ficha de museo en inglés ("Chinese", "Japanese",
 "Italian", "Indian", "French", "Mughal"). En `consulta_visual` va como adjetivo del
 motivo ("Chinese ink painting of..."); en `consulta_conceptual`, como cultura o escuela.
4. PROHIBIDO nombrar una cultura, un país, una región o un período QUE EL PEDIDO NO PIDIÓ.
 Si el pedido nombra una cultura, la de la imagen NO VA: no la menciones ni para
 contrastar. Describir la cultura de la imagen en vez de la pedida es el error más grave
 que podés cometer acá.
5. PROHIBIDO ESCRIBIR EN NEGATIVO. Nunca digas lo que la obra NO, ni "not related to",
 ni "unlike", ni "rather than": una consulta de búsqueda describe lo que se busca, y un
 buscador semántico trata "not related to Indian art" como SI pidiera arte de la India.
6. Mirá la imagen y tomá de ella SOLO lo que el pedido no cambia: el motivo, el formato, la
 técnica, la composición. Si el pedido es relativo ("más cálido", "con más vida", "lo
 mismo pero en Japón"), lo relativo se resuelve ACÁ y las consultas quedan ABSOLUTAS.
7. No inventes el título, el autor ni la fecha de la obra de la imagen, y no los nombres.

EJEMPLO del caso que más se falla. Imagen: una composición geométrica abstracta de líneas
negras y rectángulos de color. Pedido: "composiciones geométricas en el arte de la India".

 MAL → {"consulta_visual": "geometric composition with black lines and colored
          rectangles, minimalist abstract style",
          "consulta_conceptual": "abstract geometric painting influenced by early
          20th-century Western movements, not related to Indian art traditions"}
 (describe la cultura de la imagen, y encima en negativo)

 BIEN → {"consulta_visual": "Indian geometric ornamental painting with symmetrical
          patterns and bold colored fields",
          "consulta_conceptual": "Indian geometric and ornamental composition, symmetry and
          abstract pattern in South Asian painting and manuscript illumination"}

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma, sin texto alrededor:

{"consulta_visual": "...", "consulta_conceptual": "..."}\
"""

PROMPTS: dict[str, str] = {"1": SISTEMA_V1, "2": SISTEMA_V2}

VERSIÓN_DEL_PROMPT = "2"

def sistema(version: str) -> str:
    texto = PROMPTS.get(str(version))
    if texto is None:
        raise PlanInválido(
            f"`[bindings.planificador] version_del_prompt = {version!r}` y este módulo "
            f"tiene {sorted(PROMPTS)}. La versión **es ingrediente del `plan_id`**: "
            f"una versión sin texto produciría un artefacto citable cuyo prompt no se puede "
            f"leer. Las dos que hay son **la v1 —que queda citable porque su medición es un "
            f"resultado del informe— y la v2**, que es la que corre."
        )
    return texto

def armar_mensajes(estado: EstadoDeConsulta, *, version: str) -> list[dict]:
    return [
        {"rol": "sistema", "texto": sistema(version)},
        {"rol": "usuario", "texto": f"Pedido de la persona: «{estado.entrada.pedido}»"},
    ]

class PlanificadorDeConsulta(Etapa):
    NOMBRE = "planificador"
    PRECONDICIONES = ("entrada",)
    CLAVES_DE_CONFIGURACIÓN = ("bindings.planificador",)
    DEGRADA = True
    POLÍTICA = (
        "**pedido crudo al text encoder de CLIP** — que es exactamente el comportamiento de "
        "C1 en la rama visual. La clave `plan` queda ausente y `consulta_visual` cae "
        "al pedido, que es el mismo código que corre C1 (que omitir y fallar produzcan la misma forma)."
    )

    def __init__(self, *, binding, almacen_de_planes) -> None:
        self.binding = binding
        self.almacen = almacen_de_planes
        if binding.prompt_id != PROMPT_ID:
            raise PlanInválido(
                f"`[bindings.planificador]` declara `prompt_id = {binding.prompt_id!r}` y "
                f"este módulo tiene {PROMPT_ID!r}. **El prompt es parte de la identidad del "
                f"bindingy del `plan_id`**: si el texto y el id pueden "
                f"discrepar, dos planes con contenido distinto comparten id — que es lo que "
                f"la regla de que un id determina su contenido declara error duro en la mitad de clase 1 del híbrido."
            )
        self.version_del_prompt = str(binding.version_del_prompt)
        sistema(self.version_del_prompt)
        self.ultima_respuesta = None
        self.n_intentos = 0

    def plan_id_de(self, estado: EstadoDeConsulta) -> str:
        return identidad.plan_id(
            imagen_hash=estado.entrada.imagen_hash,
            pedido=estado.entrada.pedido,
            prompt_id=self.binding.prompt_id,
            version_del_prompt=self.binding.version_del_prompt,
            modelo=self.binding.modelo,
            muestreo=self.binding.muestreo.a_json(),
        )

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        plan_id = self.plan_id_de(estado)
        plan = self.almacen.obtener_o_producir(
            plan_id, lambda: self._producir(estado, plan_id)
        )
        return estado.con(plan=plan)

    def _producir(self, estado: EstadoDeConsulta, plan_id: str) -> PlanDeConsulta:
        mensajes = armar_mensajes(estado, version=self.version_del_prompt)
        tokens_in = tokens_out = 0
        último: Exception | None = None
        for intento in range(1, int(self.binding.reintentos) + 2):
            respuesta = self.binding.invocar(
                mensajes, imagenes=[estado.entrada.datos_de_imagen]
            )
            self.ultima_respuesta = respuesta
            self.n_intentos = intento
            tokens_in += int(respuesta.uso.get("tokens_in", 0) or 0)
            tokens_out += int(respuesta.uso.get("tokens_out", 0) or 0)
            try:
                visual, conceptual = self._parsear(respuesta)
                plan = PlanDeConsulta(
                    plan_id=plan_id,
                    consulta_id=estado.entrada.consulta_id,
                    imagen_hash=estado.entrada.imagen_hash,
                    pedido=estado.entrada.pedido,
                    consulta_visual=visual,
                    consulta_conceptual=conceptual,
                    binding={
                        **self.binding.identidad(),
                        "muestreo_efectivo": dict(respuesta.muestreo_efectivo),
                        "modelo_efectivo": respuesta.modelo_efectivo,
                    },
                    uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
                    procedencia=estado.entrada.procedencia,
                    n_intentos=intento,
                )
                plan.validar()
            except PlanInválido as e:
                último = e
                if intento > int(self.binding.reintentos):
                    raise
                continue
            return plan
        raise último  # pragma: no cover - el bucle siempre devuelve o levanta

    def _parsear(self, respuesta) -> tuple[str, str]:
        try:
            crudo = respuesta.json()
        except json.JSONDecodeError as e:  # pragma: no cover - el puerto ya reintentó
            raise PlanInválido(
                f"el planificador devolvió algo que no parsea ({e}). el contrato de los puertos ya reintentó "
                f"hasta `binding.reintentos`; acá es **falla de etapa** y la política de "
                f"la tabla de degradación por etapa es el **pedido crudo**."
            ) from None
        if not isinstance(crudo, dict):
            raise PlanInválido(
                f"el planificador devolvió un JSON que no es un objeto "
                f"({type(crudo).__name__}). El esquema está en el prompt del sistema y es "
                f"parte del binding."
            )
        faltan = sorted({"consulta_visual", "consulta_conceptual"} - set(crudo))
        if faltan:
            raise PlanInválido(
                f"el planificador devolvió un JSON sin {faltan}. Es: un JSON "
                f"**válido** que no satisface el esquema declarado es la misma falla que un "
                f"JSON inválido por otro camino, y el reintento corresponde igual. Claves "
                f"que vinieron: {sorted(crudo)[:6]}."
            )
        valores = []
        for clave in ("consulta_visual", "consulta_conceptual"):
            valor = crudo[clave]
            if not isinstance(valor, str):
                raise PlanInválido(
                    f"`{clave}` vino como {type(valor).__name__} y el esquema pide un "
                    f"string. **Una lista de consultas sería otra etapa**: el diseño de la arquitectura "
                    f"compromete **dos** campos, y de eso depende el conteo del criterio de costo."
                )
            valores.append(valor.strip())
        return valores[0], valores[1]

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "binding": self.binding.a_json(),
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "llamadas_al_llm_por_consulta": 1,
            "idioma_de_las_consultas": "en",
            "version_del_prompt": self.version_del_prompt,
            "versiones_del_prompt_disponibles": sorted(PROMPTS),
            "nota_de_m7_1": (
                "🔴 **El v2 es los hallazgos**: con el v1, en **4 de los 10** pedidos "
                "culturales el plan describía **la cultura de la imagen subida en vez de la "
                "pedida** (y en `c04` la **negaba**). El v2 declara la precedencia *el "
                "pedido gana sobre la imagen*, prohíbe nombrar una cultura que el pedido no "
                "pidió, prohíbe escribir en negativo y manda la cultura a **las dos** "
                "consultas. **El v1 queda citable** —es otra `version` del binding, así que "
                "es otro `plan_id`— porque *su medición es un resultado*: «el "
                "pre-retrieval ingenuo suelta la cultura pedida en el 40 % de los pedidos "
                "culturales» entra al informe del Ej. 4 como antes/después."
            ),
            "nota": (
                "**Una sola llamada por consulta y es normativo**: los dos "
                "campos salen del mismo JSON porque *partirla costaría reenviar la imagen "
                "sin ganar nada* y porque de eso depende el conteo **C1 = 1, C2 = C3 = 2, "
                "C4 = 2 + N** que el criterio de costo compromete. Las dos consultas van **en inglés**: los "
                "consumidores son el text encoder de CLIP y el texto de museo de AIC y CMA, "
                "que está en inglés — el diseño de la arquitectura da los ejemplos en español y no lo "
                "cierra. **Va reportado.**"
            ),
        }

def contar_palabras(texto: str) -> int:
    return len(str(texto).split())

def consultas_visuales_largas(planes, *, tope: int = PALABRAS_MÁXIMAS_VISUAL) -> tuple:
    return tuple(p for p in planes if contar_palabras(p.consulta_visual) > tope)
