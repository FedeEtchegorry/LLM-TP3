from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.dominio import ground_truth as gt
from tp3_rag.dominio.binding import _Binding
from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDelProyecto
from tp3_rag.nucleo.procedencia import commit_actual

VERSIÓN_DEL_APORTE_DEL_JUEZ = "1"

PROMPT_ID = "juez-grado-0-1-2-un-par"
VERSIÓN_DEL_PROMPT = "1"

ANOTADOR_DEL_JUEZ = "juez"

COLUMNAS_DEL_APORTE_DEL_JUEZ = (
    "split_id",
    "pool_id",
    "rubrica_id",
    "binding_juez",
    "ordinal",
    "consulta_id",
    "artwork_id",
    "grado",
    "estado",
    "nota",
)

ESTADO_JUZGADO = "juzgado"
ESTADO_SIN_GRADO = "sin_grado"

@dataclass(frozen=True)
class BindingDelJuez(_Binding):
    ROL: str = "juez"
    CLAVE_DE_CONFIGURACIÓN: str = "bindings.juez"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_id != PROMPT_ID or str(self.version_del_prompt) != VERSIÓN_DEL_PROMPT:
            raise ErrorDelProyecto(
                f"`[bindings.juez]` declara `prompt_id = {self.prompt_id!r}` "
                f"v{self.version_del_prompt} y este módulo tiene {PROMPT_ID!r} "
                f"v{VERSIÓN_DEL_PROMPT}. **El prompt es parte de la identidad del binding** "
                f"(el binding como único camino al puerto), y el `binding_juez` entra en la identidad del `AporteDelJuez` "
                f"(la tabla de identidades): si el texto y el id pueden discrepar, dos aportes con el mismo "
                f"nombre de archivo pudieron correr con prompts distintos y el κ dejaría de "
                f"decir **de qué juez** es."
            )

    @property
    def binding_juez(self) -> str:
        return identidad.hash_corto(self.identidad())

SISTEMA = """\
Sos el LLM-judge de la evaluación de un buscador multimodal de obras de arte. Tu tarea es \
aplicar UNA RÚBRICA que te paso entera y emitir un grado de la escala 0 / 1 / 2 para UN par \
(consulta, obra recuperada).

Recibís:

1. LA IMAGEN ORIGINAL de la consulta (la primera imagen),
2. el PEDIDO de la persona, tal como lo escribió, y su CATEGORÍA,
3. UNA obra recuperada: su IMAGEN (la segunda) y su ficha de museo.

REGLAS DEL INSTRUMENTO, y son de la rúbrica:

1. Aplicá la rúbrica que sigue. No uses tu propio criterio donde la rúbrica fija uno.
2. Cada par se juzga SOLO. No sabés qué otras obras trajo esta consulta ni en qué puesto
   salió ésta: no hay "la mejor de las cinco".
3. La imagen manda para lo que se ve; la ficha manda para lo que no se ve (cultura, año,
   artista, museo).
4. Si la rúbrica te hace derivar una lectura del pedido, derivala y DECILA en la nota.
5. Si no podés decidir, el grado es 1 — "cumple parcialmente o es defendible". NO uses 0
   como "no sé": un 0 es la afirmación "no cumple".

Devolvé EXCLUSIVAMENTE un objeto JSON con esta forma, sin texto alrededor:

{"grado": <0, 1 o 2>, "nota": "..."}

La `nota` es en español, una o dos oraciones: qué eje usaste y por qué ese grado.

────────────────────────────────────────────────────────────────────────────────
LA RÚBRICA, TEXTUAL (rubrica_id = §RUBRICA_ID§, hash del texto = §HASH§)
────────────────────────────────────────────────────────────────────────────────

§RUBRICA§
"""

HUECOS_DEL_SISTEMA = ("§RUBRICA_ID§", "§HASH§", "§RUBRICA§")

def _sistema_con(*, rubrica) -> str:
    texto = SISTEMA
    for hueco, valor in zip(
        HUECOS_DEL_SISTEMA,
        (rubrica.rubrica_id, rubrica.hash_del_texto or "?", rubrica.texto),
    ):
        texto = texto.replace(hueco, str(valor))
    return texto

CAMPOS_DE_LA_FICHA = ("obra", "artista", "cultura", "anio", "museo")

def armar_mensajes(
    *, pedido: str, categoria: str, artwork_id: str, ficha: Mapping[str, Any], rubrica
) -> list[dict]:
    partes = [f"{c}: {ficha[c]}" for c in CAMPOS_DE_LA_FICHA if ficha.get(c)]
    return [
        {"rol": "sistema", "texto": _sistema_con(rubrica=rubrica)},
        {
            "rol": "usuario",
            "texto": (
                f"Pedido de la persona: «{pedido}»\n"
                f"Categoría del pedido: {categoria}\n\n"
                f"Obra recuperada · artwork_id: {artwork_id}\n"
                f"{' · '.join(partes) or '(el catálogo no publica ficha para esta obra)'}\n\n"
                f"La primera imagen es la original de la consulta; la segunda es la de esta "
                f"obra recuperada."
            ),
        },
    ]

class GradoInválido(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class JuicioAutomático:
    consulta_id: str
    artwork_id: str
    grado: int | None
    nota: str = ""
    estado: str = ESTADO_JUZGADO
    uso: Mapping[str, Any] = None  # type: ignore[assignment]
    n_intentos: int = 1

    @property
    def par(self) -> tuple[str, str]:
        return (self.consulta_id, self.artwork_id)

    @property
    def juzgado(self) -> bool:
        return self.estado == ESTADO_JUZGADO and self.grado is not None

@dataclass(frozen=True)
class AporteDelJuez:
    procedencia: dict
    juicios: tuple[JuicioAutomático, ...]

    @property
    def identidad(self) -> str:
        return self.procedencia["identidad"]

    @property
    def split_id(self) -> str:
        return self.procedencia["split_id"]

    @property
    def binding_juez(self) -> str:
        return self.procedencia["binding_juez"]

    @property
    def ordinal(self) -> int:
        return int(self.procedencia["ordinal"])

    @property
    def pool_id(self) -> str:
        return self.procedencia["pool_id"]

    @property
    def rubrica_id(self) -> str:
        return self.procedencia["rubrica_id"]

    @property
    def hash_de_la_rubrica(self) -> str | None:
        return (self.procedencia.get("rubrica") or {}).get("hash_del_texto")

    @property
    def modelo(self) -> str:
        return (self.procedencia.get("binding") or {}).get("modelo", "")

    @property
    def anotador(self) -> str:
        return ANOTADOR_DEL_JUEZ

    @property
    def pares(self) -> frozenset[tuple[str, str]]:
        return frozenset(j.par for j in self.juicios if j.juzgado)

    def grado_de(self, par: tuple[str, str]) -> int | None:
        for j in self.juicios:
            if j.par == par and j.juzgado:
                return j.grado
        return None

    def por_grado(self) -> dict[str, int]:
        return {f"grado_{g}": sum(1 for j in self.juicios if j.grado == g) for g in GRADOS}

    @property
    def n_sin_grado(self) -> int:
        return sum(1 for j in self.juicios if not j.juzgado)

    @staticmethod
    def ruta_de(raíz: str | Path, identidad_del_aporte: str) -> Path:
        return Path(raíz) / f"{identidad_del_aporte}.csv"

    def a_texto(self) -> str:
        return gt.armar_csv(
            procedencia=self.procedencia,
            columnas=COLUMNAS_DEL_APORTE_DEL_JUEZ,
            filas=[
                {
                    "split_id": self.split_id,
                    "pool_id": self.pool_id,
                    "rubrica_id": self.rubrica_id,
                    "binding_juez": self.binding_juez,
                    "ordinal": self.ordinal,
                    "consulta_id": j.consulta_id,
                    "artwork_id": j.artwork_id,
                    "grado": "" if j.grado is None else j.grado,
                    "estado": j.estado,
                    "nota": j.nota,
                }
                for j in self.juicios
            ],
        )

    @classmethod
    def desde_texto(cls, texto: str, *, ruta=None) -> "AporteDelJuez":
        procedencia, filas = gt.partir_csv(texto, ruta=ruta)
        juicios = []
        for n, fila in enumerate(filas, 1):
            crudo = (fila.get("grado") or "").strip()
            estado = (fila.get("estado") or ESTADO_JUZGADO).strip()
            if crudo and crudo not in {str(g) for g in GRADOS}:
                raise ErrorDeArtefacto(
                    f"{ruta or 'el CSV'}: la fila {n} tiene `grado = {crudo!r}` y la escala "
                    f"es {list(GRADOS)} (`la evaluación § Escala`). **No hay default**: un grado "
                    f"ilegible no se lee como 0."
                )
            juicios.append(
                JuicioAutomático(
                    consulta_id=(fila.get("consulta_id") or "").strip(),
                    artwork_id=(fila.get("artwork_id") or "").strip(),
                    grado=int(crudo) if crudo else None,
                    nota=(fila.get("nota") or "").strip(),
                    estado=estado,
                )
            )
        aporte = cls(procedencia=procedencia, juicios=tuple(juicios))
        aporte.validar()
        return aporte

    @classmethod
    def leer(cls, almacén, raíz: str | Path, identidad_del_aporte: str) -> "AporteDelJuez":
        ruta = cls.ruta_de(raíz, identidad_del_aporte)
        aporte = cls.desde_texto(almacén.leer_texto(ruta), ruta=ruta)
        if aporte.identidad != identidad_del_aporte:
            raise ErrorDeArtefacto(
                f"el aporte de {identidad_del_aporte}.csv declara adentro "
                f"`identidad = {aporte.identidad}`. El nombre y el contenido no pueden "
                f"discrepar (la regla de que la identidad va en el nombre)."
            )
        return aporte

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.csv")))

    @classmethod
    def todos(
        cls, almacén, raíz: str | Path, *, split_id: str | None = None
    ) -> tuple["AporteDelJuez", ...]:
        salida = []
        for ident in cls.existentes(almacén, raíz):
            aporte = cls.leer(almacén, raíz, ident)
            if split_id is not None and aporte.split_id != split_id:
                continue
            salida.append(aporte)
        return tuple(salida)

    def validar(self) -> None:
        faltantes = [
            k
            for k in ("identidad", "split_id", "binding_juez", "ordinal", "pool_id", "rubrica_id")
            if self.procedencia.get(k) in (None, "")
        ]
        if faltantes:
            raise ErrorDeArtefacto(
                f"el `AporteDelJuez` no declara {faltantes}. Los tres primeros son **la "
                f"identidad** (la tabla de identidades, clase 2 con ordinal) y los dos últimos son la "
                f"procedencia sin la que el κ no sabe **sobre qué pool** ni **con qué "
                f"rúbrica** se juzgó — y κ es siempre *κ sobre una rúbrica* (el compromiso con la rúbrica que se juzgó)."
            )
        esperada = identidad_del_aporte_del_juez(
            split_id=self.split_id, binding_juez=self.binding_juez, ordinal=self.ordinal
        )
        if self.identidad != esperada:
            raise ErrorDeArtefacto(
                f"`identidad` dice {self.identidad!r} y sus tres ingredientes dan "
                f"{esperada!r}."
            )
        if not self.juicios:
            raise ErrorDeArtefacto(
                f"{self.identidad}: el aporte no tiene ni un juicio. Un aporte vacío no es "
                f"un juez que no concuerda: es un juez que no corrió."
            )
        pares = [j.par for j in self.juicios]
        repetidos = sorted({p for p in pares if pares.count(p) > 1})
        if repetidos:
            raise ErrorDeArtefacto(
                f"{self.identidad}: los pares {repetidos[:3]} están juzgados **dos veces** "
                f"en el mismo aporte. Dos grados del mismo juez para el mismo par no son dos "
                f"juicios: son uno sin resolver, y κ tomaría el que el orden del archivo "
                f"deje último. **Dos pasadas del juez son dos aportes** (ordinal, la tabla de identidades)."
            )
        conteos = self.procedencia.get("conteos") or {}
        if conteos.get("n_juicios") not in (None, len(self.juicios)):
            raise ErrorDeArtefacto(
                f"{self.identidad}: `conteos.n_juicios` dice {conteos.get('n_juicios')!r} y "
                f"hay {len(self.juicios)} filas."
            )

    def a_json(self) -> dict:
        return {
            "identidad": self.identidad,
            "split_id": self.split_id,
            "pool_id": self.pool_id,
            "rubrica_id": self.rubrica_id,
            "binding_juez": self.binding_juez,
            "ordinal": self.ordinal,
            "modelo": self.modelo,
            "n_juicios": len(self.juicios),
            "n_sin_grado": self.n_sin_grado,
            "por_grado": self.por_grado(),
        }

def identidad_del_aporte_del_juez(*, split_id: str, binding_juez: str, ordinal: int) -> str:
    return f"{split_id}-{binding_juez}-{int(ordinal)}"

@dataclass
class ResultadoDelJuez:
    aporte: AporteDelJuez
    ruta: Path
    n_llamadas: int
    tokens_in: int
    tokens_out: int
    segundos: float

class Juez:
    def __init__(
        self,
        cfg,
        *,
        almacén,
        rutas,
        binding: BindingDelJuez,
        bytes_de_la_imagen_de_consulta,
        bytes_de_la_imagen_de_la_obra,
        observabilidad=None,
    ) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.rutas = rutas
        self.binding = binding
        self.bytes_de_la_imagen_de_consulta = bytes_de_la_imagen_de_consulta
        self.bytes_de_la_imagen_de_la_obra = bytes_de_la_imagen_de_la_obra
        self.obs = observabilidad

    def juzgar(
        self,
        *,
        pool,
        rubrica,
        set_de_consultas,
        limite: int | None = None,
        al_juzgar=None,
        solo_pares=None,
    ) -> ResultadoDelJuez:
        if not rubrica.es_usable_para_anotar:
            raise ErrorDelProyecto(
                f"la rúbrica {rubrica.rubrica_id!r} está en estado {rubrica.estado!r} y el "
                f"juez aplica **la misma rúbrica que los humanos** (la evaluación). Una "
                f"plantilla no es un instrumento."
            )
        pedido_de = {c.consulta_id: c for c in set_de_consultas.consultas()}
        ítems = list(pool.items)
        if solo_pares is not None:
            ítems = [i for i in ítems if (i.consulta_id, i.artwork_id) in set(solo_pares)]
        ítems = ítems[: limite if limite else None]
        juicios: list[JuicioAutomático] = []
        n_llamadas = tokens_in = tokens_out = 0
        inicio = datetime.now(timezone.utc)
        for n, item in enumerate(ítems, start=1):
            consulta = pedido_de.get(item.consulta_id)
            if consulta is None:
                raise ErrorDelProyecto(
                    f"el pool trae `consulta_id = {item.consulta_id!r}` y el "
                    f"`SetDeConsultas` {set_de_consultas.identidad} no lo tiene. Es la referencia que resuelve: un "
                    f"`consulta_id` que no resuelve deja el juicio huérfano adentro de un "
                    f"artefacto que valida."
                )
            juicio = self._juzgar_uno(item, consulta=consulta, rubrica=rubrica)
            juicios.append(juicio)
            n_llamadas += juicio.n_intentos
            tokens_in += int((juicio.uso or {}).get("tokens_in", 0) or 0)
            tokens_out += int((juicio.uso or {}).get("tokens_out", 0) or 0)
            if al_juzgar is not None:
                al_juzgar(n, len(ítems), juicio)
        segundos = (datetime.now(timezone.utc) - inicio).total_seconds()

        ordinal, ruta = self._reservar(split_id=pool.split_id)
        aporte = AporteDelJuez(
            procedencia=self._procedencia(
                pool=pool,
                rubrica=rubrica,
                ordinal=ordinal,
                juicios=juicios,
                limite=limite,
                n_items_del_pool=len(pool.items),
                n_seleccionados=None if solo_pares is None else len(ítems),
                uso={
                    "n_llamadas": n_llamadas,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "segundos": round(segundos, 2),
                },
            ),
            juicios=tuple(juicios),
        )
        aporte.validar()
        escrita = self.almacén.escribir_texto(ruta, aporte.a_texto())
        return ResultadoDelJuez(
            aporte=aporte,
            ruta=escrita,
            n_llamadas=n_llamadas,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            segundos=segundos,
        )

    def _juzgar_uno(self, item, *, consulta, rubrica) -> JuicioAutomático:
        try:
            imagen_de_consulta = self.bytes_de_la_imagen_de_consulta(consulta)
            imagen_de_la_obra = self.bytes_de_la_imagen_de_la_obra(item.artwork_id)
        except Exception as e:  # noqa: BLE001 - la causa va al artefacto
            return JuicioAutomático(
                consulta_id=item.consulta_id,
                artwork_id=item.artwork_id,
                grado=None,
                nota=(
                    f"sin grado: no se pudo leer una de las dos imágenes "
                    f"({type(e).__name__}: {e}). La llamada **no se hizo**."
                ),
                estado=ESTADO_SIN_GRADO,
                uso={},
                n_intentos=0,
            )
        mensajes = armar_mensajes(
            pedido=consulta.pedido,
            categoria=consulta.categoria,
            artwork_id=item.artwork_id,
            ficha=item.ficha,
            rubrica=rubrica,
        )
        tokens_in = tokens_out = 0
        último: Exception | None = None
        intento = 0
        for intento in range(1, int(self.binding.reintentos) + 2):
            try:
                respuesta = self.binding.invocar(
                    mensajes, imagenes=[imagen_de_consulta, imagen_de_la_obra]
                )
            except Exception as e:  # noqa: BLE001
                último = e
                break
            tokens_in += int(respuesta.uso.get("tokens_in", 0) or 0)
            tokens_out += int(respuesta.uso.get("tokens_out", 0) or 0)
            try:
                grado, nota = self._parsear(respuesta)
            except GradoInválido as e:
                último = e
                if intento > int(self.binding.reintentos):
                    break
                continue
            return JuicioAutomático(
                consulta_id=item.consulta_id,
                artwork_id=item.artwork_id,
                grado=grado,
                nota=nota,
                uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
                n_intentos=intento,
            )
        return JuicioAutomático(
            consulta_id=item.consulta_id,
            artwork_id=item.artwork_id,
            grado=None,
            nota=(
                f"sin grado después de {intento} intento(s): {type(último).__name__}: {último}"
            ),
            estado=ESTADO_SIN_GRADO,
            uso={"tokens_in": tokens_in, "tokens_out": tokens_out},
            n_intentos=intento,
        )

    def _parsear(self, respuesta) -> tuple[int, str]:
        try:
            d = respuesta.json()
        except Exception as e:  # noqa: BLE001
            raise GradoInválido(f"la salida no es JSON: {e}") from e
        if not isinstance(d, Mapping) or "grado" not in d:
            raise GradoInválido(
                f"la salida no tiene la forma "
                f'`{{"grado": <0|1|2>, "nota": "..."}}`: {str(d)[:200]!r}'
            )
        crudo = d["grado"]
        if isinstance(crudo, bool) or not isinstance(crudo, (int, str)):
            raise GradoInválido(f"`grado = {crudo!r}` no es un entero de la escala.")
        try:
            grado = int(str(crudo).strip())
        except ValueError as e:
            raise GradoInválido(f"`grado = {crudo!r}` no es un entero.") from e
        if grado not in GRADOS:
            raise GradoInválido(
                f"`grado = {grado}` y la escala es {list(GRADOS)} "
                f"(`la evaluación § Escala`): {SIGNIFICADO_DE_LA_ESCALA}"
            )
        return grado, str(d.get("nota") or "").strip()

    def _reservar(self, *, split_id: str) -> tuple[int, Path]:
        base = f"{split_id}-{self.binding.binding_juez}"
        ordinal, ruta = self.almacén.reservar_ordinal(
            str(Path(self.rutas.juicios_juez) / f"{base}-{{ordinal}}.csv")
        )
        return ordinal, Path(self.rutas.juicios_juez) / f"{base}-{ordinal}.csv"

    def _procedencia(
        self,
        *,
        pool,
        rubrica,
        ordinal: int,
        juicios,
        limite,
        n_items_del_pool: int,
        uso,
        n_seleccionados: int | None = None,
    ) -> dict:
        por_grado = {
            f"grado_{g}": sum(1 for j in juicios if j.grado == g) for g in GRADOS
        }
        return {
            "artefacto": "AporteDelJuez",
            "version_del_aporte_del_juez": VERSIÓN_DEL_APORTE_DEL_JUEZ,
            "clase_de_identidad": 2,
            "identidad": identidad_del_aporte_del_juez(
                split_id=pool.split_id,
                binding_juez=self.binding.binding_juez,
                ordinal=ordinal,
            ),
            "split_id": pool.split_id,
            "binding_juez": self.binding.binding_juez,
            "ordinal": ordinal,
            "pool_id": pool.pool_id,
            "estado_del_pool": pool.estado,
            "rubrica_id": rubrica.rubrica_id,
            "set_de_consultas": (
                (pool.procedencia.get("cita_al_set_de_consultas") or {}).get("identidad", "")
            ),
            "nota_de_identidad": nota_de_identidad(),
            "binding": self.binding.a_json(),
            "rubrica": rubrica.cita(self.rutas.rubricas),
            "conteos": {
                "n_juicios": len(juicios),
                "n_items_del_pool": n_items_del_pool,
                "n_sin_grado": sum(1 for j in juicios if not j.juzgado),
                "por_grado": por_grado,
                "alcance": (
                    f"parcial ({n_seleccionados} de {n_items_del_pool}): solo los pares sin "
                    f"juicio previo (pooling incremental)"
                    if n_seleccionados is not None
                    else "completo"
                    if limite is None
                    else f"parcial ({limite} de {n_items_del_pool})"
                ),
            },
            "libro_mayor": {"juez": dict(uso)},
            "nota_del_libro_mayor": (
                "El `usage` que devolvió la API, atribuido al rol `juez` (la atribución de cada llamada a su rol, los tokens en el artefacto de corrida). "
                "⚠️ `n_llamadas` cuenta **los intentos**, no los pares: lo que el artefacto "
                "declara es lo que **costó**, no lo que salió bien. Y es **cota "
                "inferior** del gasto facturado: la verdad de facturación es el dashboard "
                "del proxy, y las dos **no se promedian**."
            ),
            "nota_de_la_circularidad": (
                "la evaluación § La circularidad: mitigación 3 **intacta** —el juez es `gpt-4o` y "
                "el reranker `gpt-4.1-mini` (el registro de decisiones), y es un CUARTO modelo: tampoco es "
                "el de ninguno de los dos anotadores—. Mitigación 2 **computable**: el pool "
                "declara la configuración de origen de cada ítem, así que el κ se estratifica. "
                "Mitigación 1 **formalmente rota desde la anotación por IA** (ningún humano anota) y "
                "lo que la reemplaza es separar familias; lo que SÍ se mantiene es "
                "estructural: la tabla C0–C4 consume el **consenso**, que solo el Acuerdo "
                "escribe y solo desde AporteDeAnotador — este aporte vive en otra carpeta y "
                "es otro tipo."
            ),
            "procedencia": {
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": commit_actual(),
                "capas_de_configuracion": list(self.cfg.capas),
            },
        }

def nota_de_identidad() -> str:
    return (
        "Clase 2 con ORDINAL (la tabla de identidades): la identidad es split_id + binding_juez + ordinal. El "
        "juez es **el** caso de salida no determinista con entrada fija, así que dos corridas "
        "del juez con el mismo prompt y el mismo modelo son DOS aportes y el κ puede decir de "
        "cuál salió. El rubrica_id NO es ingrediente y va adentro como procedencia con el hash "
        "de su texto: el prompt del binding ya transcribe la rúbrica. 🔴 Y es clase 2 y no 3 a "
        "propósito: la asimetría con el juicio humano ES la mitigación 1 de la evaluación —la tabla "
        "C0–C4 solo sabe consumir juicios humanos, por TIPO y no por filtro—."
    )

@dataclass(frozen=True)
class _ComoAporte:
    anotador: str
    grados: Mapping[tuple[str, str], int]
    rubrica_id: str = ""

    @property
    def pares(self) -> frozenset[tuple[str, str]]:
        return frozenset(self.grados)

    @property
    def juicios(self) -> tuple:
        return tuple(self.grados.items())

    def grado_de(self, par: tuple[str, str]) -> int | None:
        return self.grados.get(par)

def kappas_del_juez(
    aporte: AporteDelJuez,
    *,
    consenso: Mapping[tuple[str, str], int],
    aportes_humanos: Mapping[str, Mapping[tuple[str, str], int]],
    origen_por_par: Mapping[tuple[str, str], Sequence[str]],
    rubrica_id: str,
    categoria_de: Mapping[str, str] | None = None,
) -> dict:
    from tp3_rag.modulo_c.acuerdo import kappa_de_cohen

    del rubrica_id
    contra_consenso = kappa_de_cohen(
        aporte, _ComoAporte(anotador="consenso", grados=dict(consenso))
    )
    por_estrato = {}
    for configuración in sorted({c for cs in origen_por_par.values() for c in cs}):
        del_estrato = {
            par: grado
            for par, grado in consenso.items()
            if configuración in (origen_por_par.get(par) or ())
        }
        recortado = AporteDelJuez(
            procedencia=aporte.procedencia,
            juicios=tuple(j for j in aporte.juicios if j.par in del_estrato),
        )
        if not recortado.juicios:
            continue
        por_estrato[configuración] = kappa_de_cohen(
            recortado, _ComoAporte(anotador=f"consenso·{configuración}", grados=del_estrato)
        )
    contra_humanos = {
        nombre: kappa_de_cohen(aporte, _ComoAporte(anotador=nombre, grados=dict(grados)))
        for nombre, grados in sorted(aportes_humanos.items())
    }
    por_categoria: dict[str, dict] = {}
    if categoria_de:
        for categoria in sorted(set(categoria_de.values())):
            de_la_cat = {
                par: grado
                for par, grado in consenso.items()
                if categoria_de.get(par[0]) == categoria
            }
            recortado = AporteDelJuez(
                procedencia=aporte.procedencia,
                juicios=tuple(j for j in aporte.juicios if j.par in de_la_cat),
            )
            if not recortado.juicios:
                continue
            por_categoria[categoria] = {
                "contra_el_consenso": kappa_de_cohen(
                    recortado, _ComoAporte(anotador=f"consenso·{categoria}", grados=de_la_cat)
                ),
                "contra_cada_anotador": {
                    nombre: kappa_de_cohen(
                        AporteDelJuez(
                            procedencia=aporte.procedencia,
                            juicios=tuple(
                                j
                                for j in aporte.juicios
                                if categoria_de.get(j.consulta_id) == categoria
                            ),
                        ),
                        _ComoAporte(
                            anotador=nombre,
                            grados={
                                par: grado
                                for par, grado in grados.items()
                                if categoria_de.get(par[0]) == categoria
                            },
                        ),
                    )
                    for nombre, grados in sorted(aportes_humanos.items())
                },
            }
    return {
        "contra_el_consenso": contra_consenso,
        "por_configuracion_de_origen": por_estrato,
        "contra_cada_anotador": contra_humanos,
        "por_categoria": por_categoria,
        "nota_del_kappa_por_categoria": (
            "🔴 **El resultado del κ humano fue el κ por CATEGORÍA y no el agregado** "
            "(decisión del acuerdo entre anotadores: `conceptual_cultural` 0,810 · `visual_cromatico` 0,249 · "
            "`ambiguo` 0,228, estrato limpio, y el argumento *«la rúbrica es precisa "
            "exactamente donde se la refinó y está ausente donde no»*). Un **cuarto modelo** "
            "leyendo la **misma** rúbrica es la única prueba independiente que ese argumento "
            "puede tener hoy: si el patrón por categoría se reproduce, el argumento se "
            "sostiene; **si se invierte, lo que el 0,810 medía no era la precisión de la "
            "rúbrica**. ⚠️ Y es sobre **otra población** que el κ humano —los pares que el "
            "juez alcanzó a juzgar— así que las `n` van al lado."
        ),
        "nota_de_los_estratos": (
            "⚠️ **Los estratos no son disjuntos**: un ítem del pool tiene hasta "
            "**cinco** configuraciones de origen, así que un par entra a varios estratos y "
            "la suma de las `n` es mayor que 379. Elegir un origen por ítem sería **perder "
            "el dato en vez de estratificarlo**, que es la razón por la que el pool guarda "
            "una fila por (ítem, configuración)."
        ),
        "nota_de_la_mitigacion_2": (
            "🔴 **Qué contesta la estratificación** (la evaluación, mitigación 2): *κ parejo "
            "entre C0 y C4 → no hay favoritismo, y queda **demostrado**, no asumido. κ alto "
            "en C0 y bajo en C4 (con el juez puntuando más alto que los anotadores) → "
            "**midieron el sesgo en vez de especularlo***. El κ global no lo detecta: se mide "
            "sobre el pool agregado, así que un juez que coincide en promedio puede estar "
            "inflando los ítems que vienen de C4 y compensando con los de C0."
        ),
    }

def leer_json_de_respuesta(texto: str) -> Any:  # pragma: no cover - utilidad de depuración
    return json.loads(texto)
