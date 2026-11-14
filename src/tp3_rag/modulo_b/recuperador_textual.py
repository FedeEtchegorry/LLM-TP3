from __future__ import annotations

from typing import Sequence

from tp3_rag.modulo_b import filtro_de_metadata as fm
from tp3_rag.modulo_b.etapa import Candidata, Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE

AGREGACIONES: tuple[str, ...] = ("mejor_fragmento", "suma_de_los_2_mejores")

def agregar_por_obra(
    hits: Sequence[tuple[str, float, str]], *, agregacion: str
) -> list[tuple[str, float, tuple[str, ...]]]:
    if agregacion not in AGREGACIONES:
        raise ErrorDeConfiguración(
            f"`recuperacion_textual.agregacion = {agregacion!r}` no es una de "
            f"{list(AGREGACIONES)}. Es el criterio de atribución, que sigue **abierto**: lo que  "
            f"implementa es el punto de variación y la elección, el único "
            f"momento en que hay números para decidirlo* (el plan de construcción)."
        )
    por_obra: dict[str, list[tuple[float, str]]] = {}
    for artwork_id, score, fragment_id in hits:
        por_obra.setdefault(artwork_id, []).append((float(score), fragment_id))

    agregadas: list[tuple[str, float, tuple[str, ...]]] = []
    for artwork_id, puntajes in por_obra.items():
        mejores = sorted(puntajes, key=lambda p: (-p[0], p[1]))
        cuantos = 1 if agregacion == "mejor_fragmento" else 2
        usados = mejores[:cuantos]
        agregadas.append(
            (artwork_id, sum(s for s, _ in usados), tuple(f for _, f in usados))
        )
    agregadas.sort(key=lambda a: (-a[1], a[0]))
    return agregadas

class SinPlanNoHayRamaDeTexto(ErrorDelProyecto):
    pass

class RecuperadorTextual(Etapa):
    NOMBRE = "recuperador_textual"
    PRECONDICIONES = ("plan",)
    CLAVES_DE_CONFIGURACIÓN = (
        "recuperacion_textual.n_fragmentos_recuperados",
        "recuperacion_textual.agregacion",
    )
    CLAVES_OPCIONALES_DE_CONFIGURACIÓN = ("recuperacion_textual.filtro_por_metadata",)
    DEGRADA = True
    POLÍTICA = (
        "se cae la fuente `\"texto\"` y el resto corre con una sola: el `Fusionador` pasa "
        "a través con la **visual** (degradación). `fuentes_de_ranking[\"texto\"]` queda ausente "
        "**+ degradación** — sin la entrada, la forma declarada del artefacto rechaza la corrida, que es su primera "
        "dirección. Es la omisión de etapas en runtime literal: *si falla la rama de texto, el visual*."
    )

    def __init__(
        self,
        *,
        indice,
        embebedor_de_texto,
        n_fragmentos: int,
        agregacion: str,
        filtro_por_metadata: bool = False,
    ) -> None:
        if embebedor_de_texto is None:
            raise ErrorDeConfiguración(
                "el `recuperador_textual` necesita el `PuertoEmbeddingTexto` y no se le "
                "inyectó ninguno. **No cae al text encoder de CLIP**: son dos espacios "
                "incompatibles (512 vs 1.536 dims) y el stack declarado los separa en dos "
                "puertos *justamente para que esto no se pueda hacer sin darse cuenta*."
            )
        self.indice = indice
        self.embebedor_de_texto = embebedor_de_texto
        self.n_fragmentos = int(n_fragmentos)
        if self.n_fragmentos < 1:
            raise ErrorDeConfiguración(
                f"`recuperacion_textual.n_fragmentos = {n_fragmentos}` tiene que ser ≥ 1. "
                f"el diseño de la arquitectura declara **top-M fragmentos (M ≈ 50)**: con 0 la rama "
                f"propondría cero obras y la fusión *pasaría a través* con la visual sin "
                f"que nada lo registrara — una C3 que corre como C2."
            )
        if agregacion not in AGREGACIONES:
            raise ErrorDeConfiguración(
                f"`recuperacion_textual.agregacion = {agregacion!r}` no es una de "
                f"{list(AGREGACIONES)} (el criterio de atribución, **abierto**)."
            )
        self.agregacion = agregacion
        if not isinstance(filtro_por_metadata, bool):
            raise ErrorDeConfiguración(
                f"`recuperacion_textual.filtro_por_metadata = {filtro_por_metadata!r}` tiene "
                f"que ser `true` o `false`."
            )
        self.filtro_por_metadata = filtro_por_metadata
        self._culturas_por_obra: dict[str, str] | None = None

    def _filtro(self, pedido: str) -> tuple[dict | None, dict]:
        """El `where` sobre `knowledge` que restringe la búsqueda, y su registro.

        Dos pasos, los dos antes de la búsqueda por similitud: el pedido se traduce en
        valores de `culture` que existen en `artworks`, un `where` sobre `artworks` da
        las obras que cumplen, y la búsqueda en `knowledge` se limita a sus fragmentos.
        """
        registro: dict = {"campo": fm.CAMPO, "terminos": [], "valores": [], "aplicado": False}
        terminos = fm.terminos_pedidos(pedido)
        if not terminos:
            return None, {**registro, "motivo": "el pedido no nombra una cultura conocida"}
        registro["terminos"] = list(terminos)
        if self._culturas_por_obra is None:
            fichas = self.indice.metadatas(ARTWORKS, list(self.indice.ids(ARTWORKS)))
            self._culturas_por_obra = {
                i: str(m.get(fm.CAMPO) or "") for i, m in fichas.items()
            }
        valores = fm.valores_que_cumplen(terminos, self._culturas_por_obra.values())
        if not valores:
            return None, {**registro, "motivo": "ninguna obra del índice tiene esa cultura"}
        registro["valores"] = list(valores)
        obras = sorted(r.id for r in self.indice.traer(ARTWORKS, where=fm.where_de(valores)))
        registro["n_obras"] = len(obras)
        if len(obras) < fm.MÍNIMO_DE_OBRAS:
            return None, {
                **registro,
                "motivo": f"menos de {fm.MÍNIMO_DE_OBRAS} obras cumplen el filtro",
            }
        return {"artwork_id": {"$in": obras}}, {**registro, "aplicado": True}

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        if estado.plan is None:
            raise SinPlanNoHayRamaDeTexto(
                "la rama de texto necesita `plan.consulta_conceptual` y el plan **no está**: "
                "o el planificador degradó, o la cadena está mal armada (la precondición declarada la habría "
                "rechazado antes de correr). degradación, recuadro 📌: *sin plan no hay "
                "`consulta_conceptual`, así que cae **también** la rama de texto*. No se cae "
                "al pedido crudo: eso es la conducta de C1 **en la rama visual**, y acá "
                "dejaría la fuente poblada con otra entrada."
            )
        consulta = estado.plan.consulta_conceptual

        vectores = self.embebedor_de_texto.embeber(
            [consulta], etiquetas=[f"texto:{estado.entrada.consulta_id}"]
        )

        where, registro = (
            self._filtro(estado.entrada.pedido or "") if self.filtro_por_metadata else (None, None)
        )
        resultados = self.indice.consultar(KNOWLEDGE, vectores[0], self.n_fragmentos, where)
        hits = [
            (r.metadata.get("artwork_id") or "", float(r.score or 0.0), r.id)
            for r in resultados
        ]
        sin_obra = [r.id for r in resultados if not (r.metadata.get("artwork_id"))]
        if sin_obra:
            raise ErrorDelProyecto(
                f"{len(sin_obra)} fragmentos de `knowledge` volvieron sin `artwork_id` "
                f"({sin_obra[:3]}). el diseño de la arquitectura lo declara **la clave que permite que un "
                f"fragmento proponga una obra**: sin él no hay nada que agregar, y la "
                f"variante se construyó con otro esquema."
            )

        agregadas = agregar_por_obra(hits, agregacion=self.agregacion)
        fichas = self.indice.metadatas(ARTWORKS, [a[0] for a in agregadas])
        candidatas = tuple(
            Candidata(
                artwork_id=artwork_id,
                rank=i + 1,
                score=score,
                metadata={
                    **(fichas.get(artwork_id) or {}),
                    "fragmentos": list(fragmentos),
                    "n_fragmentos_de_la_obra": sum(
                        1 for h in hits if h[0] == artwork_id
                    ),
                    "agregacion": self.agregacion,
                },
            )
            for i, (artwork_id, score, fragmentos) in enumerate(agregadas)
        )
        faltan_ficha = [a[0] for a in agregadas if a[0] not in fichas]
        if faltan_ficha:
            raise ErrorDelProyecto(
                f"{len(faltan_ficha)} obras propuestas por la rama de texto no están en "
                f"`artworks` de esta variante ({faltan_ficha[:3]}). Los fragmentos de "
                f"`knowledge` salen de **fragmentar el corpus**, así que toda obra con "
                f"fragmento tiene que estar indexada: el invariante 3 de los invariantes de la registración pide "
                f"≥ 1 fragmento **por obra indexada** y esto es la otra dirección. Si pasa, "
                f"la variante tiene `knowledge` y `artworks` desalineados — y una obra sin "
                f"ficha sale **sin `source_url`**, que es el criterio de trazabilidad roto."
            )
        nuevo = estado.con(fuentes_de_ranking={"texto": candidatas})
        if registro is not None:
            nuevo = nuevo.con(sellos={**dict(nuevo.sellos), "filtro_por_metadata": registro})
        return nuevo

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "n_fragmentos": self.n_fragmentos,
            "agregacion": self.agregacion,
            "filtro_por_metadata": self.filtro_por_metadata,
            "agregaciones_declaradas": list(AGREGACIONES),
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "nota_de_p4": (
                "La estrategia de agregación es el criterio de atribución y **no se cierra en **: "
                "el pendiente dice *«¿Mejor chunk, o suma de los 2 mejores? **Decidir "
                "empíricamente**»*, y empíricamente es con los números del barrido. "
                "Lo que esta pieza implementa es el **punto de variación**, que es el mismo "
                "patrón con que trató el criterio de normalización de culture, trató el criterio de trazabilidad y trató el criterio de respuesta garantizada."
            ),
            "nota_de_m4_2": (
                "⚠️ **La `consulta_conceptual` la leen dos piezas y no es doble conteo**: el "
                "`RecolectorDeEvidencia` la usa desde C2 para ordenar fragmentos "
                "dentro de una obra ya elegida**, y esta rama **rankea obras**. En C3 con el "
                "orden por defecto son **dos embeddings** por consulta, y eso aparece en "
                "la tabla de tokens, costo y la figura; el criterio de costo cuenta **2 llamadas al LLM**, porque un embedding no es una."
            ),
        }

def fuente_de_texto(estado: EstadoDeConsulta) -> tuple[Candidata, ...]:
    return tuple(estado.fuentes_de_ranking.get("texto") or ())

FUENTE = "texto"
