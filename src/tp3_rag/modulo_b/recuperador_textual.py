from __future__ import annotations

from typing import Sequence

from tp3_rag.modulo_b import filtro_de_metadata as fm
from tp3_rag.modulo_c.candidatas import familia_de_tecnica
from tp3_rag.modulo_b.etapa import Candidata, Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE

AGREGACIONES: tuple[str, ...] = ("mejor_fragmento", "suma_de_los_2_mejores")

def _año(ficha) -> int | None:
    """`year` como entero. El índice lo guarda como texto y comparar texto con entero rompe la rama."""
    try:
        return int(ficha.get("year"))
    except (TypeError, ValueError):
        return None


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
        self._fichas: dict | None = None

    def _filtro(self, pedido: str) -> tuple[dict | None, dict]:
        """El `where` sobre `knowledge` que restringe la búsqueda, y su registro.

        El pedido se traduce en ejes (cultura, técnica, artista, período). Cada eje nombrado
        da un conjunto de obras de `artworks` y la búsqueda en `knowledge` se limita a la
        intersección de los que se pudieron aplicar, antes de la similitud. Un eje que no
        encuentra valores o deja pocas obras no filtra y queda dicho en el registro.
        """
        registro: dict = {"campo": fm.CAMPO, "terminos": [], "valores": [], "aplicado": False}
        if self._fichas is None:
            self._fichas = self.indice.metadatas(ARTWORKS, list(self.indice.ids(ARTWORKS)))
        fichas = self._fichas
        conjuntos: list[set[str]] = []
        motivo = "el pedido no nombra cultura, técnica, artista ni período conocidos"

        def aplicar(obras: set[str], minimo: int, eje: str) -> dict:
            if len(obras) < minimo:
                return {"n_obras": len(obras), "motivo": f"menos de {minimo} obras cumplen el filtro"}
            conjuntos.append(obras)
            return {"n_obras": len(obras)}

        terminos = fm.terminos_pedidos(pedido)
        if terminos:
            registro["terminos"] = list(terminos)
            valores = fm.valores_que_cumplen(terminos, [str(m.get(fm.CAMPO) or "") for m in fichas.values()])
            if not valores:
                motivo = "ninguna obra del índice tiene esa cultura"
            else:
                registro["valores"] = list(valores)
                obras = {r.id for r in self.indice.traer(ARTWORKS, where=fm.where_de(valores))}
                resultado = aplicar(obras, fm.MÍNIMO_DE_OBRAS, "cultura")
                registro["n_obras"] = resultado["n_obras"]
                motivo = resultado.get("motivo", motivo)

        familias = fm.tecnicas_pedidas(pedido)
        if familias:
            obras = {i for i, m in fichas.items() if familia_de_tecnica(m.get("medium")) in familias}
            registro["tecnica"] = {"familias": list(familias), **aplicar(obras, fm.MÍNIMO_DE_OBRAS, "tecnica")}

        artistas = fm.artistas_pedidos(pedido, [str(m.get("artist") or "") for m in fichas.values()])
        if artistas:
            obras = {i for i, m in fichas.items() if m.get("artist") in artistas}
            registro["artista"] = {"valores": list(artistas), **aplicar(obras, fm.MÍNIMO_DE_OBRAS_ARTISTA, "artista")}

        rangos = fm.periodos_pedidos(pedido)
        if rangos:
            obras = {
                i for i, m in fichas.items()
                if _año(m) is not None and any(d <= _año(m) <= h for d, h in rangos)
            }
            registro["periodo"] = {"rangos": [list(r) for r in rangos], **aplicar(obras, fm.MÍNIMO_DE_OBRAS, "periodo")}

        epocas = fm.epocas_pedidas(pedido)
        if epocas:
            culturas = {i: str(m.get(fm.CAMPO) or "") for i, m in fichas.items()}
            obras = set()
            for terminos_de_la_epoca, desde, hasta in epocas:
                valores = set(fm.valores_que_cumplen(terminos_de_la_epoca, culturas.values()))
                obras |= {
                    i for i, m in fichas.items()
                    if culturas[i] in valores and _año(m) is not None and desde <= _año(m) <= hasta
                }
            registro["epoca"] = {
                "epocas": [[list(c), d, h] for c, d, h in epocas],
                **aplicar(obras, fm.MÍNIMO_DE_OBRAS, "epoca"),
            }

        if not conjuntos:
            return None, {**registro, "motivo": motivo}
        return {"artwork_id": {"$in": sorted(set.intersection(*conjuntos))}}, {**registro, "aplicado": True}

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
