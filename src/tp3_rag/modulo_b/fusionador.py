from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tp3_rag.modulo_b.etapa import Candidata, Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto

K_POR_DEFECTO = 60

ESTRATEGIAS: tuple[str, ...] = (
    "rrf",
    "suma_de_scores_normalizados",
    "concatenacion",
    "cascada",
)

FUENTE_DEL_FILTRO = "texto"

FUENTE_DEL_PISO = "visual"

@dataclass(frozen=True)
class ObraFusionada:
    artwork_id: str
    score_fusionado: float
    rangos_de_origen: Mapping[str, int] = field(default_factory=dict)
    estrategia: str = "rrf"

    @property
    def n_ramas(self) -> int:
        return len(self.rangos_de_origen)

    def a_json(self) -> dict:
        return {
            "artwork_id": self.artwork_id,
            "score_fusionado": round(float(self.score_fusionado), 8),
            "estrategia": self.estrategia,
            "rangos_de_origen": dict(self.rangos_de_origen),
            "n_ramas": self.n_ramas,
        }

def _rangos_por_rama(
    fuentes: Mapping[str, Sequence[Candidata]]
) -> dict[str, dict[str, int]]:
    salida: dict[str, dict[str, int]] = {}
    for rama, candidatas in fuentes.items():
        por_obra: dict[str, int] = {}
        for c in candidatas:
            previo = por_obra.get(c.artwork_id)
            if previo is None or c.rank < previo:
                por_obra[c.artwork_id] = int(c.rank)
        salida[rama] = por_obra
    return salida

def rrf(
    fuentes: Mapping[str, Sequence[Candidata]], *, k: int = K_POR_DEFECTO
) -> list[ObraFusionada]:
    if int(k) < 1:
        raise ErrorDeConfiguración(
            f"`fusion.k = {k}` tiene que ser ≥ 1. la fusión recíproca de rangos con k = 60 declara **60**, y el valor sale de "
            f"la literatura de RRF: con `k` chico el primer puesto domina y la fusión deja "
            f"de premiar el acuerdo entre ramas, que es *"
            f"«las obras que aparecen en ambas ramas suben»* — el comportamiento que el "
            f"email aprobado describe."
        )
    rangos = _rangos_por_rama(fuentes)
    puntajes: dict[str, float] = {}
    origen: dict[str, dict[str, int]] = {}
    for rama, por_obra in rangos.items():
        for artwork_id, rango in por_obra.items():
            puntajes[artwork_id] = puntajes.get(artwork_id, 0.0) + 1.0 / (
                int(k) + int(rango)
            )
            origen.setdefault(artwork_id, {})[rama] = int(rango)
    filas = [
        ObraFusionada(
            artwork_id=artwork_id,
            score_fusionado=puntajes[artwork_id],
            rangos_de_origen=dict(sorted(origen[artwork_id].items())),
            estrategia="rrf",
        )
        for artwork_id in puntajes
    ]
    filas.sort(key=lambda f: (-f.score_fusionado, f.artwork_id))
    return filas

def suma_de_scores_normalizados(
    fuentes: Mapping[str, Sequence[Candidata]], **_
) -> list[ObraFusionada]:
    puntajes: dict[str, float] = {}
    origen: dict[str, dict[str, int]] = {}
    rangos = _rangos_por_rama(fuentes)
    for rama, candidatas in fuentes.items():
        valores = [float(c.score or 0.0) for c in candidatas]
        if not valores:
            continue
        lo, hi = min(valores), max(valores)
        ancho = hi - lo
        for c in candidatas:
            normalizado = 1.0 if ancho <= 0 else (float(c.score or 0.0) - lo) / ancho
            puntajes[c.artwork_id] = puntajes.get(c.artwork_id, 0.0) + normalizado
            origen.setdefault(c.artwork_id, {})[rama] = rangos[rama][c.artwork_id]
    filas = [
        ObraFusionada(
            artwork_id=artwork_id,
            score_fusionado=puntajes[artwork_id],
            rangos_de_origen=dict(sorted(origen[artwork_id].items())),
            estrategia="suma_de_scores_normalizados",
        )
        for artwork_id in puntajes
    ]
    filas.sort(key=lambda f: (-f.score_fusionado, f.artwork_id))
    return filas

def concatenacion(
    fuentes: Mapping[str, Sequence[Candidata]], **_
) -> list[ObraFusionada]:
    rangos = _rangos_por_rama(fuentes)
    colas = {rama: list(candidatas) for rama, candidatas in fuentes.items()}
    orden_de_ramas = sorted(colas, key=lambda r: (r != FUENTE_DEL_PISO, r))
    vistas: list[str] = []
    ya = set()
    i = 0
    while any(len(colas[r]) > i for r in orden_de_ramas):
        for rama in orden_de_ramas:
            if len(colas[rama]) <= i:
                continue
            artwork_id = colas[rama][i].artwork_id
            if artwork_id not in ya:
                ya.add(artwork_id)
                vistas.append(artwork_id)
        i += 1
    filas = [
        ObraFusionada(
            artwork_id=artwork_id,
            score_fusionado=1.0 / (posicion + 1),
            rangos_de_origen=dict(
                sorted(
                    (rama, por_obra[artwork_id])
                    for rama, por_obra in rangos.items()
                    if artwork_id in por_obra
                )
            ),
            estrategia="concatenacion",
        )
        for posicion, artwork_id in enumerate(vistas)
    ]
    return filas

def cascada(
    fuentes: Mapping[str, Sequence[Candidata]],
    *,
    puntajes_visuales: Mapping[str, float],
    **_,
) -> list[ObraFusionada]:
    rangos = _rangos_por_rama(fuentes)
    if FUENTE_DEL_FILTRO not in rangos:
        visual = rangos.get(FUENTE_DEL_PISO, {})
        filas = [
            ObraFusionada(
                artwork_id=a,
                score_fusionado=1.0 / r,
                rangos_de_origen={FUENTE_DEL_PISO: r},
                estrategia="cascada",
            )
            for a, r in visual.items()
        ]
        filas.sort(key=lambda f: (-f.score_fusionado, f.artwork_id))
        return filas
    elegibles = rangos[FUENTE_DEL_FILTRO]
    sin_puntaje = sorted(set(elegibles) - set(puntajes_visuales))
    if sin_puntaje:
        raise ErrorDelProyecto(
            f"la cascada no tiene puntaje visual para {len(sin_puntaje)} obras elegibles "
            f"({sin_puntaje[:3]}). Ordenar las que sí tienen y descartar el resto sería un "
            f"filtro que nadie declaró."
        )
    filas = [
        ObraFusionada(
            artwork_id=a,
            score_fusionado=float(puntajes_visuales[a]),
            rangos_de_origen=dict(
                sorted(
                    (rama, por_obra[a])
                    for rama, por_obra in rangos.items()
                    if a in por_obra
                )
            ),
            estrategia="cascada",
        )
        for a in elegibles
    ]
    filas.sort(key=lambda f: (-f.score_fusionado, f.artwork_id))
    return filas

FUNCIONES = {
    "rrf": rrf,
    "cascada": cascada,
    "suma_de_scores_normalizados": suma_de_scores_normalizados,
    "concatenacion": concatenacion,
}

class SinPiso(ErrorDelProyecto):
    pass

class Fusionador(Etapa):
    NOMBRE = "fusionador"
    PRECONDICIONES = ("fuentes_de_ranking",)
    CLAVES_DE_CONFIGURACIÓN = (
        "fusion.estrategia",
        "fusion.k",
        "consulta.top_k",
        "consulta.alfa",
    )
    DEGRADA = True
    POLÍTICA = (
        "con **una sola** fuente **pasa a través**; con dos y una falla, **se usa la "
        "visual** (degradación). Si falla el fusionador mismo, `ranking_fusionado` queda ausente "
        "+ degradación, y el `top5` sigue siendo el de la última etapa de ranking que no "
        "degradó — que es la **visual**, porque es el piso (la asimetría del piso de recuperación)."
    )

    def __init__(
        self,
        *,
        estrategia: str,
        k: int = K_POR_DEFECTO,
        top_k: int = 5,
        indice=None,
        puerto_de_imagen=None,
        alfa: float | None = None,
    ) -> None:
        if estrategia not in ESTRATEGIAS:
            raise ErrorDeConfiguración(
                f"`fusion.estrategia = {estrategia!r}` no es una de {list(ESTRATEGIAS)}. "
                f"**RRF es el default** (la fusión recíproca de rangos con k = 60) y las otras dos son **brazos "
                f"declarados** (el stack declarado): *«descartado» acá significa «no es el "
                f"default», no «no se prueba»*. Compararlos es el barrido, y además "
                f"es el reuso del caché de imagen entre variantes de la lista de recortes."
            )
        self.estrategia = estrategia
        if estrategia == "cascada" and (
            indice is None or puerto_de_imagen is None or alfa is None
        ):
            raise ErrorDeConfiguración(
                "`fusion.estrategia = \"cascada\"` ordena las obras elegibles con CLIP, así "
                "que necesita el `PuertoÍndiceVectorial`, el `PuertoEmbeddingImagen` y "
                "`consulta.alfa`, y no se le inyectó alguno."
            )
        self.indice = indice
        self.puerto = puerto_de_imagen
        self.alfa = None if alfa is None else float(alfa)
        self.k = int(k)
        if self.k < 1:
            raise ErrorDeConfiguración(f"`fusion.k = {k}` tiene que ser ≥ 1 (la fusión recíproca de rangos con k = 60: 60).")
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ErrorDeConfiguración(
                f"`consulta.top_k = {top_k}` tiene que ser ≥ 1: son las **5 obras** que "
                f"el pipeline base completo y andando, el criterio de trazabilidad y el criterio de respuesta garantizada comprometen."
            )

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        fuentes = {
            nombre: candidatas
            for nombre, candidatas in estado.fuentes_de_ranking.items()
            if candidatas
        }
        if not fuentes:
            raise SinPiso(
                "no hay **ninguna** fuente de ranking que fusionar. degradación pide *≥ 1 "
                "fuente*, y con cero no hay un ranking pobre: no hay ranking. Llegar acá "
                "significa que el `recuperador_visual` —que **no degrada**, porque es el "
                "piso (la asimetría del piso de recuperación)— no dejó su clave, y entonces la consulta ya había fallado."
            )
        if FUENTE_DEL_PISO not in fuentes:
            raise SinPiso(
                f"la fusión tiene {sorted(fuentes)} y **le falta la rama `"
                f"{FUENTE_DEL_PISO}`**. degradación manda que con dos fuentes y una falla *se "
                f"use la **visual***, y esa asimetría es la de la asimetría del piso de recuperación: la rama visual es el "
                f"**piso** y la de texto no. Entregar el ranking textual como resultado de "
                f"una corrida sin piso sería publicar una fila que dice ser C3 y es otra "
                f"cosa."
            )

        if self.estrategia == "cascada":
            filas = cascada(fuentes, puntajes_visuales=self._puntajes_visuales(estado, fuentes))
        else:
            filas = FUNCIONES[self.estrategia](fuentes, k=self.k)
        nuevo = estado.con(ranking_fusionado=tuple(filas))
        return nuevo.proyectar_top5(
            [f.artwork_id for f in filas[: self.top_k]], por=self.NOMBRE
        )

    def _puntajes_visuales(
        self, estado: EstadoDeConsulta, fuentes: Mapping[str, Sequence[Candidata]]
    ) -> dict[str, float]:
        if FUENTE_DEL_FILTRO not in fuentes:
            return {}
        from tp3_rag.modulo_b.recuperador_visual import combinar
        from tp3_rag.puertos.indice_vectorial import ARTWORKS

        e_img = self.puerto.embeber_imagen(
            estado.entrada.datos_de_imagen,
            artwork_id=estado.entrada.artwork_id_de_la_imagen or estado.entrada.consulta_id,
        )
        e_txt = self.puerto.embeber_texto(estado.consulta_visual)
        vector = combinar(e_img, e_txt, self.alfa)
        ids = [c.artwork_id for c in fuentes[FUENTE_DEL_FILTRO]]
        return {r.id: float(r.score) for r in self.indice.puntuar(ARTWORKS, vector, ids)}

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "estrategia": self.estrategia,
            "alfa": self.alfa if self.estrategia == "cascada" else None,
            "k": self.k if self.estrategia == "rrf" else None,
            "estrategias_declaradas": list(ESTRATEGIAS),
            "top_k": self.top_k,
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "nota_de_d5": (
                "**la fusión recíproca de rangos con k = 60: RRF usa solo rangos.** Un coseno de CLIP y uno de "
                "`text-embedding-3-small` **no son calibrables entre sí**, así que el "
                "default no lee `score` en ninguna línea. El brazo "
                "`suma_de_scores_normalizados` sí lo hace y es el único lugar del sistema "
                "donde dos escalas se suman: está **declarado** (el stack declarado) y "
                "compararlo requiere un barrido de variantes que no está hecho."
            ),
            "nota_del_nombre": (
                "⚠️ El campo de la fila se llama **`score_fusionado`** y el contrato de etapa/la no destrucción de lo ya registrado lo "
                "nombran `score_rrf`: con tres estrategias declaradas, un campo llamado "
                "`score_rrf` que a veces trae una suma de scores normalizados **miente**. "
                "La `estrategia` viaja en la misma fila. **Va reportado.**"
            ),
        }

def ramas_que_entraron(ranking_fusionado: Sequence[ObraFusionada]) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for fila in ranking_fusionado:
        for rama in fila.rangos_de_origen:
            conteo[rama] = conteo.get(rama, 0) + 1
    return dict(sorted(conteo.items()))
