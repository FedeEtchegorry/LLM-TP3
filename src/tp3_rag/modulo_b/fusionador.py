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
)

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
            f"`fusion.k = {k}` tiene que ser >= 1; el default es 60. Con k chico el "
            f"primer puesto domina y la fusión deja de premiar que una obra aparezca "
            f"en las dos ramas, que es justo lo que se busca."
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

FUNCIONES = {
    "rrf": rrf,
    "suma_de_scores_normalizados": suma_de_scores_normalizados,
    "concatenacion": concatenacion,
}

class SinPiso(ErrorDelProyecto):
    pass

class Fusionador(Etapa):
    NOMBRE = "fusionador"
    PRECONDICIONES = ("fuentes_de_ranking",)
    CLAVES_DE_CONFIGURACIÓN = ("fusion.estrategia", "fusion.k", "consulta.top_k")
    DEGRADA = True
    POLÍTICA = (
        "con una sola fuente pasa de largo; con dos y una falla, se usa la visual. "
        "Si falla el fusionador mismo, el ranking fusionado queda ausente con su "
        "degradación anotada, y el top-5 sigue siendo el de la última etapa de "
        "ranking que no degradó, que es la visual porque es el piso."
    )

    def __init__(self, *, estrategia: str, k: int = K_POR_DEFECTO, top_k: int = 5) -> None:
        if estrategia not in ESTRATEGIAS:
            raise ErrorDeConfiguración(
                f"`fusion.estrategia = {estrategia!r}` no es una de {list(ESTRATEGIAS)}. "
                f"RRF es el default; las otras dos están implementadas para poder "
                f"compararlas sin tocar código."
            )
        self.estrategia = estrategia
        self.k = int(k)
        if self.k < 1:
            raise ErrorDeConfiguración(f"`fusion.k = {k}` tiene que ser >= 1; el default es 60.")
        self.top_k = int(top_k)
        if self.top_k < 1:
            raise ErrorDeConfiguración(
                f"`consulta.top_k = {top_k}` tiene que ser >= 1: es cuántas obras "
                f"devuelve la consulta."
            )

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        fuentes = {
            nombre: candidatas
            for nombre, candidatas in estado.fuentes_de_ranking.items()
            if candidatas
        }
        if not fuentes:
            raise SinPiso(
                "no hay ninguna fuente de ranking que fusionar, y con cero no hay un "
                "ranking pobre: no hay ranking. Llegar acá significa que el "
                "recuperador visual no dejó su clave, y como es el piso y no degrada, "
                "la consulta ya había fallado antes."
            )
        if FUENTE_DEL_PISO not in fuentes:
            raise SinPiso(
                f"la fusión tiene {sorted(fuentes)} y le falta la rama "
                f"`{FUENTE_DEL_PISO}`. Con dos fuentes y una que falla se usa la visual, "
                f"porque es el piso y la de texto no. Entregar el ranking de texto como "
                f"resultado de una corrida sin piso publicaría una fila que dice tener "
                f"dos ramas y tiene una."
            )

        filas = FUNCIONES[self.estrategia](fuentes, k=self.k)
        nuevo = estado.con(ranking_fusionado=tuple(filas))
        return nuevo.proyectar_top5(
            [f.artwork_id for f in filas[: self.top_k]], por=self.NOMBRE
        )

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "estrategia": self.estrategia,
            "k": self.k if self.estrategia == "rrf" else None,
            "estrategias_declaradas": list(ESTRATEGIAS),
            "top_k": self.top_k,
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "nota_de_d5": (
                "RRF usa solo rangos. Un coseno de CLIP y uno de "
                "`text-embedding-3-small` no son calibrables entre sí, así que el "
                "default no lee `score` en ninguna línea. La estrategia "
                "`suma_de_scores_normalizados` sí lo hace, y es el único lugar del "
                "sistema donde dos escalas se suman."
            ),
            "nota_del_nombre": (
                "El campo de la fila se llama `score_fusionado` y no `score_rrf`: con "
                "tres estrategias posibles, un campo llamado `score_rrf` que a veces "
                "trae una suma de scores normalizados miente. La estrategia viaja en "
                "la misma fila."
            ),
        }

def ramas_que_entraron(ranking_fusionado: Sequence[ObraFusionada]) -> dict[str, int]:
    conteo: dict[str, int] = {}
    for fila in ranking_fusionado:
        for rama in fila.rangos_de_origen:
            conteo[rama] = conteo.get(rama, 0) + 1
    return dict(sorted(conteo.items()))
