from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture
from tp3_rag.modulo_a.particionador import DefiniciónDeEstrato
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeConfiguración

CUPO = 30

FAMILIAS_DE_TÉCNICA: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("oleo", ("oil",), ()),
    ("temple_sobre_papel", ("tempera",), ("paper",)),
    ("temple_sobre_tabla", ("tempera",), ("panel", "wood", "canvas")),
    ("tinta_sobre_papel_o_seda", (), ("ink", "color on silk", "watercolor")),
    ("pastel_o_gouache", (), ("pastel", "gouache")),
    ("fresco_o_muro", (), ("fresco", "plaster", "wall")),
)
FAMILIA_POR_OMISIÓN = "otra"
SIN_TÉCNICA = "sin_medium"
SIN_CULTURA = "sin_culture"

EJES = ("periodo", "cultura", "tecnica", "museo", "tiene_artista")

def familia_de_tecnica(medium: str | None) -> str:
    if not medium or not medium.strip():
        return SIN_TÉCNICA
    plegado = medium.casefold()
    for familia, todas, alguna in FAMILIAS_DE_TÉCNICA:
        if all(p in plegado for p in todas) and (
            not alguna or any(p in plegado for p in alguna)
        ):
            return familia
    return FAMILIA_POR_OMISIÓN

@dataclass(frozen=True)
class Candidata:
    artwork_id: str
    museo: str
    periodo: str
    cultura: str
    culturas: tuple[str, ...]
    tecnica: str
    tiene_artista: bool
    title: str | None
    artist: str | None
    date_display: str | None
    year: int | None
    medium: str | None
    source_url: str
    archivo_de_imagen: str
    cubre: tuple[str, ...] = field(default_factory=tuple)

    def a_json(self) -> dict:
        return {
            "artwork_id": self.artwork_id,
            "museo": self.museo,
            "periodo": self.periodo,
            "cultura": self.cultura,
            "culturas": list(self.culturas),
            "tecnica": self.tecnica,
            "tiene_artista": self.tiene_artista,
            "title": self.title,
            "artist": self.artist,
            "date_display": self.date_display,
            "year": self.year,
            "medium": self.medium,
            "source_url": self.source_url,
            "archivo_de_imagen": self.archivo_de_imagen,
            "cubre": list(self.cubre),
        }

@dataclass(frozen=True)
class Propuesta:
    candidatas: tuple[Candidata, ...]
    criterio: dict
    cobertura: dict
    afuera: dict

    def a_json(self) -> dict:
        return {
            "artefacto": None,
            "nota": (
                "🔴 **Es una propuesta, no un artefacto y no una decisión.** No se "
                "versiona: es determinista y se re-deriva con `tp3-consultas candidatas`. "
                "Qué 10 imágenes entran al set lo decide el grupo (encargo, los niveles de estabilidad)."
            ),
            "criterio": self.criterio,
            "cobertura": self.cobertura,
            "afuera": self.afuera,
            "candidatas": [c.a_json() for c in self.candidatas],
        }

class PropuestaDeCandidatas:
    def __init__(
        self,
        cfg,
        *,
        estrato: DefiniciónDeEstrato | None = None,
        criterio_de_culture: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.semilla = int(cfg["split.semilla"])
        self.estrato = estrato or DefiniciónDeEstrato.desde_configuración(cfg)
        self.criterio_de_culture = criterio_de_culture or cfg["normalizacion.culture.criterio"]
        self._normalizador = NormalizadorDeCulture(self.criterio_de_culture)
        self.raíz_de_imágenes = cfg["rutas.imagenes"]

    def _candidata(self, registro: RegistroCanónico) -> Candidata:
        culturas = tuple(self._normalizador.normalizar(registro.culture_raw))
        return Candidata(
            artwork_id=registro.artwork_id,
            museo=registro.museum,
            periodo=self.estrato.periodo_de(registro),
            cultura=culturas[0] if culturas else SIN_CULTURA,
            culturas=culturas,
            tecnica=familia_de_tecnica(registro.medium),
            tiene_artista=registro.artist is not None,
            title=registro.title,
            artist=registro.artist,
            date_display=registro.date_display,
            year=registro.year,
            medium=registro.medium,
            source_url=registro.source_url,
            archivo_de_imagen=str(
                f"{self.raíz_de_imágenes}/{registro.artwork_id.replace(':', '_')}.jpg"
            ),
        )

    @staticmethod
    def _valor(candidata: Candidata, eje: str):
        if eje == "tiene_artista":
            return "con_artista" if candidata.tiene_artista else "sin_artista"
        return getattr(candidata, eje)

    def proponer(
        self,
        registros_del_held_out: Iterable[RegistroCanónico],
        *,
        cupo: int = CUPO,
        ejes: Sequence[str] = EJES,
        incluir: Sequence[str] = (),
    ) -> Propuesta:
        candidatas = [self._candidata(r) for r in registros_del_held_out]
        if not candidatas:
            raise ErrorDeConfiguración(
                "el held-out llegó vacío: no hay de dónde proponer candidatas"
            )
        faltantes = [e for e in ejes if e not in EJES]
        if faltantes:
            raise ErrorDeConfiguración(
                f"ejes desconocidos: {faltantes}. Los declarados son {list(EJES)}."
            )
        por_id = {c.artwork_id: c for c in candidatas}
        desconocidas = [i for i in incluir if i not in por_id]
        if desconocidas:
            raise ErrorDeConfiguración(
                f"`--incluir` nombra {desconocidas}, que no están en el held-out de este "
                f"split. Las 10 imágenes salen del held-out (el diseño de la evaluación, la exclusión del held-out de toda colección): una obra del "
                f"corpus indexado se recuperaría a sí misma y el resultado de la medición "
                f"sería trivial."
            )
        cupo = min(cupo, len(candidatas))

        por_rango = sorted(
            candidatas,
            key=lambda c: identidad.rango_de_seleccion(self.semilla, c.artwork_id),
        )
        masa = {
            eje: Counter(self._valor(c, eje) for c in candidatas) for eje in ejes
        }

        elegidas: list[Candidata] = []
        tomadas: set[str] = set()
        cubre: dict[str, list[str]] = {}
        for artwork_id in dict.fromkeys(incluir):
            tomadas.add(artwork_id)
            elegidas.append(por_id[artwork_id])
            cubre.setdefault(artwork_id, []).append("pedida con --incluir")
        pasada = 0
        while len(elegidas) < cupo:
            agregó = False
            pasada += 1
            for eje in ejes:
                for valor, _ in sorted(masa[eje].items(), key=lambda p: (-p[1], p[0])):
                    if len(elegidas) >= cupo:
                        break
                    ya = any(self._valor(c, eje) == valor for c in elegidas)
                    if ya and pasada == 1:
                        continue
                    for c in por_rango:
                        if c.artwork_id in tomadas or self._valor(c, eje) != valor:
                            continue
                        tomadas.add(c.artwork_id)
                        elegidas.append(c)
                        cubre.setdefault(c.artwork_id, []).append(f"{eje}={valor}")
                        agregó = True
                        break
            if not agregó:  # pragma: no cover - el cupo se alcanza antes
                break

        elegidas = [
            Candidata(**{**c.__dict__, "cubre": tuple(cubre.get(c.artwork_id, ()))})
            for c in elegidas
        ]
        return Propuesta(
            candidatas=tuple(elegidas),
            criterio=self._criterio_declarado(cupo, ejes, len(candidatas)),
            cobertura=self._cobertura(elegidas, masa, ejes),
            afuera=self._afuera(elegidas, masa, ejes, candidatas),
        )

    def _criterio_declarado(self, cupo: int, ejes: Sequence[str], n_held_out: int) -> dict:
        return {
            "metodo": "cobertura por valor de cada eje, en pasadas",
            "ejes": list(ejes),
            "cupo": cupo,
            "n_held_out": n_held_out,
            "seleccion_dentro_del_valor": "sha256(semilla|artwork_id) ascendente",
            "semilla": self.semilla,
            "criterio_de_culture": self.criterio_de_culture,
            "familias_de_tecnica": [f for f, _, _ in FAMILIAS_DE_TÉCNICA]
            + [FAMILIA_POR_OMISIÓN, SIN_TÉCNICA],
            "tramos_de_periodo": [t.nombre for t in self.estrato.tramos]
            + [self.estrato.etiqueta_sin_periodo],
            "es_una_propuesta": True,
            "nota": (
                "🔴 Los ocho atributos del diseño de la evaluación —paisaje, retrato, interior, bodegón, "
                "abstracta, paleta fría, paleta cálida, períodos— **no son ejes de esta "
                "lista**: siete no existen en el esquema canónico y el "
                "octavo (período) sí, y es el primer eje. Los otros siete los aporta el "
                "ojo humano, que es el reparto que implica."
            ),
        }

    def _cobertura(
        self, elegidas: Sequence[Candidata], masa: Mapping[str, Counter], ejes: Sequence[str]
    ) -> dict:
        return {
            eje: {
                "valores_del_held_out": len(masa[eje]),
                "valores_cubiertos": len({self._valor(c, eje) for c in elegidas}),
                "reparto": dict(
                    sorted(Counter(self._valor(c, eje) for c in elegidas).items())
                ),
            }
            for eje in ejes
        }

    def _afuera(
        self,
        elegidas: Sequence[Candidata],
        masa: Mapping[str, Counter],
        ejes: Sequence[str],
        todas: Sequence[Candidata],
    ) -> dict:
        salida = {}
        for eje in ejes:
            cubiertos = {self._valor(c, eje) for c in elegidas}
            sin_cubrir = {
                valor: n for valor, n in masa[eje].items() if valor not in cubiertos
            }
            salida[eje] = {
                "n_valores_sin_cubrir": len(sin_cubrir),
                "n_obras_en_esos_valores": sum(sin_cubrir.values()),
                "valores": dict(sorted(sin_cubrir.items(), key=lambda p: (-p[1], p[0]))),
            }
        salida["_nota"] = (
            f"De las {len(todas)} del held-out se proponen {len(elegidas)}. Lo que queda "
            f"sin cubrir es **la cola**: valores con una o dos obras —ciudades sueltas que "
            f"AIC pone en `culture`, familias de técnica raras—. Cubrirlos todos pediría "
            f"una lista del tamaño del held-out, que es exactamente lo que esta "
            f"herramienta existe para evitar. **Si el grupo quiere un valor de la cola, "
            f"está acá nombrado**: se pide con `--incluir <artwork_id>`."
        )
        return salida
