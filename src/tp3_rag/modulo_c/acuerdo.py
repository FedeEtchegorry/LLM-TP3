from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from tp3_rag.dominio import ground_truth as gt
from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA, AporteDeAnotador
from tp3_rag.dominio.pool import Pool
from tp3_rag.dominio.rubrica import Rúbrica
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.nucleo.procedencia import commit_actual

VERSIÓN_DEL_ACUERDO = "1"

UMBRAL = 0.6

COLUMNAS_DEL_CONSENSO = (
    "pool_id",
    "rubrica_id",
    "consulta_id",
    "artwork_id",
    "grado",
    "como_se_resolvio",
    "grados_por_anotador",
)

@dataclass(frozen=True)
class Kappa:
    anotadores: tuple[str, str]
    n: int
    acuerdo_observado: float
    acuerdo_esperado: float
    kappa: float | None
    matriz: Mapping[str, Mapping[str, int]]
    marginales: Mapping[str, Mapping[str, int]]
    poblacion: str
    solo_uno: Mapping[str, tuple[str, ...]]
    razon_de_ausencia: str | None = None

    @property
    def pasa_el_umbral(self) -> bool | None:
        return None if self.kappa is None else self.kappa >= UMBRAL

    @property
    def veredicto(self) -> str:
        if self.kappa is None:
            return f"κ **ausente** — {self.razon_de_ausencia}"
        if self.kappa >= UMBRAL:
            return (
                f"κ = {self.kappa:.3f} ≥ {UMBRAL} → **acuerdo sustancial**. La rúbrica "
                f"`{{rubrica}}` se puede usar para la campaña de los ~540 sin refinarla, y "
                f"los 20 juicios del piloto **se importan en vez de re-juzgarse** para los "
                f"pares que aparezcan en el pool congelado."
            )
        return (
            f"κ = {self.kappa:.3f} < {UMBRAL} → **la rúbrica es ambigua**, y eso **es el "
            f"resultado del piloto, no un fracaso** (el acuerdo entre anotadores: *un κ bajo honestamente reportado "
            f"es un hallazgo*). Las dos salidas que el acuerdo entre anotadores declara: refinarla —derivar otra "
            f"`rubrica_id`, lo que **descarta estos 20 juicios** y no borra nada— o "
            f"reportarla como limitación explícita. Pagar ese descarte acá cuesta 20 juicios "
            f"y en la campaña definitiva costaría 540: **es para esto que el piloto existe**."
        )

    def a_json(self) -> dict:
        return {
            "kappa": self.kappa,
            "umbral": UMBRAL,
            "pasa_el_umbral": self.pasa_el_umbral,
            "anotadores": list(self.anotadores),
            "n_pares_comparados": self.n,
            "poblacion": self.poblacion,
            "acuerdo_observado": round(self.acuerdo_observado, 6),
            "acuerdo_esperado_por_azar": round(self.acuerdo_esperado, 6),
            "matriz_de_confusion": {k: dict(v) for k, v in self.matriz.items()},
            "marginales": {k: dict(v) for k, v in self.marginales.items()},
            "pares_que_juzgo_solo_uno": {k: list(v) for k, v in self.solo_uno.items()},
            "razon_de_ausencia": self.razon_de_ausencia,
            "escala": {str(g): SIGNIFICADO_DE_LA_ESCALA[g] for g in sorted(GRADOS, reverse=True)},
            "formula": "κ = (po − pe) / (1 − pe), con pe = Σ_k p1k · p2k (Cohen, 3 categorías)",
            "nota_de_ausencia": (
                "κ es **ausente y no cero** cuando pe = 1 (los dos anotadores usaron un solo "
                "grado): (1−1)/(1−1) no existe. Es la regla de —*un modelo sin precio "
                "da ausente, no cero*— sobre la cifra que el acuerdo entre anotadores publica."
            ),
        }

def kappa_de_cohen(a: AporteDeAnotador, b: AporteDeAnotador) -> Kappa:
    comunes = sorted(a.pares & b.pares)
    solo_uno = {
        a.anotador: tuple(f"{c}|{o}" for c, o in sorted(a.pares - b.pares)),
        b.anotador: tuple(f"{c}|{o}" for c, o in sorted(b.pares - a.pares)),
    }
    if not comunes:
        return Kappa(
            anotadores=(a.anotador, b.anotador),
            n=0,
            acuerdo_observado=0.0,
            acuerdo_esperado=0.0,
            kappa=None,
            matriz={},
            marginales={},
            poblacion="0 pares juzgados por los dos",
            solo_uno=solo_uno,
            razon_de_ausencia=(
                "los dos aportes **no comparten ni un par**. κ es el acuerdo sobre lo mismo: "
                "sin pares comunes no hay nada que comparar. Revisá que las dos planillas "
                "salgan del **mismo pool**."
            ),
        )

    matriz = {
        f"grado_{g1}": {f"grado_{g2}": 0 for g2 in GRADOS} for g1 in GRADOS
    }
    for par in comunes:
        matriz[f"grado_{a.grado_de(par)}"][f"grado_{b.grado_de(par)}"] += 1

    n = len(comunes)
    observado = sum(matriz[f"grado_{g}"][f"grado_{g}"] for g in GRADOS) / n
    marginales = {
        a.anotador: {f"grado_{g}": sum(1 for p in comunes if a.grado_de(p) == g) for g in GRADOS},
        b.anotador: {f"grado_{g}": sum(1 for p in comunes if b.grado_de(p) == g) for g in GRADOS},
    }
    esperado = sum(
        (marginales[a.anotador][f"grado_{g}"] / n) * (marginales[b.anotador][f"grado_{g}"] / n)
        for g in GRADOS
    )
    población = (
        f"{n} pares (consulta, obra) que **los dos** juzgaron, de "
        f"{len(a.juicios)} de {a.anotador} y {len(b.juicios)} de {b.anotador}, "
        f"sobre la rúbrica {a.rubrica_id}"
    )
    if abs(1.0 - esperado) < 1e-12:
        usados = {
            quién: [g for g, c in conteo.items() if c] for quién, conteo in marginales.items()
        }
        return Kappa(
            anotadores=(a.anotador, b.anotador),
            n=n,
            acuerdo_observado=observado,
            acuerdo_esperado=esperado,
            kappa=None,
            matriz=matriz,
            marginales=marginales,
            poblacion=población,
            solo_uno=solo_uno,
            razon_de_ausencia=(
                f"el acuerdo esperado por azar es **1** porque los dos anotadores usaron un "
                f"solo grado ({usados}), así que `κ = (po − pe)/(1 − pe)` es 0/0. **No es "
                f"κ = 0 ni κ = 1**: no está definido, y publicarlo como un número sería "
                f"inventar la cifra que el acuerdo entre anotadores compromete. Lo que sí se puede decir es el "
                f"acuerdo observado: {observado:.3f}. ⚠️ Y el hallazgo está del otro lado: "
                f"un pool donde todo se juzga igual no discrimina nada — mirá si el pool "
                f"trajo 20 obras que claramente no cumplen (y entonces el piloto dice algo "
                f"del *retrieval*, no de la rúbrica)."
            ),
        )
    return Kappa(
        anotadores=(a.anotador, b.anotador),
        n=n,
        acuerdo_observado=observado,
        acuerdo_esperado=esperado,
        kappa=(observado - esperado) / (1.0 - esperado),
        matriz=matriz,
        marginales=marginales,
        poblacion=población,
        solo_uno=solo_uno,
    )

@dataclass(frozen=True)
class FilaDeConsenso:
    consulta_id: str
    artwork_id: str
    grado: int | None
    como_se_resolvio: str
    grados_por_anotador: Mapping[str, int]

def consenso(
    pool: Pool, aportes: Sequence[AporteDeAnotador], *, pareja: tuple[str, str]
) -> tuple[FilaDeConsenso, ...]:
    por_anotador = {ap.anotador: ap for ap in aportes}
    terceros = [ap for ap in aportes if ap.anotador not in pareja]
    filas = []
    for item in pool.items:
        grados = {
            nombre: ap.grado_de(item.par)
            for nombre, ap in por_anotador.items()
            if ap.grado_de(item.par) is not None
        }
        de_la_pareja = [grados.get(n) for n in pareja]
        if any(g is None for g in de_la_pareja):
            resuelto, grado = "incompleto", None
        elif de_la_pareja[0] == de_la_pareja[1]:
            resuelto, grado = "acuerdo", de_la_pareja[0]
        else:
            desempate = next(
                (ap.grado_de(item.par) for ap in terceros if ap.grado_de(item.par) is not None),
                None,
            )
            if desempate is None:
                resuelto, grado = "sin_consenso", None
            else:
                resuelto, grado = "desempate", desempate
        filas.append(
            FilaDeConsenso(
                consulta_id=item.consulta_id,
                artwork_id=item.artwork_id,
                grado=grado,
                como_se_resolvio=resuelto,
                grados_por_anotador=grados,
            )
        )
    return tuple(filas)

@dataclass
class ResultadoDeAcuerdo:
    kappa: Kappa
    consenso: tuple[FilaDeConsenso, ...]
    ruta_del_consenso: Path
    ruta_del_kappa: Path
    pool_id: str
    rubrica_id: str

class Acuerdo:
    def __init__(self, cfg, *, almacén, rutas, observabilidad=None) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.rutas = rutas
        self.obs = observabilidad

    def calcular(
        self,
        *,
        pool: Pool,
        rubrica: Rúbrica,
        aportes: Sequence[AporteDeAnotador],
        pareja: tuple[str, str] | None = None,
    ) -> ResultadoDeAcuerdo:
        de_la_rubrica = self._exigir_e11(pool=pool, rubrica=rubrica, aportes=aportes)
        pareja = self._elegir_pareja(de_la_rubrica, pareja)
        por_nombre = {ap.anotador: ap for ap in de_la_rubrica}
        k = kappa_de_cohen(por_nombre[pareja[0]], por_nombre[pareja[1]])
        filas = consenso(pool, de_la_rubrica, pareja=pareja)

        ruta_consenso = self.almacén.escribir_texto(
            self._ruta_del_consenso(pool, rubrica),
            gt.armar_csv(
                procedencia=self._procedencia(
                    pool=pool, rubrica=rubrica, aportes=de_la_rubrica, pareja=pareja,
                    kappa=k, filas=filas, artefacto="Consenso",
                ),
                columnas=COLUMNAS_DEL_CONSENSO,
                filas=[
                    {
                        "pool_id": pool.pool_id,
                        "rubrica_id": rubrica.rubrica_id,
                        "consulta_id": f.consulta_id,
                        "artwork_id": f.artwork_id,
                        "grado": "" if f.grado is None else f.grado,
                        "como_se_resolvio": f.como_se_resolvio,
                        "grados_por_anotador": " ".join(
                            f"{n}={g}" for n, g in sorted(f.grados_por_anotador.items())
                        ),
                    }
                    for f in filas
                ],
            ),
        )
        ruta_kappa = self.almacén.escribir_json(
            self._ruta_del_kappa(pool, rubrica),
            {
                **self._procedencia(
                    pool=pool, rubrica=rubrica, aportes=de_la_rubrica, pareja=pareja,
                    kappa=k, filas=filas, artefacto="Kappa",
                ),
                "kappa": k.a_json(),
                "veredicto": k.veredicto.replace("{rubrica}", rubrica.rubrica_id),
            },
        )
        return ResultadoDeAcuerdo(
            kappa=k,
            consenso=filas,
            ruta_del_consenso=ruta_consenso,
            ruta_del_kappa=ruta_kappa,
            pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
        )

    def _exigir_e11(
        self, *, pool: Pool, rubrica: Rúbrica, aportes: Sequence[AporteDeAnotador]
    ) -> tuple[AporteDeAnotador, ...]:
        del_split = [ap for ap in aportes if ap.split_id == pool.split_id]
        de_otra_rubrica = sorted(
            {ap.rubrica_id for ap in del_split if ap.rubrica_id != rubrica.rubrica_id}
        )
        de_la_rubrica = tuple(ap for ap in del_split if ap.rubrica_id == rubrica.rubrica_id)
        if len(de_la_rubrica) < 2:
            raise ErrorDeConstrucción(
                f"hay {len(de_la_rubrica)} aporte(s) para la rúbrica "
                f"{rubrica.rubrica_id!r} en el split {pool.split_id} y **κ necesita dos**.\n"
                f" los criterios de entrega el pipeline base completo y andando lo pide así: *la ronda piloto de 20 juicios **anotada "
                f"por ≥ 2 personas**, con su κ* — *"
                f"un piloto anotado por una sola persona no sirve para lo que el piloto "
                f"existe*.\n"
                + (
                    f"  ⚠️ Hay aportes de otras rúbricas ({de_otra_rubrica}) y **no se "
                    f"mezclan**: cruzar aportes de dos rúbricas no da un κ bajo, **da un κ "
                    f"que no significa nada** (, el módulo de evaluación). Si esos son los que querías, "
                    f"pasá `--rubrica`.\n"
                    if de_otra_rubrica
                    else ""
                )
                + f"  Importá el segundo con `poetry run tp3-anotar importar --planilla …`."
            )
        esperado = rubrica.hash_del_texto
        discrepantes = {
            ap.anotador: ap.hash_de_la_rubrica
            for ap in de_la_rubrica
            if ap.hash_de_la_rubrica and ap.hash_de_la_rubrica != esperado
        }
        if discrepantes:
            raise ErrorDeConstrucción(
                f"los aportes {sorted(discrepantes)} citan la rúbrica "
                f"{rubrica.rubrica_id!r} con **otro hash de texto** "
                f"({discrepantes} ≠ {esperado}).\n"
                f"  O sea: **el texto de la rúbrica se movió sin cambiar de designación**, "
                f"que es la segunda escritura que la clase 3 prohíbe y que "
                f"el diseño de la evaluación castiga con *si la rúbrica cambia, se descartan los juicios*.\n"
                f"  κ se niega porque el número **saldría igual y no significaría nada**: κ "
                f"es siempre *κ sobre una rúbrica*, y acá hay dos textos con un nombre "
                f"— el acuerdo es siempre sobre una rúbrica.\n"
                f"  Las dos salidas: **restaurar el texto** (`git diff "
                f"{self.rutas.rubricas}` muestra qué se movió) o **derivar otra rúbrica** y "
                f"re-anotar, que es la rama del acuerdo entre anotadores y para lo que el piloto cuesta 20 "
                f"juicios y no 540."
            )
        return de_la_rubrica

    @staticmethod
    def _elegir_pareja(
        aportes: Sequence[AporteDeAnotador], pareja: tuple[str, str] | None
    ) -> tuple[str, str]:
        nombres = sorted(ap.anotador for ap in aportes)
        if pareja is None:
            if len(nombres) > 2:
                raise ErrorDeConstrucción(
                    f"hay {len(nombres)} aportes para esta rúbrica ({nombres}) y **κ de "
                    f"Cohen es entre dos**. Pasá los dos con `--anotador`.\n"
                    f" 🔑 El tercero no es un promedio: el diseño de la evaluación dice que *los "
                    f"juicios del tercer integrante son el **desempate** sobre los pares en "
                    f"los que los otros dos no coincidieron*. Entra al **consenso**, no al κ "
                    f"— y por eso el consenso es derivable y el κ es de una pareja declarada."
                )
            return (nombres[0], nombres[1])
        faltan = [n for n in pareja if n not in nombres]
        if faltan:
            raise ErrorDeConstrucción(
                f"no hay aporte de {faltan} para esta rúbrica. Los que hay: {nombres}."
            )
        if pareja[0] == pareja[1]:
            raise ErrorDeConstrucción(
                "κ de Cohen es el acuerdo entre **dos** anotadores distintos: el de alguien "
                "consigo mismo es 1 por construcción y no mide la rúbrica."
            )
        return (pareja[0], pareja[1])

    def _ruta_del_consenso(self, pool: Pool, rubrica: Rúbrica) -> Path:
        return self.rutas.acuerdo / f"consenso-{pool.pool_id}-{rubrica.rubrica_id}.csv"

    def _ruta_del_kappa(self, pool: Pool, rubrica: Rúbrica) -> Path:
        return self.rutas.acuerdo / f"kappa-{pool.pool_id}-{rubrica.rubrica_id}.json"

    def _procedencia(
        self, *, pool: Pool, rubrica: Rúbrica, aportes: Sequence[AporteDeAnotador],
        pareja: tuple[str, str], kappa: Kappa, filas: Sequence[FilaDeConsenso], artefacto: str,
    ) -> dict:
        por_resolución: dict[str, int] = {}
        for f in filas:
            por_resolución[f.como_se_resolvio] = por_resolución.get(f.como_se_resolvio, 0) + 1
        return {
            "artefacto": artefacto,
            "version_del_acuerdo": VERSIÓN_DEL_ACUERDO,
            "es_una_lectura": True,
            "nota_de_identidad": (
                "El consenso y κ son **lecturas** (domain-model la licencia de imagen y metadata), no agregados: "
                "funciones de (split, pool, **rúbrica**) sobre aportes sellados, y son "
                "derivables desde el corte del alcance —el desempate del tercero es otro AporteDeAnotador—. "
                "Van al repo con identidad `pool_id` + `rubrica_id`: sin el "
                "`rubrica_id` en el nombre, el κ de la rúbrica refinada pisaba al de la "
                "anterior (el corte de la interfaz)."
            ),
            "pool_id": pool.pool_id,
            "estado_del_pool": pool.estado,
            "split_id": pool.split_id,
            "rubrica": rubrica.cita(self.rutas.rubricas),
            "pareja_del_kappa": list(pareja),
            "aportes": [
                {
                    "identidad": ap.identidad,
                    "anotador": ap.anotador,
                    "n_juicios": len(ap.juicios),
                    "por_grado": ap.por_grado(),
                    "pool_id": ap.pool_id,
                }
                for ap in aportes
            ],
            "umbral_de_a5": UMBRAL,
            "conteos": {
                "n_items_del_pool": len(pool.items),
                "n_pares_del_kappa": kappa.n,
                "por_resolucion_del_consenso": por_resolución,
            },
            "nota_de_los_dos_kappa": (
                "⚠️ El κ del piloto y el κ del acuerdo entre anotadores son **dos números distintos**: el "
                "del piloto se calcula sobre los 20 juicios de la ronda piloto y su función "
                "es decidir si la rúbrica sirve —es el disparador (b) del registro de decisiones el corte de la interfaz—; el de "
                "el acuerdo entre anotadores se calcula sobre el ~20 % del pool definitivo (~108 juicios duplicados). "
                "**No se suman ni se promedian.**"
            ),
            "procedencia": {
                "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": commit_actual(),
                "capas_de_configuracion": list(self.cfg.capas),
            },
        }

