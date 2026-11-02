from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.dominio import juicios as dj
from tp3_rag.dominio.consultas import SetDeConsultas
from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA, AporteDeAnotador, Juicio
from tp3_rag.dominio.pool import SEGUNDOS_POR_JUICIO, Pool
from tp3_rag.dominio.rubrica import DecisiónDeA5, Rúbrica
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConstrucción
from tp3_rag.nucleo.procedencia import commit_actual, ruta_publicable

VERSIÓN_DE_LA_PLANILLA = "1"

IDENTIDAD = ("fila", "pool_id", "rubrica_id", "anotador", "consulta_id", "artwork_id")
A_LLENAR = ("grado", "nota")
AYUDA = (
    "categoria",
    "pedido",
    "imagen_de_consulta",
    "obra",
    "artista",
    "cultura",
    "anio",
    "museo",
    "imagen_de_la_obra",
    "source_url",
)
COLUMNAS = IDENTIDAD + A_LLENAR + AYUDA

ALCANCE_COMPLETA = "completa"
ALCANCE_DESEMPATE = "desempate"
ALCANCES: tuple[str, ...] = (ALCANCE_COMPLETA, ALCANCE_DESEMPATE)

FUERA_DEL_ALCANCE = "fuera"

ALCANCE = "alcance"
OPCIONALES: tuple[str, ...] = (ALCANCE,)

COLUMNAS_EXPORTADAS = COLUMNAS + OPCIONALES

ESTADOS_DEL_KAPPA: tuple[str, ...] = ("no_medido", "ausente", "sustancial", "bajo_con_decision")

@dataclass(frozen=True)
class VeredictoDeA5:
    estado: str
    kappas: tuple = ()
    decision: DecisiónDeA5 | None = None

    @property
    def hay_limitacion_declarada(self) -> bool:
        return self.estado == "bajo_con_decision"

    def a_json(self) -> dict:
        return {
            "estado": self.estado,
            "kappas_leidos": [k.a_json() for k in self.kappas],
            "decision_de_a5": self.decision.a_json() if self.decision else None,
            "nota": (
                "Una rúbrica cuyo κ **medido** está bajo el umbral del acuerdo entre anotadores no exporta "
                "sin una decisión declarada. κ **ausente** (pe = 1) y κ **no medido** "
                "no bloquean y no son lo mismo: el primero se midió y no está definido —se "
                "reporta ausente con su razón, nunca 0—, el segundo no se midió."
            ),
        }

@dataclass(frozen=True)
class AlcanceDeLaPlanilla:
    tipo: str
    declarado_por: str
    pares: frozenset = frozenset()
    n_del_pool: int = 0
    razon: str = ""
    pareja_del_kappa: tuple[str, ...] = ()

    @property
    def es_completa(self) -> bool:
        return self.tipo == ALCANCE_COMPLETA

    def cubre(self, par) -> bool:
        return self.es_completa or par in self.pares

    @property
    def n(self) -> int:
        return self.n_del_pool if self.es_completa else len(self.pares)

    def a_json(self) -> dict:
        return {
            "alcance": self.tipo,
            "declarado_por": self.declarado_por,
            "n_en_alcance": self.n,
            "n_items_del_pool": self.n_del_pool,
            "n_fuera_del_alcance": self.n_del_pool - self.n,
            "pareja_del_kappa": list(self.pareja_del_kappa),
            "razon": self.razon,
            "nota": (
                "La política de trazado compromete lo importado ES lo exportado*, NO *todas las filas "
                "tienen grado* — son propiedades distintas. El conjunto de pares se comparó "
                "contra el pool entero (la política de trazado); el conjunto de filas **llenas** se comparó "
                "contra este alcance, ni más ni menos. El alcance parcial es el desempate de "
                "el tercer anotador, y vale solo para un anotador que NO es de la pareja del κ: sin "
                "esa guardia, una planilla a medio llenar de un anotador principal haría que "
                "el κ se compute sobre menos pares en silencio (modo de falla 3)."
            ),
        }

@dataclass
class ResultadoDeExportación:
    ruta: Path
    ruta_de_instrucciones: Path
    n_filas: int
    pool_id: str
    rubrica_id: str
    anotador: str
    controles: tuple = ()
    imagenes_faltantes: tuple[str, ...] = ()
    veredicto_de_a5: VeredictoDeA5 | None = None
    alcance: AlcanceDeLaPlanilla | None = None

@dataclass
class ResultadoDeImportación:
    aporte: AporteDeAnotador
    ruta: Path
    ya_estaba: bool
    n_juicios: int = 0
    distribucion: Mapping[str, int] = field(default_factory=dict)
    alcance: AlcanceDeLaPlanilla | None = None

class PlanillaDeAnotación:
    def __init__(self, cfg, *, almacén, rutas, observabilidad=None) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.rutas = rutas
        self.obs = observabilidad

    def exportar(
        self,
        *,
        pool: Pool,
        rubrica: Rúbrica,
        anotador: str,
        set_de_consultas: SetDeConsultas,
        manifiesto,
        registros_de_variantes: Mapping[str, Any],
        salida: Path | None = None,
        alcance: str = ALCANCE_COMPLETA,
    ) -> ResultadoDeExportación:
        identidad.etiqueta_declarada(anotador, campo="anotador")
        self._exigir_rubrica_usable(rubrica)
        veredicto = self._exigir_kappa_declarado(rubrica)
        self._exigir_aporte_libre(pool=pool, rubrica=rubrica, anotador=anotador)
        el_alcance = self._alcance_para_exportar(
            pool=pool, rubrica=rubrica, anotador=anotador, tipo=alcance
        )

        from tp3_rag.modulo_c import validadores as val

        controles = val.antes_de_exportar(
            pool=pool,
            manifiesto=manifiesto,
            set_de_consultas=set_de_consultas,
            registros_de_variantes=registros_de_variantes,
            aportes=AporteDeAnotador.todos(
                self.almacén, self.rutas.juicios, split_id=pool.split_id
            ),
            pools=self._pools_del_split(pool.split_id),
        )

        filas, faltantes = self._proyectar(
            pool=pool, rubrica=rubrica, anotador=anotador, set_de_consultas=set_de_consultas,
            alcance=el_alcance,
        )
        destino = Path(salida) if salida is not None else self._ruta_por_omision(
            pool=pool, rubrica=rubrica, anotador=anotador
        )
        buffer = io.StringIO()
        escritor = csv.DictWriter(
            buffer, fieldnames=list(COLUMNAS_EXPORTADAS), lineterminator="\n"
        )
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)
        ruta = self.almacén.escribir_texto(destino, buffer.getvalue())
        ruta_instrucciones = self.almacén.escribir_texto(
            Path(str(destino).removesuffix(".csv") + "-como-anotar.md"),
            como_anotar(
                pool=pool,
                rubrica=rubrica,
                anotador=anotador,
                planilla=destino,
                n_filas=len(filas),
                imagenes_faltantes=faltantes,
                veredicto_de_a5=veredicto,
                alcance=el_alcance,
            ),
        )
        return ResultadoDeExportación(
            ruta=ruta,
            ruta_de_instrucciones=ruta_instrucciones,
            n_filas=len(filas),
            pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
            anotador=anotador,
            controles=controles,
            imagenes_faltantes=faltantes,
            veredicto_de_a5=veredicto,
            alcance=el_alcance,
        )

    def _alcance_para_exportar(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, tipo: str
    ) -> AlcanceDeLaPlanilla:
        if tipo not in ALCANCES:
            raise ErrorDeConstrucción(
                f"`alcance = {tipo!r}` no es uno de {list(ALCANCES)}.\n"
                f"  No es una lista abierta: `completa` es la planilla de un anotador de la "
                f"pareja del κ y `desempate` es la del **tercero** del tercer anotador, que "
                f"juzga *solo los pares sin consenso*. Un tercer alcance sería un "
                f"subconjunto elegido por quien corre el comando, y entonces el `%` de ítems "
                f"sin juzgar de la tabla de ablación dependería de una bandera."
            )
        if tipo == ALCANCE_COMPLETA:
            return AlcanceDeLaPlanilla(
                tipo=ALCANCE_COMPLETA,
                declarado_por="la exportación: la planilla cubre el pool entero",
                n_del_pool=len(pool.items),
                razon="la política de trazado sin alcance parcial: se juzgan los ítems del pool, todos.",
            )

        from tp3_rag.modulo_c import acuerdo as ac

        pareja = ac.pareja_declarada_del_kappa(
            self.almacén, self.rutas.acuerdo, pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
        )
        en_disputa = ac.pares_en_disputa(
            self.almacén, self.rutas.acuerdo, pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
        )
        problema = self._problema_del_alcance_parcial(
            pool=pool, rubrica=rubrica, anotador=anotador, pareja=pareja,
            sin_consenso=en_disputa,
        )
        if problema:
            raise ErrorDeConstrucción(
                f"no se exporta una planilla de **desempate** para {anotador!r}:\n"
                f"    · {problema}\n"
                f"  Se avisa **antes** de gastar la anotación, que es el momento del módulo de evaluación: "
                f"exportarla igual produciría juicios que el import va a rechazar."
            )
        return self._alcance_de_desempate(
            pool=pool,
            pares=en_disputa or (),
            pareja=pareja,
            declarado_por="la columna `alcance` de la planilla, escrita al exportar",
        )

    def _alcance_de_desempate(
        self, *, pool: Pool, pares, pareja: tuple[str, ...], declarado_por: str
    ) -> AlcanceDeLaPlanilla:
        return AlcanceDeLaPlanilla(
            tipo=ALCANCE_DESEMPATE,
            declarado_por=declarado_por,
            pares=frozenset(pares),
            n_del_pool=len(pool.items),
            pareja_del_kappa=tuple(pareja),
            razon=(
                f"el tercer anotador,: el tercer anotador es el **desempate** y juzga *solo los "
                f"pares sin consenso* — un juicio suyo, no una corrección del consenso "
                f"ajeno. El subconjunto son los {len(frozenset(pares))} pares que "
                f"`consenso-{pool.pool_id}-…csv` declara **en disputa**, y la pareja del κ "
                f"({list(pareja)}) no cambia: este aporte **no entra al κ**."
            ),
        )

    def _problema_del_alcance_parcial(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, pareja: tuple[str, ...],
        sin_consenso,
    ) -> str | None:
        if sin_consenso is None or not pareja:
            return (
                f"**no hay un κ ni un consenso declarados** para el pool "
                f"{pool.pool_id} y la rúbrica {rubrica.rubrica_id!r}, así que no hay ni "
                f"**pareja** contra la que chequear la guardia ni **subconjunto** que "
                f"juzgar. El alcance parcial no es *«las filas que quieras»*: es el "
                f"desempate del consenso, y los pares que lo forman los declara el consenso.\n"
                f"      Corré primero, con los **dos** aportes completos:\n"
                f"          poetry run tp3-anotar kappa --designacion {pool.designacion} "
                f"--rubrica {rubrica.rubrica_id}"
            )
        if anotador in pareja:
            return (
                f"🔴 **{anotador!r} es de la pareja del κ** ({list(pareja)}), y el alcance "
                f"parcial vale **solo para un anotador que no lo es**.\n"
                f"      Sin esta guardia, una planilla a medio llenar de un anotador "
                f"principal entraría como aporte legítimo y **el κ se computaría sobre "
                f"menos pares en silencio** — que es el **modo de falla 3** sobre la cifra "
                f"que el acuerdo entre anotadores publica: *toda cifra con su población al lado, y la población "
                f"tiene que ser la que la decisión usa*.\n"
                f"      El desempate es de un **tercero** (el consenso): otra etiqueta de "
                f"anotador, y su aporte entra al **consenso**, no al κ."
            )
        if not sin_consenso:
            return (
                f"el consenso de {pool.pool_id} / {rubrica.rubrica_id} **no tiene ni un par "
                f"**en disputa**: los dos anotadores coincidieron en todo, así que no hay "
                f"nada que desempatar y un aporte de desempate sería un aporte vacío "
                f"(la regla de que un id determina su contenido: lo que se congela es la anotación, no el archivo)."
            )
        return None

    def _exigir_rubrica_usable(self, rubrica: Rúbrica) -> None:
        if not rubrica.es_usable_para_anotar:
            raise ErrorDeConstrucción(
                f"la rúbrica {rubrica.rubrica_id!r} declara `estado = {rubrica.estado}` y "
                f"**no es usable para anotar**.\n"
                f"  El texto de la rúbrica lo escribe el grupo (el módulo de evaluación: *lo que escribe acá "
                f"es trabajo humano*), y exportar una planilla contra una plantilla gastaría "
                f"20 juicios bajo un instrumento a medio escribir — que es, exactamente, lo "
                f"que la ronda piloto existe para **no** hacer con los ~540 (la evaluación).\n"
                f"  Llenala y poné `estado = \"completa\"` en su front matter. Mirá qué "
                f"falta con `poetry run tp3-anotar rubrica`."
            )

    def _exigir_kappa_declarado(self, rubrica: Rúbrica) -> VeredictoDeA5:
        from tp3_rag.modulo_c import acuerdo as ac

        kappas = ac.kappas_de_la_rubrica(self.almacén, self.rutas.acuerdo, rubrica.rubrica_id)
        if not kappas:
            return VeredictoDeA5(estado="no_medido")

        de_otro_texto = [
            k for k in kappas if k.hash_del_texto and k.hash_del_texto != rubrica.hash_del_texto
        ]
        if de_otro_texto:
            raise ErrorDeConstrucción(
                f"el κ que hay de la rúbrica {rubrica.rubrica_id!r} se midió sobre **otro "
                f"texto**: cita "
                f"{sorted({k.hash_del_texto for k in de_otro_texto})} y la rúbrica en disco "
                f"hashea {rubrica.hash_del_texto}.\n"
                f"  No es un κ bajo y no es el κ de este instrumento: κ es siempre *κ sobre "
                f"una rúbrica* (el módulo de evaluación), y acá hay **dos textos con un nombre** (el compromiso con la rúbrica que se juzgó, "
                f"). Exportar así gastaría anotación bajo una rúbrica cuya única "
                f"medición es de otra versión del texto.\n"
                f"  Las dos salidas: **restaurar el texto** (`git diff "
                f"{self.rutas.rubricas}` muestra qué se movió) o **derivar otra rúbrica** y "
                f"volver a medir, que es la rama del acuerdo entre anotadores.\n"
                + "".join(f"    · {k.archivo}  (κ = {k.kappa})\n" for k in de_otro_texto)
            )

        bajos = [k for k in kappas if k.esta_bajo_el_umbral]
        if not bajos:
            ausentes = [k for k in kappas if k.es_ausente]
            return VeredictoDeA5(
                estado="ausente" if ausentes and len(ausentes) == len(kappas) else "sustancial",
                kappas=kappas,
            )

        decisión = DecisiónDeA5.leer_si_existe(
            self.almacén, self.rutas.rubricas, rubrica.rubrica_id
        )
        if decisión is None:
            raise ErrorDeConstrucción(self._mensaje_de_kappa_bajo(rubrica, bajos, kappas))
        problemas = decisión.problemas_contra(
            kappas=kappas, hash_del_texto=rubrica.hash_del_texto
        )
        if problemas:
            raise ErrorDeConstrucción(
                f"hay una decisión del acuerdo entre anotadores declarada en "
                f"{self.almacén.ruta(DecisiónDeA5.ruta_de(self.rutas.rubricas, rubrica.rubrica_id))} "
                f"y **no cubre esta medición**:\n"
                + "".join(f"    · {p}\n" for p in problemas)
                + f"  La decisión existe para que la salida del acuerdo entre anotadores esté **en un artefacto** "
                f"(el chequeo), y una autorización que no cita la medición que autoriza no es "
                f"una decisión: es una desactivación del chequeo."
            )
        return VeredictoDeA5(estado="bajo_con_decision", kappas=kappas, decision=decisión)

    def _mensaje_de_kappa_bajo(self, rubrica: Rúbrica, bajos, kappas) -> str:
        peor = min(bajos, key=lambda k: k.kappa)
        ruta = self.almacén.ruta(
            DecisiónDeA5.ruta_de(self.rutas.rubricas, rubrica.rubrica_id)
        )
        return (
            f"🔴 **No se exporta**: el κ **medido** de la rúbrica {rubrica.rubrica_id!r} está "
            f"bajo el umbral del acuerdo entre anotadores y **no hay una decisión declarada**.\n"
            f"  κ = {peor.kappa:.3f} < {peor.umbral}   ·   {peor.n} pares   ·   pool "
            f"{peor.pool_id}\n"
            f"  medición     : {peor.archivo}\n"
            f"  La rúbrica es **usable por estado** (`estado = {rubrica.estado}`) y "
            f"**ambigua por medición**, y hasta el chequeo solo se chequeaba el estado: nada "
            f"impedía gastar la anotación bajo un instrumento que el propio piloto marcó "
            f"como malo — *que es literalmente lo que la ronda piloto existe para no hacer* "
            f"(el acuerdo entre anotadores, la evaluación).\n"
            f"  **el acuerdo entre anotadores ya escribió las dos salidas, y las dos cuestan lo que dicen:**\n"
            f"    · **refinar** — derivar otra `rubrica_id` con `refina = "
            f"\"{rubrica.rubrica_id}\"` y re-anotar. No borra nada y acá cuesta "
            f"{peor.n} juicios en vez de los del pool congelado: **es para esto que el "
            f"piloto existe**. Se ejerce exportando con `--rubrica <la nueva>`.\n"
            f"    · **reportarlo como limitación explícita** — y entonces la decisión tiene "
            f"que estar **en el artefacto, no en la cabeza de quien corre el comando**:\n"
            f"        poetry run tp3-anotar decision-de-a5 --rubrica {rubrica.rubrica_id} "
            f"--escribir\n"
            f"        {ruta}\n"
            f"      La plantilla trae la medición adentro (κ, pool y hash del texto) y lo "
            f"que falta escribir es **el párrafo que va al informe**.\n"
            f"  ⚠️ Y un κ bajo honestamente reportado **es un hallazgo** (el acuerdo entre anotadores): lo que esto "
            f"bloquea no es el número, es exportar como si el número no existiera.\n"
            + (
                "  κ ausente (no definido) en: "
                + ", ".join(k.pool_id for k in kappas if k.es_ausente)
                + " — ausente **no es 0**, y no es lo que bloquea.\n"
                if any(k.es_ausente for k in kappas)
                else ""
            )
        )

    def _exigir_aporte_libre(self, *, pool: Pool, rubrica: Rúbrica, anotador: str) -> None:
        ident = identidad.identidad_del_aporte(
            split_id=pool.split_id, rubrica_id=rubrica.rubrica_id, anotador=anotador
        )
        ruta = AporteDeAnotador.ruta_de(self.rutas.juicios, ident)
        if not self.almacén.existe(ruta):
            return
        raise ErrorDeConstrucción(
            f"ya hay un aporte congelado en {self.almacén.ruta(ruta)}.\n"
            f"  Es **clase 3**: no se reescribe y no se recalcula (la regla de que un id determina su contenido), y lo que hay "
            f"adentro es trabajo humano. Exportar otra planilla con la misma identidad "
            f"produciría 20 juicios que el import va a rechazar — así que se avisa **antes** "
            f"de gastarlos.\n"
            f"  Las dos salidas legítimas, y las dos cuestan lo que dicen:\n"
            f"    · **otro anotador** (`--anotador <otra-etiqueta>`): es otro aporte, y es "
            f"lo que κ necesita — hacen falta **dos** personas;\n"
            f"    · **otra rúbrica** (`--rubrica <otra>`): es la rama del acuerdo entre anotadores con κ < 0.6. "
            f"Refinar **no borra nada**: deriva otra `Rúbrica` y los aportes nuevos cuelgan "
            f"de ella."
        )

    def _pools_del_split(self, split_id: str) -> tuple[Pool, ...]:
        return tuple(
            Pool.leer(self.almacén, self.rutas.pool, p)
            for p in Pool.del_split(self.almacén, self.rutas.pool, split_id)
        )

    def _ruta_por_omision(self, *, pool: Pool, rubrica: Rúbrica, anotador: str) -> Path:
        return self.rutas.planillas / f"{pool.pool_id}-{rubrica.rubrica_id}-{anotador}.csv"

    def _proyectar(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, set_de_consultas: SetDeConsultas,
        alcance: AlcanceDeLaPlanilla,
    ) -> tuple[list[dict], tuple[str, ...]]:
        faltantes: list[str] = []
        filas = []
        for item in pool.items:
            consulta = set_de_consultas.consulta(item.consulta_id)
            imagen_de_consulta = SetDeConsultas.ruta_de_imagen(
                self.rutas.consultas, consulta.artwork_id
            )
            imagen_de_la_obra = (
                Path(self.cfg["rutas.imagenes"]) / f"{item.artwork_id.replace(':', '_')}.jpg"
            )
            for ruta in (imagen_de_consulta, imagen_de_la_obra):
                if not self.almacén.existe(ruta):
                    faltantes.append(str(ruta))
            filas.append(
                {
                    "fila": item.fila,
                    "pool_id": pool.pool_id,
                    "rubrica_id": rubrica.rubrica_id,
                    "anotador": anotador,
                    "consulta_id": item.consulta_id,
                    "artwork_id": item.artwork_id,
                    "grado": "",
                    "nota": "",
                    ALCANCE: (
                        alcance.tipo if alcance.cubre(item.par) else FUERA_DEL_ALCANCE
                    ),
                    "categoria": consulta.categoria,
                    "pedido": consulta.pedido,
                    "imagen_de_consulta": str(imagen_de_consulta),
                    **{c: item.ficha.get(c, "") for c in ("obra", "artista", "cultura", "anio",
                                                          "museo", "source_url")},
                    "imagen_de_la_obra": str(imagen_de_la_obra),
                }
            )
        return filas, tuple(dict.fromkeys(faltantes))

    def importar(
        self,
        *,
        planilla: Path,
        set_de_consultas: SetDeConsultas,
        buscar_pool=None,
        buscar_rubrica=None,
    ) -> ResultadoDeImportación:
        ruta = Path(planilla)
        cabecera, filas = _leer_csv(self.almacén.leer_texto(ruta), ruta=ruta)
        self._validar_columnas(cabecera, ruta=ruta)
        declarada = _identidad_declarada(filas, ruta=ruta)

        pool = (buscar_pool or self._buscar_pool)(declarada["pool_id"])
        rubrica = (buscar_rubrica or self._buscar_rubrica)(declarada["rubrica_id"])
        self._exigir_rubrica_usable(rubrica)

        problemas = self._comparar_contra_lo_exportado(
            filas=filas, pool=pool, declarada=declarada, ruta=ruta
        )
        alcance, problemas_del_alcance = self._resolver_alcance(
            filas=filas, pool=pool, rubrica=rubrica, declarada=declarada
        )
        problemas += problemas_del_alcance
        problemas += self._validar_grados(filas, ruta=ruta, alcance=alcance)
        problemas += self._validar_referencias(filas, set_de_consultas=set_de_consultas)
        if problemas:
            raise ErrorDeArtefacto(_mensaje_de_rechazo(ruta, problemas, pool))

        juicios = tuple(
            Juicio(
                consulta_id=f["consulta_id"].strip(),
                artwork_id=f["artwork_id"].strip(),
                grado=int(str(f["grado"]).strip()),
                nota=(f.get("nota") or "").strip(),
            )
            for f in filas
            if (f.get("grado") or "").strip()
        )
        return self._congelar(
            pool=pool, rubrica=rubrica, anotador=declarada["anotador"],
            juicios=juicios, planilla=ruta, set_de_consultas=set_de_consultas,
            alcance=alcance,
        )

    def _validar_columnas(self, cabecera: Sequence[str], *, ruta) -> None:
        faltan = [c for c in COLUMNAS if c not in cabecera]
        de_más = [c for c in cabecera if c not in COLUMNAS and c not in OPCIONALES]
        if faltan or de_más:
            raise ErrorDeArtefacto(
                f"{ruta}: las columnas no son las de la planilla exportada.\n"
                f"  faltan  : {faltan}\n"
                f"  de más  : {de_más}\n"
                f"  Las de identidad ({list(IDENTIDAD)}) son **lo exportado** y se comparan "
                f"exactas; `grado` y `nota` son lo único que se llena; el resto es la ayuda "
                f"para poder juzgar sin leer código. Una clave desconocida es un error "
                f"(la configuración en capas), y acá una columna de menos es una planilla con la que no se "
                f"puede juzgar.\n"
                f"  🔑 El **orden de las columnas no importa** (se leen por nombre); el de "
                f"las **filas** sí."
            )

    def _comparar_contra_lo_exportado(
        self, *, filas: Sequence[Mapping[str, str]], pool: Pool, declarada: Mapping[str, str],
        ruta,
    ) -> list[str]:
        problemas: list[str] = []
        esperadas = [
            {
                "fila": str(i.fila),
                "pool_id": pool.pool_id,
                "consulta_id": i.consulta_id,
                "artwork_id": i.artwork_id,
            }
            for i in pool.items
        ]
        leídas = [
            {
                "fila": (f.get("fila") or "").strip(),
                "pool_id": (f.get("pool_id") or "").strip(),
                "consulta_id": (f.get("consulta_id") or "").strip(),
                "artwork_id": (f.get("artwork_id") or "").strip(),
            }
            for f in filas
        ]
        if len(leídas) != len(esperadas):
            problemas.append(
                f"la planilla tiene **{len(leídas)} filas** y el pool {pool.pool_id} tiene "
                f"**{len(esperadas)} ítems**. El conjunto de pares que entra tiene que ser "
                f"**igual** al que salió (la política de trazado): una fila agregada es un juicio sobre un par "
                f"que nadie puso en ninguna orden de trabajo, y eso es la procedencia del pool declarada en el aporte roto por la "
                f"puerta que la planilla **es**."
            )

        pares_esperados = {(e["consulta_id"], e["artwork_id"]): e["fila"] for e in esperadas}
        pares_leídos = {(l["consulta_id"], l["artwork_id"]) for l in leídas}
        sobran = sorted(pares_leídos - set(pares_esperados))
        faltan = sorted(set(pares_esperados) - pares_leídos)
        consultas_del_pool = set(pool.consultas)
        for par in sobran:
            if par[0] in consultas_del_pool:
                problemas.append(
                    f"el par {par} **no está en el pool**, y su `consulta_id` sí: el "
                    f"`artwork_id` fue **retocado** o la fila es de otra ronda. El par que "
                    f"falta para esa consulta está más abajo."
                )
            else:
                problemas.append(
                    f"el par {par} **no está en el pool** y su `consulta_id` tampoco es una "
                    f"de las {len(consultas_del_pool)} del pool: es una fila **agregada** o "
                    f"pegada de otra ronda."
                )
        for par in faltan:
            problemas.append(
                f"el par {par} **estaba en el pool y no está en la planilla** (fila "
                f"{pares_esperados[par]}). Lo importado tiene que ser lo exportado: un par "
                f"que desaparece no es *sin juzgar*, es un ítem que el `%` de ítems sin "
                f"juzgar de la tabla de ablación iba a contar y nadie declaró."
            )

        if not sobran and not faltan:
            desordenadas = [
                (n, l, e) for n, (l, e) in enumerate(zip(leídas, esperadas), 1) if l != e
            ]
            for n, l, e in desordenadas[:5]:
                problemas.append(
                    f"la fila {n} de la planilla dice "
                    f"`fila={l['fila']} {l['consulta_id']} / {l['artwork_id']}` y lo "
                    f"exportado en esa posición es "
                    f"`fila={e['fila']} {e['consulta_id']} / {e['artwork_id']}`: la planilla "
                    f"está **reordenada**.\n"
                    f"      El orden mezclado **es contenido del pool** —`la evaluación § Pooling` "
                    f"pide mezclar para que el puesto del ranking no ancle el juicio—, y "
                    f"sobre todo: cuando se ordena **una columna sola** en una planilla de "
                    f"cálculo, los grados se despegan de sus filas sin que nada se vea "
                    f"distinto. **Ordená por `fila` (1 → {len(esperadas)}) y volvé a "
                    f"importar**; los grados no se pierden."
                )
            if len(desordenadas) > 5:
                problemas.append(f"… y {len(desordenadas) - 5} filas más fuera de orden.")

        for n, fila in enumerate(filas, 1):
            for clave in ("pool_id", "rubrica_id", "anotador"):
                valor = (fila.get(clave) or "").strip()
                if valor != declarada[clave]:
                    problemas.append(
                        f"la fila {n} declara `{clave} = {valor!r}` y el resto de la "
                        f"planilla dice {declarada[clave]!r}. **Cada fila se describe sola a "
                        f"propósito** (el módulo de evaluación: *un par pegado de otra ronda*): un aporte es "
                        f"de **un** anotador sobre **una** rúbrica, y mezclar dos no da un κ "
                        f"bajo, da un κ que no significa nada (el compromiso con la rúbrica que se juzgó)."
                    )
        return problemas

    def _resolver_alcance(
        self, *, filas: Sequence[Mapping[str, str]], pool: Pool, rubrica: Rúbrica,
        declarada: Mapping[str, str],
    ) -> tuple[AlcanceDeLaPlanilla, list[str]]:
        from tp3_rag.modulo_c import acuerdo as ac

        completa = AlcanceDeLaPlanilla(
            tipo=ALCANCE_COMPLETA,
            declarado_por="la planilla no declara alcance parcial",
            n_del_pool=len(pool.items),
            razon="la política de trazado sin alcance parcial: se juzgan los ítems del pool, todos.",
        )
        declarado, problemas = self._alcance_declarado_en_la_planilla(filas, pool=pool)
        if problemas:
            return completa, problemas
        vacías = [f for f in filas if not (f.get("grado") or "").strip()]
        if declarado is not None and declarado.es_completa:
            return declarado, []
        if declarado is None and not vacías:
            return completa, []

        pareja = ac.pareja_declarada_del_kappa(
            self.almacén, self.rutas.acuerdo, pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
        )
        en_disputa = ac.pares_en_disputa(
            self.almacén, self.rutas.acuerdo, pool_id=pool.pool_id,
            rubrica_id=rubrica.rubrica_id,
        )
        problema = self._problema_del_alcance_parcial(
            pool=pool, rubrica=rubrica, anotador=declarada["anotador"], pareja=pareja,
            sin_consenso=en_disputa,
        )
        if problema:
            return completa, [problema]
        esperados = frozenset(en_disputa or ())
        if declarado is not None and declarado.pares != esperados:
            return completa, [
                f"la columna `alcance` marca **{len(declarado.pares)} filas** como "
                f"`{ALCANCE_DESEMPATE}` y el consenso de {pool.pool_id} declara "
                f"**{len(esperados)} pares en disputa**. El subconjunto del desempate "
                f"**no es del que exporta ni del que anota**: lo define el consenso y lo declara "
                f"el consenso, así que una planilla que marca otro conjunto se exportó "
                f"contra un consenso que ya cambió — volvé a exportarla."
            ]
        if declarado is not None:
            return replace(declarado, pareja_del_kappa=tuple(pareja)), []
        problema_de_derivar = self._problema_de_derivar_el_alcance(
            pool=pool, rubrica=rubrica, anotador=declarada["anotador"]
        )
        if problema_de_derivar:
            return completa, [problema_de_derivar]
        return self._alcance_de_desempate(
            pool=pool,
            pares=esperados,
            pareja=pareja,
            declarado_por=(
                f"**derivado** del consenso en disco (`consenso-{pool.pool_id}-"
                f"{rubrica.rubrica_id}.csv`): la planilla no trae la columna "
                f"`{ALCANCE}`, así que el alcance se dedujo de los pares "
                f"en disputa y **se dice cuál se eligió** en vez de suponerlo"
            ),
        ), []

    def _problema_de_derivar_el_alcance(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str
    ) -> str | None:
        ya_desempataron = sorted(
            ap.anotador
            for ap in AporteDeAnotador.todos(
                self.almacén,
                self.rutas.juicios,
                split_id=pool.split_id,
                rubrica_id=rubrica.rubrica_id,
            )
            if ap.anotador != anotador
            and ((ap.procedencia.get("alcance") or {}).get("alcance")) == ALCANCE_DESEMPATE
        )
        if not ya_desempataron:
            return None
        return (
            f"🔴 **esta planilla no trae la columna `{ALCANCE}` y ya hay un desempate "
            f"congelado** ({ya_desempataron}), así que el alcance **no se puede derivar** "
            f".\n"
            f"  Derivarlo del `consenso-{pool.pool_id}-{rubrica.rubrica_id}.csv` valió **una "
            f"vez**: en ese momento los pares en disputa del consenso eran exactamente los "
            f"que el tercero tenía que juzgar. Ahora el consenso **ya los resolvió** con ese "
            f"desempate, así que el conjunto que hay en disco no es el que se exportó — y "
            f"cambia cada vez que alguien corre `kappa`. **El alcance lo definiría el "
            f"consenso y el consenso se recomputa con este aporte**: es circular.\n"
            f"  El alcance tiene que **viajar en la planilla**, y no hay que editar nada a "
            f"mano: volvé a exportar y la columna sale escrita fila por fila.\n"
            f"      poetry run tp3-anotar exportar --designacion {pool.designacion} "
            f"--rubrica {rubrica.rubrica_id} \\\n"
            f"          --anotador {anotador} --alcance {ALCANCE_DESEMPATE}"
        )

    def _alcance_declarado_en_la_planilla(
        self, filas: Sequence[Mapping[str, str]], *, pool: Pool
    ) -> tuple[AlcanceDeLaPlanilla | None, list[str]]:
        valores = [(f.get(ALCANCE) or "").strip() for f in filas]
        puestos = [v for v in valores if v]
        if not puestos:
            return None, []
        if len(puestos) != len(valores):
            return None, [
                f"la columna `{ALCANCE}` está **a medio llenar**: {len(puestos)} de "
                f"{len(valores)} filas la declaran. O la lleva toda la planilla o ninguna — "
                f"una fila sin alcance no es *«completa por omisión»*: es una fila de la que "
                f"no se sabe si le tocaba juzgarse, y eso es justo lo que el alcance existe "
                f"para que no pase."
            ]
        tipos = sorted({v for v in puestos if v != FUERA_DEL_ALCANCE})
        desconocidos = sorted({v for v in puestos if v not in ALCANCES + (FUERA_DEL_ALCANCE,)})
        if desconocidos:
            return None, [
                f"la columna `{ALCANCE}` dice {desconocidos} y los únicos valores son "
                f"{list(ALCANCES)} —el alcance de la planilla— y `{FUERA_DEL_ALCANCE}` para "
                f"una fila que no le toca. Una clave desconocida es un error (la configuración en capas)."
            ]
        if len(tipos) != 1:
            return None, [
                f"la columna `{ALCANCE}` declara {tipos} a la vez. Un aporte es de **un** "
                f"alcance, igual que es de **un** anotador y de **una** rúbrica: mezclar dos "
                f"no da un aporte parcial, da un aporte del que no se sabe qué población "
                f"cubre."
            ]
        tipo = tipos[0]
        if tipo == ALCANCE_COMPLETA and FUERA_DEL_ALCANCE in puestos:
            return None, [
                f"la columna `{ALCANCE}` dice `{ALCANCE_COMPLETA}` y además marca "
                f"{puestos.count(FUERA_DEL_ALCANCE)} filas como `{FUERA_DEL_ALCANCE}`. Una "
                f"planilla **completa** no tiene filas que no le toquen: eso es el alcance "
                f"`{ALCANCE_DESEMPATE}`."
            ]
        pares = frozenset(
            ((f.get("consulta_id") or "").strip(), (f.get("artwork_id") or "").strip())
            for f, v in zip(filas, valores)
            if v == tipo
        )
        if tipo == ALCANCE_COMPLETA:
            return AlcanceDeLaPlanilla(
                tipo=ALCANCE_COMPLETA,
                declarado_por=f"la columna `{ALCANCE}` de la planilla",
                n_del_pool=len(pool.items),
                razon="la política de trazado sin alcance parcial: se juzgan los ítems del pool, todos.",
            ), []
        return AlcanceDeLaPlanilla(
            tipo=ALCANCE_DESEMPATE,
            declarado_por=f"la columna `{ALCANCE}` de la planilla, fila por fila",
            pares=pares,
            n_del_pool=len(pool.items),
            razon=(
                f"el tercer anotador,: el desempate juzga *solo los pares sin consenso*, y la "
                f"planilla declara cuáles son las {len(pares)} filas que le tocan."
            ),
        ), []

    def _validar_grados(
        self, filas: Sequence[Mapping[str, str]], *, ruta,
        alcance: AlcanceDeLaPlanilla | None = None,
    ) -> list[str]:
        alcance = alcance or AlcanceDeLaPlanilla(
            tipo=ALCANCE_COMPLETA, declarado_por="sin alcance declarado", n_del_pool=len(filas)
        )
        vacías, inválidas, de_más = [], [], []
        for n, fila in enumerate(filas, 1):
            grado = (fila.get("grado") or "").strip()
            par = (
                (fila.get("consulta_id") or "").strip(),
                (fila.get("artwork_id") or "").strip(),
            )
            if not grado:
                if alcance.cubre(par):
                    vacías.append((n, fila.get("consulta_id"), fila.get("artwork_id")))
            elif grado not in {str(g) for g in GRADOS}:
                inválidas.append((n, grado))
            elif not alcance.cubre(par):
                de_más.append((n, fila.get("consulta_id"), fila.get("artwork_id")))
        problemas = []
        if vacías and alcance.es_completa:
            problemas.append(
                f"quedan **{len(vacías)} filas sin `grado`** (p. ej. la fila {vacías[0][0]}: "
                f"{vacías[0][1]} / {vacías[0][2]}). 🔑 **Así es como sabés que terminaste**: "
                f"mientras falte una, el import no pasa. Una celda vacía **no se lee como "
                f"0** — un 0 es *'no cumple'*, que es un juicio."
            )
        elif vacías:
            problemas.append(
                f"quedan **{len(vacías)} filas sin `grado` dentro del alcance "
                f"`{alcance.tipo}`** (p. ej. la fila {vacías[0][0]}: {vacías[0][1]} / "
                f"{vacías[0][2]}), que son {alcance.n} de las {alcance.n_del_pool} del pool. "
                f"El alcance se validó contra lo que lo declara y **falta parte de él**: una "
                f"celda vacía dentro del alcance **no se lee como 0** — un 0 es *'no "
                f"cumple'*, que es un juicio."
            )
        if de_más:
            problemas.append(
                f"hay **{len(de_más)} filas con `grado` FUERA del alcance `{alcance.tipo}`** "
                f"(p. ej. la fila {de_más[0][0]}: {de_más[0][1]} / {de_más[0][2]}). El "
                f"alcance se valida en las **dos** direcciones a propósito: *ni más filas "
                f"llenas, ni menos*. Un juicio de más sobre un par que ya tenía consenso no "
                f"es un desempate — es el tercer anotador **corrigiendo** el consenso ajeno, "
                f"que es exactamente lo que el consenso y la precisión del 2026-10-12 de la evaluación "
                f"dicen que el desempate **no** es."
            )
        for n, grado in inválidas[:5]:
            problemas.append(
                f"la fila {n} tiene `grado = {grado!r}` y los únicos valores son "
                f"{list(GRADOS)} (`la evaluación § Escala`): "
                + " · ".join(f"**{g}** = {SIGNIFICADO_DE_LA_ESCALA[g]}" for g in sorted(
                    GRADOS, reverse=True))
            )
        return problemas

    def _validar_referencias(
        self, filas: Sequence[Mapping[str, str]], *, set_de_consultas: SetDeConsultas
    ) -> list[str]:
        declarados = {c.consulta_id for c in set_de_consultas.consultas()}
        problemas = []
        for consulta_id in sorted({(f.get("consulta_id") or "").strip() for f in filas}):
            if consulta_id not in declarados:
                problemas.append(
                    f"el `consulta_id` {consulta_id!r} **no resuelve** en el set "
                    f"{set_de_consultas.identidad} ({len(declarados)} consultas declaradas). "
                    f"Es la referencia que resuelve, y el detector comprometido era el `%` de ítems sin juzgar de "
                    f"la tabla de ablación — o sea **después** de gastar la anotación."
                )
        return problemas

    def _congelar(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, juicios: Sequence[Juicio],
        planilla: Path, set_de_consultas: SetDeConsultas,
        alcance: AlcanceDeLaPlanilla | None = None,
    ) -> ResultadoDeImportación:
        ident = identidad.identidad_del_aporte(
            split_id=pool.split_id, rubrica_id=rubrica.rubrica_id, anotador=anotador
        )
        ruta = AporteDeAnotador.ruta_de(self.rutas.juicios, ident)
        nuevo = AporteDeAnotador(
            procedencia=self._procedencia(
                identidad_del_aporte=ident,
                pool=pool,
                rubrica=rubrica,
                anotador=anotador,
                juicios=juicios,
                planilla=planilla,
                set_de_consultas=set_de_consultas,
                alcance=alcance,
            ),
            juicios=tuple(juicios),
        )
        nuevo.validar()

        if self.almacén.existe(ruta):
            previo = AporteDeAnotador.leer(self.almacén, self.rutas.juicios, ident)
            if {j.par: (j.grado, j.nota) for j in previo.juicios} == {
                j.par: (j.grado, j.nota) for j in nuevo.juicios
            }:
                return ResultadoDeImportación(
                    aporte=previo,
                    ruta=self.almacén.ruta(ruta),
                    ya_estaba=True,
                    n_juicios=len(previo.juicios),
                    distribucion=previo.por_grado(),
                    alcance=alcance,
                )
            raise ErrorDeArtefacto(
                f"ya hay un aporte congelado en {self.almacén.ruta(ruta)} **con otros "
                f"juicios**.\n"
                f"  Clase 3: *no se reescribe y no se recalcula* (la regla de que un id determina su contenido). Lo que hay en "
                f"disco es trabajo humano y este import lo perdería.\n"
                f"  Si cambió un grado a propósito, lo que cambió es el instrumento o la "
                f"persona: **otra rúbrica** (el acuerdo entre anotadores con κ < 0.6 manda refinar y re-anotar) o "
                f"**otra etiqueta de anotador**. Las dos derivan **otro** aporte y los dos "
                f"conviven, que es justo lo que compró."
            )
        escrita = self.almacén.escribir_texto(ruta, nuevo.a_texto())
        return ResultadoDeImportación(
            aporte=nuevo,
            ruta=escrita,
            ya_estaba=False,
            n_juicios=len(nuevo.juicios),
            distribucion=nuevo.por_grado(),
            alcance=alcance,
        )

    def _procedencia(
        self, *, identidad_del_aporte: str, pool: Pool, rubrica: Rúbrica, anotador: str,
        juicios: Sequence[Juicio], planilla: Path, set_de_consultas: SetDeConsultas,
        alcance: AlcanceDeLaPlanilla | None = None,
    ) -> dict:
        por_grado = {f"grado_{g}": sum(1 for j in juicios if j.grado == g) for g in GRADOS}
        return {
            "artefacto": "AporteDeAnotador",
            "version_del_aporte": dj.VERSIÓN_DEL_APORTE,
            "clase_de_identidad": 3,
            "identidad": identidad_del_aporte,
            "split_id": pool.split_id,
            "rubrica_id": rubrica.rubrica_id,
            "anotador": anotador,
            "nota_de_identidad": dj.nota_de_identidad(),
            "pool_id": pool.pool_id,
            "estado_del_pool": pool.estado,
            "set_de_consultas": set_de_consultas.identidad,
            "planilla": ruta_publicable(planilla),
            "fecha_de_import": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": commit_actual(),
            "rubrica": rubrica.cita(self.rutas.rubricas),
            "conteos": {"n_juicios": len(juicios), "por_grado": por_grado},
            "alcance": (alcance or AlcanceDeLaPlanilla(
                tipo=ALCANCE_COMPLETA,
                declarado_por="sin alcance declarado",
                n_del_pool=len(pool.items),
            )).a_json(),
            "validacion_del_import": {
                "requisito": "la política de trazado",
                "pares_comparados_contra": pool.pool_id,
                "secuencia_exacta": True,
                "nota": (
                    "la política de trazado: el conjunto de pares importado es **igual** al exportado para este "
                    "pool_id, en el **mismo orden** (la columna `fila` es el orden mezclado y "
                    "es contenido), y cada `consulta_id` resuelve en el set del mismo split "
                    "(la referencia que resuelve). Sin esto, la procedencia del pool declarada en el aporte era 'garantizado por único camino' y el camino tiene "
                    "un humano adentro. ⚠️ Y esto NO es *todas las filas tienen grado*: esa "
                    "es otra propiedad y se verifica contra el alcance declarado."
                ),
            },
        }

def _leer_csv(texto: str, *, ruta) -> tuple[tuple[str, ...], list[dict]]:
    lector = csv.DictReader(io.StringIO(texto))
    if lector.fieldnames is None:
        raise ErrorDeArtefacto(
            f"{ruta} está vacía: no tiene ni encabezado. La planilla se exporta con "
            f"`tp3-anotar exportar`."
        )
    filas = [f for f in lector if any((v or "").strip() for v in f.values())]
    if not filas:
        raise ErrorDeArtefacto(
            f"{ruta} no tiene ni una fila con datos. **Un import vacío no es un aporte "
            f"vacío**: es una planilla que nadie llenó, y congelarla escribiría un artefacto "
            f"de clase 3 sin el trabajo humano que lo justifica."
        )
    return tuple(lector.fieldnames), filas

def _identidad_declarada(filas: Sequence[Mapping[str, str]], *, ruta) -> dict:
    primera = filas[0]
    declarada = {c: (primera.get(c) or "").strip() for c in ("pool_id", "rubrica_id", "anotador")}
    faltan = [k for k, v in declarada.items() if not v]
    if faltan:
        raise ErrorDeArtefacto(
            f"{ruta}: la primera fila no declara {faltan}. Cada fila lleva su `pool_id`, su "
            f"`rubrica_id` y su `anotador` **a propósito**: es lo que vuelve a una fila "
            f"pegada de otra ronda detectable por su propio contenido (el módulo de evaluación), y lo que "
            f"evita que el import pida a mano un dato que ya viene en el archivo."
        )
    return declarada

def _mensaje_de_rechazo(ruta, problemas: Sequence[str], pool: Pool) -> str:
    cuerpo = "\n".join(f"    · {p}" for p in problemas)
    return (
        f"🔴 **El import se rechaza**: lo importado no es lo exportado (la política de trazado).\n"
        f"  planilla : {ruta}\n"
        f"  pool     : {pool.pool_id}  ({len(pool.items)} ítems, estado {pool.estado})\n"
        f"  {len(problemas)} problema{'s' if len(problemas) != 1 else ''}:\n"
        f"{cuerpo}\n"
        f"  Por qué se rechaza en vez de importar lo que se entienda: la planilla es *un "
        f"puente de ida y vuelta* (el módulo de evaluación) y **el único camino** por el que entran los juicios. "
        f"Sin esta validación, la procedencia del pool declarada en el aporte está *garantizado por único camino* y el camino tiene un "
        f"humano adentro (la política de trazado): una fila agregada o un `artwork_id` retocado serían un "
        f"**juicio silencioso**, y lo único que lo acusaría es el `%` de ítems sin juzgar de "
        f"la tabla de ablación, que es **después**.\n"
        f"  **Nada se perdió**: la planilla no se tocó. Arreglá lo de arriba y volvé a "
        f"importar."
    )

def como_anotar(
    *, pool: Pool, rubrica: Rúbrica, anotador: str, planilla: Path, n_filas: int,
    imagenes_faltantes: Sequence[str] = (),
    veredicto_de_a5: VeredictoDeA5 | None = None,
    alcance: AlcanceDeLaPlanilla | None = None,
) -> str:
    consultas = pool.por_consulta()
    minutos = round(n_filas * SEGUNDOS_POR_JUICIO / 60)
    líneas = [
        f"# Cómo anotar — `{anotador}`, {n_filas} juicios (~{minutos} minutos)",
        "",
        f"- **Planilla**: `{planilla}`",
        f"- **Rúbrica**: `{Rúbrica.ruta_de('artifacts/ground-truth/rubricas', rubrica.rubrica_id)}`"
        f"  (`rubrica_id = {rubrica.rubrica_id}`)",
        f"- **Pool**: `{pool.pool_id}` — estado **{pool.estado}**, "
        f"{len(consultas)} consultas, "
        + (
            "**la unión de los top-5 de "
            + ", ".join(pool.nombres_de_configuracion())
            + "** (ciega: no sabés cuál trajo cada obra, y es a propósito)"
            if pool.es_congelado
            else "top-5 de una configuración"
        ),
        "",
        f"> Los ~{minutos} minutos son los **{SEGUNDOS_POR_JUICIO} s por juicio** de",
        "> `la evaluación § Pooling`. No es una meta: si una obra te toma más, tomátelo.",
        "",
    ]
    if veredicto_de_a5 is not None and veredicto_de_a5.hay_limitacion_declarada:
        decisión = veredicto_de_a5.decision
        peor = min(
            (k for k in veredicto_de_a5.kappas if k.esta_bajo_el_umbral),
            key=lambda k: k.kappa,
        )
        ruta_de_la_decision = DecisiónDeA5.ruta_de(
            "artifacts/ground-truth/rubricas", rubrica.rubrica_id
        )
        líneas += [
            f"> 🔴 **Esta rúbrica tiene un κ medido bajo el umbral del acuerdo entre anotadores: "
            f"{peor.kappa:.3f} < {peor.umbral}** sobre {peor.n} pares del pool "
            f"`{peor.pool_id}`.",
            f"> Se exporta igual porque hay una **decisión declarada** "
            f"(`salida = {decisión.salida}`,",
            f"> decidida por {decisión.decidida_por} el {decisión.fecha}).",
            ">",
            f"> 📄 **La limitación está escrita en `{ruta_de_la_decision}`",
            "> y NO se copia acá a propósito: lleva el κ **por categoría**, qué",
            "> consultas quedaron en el estrato pre-juzgado y **cómo resolvió cada anotador",
            "> anterior** los casos dudosos — o sea que **pre-juzga lo que vos tenés que",
            "> juzgar**. Si la querés leer, **leela después de anotar**: lo que mide tu aporte",
            "> es la rúbrica, y un juicio informado por los juicios ajenos no la mide.",
            ">",
            "> ⚠️ **Anotá igual y anotá bien.** Lo que el κ bajo dice es que la rúbrica no",
            "> alcanzó para que dos personas coincidan, no que tus juicios no valgan: usá la",
            "> columna `nota` cada vez que dudes — **es el insumo para refinarla** (el acuerdo entre anotadores).",
            "",
        ]
    if alcance is not None and not alcance.es_completa:
        líneas += [
            f"> 🔴 **Tu alcance es el DESEMPATE: {alcance.n} de las {alcance.n_del_pool} "
            f"filas** (el tercer anotador).",
            f"> Son las que la columna `{ALCANCE}` marca como `{alcance.tipo}`; las que dicen",
            f"> `{FUERA_DEL_ALCANCE}` **se dejan vacías** y el import lo verifica en las dos",
            "> direcciones: ni más filas llenas, ni menos.",
            ">",
            "> 🔑 **Y es un juicio tuyo, no una corrección del de nadie.** No vas a ver qué",
            "> pusieron los anotadores anteriores, y es a propósito: tu grado **resuelve** el",
            "> par. Juzgá el par contra el pedido como si fuera el primero.",
            "",
        ]
    líneas += [
        "## 1 · Leé la rúbrica **antes** de abrir la planilla",
        "",
        "κ es siempre *κ sobre una rúbrica*: dos personas que no leyeron la misma no",
        "produjeron juicios comparables. Si algo de la rúbrica no se entiende, **anotalo en",
        "la columna `nota`** en vez de resolverlo por tu cuenta: en el piloto eso **es el",
        "resultado**, y en la campaña definitiva es lo que explica un κ bajo.",
        "",
        "## 2 · Abrí la planilla",
        "",
        "Es un CSV: LibreOffice, Excel, Google Sheets o un editor de texto. **Importalo con",
        "separador `,` y comillas `\"`, y todas las columnas como texto** (si no, un",
        "`artwork_id` como `aic:111728` puede quedar reinterpretado).",
        "",
        "## 3 · Llená **dos** columnas, y solo dos",
        "",
        "| columna | qué poné |",
        "|---|---|",
        "| **`grado`** | "
        + " · ".join(f"`{g}` = {SIGNIFICADO_DE_LA_ESCALA[g]}" for g in sorted(GRADOS, reverse=True))
        + " |",
        "| **`nota`** | opcional, y **vale mucho acá**: por qué dudaste, qué regla faltó. Es "
        "el insumo para refinar la rúbrica |",
        "",
        "Las demás columnas son de lectura:",
        "",
        "| columna | para qué |",
        "|---|---|",
        "| `pedido` | **lo que el usuario pidió.** Es contra esto que se juzga |",
        "| `imagen_de_consulta` | la **imagen original** del pedido (un archivo del repo) |",
        "| `imagen_de_la_obra` | la **obra recuperada**, que es la que estás juzgando |",
        "| `obra`, `artista`, `cultura`, `anio`, `museo` | la ficha, para los pedidos "
        "culturales |",
        "| `source_url` | la página del museo, si querés ver la obra en grande |",
        "| `fila`, `pool_id`, `rubrica_id`, `anotador`, `consulta_id`, `artwork_id` | la "
        "identidad. **No se tocan** |",
        f"| `{ALCANCE}` | **qué filas te tocan**: `{FUERA_DEL_ALCANCE}` = dejala "
        f"vacía. **No se toca** |",
        "",
        "## 4 · 🔴 Tres cosas que **rompen** el import",
        "",
        "1. **No agregues ni borres filas.** Son exactamente "
        f"{n_filas}"
        + (
            f", y las que llenás son las {alcance.n} de tu alcance."
            if alcance is not None and not alcance.es_completa
            else "."
        ),
        "2. **No cambies un `artwork_id` ni un `consulta_id`.**",
        "3. **No reordenes las filas** (ni ordenes una columna sola: los grados se despegan).",
        "   Si ya las ordenaste, ordená de nuevo por `fila` de 1 a "
        f"{n_filas} y listo.",
        "",
        "No es burocracia: la planilla es el **único** camino por el que entran los juicios,",
        "así que una fila de más sería un juicio que nadie pidió y nadie vería (la política de trazado).",
        "",
        "## 5 · Cómo sabés que terminaste",
        "",
        "```bash",
        f"poetry run tp3-anotar importar --planilla {planilla}",
        "```",
        "",
        (
            f"Si pasa, imprime **{alcance.n}/{alcance.n}** juicios —tu alcance— y congela"
            if alcance is not None and not alcance.es_completa
            else f"Si pasa, imprime **{n_filas}/{n_filas}** y congela"
        )
        + " tus juicios en",
        "`artifacts/ground-truth/juicios/`. Si falta una celda, **no pasa** y dice cuál: eso",
        "es la señal de que terminaste, no una cuenta mental.",
        "",
    ]
    if alcance is not None and not alcance.es_completa:
        líneas += [
            "## 6 · Y después, el **consenso** (no el κ)",
            "",
            "Tu aporte **no entra al κ** y eso no es un detalle de implementación: κ de Cohen",
            "es entre **dos** anotadores, y la pareja ya está declarada",
            f"({', '.join(alcance.pareja_del_kappa) or 'en el `kappa-…json`'}). Lo que tu",
            "grado hace es **resolver** el par en el consenso (la evaluación, el desempate del tercero: *el desempate",
            "es una anotación más, no una corrección del consenso*). Cuando importes:",
            "",
            "```bash",
            f"poetry run tp3-anotar kappa --designacion {pool.designacion} "
            f"--rubrica {rubrica.rubrica_id}",
            "```",
            "",
            "y el consenso se **recalcula** —es una lectura, no un agregado—: los pares que",
            "decían `sin_consenso` pasan a decir `desempate` con tu grado. El κ no se mueve.",
            "",
        ]
    else:
        líneas += [
            "## 6 · Y después, el κ",
            "",
            "Hacen falta **dos** personas: κ mide el acuerdo entre dos anotaciones",
            "independientes. **No se consulten mientras anotan** — un acuerdo conversado no "
            "mide",
            "la rúbrica, mide la conversación. Cuando las dos importaron:",
            "",
            "```bash",
            f"poetry run tp3-anotar kappa --designacion {pool.designacion} "
            f"--rubrica {rubrica.rubrica_id}",
            "```",
            "",
            "Y lo que salga, **sale**: un κ bajo honestamente reportado **es un hallazgo**",
            "(el acuerdo entre anotadores). Si κ < 0.6 la rúbrica es ambigua: se refina, se deriva otra",
            f"`rubrica_id` y se re-anotan estos {n_filas}"
            + (
                " — y por eso el piloto existe: cuesta estos juicios y no los del pool "
                "congelado."
                if not pool.es_congelado
                else ". Para eso corrió antes la **ronda piloto**: refinar la rúbrica con 20 "
                "juicios en vez de con estos."
            ),
            "",
        ]
    if imagenes_faltantes:
        líneas += [
            "---",
            "",
            f"> ⚠️ **{len(imagenes_faltantes)} archivos de imagen no están en disco** (p. ej.",
            f"> `{imagenes_faltantes[0]}`). `data/imagenes/` es **regenerable y no**",
            "> **versionado** (la regla de qué se versiona): si falta, corré `poetry run tp3-ingesta` o mirá la obra",
            "> por `source_url`. Las imágenes **de consulta** sí están en el repo.",
            "",
        ]
    return "\n".join(líneas)
