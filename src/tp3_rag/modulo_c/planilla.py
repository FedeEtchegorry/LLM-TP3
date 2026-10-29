from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.dominio import juicios as dj
from tp3_rag.dominio.consultas import SetDeConsultas
from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA, AporteDeAnotador, Juicio
from tp3_rag.dominio.pool import SEGUNDOS_POR_JUICIO, Pool
from tp3_rag.dominio.rubrica import Rúbrica
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

@dataclass
class ResultadoDeImportación:
    aporte: AporteDeAnotador
    ruta: Path
    ya_estaba: bool
    n_juicios: int = 0
    distribucion: Mapping[str, int] = field(default_factory=dict)

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
    ) -> ResultadoDeExportación:
        identidad.etiqueta_declarada(anotador, campo="anotador")
        self._exigir_rubrica_usable(rubrica)
        self._exigir_aporte_libre(pool=pool, rubrica=rubrica, anotador=anotador)

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
            pool=pool, rubrica=rubrica, anotador=anotador, set_de_consultas=set_de_consultas
        )
        destino = Path(salida) if salida is not None else self._ruta_por_omision(
            pool=pool, rubrica=rubrica, anotador=anotador
        )
        buffer = io.StringIO()
        escritor = csv.DictWriter(buffer, fieldnames=list(COLUMNAS), lineterminator="\n")
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
        )

    def _exigir_rubrica_usable(self, rubrica: Rúbrica) -> None:
        if not rubrica.es_usable_para_anotar:
            raise ErrorDeConstrucción(
                f"la rúbrica {rubrica.rubrica_id!r} declara `estado = {rubrica.estado}` y "
                f"**no es usable para anotar**.\n"
                f" El texto de la rúbrica lo escribe el grupo (el módulo de evaluación: *lo que escribe acá "
                f"es trabajo humano*), y exportar una planilla contra una plantilla gastaría "
                f"20 juicios bajo un instrumento a medio escribir — que, exactamente, lo "
                f"que la ronda piloto existe para **no** hacer con los ~540.\n"
                f"  Llenala y poné `estado = \"completa\"` en su front matter. Mirá qué "
                f"falta con `poetry run tp3-anotar rubrica`."
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
            f" Es **clase 3**: no se reescribe y no se recalcula, y lo que hay "
            f"adentro es trabajo humano. Exportar otra planilla con la misma identidad "
            f"produciría 20 juicios que el import va a rechazar — así que se avisa **antes** "
            f"de gastarlos.\n"
            f"  Las dos salidas legítimas, y las dos cuestan lo que dicen:\n"
            f"    · **otro anotador** (`--anotador <otra-etiqueta>`): es otro aporte, y es "
            f"lo que κ necesita — hacen falta **dos** personas;\n"
            f" · **otra rúbrica** (`--rubrica <otra>`): es la rama del acuerdo entre anotadores con κ < 0.6. "
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
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, set_de_consultas: SetDeConsultas
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
        problemas += self._validar_grados(filas, ruta=ruta)
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
        )
        return self._congelar(
            pool=pool, rubrica=rubrica, anotador=declarada["anotador"],
            juicios=juicios, planilla=ruta, set_de_consultas=set_de_consultas,
        )

    def _validar_columnas(self, cabecera: Sequence[str], *, ruta) -> None:
        faltan = [c for c in COLUMNAS if c not in cabecera]
        de_más = [c for c in cabecera if c not in COLUMNAS]
        if faltan or de_más:
            raise ErrorDeArtefacto(
                f"{ruta}: las columnas no son las de la planilla exportada.\n"
                f" faltan: {faltan}\n"
                f" de más: {de_más}\n"
                f"  Las de identidad ({list(IDENTIDAD)}) son **lo exportado** y se comparan "
                f"exactas; `grado` y `nota` son lo único que se llena; el resto es la ayuda "
                f"para poder juzgar sin leer código. Una clave desconocida es un error "
                f", y acá una columna de menos es una planilla con la que no se "
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
                f"**igual** al que salió: una fila agregada es un juicio sobre un par "
                f"que nadie puso en ninguna orden de trabajo, y eso es roto por la "
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
                    f" El orden mezclado **es contenido del pool** —el diseño de la evaluación "
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
                        f"bajo, da un κ que no significa nada."
                    )
        return problemas

    def _validar_grados(self, filas: Sequence[Mapping[str, str]], *, ruta) -> list[str]:
        vacías, inválidas = [], []
        for n, fila in enumerate(filas, 1):
            grado = (fila.get("grado") or "").strip()
            if not grado:
                vacías.append((n, fila.get("consulta_id"), fila.get("artwork_id")))
            elif grado not in {str(g) for g in GRADOS}:
                inválidas.append((n, grado))
        problemas = []
        if vacías:
            problemas.append(
                f"quedan **{len(vacías)} filas sin `grado`** (p. ej. la fila {vacías[0][0]}: "
                f"{vacías[0][1]} / {vacías[0][2]}). 🔑 **Así es como sabés que terminaste**: "
                f"mientras falte una, el import no pasa. Una celda vacía **no se lee como "
                f"0** — un 0 es *'no cumple'*, que es un juicio."
            )
        for n, grado in inválidas[:5]:
            problemas.append(
                f"la fila {n} tiene `grado = {grado!r}` y los únicos valores son "
                f"{list(GRADOS)}: "
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
                    f"Es, y el detector comprometido era el `%` de ítems sin juzgar de "
                    f"la tabla de ablación — o sea **después** de gastar la anotación."
                )
        return problemas

    def _congelar(
        self, *, pool: Pool, rubrica: Rúbrica, anotador: str, juicios: Sequence[Juicio],
        planilla: Path, set_de_consultas: SetDeConsultas,
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
                )
            raise ErrorDeArtefacto(
                f"ya hay un aporte congelado en {self.almacén.ruta(ruta)} **con otros "
                f"juicios**.\n"
                f" Clase 3: *no se reescribe y no se recalcula*. Lo que hay en "
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
        )

    def _procedencia(
        self, *, identidad_del_aporte: str, pool: Pool, rubrica: Rúbrica, anotador: str,
        juicios: Sequence[Juicio], planilla: Path, set_de_consultas: SetDeConsultas,
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
            "validacion_del_import": {
                "requisito": "politica_de_trazado",
                "pares_comparados_contra": pool.pool_id,
                "secuencia_exacta": True,
                "nota": (
                    "la política de trazado: el conjunto de pares importado es **igual** al exportado para este "
                    "pool_id, en el **mismo orden** (la columna `fila` es el orden mezclado y "
                    "es contenido), y cada `consulta_id` resuelve en el set del mismo split "
                    "(). Sin esto, era 'garantizado por único camino' y el camino tiene "
                    "un humano adentro."
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
            f"pegada de otra ronda detectable por su propio contenido, y lo que "
            f"evita que el import pida a mano un dato que ya viene en el archivo."
        )
    return declarada

def _mensaje_de_rechazo(ruta, problemas: Sequence[str], pool: Pool) -> str:
    cuerpo = "\n".join(f"    · {p}" for p in problemas)
    return (
        f"🔴 **El import se rechaza**: lo importado no es lo exportado.\n"
        f" planilla: {ruta}\n"
        f" pool: {pool.pool_id} ({len(pool.items)} ítems, estado {pool.estado})\n"
        f"  {len(problemas)} problema{'s' if len(problemas) != 1 else ''}:\n"
        f"{cuerpo}\n"
        f"  Por qué se rechaza en vez de importar lo que se entienda: la planilla es *un "
        f"puente de ida y vuelta* y **el único camino** por el que entran los juicios. "
        f"Sin esta validación, está *garantizado por único camino* y el camino tiene un "
        f"humano adentro: una fila agregada o un `artwork_id` retocado serían un "
        f"**juicio silencioso**, y lo único que lo acusaría es el `%` de ítems sin juzgar de "
        f"la tabla de ablación, que es **después**.\n"
        f"  **Nada se perdió**: la planilla no se tocó. Arreglá lo de arriba y volvé a "
        f"importar."
    )

def como_anotar(
    *, pool: Pool, rubrica: Rúbrica, anotador: str, planilla: Path, n_filas: int,
    imagenes_faltantes: Sequence[str] = (),
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
        "> el diseño de la evaluación. No es una meta: si una obra te toma más, tomátelo.",
        "",
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
        "",
        "## 4 · 🔴 Tres cosas que **rompen** el import",
        "",
        "1. **No agregues ni borres filas.** Son exactamente "
        f"{n_filas}.",
        "2. **No cambies un `artwork_id` ni un `consulta_id`.**",
        "3. **No reordenes las filas** (ni ordenes una columna sola: los grados se despegan).",
        "   Si ya las ordenaste, ordená de nuevo por `fila` de 1 a "
        f"{n_filas} y listo.",
        "",
        "No es burocracia: la planilla es el **único** camino por el que entran los juicios,",
        "así que una fila de más sería un juicio que nadie pidió y nadie vería.",
        "",
        "## 5 · Cómo sabés que terminaste",
        "",
        "```bash",
        f"poetry run tp3-anotar importar --planilla {planilla}",
        "```",
        "",
        f"Si pasa, imprime **{n_filas}/{n_filas}** y congela tus juicios en",
        "`artifacts/ground-truth/juicios/`. Si falta una celda, **no pasa** y dice cuál: eso",
        "es la señal de que terminaste, no una cuenta mental.",
        "",
        "## 6 · Y después, el κ",
        "",
        "Hacen falta **dos** personas: κ mide el acuerdo entre dos anotaciones",
        "independientes. **No se consulten mientras anotan** — un acuerdo conversado no mide",
        "la rúbrica, mide la conversación. Cuando las dos importaron:",
        "",
        "```bash",
        f"poetry run tp3-anotar kappa --designacion {pool.designacion} "
        f"--rubrica {rubrica.rubrica_id}",
        "```",
        "",
        "Y lo que salga, **sale**: un κ bajo honestamente reportado **es un hallazgo**",
        "(los criterios de entrega el acuerdo entre anotadores). Si κ < 0.6 la rúbrica es ambigua: se refina, se deriva otra",
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
            "> **versionado**: si falta, corré `poetry run tp3-ingesta` o mirá la obra",
            "> por `source_url`. Las imágenes **de consulta** sí están en el repo.",
            "",
        ]
    return "\n".join(líneas)

