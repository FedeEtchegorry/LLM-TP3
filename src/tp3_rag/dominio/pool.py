from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DEL_POOL = "2"

ESTADOS: tuple[str, ...] = ("piloto", "congelado")

FRACCION_DUPLICADA_DE_A5 = 0.20

SEGUNDOS_POR_JUICIO = 15

COLUMNAS = (
    "pool_id",
    "fila",
    "consulta_id",
    "artwork_id",
    "obra",
    "artista",
    "cultura",
    "anio",
    "museo",
    "source_url",
)

CLAVES_OBLIGATORIAS_DEL_JSON = (
    "identidad",
    "split_id",
    "designacion",
    "estado",
    "configuraciones",
    "semilla_de_mezcla",
    "regla",
    "origen",
    "conteos",
)

def aporte_por_configuracion(
    origen: Sequence[Mapping[str, Any]], nombres: Sequence[str]
) -> dict[str, int]:
    salida = {nombre: 0 for nombre in nombres}
    for o in origen:
        salida[o["configuracion"]] = salida.get(o["configuracion"], 0) + 1
    return salida

def exclusivos_por_configuracion(
    origen: Sequence[Mapping[str, Any]], nombres: Sequence[str]
) -> dict[str, int]:
    por_par: dict[tuple[str, str], set[str]] = {}
    for o in origen:
        por_par.setdefault((o["consulta_id"], o["artwork_id"]), set()).add(
            o["configuracion"]
        )
    salida = {nombre: 0 for nombre in nombres}
    for configuraciones in por_par.values():
        if len(configuraciones) == 1:
            solo = next(iter(configuraciones))
            salida[solo] = salida.get(solo, 0) + 1
    return salida

def items_del_top5_sin_juzgar(
    origen: Sequence[Mapping[str, Any]],
    pares: frozenset[tuple[str, str]],
    nombres: Sequence[str],
) -> dict[str, int]:
    sin_juzgar = {nombre: 0 for nombre in nombres}
    for o in origen:
        if (o["consulta_id"], o["artwork_id"]) not in pares:
            sin_juzgar[o["configuracion"]] = sin_juzgar.get(o["configuracion"], 0) + 1
    return sin_juzgar

@dataclass(frozen=True)
class ÍtemDePool:
    fila: int
    consulta_id: str
    artwork_id: str
    ficha: Mapping[str, Any]

    @property
    def par(self) -> tuple[str, str]:
        return (self.consulta_id, self.artwork_id)

@dataclass(frozen=True)
class Pool:
    procedencia: dict
    items: tuple[ÍtemDePool, ...]

    @property
    def pool_id(self) -> str:
        return self.procedencia["identidad"]

    @property
    def split_id(self) -> str:
        return self.procedencia["split_id"]

    @property
    def designacion(self) -> str:
        return self.procedencia["designacion"]

    @property
    def estado(self) -> str:
        return self.procedencia["estado"]

    @property
    def es_congelado(self) -> bool:
        return self.estado == "congelado"

    @property
    def configuraciones(self) -> tuple[dict, ...]:
        return tuple(dict(c) for c in self.procedencia["configuraciones"])

    @property
    def semilla_de_mezcla(self) -> int:
        return int(self.procedencia["semilla_de_mezcla"])

    @property
    def regla(self) -> dict:
        return dict(self.procedencia["regla"])

    @property
    def pares(self) -> frozenset[tuple[str, str]]:
        return frozenset(i.par for i in self.items)

    @property
    def consultas(self) -> tuple[str, ...]:
        vistas = []
        for i in self.items:
            if i.consulta_id not in vistas:
                vistas.append(i.consulta_id)
        return tuple(vistas)

    @property
    def artworks(self) -> frozenset[str]:
        return frozenset(i.artwork_id for i in self.items)

    def origen_de(self, par: tuple[str, str]) -> dict:
        return self.origenes_de(par)[0]

    def origenes_de(self, par: tuple[str, str]) -> tuple[dict, ...]:
        encontrados = tuple(
            dict(o)
            for o in self.procedencia["origen"]
            if (o["consulta_id"], o["artwork_id"]) == par
        )
        if not encontrados:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: el par {par} no tiene origen declarado. El origen va "
                f"aparte y completo: cegar al anotador y perder el dato son dos cosas "
                f"distintas, y sin él el acuerdo no se puede estratificar por origen."
            )
        return encontrados

    def variantes_de_origen(self) -> tuple[str, ...]:
        return tuple(sorted({c["variante_id"] for c in self.configuraciones}))

    def nombres_de_configuracion(self) -> tuple[str, ...]:
        return tuple(c["configuracion"] for c in self.configuraciones)

    def aporte_por_configuracion(self) -> dict[str, int]:
        return aporte_por_configuracion(
            self.procedencia["origen"], self.nombres_de_configuracion()
        )

    def exclusivos_por_configuracion(self) -> dict[str, int]:
        return exclusivos_por_configuracion(
            self.procedencia["origen"], self.nombres_de_configuracion()
        )

    def items_del_top5_sin_juzgar(self) -> dict[str, int]:
        return items_del_top5_sin_juzgar(
            self.procedencia["origen"], self.pares, self.nombres_de_configuracion()
        )

    def por_consulta(self) -> dict[str, tuple[ÍtemDePool, ...]]:
        salida: dict[str, list[ÍtemDePool]] = {}
        for i in self.items:
            salida.setdefault(i.consulta_id, []).append(i)
        return {k: tuple(v) for k, v in salida.items()}

    @staticmethod
    def ruta_de(raíz: str | Path, pool_id: str) -> Path:
        return Path(raíz) / f"{pool_id}.csv"

    @staticmethod
    def ruta_de_procedencia(raíz: str | Path, pool_id: str) -> Path:
        return Path(raíz) / f"{pool_id}.json"

    def a_texto(self) -> str:
        import csv
        import io

        buffer = io.StringIO()
        escritor = csv.DictWriter(buffer, fieldnames=list(COLUMNAS), lineterminator="\n")
        escritor.writeheader()
        for i in self.items:
            escritor.writerow(
                {
                    "pool_id": self.pool_id,
                    "fila": i.fila,
                    "consulta_id": i.consulta_id,
                    "artwork_id": i.artwork_id,
                    **{c: i.ficha.get(c, "") for c in COLUMNAS[4:]},
                }
            )
        return buffer.getvalue()

    @classmethod
    def leer(cls, almacén, raíz: str | Path, pool_id: str) -> "Pool":
        import csv
        import io

        procedencia = almacén.leer_json(
            cls.ruta_de_procedencia(raíz, pool_id),
            claves_obligatorias=CLAVES_OBLIGATORIAS_DEL_JSON,
        )
        texto = almacén.leer_texto(cls.ruta_de(raíz, pool_id))
        items = []
        for n, fila in enumerate(csv.DictReader(io.StringIO(texto)), 1):
            faltan = [c for c in ("fila", "consulta_id", "artwork_id") if not fila.get(c)]
            if faltan:
                raise ErrorDeArtefacto(
                    f"{pool_id}.csv: la fila {n} no tiene {faltan}. La `fila` **es "
                    f"contenido**: es el orden mezclado, y es lo que la planilla compara al "
                    f"importar."
                )
            items.append(
                ÍtemDePool(
                    fila=int(fila["fila"]),
                    consulta_id=fila["consulta_id"].strip(),
                    artwork_id=fila["artwork_id"].strip(),
                    ficha={c: (fila.get(c) or "") for c in COLUMNAS[4:]},
                )
            )
        pool = cls(procedencia=procedencia, items=tuple(items))
        pool.validar()
        if pool.pool_id != pool_id:
            raise ErrorDeArtefacto(
                f"el pool de {pool_id} declara adentro `identidad = {pool.pool_id}`. El "
                f"nombre y el contenido no pueden discrepar."
            )
        return pool

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.csv")))

    @classmethod
    def del_split(cls, almacén, raíz: str | Path, split_id: str) -> tuple[str, ...]:
        return tuple(
            p for p in cls.existentes(almacén, raíz)
            if identidad.partir_identidad_del_pool(p)[0] == split_id
        )

    def validar(self) -> None:
        esperada = identidad.identidad_del_pool(
            split_id=self.split_id, designacion=self.designacion
        )
        if self.pool_id != esperada:
            raise ErrorDeArtefacto(
                f"`identidad` dice {self.pool_id!r} y sus dos ingredientes dan "
                f"{esperada!r}. La identidad de un pool **es** el compromiso de abrir una "
                f"ronda de anotación."
            )
        if self.estado not in ESTADOS:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: `estado` = {self.estado!r} no es uno de {list(ESTADOS)}. "
                f"El estado es lo que **impide que un pool provisorio se confunda con uno "
                f"definitivo, y por eso no tiene default."
            )
        if self.es_congelado and len(self.configuraciones) < 2:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: declara `estado = congelado` y **una sola** configuración "
                f"de origen ({list(self.nombres_de_configuracion())}). El pool "
                f"abre con eso: *\"no se anota todo el corpus **ni solo los resultados de una "
                f"configuración** — eso sesgaría la comparación a favor de la configuración "
                f"anotada\"*. 🔴 Se verifica **al leer** y no solo al derivar, porque es el "
                f"invariante que mantiene honesta la comparación de la tabla de ablación y el artefacto lo "
                f"puede decir de sí mismo: el estado y las configuraciones viajan los dos "
                f"adentro."
            )
        if not self.items:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: el pool no tiene ni un ítem. Un pool vacío no es una "
                f"orden de trabajo."
            )
        pares = [i.par for i in self.items]
        repetidos = sorted({p for p in pares if pares.count(p) > 1})
        if repetidos:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: los pares {repetidos[:3]} están **dos veces**. El pool es "
                f"la unión deduplicada de los top-5: un par dos "
                f"veces gasta anotación humana dos veces sobre el mismo juicio."
            )
        filas = sorted(i.fila for i in self.items)
        if filas != list(range(1, len(self.items) + 1)):
            raise ErrorDeArtefacto(
                f"{self.pool_id}: las `fila` no son 1..{len(self.items)} sin huecos. La "
                f"fila **es el orden mezclado y es contenido**: es lo que la política de trazado compara al "
                f"importar, así que un hueco o un salto haría que *lo importado sea lo "
                f"exportado* no se pueda decidir."
            )
        self._validar_origen()
        self._validar_conteos()

    def _validar_origen(self) -> None:
        filas = [
            (o["consulta_id"], o["artwork_id"], o["configuracion"])
            for o in self.procedencia["origen"]
        ]
        repetidas = sorted({f for f in filas if filas.count(f) > 1})
        if repetidas:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: el origen declara dos veces {repetidas[:3]}. Es **una fila "
                f"por `(consulta, obra, configuración)`**: una configuración no puede traer el "
                f"mismo par dos veces en un top-5 deduplicado, y contarlo dos veces inflaría "
                f"el aporte de esa fila de la tabla de ablación."
            )
        del_origen = {(o["consulta_id"], o["artwork_id"]) for o in self.procedencia["origen"]}
        if del_origen != self.pares:
            faltan = sorted(self.pares - del_origen)
            sobran = sorted(del_origen - self.pares)
            raise ErrorDeArtefacto(
                f"{self.pool_id}: el origen y el pool ciego **no cubren los mismos pares** "
                f"(sin origen: {faltan[:3]} · origen sin ítem: {sobran[:3]}). Son dos "
                f"archivos de un agregado: si no cierran, o hay un ítem que nadie puede "
                f"estratificar por configuración de origen o hay un origen que "
                f"nadie va a juzgar."
            )
        configuraciones = {c["configuracion"] for c in self.configuraciones}
        huérfanos = sorted(
            {o["configuracion"] for o in self.procedencia["origen"]} - configuraciones
        )
        if huérfanos:
            raise ErrorDeArtefacto(
                f"{self.pool_id}: el origen nombra las configuraciones {huérfanos}, que no "
                f"están en `configuraciones`. El pool **declara qué configuraciones lo "
                f"formaron: una que aparece solo en una fila no está declarada."
            )

    def _validar_conteos(self) -> None:
        c = self.procedencia["conteos"]
        esperados = {
            "n_items": len(self.items),
            "n_consultas": len(self.consultas),
            "n_obras_distintas": len(self.artworks),
        }
        derivadas = {
            "aporte_por_configuracion": self.aporte_por_configuracion(),
            "exclusivos_por_configuracion": self.exclusivos_por_configuracion(),
        }
        esperados.update({k: v for k, v in derivadas.items() if k in c})
        for clave, esperado in esperados.items():
            if c.get(clave) != esperado:
                raise ErrorDeArtefacto(
                    f"{self.pool_id}: `conteos.{clave}` dice {c.get(clave)!r} y el "
                    f"contenido dice {esperado}. Un conteo que no cierra contra su propio "
                    f"contenido se rechaza — y acá el conteo es el "
                    f"presupuesto de anotación humana**."
                )

    def filas_para_mostrar(self):
        for i in self.items:
            orígenes = self.origenes_de(i.par)
            yield (
                str(i.fila),
                i.consulta_id,
                i.artwork_id,
                (i.ficha.get("obra") or "—")[:40],
                " ".join(f"{o['configuracion']}#{o['rank']}" for o in orígenes),
            )

def nota_de_identidad() -> str:
    return (
        "Ancla de trabajo humano. La identidad es `split_id` + "
        "`designación`, y la designación **la asigna una persona** cuando decide abrir una "
        "ronda de anotación. El **estado**, las **configuraciones de origen** y la "
        "**semilla de mezcla** NO son ingredientes y van acá adentro: el estado no alcanza "
        "porque dos pools congelados del mismo split tendrían el mismo id (la verificación de la norma de los vectores), y las "
        "configuraciones tampoco porque **ningún conjunto de entradas determina el "
        "contenido de un pool: los top-5 no son deterministas** (). Congelar y agregar un "
        "finalista son derivar otro pool, nunca reescribir este."
    )

def nota_de_estado(estado: str) -> str:
    if estado == "piloto":
        return (
            "Pool PROVISORIO de la ronda piloto. NO es el "
            "pool congelado: no es la unión de los top-5 de C0–C4, es el top-5 de una sola "
            "configuración sobre 4 de las 30 consultas. Lo que produce son 20 juicios que "
            "**no cuentan contra los ~540**, que **se descartan si la rúbrica cambia** —es "
            "el resultado que el piloto existe para producir— y que, si no cambia, se "
            "**importan** para los pares que además aparezcan en el pool congelado, porque "
            "un juicio es sobre el par (consulta, obra) y no sobre la configuración que lo "
            "trajo. Armar el pool congelado."
        )
    return (
        "Pool **congelado**: la unión deduplicada de los top-5 de las configuraciones "
        "declaradas arriba. Una vez anotado no se toca; agregar un finalista es **derivar "
        "otro pool** con otra designación y juzgar la diferencia (pooling incremental, "
        "la campaña definitiva)."
    )

def nota_de_ceguera(configuraciones: Sequence[Mapping[str, Any]]) -> str:
    base = (
        "El CSV es **ciego**: no lleva ni la configuración de origen ni el `rank`. El origen "
        "vive en el `.json` porque *cegar al anotador y perder el dato son dos cosas "
        "distintas — el acuerdo se estratifica por configuración de "
        "origen y necesita el dato. "
    )
    if len(configuraciones) == 1:
        base += (
            f"⚠️ **Acá la única configuración es {configuraciones[0]['configuracion']}, así "
            f"que no hay origen que cegar**: cegarlo, cuando el pool sea la unión "
            f"de cinco. "
        )
    return base + (
        "**Pero el `rank` se ciega igual**, y es una decisión de esta pieza: el puesto del "
        "ranking filtra más que el nombre de la configuración. Ver *'se mezcla el orden'* en "
        "lo que se juzga es la obra, no el puesto que el sistema le dio."
    )
