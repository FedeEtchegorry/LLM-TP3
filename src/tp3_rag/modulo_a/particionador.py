from __future__ import annotations

import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.dominio.split import VERSIÓN_DEL_MANIFIESTO, ManifiestoDeSplit
from tp3_rag.modulo_a.selector_de_corpus import Elegibles
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDeConstrucción

MÉTODO_DE_REPARTO = "cuota_de_hare"
MÉTODO_DE_SELECCIÓN = "sha256(semilla|artwork_id) ascendente, los k más chicos"

@dataclass(frozen=True)
class Tramo:
    nombre: str
    desde: int | None
    hasta: int | None

    def contiene(self, year: int) -> bool:
        return (self.desde is None or year >= self.desde) and (
            self.hasta is None or year <= self.hasta
        )

    def a_json(self) -> dict:
        return {"nombre": self.nombre, "desde": self.desde, "hasta": self.hasta}

@dataclass(frozen=True)
class DefiniciónDeEstrato:
    ejes: tuple[str, ...]
    campo_de_periodo: str
    tramos: tuple[Tramo, ...]
    etiqueta_sin_periodo: str

    @classmethod
    def desde_configuración(cls, cfg) -> "DefiniciónDeEstrato":
        crudos = cfg["split.estrato.tramos"]
        if not crudos:
            raise ErrorDeConfiguración("`split.estrato.tramos` está vacío: no hay período")
        tramos = tuple(
            Tramo(t["nombre"], t.get("desde"), t.get("hasta")) for t in crudos
        )
        definición = cls(
            ejes=tuple(cfg["split.estrato.ejes"]),
            campo_de_periodo=cfg["split.estrato.campo_de_periodo"],
            tramos=tramos,
            etiqueta_sin_periodo=cfg["split.estrato.etiqueta_sin_periodo"],
        )
        definición.validar()
        return definición

    def validar(self) -> None:
        nombres = [t.nombre for t in self.tramos]
        if len(set(nombres)) != len(nombres):
            raise ErrorDeConfiguración(f"`split.estrato.tramos` repite nombres: {nombres}")
        if self.etiqueta_sin_periodo in nombres:
            raise ErrorDeConfiguración(
                f"`split.estrato.etiqueta_sin_periodo` = "
                f"{self.etiqueta_sin_periodo!r} colisiona con un tramo"
            )
        ordenados = sorted(self.tramos, key=lambda t: (t.desde is not None, t.desde or 0))
        if ordenados[0].desde is not None or ordenados[-1].hasta is not None:
            raise ErrorDeConfiguración(
                "los tramos tienen que cubrir sin techo ni piso: el primero sin `desde` "
                "y el último sin `hasta`. Un `year` fuera de todo tramo quedaría sin "
                "estrato, y el síntoma no apuntaría a la causa."
            )
        for anterior, siguiente in zip(ordenados, ordenados[1:]):
            if anterior.hasta is None or siguiente.desde is None:
                raise ErrorDeConfiguración(
                    f"solo el primer tramo puede no tener `desde` y solo el último no "
                    f"tener `hasta`: {anterior.nombre!r} / {siguiente.nombre!r}"
                )
            if siguiente.desde != anterior.hasta + 1:
                raise ErrorDeConfiguración(
                    f"los tramos {anterior.nombre!r} y {siguiente.nombre!r} no son "
                    f"contiguos ({anterior.hasta} → {siguiente.desde}): "
                    f"{'se solapan' if siguiente.desde <= anterior.hasta else 'hay un hueco'}"
                )

    def periodo_de(self, registro: RegistroCanónico) -> str:
        year = getattr(registro, self.campo_de_periodo, None)
        if year is None:
            return self.etiqueta_sin_periodo
        for tramo in self.tramos:
            if tramo.contiene(year):
                return tramo.nombre
        raise ErrorDeConstrucción(
            f"{registro.artwork_id}: `{self.campo_de_periodo}` = {year} no cae en ningún "
            f"tramo declarado. Los tramos cubren sin huecos por validación, así que esto "
            f"es un bug y no un dato raro."
        )

    def estrato_de(self, registro: RegistroCanónico) -> str:
        partes = {"museo": registro.museum, "periodo": self.periodo_de(registro)}
        faltantes = sorted(set(self.ejes) - set(partes))
        if faltantes:
            raise ErrorDeConfiguración(
                f"`split.estrato.ejes` nombra ejes que no existen: {faltantes}. "
                f"Hay {sorted(partes)}."
            )
        return "|".join(partes[eje] for eje in self.ejes)

    def a_json(self) -> dict:
        return {
            "ejes": list(self.ejes),
            "campo_de_periodo": self.campo_de_periodo,
            "tramos": [t.a_json() for t in self.tramos],
            "etiqueta_sin_periodo": self.etiqueta_sin_periodo,
            "nota": (
                "Seis tramos de la periodización corriente del arte occidental; el corte "
                "de 1875 es la primera exposición impresionista (1874). `sin_periodo` es "
                "para las obras sin `year`: 1 de 2.894 (aic:45753), que `date_start` de "
                "AIC devuelve en `null`."
            ),
        }

def cuota_de_hare(tamaños: Mapping[str, int], proporción: float) -> dict[str, int]:
    if not 0 < proporción < 1:
        raise ErrorDeConfiguración(
            f"`split.proporcion_held_out` = {proporción}: tiene que estar en (0, 1). "
            f"Un held-out de 0 deja el set de evaluación sin obras y uno de 1 deja el "
            f"corpus vacío."
        )
    objetivo = math.floor(sum(tamaños.values()) * proporción)
    base = {clave: math.floor(n * proporción) for clave, n in tamaños.items()}
    sobran = objetivo - sum(base.values())
    orden = sorted(
        tamaños, key=lambda clave: (-(tamaños[clave] * proporción - base[clave]), clave)
    )
    for clave in orden[:sobran]:
        base[clave] += 1
    return base

@dataclass
class ResultadoDeSplit:
    manifiesto: ManifiestoDeSplit
    ruta: Path
    ya_estaba_congelado: bool
    elegibles: Elegibles

class Particionador:
    def __init__(self, cfg, *, almacén, observabilidad=None) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.obs = observabilidad
        self.semilla = int(cfg["split.semilla"])
        self.proporción = float(cfg["split.proporcion_held_out"])
        self.estrato = DefiniciónDeEstrato.desde_configuración(cfg)
        self.raíz_de_splits = cfg["rutas.artefactos_splits"]

    def partir(self, elegibles: Elegibles) -> dict[str, dict]:
        por_estrato: dict[str, list[RegistroCanónico]] = {}
        for registro in elegibles.registros:
            por_estrato.setdefault(self.estrato.estrato_de(registro), []).append(registro)

        cupos = cuota_de_hare(
            {clave: len(v) for clave, v in por_estrato.items()}, self.proporción
        )

        salida: dict[str, dict] = {}
        for clave, registros in sorted(por_estrato.items()):
            ordenados = sorted(
                (r.artwork_id for r in registros),
                key=lambda aid: identidad.rango_de_seleccion(self.semilla, aid),
            )
            k = cupos[clave]
            salida[clave] = {
                "held_out": sorted(ordenados[:k]),
                "corpus": sorted(ordenados[k:]),
            }
        return salida

    def congelar(
        self,
        elegibles: Elegibles,
        *,
        catalogo_id: str,
        version_del_normalizador: str,
        derivar_otro: bool = False,
    ) -> ResultadoDeSplit:
        partición = self.partir(elegibles)
        ids_corpus = sorted(i for e in partición.values() for i in e["corpus"])
        ids_held_out = sorted(i for e in partición.values() for i in e["held_out"])

        sid = identidad.split_id(
            criterio_de_inclusion=elegibles.criterio,
            semilla=self.semilla,
            proporcion_held_out=self.proporción,
            definicion_de_estrato=self.estrato.a_json(),
            ids_corpus=ids_corpus,
            ids_held_out=ids_held_out,
        )

        existentes = ManifiestoDeSplit.existentes(self.almacén, self.raíz_de_splits)
        if sid in existentes:
            manifiesto = ManifiestoDeSplit.leer(self.almacén, self.raíz_de_splits, sid)
            return ResultadoDeSplit(
                manifiesto=manifiesto,
                ruta=self.almacén.ruta(ManifiestoDeSplit.ruta_de(self.raíz_de_splits, sid)),
                ya_estaba_congelado=True,
                elegibles=elegibles,
            )
        if existentes and not derivar_otro:
            raise ErrorDeConstrucción(
                f"ya hay {len(existentes)} split congelado ({', '.join(existentes)}) y esta "
                f"corrida derivaría otro ({sid}).\n"
                f"  **Derivar otro split no es gratis**: el `SetDeConsultas` se ancla por "
                f"`split_id` y cada juicio entra por él (la tabla de identidades, clase 3), así que un split "
                f"nuevo deja los ~540 juicios del anterior sin poder compararse con nada "
                f"construido sobre este.\n"
                f"  Si es a propósito —otro criterio de inclusión, otra semilla, otros "
                f"tramos—, pasá `--derivar-otro`. Que haya que decirlo **es** el "
                f"solo pasa si alguien deriva otro a propósito."
            )

        contenido = self._armar(
            split_id=sid,
            elegibles=elegibles,
            partición=partición,
            ids_corpus=ids_corpus,
            ids_held_out=ids_held_out,
            catalogo_id=catalogo_id,
            version_del_normalizador=version_del_normalizador,
        )
        manifiesto = ManifiestoDeSplit(contenido)
        manifiesto.validar()
        ruta = self.almacén.escribir_json(
            ManifiestoDeSplit.ruta_de(self.raíz_de_splits, sid), contenido
        )
        return ResultadoDeSplit(
            manifiesto=manifiesto, ruta=ruta, ya_estaba_congelado=False, elegibles=elegibles
        )

    def _armar(
        self,
        *,
        split_id: str,
        elegibles: Elegibles,
        partición: Mapping[str, dict],
        ids_corpus: Sequence[str],
        ids_held_out: Sequence[str],
        catalogo_id: str,
        version_del_normalizador: str,
    ) -> dict:
        estratos = {
            clave: {
                "n_elegibles": len(lados["corpus"]) + len(lados["held_out"]),
                "n_corpus": len(lados["corpus"]),
                "n_held_out": len(lados["held_out"]),
                "pct_held_out": round(
                    100
                    * len(lados["held_out"])
                    / max(len(lados["corpus"]) + len(lados["held_out"]), 1),
                    1,
                ),
                "held_out_ids": list(lados["held_out"]),
            }
            for clave, lados in sorted(partición.items())
        }

        return {
            "artefacto": "ManifiestoDeSplit",
            "version_del_manifiesto": VERSIÓN_DEL_MANIFIESTO,
            "clase_de_identidad": 3,
            "split_id": split_id,
            "nota_de_identidad": (
                "Clase 3: ancla de trabajo humano. La identidad **lleva los ids de "
                "cada lado adentro**, así que la identidad *es* el compromiso: mover el "
                "split no se puede hacer callado. El `catalogo_id` **no** es ingrediente "
                "—está en `procedencia`—, que es lo que hace que una re-ingesta no mueva "
                "el split ni huerfane los ~540 juicios. No se reescribe y no se recalcula "
                ": solo cambia derivando **otro**, con otro id."
            ),
            "criterio_de_inclusion": elegibles.criterio,
            "semilla": self.semilla,
            "proporcion_held_out": self.proporción,
            "definicion_de_estrato": self.estrato.a_json(),
            "corpus_indexado": {"ids": list(ids_corpus)},
            "held_out": {"ids": list(ids_held_out)},
            "conteos": {
                "elegibles_con_texto_e_imagen": len(ids_corpus) + len(ids_held_out),
                "corpus_indexado": len(ids_corpus),
                "held_out": len(ids_held_out),
            },
            "nota_de_conteos": (
                "Este artefacto es el **dueño** de los tres (los almacenes de artefactos, la regla de qué se versiona): nadie más los "
                "copia. `n embebidas` es del `ReporteDeIngesta` y `n indexadas` del "
                "`RegistroDeÍndices`; quien necesite dos juntos **cita el id del otro** "
                "en vez de copiar el número."
            ),
            "estratos": estratos,
            "distribucion_del_held_out": _distribución(estratos, self.estrato),
            "seleccion_del_universo": elegibles.a_json(),
            "procedencia": {
                "catalogo_id": catalogo_id,
                "nota_catalogo_id": (
                    "Declarado adentro y **no** en la identidad (la tabla de identidades, clase 3): es lo "
                    "auditable, y alcanza. Si entrara, una re-ingesta movería el split."
                ),
                "fecha_de_derivacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": _commit(),
                "version_del_normalizador": version_del_normalizador,
                "metodo_de_reparto": MÉTODO_DE_REPARTO,
                "metodo_de_seleccion": MÉTODO_DE_SELECCIÓN,
                "nota_de_metodo": (
                    "El método **no es ingrediente del id** y no hace falta que lo sea: "
                    "los ids de cada lado ya están adentro, así que cambiar el reparto o "
                    "el ranking cambia el id. Baja a procedencia declarada, que es lo que "
                    "la tabla de identidades manda hacer con lo que no justifica su costo. La selección es "
                    "un **ranking por hash** y no un PRNG porque la declaración tiene que "
                    "alcanzar para re-derivar el artefacto en cualquier lenguaje."
                ),
                "capas_de_configuracion": list(self.cfg.capas),
            },
            "configuracion_resuelta": self.cfg.a_dict(),
            "config_id": self.cfg.config_id,
        }

def _distribución(estratos: Mapping[str, dict], definición: DefiniciónDeEstrato) -> dict:
    por_eje: dict[str, Counter] = {eje: Counter() for eje in definición.ejes}
    elegibles_por_eje: dict[str, Counter] = {eje: Counter() for eje in definición.ejes}
    for clave, datos in estratos.items():
        partes = dict(zip(definición.ejes, clave.split("|")))
        for eje, valor in partes.items():
            por_eje[eje][valor] += datos["n_held_out"]
            elegibles_por_eje[eje][valor] += datos["n_elegibles"]
    return {
        f"por_{eje}": {
            valor: {
                "n_elegibles": elegibles_por_eje[eje][valor],
                "n_held_out": n,
                "pct": round(100 * n / max(elegibles_por_eje[eje][valor], 1), 1),
            }
            for valor, n in sorted(por_eje[eje].items())
        }
        for eje in definición.ejes
    }

def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAÍZ_DEL_REPO,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
