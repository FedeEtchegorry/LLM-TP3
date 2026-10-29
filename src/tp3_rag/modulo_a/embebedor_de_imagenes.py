from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.puertos.embedding_imagen import IdentidadDelModelo

VERSIÓN_DEL_ARTEFACTO = "1"

class CachéDeEmbeddingsDeImagen:
    def __init__(self, raíz: Path, modelo: IdentidadDelModelo) -> None:
        self.raíz = Path(raíz)
        self.modelo = modelo

    def ruta(self, artwork_id: str, hash_bytes: str) -> Path:
        return (
            self.raíz
            / _seguro(self.modelo.checkpoint)
            / _seguro(self.modelo.version_del_preprocesado)
            / f"{_seguro(artwork_id)}-{_seguro(hash_bytes)}.npy"
        )

    def leer(self, artwork_id: str, hash_bytes: str):
        import numpy as np

        ruta = self.ruta(artwork_id, hash_bytes)
        if not ruta.exists():
            return None
        return np.load(ruta)

    def escribir(self, artwork_id: str, hash_bytes: str, vector) -> Path:
        import numpy as np

        ruta = self.ruta(artwork_id, hash_bytes)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_name(ruta.name + ".parcial")
        with tmp.open("wb") as fh:
            np.save(fh, np.asarray(vector, dtype=np.float32))
        tmp.replace(ruta)
        return ruta

def _seguro(valor: str) -> str:
    return valor.replace("/", "__").replace(":", "_")

@dataclass
class ResultadoDeEmbebido:
    artefacto: dict
    ruta: Path
    n_embebidas: int
    n_desde_cache: int
    img_por_segundo: float
    fallas: list = field(default_factory=list)

class EmbebedorDeImágenes:
    def __init__(self, cfg, *, puerto, almacén, observabilidad) -> None:
        self.cfg = cfg
        self.puerto = puerto
        self.almacén = almacén
        self.obs = observabilidad
        self.modelo = puerto.identidad()
        self.caché = CachéDeEmbeddingsDeImagen(
            almacén.ruta(cfg["rutas.embeddings_imagen"]), self.modelo
        )
        self.raíz_de_imágenes = almacén.ruta(cfg["rutas.imagenes"])
        self.batch = int(cfg["embebido.batch"])

    def universo_con_imagen(self, registros: Sequence[RegistroCanónico]):
        return [r for r in registros if r.tiene_imagen_real]

    def ruta_de_imagen(self, artwork_id: str) -> Path:
        return self.raíz_de_imágenes / f"{artwork_id.replace(':', '_')}.jpg"

    def correr(
        self, registros: Sequence[RegistroCanónico], *, catalogo_id: str, limite: int | None = None
    ) -> ResultadoDeEmbebido:
        pendientes = self.universo_con_imagen(registros)
        pendientes.sort(key=lambda r: r.artwork_id)
        if limite:
            pendientes = pendientes[:limite]

        self.obs.decir(
            f"universo con imagen: {len(pendientes)}  "
            f"(el corpus es otro número y es del manifiesto — guardrail 🔒)\n"
            f"checkpoint: {self.modelo.checkpoint}  "
            f"preproc={self.modelo.version_del_preprocesado}  batch={self.batch}"
        )

        hechas, desde_caché, fallas = 0, 0, []
        lote: list[tuple[RegistroCanónico, bytes]] = []
        t0 = time.perf_counter()
        t_cpu = 0.0

        for registro in pendientes:
            hash_bytes = registro.imagen.hash_bytes
            if self.caché.leer(registro.artwork_id, hash_bytes) is not None:
                desde_caché += 1
                hechas += 1
                continue
            ruta = self.ruta_de_imagen(registro.artwork_id)
            if not ruta.exists():
                fallas.append({"artwork_id": registro.artwork_id, "motivo": "bytes_ausentes"})
                continue
            lote.append((registro, ruta.read_bytes()))
            if len(lote) >= self.batch:
                t_cpu += self._embeber_lote(lote, fallas)
                hechas += len(lote)
                lote = []
                self._progreso(hechas, len(pendientes), t_cpu, desde_caché)
        if lote:
            t_cpu += self._embeber_lote(lote, fallas)
            hechas += len(lote)
        self.obs.cerrar_progreso()

        segundos_de_pared = time.perf_counter() - t0
        n_calculadas = hechas - desde_caché
        ips = n_calculadas / t_cpu if t_cpu > 0 else 0.0

        artefacto = self._armar(
            catalogo_id=catalogo_id,
            n_universo_con_imagen=len(pendientes),
            n_embebidas=hechas,
            n_desde_cache=desde_caché,
            n_calculadas=n_calculadas,
            segundos_de_cpu=t_cpu,
            segundos_de_pared=segundos_de_pared,
            img_por_segundo=ips,
            fallas=fallas,
        )
        ordinal = artefacto["ordinal"]
        ruta = self.almacén.escribir_json(
            Path(self.cfg["rutas.artefactos_embebido"])
            / f"{catalogo_id}-{_seguro(self.modelo.checkpoint)}-{ordinal}.json",
            artefacto,
        )
        return ResultadoDeEmbebido(
            artefacto=artefacto,
            ruta=ruta,
            n_embebidas=hechas,
            n_desde_cache=desde_caché,
            img_por_segundo=ips,
            fallas=fallas,
        )

    def _embeber_lote(self, lote, fallas: list) -> float:
        t0 = time.perf_counter()
        try:
            vectores = self.puerto.embeber_imagenes(
                [(r.artwork_id, datos) for r, datos in lote]
            )
        except Exception as e:
            segundos = time.perf_counter() - t0
            for registro, datos in lote:
                try:
                    vector = self.puerto.embeber_imagen(datos, artwork_id=registro.artwork_id)
                except Exception as individual:
                    fallas.append(
                        {
                            "artwork_id": registro.artwork_id,
                            "motivo": f"{type(individual).__name__}",
                            "detalle": str(individual)[:200],
                        }
                    )
                    continue
                self.caché.escribir(registro.artwork_id, registro.imagen.hash_bytes, vector)
            if not fallas:
                raise ErrorDeConstrucción(
                    f"el lote falló ({type(e).__name__}: {e}) y ninguna obra falló sola"
                ) from None
            return time.perf_counter() - t0 + segundos
        for (registro, _), vector in zip(lote, vectores):
            self.caché.escribir(registro.artwork_id, registro.imagen.hash_bytes, vector)
        return time.perf_counter() - t0

    def _progreso(self, hechas: int, total: int, t_cpu: float, desde_caché: int) -> None:
        calculadas = hechas - desde_caché
        ips = calculadas / t_cpu if t_cpu > 0 else 0.0
        falta = (total - hechas) / ips / 60 if ips > 0 else 0.0
        self.obs.progreso(
            f"{hechas}/{total} · {ips:.1f} img/s · caché={desde_caché} · "
            f"faltan ~{falta:.1f} min"
        )

    def _armar(
        self,
        *,
        catalogo_id: str,
        n_universo_con_imagen: int,
        n_embebidas: int,
        n_desde_cache: int,
        n_calculadas: int,
        segundos_de_cpu: float,
        segundos_de_pared: float,
        img_por_segundo: float,
        fallas: list,
    ) -> dict:
        plantilla = (
            f"{self.cfg['rutas.artefactos_embebido']}/"
            f"{catalogo_id}-{_seguro(self.modelo.checkpoint)}-{{ordinal}}.json"
        )
        ordinal, _ = self.almacén.reservar_ordinal(plantilla)

        return {
            "artefacto": "ReporteDeEmbebido",
            "version_del_artefacto": VERSIÓN_DEL_ARTEFACTO,
            "clase_de_identidad": 2,
            "catalogo_id": catalogo_id,
            "checkpoint": self.modelo.checkpoint,
            "ordinal": ordinal,
            "nota_de_identidad": (
                "Clase 2 y artefacto **propio**, no un campo del `ReporteDeIngesta` "
                ": el reporte de ingesta **sella** al terminar la ingesta y "
                "el embebido corre después, así que meterle la `n` embebidas adentro "
                "rompía la semántica de sellado y la regla de un escritor por artefacto (un escritor por artefacto: el "
                "Embebedor no es la Ingesta). **el dataset documentado y reproducible lee los dos.** La identidad es "
                "`catalogo_id` + checkpoint + **ordinal**; la fecha y el throughput van "
                "adentro porque no están determinados por las entradas. ⚠️ El ordinal "
                "**no sube** al `variante_id`: si subiera, cada re-corrida del embebedor "
                "sería una variante nueva (la tabla de identidades, recuadro 🔴)."
            ),
            "modelo": self.modelo.a_json(),
            "embebido": {
                "n_universo_con_imagen": n_universo_con_imagen,
                "n_embebidas": n_embebidas,
                "n_calculadas_en_esta_corrida": n_calculadas,
                "n_desde_cache": n_desde_cache,
                "n_fallas": len(fallas),
                "fallas": fallas[:50],
                "img_por_segundo": round(img_por_segundo, 2),
                "segundos_de_embebido": round(segundos_de_cpu, 1),
                "segundos_de_pared": round(segundos_de_pared, 1),
                "batch": self.batch,
                "threads": self.cfg["embebido.threads"],
            },
            "nota_de_a6": (
                "`n_embebidas` es del **universo con imagen**, no del corpus: se embebe "
                "todo lo que tiene imagen real porque el criterio de inclusión filtra por "
                "texto y el texto no cambia el embedding de la imagen. "
                "Que difiera de `n indexadas` **es el resultado esperado** y es lo que el dataset documentado y reproducible "
                "verifica; la otra cifra vive en el `RegistroDeÍndices`."
            ),
            "a9c_fallas_de_embedding_por_formato": len(
                [f for f in fallas if "Formato" in f.get("motivo", "")]
            ),
            "nota_de_a9c": (
                "Es la mitad del conteo de conversiones de imagen que dejó en `null` por no tener embebedor: "
                "*0 fallas de embedding por formato*. La otra mitad —grises, alfa y "
                "conversión a RGB— la emitió y **no se copia acá**: vive en el "
                "`ReporteDeIngesta`, cuya identidad la determina."
            ),
            "libro_mayor": self.obs.libro_mayor(),
            "procedencia": {
                "fecha_de_embebido": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": _commit(),
                "capas_de_configuracion": list(self.cfg.capas),
            },
            "configuracion_resuelta": self.cfg.a_dict(),
            "config_id": self.cfg.config_id,
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
