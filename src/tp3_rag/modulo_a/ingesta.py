from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tp3_rag.dominio.catalogo import prefijar
from tp3_rag.modulo_a.normalizador import VERSIÓN_DEL_NORMALIZADOR, Normalizador
from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture
from tp3_rag.modulo_a.preparador_de_imagen import PreparadorDeImagen
from tp3_rag.modulo_a.reporte_de_ingesta import ReporteDeIngesta
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.cliente_http import ClienteHTTP
from tp3_rag.puertos.museos.aic import AdaptadorAIC
from tp3_rag.puertos.museos.cma import AdaptadorCMA

ADAPTADORES = {"aic": AdaptadorAIC, "cma": AdaptadorCMA}

@dataclass
class ResultadoDeIngesta:
    catalogo_id: str
    ordinal: int
    registros: list
    reporte: ReporteDeIngesta
    ruta_del_reporte: Path
    ruta_del_catálogo: Path

class Ingesta:
    def __init__(
        self,
        cfg,
        credenciales,
        *,
        observabilidad: Observabilidad | None = None,
        cliente: ClienteHTTP | None = None,
        raíz: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.credenciales = credenciales
        self.obs = observabilidad or Observabilidad()
        self.raíz = raíz or RAÍZ_DEL_REPO
        self.cliente = cliente or ClienteHTTP.desde_configuración(cfg, credenciales)
        self.normalizador = Normalizador(cfg)
        self.culture = NormalizadorDeCulture(cfg["normalizacion.culture.criterio"])
        self.preparador = PreparadorDeImagen(
            self.cliente, self.raíz / cfg["rutas.imagenes"], self.obs
        )
        self.almacén = AlmacenamientoEnArchivos(self.raíz)

    def correr(self) -> ResultadoDeIngesta:
        museos = list(self.cfg["ingesta.museos"])
        tipos = list(self.cfg["ingesta.tipos"])
        límite = self.cfg["ingesta.limite_n"] or None
        segundos: dict = {}

        self.obs.decir(
            f"museos={museos} · tipos={tipos} · "
            f"límite_n={límite if límite is not None else 'sin límite (universo completo)'}"
        )

        registros, crudos, preparadas, coberturas = [], [], [], []
        t_total = time.perf_counter()

        for museo in museos:
            fábrica = ADAPTADORES.get(museo)
            if fábrica is None:
                raise ErrorDeConstrucción(
                    f"no hay adaptador para el museo {museo!r}; hay {sorted(ADAPTADORES)}"
                )
            adaptador = fábrica(self.cliente, self.cfg)

            t0 = time.perf_counter()
            self.obs.decir(f"\n[{museo}] listando…")
            ids = adaptador.listar(tipos, self.cfg["ingesta.licencia"], límite)
            cobertura = adaptador.cobertura_de_paginación()
            coberturas.append(cobertura)
            segundos[f"listar.{museo}"] = round(time.perf_counter() - t0, 2)
            self.obs.decir(
                f"[{museo}] {cobertura.n_alcanzadas} de {cobertura.n_total_declarada} declaradas"
                f" · no alcanzadas={cobertura.n_no_alcanzadas}"
                f"{' (truncado por límite_n)' if cobertura.truncado_por_limite else ''}"
                f" · {segundos[f'listar.{museo}']}s"
            )

            t0 = time.perf_counter()
            for i, id_del_museo in enumerate(ids, 1):
                crudo = adaptador.traer(id_del_museo)
                artwork_id = prefijar(museo, id_del_museo)
                url = self.normalizador.url_de_imagen(museo, crudo)

                preparada = self.preparador.preparar(artwork_id, url)
                preparadas.append(preparada)

                registro = self.normalizador.normalizar(crudo, museo, preparada.hash_bytes)
                registros.append(registro)
                crudos.append({"artwork_id": artwork_id, "crudo": crudo})

                if i % 25 == 0 or i == len(ids):
                    hecho = time.perf_counter() - t0
                    self.obs.progreso(
                        f"[{museo}] {i}/{len(ids)} · {i / max(hecho, 1e-9):.1f} obras/s"
                        f" · img ok={sum(1 for p in preparadas if p.ok)}"
                    )
            self.obs.cerrar_progreso()
            segundos[f"traer_y_normalizar.{museo}"] = round(time.perf_counter() - t0, 2)

        segundos["total"] = round(time.perf_counter() - t_total, 2)

        json_de_registros = [r.a_json() for r in registros]
        catalogo_id = identidad.catalogo_id(
            museos=museos,
            tipos=tipos,
            version_del_normalizador=VERSIÓN_DEL_NORMALIZADOR,
            registros=json_de_registros,
        )
        self.obs.decir(f"\ncatalogo_id = {catalogo_id}  (hash de los registros, no de la fecha)")

        ruta_catálogo = self._escribir_catálogo(catalogo_id, json_de_registros, crudos, preparadas)

        plantilla = f"{self.cfg['rutas.artefactos_ingesta']}/{catalogo_id}-{{ordinal}}.json"
        ordinal, ruta_reporte = self.almacén.reservar_ordinal(plantilla)

        reporte = ReporteDeIngesta.armar(
            catalogo_id=catalogo_id,
            ordinal=ordinal,
            fecha_de_ingesta=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            commit=_commit(),
            configuración=self.cfg,
            credenciales_presentes=self.credenciales.constancia(),
            registros=registros,
            coberturas_de_paginación=coberturas,
            preparadas=preparadas,
            a9b=self.culture.a9b(registros),
            version_del_normalizador=VERSIÓN_DEL_NORMALIZADOR,
            segundos=segundos,
        )
        self.almacén.escribir_json(ruta_reporte.relative_to(self.raíz), reporte.a_json())

        return ResultadoDeIngesta(
            catalogo_id=catalogo_id,
            ordinal=ordinal,
            registros=registros,
            reporte=reporte,
            ruta_del_reporte=ruta_reporte,
            ruta_del_catálogo=ruta_catálogo,
        )

    def _escribir_catálogo(self, catalogo_id, json_de_registros, crudos, preparadas) -> Path:
        base = Path(self.cfg["rutas.catalogo"]) / catalogo_id
        self.almacén.escribir_jsonl(
            base / "registros.jsonl", sorted(json_de_registros, key=lambda r: r["artwork_id"])
        )
        self.almacén.escribir_jsonl(
            base / "crudos.jsonl", sorted(crudos, key=lambda c: c["artwork_id"])
        )
        self.almacén.escribir_jsonl(
            base / "validacion-de-imagenes.jsonl",
            sorted((p.a_json() for p in preparadas), key=lambda p: p["artwork_id"]),
        )
        self.almacén.escribir_json(
            base / "catalogo.json",
            {
                "artefacto": "CatalogoCanonico",
                "clase_de_identidad": 1,
                "catalogo_id": catalogo_id,
                "museos": sorted(self.cfg["ingesta.museos"]),
                "tipos": sorted(self.cfg["ingesta.tipos"]),
                "version_del_normalizador": VERSIÓN_DEL_NORMALIZADOR,
                "hash_de_registros": identidad.hash_de_registros(json_de_registros),
                "n_registros": len(json_de_registros),
            },
        )
        return self.almacén.ruta(base)

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
