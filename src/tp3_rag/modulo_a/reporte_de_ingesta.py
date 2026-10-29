from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico

VERSIÓN_DEL_REPORTE = "1"

@dataclass(frozen=True)
class ReporteDeIngesta:
    contenido: dict

    @property
    def catalogo_id(self) -> str:
        return self.contenido["catalogo_id"]

    @property
    def ordinal(self) -> int:
        return self.contenido["ordinal"]

    def a_json(self) -> dict:
        return self.contenido

    @staticmethod
    def cobertura_de_normalización(registros: Sequence[RegistroCanónico]) -> dict:
        por_museo: dict[str, dict] = {}
        for r in registros:
            m = por_museo.setdefault(r.museum, {"n": 0, **{c: 0 for c in RegistroCanónico.CAMPOS_DE_A9A}})
            m["n"] += 1
            for campo in RegistroCanónico.CAMPOS_DE_A9A:
                if r.campo_de_a9a_resuelto(campo):
                    m[campo] += 1

        def porcentajes(m: dict) -> dict:
            n = m["n"] or 1
            return {
                "n_universo_normalizado": m["n"],
                **{
                    campo: {"n": m[campo], "pct": round(100 * m[campo] / n, 1)}
                    for campo in RegistroCanónico.CAMPOS_DE_A9A
                },
            }

        total = {"n": len(registros), **{c: 0 for c in RegistroCanónico.CAMPOS_DE_A9A}}
        for m in por_museo.values():
            for campo in RegistroCanónico.CAMPOS_DE_A9A:
                total[campo] += m[campo]

        return {
            "denominador": "universo normalizado, no obras indexadas",
            "campos": list(RegistroCanónico.CAMPOS_DE_A9A),
            "total": porcentajes(total),
            "por_museo": {museo: porcentajes(m) for museo, m in sorted(por_museo.items())},
        }

    @staticmethod
    def cobertura_de_texto(registros: Sequence[RegistroCanónico]) -> dict:
        por_museo: dict[str, Counter] = {}
        n_por_museo: Counter = Counter()
        for r in registros:
            n_por_museo[r.museum] += 1
            c = por_museo.setdefault(r.museum, Counter())
            for campo in r.campos_de_texto:
                if campo.texto.strip():
                    c[campo.section] += 1
        return {
            museo: {
                "n_universo_normalizado": n_por_museo[museo],
                "secciones": {
                    section: {"n": n, "pct": round(100 * n / (n_por_museo[museo] or 1), 1)}
                    for section, n in sorted(c.items())
                },
            }
            for museo, c in sorted(por_museo.items())
        }

    @staticmethod
    def licencias_por_seccion(registros: Sequence[RegistroCanónico]) -> dict:
        cuenta: Counter = Counter()
        for r in registros:
            for campo in r.campos_de_texto:
                cuenta[(r.museum, campo.section, campo.license)] += 1
        salida: dict = {}
        for (museo, section, lic), n in sorted(cuenta.items()):
            salida.setdefault(museo, {}).setdefault(section, {})[lic] = n
        return salida

    @classmethod
    def armar(
        cls,
        *,
        catalogo_id: str,
        ordinal: int,
        fecha_de_ingesta: str,
        commit: str | None,
        configuración,
        credenciales_presentes: dict,
        registros: Sequence[RegistroCanónico],
        coberturas_de_paginación: Sequence,
        preparadas: Sequence,
        a9b: dict,
        version_del_normalizador: str,
        segundos: dict,
    ) -> "ReporteDeIngesta":
        from tp3_rag.modulo_a.preparador_de_imagen import PreparadorDeImagen

        n_por_museo = Counter(r.museum for r in registros)
        con_imagen = Counter(r.museum for r in registros if r.tiene_imagen_real)
        preparadas_por_museo: dict[str, list] = {}
        for p in preparadas:
            preparadas_por_museo.setdefault(p.artwork_id.split(":", 1)[0], []).append(p)

        universo = {"n_normalizadas": len(registros), "por_museo": {}}
        for cob in coberturas_de_paginación:
            museo = cob.museo
            ps = preparadas_por_museo.get(museo, [])
            universo["por_museo"][museo] = {
                "paginacion": cob.a_json(),
                "n_normalizadas": n_por_museo[museo],
                "n_descargadas": sum(1 for p in ps if p.ok),
                "bytes_descargados": sum(p.bytes_descargados for p in ps),
                "n_con_imagen_real": con_imagen[museo],
                "n_sin_imagen_real": n_por_museo[museo] - con_imagen[museo],
                "motivos_sin_imagen": _conteo(p.motivo for p in ps if not p.ok),
            }

        contenido = {
            "artefacto": "ReporteDeIngesta",
            "version_del_reporte": VERSIÓN_DEL_REPORTE,
            "clase_de_identidad": 2,
            "catalogo_id": catalogo_id,
            "ordinal": ordinal,
            "nota_de_identidad": (
                "Registra una ejecución. El `catalogo_id` está adentro como "
                "procedencia además de en el nombre, y la fecha y el throughput no "
                "entran en la identidad porque no están determinados por el catálogo. "
                "El ordinal no sube al `variante_id`."
            ),
            "procedencia": {
                "fecha_de_ingesta": fecha_de_ingesta,
                "commit": commit,
                "version_del_normalizador": version_del_normalizador,
                "credenciales": credenciales_presentes,
                "capas_de_configuracion": list(configuración.capas),
            },
            "configuracion_resuelta": configuración.a_dict(),
            "config_id": configuración.config_id,
            "universo": universo,
            "a9a_cobertura_de_normalizacion": cls.cobertura_de_normalización(registros),
            "a9b_culture": a9b,
            "a9c_imagen": PreparadorDeImagen.a9c(preparadas),
            "cobertura_por_campo_de_texto": cls.cobertura_de_texto(registros),
            "licencias_por_seccion": cls.licencias_por_seccion(registros),
            "embebido": None,
            "nota_de_embebido": (
                "`null` a propósito: el embebido de imágenes corre como una etapa "
                "aparte, así que `n_embebidas` y el throughput de CLIP —dos campos que "
                "este artefacto declara como suyos— todavía no existen cuando se lo "
                "sella. Un artefacto que registra una ejecución se abre, se llena y se "
                "sella, y estos dos campos llegan después de sellarlo."
            ),
            "segundos": segundos,
        }
        return cls(contenido)

def _conteo(valores) -> dict:
    return dict(sorted(Counter(v for v in valores if v).items()))
