from __future__ import annotations

from typing import Mapping, Sequence

from tp3_rag.dominio.salida import (
    ABSTENCIÓN_POR_REFS_COLGADAS,
    FormatoDeSalida,
    ObligaciónIncumplida,
    ObraCitada,
    Referencia,
    SalidaCitada,
)
from tp3_rag.modulo_b.etapa import Etapa, EstadoDeConsulta

CAMPOS_DE_LA_FICHA = ("title", "artist", "date_display", "year", "medium", "style", "culture")

def _atribucion(fragmento, formato: FormatoDeSalida) -> str | None:
    if not formato.exige_atribucion(fragmento.license):
        return None
    autor = fragmento.artist or "autor no declarado por el museo"
    título = fragmento.title or fragmento.artwork_id
    museo = formato.nombre_de_museo(fragmento.museum)
    licencia = formato.nombre_de_licencia(fragmento.license)
    return f"“{título}”, {autor}. {museo}. Texto bajo {licencia} — {fragmento.source_url}"

class RenderizadorDeSalida(Etapa):
    NOMBRE = "renderizador"
    PRECONDICIONES = ("top5",)
    CLAVES_DE_CONFIGURACIÓN = ("salida",)
    DEGRADA = False
    POLÍTICA = (
        "las **obligaciones** no degradan: fallan (las obligaciones de la cita). Una `ref` colgada **sí** degrada: "
        "se cae la afirmación y se cuenta (el descarte contado de afirmaciones), y si se cae la última de una obra, esa "
        "obra sale con **abstención**."
    )

    def __init__(self, *, formato: FormatoDeSalida) -> None:
        self.formato = formato

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        evidencia = dict(estado.evidencia or {})
        explicaciones = {e.artwork_id: e for e in (estado.explicacion or ())}
        candidatas: dict[str, object] = {}
        for nombre in ("texto", *sorted(estado.fuentes_de_ranking), "visual"):
            for c in estado.fuentes_de_ranking.get(nombre, ()):
                candidatas[c.artwork_id] = c
        score_fusionado = {
            f.artwork_id: f.score_fusionado for f in (estado.ranking_fusionado or ())
        }

        obras: list[ObraCitada] = []
        descartes: list[Mapping] = []
        fragmentos_por_id: dict[str, object] = {}

        try:
            for rango, artwork_id in enumerate(estado.top5 or (), start=1):
                fragmentos = tuple(evidencia.get(artwork_id) or ())
                for f in fragmentos:
                    fragmentos_por_id[f.fragment_id] = f
                explicación = explicaciones.get(artwork_id)

                if explicación is not None:
                    explicación, caídas = explicación.sin_refs_colgadas(
                        [f.fragment_id for f in fragmentos],
                        texto_de_abstención=ABSTENCIÓN_POR_REFS_COLGADAS,
                    )
                    descartes.extend(caídas)

                obras.append(
                    self._una_obra(
                        artwork_id=artwork_id,
                        rango=rango,
                        candidata=candidatas.get(artwork_id),
                        score=score_fusionado.get(artwork_id),
                        fragmentos=fragmentos,
                        explicación=explicación,
                    )
                )

            salida = SalidaCitada(
                consulta_id=estado.entrada.consulta_id,
                obras=tuple(obras),
                formato=self.formato,
            )
            salida.verificar_obligaciones(fragmentos_por_id)
        except ObligaciónIncumplida as e:
            marcado = estado.descartar_afirmaciones(descartes) if descartes else estado
            return marcado.marcar_fallida(self.NOMBRE, e.obligacion).degradar(
                self.NOMBRE, causa=e.detalle, politica="falla: es una obligación, no calidad"
            )

        nuevo = estado.descartar_afirmaciones(descartes) if descartes else estado
        if descartes:
            nuevo = nuevo.degradar(
                self.NOMBRE,
                causa=(
                    f"el descarte contado de afirmaciones: se descartaron {len(descartes)} afirmación(es) con `refs` que no "
                    f"resuelven contra la evidencia de esta consulta. **Es lo que `el conteo de referencias colgadas` "
                    f"cuenta**, y por eso no falla: hacerlo fallar entregaría 0 obras en "
                    f"vez de 5 (rompiendo el criterio de respuesta garantizada) y volvería circular a el conteo de referencias colgadas."
                ),
                politica="se cae la afirmación entera; si era la última de la obra, abstención",
            )
        return nuevo.con(salida=salida)

    def _una_obra(
        self,
        *,
        artwork_id: str,
        rango: int,
        candidata,
        fragmentos: Sequence,
        explicación,
        score=None,
    ) -> ObraCitada:
        por_id = {f.fragment_id: f for f in fragmentos}
        numeración: dict[str, int] = {}
        líneas: list[str] = []

        for afirmación in (explicación.afirmaciones if explicación else ()):
            marcas: list[str] = []
            for ref in afirmación.refs:
                if ref not in numeración:
                    numeración[ref] = len(numeración) + 1
                marcas.append(f"[{numeración[ref]}]")
            sufijo = f" {' '.join(marcas)}" if marcas else ""
            líneas.append(f"{afirmación.texto}{sufijo}")

        referencias = tuple(
            Referencia(
                n=n,
                fragment_id=fid,
                texto_citado=por_id[fid].texto,
                title=por_id[fid].title,
                artist=por_id[fid].artist,
                source_url=por_id[fid].source_url,
                license=por_id[fid].license,
                museum=por_id[fid].museum,
                atribucion=_atribucion(por_id[fid], self.formato),
            )
            for fid, n in sorted(numeración.items(), key=lambda kv: kv[1])
            if fid in por_id
        )

        metadata = dict(candidata.metadata) if candidata is not None else {}
        ficha = {c: metadata[c] for c in CAMPOS_DE_LA_FICHA if metadata.get(c) not in (None, "")}

        return ObraCitada(
            artwork_id=artwork_id,
            rango=rango,
            score=(
                score
                if score is not None
                else (candidata.score if candidata is not None else None)
            ),
            ficha=ficha,
            source_url=metadata.get("source_url") or "",
            lineas=tuple(líneas),
            abstencion=(explicación.abstencion if explicación else _SIN_EXPLICACIÓN),
            referencias=referencias,
        )

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "formato": self.formato.a_json(),
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "nota": (
                "La numeración `[n]` nace acá y es lo que vuelve a `el conteo de referencias colgadas` un evento "
                "contable. las obligaciones de la cita se verifica **contra el texto renderizado** porque el criterio de atribución "
                "compromete *en toda superficie donde se muestren*. Y *fallar* es escribir "
                "`fallida` en el estado, no lanzar: la clausura de la etapa sigue valiendo."
            ),
        }

_SIN_EXPLICACIÓN = (
    "Esta obra se devuelve sin explicación: la etapa de generación no corrió en esta "
    "configuración o degradó, y queda registrado en la corrida. La ficha y el enlace al "
    "museo son del catálogo y no dependen del modelo."
)
