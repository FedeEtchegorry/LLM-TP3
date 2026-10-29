from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tp3_rag.dominio.fragmento import SEPARADOR_DE_ID
from tp3_rag.nucleo.errores import ErrorDelProyecto

TIPOS: tuple[str, ...] = ("dato", "observacion")

_REF_NUMÉRICA = re.compile(r"^\s*\[?\s*\d+\s*\]?\s*$")

class ViolaciónDelContratoDeSalida(ErrorDelProyecto):
    pass

@dataclass(frozen=True)
class Afirmación:
    texto: str
    tipo: str
    refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ViolaciónDelContratoDeSalida(
                f"`tipo = {self.tipo!r}` no es uno de {list(TIPOS)}. la regla de que un dato lleva referencia reparte las "
                f"afirmaciones en exactamente dos clases, y un tercer tipo dejaría una "
                f"afirmación que ni necesita cita ni se declara observación propia."
            )
        if not str(self.texto).strip():
            raise ViolaciónDelContratoDeSalida(
                "una afirmación sin texto no es una afirmación. Si no había nada que decir "
                "sobre la obra, la forma correcta es la **abstención** del descarte contado de afirmaciones, que es un "
                "campo de la `Explicación` y no una afirmación vacía: *una obra sin "
                "afirmaciones y sin abstención es indistinguible de una obra que nadie "
                "intentó explicar*."
            )
        if self.tipo == "dato" and not self.refs:
            raise ViolaciónDelContratoDeSalida(
                f"una afirmación de tipo `dato` sin `refs`. Es la regla de honestidad "
                f"como invariante de forma: todo dato sobre la obra, el artista, la "
                f"procedencia o la historia se cita. Afirmación: {self.texto[:80]!r}"
            )
        if self.tipo == "observacion" and self.refs:
            raise ViolaciónDelContratoDeSalida(
                f"la otra dirección del mismo invariante: una `observacion` con `refs` "
                f"{list(self.refs)}. Las observaciones sobre la imagen que subió el "
                f"usuario —paleta, pincelada, composición— se presentan como propias, "
                f"sin cita: una observación con cita es exactamente la observación "
                f"visual presentada con cita falsa que la revisión humana mide."
            )
        for ref in self.refs:
            if _REF_NUMÉRICA.match(str(ref)):
                raise ViolaciónDelContratoDeSalida(
                    f"que se rompió el generador emite ids y no números: la `ref` {ref!r} es un **número**. *"
                    f"El generador emite `fragment_id`, no números; la numeración `[n]` la "
                    f"asigna el renderizador sobre la evidencia de esa consulta.* Con un "
                    f"`[7]` habría que **inferir** a qué quiso apuntar, y `el conteo de referencias colgadas` dejaría "
                    f"de ser un evento detectable para volverse una inspección."
                )
            if SEPARADOR_DE_ID not in str(ref):
                raise ViolaciónDelContratoDeSalida(
                    f"la `ref` {ref!r} no tiene la forma de un `fragment_id` "
                    f"(`{{artwork_id}}{SEPARADOR_DE_ID}{{section}}"
                    f"{SEPARADOR_DE_ID}{{chunk_index}}`). El generador emite ids, y "
                    f"un id mal formado no es una `ref` colgada: es el contrato de salida "
                    f"roto, y contarlo como referencia colgada inflaría el numerador con "
                    f"un error de forma."
                )
        if len(set(self.refs)) != len(self.refs):
            raise ViolaciónDelContratoDeSalida(
                f"la afirmación repite `refs` ({list(self.refs)}). Son un **conjunto**: "
                f"citar dos veces el mismo fragmento no agrega respaldo y le daría dos "
                f"números `[n]` distintos al mismo texto."
            )

    @property
    def es_dato(self) -> bool:
        return self.tipo == "dato"

    def a_json(self) -> dict:
        return {"texto": self.texto, "tipo": self.tipo, "refs": list(self.refs)}

    @classmethod
    def desde_json(cls, d: Mapping[str, Any]) -> "Afirmación":
        if not isinstance(d, Mapping):
            raise ViolaciónDelContratoDeSalida(
                f"una afirmación tiene que ser un objeto con `texto`, `tipo` y `refs`; "
                f"llegó {type(d).__name__}."
            )
        faltantes = sorted({"texto", "tipo"} - set(d))
        if faltantes:
            raise ViolaciónDelContratoDeSalida(
                f"a una afirmación le faltan {faltantes}. No se completa con defaults: un "
                f"`tipo` por omisión sería **elegir si hace falta citar**, que es "
                f"exactamente la decisión que la regla de que un dato lleva referencia le saca al modelo."
            )
        refs = d.get("refs") or ()
        if isinstance(refs, str):
            raise ViolaciónDelContratoDeSalida(
                f"`refs` llegó como string ({refs!r}) y es una **lista** de `fragment_id`. "
                f"Un string se iteraría carácter por carácter y daría 30 refs de una letra."
            )
        return cls(
            texto=str(d["texto"]).strip(),
            tipo=str(d["tipo"]).strip().lower(),
            refs=tuple(str(r).strip() for r in refs),
        )

@dataclass(frozen=True)
class Explicación:
    artwork_id: str
    afirmaciones: tuple[Afirmación, ...] = ()
    abstencion: str | None = None

    def __post_init__(self) -> None:
        if not self.afirmaciones and not self.abstencion:
            raise ViolaciónDelContratoDeSalida(
                f"{self.artwork_id}: una explicación **sin afirmaciones y sin "
                f"abstención** es el hueco silencioso que el descarte contado de afirmaciones prohíbe — es "
                f"indistinguible de una obra que nadie intentó explicar. Es el mismo "
                f"criterio que la evidencia ordenada por similitud aplica del lado de la evidencia."
            )

    @property
    def refs(self) -> tuple[str, ...]:
        vistas: list[str] = []
        for a in self.afirmaciones:
            for r in a.refs:
                if r not in vistas:
                    vistas.append(r)
        return tuple(vistas)

    @property
    def n_datos(self) -> int:
        return sum(1 for a in self.afirmaciones if a.es_dato)

    def a_json(self) -> dict:
        d: dict[str, Any] = {
            "artwork_id": self.artwork_id,
            "afirmaciones": [a.a_json() for a in self.afirmaciones],
        }
        if self.abstencion:
            d["abstencion"] = self.abstencion
        return d

    @classmethod
    def desde_json(cls, d: Mapping[str, Any]) -> "Explicación":
        if "artwork_id" not in d:
            raise ViolaciónDelContratoDeSalida(
                "una explicación sin `artwork_id` no se puede atribuir a ninguna obra del "
                "top-5, y la evidencia registrada en la corrida/la forma declarada del artefacto necesitan esa correspondencia."
            )
        return cls(
            artwork_id=str(d["artwork_id"]).strip(),
            afirmaciones=tuple(
                Afirmación.desde_json(a) for a in (d.get("afirmaciones") or ())
            ),
            abstencion=(str(d["abstencion"]).strip() if d.get("abstencion") else None),
        )

    def sin_refs_colgadas(
        self, ids_de_la_evidencia: Sequence[str], *, texto_de_abstención: str
    ) -> tuple["Explicación", tuple[dict, ...]]:
        válidos = set(ids_de_la_evidencia)
        quedan: list[Afirmación] = []
        descartes: list[dict] = []
        for a in self.afirmaciones:
            colgadas = [r for r in a.refs if r not in válidos]
            if colgadas:
                descartes.append(
                    {
                        "artwork_id": self.artwork_id,
                        "refs_colgadas": colgadas,
                        "tipo": a.tipo,
                        "n_refs_de_la_afirmacion": len(a.refs),
                        "causa": (
                            "la `ref` no resuelve contra la evidencia de **esta** "
                            "consulta. Se cae la afirmación entera (el descarte contado de afirmaciones) y no solo la "
                            "`ref`, porque dejar un `dato` sin respaldo violaría la regla de que un dato lleva referencia."
                        ),
                    }
                )
                continue
            quedan.append(a)

        if quedan or self.abstencion:
            return (
                Explicación(
                    artwork_id=self.artwork_id,
                    afirmaciones=tuple(quedan),
                    abstencion=self.abstencion,
                ),
                tuple(descartes),
            )
        return (
            Explicación(
                artwork_id=self.artwork_id, afirmaciones=(), abstencion=texto_de_abstención
            ),
            tuple(descartes),
        )
