from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from tp3_rag.nucleo.errores import ErrorDelProyecto

ABSTENCIÓN_POR_REFS_COLGADAS = (
    "La evidencia recuperada no alcanza para explicar esta obra: las afirmaciones que el "
    "modelo produjo citaban fragmentos que no se recuperaron en esta consulta y se "
    "descartaron."
)

class ObligaciónIncumplida(ErrorDelProyecto):
    def __init__(self, obligacion: str, detalle: str) -> None:
        super().__init__(f"las obligaciones de la cita · {obligacion}: {detalle}")
        self.obligacion = obligacion
        self.detalle = detalle

@dataclass(frozen=True)
class FormatoDeSalida:
    formato: str = "markdown"
    atribucion: str = "al_pie_de_la_referencia"
    campos_de_la_ficha: tuple[str, ...] = ("artist", "date_display", "medium", "style", "museum")
    nombre_del_museo: Mapping[str, str] = field(default_factory=dict)
    nombre_de_la_licencia: Mapping[str, str] = field(default_factory=dict)
    exigen_atribucion: Mapping[str, bool] = field(default_factory=dict)

    FORMATOS: ClassVar[tuple[str, ...]] = ("markdown", "texto")
    UBICACIONES: ClassVar[tuple[str, ...]] = ("al_pie_de_la_referencia", "en_bloque_final")

    def __post_init__(self) -> None:
        if self.formato not in self.FORMATOS:
            raise ErrorDelProyecto(
                f"`salida.formato = {self.formato!r}` no es uno de {list(self.FORMATOS)}"
            )
        if self.atribucion not in self.UBICACIONES:
            raise ErrorDelProyecto(
                f"`salida.atribucion = {self.atribucion!r}` no es una de "
                f"{list(self.UBICACIONES)}"
            )

    @classmethod
    def desde_configuración(cls, cfg) -> "FormatoDeSalida":
        d = cfg["salida"]
        return cls(
            formato=d["formato"],
            atribucion=d["atribucion"],
            campos_de_la_ficha=tuple(d["campos_de_la_ficha"]),
            nombre_del_museo=dict(d["nombre_del_museo"]),
            nombre_de_la_licencia=dict(d["nombre_de_la_licencia"]),
            exigen_atribucion={k: bool(v) for k, v in dict(d["exigen_atribucion"]).items()},
        )

    def exige_atribucion(self, licencia: str) -> bool:
        if licencia not in self.exigen_atribucion:
            raise ObligaciónIncumplida(
                "licencia no declarada",
                f"la licencia {licencia!r} no tiene entrada en "
                f"`salida.exigen_atribucion`, así que no se sabe si exige atribución. **No "
                f"hay default**: un `False` por omisión publicaría una cita sin atribuir, "
                f"que los modos de falla del proyecto lista como modo de falla **no aceptable**. Es la "
                f"regla de la licencia declarada y no asumida aplicada al último eslabón de la licencia por fragmento de punta a punta.",
            )
        return self.exigen_atribucion[licencia]

    def nombre_de_licencia(self, licencia: str) -> str:
        if licencia not in self.nombre_de_la_licencia:
            raise ObligaciónIncumplida(
                "licencia no declarada",
                f"la licencia {licencia!r} no tiene nombre público en "
                f"`salida.nombre_de_la_licencia`. Una atribución que nombra la licencia con "
                f"su código interno no es una atribución legible.",
            )
        return self.nombre_de_la_licencia[licencia]

    def nombre_de_museo(self, museo: str) -> str:
        return self.nombre_del_museo.get(museo, museo)

    def a_json(self) -> dict:
        return {
            "formato": self.formato,
            "atribucion": self.atribucion,
            "campos_de_la_ficha": list(self.campos_de_la_ficha),
            "exigen_atribucion": dict(self.exigen_atribucion),
            "nota": (
                "El formato concreto **sigue abierto**: esto es la **cosmética declarada**, no el "
                "contrato. El contrato —`[n]` que asigna el renderizador, toda afirmación "
                "con dato citando un `fragment_id` recuperado en esta consulta, 100 % de "
                "los CC-BY con atribución, español + cita textual en inglés— vive en el "
                "código y es el conteo de referencias colgadas, el criterio de atribución, el criterio de idioma y las obligaciones de la cita."
            ),
        }

@dataclass(frozen=True)
class Referencia:
    n: int
    fragment_id: str
    texto_citado: str
    title: str | None
    artist: str | None
    source_url: str
    license: str
    museum: str
    atribucion: str | None
    campos: tuple[str, ...] = ()
    licencias: tuple[str, ...] = ()

    @property
    def licencias_citadas(self) -> tuple[str, ...]:
        return self.licencias or (self.license,)

    @property
    def mezcla_licencias(self) -> bool:
        return len(set(self.licencias_citadas)) > 1

    def a_json(self) -> dict:
        return {
            "n": self.n,
            "fragment_id": self.fragment_id,
            "texto_citado": self.texto_citado,
            "title": self.title,
            "artist": self.artist,
            "source_url": self.source_url,
            "license": self.license,
            "museum": self.museum,
            "atribucion": self.atribucion,
            "campos": list(self.campos),
            "licencias": list(self.licencias_citadas),
            "mezcla_licencias": self.mezcla_licencias,
        }

@dataclass(frozen=True)
class ObraCitada:
    artwork_id: str
    rango: int
    score: float | None
    ficha: Mapping[str, Any]
    source_url: str
    lineas: tuple[str, ...]
    abstencion: str | None
    referencias: tuple[Referencia, ...]

    def a_json(self) -> dict:
        return {
            "artwork_id": self.artwork_id,
            "rango": self.rango,
            "score": self.score,
            "ficha": dict(self.ficha),
            "source_url": self.source_url,
            "lineas": list(self.lineas),
            "abstencion": self.abstencion,
            "referencias": [r.a_json() for r in self.referencias],
        }

@dataclass(frozen=True)
class SalidaCitada:
    consulta_id: str
    obras: tuple[ObraCitada, ...]
    formato: FormatoDeSalida
    idioma_de_la_explicacion: str = "es"
    idioma_de_las_citas: str = "en"

    def render(self) -> str:
        marca = "**" if self.formato.formato == "markdown" else ""
        partes: list[str] = []
        for obra in self.obras:
            cabecera = f"{obra.rango}. {marca}{obra.ficha.get('title') or obra.artwork_id}{marca}"
            partes.append(cabecera)
            ficha = " · ".join(
                str(obra.ficha[c]) for c in self.formato.campos_de_la_ficha if obra.ficha.get(c)
            )
            if ficha:
                partes.append(f"   {ficha}")
            partes.append(f"   Ficha oficial: {obra.source_url}")
            partes.append("")
            for línea in obra.lineas:
                partes.append(f"   {línea}")
            if obra.abstencion:
                partes.append(f"   {obra.abstencion}")
            if obra.referencias:
                partes.append("")
                partes.append(f"   {marca}Fuentes{marca}")
                for ref in obra.referencias:
                    partes.append(f"   [{ref.n}] “{ref.texto_citado}”")
                    licencias = " + ".join(
                        self.formato.nombre_de_licencia(x) for x in ref.licencias_citadas
                    )
                    pie = f"       {ref.source_url} · {licencias}"
                    partes.append(pie)
                    if ref.atribucion and self.formato.atribucion == "al_pie_de_la_referencia":
                        partes.append(f"       {ref.atribucion}")
            if (
                self.formato.atribucion == "en_bloque_final"
                and any(r.atribucion for r in obra.referencias)
            ):
                partes.append("")
                partes.append(f"   {marca}Atribución{marca}")
                for ref in obra.referencias:
                    if ref.atribucion:
                        partes.append(f"       [{ref.n}] {ref.atribucion}")
            partes.append("")
        return "\n".join(partes).rstrip() + "\n"

    def verificar_obligaciones(self, fragmentos_por_id: Mapping[str, Any]) -> dict:
        texto = self.render()
        n_exigen = 0
        n_mezclan = 0
        n_mezclan_y_exigen = 0
        for obra in self.obras:
            for ref in obra.referencias:
                fragmento = fragmentos_por_id.get(ref.fragment_id)
                if fragmento is None:
                    raise ObligaciónIncumplida(
                        "cita sin fragmento",
                        f"la referencia [{ref.n}] de {obra.artwork_id} cita "
                        f"{ref.fragment_id!r}, que **no está en la evidencia de esta "
                        f"consulta**. Una `ref` colgada del generador es el descarte contado de afirmaciones y se "
                        f"descarta antes de llegar acá; que llegue significa que el "
                        f"renderizador la inventó, y eso es la obligación de "
                        f"verificabilidad de la cita.",
                    )
                if ref.texto_citado != fragmento.texto:
                    raise ObligaciónIncumplida(
                        "cita no textual",
                        f"la referencia [{ref.n}] de {obra.artwork_id} no es **byte a "
                        f"byte** el texto de {ref.fragment_id}. La cita es *transporte de "
                        f"texto, no generación* (el contrato de salida): si se reescribe, el criterio de idioma deja de "
                        f"ser verificable y *groundedness* mide nuestra edición en vez de "
                        f"la fuente.",
                    )
                if ref.texto_citado not in texto:
                    raise ObligaciónIncumplida(
                        "cita no renderizada",
                        f"la referencia [{ref.n}] de {obra.artwork_id} tiene "
                        f"`texto_citado` y **no aparece en la superficie**. Lo que el criterio de atribución y "
                        f"el criterio de idioma comprometen es sobre lo que se muestra.",
                    )
                exigen = [x for x in ref.licencias_citadas if self.formato.exige_atribucion(x)]
                if ref.mezcla_licencias:
                    n_mezclan += 1
                if not exigen:
                    continue
                n_exigen += 1
                if ref.mezcla_licencias:
                    n_mezclan_y_exigen += 1
                if not ref.atribucion:
                    raise ObligaciónIncumplida(
                        "atribución faltante",
                        f"el fragmento {ref.fragment_id} abarca "
                        f"{list(ref.licencias_citadas)} —{exigen} exige(n) atribución— y "
                        f"**no lleva atribución**. el criterio de atribución es un umbral **duro de 100 %** y "
                        f"la política de fallas explica por qué esta pieza falla en vez de degradar: *una "
                        f"respuesta sin explicación es un resultado pobre; una cita sin "
                        f"atribución es una infracción* (los modos de falla del proyecto).",
                    )
                if ref.atribucion not in texto:
                    raise ObligaciónIncumplida(
                        "atribución no renderizada",
                        f"el fragmento {ref.fragment_id} ({list(ref.licencias_citadas)}) "
                        f"tiene línea de atribución y **no aparece en la superficie**. el criterio de atribución "
                        f"compromete *'se renderizan con atribución en toda superficie "
                        f"donde se muestren'*: un campo lleno que el render no imprime "
                        f"cumple el chequeo y viola el criterio.",
                    )
        return {
            "cumple_sa3": True,
            "n_referencias": sum(len(o.referencias) for o in self.obras),
            "n_que_exigen_atribucion": n_exigen,
            "n_con_atribucion_renderizada": n_exigen,
            "pct_con_atribucion": 100.0 if n_exigen else None,
            "n_que_mezclan_licencias": n_mezclan,
            "n_que_mezclan_licencias_y_exigen_atribucion": n_mezclan_y_exigen,
        }

    def referencias(self) -> tuple[Referencia, ...]:
        return tuple(r for o in self.obras for r in o.referencias)

    def a_json(self) -> dict:
        return {
            "consulta_id": self.consulta_id,
            "idioma_de_la_explicacion": self.idioma_de_la_explicacion,
            "idioma_de_las_citas": self.idioma_de_las_citas,
            "nota_de_idioma": (
                "el contrato de la salida citada / el criterio de idioma: la explicación en **español**, el fragmento "
                "citado **textual en inglés, sin traducir**. La cita viaja desde "
                "`knowledge` hasta acá **sin pasar por el modelo** (el contrato de salida), y las obligaciones de la cita lo "
                "verifica byte a byte: si el formateo viviera dentro del prompt, "
                "*'textual'* dependería de que el modelo no lo reescriba."
            ),
            "formato": self.formato.a_json(),
            "obras": [o.a_json() for o in self.obras],
            "render": self.render(),
        }
