from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture
from tp3_rag.nucleo.errores import CulturaPedidaFueraDelCorpus

GRUPOS_PROPUESTOS: tuple[dict, ...] = (
    {
        "clave": "india",
        "nombre": "la India",
        "etiquetas": [
            "Mughal India", "Northern India", "Northwestern India", "India",
            "Eastern India", "Western India", "Central India", "Southern India",
            "Sultanate India", "British India", "Northwest India",
        ],
        "dudosas": {
            "Jodhpur": "ciudad de Rajastán — el caso en que AIC da una ciudad",
            "made for Abd al-Rahim the Khan Khanan": (
                "**no nombra un lugar**: es el comitente mogol. Un criterio de "
                "normalización no tiene qué hacer con este valor, y es el ejemplo más "
                "claro de que normalizar no es solo cortar por coma"
            ),
            "Sri Lanka": "país distinto; entra solo si la cultura pedida es 'el sur de Asia'",
            "Nepal": "ídem — país distinto, 21 obras propias",
        },
        "por_que": (
            "Es el caso que existe para resolver. Con `India` sola el `n` que la cobertura cultural del resultado "
            "publica deja afuera a las otras diez etiquetas, y **once de once nombran la "
            "misma cultura**. Fijate además que `Northwestern India` y `Northwest India` "
            "son la misma región escrita de dos formas: el vocabulario no se parte solo "
            "por región, se parte por **ortografía**"
        ),
    },
    {
        "clave": "estados_unidos",
        "nombre": "los Estados Unidos",
        "etiquetas": ["United States", "America"],
        "dudosas": {
            "New York": "AIC: lugar de producción. `New York` y `New York City` son el mismo lugar",
            "New York City": "ídem",
            "Philadelphia": "ciudad",
            "Boston": "ciudad",
            "Baltimore": "ciudad",
            "Bennington": "ciudad (Vermont)",
            "Saint Louis": "ciudad",
            "Roxbury": "barrio de Boston",
            "Gloucester": "ciudad — **y hay un Gloucester en Inglaterra**",
            "Greenwich": "**homónima**: Connecticut y Londres",
            "Ipswich": "**homónima**: Massachusetts e Inglaterra",
            "Lancaster": "**homónima**: Pensilvania e Inglaterra (4 obras, todas de AIC)",
            "Newport": "**homónima**: Rhode Island y Gales",
            "Connecticut": "estado",
            "Massachusetts": "estado",
            "New Hampshire": "estado",
            "Pennsylvania": "estado",
            "Florida": "estado",
            "New England": "región",
            "Long Island": "isla",
            "Nantucket Island": "isla",
            "Niagara Falls": "accidente geográfico, y está en los dos países",
            "Prouts Neck": "cabo de Maine",
            "York Harbor": "pueblo de Maine",
        },
        "por_que": (
            "`United States` (AIC) y `America` (CMA) son **el mismo país con dos nombres**, "
            "y es el ejemplo que el criterio de normalización de culture ya trae. Lo que agrega el corpus es la cola: "
            "**dos docenas de etiquetas más** que son ciudades, estados, islas y cabos "
            "estadounidenses, cuatro de ellas **homónimas de lugares ingleses**. Agrupar "
            "'Estados Unidos' bien no es una decisión de dos líneas: es la cola la que "
            "manda, y la cola es ambigua"
        ),
    },
    {
        "clave": "paises_bajos",
        "nombre": "los Países Bajos",
        "etiquetas": [
            "Netherlands", "Holland", "Netherlandish", "Northern Netherlands",
            "South Netherlands",
        ],
        "dudosas": {
            "Delft": "ciudad",
            "Dordrecht": "ciudad",
            "Flanders": (
                "🔑 **no es sinónimo**: en historia del arte flamenco y neerlandés son dos "
                "escuelas. Son **44 obras**, así que la decisión mueve el número"
            ),
            "Belgium": "país distinto, y es donde cae Flandes hoy",
            "Northeastern France or the Burgundian Netherlands": (
                "**un valor que nombra dos lugares con un 'or'**. Ningún criterio de corte "
                "lo resuelve: o se elige uno, o se descarta, y las dos son decisiones"
            ),
        },
        "por_que": (
            "`Netherlands` y `Holland` conviven —un ejemplo del problema— y el corpus agrega "
            "`Netherlandish`, que es **la forma adjetival del mismo nombre**: el criterio "
            "`primer_componente_por_coma` no une `China`/`Chinese` ni "
            "`Netherlands`/`Netherlandish`, y es un patrón, no un caso"
        ),
    },
    {
        "clave": "china",
        "nombre": "China",
        "etiquetas": ["China", "Chinese", "South China"],
        "dudosas": {},
        "por_que": (
            "El par **nominal/adjetival** (`China` / `Chinese`) otra vez. Es chico en `n` "
            "pero importa como patrón: un criterio que no normaliza la forma del gentilicio "
            "va a dejar sinónimos en **toda** cultura que algún museo escriba como adjetivo"
        ),
    },
    {
        "clave": "japon",
        "nombre": "Japón",
        "etiquetas": ["Japan"],
        "dudosas": {},
        "por_que": (
            "✅ **No necesita agrupar**: una etiqueta sola cubre toda la cultura japonesa "
            "del corpus. Es la cultura con más masa donde el vocabulario **no** se parte, "
            "así que es la más segura para escribir un pedido conceptual/cultural — y, "
            "literal, el ejemplo del email aprobado (*'el mismo tema pero en la cultura "
            "japonesa'*)"
        ),
    },
    {
        "clave": "corea",
        "nombre": "Corea",
        "etiquetas": ["Korea"],
        "dudosas": {},
        "por_que": "✅ **Tampoco**. Misma propiedad que `japon`, con menos masa",
    },
    {
        "clave": "tibet",
        "nombre": "el Tíbet",
        "etiquetas": ["Tibet", "Central Tibet", "Western Tibet"],
        "dudosas": {},
        "por_que": (
            "El patrón de la India en chico: el eje cardinal parte la cultura en tres sin "
            "que ninguna de las tres sea una cultura distinta"
        ),
    },
    {
        "clave": "italia",
        "nombre": "Italia",
        "etiquetas": ["Italy", "Northern Italy"],
        "dudosas": {
            "Venice": "ciudad — y `Republic of Venice` es **la misma ciudad como estado**",
            "Republic of Venice": "ídem",
            "Florence": "ciudad",
            "Rome": "ciudad",
            "Frascati": "ciudad",
            "Umbria": "región",
            "possibly Italy": "**valor con hedge**: el museo no está seguro y lo escribió adentro del campo",
            "Spain or Northern Italy": "**dos países con un 'or'**",
            "Antonine-era Roman Empire": "imperio, no país — y el período está en el valor",
        },
        "por_que": (
            "Acá aparecen los dos valores con **hedge** (`possibly Italy`, `Spain or "
            "Northern Italy`): no son sinónimos, son **incertidumbre del museo escrita "
            "dentro del dato**. el criterio de normalización tiene que decidir qué hace con ellos, y ninguna de las "
            "tres salidas (elegir, descartar, contar en los dos) es gratis"
        ),
    },
    {
        "clave": "francia",
        "nombre": "Francia",
        "etiquetas": ["France"],
        "dudosas": {
            "Paris": "ciudad (8 obras)",
            "Giverny": "pueblo — donde Monet pintó la serie de las parvas",
            "Brittany": "región",
            "Saint-Rémy-de-Provence": "pueblo — el ejemplo literal del problema",
            "Northeastern France or the Burgundian Netherlands": "el valor con 'or' de arriba",
        },
        "por_que": (
            "La cultura con más masa del corpus y la etiqueta sola alcanza **casi** del "
            "todo: lo que queda afuera son lugares franceses que AIC escribió como ciudad "
            "o región, más el valor con 'or'. `Saint-Rémy-de-Provence` es el ejemplo "
            "con el que el criterio de normalización de culture está escrito"
        ),
    },
    {
        "clave": "reino_unido",
        "nombre": "Inglaterra / el Reino Unido",
        "etiquetas": ["England", "United Kingdom"],
        "dudosas": {
            "Scotland": "🔑 **país del Reino Unido, no de Inglaterra**: el nombre del grupo cambia la respuesta",
            "London": "ciudad",
            "Greenwich": "la homónima de arriba",
            "Ipswich": "ídem",
            "Lancaster": "ídem",
            "Newport": "ídem",
        },
        "por_que": (
            "Es el grupo donde **el nombre de la `cultura_pedida` decide las etiquetas** y "
            "no al revés: 'inglesa' y 'británica' no son el mismo conjunto, y el pedido que "
            "una persona escribe elige por las dos"
        ),
    },
    {
        "clave": "alemania",
        "nombre": "Alemania",
        "etiquetas": ["Germany", "Southern Germany"],
        "dudosas": {"Munich": "ciudad", "Austria": "país distinto, 19 obras propias"},
        "por_que": "El eje cardinal otra vez, y chico: `Southern Germany` son 2 obras",
    },
)

@dataclass(frozen=True)
class UniónDeEtiquetas:
    etiquetas: tuple[str, ...]
    n: int
    n_por_etiqueta: dict[str, int]
    poblacion: str
    n_poblacion: int
    criterio: str

    @property
    def suma_de_etiquetas(self) -> int:
        return sum(self.n_por_etiqueta.values())

    @property
    def solapamiento(self) -> int:
        return self.suma_de_etiquetas - self.n

class VocabularioDeCulture:
    def __init__(
        self,
        registros_del_corpus: Iterable[RegistroCanónico],
        *,
        criterio: str,
        poblacion: str,
    ) -> None:
        self.criterio = criterio
        self.poblacion = poblacion
        self._normalizador = NormalizadorDeCulture(criterio)
        self._por_etiqueta: dict[str, set[str]] = {}
        self._sin_culture: list[str] = []
        self.n_poblacion = 0
        for registro in registros_del_corpus:
            self.n_poblacion += 1
            etiquetas = self._normalizador.normalizar(registro.culture_raw)
            if not etiquetas:
                self._sin_culture.append(registro.artwork_id)
            for etiqueta in etiquetas:
                self._por_etiqueta.setdefault(etiqueta, set()).add(registro.artwork_id)

    @classmethod
    def del_corpus_del_split(
        cls, catálogo, manifiesto, *, criterio: str
    ) -> "VocabularioDeCulture":
        ids = set(manifiesto.ids_corpus)
        return cls(
            (r for r in catálogo.registros if r.artwork_id in ids),
            criterio=criterio,
            poblacion=f"corpus indexado del split {manifiesto.split_id}",
        )

    @property
    def etiquetas(self) -> dict[str, int]:
        return dict(
            sorted(
                ((e, len(obras)) for e, obras in self._por_etiqueta.items()),
                key=lambda par: (-par[1], par[0]),
            )
        )

    @property
    def n_etiquetas(self) -> int:
        return len(self._por_etiqueta)

    @property
    def n_sin_culture(self) -> int:
        return len(self._sin_culture)

    def normalizar(self, culture_raw) -> list[str]:
        return self._normalizador.normalizar(culture_raw)

    def existe(self, etiqueta: str) -> bool:
        return etiqueta in self._por_etiqueta

    def n_de(self, etiqueta: str) -> int:
        return len(self._por_etiqueta.get(etiqueta, ()))

    def validar_etiquetas(self, etiquetas: Sequence[str]) -> None:
        for etiqueta in etiquetas:
            if not self.existe(etiqueta):
                raise CulturaPedidaFueraDelCorpus(
                    etiqueta,
                    criterio=self.criterio,
                    poblacion=self.poblacion,
                    sugerencias=self.sugerencias(etiqueta),
                )

    def union(self, etiquetas: Sequence[str]) -> UniónDeEtiquetas:
        self.validar_etiquetas(etiquetas)
        obras: set[str] = set()
        for etiqueta in etiquetas:
            obras |= self._por_etiqueta[etiqueta]
        return UniónDeEtiquetas(
            etiquetas=tuple(etiquetas),
            n=len(obras),
            n_por_etiqueta={e: self.n_de(e) for e in etiquetas},
            poblacion=self.poblacion,
            n_poblacion=self.n_poblacion,
            criterio=self.criterio,
        )

    def grupos_propuestos(self) -> list[dict]:
        salida = []
        for grupo in GRUPOS_PROPUESTOS:
            presentes = [e for e in grupo["etiquetas"] if self.existe(e)]
            ausentes = [e for e in grupo["etiquetas"] if not self.existe(e)]
            unión = self.union(presentes) if presentes else None
            salida.append(
                {
                    "clave": grupo["clave"],
                    "nombre": grupo["nombre"],
                    "etiquetas": presentes,
                    "ausentes": ausentes,
                    "n": unión.n if unión else 0,
                    "n_por_etiqueta": unión.n_por_etiqueta if unión else {},
                    "solapamiento": unión.solapamiento if unión else 0,
                    "dudosas": {
                        e: razón
                        for e, razón in grupo["dudosas"].items()
                        if self.existe(e)
                    },
                    "n_dudosas": sum(
                        self.n_de(e) for e in grupo["dudosas"] if self.existe(e)
                    ),
                    "por_que": grupo["por_que"],
                }
            )
        return salida

    def sugerencias(self, etiqueta: str, tope: int = 5) -> tuple[str, ...]:
        import difflib

        objetivo = str(etiqueta).casefold()
        palabras = {p for p in objetivo.replace(",", " ").split() if p}
        puntuadas: list[tuple[float, int, str]] = []
        for candidata in self._por_etiqueta:
            plegada = candidata.casefold()
            comunes = len(palabras & set(plegada.split()))
            contiene = objetivo in plegada or plegada in objetivo
            parecido = difflib.SequenceMatcher(None, objetivo, plegada).ratio()
            if comunes or contiene or parecido >= 0.6:
                puntaje = comunes * 2 + (1.5 if contiene else 0) + parecido
                puntuadas.append((-puntaje, -self.n_de(candidata), candidata))
        return tuple(c for _, _, c in sorted(puntuadas)[:tope])

    def distribucion(self, tope: int | None = None) -> dict:
        etiquetas = self.etiquetas
        if tope is not None:
            etiquetas = dict(list(etiquetas.items())[:tope])
        return {
            "criterio": self.criterio,
            "poblacion": self.poblacion,
            "n_poblacion": self.n_poblacion,
            "n_etiquetas_distintas": self.n_etiquetas,
            "n_obras_sin_culture": self.n_sin_culture,
            "etiquetas": etiquetas,
            "nota": (
                "La población es **el corpus indexado**, que es la que la cobertura cultural del resultado usa — no el "
                "universo normalizado. Las cifras del criterio de normalización de culture están medidas sobre el "
                "universo y son 2–3× más grandes; las dos son correctas y solo una es la "
                "que la decisión necesita (los hallazgos, modo de falla 3)."
            ),
        }

def conteo(valores: Iterable[str]) -> Mapping[str, int]:
    return dict(sorted(Counter(valores).items()))
