from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tp3_rag.nucleo.errores import ErrorDeConfiguración

if sys.version_info >= (3, 11):  # pragma: no cover - el lock fija 3.10
    import tomllib
else:
    import tomli as tomllib

RAÍZ_DEL_REPO = Path(__file__).resolve().parents[3]
DEFAULT_TOML = RAÍZ_DEL_REPO / "config" / "default.toml"

_AUSENTE = object()

def _congelar(valor: Any) -> Any:
    if isinstance(valor, dict):
        return _MapaInmutable({k: _congelar(v) for k, v in valor.items()})
    if isinstance(valor, list):
        return tuple(_congelar(v) for v in valor)
    return valor

class _MapaInmutable(Mapping):
    __slots__ = ("_d",)

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getitem__(self, k):
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f"{self._d!r}"

    def a_dict(self) -> dict:
        return {k: _descongelar(v) for k, v in self._d.items()}

def _descongelar(valor: Any) -> Any:
    if isinstance(valor, _MapaInmutable):
        return valor.a_dict()
    if isinstance(valor, (tuple, list)):
        return [_descongelar(v) for v in valor]
    return valor

TABLAS_ABIERTAS = (
    "ingesta.tabla_de_licencias",
    "ingesta.tabla_de_licencias.*",
    "ingesta.campos_de_texto",
    "ingesta.campos_de_texto.*",
    "ingesta.campos_canonicos",
    "ingesta.campos_canonicos.*",
    "ingesta.media_license",
    "ingesta.urls",
    "ingesta.urls.*",
    "museos",
    "museos.*",
    "http.headers_por_dominio",
    "http.headers_por_dominio.*",
    "http.rate_limit_por_minuto",
    "proveedor.precio_usd_por_millon_de_tokens",
    "consulta.plan_de_falla",
    "salida.nombre_de_la_licencia",
    "salida.exigen_atribucion",
    "salida.nombre_del_museo",
    "medicion.terminos_de_cultura",
)

NO_DETERMINAN_EL_CONFIG_ID: tuple[str, ...] = (
    "rutas",
    "embebido.threads",
    "embebido.batch",
    "embebido.batch_maximo",
    "embebido_texto.batch",
    "indice.batch_de_upsert",
    "proveedor.base_url",
    "proveedor.precio_usd_por_millon_de_tokens",
    "http",
    "medicion",
    "trazas.proyecto",
    "trazas.timeout_segundos",
    "consulta.alfa_congelado",
)

CLAVES_DE_TODA_CORRIDA: tuple[str, ...] = (
    "ingesta",
    "normalizacion",
    "museos",
    "split",
    "fragmentacion",
    "embebido",
    "embebido_texto",
    "indice",
    "consulta.configuracion",
    "consulta.etapas",
    "consulta.top_k",
    "consulta.plan_de_falla",
    "retencion",
    "trazas.proveedor",
)

def _minimizar(rutas) -> tuple[str, ...]:
    ordenadas = sorted(rutas)
    return tuple(
        r
        for r in ordenadas
        if not any(r != d and r.startswith(f"{d}.") for d in ordenadas)
    )

def _cubre(alcance, ruta: str) -> bool:
    return any(ruta == d or ruta.startswith(f"{d}.") for d in alcance)

def fuera_del_alcance(leídas, *, alcance) -> tuple[str, ...]:
    permitido = tuple(alcance) + tuple(CLAVES_DE_TODA_CORRIDA) + NO_DETERMINAN_EL_CONFIG_ID
    return tuple(sorted({r for r in leídas if not _cubre(permitido, r)}))

def _recortar_a(árbol: Mapping[str, Any], rutas) -> dict:
    entero = json.loads(json.dumps(dict(árbol)))
    salida: dict = {}
    for ruta in rutas:
        partes = ruta.split(".")
        nodo: Any = entero
        for parte in partes:
            if not isinstance(nodo, dict) or parte not in nodo:
                nodo = _AUSENTE
                break
            nodo = nodo[parte]
        if nodo is _AUSENTE:
            continue
        destino = salida
        for parte in partes[:-1]:
            destino = destino.setdefault(parte, {})
            if not isinstance(destino, dict):  # pragma: no cover - rutas disjuntas
                break
        if isinstance(destino, dict):
            if partes[-1] in destino and isinstance(destino[partes[-1]], dict) and isinstance(nodo, dict):
                destino[partes[-1]].update(nodo)
            else:
                destino[partes[-1]] = nodo
    return salida

def sin_las_no_determinantes(árbol: Mapping[str, Any]) -> dict:
    recortado = json.loads(json.dumps(dict(árbol)))
    for ruta in NO_DETERMINAN_EL_CONFIG_ID:
        partes = ruta.split(".")
        nodo = recortado
        for parte in partes[:-1]:
            nodo = nodo.get(parte) if isinstance(nodo, dict) else None
            if nodo is None:
                break
        if isinstance(nodo, dict):
            nodo.pop(partes[-1], None)
    return recortado

def _es_tabla_abierta(ruta: str) -> bool:
    if ruta in TABLAS_ABIERTAS:
        return True
    partes = ruta.split(".")
    for i in range(len(partes)):
        patrón = ".".join(partes[:i] + ["*"] + partes[i + 1 :])
        if patrón in TABLAS_ABIERTAS:
            return True
    return False

def _fusionar(base: dict, encima: dict, prefijo: str = "") -> dict:
    salida = dict(base)
    abierta = prefijo and _es_tabla_abierta(prefijo.rstrip("."))
    for clave, valor in encima.items():
        ruta = f"{prefijo}{clave}"
        if clave not in base and not abierta:
            raise ErrorDeConfiguración(
                f"clave desconocida en la configuración: {ruta!r}. "
                f"`config/default.toml` es el esquema; declarala ahí primero."
            )
        if isinstance(base.get(clave), dict) and isinstance(valor, dict):
            salida[clave] = _fusionar(base[clave], valor, prefijo=f"{ruta}.")
        else:
            salida[clave] = valor
    return salida

def _interpretar(texto: str) -> Any:
    bajo = texto.strip()
    if bajo in ("true", "false"):
        return bajo == "true"
    if bajo in ("null", "none", ""):
        return None
    try:
        return int(bajo)
    except ValueError:
        pass
    try:
        return float(bajo)
    except ValueError:
        pass
    if bajo.startswith("[") and bajo.endswith("]"):
        return [p.strip().strip("\"'") for p in bajo[1:-1].split(",") if p.strip()]
    return bajo

@dataclass(frozen=True)
class Configuración:
    _árbol: _MapaInmutable
    capas: tuple[str, ...]
    _lecturas: set = field(default_factory=set, compare=False, repr=False)

    def __getitem__(self, ruta: str) -> Any:
        nodo: Any = self._árbol
        for parte in ruta.split("."):
            try:
                nodo = nodo[parte]
            except (KeyError, TypeError):
                raise ErrorDeConfiguración(
                    f"la configuración no tiene la clave {ruta!r}"
                ) from None
        self._lecturas.add(ruta)
        return nodo

    def lecturas(self) -> frozenset[str]:
        return frozenset(self._lecturas)

    def olvidar_lecturas(self) -> None:
        self._lecturas.clear()

    def get(self, ruta: str, default: Any = None) -> Any:
        try:
            return self[ruta]
        except ErrorDeConfiguración:
            return default

    def a_dict(self) -> dict:
        return self._árbol.a_dict()

    def alcance_del_config_id(self) -> tuple[str, ...]:
        from tp3_rag.modulo_b.armador import alcance_de_la_cadena

        cadena = tuple(self._árbol["consulta"]["etapas"])
        return _minimizar(set(CLAVES_DE_TODA_CORRIDA) | set(alcance_de_la_cadena(cadena)))

    def a_dict_determinante(self) -> dict:
        return sin_las_no_determinantes(
            _recortar_a(self.a_dict(), self.alcance_del_config_id())
        )

    @property
    def config_id(self) -> str:
        from tp3_rag.nucleo.identidad import hash_corto

        return hash_corto(self.a_dict_determinante())

    def no_determinantes(self) -> dict:
        presentes: dict[str, Any] = {}
        for ruta in NO_DETERMINAN_EL_CONFIG_ID:
            valor = self.get(ruta, _AUSENTE)
            if valor is not _AUSENTE:
                presentes[ruta] = _descongelar(valor)
        return presentes

    @classmethod
    def cargar(
        cls,
        archivo: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
        default: Path | None = None,
    ) -> "Configuración":
        ruta_default = default or DEFAULT_TOML
        if not ruta_default.exists():
            raise ErrorDeConfiguración(f"no existe {ruta_default}")
        with ruta_default.open("rb") as fh:
            árbol = tomllib.load(fh)
        capas = [str(ruta_default.relative_to(RAÍZ_DEL_REPO))]

        if archivo is not None:
            if not archivo.exists():
                raise ErrorDeConfiguración(f"no existe el archivo de configuración {archivo}")
            with archivo.open("rb") as fh:
                árbol = _fusionar(árbol, tomllib.load(fh))
            capas.append(str(archivo))

        for ruta, valor in (overrides or {}).items():
            partes = ruta.split(".")
            nodo = árbol
            for parte in partes[:-1]:
                if parte not in nodo or not isinstance(nodo[parte], dict):
                    raise ErrorDeConfiguración(
                        f"clave desconocida en la configuración: {ruta!r} "
                        f"(`config/default.toml` es el esquema)"
                    )
                nodo = nodo[parte]
            padre = ".".join(partes[:-1])
            if partes[-1] not in nodo and not _es_tabla_abierta(padre):
                raise ErrorDeConfiguración(
                    f"clave desconocida en la configuración: {ruta!r} "
                    f"(`config/default.toml` es el esquema). Si {padre!r} fuera un mapa "
                    f"cuyas claves son nombres —un museo, una `section`, una etapa del plan "
                    f"de falla—, agregar una entrada sería usar el punto de variación, "
                    f"y entonces tendría que estar en `TABLAS_ABIERTAS`."
                )
            nodo[partes[-1]] = _interpretar(valor) if isinstance(valor, str) else valor
            capas.append(f"--set {ruta}")

        cfg = cls(_congelar(árbol), tuple(capas))
        cfg.validar()
        return cfg

    def validar(self) -> None:
        museos = self["ingesta.museos"]
        if not museos:
            raise ErrorDeConfiguración("`ingesta.museos` está vacío: no hay nada que ingestar")
        if not self["ingesta.tipos"]:
            raise ErrorDeConfiguración("`ingesta.tipos` está vacío")

        tabla = self["ingesta.tabla_de_licencias"]
        campos = self["ingesta.campos_de_texto"]
        for museo in museos:
            if museo not in campos:
                raise ErrorDeConfiguración(
                    f"el museo {museo!r} está en `ingesta.museos` y no tiene "
                    f"`ingesta.campos_de_texto.{museo}`"
                )
            if museo not in self["ingesta.campos_canonicos"]:
                raise ErrorDeConfiguración(
                    f"el museo {museo!r} no tiene `ingesta.campos_canonicos.{museo}`"
                )
            if museo not in self["ingesta.media_license"]:
                raise ErrorDeConfiguración(
                    f"el museo {museo!r} no tiene `ingesta.media_license.{museo}`: "
                    f"la licencia de imagen y metadata también se declara, no se asume"
                )
            faltantes = sorted(set(campos[museo]) - set(tabla.get(museo, {})))
            if faltantes:
                raise ErrorDeConfiguración(
                    f"`ingesta.tabla_de_licencias.{museo}` no declara "
                    f"{faltantes}. Un par (museo, section) sin entrada es error duro, "
                    f"no un default."
                )

        criterio = self["normalizacion.culture.criterio"]
        from tp3_rag.modulo_a.normalizador_de_culture import CRITERIOS

        if criterio not in CRITERIOS:
            raise ErrorDeConfiguración(
                f"`normalizacion.culture.criterio` = {criterio!r} no existe. "
                f"Disponibles: {sorted(CRITERIOS)} (el criterio definitivo sigue "
                f"abierto: lo que está implementado es el punto de variación, no "
                f"el valor)"
            )

        from tp3_rag.modulo_a.particionador import DefiniciónDeEstrato, cuota_de_hare
        from tp3_rag.modulo_a.selector_de_corpus import SelectorDeCorpus

        SelectorDeCorpus(self["ingesta.criterio_de_inclusion"])
        DefiniciónDeEstrato.desde_configuración(self)
        cuota_de_hare({"_": 1}, float(self["split.proporcion_held_out"]))

        from tp3_rag.modulo_a.fragmentador import EstrategiaDeChunking

        EstrategiaDeChunking.desde_configuración(self)
        if not self["fragmentacion.campos_de_knowledge"]:
            raise ErrorDeConfiguración(
                "`fragmentacion.campos_de_knowledge` está vacío: la colección de texto "
                "no tendría ni un fragmento y la registración rechazaría la variante. "
                "La ablación de campos elige cuáles, no si hay."
            )
        secciones_declaradas = {
            section
            for museo in museos
            for section in (campos.get(museo) or {})
        }
        desconocidas = sorted(
            set(self["fragmentacion.campos_de_knowledge"]) - secciones_declaradas
        )
        if desconocidas:
            raise ErrorDeConfiguración(
                f"`fragmentacion.campos_de_knowledge` pide {desconocidas}, que ningún "
                f"museo configurado produce (`ingesta.campos_de_texto`). Las disponibles "
                f"son {sorted(secciones_declaradas)}. Una `section` inexistente no puede "
                f"leerse como \"no hay texto\": daría obras con 0 fragmentos, y la "
                f"registración culparía al criterio de inclusión en vez de al typo."
            )

        base_url = self["proveedor.base_url"]
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ErrorDeConfiguración(
                f"`proveedor.base_url` = {base_url!r} no es una URL absoluta. La key de "
                f"la cátedra es una virtual key de un proxy LiteLLM, no de "
                f"`api.openai.com`: sin `base_url` el SDK apunta a su host por defecto "
                f"y la respuesta es 401. Vacía no es usar el default: es pegarle a otro "
                f"proveedor sin querer."
            )

        if int(self["embebido_texto.batch"]) <= 0:
            raise ErrorDeConfiguración("`embebido_texto.batch` tiene que ser > 0")
        if int(self["indice.batch_de_upsert"]) <= 0:
            raise ErrorDeConfiguración("`indice.batch_de_upsert` tiene que ser > 0")
        if self["indice.metrica"] not in ("cosine", "l2", "ip"):
            raise ErrorDeConfiguración(
                f"`indice.metrica` = {self['indice.metrica']!r} no es una de Chroma "
                f"(`cosine`, `l2`, `ip`). El diseño de recuperación asume coseno."
            )

        batch = int(self["embebido.batch"])
        if not 0 < batch <= int(self["embebido.batch_maximo"]):
            raise ErrorDeConfiguración(
                f"`embebido.batch` = {batch} tiene que estar en "
                f"(0, {self['embebido.batch_maximo']}]. El tope existe para que un batch "
                f"grande falle acá y no en el OOM killer a los 40 minutos de corrida."
            )

    def __str__(self) -> str:
        return json.dumps(self.a_dict(), ensure_ascii=False, indent=2, sort_keys=True)
