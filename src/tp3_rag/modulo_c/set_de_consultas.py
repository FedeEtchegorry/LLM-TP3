from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.dominio.consultas import (
    CATEGORÍA_CULTURAL,
    CATEGORÍAS,
    ESTADOS,
    N_IMAGENES,
    QUÉ_ESTRESA,
    VERSIÓN_DEL_SET,
    SetDeConsultas,
    nota_de_identidad,
)
from tp3_rag.modulo_c.vocabulario_de_culture import VocabularioDeCulture
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import (
    ErrorDeArtefacto,
    ErrorDeConfiguración,
    ErrorDeConstrucción,
)

if sys.version_info >= (3, 11):  # pragma: no cover - el lock fija 3.10
    import tomllib
else:
    import tomli as tomllib

CLAVES_DE_LA_DECLARACIÓN = ("split_id", "autoria", "estado", "revision", "imagenes")
CLAVES_DE_IMAGEN = ("artwork_id", "atributos", "consultas")
CLAVES_DE_CONSULTA = ("consulta_id", "categoria", "pedido")

@dataclass(frozen=True)
class Declaración:
    contenido: dict
    ruta: Path | None = None

    @property
    def split_id(self) -> str:
        return self.contenido["split_id"]

    @property
    def autoria(self) -> str:
        return self.contenido["autoria"]

    @property
    def estado(self) -> str:
        return self.contenido["estado"]

    @property
    def imagenes(self) -> list[dict]:
        return self.contenido["imagenes"]

    @classmethod
    def desde_toml(cls, ruta: str | Path) -> "Declaración":
        ruta = Path(ruta)
        if not ruta.exists():
            raise ErrorDeConfiguración(
                f"no existe la declaración {ruta}. Sacá una plantilla con "
                f"`poetry run tp3-consultas plantilla`."
            )
        with ruta.open("rb") as fh:
            contenido = tomllib.load(fh)
        declaración = cls(contenido, ruta=ruta)
        declaración.validar_forma()
        return declaración

    def validar_forma(self) -> None:
        d = self.contenido
        faltantes = [k for k in CLAVES_DE_LA_DECLARACIÓN if k not in d]
        if faltantes:
            raise ErrorDeConfiguración(
                f"la declaración no tiene {faltantes}. Las obligatorias son "
                f"{list(CLAVES_DE_LA_DECLARACIÓN)}: `split_id` es el ancla, "
                f"`autoria` es la mitad de la identidad y `revision` es el "
                f"*cuántas se revisaron* que el set de consultas pide."
            )
        de_más = sorted(
            set(d) - set(CLAVES_DE_LA_DECLARACIÓN) - {"criterio_de_culture", "nota"}
        )
        if de_más:
            raise ErrorDeConfiguración(
                f"la declaración trae claves que no se guardan: {de_más}. Lo que no está "
                f"en el esquema no existe, y en un artefacto de clase 3 una clave "
                f"ignorada en silencio es trabajo humano que se pierde."
            )
        if self.estado not in ESTADOS:
            raise ErrorDeConfiguración(
                f"`estado` = {self.estado!r} no es uno de {list(ESTADOS)}. "
                f"`plantilla` es el ejemplo que muestra la forma; `completo` son las 30."
            )
        identidad.etiqueta_declarada(self.autoria, campo="autoria")
        if not isinstance(d["revision"], Mapping) or "n_imagenes_revisadas" not in d["revision"]:
            raise ErrorDeConfiguración(
                "falta `revision.n_imagenes_revisadas`. Es lo que vuelve auditable el "
                "*'elegidas para cubrir variedad'* del diseño de la evaluación (el set de consultas, punto 2): si se "
                "revisaron 30 de 289, eso se dice."
            )
        if not self.imagenes:
            raise ErrorDeConfiguración("`[[imagenes]]` está vacío: no hay set.")
        for img in self.imagenes:
            self._validar_forma_de_imagen(img)

    @staticmethod
    def _validar_forma_de_imagen(img: Mapping[str, Any]) -> None:
        faltantes = [k for k in CLAVES_DE_IMAGEN if k not in img]
        if faltantes:
            raise ErrorDeConfiguración(
                f"una `[[imagenes]]` no tiene {faltantes} (obligatorias: "
                f"{list(CLAVES_DE_IMAGEN)}). Los `atributos` son el diseño y no "
                f"tienen default: son el juicio humano sobre la imagen que el esquema "
                f"canónico no tiene."
            )
        de_más = sorted(set(img) - set(CLAVES_DE_IMAGEN) - {"nota", "es_ejemplo"})
        if de_más:
            raise ErrorDeConfiguración(
                f"la imagen {img.get('artwork_id')!r} trae claves que no se guardan: "
                f"{de_más}."
            )
        for consulta in img["consultas"]:
            faltan = [k for k in CLAVES_DE_CONSULTA if k not in consulta]
            if faltan:
                raise ErrorDeConfiguración(
                    f"una consulta de {img.get('artwork_id')!r} no tiene {faltan}. El "
                    f"`consulta_id` **no tiene default a propósito**: los 30 se declaran "
                    f", y generarlo acá sería derivarlo — la puerta que  "
                    f"cerró. Usá un id legible y citable (`c01-visual`)."
                )
            sobran = sorted(
                set(consulta) - set(CLAVES_DE_CONSULTA) - {"cultura_pedida", "es_ejemplo"}
            )
            if sobran:
                raise ErrorDeConfiguración(
                    f"la consulta {consulta.get('consulta_id')!r} trae claves que no se "
                    f"guardan: {sobran}."
                )
            cultura = consulta.get("cultura_pedida")
            if cultura is not None:
                if not isinstance(cultura, Mapping) or "nombre" not in cultura or "etiquetas" not in cultura:
                    raise ErrorDeConfiguración(
                        f"`cultura_pedida` de {consulta.get('consulta_id')!r} necesita "
                        f"`nombre` y `etiquetas`.: es un *conjunto declarado de "
                        f"etiquetas* del vocabulario normalizado, no una etiqueta sola."
                    )
                sobran_cp = sorted(set(cultura) - {"nombre", "etiquetas", "nota"})
                if sobran_cp:
                    raise ErrorDeConfiguración(
                        f"`cultura_pedida` de {consulta.get('consulta_id')!r} trae "
                        f"{sobran_cp}. **El `n` no se declara: lo computa el autor** "
                        f"contra el corpus. Un `n` escrito a mano es un número sin "
                        f"fuente, y es el número que la cobertura cultural del resultado publica."
                    )

@dataclass
class ResultadoDeAutoría:
    set_de_consultas: SetDeConsultas
    ruta: Path
    rutas_de_imagenes: tuple[Path, ...]
    ya_estaba_congelado: bool
    vocabulario: VocabularioDeCulture | None = None

class AutorDelSetDeConsultas:
    def __init__(self, cfg, *, almacén, observabilidad=None) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.obs = observabilidad
        self.raíz = cfg["rutas.ground_truth_consultas"]
        self.raíz_de_imágenes = cfg["rutas.imagenes"]

    def autorar(
        self,
        declaración: Declaración,
        *,
        catálogo,
        manifiesto,
        derivar_otro: bool = False,
    ) -> ResultadoDeAutoría:
        if declaración.split_id != manifiesto.split_id:
            raise ErrorDeConstrucción(
                f"la declaración ancla a `split_id = {declaración.split_id}` y el "
                f"manifiesto leído es {manifiesto.split_id}. El set **ancla a un split** "
                f": anclarlo al que no es sería congelar 30 consultas sobre un "
                f"held-out que no es el que se midió."
            )
        criterio = declaración.contenido.get(
            "criterio_de_culture", self.cfg["normalizacion.culture.criterio"]
        )
        ident = identidad.identidad_del_set_de_consultas(
            split_id=declaración.split_id, autoria=declaración.autoria
        )

        existentes = SetDeConsultas.existentes(self.almacén, self.raíz)
        if ident in existentes:
            set_ = SetDeConsultas.leer(self.almacén, self.raíz, ident)
            return ResultadoDeAutoría(
                set_de_consultas=set_,
                ruta=self.almacén.ruta(SetDeConsultas.ruta_de(self.raíz, ident)),
                rutas_de_imagenes=tuple(
                    self.almacén.ruta(SetDeConsultas.ruta_de_imagen(self.raíz, i.artwork_id))
                    for i in set_.imagenes()
                ),
                ya_estaba_congelado=True,
            )
        del_mismo_split = SetDeConsultas.del_split(
            self.almacén, self.raíz, declaración.split_id
        )
        if del_mismo_split and not derivar_otro:
            raise ErrorDeConstrucción(
                f"ya hay {len(del_mismo_split)} set de consultas congelado para el split "
                f"{declaración.split_id} ({', '.join(del_mismo_split)}) y esta corrida "
                f"autoraría otro ({ident}).\n"
                f"  **Autorar otro set no es gratis**: cada juicio entra por un "
                f"`consulta_id` (la tabla de identidades, clase 3), así que los ~540 juicios del set "
                f"anterior quedan sin con qué compararse con lo que se mida sobre este.\n"
                f"  Si es a propósito —re-autorar las `cultura_pedida` porque cambió "
                f"el criterio de normalización de culture, u otra tanda de pedidos—, pasá `--derivar-otro`. Que haya "
                f"que decirlo **es** la clase 3 vuelta ejecutable, igual que en el "
                f"`Particionador`.\n"
                f"  🔑 Y si lo que querés es **re-autorar sin perder los juicios**: los 30 "
                f"`consulta_id` y los ~540 juicios **no dependen de la `autoría`** "
                f", así que un set nuevo que declare **los mismos `consulta_id` "
                f"sobre las mismas imágenes** los conserva. Lo que se mueve es el "
                f"vocabulario de las `cultura_pedida`, que es lo que se estaba re-autorando."
            )

        vocabulario = VocabularioDeCulture.del_corpus_del_split(
            catálogo, manifiesto, criterio=criterio
        )
        contenido = self._armar(
            identidad_del_set=ident,
            declaración=declaración,
            catálogo=catálogo,
            manifiesto=manifiesto,
            vocabulario=vocabulario,
            criterio=criterio,
        )
        set_ = SetDeConsultas(contenido)
        set_.validar()

        rutas = self._versionar_bytes(set_)
        set_.verificar_bytes(self.almacén, self.raíz)
        ruta = self.almacén.escribir_json(
            SetDeConsultas.ruta_de(self.raíz, ident), contenido
        )
        return ResultadoDeAutoría(
            set_de_consultas=set_,
            ruta=ruta,
            rutas_de_imagenes=rutas,
            ya_estaba_congelado=False,
            vocabulario=vocabulario,
        )

    def _versionar_bytes(self, set_: SetDeConsultas) -> tuple[Path, ...]:
        rutas = []
        for img in set_.imagenes():
            origen = self.almacén.ruta(
                Path(self.raíz_de_imágenes) / f"{img.artwork_id.replace(':', '_')}.jpg"
            )
            if not origen.exists():
                raise ErrorDeConstrucción(
                    f"no están los bytes de {img.artwork_id} en {origen}. El "
                    f"`AlmacénDeImágenes` es regenerable: corré `tp3-ingesta` antes de "
                    f"autorar, porque lo que se versiona acá es **una copia** de esos "
                    f"bytes (el set de consultas, )."
                )
            rutas.append(
                self.almacén.escribir_bytes(
                    SetDeConsultas.ruta_de_imagen(self.raíz, img.artwork_id),
                    origen.read_bytes(),
                )
            )
        return tuple(rutas)

    def _armar(
        self,
        *,
        identidad_del_set: str,
        declaración: Declaración,
        catálogo,
        manifiesto,
        vocabulario: VocabularioDeCulture,
        criterio: str,
    ) -> dict:
        registros = {r.artwork_id: r for r in catálogo.registros}
        held_out = set(manifiesto.ids_held_out)
        imágenes = []
        for img in declaración.imagenes:
            artwork_id = img["artwork_id"]
            registro = self._registro_del_held_out(artwork_id, registros, held_out, manifiesto)
            imágenes.append(
                {
                    "artwork_id": artwork_id,
                    "imagen_hash": registro.imagen.hash_bytes,
                    "archivo": str(SetDeConsultas.ruta_de_imagen(self.raíz, artwork_id)),
                    "atributos": list(img["atributos"]),
                    **({"nota": img["nota"]} if img.get("nota") else {}),
                    "es_ejemplo": bool(img.get("es_ejemplo", False)),
                    "consultas": [
                        self._armar_consulta(c, vocabulario=vocabulario, criterio=criterio)
                        for c in img["consultas"]
                    ],
                    "procedencia": {
                        "catalogo_id": catálogo.catalogo_id,
                        "copiada_de": str(
                            Path(self.raíz_de_imágenes)
                            / f"{artwork_id.replace(':', '_')}.jpg"
                        ),
                        "title": registro.title,
                        "artist": registro.artist,
                        "date_display": registro.date_display,
                        "year": registro.year,
                        "medium": registro.medium,
                        "museum": registro.museum,
                        "culture_normalizada": vocabulario.normalizar(registro.culture_raw),
                        "source_url": registro.source_url,
                        "media_license": registro.media_license,
                        "nota": (
                            "El `imagen_hash` de arriba es el `hash_bytes` del registro "
                            "canónico y el de la copia versionada: se verifican iguales al "
                            "autorar y la copia se re-verifica al leer (el set de consultas, )."
                        ),
                    },
                }
            )

        set_parcial = SetDeConsultas(
            {
                "identidad": identidad_del_set,
                "split_id": declaración.split_id,
                "autoria": declaración.autoria,
                "estado": declaración.estado,
                "criterio_de_culture": criterio,
                "revision": dict(declaración.contenido["revision"]),
                "imagenes": imágenes,
                "conteos": {},
            }
        )
        consultas = set_parcial.consultas()

        return {
            "artefacto": "SetDeConsultas",
            "version_del_set": VERSIÓN_DEL_SET,
            "clase_de_identidad": 3,
            "identidad": identidad_del_set,
            "split_id": declaración.split_id,
            "autoria": declaración.autoria,
            "nota_de_identidad": nota_de_identidad(),
            "estado": declaración.estado,
            "nota_de_estado": (
                "El `estado` va **adentro y no en la identidad**, igual que el del `Pool` "
                "(la tabla de identidades, recuadro 📌): pasar de `plantilla` a `completo` es **autorar otro "
                "set con otra `autoría`**, que es lo único que la clase 3 permite. Un set "
                "`plantilla` no es usable para medir y lo dice."
            ),
            "criterio_de_culture": criterio,
            "nota_de_criterio_de_culture": (
                ": el set declara el vocabulario en el que se autoraron las "
                "`cultura_pedida`. **No es ingrediente del id**: si entrara, "
                "cambiar el criterio de normalización de culture renombraría el set y desanclaría las 30 consultas "
                "por un cambio de vocabulario. La coherencia del par (set, variante) se "
                "chequea **al generar el informe** —con la identificación de variantes hay variantes coexistentes y "
                "cada una trae su criterio en su receta—, y cuando no valida, la cobertura cultural del resultado no se "
                "computa para esa variante: es una restricción de alcance declarada, no un "
                "invariante gratis."
            ),
            "revision": {
                **dict(declaración.contenido["revision"]),
                "n_held_out": len(held_out),
                "n_imagenes_elegidas": len(imágenes),
                "nota": (
                    "el set de consultas, punto 2: se etiquetan **las imágenes que se revisan para elegir "
                    "las 10**, no las 289 del held-out, y el artefacto registra cuántas se "
                    "miraron. Es lo que vuelve auditable el *'elegidas para cubrir "
                    "variedad'* del diseño de la evaluación."
                ),
            },
            "imagenes": imágenes,
            "conteos": {
                "n_imagenes": len(imágenes),
                "n_consultas": len(consultas),
                "n_pedidos_culturales": sum(1 for c in consultas if c.cultura_pedida),
                "por_categoria": set_parcial.por_categoria(),
            },
            "nota_de_conteos": (
                "`por_categoria` es lo que la tabla de ablación desglosa y el delta por categoría de pedido usa para el delta "
                "C3−C2. Los tres conteos del split (~2.880 / 2.605 / 289) **no se copian "
                "acá**: son del `ManifiestoDeSplit`, y lo que este artefacto cita es "
                "su `split_id`."
            ),
            "nota_de_declaracion": (
                "**La declaración no se guarda aparte, y es a propósito.** Lo de arriba "
                "ya es un **superconjunto** de lo que la persona escribió —mismo texto, "
                "más el `imagen_hash`, el `n` computado y la procedencia—, así que una "
                "copia cruda del TOML sería el mismo texto dos veces en el mismo archivo, "
                "que es el modo de falla 1 del proyecto en miniatura (los modos de falla del proyecto, duda 10). Lo "
                "que vuelve seguro no guardarla es que **nada se descarta en silencio**: "
                "la declaración rechaza toda clave que no vaya a terminar acá. Y el TOML "
                "es entonces un borrador que se puede tirar: lo irreproducible —el texto "
                "de los pedidos, los atributos, las etiquetas— está adentro **por valor**, "
                "que es el mismo criterio con el que la `RevisiónDeExplicaciones` copia "
                "las 25 explicaciones."
            ),
            "procedencia": {
                "catalogo_id": catálogo.catalogo_id,
                "fecha_de_autoria": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": _commit(),
                "declaracion_en": _ruta_publicable(declaración.ruta),
                "vocabulario_de_culture": {
                    "criterio": criterio,
                    "poblacion": vocabulario.poblacion,
                    "n_poblacion": vocabulario.n_poblacion,
                    "n_etiquetas_distintas": vocabulario.n_etiquetas,
                    "n_obras_sin_culture": vocabulario.n_sin_culture,
                },
                "capas_de_configuracion": list(self.cfg.capas),
            },
            "configuracion_resuelta": self.cfg.a_dict(),
            "config_id": self.cfg.config_id,
        }

    @staticmethod
    def _registro_del_held_out(artwork_id, registros, held_out, manifiesto):
        if artwork_id not in registros:
            raise ErrorDeConstrucción(
                f"{artwork_id} no está en el catálogo. Una imagen de consulta tiene que "
                f"ser una **obra real** del held-out: *'son obras reales que "
                f"el sistema nunca vio, así que no puede recuperarse a sí mismo'*."
            )
        if artwork_id not in held_out:
            raise ErrorDeConstrucción(
                f"{artwork_id} **no está en el held-out** del split "
                f"{manifiesto.split_id}. Las 10 imágenes salen del held-out (el diseño de la evaluación, "
                f"la exclusión del held-out de toda colección): una obra indexada se recuperaría a sí misma en top-1 y el "
                f"resultado sería el trivial que el held-out existe para evitar. Mirá "
                f"candidatas con `poetry run tp3-consultas candidatas`."
            )
        registro = registros[artwork_id]
        if registro.imagen is None:
            raise ErrorDeConstrucción(
                f"{artwork_id} no tiene imagen real en el catálogo, así que no hay bytes "
                f"que versionar ni `imagen_hash` que declarar."
            )
        return registro

    def _armar_consulta(
        self, consulta: Mapping[str, Any], *, vocabulario: VocabularioDeCulture, criterio: str
    ) -> dict:
        salida = {
            "consulta_id": consulta["consulta_id"],
            "categoria": consulta["categoria"],
            "pedido": consulta["pedido"],
            "es_ejemplo": bool(consulta.get("es_ejemplo", False)),
        }
        cultura = consulta.get("cultura_pedida")
        if cultura is None:
            return salida

        etiquetas = list(dict.fromkeys(cultura["etiquetas"]))
        unión = vocabulario.union(etiquetas)
        salida["cultura_pedida"] = {
            "nombre": cultura["nombre"],
            "etiquetas": etiquetas,
            "n": unión.n,
            "n_por_etiqueta": unión.n_por_etiqueta,
            "poblacion": unión.poblacion,
            "n_poblacion": unión.n_poblacion,
            "criterio_de_culture": criterio,
            **({"nota_del_autor": cultura["nota"]} if cultura.get("nota") else {}),
            "nota": (
                ": `cultura_pedida` es un **conjunto declarado de etiquetas** del "
                "vocabulario normalizado. El `n` es el tamaño de la **unión de obras** "
                f"(la suma de `n_por_etiqueta` da {unión.suma_de_etiquetas} y el "
                f"solapamiento es {unión.solapamiento}: una obra con dos etiquetas del "
                "grupo cuenta una vez). La población va al lado del número porque una "
                "cifra sin su población es el modo de falla 3 del proyecto. **Y el "
                "agrupamiento es evidencia del criterio de normalización de culture, que sigue abierto.**"
            ),
        }
        return salida

    def plantilla(
        self,
        *,
        split_id: str,
        autoria: str = "revision-1",
        artwork_ids: Sequence[str] = (),
        n_imagenes_revisadas: int | None = None,
        estado: str = "completo",
        criterio_de_culture: str | None = None,
    ) -> str:
        criterio = criterio_de_culture or self.cfg["normalizacion.culture.criterio"]
        ids = list(artwork_ids) or [f"REEMPLAZAR-{i + 1:02d}" for i in range(N_IMAGENES)]
        líneas = [
            "# Declaración del `SetDeConsultas` — el set de consultas, el diseño de la evaluación.",
            "#",
            "# La escribe **una persona**. Se congela:",
            f"#      poetry run tp3-consultas autorar --declaracion {'<este archivo>'}",
            "#",
            "#  🔴 Clase 3: **se congela una vez**. Corregir después de congelar no edita",
            "#  el set — autora **otro**, con otra `autoria` (y eso cuesta, porque los",
            "#  juicios del anterior quedan sin con qué compararse).",
            "#",
            f"#  Las {N_IMAGENES} imágenes salen del **held-out** y las elige el grupo.",
            "#  Mirá la lista corta con `poetry run tp3-consultas candidatas`.",
            "",
            f'split_id = "{split_id}"',
            f'autoria  = "{autoria}"          # etiqueta declarada: la pone una persona',
            f'estado   = "{estado}"           # plantilla | completo',
            f'criterio_de_culture = "{criterio}" # · el vocabulario de `cultura_pedida`',
            "",
            "[revision]",
            "# 🔑 el set de consultas: el artefacto registra **cuántas imágenes se miraron** para elegir",
            f"#  las {N_IMAGENES}. Es lo que vuelve auditable el *'elegidas para cubrir",
            "# variedad'* del diseño de la evaluación. Si se revisaron 30 de 289, se dice 30.",
            f"n_imagenes_revisadas = {n_imagenes_revisadas or len(ids)}",
            'como_se_revisaron = "REEMPLAZAR: p. ej. la lista corta de `tp3-consultas candidatas`"',
            "",
        ]
        for n, artwork_id in enumerate(ids, 1):
            líneas += [
                "# " + "─" * 74,
                "[[imagenes]]",
                f'artwork_id = "{artwork_id}"',
                "# 🔑 `atributos` = el diseño: paisaje, retrato, escena de",
                "#  interior, bodegón, abstracta, paleta fría, paleta cálida, período.",
                "#  **No existen en el esquema canónico**: son juicio humano sobre la",
                "#  imagen, y por eso no tienen default.",
                'atributos = ["REEMPLAZAR", "REEMPLAZAR"]',
                "",
            ]
            for categoría in CATEGORÍAS:
                sufijo = categoría.split("_")[0]
                líneas += [
                    "[[imagenes.consultas]]",
                    f'consulta_id = "c{n:02d}-{sufijo}"      # declarado, legible y citable',
                    f'categoria   = "{categoría}"   # estresa: {QUÉ_ESTRESA[categoría]}',
                    'pedido      = "REEMPLAZAR"',
                ]
                if categoría == CATEGORÍA_CULTURAL:
                    líneas += [
                        "#: **conjunto** de etiquetas del vocabulario normalizado, no",
                        "#  una sola. El `n` lo computa el autor contra el corpus; acá no va.",
                        "#  Mirá los grupos propuestos con `tp3-consultas culturas`.",
                        'cultura_pedida = { nombre = "REEMPLAZAR", etiquetas = ["REEMPLAZAR"] }',
                    ]
                líneas.append("")
        return "\n".join(líneas)

def _ruta_publicable(ruta: Path | None) -> str | None:
    if ruta is None:
        return None
    try:
        return str(Path(ruta).resolve().relative_to(RAÍZ_DEL_REPO))
    except ValueError:
        return Path(ruta).name

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

def verificar_hash_contra_el_catalogo(set_: SetDeConsultas, catálogo) -> None:
    registros = {r.artwork_id: r for r in catálogo.registros}
    for img in set_.imagenes():
        registro = registros.get(img.artwork_id)
        if registro is None or registro.imagen is None:
            raise ErrorDeArtefacto(
                f"{set_.identidad}: {img.artwork_id} no está en el catálogo "
                f"{catálogo.catalogo_id} o no tiene imagen. El set ancla a un split del "
                f"que esta obra era held-out: si el catálogo cambió, lo que cambió es el "
                f"universo, no el set."
            )
        if registro.imagen.hash_bytes != img.imagen_hash:
            raise ErrorDeArtefacto(
                f"{set_.identidad}: {img.artwork_id} tiene `imagen_hash` "
                f"{img.imagen_hash[:23]}… en el set y {registro.imagen.hash_bytes[:23]}… "
                f"en el catálogo {catálogo.catalogo_id}. **El museo reemplazó el JPEG.** "
                f"Los bytes versionados del set son los que se midieron y **no se "
                f"tocan**: lo que hay que decidir es si se re-ingesta."
            )
