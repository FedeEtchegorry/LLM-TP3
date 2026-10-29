from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from tp3_rag.dominio.catalogo import RegistroCanónico
from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.dominio.variante import (
    INGREDIENTES_DEL_ID,
    VERSIÓN_DEL_REGISTRO,
    RegistroDeÍndices,
)
from tp3_rag.modulo_a.embebedor_de_imagenes import CachéDeEmbeddingsDeImagen
from tp3_rag.modulo_a.fragmentador import Fragmentador
from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDeConstrucción
from tp3_rag.puertos.embedding_imagen import IdentidadDelModelo
from tp3_rag.puertos.embedding_texto import normalizar as normalizar_texto
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE

TOLERANCIA_DE_NORMA = 1e-3

CAMPOS_DE_ARTWORKS = (
    "museum",
    "title",
    "artist",
    "date_display",
    "year",
    "medium",
    "style",
    "culture",
    "type",
    "image_url",
    "source_url",
)

def derivar_variante_id(cfg, *, catalogo_id: str, split_id: str, fragmentador: Fragmentador) -> str:
    return identidad.variante_id(
        catalogo_id=catalogo_id,
        split_id=split_id,
        criterio_de_inclusion=dict(cfg["ingesta.criterio_de_inclusion"]),
        checkpoint_de_clip=cfg["embebido.checkpoint"],
        version_del_preprocesado=cfg["embebido.version_del_preprocesado"],
        estrategia_de_chunking=fragmentador.estrategia.a_json(),
        campos_de_knowledge=fragmentador.campos_de_knowledge,
        modelo_de_embedding_de_texto=cfg["embebido_texto.modelo"],
        criterio_de_culture=cfg["normalizacion.culture.criterio"],
    )

@dataclass
class ResultadoDeConstrucción:
    variante_id: str
    n_indexadas: int = 0
    n_fragmentos: int = 0
    ids_indexados: tuple[str, ...] = ()
    fragmentos_por_obra: dict = field(default_factory=dict)
    conteos_por_seccion: dict = field(default_factory=dict)
    registro: RegistroDeÍndices | None = None
    ruta_del_registro: Path | None = None
    ya_estaba_registrada: bool = False
    hasta: str = "registro"

    @property
    def registrada(self) -> bool:
        return self.registro is not None

class ConstructorDeÍndice:
    def __init__(
        self,
        cfg,
        *,
        almacén,
        indice,
        manifiesto: ManifiestoDeSplit,
        catalogo_id: str,
        embebedor_de_texto=None,
        observabilidad=None,
        fragmentador: Fragmentador | None = None,
    ) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.indice = indice
        self.manifiesto = manifiesto
        self.catalogo_id = catalogo_id
        self.embebedor_de_texto = embebedor_de_texto
        self.obs = observabilidad or _ObservabilidadMuda()
        self.fragmentador = fragmentador or Fragmentador(cfg)
        self.culture = NormalizadorDeCulture(cfg["normalizacion.culture.criterio"])

        self.modelo_de_imagen = IdentidadDelModelo(
            checkpoint=cfg["embebido.checkpoint"],
            version_del_preprocesado=cfg["embebido.version_del_preprocesado"],
            dim=512,
        )
        self.caché = CachéDeEmbeddingsDeImagen(
            almacén.ruta(cfg["rutas.embeddings_imagen"]), self.modelo_de_imagen
        )
        self.variante_id = derivar_variante_id(
            cfg,
            catalogo_id=catalogo_id,
            split_id=manifiesto.split_id,
            fragmentador=self.fragmentador,
        )
        if self.indice.variante_id != self.variante_id:
            raise ErrorDeConstrucción(
                f"el índice está abierto en {self.indice.variante_id} y esta receta da "
                f"{self.variante_id}. El `variante_id` va en el path de las colecciones "
                f" y no puede discrepar de la receta que las llena."
            )

    def receta(self) -> dict:
        return {
            "catalogo_id": self.catalogo_id,
            "split_id": self.manifiesto.split_id,
            "criterio_de_inclusion": dict(self.cfg["ingesta.criterio_de_inclusion"]),
            "checkpoint_de_clip": self.modelo_de_imagen.checkpoint,
            "version_del_preprocesado": self.modelo_de_imagen.version_del_preprocesado,
            "estrategia_de_chunking": self.fragmentador.estrategia.a_json(),
            "campos_de_knowledge": list(self.fragmentador.campos_de_knowledge),
            "modelo_de_embedding_de_texto": self.cfg["embebido_texto.modelo"],
            "criterio_de_culture": self.culture.criterio,
            "limite_n_de_la_ingesta": self.cfg["ingesta.limite_n"],
            "dim_del_embedding_de_texto": self.cfg["embebido_texto.dim"],
            "base_url_del_proveedor": self.cfg["proveedor.base_url"],
            "nota_del_base_url": (
                "**Procedencia, no ingrediente** (los hallazgos, decisión de identidad "
                "del encargo ). El `base_url` es **ruteo**: lo que determina los "
                "vectores es el modelo, y el modelo ya está en el id — que es la regla de "
                " (*un ingrediente queda fuera solo si otro ya lo determina*). Si "
                "entrara, el día que la cátedra mueva el proxy **se invalidaría el índice "
                "entero**, que es el costo injustificado que rechazó para el valor "
                "de los headers. ⚠️ **Riesgo residual declarado**: si el proxy sirviera "
                "otro modelo bajo el mismo nombre, la identidad mentiría. Es la misma "
                "confianza que ya se le da a `api.openai.com`; por eso el host se declara "
                "acá en vez de no quedar registrado en ningún lado."
            ),
            "metrica_del_indice": self.cfg["indice.metrica"],
            "campos_de_metadata_de_artworks": list(CAMPOS_DE_ARTWORKS),
            "tabla_de_licencias_de_nr1": _a_dict(self.cfg["ingesta.tabla_de_licencias"]),
            "nota_de_la_tabla_de_licencias": (
                "Copia **como procedencia** (los invariantes de la registración inv. 4, la configuración en capas deuda el modelo del LLM por rol). Su hogar "
                "es la configuración de la ingesta, que es anterior a cualquier variante: "
                "el `Normalizador` la necesita cuando todavía no existe ninguna. "
                "Modelarlo al revés pondría la ingesta a esperar una variante."
            ),
            "nota_de_los_ingredientes": (
                "Los nueve primeros son los ingredientes del `variante_id`. "
                "⚠️ **Dos de ellos no están en la lista que los invariantes de la registración inv. 4 escribe**: el "
                "`tamano_objetivo` (dentro de `estrategia_de_chunking`) y la "
                "`version_del_preprocesado`. Los dos **cambian el contenido** de la "
                "variante y ningún otro ingrediente los determina, así que dejarlos fuera "
                "haría falsa a la regla de que un id determina su contenido. El límite de `n` y la tabla de licencias sí "
                "quedan fuera del id, pero no por ser 'parámetros': porque el "
                "`catalogo_id` ya los determina. **Va reportado.**"
            ),
            "configuracion_de_indexado_resuelta": {
                "fragmentacion": _a_dict(self.cfg["fragmentacion"]),
                "embebido": _a_dict(self.cfg["embebido"]),
                "embebido_texto": _a_dict(self.cfg["embebido_texto"]),
                "indice": _a_dict(self.cfg["indice"]),
                "normalizacion": _a_dict(self.cfg["normalizacion"]),
            },
        }

    def verificar_cobertura(self, registros: Sequence[RegistroCanónico]) -> tuple[RegistroCanónico, ...]:
        por_id = {r.artwork_id: r for r in registros}
        faltantes = self.manifiesto.cobertura_contra(por_id)
        if faltantes:
            raise ErrorDeConstrucción(
                f"el catálogo {self.catalogo_id} **no cubre** el lado corpus del split "
                f"{self.manifiesto.split_id}: faltan {len(faltantes)} ids "
                f"(p. ej. {list(faltantes[:5])}). La igualdad de conjuntos del invariante "
                f"2 de los invariantes de la registración es **insatisfacible** y ninguna variante de este split se "
                f"puede registrar. Salidas: (a) re-ingestar esas obras, que se piden por "
                f"id, o (b) derivar otro split con `--derivar-otro` y su precio de "
                f"anotación a la vista. **Registrar la intersección no es una salida**: "
                f"publicaría una `n` indexada menor que el "
                f"{self.manifiesto.conteos['corpus_indexado']} anunciado."
            )
        return tuple(por_id[i] for i in sorted(self.manifiesto.ids_corpus))

    def construir_artworks(self, corpus: Sequence[RegistroCanónico]) -> int:
        self.indice.crear_coleccion(ARTWORKS, self.variante_id)
        ids, vectores, metadatas = [], [], []
        for registro in corpus:
            ids.append(registro.artwork_id)
            vectores.append(self._vector_de(registro))
            metadatas.append(self._metadata_de(registro))
        n = self.indice.upsert(
            ARTWORKS, ids=ids, vectores=vectores, metadatas=metadatas
        )
        self.obs.contar("artworks.upsert", n)
        return n

    def _vector_de(self, registro: RegistroCanónico):
        if registro.imagen is None:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id} está en el lado corpus del split y **no tiene "
                f"imagen**, así que no hay vector que guardar en `artworks`. Es el caso "
                f"que el relevamiento de datos y fuentes describe: "
                f"un predicado de texto solo vuelve insatisfacible el invariante 2 de "
                f"los invariantes de la registración. El criterio de este split es "
                f"{dict(self.cfg['ingesta.criterio_de_inclusion'])}."
            )
        vector = self.caché.leer(registro.artwork_id, registro.imagen.hash_bytes)
        if vector is None:
            raise ErrorDeConstrucción(
                f"{registro.artwork_id}: no hay vector en el caché de embeddings para "
                f"(hash {registro.imagen.hash_bytes[:20]}…, "
                f"checkpoint {self.modelo_de_imagen.checkpoint}, preproc "
                f"{self.modelo_de_imagen.version_del_preprocesado}). `artworks` se llena "
                f"**leyendo el caché** (el reuso del caché de imagen entre variantes): el constructor no embebe. Corré "
                f"`poetry run tp3-embeber`."
            )
        norma = float(sum(float(x) * float(x) for x in vector) ** 0.5)
        if abs(norma - 1.0) > TOLERANCIA_DE_NORMA:
            raise ErrorDeConstrucción(
                f"la verificación de la norma de los vectores: el vector cacheado de {registro.artwork_id} tiene norma {norma:.6f} "
                f"y no 1. Ya venía del caché, y se verifica igual porque un vector no "
                f"unitario **arruina `α·e_img + (1−α)·e_txt`** y el síntoma aparecería "
                f"recién en la consulta. Un `.npy` truncado se lee como un vector válido."
            )
        return [float(x) for x in vector]

    def _metadata_de(self, registro: RegistroCanónico) -> dict:
        crudo = {
            "museum": registro.museum,
            "title": registro.title,
            "artist": registro.artist,
            "date_display": registro.date_display,
            "year": registro.year,
            "medium": registro.medium,
            "style": registro.style,
            "culture": self.culture.normalizar(registro.culture_raw),
            "type": registro.type,
            "image_url": registro.imagen.image_url if registro.imagen else None,
            "source_url": registro.source_url,
        }
        return {clave: crudo[clave] for clave in CAMPOS_DE_ARTWORKS}

    def fragmentar(self, corpus: Sequence[RegistroCanónico]) -> dict[str, list[Fragmento]]:
        por_obra = self.fragmentador.fragmentar_muchos(corpus)
        self.obs.contar("fragmentos", sum(len(v) for v in por_obra.values()))
        return por_obra

    def construir_knowledge(self, por_obra: Mapping[str, list[Fragmento]]) -> int:
        if self.embebedor_de_texto is None:
            raise ErrorDeConstrucción(
                "`construir_knowledge` necesita un `PuertoEmbeddingTexto` y no se le "
                "pasó ninguno. Sin `knowledge` **no se puede registrar**: el invariante "
                "3 de los invariantes de la registración es parte de la transacción de registración."
            )
        fragmentos = [f for obra in sorted(por_obra) for f in por_obra[obra]]
        if not fragmentos:
            raise ErrorDeConstrucción(
                "no hay ni un fragmento que indexar en `knowledge`. Con el conjunto de "
                f"campos {list(self.fragmentador.campos_de_knowledge)} ninguna obra del "
                f"corpus produjo texto, así que el invariante 3 de los invariantes de la registración es falso para "
                f"las {len(por_obra)} obras."
            )
        self.indice.crear_coleccion(KNOWLEDGE, self.variante_id)
        batch = int(self.cfg["embebido_texto.batch"])
        total = 0
        for desde in range(0, len(fragmentos), batch):
            lote = fragmentos[desde : desde + batch]
            vectores = self.embebedor_de_texto.embeber(
                [f.texto for f in lote], etiquetas=[f.fragment_id for f in lote]
            )
            total += self.indice.upsert(
                KNOWLEDGE,
                ids=[f.fragment_id for f in lote],
                vectores=[
                    normalizar_texto(v, etiqueta=f.fragment_id)
                    for v, f in zip(vectores, lote)
                ],
                metadatas=[f.a_metadata() for f in lote],
                documentos=[f.texto for f in lote],
            )
            self.obs.progreso(f"knowledge {total}/{len(fragmentos)} fragmentos")
        self.obs.cerrar_progreso()
        return total

    def medición_del_embebido_de_texto(self, por_obra: Mapping[str, list[Fragmento]]) -> dict:
        libro = getattr(self.obs, "libro_mayor", lambda: {})() or {}
        rol = dict(libro.get("embedding_texto") or {})
        modelo = self.cfg["embebido_texto.modelo"]
        precios = self.cfg.get("proveedor.precio_usd_por_millon_de_tokens") or {}
        precio = precios.get(modelo)
        tokens = int(rol.get("tokens_in", 0))
        n_fragmentos = sum(len(v) for v in por_obra.values())
        return {
            "poblacion": {
                "n_obras_del_corpus": len(por_obra),
                "n_fragmentos": n_fragmentos,
                "campos_de_knowledge": list(self.fragmentador.campos_de_knowledge),
                "modelo": modelo,
                "nota": (
                    "**El corpus indexado, no el universo.** Es la población que la tabla de tokens, costo y la figura "
                    "necesita y la misma sobre la que la ablación de campos de la colección de texto proyectó: agregar "
                    "`exhibitions` y `citations` la mueve, así que la cifra sin la "
                    "población al lado no se puede comparar con nada."
                ),
            },
            "medido": {
                "hubo_llamadas": bool(rol),
                "n_llamadas": rol.get("n_llamadas", 0),
                "n_fragmentos_embebidos": rol.get("n_items", 0),
                "tokens_in": tokens,
                "tokens_out": rol.get("tokens_out", 0),
                "segundos_de_api": rol.get("segundos", 0.0),
                "tokens_por_fragmento": round(tokens / n_fragmentos, 1) if n_fragmentos else None,
                "resultados": rol.get("resultados", {}),
            },
            "precio_usd_por_millon_de_tokens": precio,
            "costo_usd": round(tokens * float(precio) / 1_000_000, 6) if precio else None,
            "nota_del_precio": (
                "Los tokens son **medidos** (el `usage` de la API, vía el libro mayor de "
                "los tokens en el artefacto de corrida); el precio es **declarado** en `proveedor."
                "precio_usd_por_millon_de_tokens` y esta corrida no lo pudo verificar "
                "contra una fuente: `/key/info` del proxy está bloqueado para virtual "
                "keys. El USD es entonces un **derivado declarado**, y si el modelo no "
                "tuviera precio declarado sería `null` en vez de 0."
            ),
        }

    def registrar(
        self,
        *,
        por_obra: Mapping[str, list[Fragmento]],
        cita_al_embebido: Mapping[str, object],
    ) -> ResultadoDeConstrucción:
        invariantes = self.verificar_invariantes(por_obra=por_obra)

        contenido = {
            "artefacto": "RegistroDeÍndices",
            "version_del_registro": VERSIÓN_DEL_REGISTRO,
            "clase_de_identidad": 1,
            "variante_id": self.variante_id,
            "nota_de_identidad": (
                "Clase 1: mismo id ⇒ mismo contenido. Incluye el `catalogo_id` "
                "**a propósito** —una re-ingesta puede cambiar el contenido de los "
                "registros que indexa, así que una variante nueva es honestamente otra—; "
                "y **excluye el ordinal** del reporte del embebido, que vive en "
                "`procedencia`: si subiera, cada re-corrida del `EmbebedorDeImágenes` "
                "sería una variante nueva y habría que reindexar sin que cambiara un solo "
                "registro ni un solo vector (la tabla de identidades, recuadro 🔴)."
            ),
            "receta": self.receta(),
            "conteos": {
                "n_indexadas": invariantes["n_indexadas"],
                "n_fragmentos": invariantes["n_fragmentos"],
                "fragmentos_por_seccion": self.fragmentador.conteos_por_seccion(),
            },
            "nota_de_conteos": (
                "Este artefacto es el **dueño** de `n indexadas` y `n fragmentos` y de "
                "nada más. `n embebidas` es del **universo** y vive en el "
                "reporte del embebido: se **cita** por `catalogo_id` + ordinal en vez de "
                "copiarse, porque una cifra con dos copias es el modo de falla 1 del "
                "proyecto. Los tres conteos del split son del `ManifiestoDeSplit`."
            ),
            "colecciones": {
                "raiz": str(Path(self.cfg["rutas.chroma"]) / self.variante_id),
                "artworks": ARTWORKS,
                "knowledge": KNOWLEDGE,
                "nota": (
                    "El registro dice **dónde** están los vectores; los vectores no van "
                    "al repo. el contrato de los puertos y el layout de artefactos discrepan sobre si el `variante_id` "
                    "va en el nombre de la colección o en el path: se sigue el layout de artefactos (el "
                    "path) y además va en la metadata de la colección y se verifica al "
                    "abrir. **Va reportado.**"
                ),
            },
            "invariantes_de_9_6": invariantes,
            "costo_del_embebido_de_texto": self.medición_del_embebido_de_texto(por_obra),
            "libro_mayor": getattr(self.obs, "libro_mayor", lambda: {})(),
            "procedencia": {
                "catalogo_id": self.catalogo_id,
                "split_id": self.manifiesto.split_id,
                "embebido": dict(cita_al_embebido),
                "nota_del_embebido": (
                    "**Cita, no ingrediente** (la tabla de identidades, recuadro 🔴). Es lo que vuelve "
                    "exacta la pregunta *de qué ejecución salió esta `n` embebidas*, y "
                    "lo que permite verificar `n embebidas ≥ n indexadas` contra **un** "
                    "reporte y no contra un `catalogo_id` que nombra varios."
                ),
                "fecha_de_construccion": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "commit": _commit(),
                "capas_de_configuracion": list(self.cfg.capas),
            },
            "configuracion_resuelta": self.cfg.a_dict(),
            "config_id": self.cfg.config_id,
        }

        registro = RegistroDeÍndices(contenido)
        registro.validar()

        raíz = self.cfg["rutas.artefactos_indices"]
        ruta_relativa = RegistroDeÍndices.ruta_de(raíz, self.variante_id)
        ya_estaba = self.almacén.existe(ruta_relativa)
        if ya_estaba:
            previo = RegistroDeÍndices.leer(self.almacén, raíz, self.variante_id)
            _exigir_mismo_contenido(previo, registro)
            return ResultadoDeConstrucción(
                variante_id=self.variante_id,
                n_indexadas=invariantes["n_indexadas"],
                n_fragmentos=invariantes["n_fragmentos"],
                ids_indexados=tuple(sorted(self.indice.ids(ARTWORKS))),
                fragmentos_por_obra={k: len(v) for k, v in por_obra.items()},
                conteos_por_seccion=self.fragmentador.conteos_por_seccion(),
                registro=previo,
                ruta_del_registro=self.almacén.ruta(ruta_relativa),
                ya_estaba_registrada=True,
            )

        ruta = self.almacén.escribir_json(ruta_relativa, contenido)
        return ResultadoDeConstrucción(
            variante_id=self.variante_id,
            n_indexadas=invariantes["n_indexadas"],
            n_fragmentos=invariantes["n_fragmentos"],
            ids_indexados=tuple(sorted(self.indice.ids(ARTWORKS))),
            fragmentos_por_obra={k: len(v) for k, v in por_obra.items()},
            conteos_por_seccion=self.fragmentador.conteos_por_seccion(),
            registro=registro,
            ruta_del_registro=ruta,
        )

    def verificar_invariantes(
        self, *, por_obra: Mapping[str, list[Fragmento]], materializado: bool = True
    ) -> dict:
        corpus = set(self.manifiesto.ids_corpus)
        held_out = set(self.manifiesto.ids_held_out)
        ids_indexados = (
            set(self.indice.ids(ARTWORKS)) if materializado else set(por_obra) & corpus
        )

        intrusos = sorted(ids_indexados & held_out)
        if intrusos:
            raise ErrorDeConstrucción(
                f"invariante 1 de se rompió los invariantes de la registración: {len(intrusos)} obras del "
                f"held-out están indexadas (p. ej. {intrusos[:3]}). Se verifica contra el "
                f"`ManifiestoDeSplit` y **no** contra un flag de los registros, que no "
                f"existe: un flag valdría `false` en el 100 % de los indexados y no "
                f"distinguiría nada. El held-out es la fuente de las 10 imágenes "
                f"de consulta: si entra al índice, el self-retrieval mide memoria."
            )

        faltan = sorted(corpus - ids_indexados)
        sobran = sorted(ids_indexados - corpus)
        if faltan or sobran:
            raise ErrorDeConstrucción(
                f"invariante 2 de se rompió los invariantes de la registración (el **guardrail 🔒** y el dataset documentado y reproducible): los ids "
                f"indexados no **son** el lado corpus del manifiesto. Faltan "
                f"{len(faltan)} (p. ej. {faltan[:3]}) y sobran {len(sobran)} "
                f"(p. ej. {sobran[:3]}). Es igualdad de **conjuntos** y no de conteos a "
                f"propósito: dos errores que se compensan dan el mismo número. El caso "
                f"que atrapa es indexar el universo con imagen y publicar eso donde "
                f"corresponde {len(corpus)}."
            )

        n_indexadas = len(ids_indexados)
        n_embebidas = self._n_embebidas_citada()
        if n_embebidas is not None and n_embebidas < n_indexadas:
            raise ErrorDeConstrucción(
                f"invariante 2 de los invariantes de la registración, segunda mitad: `n embebidas` = {n_embebidas} "
                f"< `n indexadas` = {n_indexadas}. Son **dos números separados** (el dataset documentado y reproducible) y "
                f"la diferencia tiene que reportarse como **esperada**, no al revés: "
                f"indexar más de lo que se embebió es imposible salvo que el caché "
                f"tuviera vectores de otra ejecución."
            )

        sin_fragmentos = sorted(i for i in corpus if not por_obra.get(i))
        if sin_fragmentos:
            raise ErrorDeConstrucción(
                f"invariante 3 de los invariantes de la registración (al menos un fragmento por obra indexada, y los criterios de entrega el criterio de trazabilidad por "
                f"construcción) roto: {len(sin_fragmentos)} obras indexadas en `artworks` "
                f"no tienen ni un fragmento en `knowledge` "
                f"(p. ej. {sin_fragmentos[:3]}). Con "
                f"`campos_de_knowledge = {list(self.fragmentador.campos_de_knowledge)}`: "
                f"un conjunto de campos que excluya `description` **se rechaza al "
                f"registrar, no al medir**. Es además el detector de "
                f"el corpus parametrizado: si alguna pieza duplica el criterio de inclusión —el candidato "
                f"es el `Fragmentador`—, con un criterio más amplio la obra sin texto "
                f"entra a la membresía y sale con 0 fragmentos."
            )

        receta = self.receta()
        faltantes = sorted(set(INGREDIENTES_DEL_ID) - set(receta))
        sin_valor = sorted(
            k for k in INGREDIENTES_DEL_ID if receta.get(k) in (None, "", [], {})
        )
        if faltantes or sin_valor:
            raise ErrorDeConstrucción(
                f"invariante 4 de se rompió los invariantes de la registración: la receta no es un superconjunto de los "
                f"ingredientes del `variante_id`. Faltan {faltantes}; sin valor: "
                f"{sin_valor}. *Una variante cuya receta no dice con qué criterio de "
                f"`culture` se construyó es una variante que los informes no pueden "
                f"describir*."
            )

        n_fragmentos = sum(len(por_obra[i]) for i in sorted(corpus) if i in por_obra)
        return {
            "1_held_out_fuera": {
                "cumple": True,
                "n_held_out": len(held_out),
                "verificado_contra": f"ManifiestoDeSplit {self.manifiesto.split_id}",
            },
            "2_igualdad_de_conjuntos_y_a6": {
                "cumple": True,
                "n_indexadas": n_indexadas,
                "n_corpus_del_manifiesto": len(corpus),
                "igualdad_de_conjuntos": True,
                "n_embebidas_citada": n_embebidas,
                "diferencia_embebidas_menos_indexadas": (
                    n_embebidas - n_indexadas if n_embebidas is not None else None
                ),
                "diferencia_declarada_esperada": (
                    "**Sí, y es el resultado correcto.** `n embebidas` es del **universo "
                    "con imagen** y `n indexadas` es del **corpus**: se embebe todo lo que "
                    "tiene imagen real porque el criterio de inclusión filtra por texto y "
                    "el texto no cambia el embedding de la imagen. "
                    "La diferencia son las obras con imagen que no pasan el criterio más "
                    "el held-out, que no entra a ninguna colección (la exclusión del held-out de toda colección)."
                ),
            },
            "3_toda_obra_indexada_tiene_fragmento": {
                "cumple": True,
                "n_obras_con_fragmentos": sum(1 for i in corpus if por_obra.get(i)),
                "n_fragmentos": n_fragmentos,
                "min_fragmentos_por_obra": min(
                    (len(por_obra.get(i, ())) for i in corpus), default=0
                ),
            },
            "4_receta_superconjunto": {
                "cumple": True,
                "n_ingredientes_del_id": len(INGREDIENTES_DEL_ID),
                "n_claves_de_la_receta": len(receta),
            },
            "n_indexadas": n_indexadas,
            "n_fragmentos": n_fragmentos,
        }

    def _n_embebidas_citada(self) -> int | None:
        reportes = self._reportes_del_embebido()
        if not reportes:
            return None
        contenido = self.almacén.leer_json(reportes[-1][1])
        return int(contenido["embebido"]["n_embebidas"])

    def _reportes_del_embebido(self) -> list[tuple[int, Path]]:
        raíz = self.almacén.ruta(self.cfg["rutas.artefactos_embebido"])
        if not raíz.exists():
            return []
        prefijo = f"{self.catalogo_id}-{self.modelo_de_imagen.checkpoint.replace('/', '__')}-"
        encontrados = []
        for ruta in raíz.glob(f"{prefijo}*.json"):
            try:
                ordinal = int(ruta.stem[len(prefijo) :])
            except ValueError:  # pragma: no cover - un nombre que no es ordinal
                continue
            encontrados.append((ordinal, Path(self.cfg["rutas.artefactos_embebido"]) / ruta.name))
        return sorted(encontrados)

    def cita_al_embebido(self) -> dict:
        reportes = self._reportes_del_embebido()
        if not reportes:
            return {
                "catalogo_id": self.catalogo_id,
                "checkpoint": self.modelo_de_imagen.checkpoint,
                "ordinal": None,
                "nota": (
                    "No hay reporte del embebido para este (catálogo, checkpoint). La "
                    "segunda mitad del invariante 2 —`n embebidas ≥ n indexadas`— queda "
                    "**sin verificar**, y se dice en vez de verificarse contra un número "
                    "inventado."
                ),
            }
        ordinal, ruta = reportes[-1]
        contenido = self.almacén.leer_json(ruta)
        return {
            "catalogo_id": self.catalogo_id,
            "checkpoint": self.modelo_de_imagen.checkpoint,
            "ordinal": ordinal,
            "artefacto": str(ruta),
            "n_embebidas": int(contenido["embebido"]["n_embebidas"]),
            "img_por_segundo": contenido["embebido"].get("img_por_segundo"),
            "n_ordinales_disponibles": len(reportes),
        }

def _exigir_mismo_contenido(previo: RegistroDeÍndices, nuevo: RegistroDeÍndices) -> None:
    diferencias = [
        clave
        for clave in sorted(set(previo.receta) | set(nuevo.receta))
        if previo.receta.get(clave) != nuevo.receta.get(clave)
    ]
    if diferencias:
        raise ErrorDeArtefacto(
            f"la variante {nuevo.variante_id} ya está registrada con **otra receta**: "
            f"difieren {diferencias}. la regla de que un id determina su contenido lo declara error duro de clase 1, y su "
            f"causa declarada es *falta un ingrediente en la tabla de identidades*: dos recetas distintas "
            f"con el mismo id significan que algún eje de la receta **no está en la "
            f"identidad**. Es el detector de la identificación de variantes."
        )
    for clave in ("n_indexadas", "n_fragmentos"):
        if previo.conteos.get(clave) != nuevo.conteos.get(clave):
            raise ErrorDeArtefacto(
                f"la variante {nuevo.variante_id} ya está registrada con "
                f"`{clave} = {previo.conteos.get(clave)}` y esta construcción da "
                f"{nuevo.conteos.get(clave)}. Misma receta y otro conteo es la clase 1 "
                f"rota por un camino peor: el contenido del catálogo o del caché cambió "
                f"sin que cambiara ninguna identidad."
            )

def _a_dict(valor):
    return valor.a_dict() if hasattr(valor, "a_dict") else dict(valor)

class _ObservabilidadMuda:
    def libro_mayor(self) -> dict:
        return {}

    def contar(self, *_a, **_k):
        pass

    def decir(self, *_a, **_k):
        pass

    def progreso(self, *_a, **_k):
        pass

    def cerrar_progreso(self, *_a, **_k):
        pass

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
