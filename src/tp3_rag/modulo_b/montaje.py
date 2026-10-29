from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tp3_rag.dominio.consultas import SetDeConsultas
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.dominio.variante import RegistroDeÍndices
from tp3_rag.nucleo.errores import ErrorDeArtefacto

@dataclass
class Montaje:
    registro: RegistroDeÍndices
    manifiesto: ManifiestoDeSplit
    indice: Any
    puerto_de_imagen: Any = None
    embebedor_de_texto: Any = None
    puerto_de_llm: Any = None

    @property
    def variante_id(self) -> str:
        return self.registro.variante_id

    @property
    def split_id(self) -> str:
        return self.manifiesto.split_id

    def cerrar(self) -> None:
        if hasattr(self.indice, "cerrar"):
            self.indice.cerrar()

def registro_de_indices(cfg, almacén, variante_id: str | None) -> RegistroDeÍndices:
    raíz = cfg["rutas.artefactos_indices"]
    if variante_id:
        return RegistroDeÍndices.leer(almacén, raíz, variante_id)
    existentes = RegistroDeÍndices.existentes(almacén, raíz)
    if not existentes:
        raise ErrorDeArtefacto(
            "no hay ninguna variante registrada. Corré `poetry run tp3-indexar` primero: "
            "el contrato de los puertos convierte una colección ausente en **error duro nombrando el "
            "`variante_id`** —*un ranking vacío se degradaría elegantemente y nadie se "
            "enteraría de que el índice no estaba*— así que consultar sin variante no es "
            "una corrida pobre, es una corrida imposible."
        )
    if len(existentes) > 1:
        raise ErrorDeArtefacto(
            f"hay {len(existentes)} variantes registradas: {list(existentes)}. Pasá "
            f"`--variante-id`: con la identificación de variantes exigiendo variantes coexistentes, **cuál se consulta "
            f"es una decisión y no un `mtime`** — y el `variante_id` entra en el `config_id`, "
            f"que es lo que vuelve reproducible una fila de la tabla C0–C4."
        )
    return RegistroDeÍndices.leer(almacén, raíz, existentes[0])

def manifiesto_del_registro(cfg, almacén, registro: RegistroDeÍndices) -> ManifiestoDeSplit:
    return ManifiestoDeSplit.leer(almacén, cfg["rutas.artefactos_splits"], registro.split_id)

def set_de_consultas(cfg, almacén, manifiesto, autoria: str | None) -> SetDeConsultas:
    raíz = cfg["rutas.ground_truth_consultas"]
    if autoria:
        identidad_del_set = f"{manifiesto.split_id}-{autoria}"
    else:
        autorías = SetDeConsultas.del_split(almacén, raíz, manifiesto.split_id)
        completos = []
        for i in autorías:
            try:
                candidato = SetDeConsultas.leer(almacén, raíz, i, verificar_bytes=False)
            except ErrorDeArtefacto:
                continue
            if candidato.es_usable_para_medir:
                completos.append(i)
        if not completos:
            raise ErrorDeArtefacto(
                f"no hay ningún `SetDeConsultas` **completo** para el split "
                f"{manifiesto.split_id} (existen: {list(autorías)}). Un set `plantilla` "
                f"*declara que no es usable para medir* y el pipeline base completo y andando compromete **30/30**: "
                f"correrlo daría un marcador sobre un instrumento que no está terminado."
            )
        if len(completos) > 1:
            raise ErrorDeArtefacto(
                f"hay {len(completos)} sets completos para el split {manifiesto.split_id}: "
                f"{completos}. Pasá `--autoria`: los ~540 juicios entran por `consulta_id` y "
                f"**dos sets son dos instrumentos** (la tabla de identidades, clase 3)."
            )
        identidad_del_set = completos[0]
    return SetDeConsultas.leer(almacén, raíz, identidad_del_set)

def bytes_de_la_imagen_del_set(cfg, almacén, consulta) -> bytes:
    ruta = almacén.ruta(
        SetDeConsultas.ruta_de_imagen(
            cfg["rutas.ground_truth_consultas"], consulta.artwork_id
        )
    )
    if not ruta.exists():
        raise ErrorDeArtefacto(
            f"falta la imagen versionada de {consulta.artwork_id} ({ruta}). Las 10 son "
            f"**parte del instrumento de medición** y van al repo (el set de consultas, la regla de qué se versiona)."
        )
    return ruta.read_bytes()

def bytes_del_corpus(cfg, almacén):
    raíz = Path(cfg["rutas.imagenes"])

    def leer(artwork_id: str) -> bytes:
        ruta = almacén.ruta(raíz / f"{artwork_id.replace(':', '_')}.jpg")
        if not ruta.exists():
            raise ErrorDeArtefacto(
                f"falta la imagen de {artwork_id} en el `AlmacénDeImágenes` ({ruta}). Es "
                f"**regenerable y no versionado**: corré `poetry run tp3-ingesta`. "
                f"El self-retrieval necesita los bytes porque compara contra el vector "
                f"indexado, y ese vector salió del caché cuya clave lleva su hash."
            )
        return ruta.read_bytes()

    return leer

def abrir_indice(cfg, almacén, variante_id: str):
    from tp3_rag.puertos.indice_vectorial import ÍndiceVectorialChroma

    return ÍndiceVectorialChroma(
        almacén.ruta(cfg["rutas.chroma"]),
        variante_id=variante_id,
        metrica=cfg["indice.metrica"],
        batch_de_upsert=cfg["indice.batch_de_upsert"],
    )

def montar(
    cfg,
    almacén,
    *,
    variante_id: str | None = None,
    observabilidad=None,
    credenciales=None,
    con_clip: bool = True,
    con_llm: bool = True,
    con_embeddings_de_texto: bool | None = None,
) -> Montaje:
    registro = registro_de_indices(cfg, almacén, variante_id)
    manifiesto = manifiesto_del_registro(cfg, almacén, registro)
    indice = abrir_indice(cfg, almacén, registro.variante_id)
    if observabilidad is not None:
        indice = observabilidad.instrumentar(indice, rol="indice")

    montaje = Montaje(registro=registro, manifiesto=manifiesto, indice=indice)

    if con_clip:
        from tp3_rag.puertos.embedding_imagen import EmbeddingImagenCLIP

        clip = EmbeddingImagenCLIP(
            checkpoint=cfg["embebido.checkpoint"],
            version_del_preprocesado=cfg["embebido.version_del_preprocesado"],
            threads=cfg["embebido.threads"],
        )
        montaje.puerto_de_imagen = (
            observabilidad.instrumentar(clip, rol="embedding_imagen")
            if observabilidad is not None
            else clip
        )

    necesita_texto = (
        con_embeddings_de_texto
        if con_embeddings_de_texto is not None
        else (
            cfg["evidencia.orden"] == "por_similitud_al_pedido"
            or "recuperador_textual" in tuple(cfg["consulta.etapas"])
        )
    )
    if necesita_texto:
        from tp3_rag.puertos.embedding_texto import EmbeddingTextoOpenAI

        texto = EmbeddingTextoOpenAI(
            api_key=credenciales.exigir_openai(),
            base_url=cfg["proveedor.base_url"],
            modelo=cfg["embebido_texto.modelo"],
            dim=cfg["embebido_texto.dim"],
            limite_de_tokens=cfg["embebido_texto.limite_de_tokens"],
            reintentos=cfg["http.reintentos"],
            backoff_base=cfg["http.backoff_base"],
        )
        montaje.embebedor_de_texto = (
            observabilidad.instrumentar(texto, rol="embedding_texto")
            if observabilidad is not None
            else texto
        )

    if con_llm:
        from tp3_rag.puertos.llm import LLMOpenAI

        montaje.puerto_de_llm = LLMOpenAI(
            api_key=credenciales.exigir_openai(),
            base_url=cfg["proveedor.base_url"],
            timeout_segundos=float(cfg["http.timeout_segundos"]),
            backoff_base=float(cfg["http.backoff_base"]),
        )
    return montaje
