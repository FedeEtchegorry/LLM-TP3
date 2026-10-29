from __future__ import annotations

from typing import Mapping, Sequence

from tp3_rag.dominio.fragmento import Fragmento
from tp3_rag.modulo_b.etapa import Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración, ErrorDelProyecto
from tp3_rag.puertos.indice_vectorial import AUSENTE, KNOWLEDGE

ÓRDENES: tuple[str, ...] = ("por_similitud_al_pedido", "por_seccion_declarada")

class EvidenciaVacía(ErrorDelProyecto):
    pass

def _fragmento_desde_chroma(fragment_id: str, texto: str, metadata: Mapping) -> Fragmento:
    faltantes = sorted(
        {"artwork_id", "museum", "section", "chunk_index", "source_url", "license"}
        - set(metadata)
    )
    if faltantes:
        raise ErrorDelProyecto(
            f"se rompió la evidencia copiada por valor en {fragment_id}: la metadata de `knowledge` no trae {faltantes}. "
            f"*La proyección es la **identidad** sobre `license`, `source_url`, `texto`, "
            f"`title` y `artist`*: esta pieza es la *capa intermedia* que la licencia por fragmento de punta a punta nombra, y "
            f"**un `SELECT` que se olvide de dos columnas es toda la falla**. Si el campo no "
            f"está en el índice, la variante se construyó con otro esquema."
        )
    return Fragmento(
        artwork_id=metadata["artwork_id"],
        museum=metadata["museum"],
        section=metadata["section"],
        chunk_index=int(metadata["chunk_index"]),
        texto=texto,
        source_url=metadata["source_url"],
        license=metadata["license"],
        title=_desmaterializar(metadata.get("title")),
        artist=_desmaterializar(metadata.get("artist")),
    )

def _desmaterializar(valor):
    return None if valor in (None, AUSENTE) else valor

class RecolectorDeEvidencia(Etapa):
    NOMBRE = "recolector_de_evidencia"
    PRECONDICIONES = ("top5",)
    CLAVES_DE_CONFIGURACIÓN = (
        "evidencia.n_fragmentos_por_obra",
        "evidencia.orden",
        "fragmentacion.campos_de_knowledge",
    )
    DEGRADA = True
    POLÍTICA = (
        "evidencia vacía para la obra afectada → el generador **abstiene** por esa obra "
        ". La obra sale igual, con su metadata y su link."
    )

    def __init__(
        self,
        *,
        indice,
        n_fragmentos_por_obra: int,
        orden: str,
        campos_de_knowledge: Sequence[str],
        embebedor_de_texto=None,
    ) -> None:
        if orden not in ÓRDENES:
            raise ErrorDeConfiguración(
                f"`evidencia.orden = {orden!r}` no es uno de {list(ÓRDENES)}."
            )
        if orden == "por_similitud_al_pedido" and embebedor_de_texto is None:
            raise ErrorDeConfiguración(
                "`evidencia.orden = \"por_similitud_al_pedido\"` necesita el "
                "`PuertoEmbeddingTexto` y no se le pasó ninguno. **No cae al otro orden en "
                "silencio**: los dos eligen *cuáles* fragmentos se citan, así que una "
                "corrida que creyó ordenar por similitud y ordenó por sección publicaría "
                "una evidencia distinta bajo el mismo `config_id`. El otro orden es "
                "legítimo y se pide: `--set evidencia.orden=por_seccion_declarada`."
            )
        self.indice = indice
        self.n = int(n_fragmentos_por_obra)
        if self.n < 1:
            raise ErrorDeConfiguración(
                f"`evidencia.n_fragmentos_por_obra = {n_fragmentos_por_obra}` tiene que ser "
                f"≥ 1: el criterio de trazabilidad compromete **≥ 1 fragmento disponible para citar** por ítem."
            )
        self.orden = orden
        self.campos_de_knowledge = tuple(campos_de_knowledge)
        self.embebedor_de_texto = embebedor_de_texto

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        obras = tuple(estado.top5 or ())
        vector = None
        if self.orden == "por_similitud_al_pedido":
            vectores = self.embebedor_de_texto.embeber(
                [estado.consulta_conceptual],
                etiquetas=[f"evidencia:{estado.entrada.consulta_id}"],
            )
            vector = vectores[0]

        evidencia: dict[str, tuple[Fragmento, ...]] = {}
        vacías: list[str] = []
        for artwork_id in obras:
            fragmentos = self._de_una_obra(artwork_id, vector)
            evidencia[artwork_id] = fragmentos
            if not fragmentos:
                vacías.append(artwork_id)

        nuevo = estado.con(evidencia=evidencia)
        for artwork_id in vacías:
            nuevo = nuevo.degradar(
                self.NOMBRE,
                causa=(
                    f"la evidencia ordenada por similitud: {artwork_id} volvió con **0 fragmentos** de `knowledge`. Con "
                    f"al menos un fragmento por obra indexada (invariante 3 de los invariantes de la registración) debería ser imposible: es el síntoma de "
                    f"una variante mal construida, y es una violación del criterio de trazabilidad (*≥ 1 "
                    f"fragmento disponible para citar*)."
                ),
                politica="el generador abstiene por esa obra",
            )
        return nuevo

    def _de_una_obra(self, artwork_id: str, vector) -> tuple[Fragmento, ...]:
        where = {"artwork_id": artwork_id}
        if vector is not None:
            resultados = self.indice.consultar(KNOWLEDGE, vector, self.n, where)
            return tuple(
                _fragmento_desde_chroma(r.id, r.documento or "", r.metadata)
                for r in resultados
            )
        todos = self.indice.traer(KNOWLEDGE, where=where)
        orden = {s: i for i, s in enumerate(self.campos_de_knowledge)}
        ordenados = sorted(
            todos,
            key=lambda r: (
                orden.get(r.metadata.get("section"), len(orden)),
                int(r.metadata.get("chunk_index", 0)),
                r.id,
            ),
        )
        return tuple(
            _fragmento_desde_chroma(r.id, r.documento or "", r.metadata)
            for r in ordenados[: self.n]
        )

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "n_fragmentos_por_obra": self.n,
            "orden": self.orden,
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
            "nota": (
                "la evidencia copiada por valor es la **identidad** sobre `license`, `source_url`, `texto`, `title` y "
                "`artist`: esta pieza devuelve `Fragmento`s completos y no dicts "
                "proyectados, así que *no hay proyección* que pueda olvidar una columna. Y "
                "`media_license` **no está en el tipo**, así que el renderizador no "
                "puede leerlo mal."
            ),
        }
