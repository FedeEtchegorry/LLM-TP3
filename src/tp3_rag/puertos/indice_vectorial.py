from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.nucleo.errores import ErrorDeConstrucción

ARTWORKS = "artworks"
KNOWLEDGE = "knowledge"
COLECCIONES = (ARTWORKS, KNOWLEDGE)

AUSENTE = ""
SEPARADOR_DE_LISTA = " | "

class ColecciónAusente(ErrorDeConstrucción):
    def __init__(self, nombre: str, variante_id: str, causa: str) -> None:
        super().__init__(
            f"la colección {nombre!r} de la variante {variante_id} {causa}. Un ranking "
            f"vacío se degradaría elegantemente y nadie se enteraría de que el índice no "
            f"estaba, así que esto es error duro (el contrato de los puertos, decisión 2). Construí la variante "
            f"con `poetry run tp3-indexar`."
        )
        self.nombre = nombre
        self.variante_id = variante_id

def aplanar(valor: Any) -> str | int | float | bool:
    if valor is None:
        return AUSENTE
    if isinstance(valor, bool) or isinstance(valor, (int, float)):
        return valor
    if isinstance(valor, str):
        return valor
    if isinstance(valor, (list, tuple, set)):
        partes = [str(v).strip() for v in valor if v is not None and str(v).strip()]
        return SEPARADOR_DE_LISTA.join(partes) if partes else AUSENTE
    return str(valor)

def aplanar_metadata(metadata: Mapping[str, Any]) -> dict:
    return {clave: aplanar(valor) for clave, valor in metadata.items()}

@dataclass(frozen=True)
class Resultado:
    id: str
    score: float | None
    metadata: dict
    documento: str | None = None

class _BaseDeÍndice:
    OPERACIONES = (
        "crear_coleccion",
        "upsert",
        "consultar",
        "traer",
        "metadatas",
        "contar",
    )

    def __init__(self, *, variante_id: str) -> None:
        if not variante_id:
            raise ErrorDeConstrucción(
                "un índice sin `variante_id` no se puede construir: el contrato de los puertos pide que no "
                "haya **ninguna** consulta sin `variante_id` explícito, y la forma de "
                "garantizarlo es que el adaptador no exista sin él."
            )
        self.variante_id = variante_id

    def _validar_nombre(self, nombre: str) -> str:
        if nombre not in COLECCIONES:
            raise ErrorDeConstrucción(
                f"colección {nombre!r} desconocida; las del alcance aprobado son "
                f"{list(COLECCIONES)}."
            )
        return nombre

class ÍndiceVectorialChroma(_BaseDeÍndice):
    def __init__(
        self,
        raíz: str | Path,
        *,
        variante_id: str,
        metrica: str = "cosine",
        batch_de_upsert: int = 5000,
    ) -> None:
        super().__init__(variante_id=variante_id)
        self.raíz = Path(raíz) / variante_id
        self.metrica = metrica
        self.batch_de_upsert = int(batch_de_upsert)
        self._cliente = None
        self._colecciones: dict[str, Any] = {}

    def _cli(self):
        if self._cliente is None:
            import chromadb

            self.raíz.mkdir(parents=True, exist_ok=True)
            self._cliente = chromadb.PersistentClient(path=str(self.raíz))
        return self._cliente

    def crear_coleccion(self, nombre: str, variante_id: str):
        self._validar_nombre(nombre)
        if variante_id != self.variante_id:
            raise ErrorDeConstrucción(
                f"se pidió crear {nombre!r} con `variante_id = {variante_id}` sobre un "
                f"índice abierto en {self.variante_id}. La identidad va en el path "
                f" y no puede discrepar del contenido."
            )
        coleccion = self._cli().get_or_create_collection(
            name=nombre,
            configuration={"hnsw": {"space": self.metrica}},
            metadata={"variante_id": variante_id},
            embedding_function=None,
        )
        self._verificar_identidad(nombre, coleccion)
        self._colecciones[nombre] = coleccion
        return coleccion

    def _abrir(self, nombre: str, *, exigir_no_vacia: bool = True):
        self._validar_nombre(nombre)
        if nombre not in self._colecciones:
            try:
                self._colecciones[nombre] = self._cli().get_collection(
                    nombre, embedding_function=None
                )
            except Exception:
                raise ColecciónAusente(nombre, self.variante_id, "no existe") from None
            self._verificar_identidad(nombre, self._colecciones[nombre])
        coleccion = self._colecciones[nombre]
        if exigir_no_vacia and coleccion.count() == 0:
            raise ColecciónAusente(nombre, self.variante_id, "está vacía")
        return coleccion

    def _verificar_identidad(self, nombre: str, coleccion) -> None:
        declarado = (coleccion.metadata or {}).get("variante_id")
        if declarado and declarado != self.variante_id:
            raise ErrorDeConstrucción(
                f"la colección {nombre!r} en {self.raíz} declara adentro "
                f"`variante_id = {declarado}` y el path dice {self.variante_id}. El "
                f"artefacto no valida y se rechaza (el ciclo de vida de los artefactos, punto 5)."
            )

    def upsert(
        self,
        nombre: str,
        *,
        ids: Sequence[str],
        vectores: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        documentos: Sequence[str] | None = None,
    ) -> int:
        if not (len(ids) == len(vectores) == len(metadatas)):
            raise ErrorDeConstrucción(
                f"upsert desalineado en {nombre!r}: {len(ids)} ids, {len(vectores)} "
                f"vectores, {len(metadatas)} metadatas."
            )
        coleccion = self._colecciones.get(nombre) or self.crear_coleccion(
            nombre, self.variante_id
        )
        planas = [aplanar_metadata(m) or None for m in metadatas]
        for desde in range(0, len(ids), self.batch_de_upsert):
            hasta = desde + self.batch_de_upsert
            coleccion.upsert(
                ids=list(ids[desde:hasta]),
                embeddings=[list(map(float, v)) for v in vectores[desde:hasta]],
                metadatas=planas[desde:hasta],
                documents=list(documentos[desde:hasta]) if documentos is not None else None,
            )
        return len(ids)

    def consultar(
        self,
        nombre: str,
        vector: Sequence[float],
        n: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[Resultado]:
        coleccion = self._abrir(nombre)
        crudo = coleccion.query(
            query_embeddings=[list(map(float, vector))],
            n_results=max(1, int(n)),
            where=dict(where) if where else None,
            include=["metadatas", "distances", "documents"],
        )
        ids = (crudo.get("ids") or [[]])[0]
        distancias = (crudo.get("distances") or [[]])[0]
        metadatas = (crudo.get("metadatas") or [[]])[0]
        documentos = (crudo.get("documents") or [[None] * len(ids)])[0]
        return [
            Resultado(id=i, score=1.0 - float(d), metadata=dict(m or {}), documento=doc)
            for i, d, m, doc in zip(ids, distancias, metadatas, documentos)
        ]

    def traer(self, nombre: str, *, where: Mapping[str, Any], limite: int | None = None):
        coleccion = self._abrir(nombre)
        crudo = coleccion.get(
            where=dict(where), limit=limite, include=["metadatas", "documents"]
        )
        ids = crudo.get("ids") or []
        metadatas = crudo.get("metadatas") or [{}] * len(ids)
        documentos = crudo.get("documents") or [None] * len(ids)
        return [
            Resultado(id=i, score=None, metadata=dict(m or {}), documento=doc)
            for i, m, doc in zip(ids, metadatas, documentos)
        ]

    def metadatas(self, nombre: str, ids: Sequence[str]) -> dict[str, dict]:
        if not ids:
            return {}
        coleccion = self._abrir(nombre, exigir_no_vacia=False)
        crudo = coleccion.get(ids=list(ids), include=["metadatas"])
        return {
            i: dict(m or {})
            for i, m in zip(crudo.get("ids") or [], crudo.get("metadatas") or [])
        }

    def contar(self, nombre: str) -> int:
        return self._abrir(nombre, exigir_no_vacia=False).count()

    def existe(self, nombre: str) -> bool:
        try:
            self._abrir(nombre, exigir_no_vacia=False)
            return True
        except ColecciónAusente:
            return False

    def ids(self, nombre: str) -> tuple[str, ...]:
        coleccion = self._abrir(nombre, exigir_no_vacia=False)
        return tuple(coleccion.get(include=[])["ids"])

    def cerrar(self) -> None:
        if self._cliente is not None:
            self._colecciones.clear()
            self._cliente = None

class ÍndiceVectorialEnMemoria(_BaseDeÍndice):
    def __init__(self, *, variante_id: str, metrica: str = "cosine") -> None:
        super().__init__(variante_id=variante_id)
        self.metrica = metrica
        self._datos: dict[str, dict[str, tuple]] = {}
        self.n_consultas = 0

    def crear_coleccion(self, nombre: str, variante_id: str):
        self._validar_nombre(nombre)
        if variante_id != self.variante_id:
            raise ErrorDeConstrucción(
                f"se pidió crear {nombre!r} con `variante_id = {variante_id}` sobre un "
                f"índice abierto en {self.variante_id}."
            )
        self._datos.setdefault(nombre, {})
        return nombre

    def upsert(
        self,
        nombre: str,
        *,
        ids: Sequence[str],
        vectores: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
        documentos: Sequence[str] | None = None,
    ) -> int:
        if not (len(ids) == len(vectores) == len(metadatas)):
            raise ErrorDeConstrucción(
                f"upsert desalineado en {nombre!r}: {len(ids)} ids, {len(vectores)} "
                f"vectores, {len(metadatas)} metadatas."
            )
        self.crear_coleccion(nombre, self.variante_id)
        for i, (clave, vector, metadata) in enumerate(zip(ids, vectores, metadatas)):
            self._datos[nombre][clave] = (
                tuple(float(x) for x in vector),
                aplanar_metadata(metadata),
                documentos[i] if documentos is not None else None,
            )
        return len(ids)

    def consultar(
        self,
        nombre: str,
        vector: Sequence[float],
        n: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[Resultado]:
        self.n_consultas += 1
        contenido = self._abrir(nombre)
        candidatos = [
            Resultado(clave, _coseno(vector, v), dict(m), doc)
            for clave, (v, m, doc) in contenido.items()
            if _cumple(m, where)
        ]
        candidatos.sort(key=lambda r: (-(r.score or 0.0), r.id))
        return candidatos[: max(1, int(n))]

    def traer(self, nombre: str, *, where: Mapping[str, Any], limite: int | None = None):
        self.n_consultas += 1
        contenido = self._abrir(nombre)
        salida = [
            Resultado(clave, None, dict(m), doc)
            for clave, (_, m, doc) in sorted(contenido.items())
            if _cumple(m, where)
        ]
        return salida[:limite] if limite else salida

    def metadatas(self, nombre: str, ids: Sequence[str]) -> dict[str, dict]:
        self._validar_nombre(nombre)
        contenido = self._datos.get(nombre) or {}
        return {i: dict(contenido[i][1]) for i in ids if i in contenido}

    def contar(self, nombre: str) -> int:
        self._validar_nombre(nombre)
        if nombre not in self._datos:
            raise ColecciónAusente(nombre, self.variante_id, "no existe")
        return len(self._datos[nombre])

    def existe(self, nombre: str) -> bool:
        self._validar_nombre(nombre)
        return nombre in self._datos

    def ids(self, nombre: str) -> tuple[str, ...]:
        self._validar_nombre(nombre)
        return tuple(sorted(self._datos.get(nombre) or {}))

    def documento(self, nombre: str, clave: str) -> str | None:
        return (self._datos.get(nombre) or {}).get(clave, (None, None, None))[2]

    def cerrar(self) -> None:
        pass

    def _abrir(self, nombre: str) -> dict:
        self._validar_nombre(nombre)
        if nombre not in self._datos:
            raise ColecciónAusente(nombre, self.variante_id, "no existe")
        if not self._datos[nombre]:
            raise ColecciónAusente(nombre, self.variante_id, "está vacía")
        return self._datos[nombre]

def _coseno(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ErrorDeConstrucción(
            f"se consultó con un vector de {len(a)} dims contra uno de {len(b)}. Es el "
            f"síntoma de mezclar los dos espacios: CLIP da 512 y "
            f"`text-embedding-3-small` da 1.536, y el stack declarado los separa en dos "
            f"puertos justamente para que esto no se pueda hacer sin darse cuenta."
        )
    punto = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return punto / (na * nb)

def _cumple(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    if not where:
        return True
    for clave, condición in where.items():
        if clave == "$and":
            if not all(_cumple(metadata, c) for c in condición):
                return False
            continue
        if clave == "$or":
            if not any(_cumple(metadata, c) for c in condición):
                return False
            continue
        valor = metadata.get(clave, AUSENTE)
        if not isinstance(condición, Mapping):
            if valor != condición:
                return False
            continue
        for operador, esperado in condición.items():
            if operador == "$eq" and valor != esperado:
                return False
            elif operador == "$ne" and valor == esperado:
                return False
            elif operador == "$in" and valor not in esperado:
                return False
            elif operador == "$nin" and valor in esperado:
                return False
            elif operador not in ("$eq", "$ne", "$in", "$nin"):
                raise ErrorDeConstrucción(
                    f"el índice en memoria no soporta el operador {operador!r}. Se "
                    f"levanta en vez de ignorarlo: un `where` que no filtra haría pasar "
                    f"un test de la evidencia registrada en la corrida contra un índice que devolvió todo."
                )
    return True
