from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeArtefacto

VERSIÓN_DEL_SET = "1"

CATEGORÍAS: tuple[str, ...] = ("visual_cromatico", "conceptual_cultural", "ambiguo")

QUÉ_ESTRESA = {
    "visual_cromatico": "la rama visual y el valor de α",
    "conceptual_cultural": "la rama de texto (la similitud visual no lo resuelve)",
    "ambiguo": "el pre-retrieval con LLM (hay que interpretar)",
}

CATEGORÍA_CULTURAL = "conceptual_cultural"

ESTADOS: tuple[str, ...] = ("plantilla", "completo")

N_IMAGENES = 10
N_PEDIDOS_POR_IMAGEN = 3
N_CONSULTAS = N_IMAGENES * N_PEDIDOS_POR_IMAGEN

CLAVES_OBLIGATORIAS = (
    "identidad",
    "split_id",
    "autoria",
    "estado",
    "criterio_de_culture",
    "revision",
    "imagenes",
    "conteos",
)

@dataclass(frozen=True)
class CulturaPedida:
    nombre: str
    etiquetas: tuple[str, ...]
    n: int
    n_por_etiqueta: Mapping[str, int]
    poblacion: str
    n_poblacion: int
    criterio_de_culture: str

    def a_json(self) -> dict:
        return {
            "nombre": self.nombre,
            "etiquetas": list(self.etiquetas),
            "n": self.n,
            "n_por_etiqueta": dict(self.n_por_etiqueta),
            "poblacion": self.poblacion,
            "n_poblacion": self.n_poblacion,
            "criterio_de_culture": self.criterio_de_culture,
            "nota": (
                ": `cultura_pedida` es un **conjunto declarado de etiquetas** del "
                "vocabulario normalizado. `n` es el tamaño de la **unión de obras** y no "
                "la suma de `n_por_etiqueta`: una obra con dos etiquetas del grupo cuenta "
                "una vez. La población va al lado del número a propósito "
                "(los hallazgos, modo de falla 3)."
            ),
        }

    @classmethod
    def desde_json(cls, d: Mapping[str, Any]) -> "CulturaPedida":
        return cls(
            nombre=d["nombre"],
            etiquetas=tuple(d["etiquetas"]),
            n=d["n"],
            n_por_etiqueta=dict(d.get("n_por_etiqueta") or {}),
            poblacion=d["poblacion"],
            n_poblacion=d["n_poblacion"],
            criterio_de_culture=d["criterio_de_culture"],
        )

@dataclass(frozen=True)
class ConsultaDelSet:
    consulta_id: str
    categoria: str
    pedido: str
    artwork_id: str
    archivo_de_imagen: str
    imagen_hash: str
    cultura_pedida: CulturaPedida | None = None
    es_ejemplo: bool = False

    @property
    def es_cultural(self) -> bool:
        return self.categoria == CATEGORÍA_CULTURAL

@dataclass(frozen=True)
class ImagenDeConsulta:
    artwork_id: str
    imagen_hash: str
    archivo: str
    atributos: tuple[str, ...]
    consultas: tuple[ConsultaDelSet, ...]
    nota: str | None = None
    es_ejemplo: bool = False
    procedencia: Mapping[str, Any] | None = None

@dataclass(frozen=True)
class SetDeConsultas:
    contenido: dict

    @property
    def identidad(self) -> str:
        return self.contenido["identidad"]

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
    def es_usable_para_medir(self) -> bool:
        return self.estado == "completo"

    @property
    def criterio_de_culture(self) -> str:
        return self.contenido["criterio_de_culture"]

    @property
    def n_imagenes_revisadas(self) -> int:
        return self.contenido["revision"]["n_imagenes_revisadas"]

    def a_json(self) -> dict:
        return self.contenido

    def imagenes(self) -> tuple[ImagenDeConsulta, ...]:
        return tuple(self._imagen(d) for d in self.contenido["imagenes"])

    def consultas(self) -> tuple[ConsultaDelSet, ...]:
        return tuple(c for img in self.imagenes() for c in img.consultas)

    def consulta(self, consulta_id: str) -> ConsultaDelSet:
        for c in self.consultas():
            if c.consulta_id == consulta_id:
                return c
        raise ErrorDeArtefacto(
            f"{self.identidad}: no hay ninguna consulta con `consulta_id = "
            f"{consulta_id!r}`. Una referencia que no resuelve es /, y el "
            f"detector comprometido es el `%` de ítems sin juzgar de la tabla de ablación."
        )

    def culturas_pedidas(self) -> tuple[CulturaPedida, ...]:
        return tuple(c.cultura_pedida for c in self.consultas() if c.cultura_pedida)

    def por_categoria(self) -> dict[str, int]:
        conteo = {cat: 0 for cat in CATEGORÍAS}
        for c in self.consultas():
            conteo[c.categoria] = conteo.get(c.categoria, 0) + 1
        return conteo

    def _imagen(self, d: Mapping[str, Any]) -> ImagenDeConsulta:
        consultas = tuple(
            ConsultaDelSet(
                consulta_id=c["consulta_id"],
                categoria=c["categoria"],
                pedido=c["pedido"],
                artwork_id=d["artwork_id"],
                archivo_de_imagen=d["archivo"],
                imagen_hash=d["imagen_hash"],
                cultura_pedida=(
                    CulturaPedida.desde_json(c["cultura_pedida"])
                    if c.get("cultura_pedida")
                    else None
                ),
                es_ejemplo=bool(c.get("es_ejemplo", False)),
            )
            for c in d.get("consultas", ())
        )
        return ImagenDeConsulta(
            artwork_id=d["artwork_id"],
            imagen_hash=d["imagen_hash"],
            archivo=d["archivo"],
            atributos=tuple(d.get("atributos", ())),
            consultas=consultas,
            nota=d.get("nota"),
            es_ejemplo=bool(d.get("es_ejemplo", False)),
            procedencia=d.get("procedencia"),
        )

    @staticmethod
    def ruta_de(raíz: str | Path, identidad_del_set: str) -> Path:
        return Path(raíz) / f"{identidad_del_set}.json"

    @staticmethod
    def ruta_de_imagen(raíz: str | Path, artwork_id: str) -> Path:
        return Path(raíz) / "imagenes" / f"{artwork_id.replace(':', '_')}.jpg"

    @classmethod
    def existentes(cls, almacén, raíz: str | Path) -> tuple[str, ...]:
        carpeta = almacén.ruta(raíz)
        if not carpeta.exists():
            return ()
        return tuple(sorted(p.stem for p in carpeta.glob("*.json")))

    @classmethod
    def del_split(cls, almacén, raíz: str | Path, split_id: str) -> tuple[str, ...]:
        return tuple(
            i for i in cls.existentes(almacén, raíz)
            if identidad.partir_identidad_del_set(i)[0] == split_id
        )

    @classmethod
    def leer(
        cls, almacén, raíz: str | Path, identidad_del_set: str, *, verificar_bytes: bool = True
    ) -> "SetDeConsultas":
        contenido = almacén.leer_json(
            cls.ruta_de(raíz, identidad_del_set), claves_obligatorias=CLAVES_OBLIGATORIAS
        )
        set_ = cls(contenido)
        set_.validar()
        if set_.identidad != identidad_del_set:
            raise ErrorDeArtefacto(
                f"el set de {identidad_del_set}.json declara adentro "
                f"`identidad = {set_.identidad}`. El nombre y el contenido no pueden "
                f"discrepar: la identidad va en el nombre."
            )
        if verificar_bytes:
            set_.verificar_bytes(almacén, raíz)
        return set_

    def validar(self) -> None:
        self._validar_identidad()
        self._validar_estado()
        self._validar_imagenes()
        self._validar_consultas()
        self._validar_revision()
        self._validar_conteos()

    def _validar_identidad(self) -> None:
        esperada = identidad.identidad_del_set_de_consultas(
            split_id=self.split_id, autoria=self.autoria
        )
        if self.identidad != esperada:
            raise ErrorDeArtefacto(
                f"`identidad` dice {self.identidad!r} y sus dos ingredientes dan "
                f"{esperada!r}. La identidad de un artefacto de clase 3 **es** el "
                f"compromiso: si no coincide con sus ingredientes, no se sabe a "
                f"qué split ancla ni qué autoría la produjo."
            )

    def _validar_estado(self) -> None:
        if self.estado not in ESTADOS:
            raise ErrorDeArtefacto(
                f"{self.identidad}: `estado` = {self.estado!r} no es uno de {list(ESTADOS)}."
            )

    def _validar_imagenes(self) -> None:
        imágenes = self.imagenes()
        if not imágenes:
            raise ErrorDeArtefacto(f"{self.identidad}: el set no tiene ni una imagen.")

        ids = [i.artwork_id for i in imágenes]
        if len(set(ids)) != len(ids):
            raise ErrorDeArtefacto(
                f"{self.identidad}: hay `artwork_id` repetidos entre las imágenes de "
                f"consulta. el diseño de la evaluación pide **10 imágenes** para cubrir variedad; dos "
                f"veces la misma obra no cubre variedad y además duplicaría sus juicios."
            )
        for img in imágenes:
            if not img.atributos:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {img.artwork_id} no declara `atributos`. Son "
                    f"el diseño y lo único que vuelve auditable el *'elegidas "
                    f"para cubrir variedad'* del diseño de la evaluación: sin ellos el set no dice por "
                    f"qué se eligió esta obra y no otra."
                )
            if not str(img.imagen_hash).startswith("sha256:"):
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {img.artwork_id} declara "
                    f"`imagen_hash = {img.imagen_hash!r}`, sin prefijo de algoritmo. "
                    f"El prefijo es de `identidad.hash_de_bytes` y existe para que un "
                    f"cambio de algoritmo lo diga en vez de colisionar."
                )
            if len(img.consultas) != N_PEDIDOS_POR_IMAGEN:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {img.artwork_id} tiene {len(img.consultas)} "
                    f"pedidos y tienen que ser {N_PEDIDOS_POR_IMAGEN}. *'Cada imagen "
                    f"recibe tres pedidos de categorías distintas'*, y que "
                    f"las tres categorías estén **balanceadas es deliberado**: es lo que "
                    f"permite el desglose de la tabla de ablación y el delta por categoría del delta por categoría de pedido."
                )
            categorías = [c.categoria for c in img.consultas]
            if sorted(categorías) != sorted(CATEGORÍAS):
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {img.artwork_id} tiene las categorías "
                    f"{categorías} y tiene que tener **una de cada**: {list(CATEGORÍAS)}. "
                    f"Dos pedidos de la misma categoría sobre la misma imagen rompen el "
                    f"balance que la tabla de ablación y el delta por categoría de pedido desglosan."
                )

    def _validar_consultas(self) -> None:
        consultas = self.consultas()
        ids = [c.consulta_id for c in consultas]
        repetidos = sorted({i for i in ids if ids.count(i) > 1})
        if repetidos:
            raise ErrorDeArtefacto(
                f"{self.identidad}: los `consulta_id` {repetidos} están repetidos. Cada "
                f"juicio entra por un `consulta_id`: dos consultas con el mismo "
                f"id mezclan los juicios de dos pedidos distintos y el `juicio` no tiene "
                f"con qué distinguirlos."
            )
        for c in consultas:
            if not c.consulta_id or not str(c.consulta_id).strip():
                raise ErrorDeArtefacto(
                    f"{self.identidad}: hay una consulta de {c.artwork_id} sin "
                    f"`consulta_id`. Los 30 **se declaran**: no hay default."
                )
            if identidad.parece_hash_corto(c.consulta_id):
                raise ErrorDeArtefacto(
                    f"{self.identidad}: `consulta_id = {c.consulta_id!r}` tiene forma de "
                    f"`hash_corto`. **Los 30 del set se declaran, no se derivan de los "
                    f"bytes** — es la puerta que cerró (el set de consultas, consecuencia 1): un "
                    f"id derivado de bytes se mueve cuando el museo reemplaza un JPEG, y "
                    f"con él se huerfanan los ~540 juicios. Usá un id **legible y "
                    f"citable** (p. ej. `c01-visual`), por la misma razón por la que el "
                    f"ordinal de la clase 2 no es un timestamp."
                )
            if not str(c.pedido).strip():
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {c.consulta_id} no tiene `pedido`. El pedido es "
                    f"la mitad de la consulta (`Consulta` = imagen + pedido, el plan de consulta)."
                )
            if c.categoria not in CATEGORÍAS:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {c.consulta_id} declara "
                    f"`categoria = {c.categoria!r}`, que no es una de {list(CATEGORÍAS)}."
                )
            self._validar_cultura_pedida(c)

    def _validar_cultura_pedida(self, c: ConsultaDelSet) -> None:
        if c.cultura_pedida is None:
            if c.es_cultural:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: {c.consulta_id} es de categoría "
                    f"`{CATEGORÍA_CULTURAL}` y no declara `cultura_pedida`. **la cobertura cultural del resultado** "
                    f"compromete publicar, para cada pedido cultural, el `n` de obras del "
                    f"corpus que pertenecen a la cultura pedida: sin las etiquetas no hay "
                    f"`n`, y el criterio de cobertura publicaría dos de las tres cosas que manda publicar "
                    f"juntas."
                )
            return
        if not c.es_cultural:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} es de categoría {c.categoria!r} y "
                f"declara `cultura_pedida`. Solo la categoría `{CATEGORÍA_CULTURAL}` la "
                f"lleva: un pedido cromático con cultura pedida haría que **la cobertura cultural del resultado** "
                f"publique un `n` al lado de un resultado que no lo usa, y el desglose de "
                f"la tabla de ablación dejaría de separar lo que separa."
            )
        cp = c.cultura_pedida
        if not cp.etiquetas:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} declara una `cultura_pedida` sin "
                f"etiquetas.: es un *conjunto declarado de etiquetas* del "
                f"vocabulario normalizado, y un conjunto vacío daría el `n = 0` que la "
                f"decisión existe para no producir."
            )
        if len(set(cp.etiquetas)) != len(cp.etiquetas):
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} repite etiquetas en "
                f"`cultura_pedida.etiquetas` ({list(cp.etiquetas)}). Es un **conjunto**: "
                f"una etiqueta dos veces no cambia la unión y sugiere que el agrupamiento "
                f"se armó sin mirarlo."
            )
        if cp.criterio_de_culture != self.criterio_de_culture:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} dice que su `cultura_pedida` se "
                f"computó con {cp.criterio_de_culture!r} y el set declara "
                f"{self.criterio_de_culture!r}. es sobre *un* vocabulario: dos "
                f"criterios dentro del mismo set volverían incomparables dos `n` que el criterio de cobertura "
                f"publica juntos."
            )
        if not isinstance(cp.n, int) or cp.n < 1:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} declara `cultura_pedida.n = {cp.n!r}`. "
                f"**Tiene que ser ≥ 1**: una etiqueta que no existe en el corpus es error "
                f"al crear el set, no un `n = 0`. Un cero publicado por la cobertura cultural del resultado se "
                f"lee como falla de cobertura y es un desajuste de vocabulario — que es "
                f"la falla que el criterio de cobertura existe para distinguir."
            )
        if cp.n > cp.n_poblacion:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} declara `n = {cp.n}` sobre una "
                f"población de {cp.n_poblacion}. Un `n` mayor que su población no es un "
                f"número, es dos poblaciones mezcladas — el **modo de falla 3** del "
                f"proyecto."
            )
        if sorted(cp.n_por_etiqueta) != sorted(cp.etiquetas):
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} declara `n_por_etiqueta` para "
                f"{sorted(cp.n_por_etiqueta)} y etiquetas {sorted(cp.etiquetas)}. El "
                f"desglose es lo que vuelve **auditable el agrupamiento**, que es el "
                f"insumo del criterio de normalización de culture: tiene que cubrir exactamente las etiquetas "
                f"declaradas."
            )
        suma = sum(cp.n_por_etiqueta.values())
        mayor = max(cp.n_por_etiqueta.values())
        if not mayor <= cp.n <= suma:
            raise ErrorDeArtefacto(
                f"{self.identidad}: {c.consulta_id} declara `n = {cp.n}` con un desglose "
                f"que suma {suma} y cuyo máximo es {mayor}. El `n` es el tamaño de la "
                f"**unión de obras**, así que tiene que estar entre el máximo (todas las "
                f"etiquetas sobre las mismas obras) y la suma (sin solapamiento)."
            )

    def _validar_revision(self) -> None:
        revisadas = self.contenido["revision"].get("n_imagenes_revisadas")
        if not isinstance(revisadas, int) or revisadas < 1:
            raise ErrorDeArtefacto(
                f"{self.identidad}: `revision.n_imagenes_revisadas` = {revisadas!r}. El "
                f"artefacto **registra cuántas imágenes se miraron** para elegir las 10 "
                f"(el set de consultas, punto 2): es lo que vuelve auditable el *'elegidas para cubrir "
                f"variedad'* del diseño de la evaluación. Si se revisaron 30 de 289, eso se dice."
            )
        if revisadas < len(self.imagenes()):
            raise ErrorDeArtefacto(
                f"{self.identidad}: dice que se revisaron {revisadas} imágenes y el set "
                f"tiene {len(self.imagenes())}. No se pueden elegir más de las que se "
                f"miraron."
            )

    def _validar_conteos(self) -> None:
        imágenes = self.imagenes()
        consultas = self.consultas()
        c = self.contenido["conteos"]
        esperados = {
            "n_imagenes": len(imágenes),
            "n_consultas": len(consultas),
            "n_pedidos_culturales": sum(1 for x in consultas if x.cultura_pedida),
        }
        for clave, esperado in esperados.items():
            if c.get(clave) != esperado:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: `conteos.{clave}` dice {c.get(clave)!r} y el "
                    f"contenido dice {esperado}. Un conteo que no cierra contra su propio "
                    f"contenido no se completa, se rechaza (el ciclo de vida de los artefactos, punto 5)."
                )
        if c.get("por_categoria") != self.por_categoria():
            raise ErrorDeArtefacto(
                f"{self.identidad}: `conteos.por_categoria` dice {c.get('por_categoria')} "
                f"y el contenido dice {self.por_categoria()}. Es el conteo que la tabla de ablación "
                f"desglosa."
            )

        ejemplos = [x for x in consultas if x.es_ejemplo]
        if self.estado == "completo":
            if len(imágenes) != N_IMAGENES or len(consultas) != N_CONSULTAS:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: `estado = completo` con {len(imágenes)} imágenes y "
                    f"{len(consultas)} consultas. **30 consultas = 10 imágenes × 3 "
                    f"pedidos**, y es lo que el pipeline base completo y andando mide como *30/30*. Un set "
                    f"a medio llenar se declara `plantilla`: pasar de uno al otro es "
                    f"**autorar otro** con otra `autoría`, que es lo único que la clase 3 "
                    f"permite."
                )
            if ejemplos:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: `estado = completo` y hay {len(ejemplos)} "
                    f"consultas marcadas `es_ejemplo`. Un ejemplo **no es dato**: si "
                    f"entrara al pool, gastaría anotación humana sobre un pedido que "
                    f"nadie escribió para medir."
                )
        else:
            no_marcadas = [x.consulta_id for x in consultas if not x.es_ejemplo]
            if no_marcadas:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: `estado = plantilla` y {no_marcadas} no están "
                    f"marcadas `es_ejemplo = true`. En una plantilla **todo** es ejemplo: "
                    f"es lo que impide que la forma se confunda con el dato."
                )

    def verificar_bytes(self, almacén, raíz: str | Path) -> None:
        from tp3_rag.nucleo.identidad import hash_de_bytes

        for img in self.imagenes():
            ruta = almacén.ruta(self.ruta_de_imagen(raíz, img.artwork_id))
            if not ruta.exists():
                raise ErrorDeArtefacto(
                    f"{self.identidad}: falta la imagen versionada de {img.artwork_id} "
                    f"({ruta}). Las 10 imágenes **son parte del instrumento de medición** "
                    f"y van al repo (el set de consultas, la regla de qué se versiona): el `AlmacénDeImágenes` es `data/`, "
                    f"regenerable y no versionado, así que no sirve de fuente de verdad."
                )
            medido = hash_de_bytes(ruta.read_bytes())
            if medido != img.imagen_hash:
                raise ErrorDeArtefacto(
                    f"{self.identidad}: los bytes de {img.artwork_id} no son los "
                    f"declarados (declarado {img.imagen_hash[:23]}…, medido "
                    f"{medido[:23]}…). Es **exactamente** lo que el `imagen_hash` adentro "
                    f"existe para detectar: con otros bytes, los `consulta_id` de esta "
                    f"imagen dejaron de describir lo que se juzgó. **No se re-hashea: se "
                    f"restaura el archivo** o se autora otro set."
                )

    def filas_para_mostrar(self) -> Iterator[tuple[str, ...]]:
        for img in self.imagenes():
            for c in img.consultas:
                cultura = (
                    f"{c.cultura_pedida.nombre} (n={c.cultura_pedida.n})"
                    if c.cultura_pedida
                    else "—"
                )
                yield (c.consulta_id, img.artwork_id, c.categoria, c.pedido, cultura)

def nota_de_identidad() -> str:
    return (
        "Clase 3: ancla de trabajo humano. La identidad es `split_id` + `autoría`, "
        "y la `autoría` **la asigna una persona**: es lo que vuelve alcanzable el costo "
        "que el set de consultas declara —si cambia el criterio de normalización de culture y hay que **re-autorar** las "
        "`cultura_pedida`, mismo split + otra autoría = **otro artefacto**, y ni los 30 "
        "`consulta_id` ni los ~540 juicios se mueven—. El criterio de `culture` y el "
        "`variante_id` **no** son ingredientes: el primero va adentro (si entrara, cambiar "
        "cambiar el criterio de normalización renombraría el set) y la coherencia del par (set, variante) se chequea al "
        "generar el informe (). No se reescribe y no se recalcula: solo cambia "
        "autorando **otro**."
    )

def categorias_de(declaradas: Sequence[str]) -> list[str]:
    return [c for c in CATEGORÍAS if c not in declaradas]
