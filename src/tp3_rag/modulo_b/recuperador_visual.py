from __future__ import annotations

from tp3_rag.modulo_b.etapa import Candidata, Etapa, EstadoDeConsulta
from tp3_rag.nucleo.errores import ErrorDeConfiguración
from tp3_rag.puertos.indice_vectorial import ARTWORKS

TOLERANCIA_DE_NORMA = 1e-3

def combinar(e_img, e_txt, alfa: float) -> tuple[float, ...]:
    if not 0.0 <= alfa <= 1.0:
        raise ErrorDeConfiguración(
            f"`consulta.alfa = {alfa}` tiene que estar en [0, 1]. Es el peso de la imagen "
            f"contra el texto en `normalize(α·e_img + (1−α)·e_txt)`: fuera del rango, el "
            f"vector resultante no es una interpolación y `α` deja de tener la lectura que "
            f"el diseño de la arquitectura le da (*parecido a esto ↔ movido hacia allá*)."
        )
    if alfa == 1.0:
        return tuple(float(x) for x in e_img)
    if alfa == 0.0:
        return tuple(float(x) for x in e_txt)
    if len(e_img) != len(e_txt):
        raise ErrorDeConfiguración(
            f"`e_img` tiene {len(e_img)} dims y `e_txt` {len(e_txt)}. Los dos tienen que "
            f"venir del **mismo checkpoint de CLIP**: por eso el `PuertoEmbeddingImagen` es "
            f"*el único puerto con dos operaciones* — si se pidieran por separado, "
            f"nada impediría mezclar checkpoints y `α` dejaría de tener sentido geométrico."
        )
    crudo = [alfa * float(a) + (1.0 - alfa) * float(b) for a, b in zip(e_img, e_txt)]
    norma = sum(x * x for x in crudo) ** 0.5
    if norma <= 0:
        raise ErrorDeConfiguración(
            "la combinación `α·e_img + (1−α)·e_txt` dio norma 0, así que normalizarla daría "
            "`nan`. Pasa si los dos vectores son opuestos exactos con `α = 0.5`. Un `nan` "
            "en el índice **no falla, ordena mal**, que es peor (la verificación de la norma de los vectores)."
        )
    return tuple(x / norma for x in crudo)

class RecuperadorVisual(Etapa):
    NOMBRE = "recuperador_visual"
    PRECONDICIONES = ("entrada",)
    CLAVES_DE_CONFIGURACIÓN = ("consulta.alfa", "consulta.n_candidatos", "consulta.top_k")
    DEGRADA = False
    POLÍTICA = "no degrada: es el piso. Si falla, la consulta falla."

    def __init__(
        self,
        *,
        puerto_de_imagen,
        indice,
        alfa: float,
        n_candidatos: int,
        top_k: int = 5,
    ) -> None:
        self.puerto = puerto_de_imagen
        self.indice = indice
        self.alfa = float(alfa)
        if not 0.0 <= self.alfa <= 1.0:
            raise ErrorDeConfiguración(
                f"`consulta.alfa = {alfa}` tiene que estar en [0, 1]. Se valida al construir "
                f"y no al correr: es clase 1 de falla, *falla inmediata con el "
                f"nombre de la clave*, y un `α` fuera de rango descubierto en la consulta 17 "
                f"de 30 ya gastó 16 llamadas."
            )
        self.n_candidatos = int(n_candidatos)
        if self.n_candidatos < 1:
            raise ErrorDeConfiguración(
                f"`consulta.n_candidatos = {n_candidatos}` tiene que ser ≥ 1."
            )
        self.top_k = int(top_k)
        if self.top_k < 1 or self.top_k > self.n_candidatos:
            raise ErrorDeConfiguración(
                f"`consulta.top_k = {top_k}` tiene que estar en [1, n_candidatos="
                f"{self.n_candidatos}]. Pedir un top-5 de un ranking de 3 candidatas "
                f"entregaría 3 obras donde el pipeline base completo y andando y el criterio de respuesta garantizada comprometen **5**, y el síntoma "
                f"aparecería como *'devolvió 3 obras'* en vez de como una N mal declarada."
            )

    def correr(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        e_img = self.puerto.embeber_imagen(
            estado.entrada.datos_de_imagen,
            artwork_id=estado.entrada.artwork_id_de_la_imagen or estado.entrada.consulta_id,
        )
        e_txt = self.puerto.embeber_texto(estado.consulta_visual)
        vector = combinar(e_img, e_txt, self.alfa)

        resultados = self.indice.consultar(ARTWORKS, vector, self.n_candidatos)
        candidatas = tuple(
            Candidata(artwork_id=r.id, rank=i + 1, score=r.score, metadata=r.metadata)
            for i, r in enumerate(resultados)
        )
        nuevo = estado.con(fuentes_de_ranking={"visual": candidatas})
        return nuevo.proyectar_top5(
            [c.artwork_id for c in candidatas[: self.top_k]], por=self.NOMBRE
        )

    def vector_de_consulta(self, datos_de_imagen: bytes, texto: str) -> tuple[float, ...]:
        e_img = self.puerto.embeber_imagen(datos_de_imagen)
        e_txt = self.puerto.embeber_texto(texto)
        return combinar(e_img, e_txt, self.alfa)

    def a_json(self) -> dict:
        return {
            "etapa": self.NOMBRE,
            "alfa": self.alfa,
            "n_candidatos": self.n_candidatos,
            "top_k": self.top_k,
            "degrada": self.DEGRADA,
            "politica": self.POLÍTICA,
        }
