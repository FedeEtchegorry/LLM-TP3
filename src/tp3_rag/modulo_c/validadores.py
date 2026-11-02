from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.puertos.cliente_http import (
    CacheMiss,
    LazoDeRedirecciones,
    ProhibidoPorElRemoto,
    RecursoNoEncontrado,
)
from tp3_rag.puertos.indice_vectorial import ARTWORKS

@dataclass
class Resultado:
    criterio: str
    pasa: bool
    n: int
    de: int
    poblacion: str
    detalle: dict = field(default_factory=dict)
    fallas: tuple[dict, ...] = ()
    evaluable: bool = True

    @property
    def marcador(self) -> str:
        return f"{self.n}/{self.de}"

    @property
    def marca(self) -> str:
        if not self.evaluable:
            return "⊘"
        return "✓" if self.pasa else "✗"

    def a_json(self) -> dict:
        return {
            "criterio": self.criterio,
            "pasa": self.pasa,
            "evaluable": self.evaluable,
            "estado": "pasa" if self.pasa else ("falla" if self.evaluable else "no_evaluable"),
            "n": self.n,
            "de": self.de,
            "marcador": self.marcador,
            "poblacion": self.poblacion,
            "detalle": dict(self.detalle),
            "fallas": [dict(f) for f in self.fallas],
        }

PEDIDO_NEUTRO = "una pintura"

def self_retrieval(
    *,
    puerto_de_imagen,
    indice,
    manifiesto,
    bytes_de,
    n: int = 20,
    alfa: float = 1.0,
) -> Resultado:
    if alfa != 1.0:
        raise ErrorDelProyecto(
            f"el self-retrieval se pidió con `α = {alfa}` y **tiene que ser 1** "
            f" Con `α < 1` el "
            f"vector de consulta es una mezcla con el texto del pedido, así que el test "
            f"**puede fallar con un índice sano o pasar con uno roto**: deja de medir lo que "
            f"dice medir. No es un default que se pueda ajustar — es el criterio."
        )
    from tp3_rag.modulo_b.recuperador_visual import RecuperadorVisual

    corpus = tuple(manifiesto.ids_corpus)
    held_out = set(manifiesto.ids_held_out)
    ordenadas = sorted(corpus)
    paso = max(1, len(ordenadas) // n)
    elegidas = ordenadas[::paso][:n]

    intrusas = sorted(set(elegidas) & held_out)
    if intrusas:  # pragma: no cover - el manifiesto es clase 3 y los lados son disjuntos
        raise ErrorDelProyecto(
            f"el self-retrieval eligió {intrusas}, que están en el **held-out**. los detectores estructurales lo "
            f"nombra como el segundo modo de ruptura del control de self-retrieval: *correr el test sobre "
            f"imágenes del held-out, que no están indexadas, **falla con un índice sano*** — "
            f"el falso positivo del diseño."
        )

    recuperador = RecuperadorVisual(
        puerto_de_imagen=puerto_de_imagen, indice=indice, alfa=alfa, n_candidatos=5, top_k=5
    )
    aciertos, fallas = 0, []
    for artwork_id in elegidas:
        vector = recuperador.vector_de_consulta(bytes_de(artwork_id), PEDIDO_NEUTRO)
        resultados = indice.consultar(ARTWORKS, vector, 5)
        top1 = resultados[0].id if resultados else None
        if top1 == artwork_id:
            aciertos += 1
            continue
        fallas.append(
            {
                "artwork_id": artwork_id,
                "top1": top1,
                "score_top1": resultados[0].score if resultados else None,
                "posicion": next(
                    (i + 1 for i, r in enumerate(resultados) if r.id == artwork_id), None
                ),
            }
        )
    return Resultado(
        criterio="el sanity check de self-retrieval · self-retrieval, α = 1",
        pasa=aciertos == len(elegidas),
        n=aciertos,
        de=len(elegidas),
        poblacion=(
            f"{len(elegidas)} obras tomadas del **lado corpus** del manifiesto "
            f"{manifiesto.split_id} ({len(corpus)} obras), espaciadas sobre el orden "
            f"canónico para cubrir los dos museos"
        ),
        detalle={
            "alfa": alfa,
            "pedido": PEDIDO_NEUTRO,
            "nota_del_alfa": (
                "El peso de fusión tiene que ser 1: con `α < 1` el vector "
                "de consulta deja de ser el embedding de la obra y el test puede fallar con "
                "un índice sano. El `α` lo **fija el validador**, no la pieza — los detectores estructurales nombra "
                "*un α por defecto dentro de la pieza que el validador herede* como el modo "
                "de ruptura del control de self-retrieval."
            ),
            "nota_del_pedido": (
                "Con `α = 1` el texto **no entra** en el vector —`combinar` devuelve `e_img` "
                "sin tocar—, así que el pedido neutro ejercita el camino sin afectar el "
                "resultado. Es deliberado: el test mide el índice, no el pedido."
            ),
        },
        fallas=tuple(fallas),
    )

def anclaje_a_la_imagen(
    *,
    puerto_de_imagen,
    indice,
    manifiesto,
    bytes_de,
    n: int = 20,
    alfa: float,
) -> Resultado:
    from tp3_rag.modulo_b.recuperador_visual import RecuperadorVisual

    corpus = tuple(manifiesto.ids_corpus)
    held_out = set(manifiesto.ids_held_out)
    ordenadas = sorted(corpus)
    paso = max(1, len(ordenadas) // n)
    elegidas = ordenadas[::paso][:n]
    intrusas = sorted(set(elegidas) & held_out)
    if intrusas:  # pragma: no cover - los dos lados del manifiesto son disjuntos
        raise ErrorDelProyecto(
            f"el anclaje eligió {intrusas}, que están en el **held-out**: *falla con un "
            f"índice sano* (los detectores estructurales, segundo modo de ruptura del control de self-retrieval)."
        )
    recuperador = RecuperadorVisual(
        puerto_de_imagen=puerto_de_imagen, indice=indice, alfa=alfa, n_candidatos=5, top_k=5
    )
    aciertos, en_el_top5, posiciones, fallas = 0, 0, [], []
    for artwork_id in elegidas:
        vector = recuperador.vector_de_consulta(bytes_de(artwork_id), PEDIDO_NEUTRO)
        resultados = indice.consultar(ARTWORKS, vector, 5)
        ids = [r.id for r in resultados]
        posición = ids.index(artwork_id) + 1 if artwork_id in ids else None
        if posición == 1:
            aciertos += 1
        if posición is not None:
            en_el_top5 += 1
            posiciones.append(posición)
        else:
            fallas.append({"artwork_id": artwork_id, "top1": ids[0] if ids else None})
    return Resultado(
        criterio=f"anclaje a la imagen · α = {alfa}  (NO es el sanity check de self-retrieval)",
        pasa=True,
        n=aciertos,
        de=len(elegidas),
        poblacion=(
            f"{len(elegidas)} obras del **lado corpus** del manifiesto "
            f"{manifiesto.split_id} ({len(corpus)} obras), espaciadas sobre el orden "
            f"canónico — **las mismas que el sanity check de self-retrieval**, para que los números se puedan comparar"
        ),
        detalle={
            "alfa": alfa,
            "pedido": PEDIDO_NEUTRO,
            "en_el_top5": en_el_top5,
            "posicion_media_cuando_aparece": (
                round(sum(posiciones) / len(posiciones), 2) if posiciones else None
            ),
            "nota": (
                "🔴 **Esto NO es el sanity check de self-retrieval y no lo reemplaza.** el sanity check de self-retrieval es un **criterio** "
                "pinchado en `α = 1` (el sanity check de self-retrieval, el control de self-retrieval, el diseño) "
                "y el validador rechaza otro valor con error duro, porque con `α < 1` el "
                "vector deja de ser el embedding de la obra. Esto es un **instrumento de "
                "barrido**: el mismo procedimiento al `α` que se está evaluando, para leer "
                "cuánto anclaje a la imagen cuesta mover la perilla. `pasa` es True porque se "
                "pudo medir; no hay umbral."
            ),
            "nota_de_la_rama": (
                "⚠️ **Mide la rama visual, no C3.** El anclaje es una propiedad del vector "
                "`α·e_img + (1−α)·e_txt`, que es del `RecuperadorVisual`; la rama de texto y "
                "la fusión RRF **enmascararían** la pérdida de anclaje en vez de medirla. Y "
                "no cuesta ninguna llamada al proxy: CLIP y Chroma son locales."
            ),
        },
        fallas=tuple(fallas),
    )

HEADER_EXIGIDO_POR_DOMINIO: tuple[tuple[str, str], ...] = (("artic.edu", "AIC-User-Agent"),)

RESUELVE = "resuelve"
NO_RESUELVE = "no_resuelve"
SIN_EL_HEADER = "sin_el_header"
NO_CONCLUYENTE = "no_concluyente"
SIN_VERIFICAR = "sin_verificar"

PISO_DE_P2 = 0.98

INTENTOS_DEL_CHEQUEO_DE_URL = 3

PAUSA_DEL_CHEQUEO_DE_URL = 2.0

def headers_que_r6_exige_y_faltan(url: str, cliente_http) -> list[str] | None:
    declarar = getattr(cliente_http, "headers_de", None)
    if declarar is None:
        return None
    host = urlsplit(url).hostname or ""
    enviados = {str(h).lower() for h in declarar(url)}
    return [
        header
        for dominio, header in HEADER_EXIGIDO_POR_DOMINIO
        if (host == dominio or host.endswith("." + dominio)) and header.lower() not in enviados
    ]

def resolver_source_url(url: str, cliente_http, *, dormir=None) -> dict:
    dormir = time.sleep if dormir is None else dormir
    if cliente_http is None:
        return {
            "estado": SIN_VERIFICAR,
            "intentos": 0,
            "detalle": (
                "no se pasó `ClienteHTTP`, así que la resolución queda sin comprobar en vez "
                "de darse por buena (el ciclo de vida de los artefactos, punto 5)"
            ),
        }
    faltan = headers_que_r6_exige_y_faltan(url, cliente_http)
    if faltan is None:
        return {
            "estado": SIN_VERIFICAR,
            "intentos": 0,
            "detalle": (
                "el cliente **no declara** qué headers manda por dominio, así que la política de headers no se "
                "pudo verificar para esta URL. No se da por bueno: un 403 de AIC sin saber si "
                "el header iba es indistinguible de los dos síntomas conocidos"
            ),
        }
    if faltan:
        return {
            "estado": SIN_EL_HEADER,
            "intentos": 0,
            "detalle": (
                f"🔴 **la política de headers roto en el chequeo del criterio de trazabilidad**: el cliente **no** manda {faltan} a "
                f"este dominio de AIC, y la política de headers lo exige en **todas** las requests a dominios de "
                f"AIC — API **e** imágenes (`el relevamiento § Verificación del 403`). Sin el header "
                f"Cloudflare devuelve 403 con HTML, y el síntoma no apunta a la causa. **Esto "
                f"es nuestro: el piso del criterio de trazabilidad no lo absorbe.**"
            ),
        }
    último = ""
    for intento in range(1, INTENTOS_DEL_CHEQUEO_DE_URL + 1):
        try:
            status = cliente_http.get(url).status
        except LazoDeRedirecciones as e:
            return {
                "estado": NO_CONCLUYENTE,
                "intentos": intento,
                "detalle": f"lazo de redirecciones del remoto, **no se reintenta**: {e}",
            }
        except ProhibidoPorElRemoto as e:
            último = f"403 **con el header puesto**: {e}"
        except RecursoNoEncontrado as e:
            return {
                "estado": NO_RESUELVE,
                "intentos": intento,
                "detalle": f"404: la ficha que publicamos no existe en el museo ({e})",
            }
        except CacheMiss as e:
            return {
                "estado": SIN_VERIFICAR,
                "intentos": intento,
                "detalle": f"la red está deshabilitada y la respuesta no está en el caché ({e})",
            }
        except ErrorDelProyecto as e:
            return {
                "estado": NO_CONCLUYENTE,
                "intentos": intento,
                "detalle": f"el remoto no cerró después de los reintentos del `ClienteHTTP`: {e}",
            }
        except Exception as e:  # noqa: BLE001 — lo desconocido se trata como nuestro
            return {
                "estado": NO_RESUELVE,
                "intentos": intento,
                "detalle": f"{type(e).__name__}: {e}",
            }
        else:
            if status == 200:
                return {"estado": RESUELVE, "intentos": intento, "detalle": "HTTP 200"}
            return {
                "estado": NO_RESUELVE,
                "intentos": intento,
                "detalle": f"HTTP {status} (no 200)",
            }
        if intento < INTENTOS_DEL_CHEQUEO_DE_URL:
            dormir(PAUSA_DEL_CHEQUEO_DE_URL)
    return {
        "estado": NO_CONCLUYENTE,
        "intentos": INTENTOS_DEL_CHEQUEO_DE_URL,
        "detalle": (
            f"{último} — persistió en los {INTENTOS_DEL_CHEQUEO_DE_URL} intentos del chequeo "
            f"con la política de headers verificado, así que **no es nuestra configuración**"
        ),
    }

def piso_de_p2(ítems: int) -> int:
    return math.ceil(PISO_DE_P2 * ítems)

def p2_trazabilidad(
    corridas: Sequence[Any],
    *,
    ids_del_corpus: Sequence[str],
    cliente_http=None,
    dormir=None,
) -> Resultado:
    corpus = set(ids_del_corpus)
    ítems, fallas, no_concluyentes = 0, [], []
    urls: dict[str, dict] = {}
    for corrida in corridas:
        evidencia = (corrida.estado.get("evidencia") or {}) if corrida.estado else {}
        salida = corrida.salida or {}
        por_obra = {o["artwork_id"]: o for o in (salida.get("obras") or ())}
        for artwork_id in corrida.top5:
            ítems += 1
            problemas, inestabilidad = [], []
            if artwork_id not in corpus:
                problemas.append(
                    "el `artwork_id` **no está en el corpus indexado** del split: la obra "
                    "devuelta no existe donde la corrida dice que existe"
                )
            fragmentos = evidencia.get(artwork_id) or []
            if not fragmentos:
                problemas.append(
                    "**0 fragmentos** de `knowledge` disponibles para citar. Con al menos un fragmento por obra indexada "
                    "(invariante 3 de los invariantes de la registración) debería ser imposible: es la evidencia ordenada por similitud y es una "
                    "violación del criterio de trazabilidad por el camino que la variante tendría que haber cerrado"
                )
            url = (por_obra.get(artwork_id) or {}).get("source_url")
            if not url:
                problemas.append("la salida **no publicó `source_url`**: la obra no es trazable")
            else:
                if url not in urls:
                    urls[url] = resolver_source_url(url, cliente_http, dormir=dormir)
                resolución = urls[url]
                estado, detalle = resolución["estado"], resolución["detalle"]
                if estado == NO_RESUELVE:
                    problemas.append(
                        f"`source_url` **no resuelve** (no dio 200): {url} — {detalle}"
                    )
                elif estado == SIN_EL_HEADER:
                    problemas.append(f"`source_url` **no se chequeó bien**: {url} — {detalle}")
                elif estado == SIN_VERIFICAR:
                    problemas.append(f"`source_url` **no verificada**: {detalle} ({url})")
                elif estado == NO_CONCLUYENTE:
                    inestabilidad.append({"url": url, **resolución})
            if problemas:
                fallas.append(
                    {
                        "consulta_id": corrida.consulta_id,
                        "artwork_id": artwork_id,
                        "problemas": problemas,
                    }
                )
            elif inestabilidad:
                no_concluyentes.append(
                    {
                        "consulta_id": corrida.consulta_id,
                        "artwork_id": artwork_id,
                        **inestabilidad[0],
                    }
                )
    piso = piso_de_p2(ítems)
    resueltos = ítems - len(fallas) - len(no_concluyentes)
    return Resultado(
        criterio="el criterio de trazabilidad · toda obra devuelta existe, es trazable y es explicable",
        pasa=not fallas and ítems > 0 and resueltos >= piso,
        n=resueltos,
        de=ítems,
        poblacion=(
            f"{len(corridas)} corridas × top-5 = {ítems} ítems · corpus indexado de "
            f"{len(corpus)} obras"
        ),
        detalle={
            "n_urls_distintas": len(urls),
            "n_urls_que_resuelven": sum(1 for v in urls.values() if v["estado"] == RESUELVE),
            "n_urls_sin_verificar": sum(
                1 for v in urls.values() if v["estado"] == SIN_VERIFICAR
            ),
            "piso": piso,
            "piso_declarado": f"{PISO_DE_P2:.0%} de los {ítems} ítems = {piso}",
            "n_items_no_concluyentes": len(no_concluyentes),
            "n_urls_sin_el_header_de_r6": sum(
                1 for v in urls.values() if v["estado"] == SIN_EL_HEADER
            ),
            "r6_verificado_antes_de_pedir": [
                f"{dominio} → {header}" for dominio, header in HEADER_EXIGIDO_POR_DOMINIO
            ],
            "no_concluyentes": [dict(x) for x in no_concluyentes],
            "politica_de_reintento": (
                f"🔴 Los headers se verifica **antes** de pedir: sin el header, el "
                f"403 es **nuestro** y es falla dura que el piso **no** absorbe. Con el header "
                f"puesto, el 403 se reintenta hasta {INTENTOS_DEL_CHEQUEO_DE_URL} veces "
                f"({PAUSA_DEL_CHEQUEO_DE_URL:.0f} s entre intentos) y el **lazo de "
                f"redirecciones no se reintenta** —es determinístico del remoto—; los dos, si "
                f"persisten, caen en `no_concluyente`. Ni 429 ni 5xx llegan acá: los agota el "
                f"`ClienteHTTP` (el contrato de los puertos)."
            ),
            "nota_del_piso": (
                "🔴 Un `source_url` que el tercero no quiso contestar **no es una obra no "
                "trazable**: es una obra sobre la que no se pudo medir, y `n` no la cuenta "
                "—el marcador dice lo que se verificó, no lo que se esperaba—. El veredicto "
                "se compara contra el piso y las fallas **propias** bloquean igual. Es  "
                "aplicado a un criterio evaluable: *decir que falla es tan falso como decir "
                "que pasa*."
            ),
            "nota": (
                "Las tres condiciones se leen **del artefacto**, no se recalculan (el módulo de evaluación: "
                "*los informes leen almacenes; no recalculan nada*). La tercera es "
                "*disponible para citar*, no *citado*: lo segundo es el criterio de honestidad."
            ),
        },
        fallas=tuple(fallas),
    )

def p3a_referencias_colgadas(corridas: Sequence[Any]) -> Resultado:
    publicadas, colgadas_publicadas, descartadas = 0, [], 0
    for corrida in corridas:
        evidencia = (corrida.estado.get("evidencia") or {}) if corrida.estado else {}
        válidos = {
            f["fragment_id"]
            for fragmentos in evidencia.values()
            for f in (fragmentos or [])
        }
        descartadas += len(corrida.afirmaciones_descartadas)
        for obra in ((corrida.salida or {}).get("obras") or ()):
            for ref in obra.get("referencias") or ():
                publicadas += 1
                if ref["fragment_id"] not in válidos:
                    colgadas_publicadas.append(
                        {
                            "consulta_id": corrida.consulta_id,
                            "artwork_id": obra["artwork_id"],
                            "n": ref["n"],
                            "fragment_id": ref["fragment_id"],
                        }
                    )
    return Resultado(
        criterio="el conteo de referencias colgadas · 0 referencias a un fragmento inexistente o no recuperado",
        pasa=not colgadas_publicadas,
        n=publicadas - len(colgadas_publicadas),
        de=publicadas,
        poblacion=(
            f"{publicadas} referencias `[n]` publicadas en {len(corridas)} corridas"
        ),
        detalle={
            "n_colgadas_publicadas": len(colgadas_publicadas),
            "umbral": 0,
            "n_afirmaciones_descartadas_por_sa5": descartadas,
            "nota_de_sa5": (
                "Las descartadas **no son fallas del conteo de referencias colgadas**: son el evento que el conteo de referencias colgadas existe "
                "para medir. el descarte contado de afirmaciones se cae la afirmación entera —no solo la `ref`, porque eso "
                "violaría la regla de que un dato lleva referencia— y **lo cuenta**. El umbral duro es sobre lo **publicado**."
            ),
        },
        fallas=tuple(colgadas_publicadas),
    )

def p4_atribucion(corridas: Sequence[Any], *, exigen_atribucion: Mapping[str, bool]) -> Resultado:
    render_por_corrida = {}
    exigen, con_atribución, fallas = 0, 0, []
    mezclan, mezclan_y_exigen = 0, 0
    for corrida in corridas:
        salida = corrida.salida or {}
        render = salida.get("render") or ""
        render_por_corrida[corrida.consulta_id] = render
        for obra in salida.get("obras") or ():
            for ref in obra.get("referencias") or ():
                licencias = tuple(ref.get("licencias") or ()) or (ref["license"],)
                sin_declarar = sorted(set(licencias) - set(exigen_atribucion))
                if sin_declarar:
                    fallas.append(
                        {
                            "consulta_id": corrida.consulta_id,
                            "fragment_id": ref["fragment_id"],
                            "problema": (
                                f"las licencias {sin_declarar} **no están declaradas** en "
                                f"`salida.exigen_atribucion`, así que no se sabe si exigen "
                                f"atribución. No hay default: es la regla de la licencia declarada y no asumida en el "
                                f"último eslabón de la licencia por fragmento de punta a punta"
                            ),
                        }
                    )
                    continue
                licencia = " + ".join(sorted(set(licencias)))
                if len(set(licencias)) > 1:
                    mezclan += 1
                if not any(exigen_atribucion[x] for x in licencias):
                    continue
                exigen += 1
                if len(set(licencias)) > 1:
                    mezclan_y_exigen += 1
                if ref.get("atribucion") and ref["atribucion"] in render:
                    con_atribución += 1
                    continue
                fallas.append(
                    {
                        "consulta_id": corrida.consulta_id,
                        "fragment_id": ref["fragment_id"],
                        "license": licencia,
                        "problema": (
                            "sin línea de atribución"
                            if not ref.get("atribucion")
                            else "la atribución existe como campo y **no aparece en la "
                            "superficie renderizada**"
                        ),
                    }
                )
    return Resultado(
        criterio="el criterio de atribución · 100 % de los fragmentos que exigen atribución, con atribución",
        pasa=not fallas,
        n=con_atribución,
        de=exigen,
        poblacion=(
            f"{exigen} referencias publicadas con licencia que **exige** atribución, en "
            f"{len(corridas)} corridas. No es sobre los fragmentos de `knowledge`: un "
            f"fragmento que ninguna consulta citó no tiene superficie donde mostrarse"
        ),
        detalle={
            "licencias_que_exigen": sorted(k for k, v in exigen_atribucion.items() if v),
            "umbral": "100 %",
            "verificado_contra": "el texto renderizado, no el campo",
            "sin_llamar_al_llm": True,
            "n_citas_que_mezclan_licencias": mezclan,
            "n_citas_que_mezclan_y_exigen_atribucion": mezclan_y_exigen,
        },
        fallas=tuple(fallas),
    )

PALABRAS_DEL_ESPAÑOL = frozenset(
    """de la el los las un una y o que en con por para se su sus del al es son
    esta este esto como más pero sobre entre sin no lo le les muy también donde
    cuando porque aunque desde hasta ya tiene tienen puede pueden fue fueron""".split()
)
_PALABRAS = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)
UMBRAL_DE_ESPAÑOL = 0.15

def proporcion_de_palabras_del_espanol(texto: str) -> float:
    palabras = [p.lower() for p in _PALABRAS.findall(texto)]
    if not palabras:
        return 0.0
    return sum(1 for p in palabras if p in PALABRAS_DEL_ESPAÑOL) / len(palabras)

def p8_idioma(corridas: Sequence[Any]) -> Resultado:
    explicaciones, en_español, fallas = 0, 0, []
    citas, textuales = 0, 0
    for corrida in corridas:
        evidencia = (corrida.estado.get("evidencia") or {}) if corrida.estado else {}
        texto_por_id = {
            f["fragment_id"]: f["texto"]
            for fragmentos in evidencia.values()
            for f in (fragmentos or [])
        }
        for obra in ((corrida.salida or {}).get("obras") or ()):
            líneas = [l for l in (obra.get("lineas") or ()) if l.strip()]
            if líneas:
                explicaciones += 1
                proporción = proporcion_de_palabras_del_espanol(" ".join(líneas))
                if proporción >= UMBRAL_DE_ESPAÑOL:
                    en_español += 1
                else:
                    fallas.append(
                        {
                            "consulta_id": corrida.consulta_id,
                            "artwork_id": obra["artwork_id"],
                            "proporcion_de_palabras_del_espanol": round(proporción, 3),
                            "umbral": UMBRAL_DE_ESPAÑOL,
                            "problema": "la explicación no pasa el heurístico de español",
                        }
                    )
            for ref in obra.get("referencias") or ():
                citas += 1
                esperado = texto_por_id.get(ref["fragment_id"])
                if esperado is not None and ref["texto_citado"] == esperado:
                    textuales += 1
                else:
                    fallas.append(
                        {
                            "consulta_id": corrida.consulta_id,
                            "fragment_id": ref["fragment_id"],
                            "problema": (
                                "el `texto_citado` **no es byte a byte** el del fragmento: la "
                                "cita se reescribió, y *groundedness* mediría nuestra edición"
                            ),
                        }
                    )
    return Resultado(
        criterio="el criterio de idioma · explicación en español, cita textual en inglés sin traducir",
        pasa=not fallas,
        n=en_español + textuales,
        de=explicaciones + citas,
        poblacion=(
            f"{explicaciones} explicaciones por obra (con afirmaciones) + {citas} citas "
            f"publicadas, en {len(corridas)} corridas"
        ),
        detalle={
            "explicaciones_en_espanol": f"{en_español}/{explicaciones}",
            "citas_byte_a_byte": f"{textuales}/{citas}",
            "umbral_del_heuristico": UMBRAL_DE_ESPAÑOL,
            "nota_de_la_asimetria": (
                "La mitad de la **cita** es **exacta** (byte a byte) *porque la cita no pasa "
                "por el modelo* (el contrato de salida): esa decisión de diseño es la razón por la que la "
                "mitad difícil del criterio de idioma no existe. La mitad del **español** es un **heurístico "
                "declarado** —proporción de palabras función sobre un umbral elegido por "
                "nosotros— y lo que la cierra es la muestra humana de 25 del criterio de honestidad "
                "(`RevisiónDeExplicaciones`, +). Se dice en vez de presentarse como un "
                "detector de idioma."
            ),
        },
        fallas=tuple(fallas),
    )

def e5_pares_juzgados_en_algun_pool(
    *, aportes: Sequence[Any], pools: Sequence[Any], split_id: str
) -> Resultado:
    del_split = [p for p in pools if p.split_id == split_id]
    cubiertos: set[tuple[str, str]] = set()
    for p in del_split:
        cubiertos |= p.pares
    juzgados, fallas = 0, []
    for aporte in aportes:
        if aporte.split_id != split_id:
            continue
        for juicio in aporte.juicios:
            juzgados += 1
            if juicio.par in cubiertos:
                continue
            fallas.append(
                {
                    "aporte": aporte.identidad,
                    "anotador": aporte.anotador,
                    "consulta_id": juicio.consulta_id,
                    "artwork_id": juicio.artwork_id,
                    "pool_declarado": aporte.pool_id,
                    "problema": (
                        "el par **no está en ningún pool de este split**: es un juicio sobre "
                        "un par que nadie puso en una orden de trabajo. Lo único que lo "
                        "acusaba era el `%` de ítems sin juzgar de la tabla de ablación, que es después"
                    ),
                }
            )
    return Resultado(
        criterio="la procedencia del pool declarada en el aporte · todo par juzgado estuvo en algún pool de ese split",
        pasa=not fallas,
        n=juzgados - len(fallas),
        de=juzgados,
        poblacion=(
            f"{juzgados} juicios de {len([a for a in aportes if a.split_id == split_id])} "
            f"aporte(s) del split {split_id}, contra la unión de {len(del_split)} pool(s) "
            f"({len(cubiertos)} pares)"
        ),
        detalle={
            "pools_del_split": [p.pool_id for p in del_split],
            "n_pares_en_algun_pool": len(cubiertos),
            "momento": "antes de exportar la planilla",
            "nota": (
                "`algún` y no `el que lo exportó`: un juicio es sobre el par, no sobre el "
                "pool que lo trajo, y eso es lo que vuelve legales el pooling incremental y "
                "la importación de los 20 del piloto (la evaluación, la tabla de identidades)."
            ),
        },
        fallas=tuple(fallas),
    )

def e10_puerta_de_la_anotacion(
    *, pool: Any, manifiesto: Any, registros_de_variantes: Mapping[str, Any]
) -> Resultado:
    held_out = set(manifiesto.ids_held_out)
    corpus = set(manifiesto.ids_corpus)
    fallas = []

    for configuración in pool.configuraciones:
        variante_id = configuración.get("variante_id")
        registro = registros_de_variantes.get(variante_id) if variante_id else None
        if registro is None:
            fallas.append(
                {
                    "mitad": "a",
                    "configuracion": configuración.get("configuracion"),
                    "variante_id": variante_id,
                    "problema": (
                        "la variante de origen **no se pudo resolver** en el "
                        "`RegistroDeÍndices`. No se da por buena: sin la receta no se sabe "
                        "de qué split salió el corpus que trajo estos candidatos (es la "
                        "regla del ciclo de vida de los artefactos punto 5 aplicada a un validador)"
                    ),
                }
            )
            continue
        if registro.split_id != pool.split_id:
            fallas.append(
                {
                    "mitad": "a",
                    "configuracion": configuración.get("configuracion"),
                    "variante_id": variante_id,
                    "split_de_la_variante": registro.split_id,
                    "split_del_pool": pool.split_id,
                    "problema": (
                        "la configuración de origen usó una variante de **otro split**, así "
                        "que su top-5 sale de otro corpus: entran obras que no están en el "
                        "lado corpus de este manifiesto y puede entrar una del **held-out** "
                        "de este split — la exclusión del held-out de toda colección violado por la puerta de la anotación"
                    ),
                }
            )

    for item in pool.items:
        if item.artwork_id in held_out:
            fallas.append(
                {
                    "mitad": "b",
                    "consulta_id": item.consulta_id,
                    "artwork_id": item.artwork_id,
                    "problema": (
                        "la obra a juzgar está en el **held-out** del split. Es la exclusión del held-out de toda colección por la "
                        "puerta de la anotación: se gastaría juicio humano midiendo al "
                        "sistema contra una obra que el held-out existe para que no vea"
                    ),
                }
            )
        elif item.artwork_id not in corpus:
            fallas.append(
                {
                    "mitad": "b",
                    "consulta_id": item.consulta_id,
                    "artwork_id": item.artwork_id,
                    "problema": (
                        "la obra a juzgar **no está en ninguno de los dos lados** del "
                        "manifiesto: o el pool salió de una variante de otro split, o el "
                        "manifiesto no es el que se indexó"
                    ),
                }
            )
    return Resultado(
        criterio="la procedencia del pool dentro de su split · ninguna obra del held-out entra por la puerta de la anotación",
        pasa=not fallas,
        n=len(pool.items) - len([f for f in fallas if f["mitad"] == "b"]),
        de=len(pool.items),
        poblacion=(
            f"{len(pool.items)} obras del pool {pool.pool_id} contra el manifiesto "
            f"{manifiesto.split_id} ({len(corpus)} corpus + {len(held_out)} held-out), y "
            f"{len(pool.configuraciones)} configuración(es) de origen contra el "
            f"`RegistroDeÍndices`"
        ),
        detalle={
            "variantes_de_origen": list(pool.variantes_de_origen()),
            "momento": "antes de exportar la planilla",
            "nota_de_las_dos_mitades": (
                "(a) las configuraciones de origen apuntan a variantes del mismo split; "
                "(b) ninguna obra del pool es del held-out. Las dos se cumplen **por "
                "construcción** con el pool saliendo de C1 — y por eso se verifican: un "
                "invariante que se cumple por construcción deja de cumplirse el día que la "
                "construcción cambia por construcción."
            ),
            "nota_de_la_imagen_de_consulta": (
                "Las 10 imágenes de consulta **son** del held-out a propósito (el set de consultas, la exclusión del held-out de toda colección). "
                "Lo que este validador mira son las obras **recuperadas**."
            ),
        },
        fallas=tuple(fallas),
    )

def antes_de_exportar(
    *,
    pool: Any,
    manifiesto: Any,
    set_de_consultas: Any,
    registros_de_variantes: Mapping[str, Any],
    aportes: Sequence[Any],
    pools: Sequence[Any],
) -> tuple[Resultado, ...]:
    if set_de_consultas.split_id != pool.split_id:
        raise ErrorDelProyecto(
            f"el set {set_de_consultas.identidad} ancla al split "
            f"{set_de_consultas.split_id} y el pool {pool.pool_id} al "
            f"{pool.split_id}. Los juicios entran por un `consulta_id` del set y cuelgan "
            f"del `split_id` del pool: con dos splits, la referencia que resuelve queda roto desde el export."
        )
    resultados = (
        e10_puerta_de_la_anotacion(
            pool=pool, manifiesto=manifiesto, registros_de_variantes=registros_de_variantes
        ),
        e5_pares_juzgados_en_algun_pool(
            aportes=aportes, pools=pools, split_id=pool.split_id
        ),
    )
    fallidos = [r for r in resultados if not r.pasa]
    if fallidos:
        detalle = "\n".join(
            f"    ✗ {r.criterio} — {r.marcador}\n"
            + "\n".join(f"        · {f}" for f in r.fallas[:5])
            for r in fallidos
        )
        raise ErrorDelProyecto(
            f"🔴 **No se exporta la planilla**: {len(fallidos)} control(es) del ground truth "
            f"no pasa(n).\n{detalle}\n"
            f"  El momento es a propósito (el módulo de evaluación): **este es el último punto en que todavía "
            f"no se gastó anotación**. Exportar igual significaría descubrirlo con los "
            f"juicios hechos, y el trabajo humano es *lo único que no se puede regenerar*."
        )
    return resultados

def _planes_de(corridas: Sequence[Any]) -> tuple[dict, ...]:
    return tuple(c.plan for c in corridas if getattr(c, "plan", None))

def pl3_evidencia_de_a3(corridas: Sequence[Any]) -> Resultado:
    planes = _planes_de(corridas)
    conteo = {
        c: sum(1 for p in planes if (p.get("pl3") or {}).get(c))
        for c in ("no_vacias", "distintas_del_pedido_crudo", "distintas_entre_si")
    }
    las_tres = sum(1 for p in planes if p.get("cumple_pl3"))
    fallas = tuple(
        {
            "consulta_id": p.get("consulta_id"),
            "plan_id": p.get("plan_id"),
            "clausulas_rotas": sorted(k for k, v in (p.get("pl3") or {}).items() if not v),
            "consulta_visual": (p.get("consulta_visual") or "")[:80],
            "consulta_conceptual": (p.get("consulta_conceptual") or "")[:80],
        }
        for p in planes
        if not p.get("cumple_pl3")
    )
    return Resultado(
        criterio="las técnicas avanzadas aisladas por medición · las tres cláusulas de las consultas del plan — las tres cláusulas del plan (rewriting + expansion)",
        pasa=bool(planes) and las_tres == len(planes),
        n=las_tres,
        de=len(planes),
        poblacion=(
            f"{len(planes)} planes citados por las corridas leídas (las corridas sin "
            f"`plan` son C0/C1 y **no cuentan**: ahí el planificador no está en la cadena)"
        ),
        detalle={
            "no_vacias": f"{conteo['no_vacias']}/{len(planes)}",
            "distintas_del_pedido_crudo": (
                f"{conteo['distintas_del_pedido_crudo']}/{len(planes)}"
            ),
            "distintas_entre_si": f"{conteo['distintas_entre_si']}/{len(planes)}",
            "planes_distintos": len({p.get("plan_id") for p in planes}),
            "nota_de_las_tres": (
                "**Tres números y no un porcentaje**: *que difieran del crudo es "
                "**rewriting**; que difieran **entre sí** es lo que distingue **expansion** "
                "de **rewriting*** (el plan de consulta). Un agregado no distingue *«el modelo devolvió "
                "una vacía»* de *«no hubo expansión»*, que son dos hallazgos distintos."
            ),
            "nota_del_idioma": (
                "⚠️ La segunda cláusula es **necesaria y no suficiente**: los pedidos del "
                "set están en español y las dos consultas se emiten en inglés (los "
                "consumidores son el text encoder de CLIP y texto de museo en inglés), así "
                "que *«distinta del pedido crudo»* se cumple por el idioma antes que por la "
                "reescritura. Lo que la cláusula descarta es el caso degenerado —el modelo "
                "copió el pedido—; que la reescritura sea **buena** lo mide la tabla de ablación, no esto. "
                "**Va reportado.**"
            ),
        },
        fallas=fallas,
    )

def pl5_escritura_unica(corridas: Sequence[Any]) -> Resultado:
    planes = _planes_de(corridas)
    producidos: dict[str, int] = {}
    for p in planes:
        if not p.get("reusado"):
            producidos[p["plan_id"]] = producidos.get(p["plan_id"], 0) + 1
    duplicados = sorted(pid for pid, n in producidos.items() if n > 1)
    distintos = {p["plan_id"] for p in planes}
    n_reusados = sum(1 for p in planes if p.get("reusado"))
    descartadas = sum(1 for p in planes if p.get("produccion_descartada"))
    return Resultado(
        criterio="la escritura única del plan · escritura única del plan, reuso registrado",
        pasa=not duplicados,
        n=len(distintos) - len(duplicados),
        de=len(distintos),
        poblacion=(
            f"{len(distintos)} `plan_id` distintos citados por {len(planes)} corridas con "
            f"planificador en la cadena"
        ),
        detalle={
            "producidos": len(producidos),
            "reusados": n_reusados,
            "producciones_descartadas_por_carrera": descartadas,
            "nota": (
                "la escritura única del plan es lo que resuelve la tensión del **híbrido** de la tabla de identidades: el plan se "
                "*direcciona* como caché (eso es el reuso del plan al barrer el peso de fusión) y su contenido es salida **no "
                "determinista** de un LLM. *Si se reescribiera, la regla de que un id determina su contenido haría error de algo "
                "que no lo es; si se reusara sin registrarlo, no habría forma de saber qué "
                "corrida pagó la llamada que la tabla de tokens, costo y la figura cuenta.*"
            ),
        },
        fallas=tuple({"plan_id": pid, "producido_por": producidos[pid]} for pid in duplicados),
    )

def r11_un_plan_cinco_corridas(
    corridas: Sequence[Any], *, consulta_id: str, alfas_esperados: int = 5
) -> Resultado:
    del_consulta = [c for c in corridas if c.consulta_id == consulta_id and c.plan]
    por_config: dict[str, Any] = {}
    for c in del_consulta:
        previa = por_config.get(c.config_id)
        if previa is None or c.ordinal > previa.ordinal:
            por_config[c.config_id] = c
    del_barrido = tuple(por_config.values())
    plan_ids = {c.plan_id for c in del_barrido}
    corrida_ids = {c.corrida_id for c in del_barrido}
    alfas = {
        (c.contenido.get("configuracion_resuelta") or {})
        .get("consulta", {})
        .get("alfa")
        for c in del_barrido
    }
    producidos = sum(1 for c in del_barrido if c.plan_reusado is False)
    pasa = (
        len(plan_ids) == 1
        and len(corrida_ids) == alfas_esperados
        and len(alfas) == alfas_esperados
    )
    return Resultado(
        criterio=f"el reuso del plan al barrer el peso de fusión · barrido de α sobre `{consulta_id}` — 1 plan_id / {alfas_esperados} corrida_id",
        pasa=pasa,
        n=len(corrida_ids),
        de=alfas_esperados,
        poblacion=(
            f"las corridas de `{consulta_id}` con planificador en la cadena, **de cualquier "
            f"`config_id`** (último ordinal de cada uno): `α` entra en el `config_id` y lo "
            f"que lo que no entra en la identidad del plan promete es que **no** entra en el `plan_id`. {len(del_consulta)} "
            f"corrida(s) en disco → {len(del_barrido)} del barrido"
        ),
        detalle={
            "plan_id": sorted(plan_ids),
            "n_plan_id": len(plan_ids),
            "n_corrida_id": len(corrida_ids),
            "alfas": sorted(a for a in alfas if a is not None),
            "planes_producidos": producidos,
            "planes_reusados": len(del_barrido) - producidos,
            "nota": (
                "**lo que no entra en la identidad del plan es el reuso del plan al barrer el peso de fusión literal.** `α`, N, la fusión y el `variante_id` **no** son "
                "ingredientes del `plan_id`, así que el barrido paga **una** llamada de "
                "pre-retrieval y produce **cinco** corridas comparables. Si el `α` entrara, "
                "barrer costaría 5× el pre-retrieval y el diseño de la arquitectura —*una sola llamada por "
                "consulta*— dejaría de ser verdad en el barrido."
            ),
        },
        fallas=(
            ()
            if pasa
            else (
                {
                    "n_plan_id": len(plan_ids),
                    "n_corrida_id": len(corrida_ids),
                    "n_alfas": len(alfas),
                    "esperado": f"1 plan_id / {alfas_esperados} corrida_id / {alfas_esperados} α",
                },
            )
        ),
    )

def todos_sobre_corridas(
    corridas: Sequence[Any],
    *,
    ids_del_corpus: Sequence[str],
    exigen_atribucion: Mapping[str, bool],
    cliente_http=None,
    n_candidatos: int | None = None,
) -> tuple[Resultado, ...]:
    base = (
        p2_trazabilidad(corridas, ids_del_corpus=ids_del_corpus, cliente_http=cliente_http),
        p3a_referencias_colgadas(corridas),
        p4_atribucion(corridas, exigen_atribucion=exigen_atribucion),
        p8_idioma(corridas),
    )
    if _planes_de(corridas):
        base += (pl3_evidencia_de_a3(corridas), pl5_escritura_unica(corridas))
    if _rankings_fusionados(corridas):
        base += (a3_fusion_con_rangos_de_origen(corridas),)
    if _reranks_de(corridas):
        base += (a3_reranking_con_justificacion(corridas),)
    base += (p6_llamadas_al_llm(corridas, n_candidatos=n_candidatos),)
    return base

def _rankings_fusionados(corridas: Sequence[Any]) -> tuple[tuple[Any, list], ...]:
    salida = []
    for c in corridas:
        filas = (getattr(c, "estado", None) or {}).get("ranking_fusionado")
        if filas:
            salida.append((c, list(filas)))
    return tuple(salida)

def a3_fusion_con_rangos_de_origen(corridas: Sequence[Any]) -> Resultado:
    con_fusion = _rankings_fusionados(corridas)
    bien = []
    fallas = []
    con_dos_ramas = 0
    ramas_vistas: dict[str, int] = {}
    for corrida, filas in con_fusion:
        sin_rangos = [f for f in filas if not f.get("rangos_de_origen")]
        incoherentes = [
            f
            for f in filas
            if int(f.get("n_ramas") or 0) != len(f.get("rangos_de_origen") or {})
        ]
        for f in filas:
            for rama in f.get("rangos_de_origen") or {}:
                ramas_vistas[rama] = ramas_vistas.get(rama, 0) + 1
        if sin_rangos or incoherentes:
            fallas.append(
                {
                    "consulta_id": corrida.consulta_id,
                    "filas_sin_rangos_de_origen": len(sin_rangos),
                    "filas_con_n_ramas_incoherente": len(incoherentes),
                }
            )
            continue
        bien.append(corrida)
        if max((int(f.get("n_ramas") or 0) for f in filas), default=0) >= 2:
            con_dos_ramas += 1
    return Resultado(
        criterio="las técnicas avanzadas aisladas por medición · 3ª técnica — el ranking fusionado con sus rangos de origen por rama",
        pasa=bool(con_fusion) and not fallas,
        n=len(bien),
        de=len(con_fusion),
        poblacion=(
            f"{len(con_fusion)} corridas con `ranking_fusionado` (las corridas sin él son "
            f"C0–C2 y **no cuentan**: ahí el fusionador no está en la cadena)"
        ),
        detalle={
            "con_las_dos_ramas": f"{con_dos_ramas}/{len(con_fusion)}",
            "obras_aportadas_por_rama": dict(sorted(ramas_vistas.items())),
            "nota_de_las_dos_ramas": (
                "⚠️ **`con_las_dos_ramas` no es un umbral y no puede serlo.** Que una obra "
                "aparezca en las dos ramas depende del corpus y del pedido; con la rama de "
                "texto caída son **cero por diseño** (el criterio de respuesta garantizada), y eso es la degradación "
                "funcionando. Lo que las técnicas avanzadas aisladas por medición exige es **el artefacto**: el ranking fusionado "
                "con sus rangos de origen, que es lo que esta fila cuenta."
            ),
            "nota_de_d5": (
                "El `score_fusionado` sale de **rangos** y no de scores (la fusión recíproca de rangos con k = 60): *un "
                "coseno de CLIP y uno de `text-embedding-3-small` no son calibrables entre "
                "sí*. La estrategia viaja en cada fila, así que el artefacto dice con qué "
                "se fusionó."
            ),
        },
        fallas=tuple(fallas),
    )

ROLES_DEL_LLM: tuple[str, ...] = ("planificador", "reranker", "generador", "juez")

LLAMADAS_COMPROMETIDAS: dict[str, Any] = {"C0": 0, "C1": 1, "C2": 2, "C3": 2, "C4": None}

def _reranks_de(corridas: Sequence[Any]) -> tuple[tuple[Any, list], ...]:
    salida = []
    for c in corridas:
        filas = (getattr(c, "estado", None) or {}).get("rerank")
        if filas:
            salida.append((c, list(filas)))
    return tuple(salida)

def a3_reranking_con_justificacion(corridas: Sequence[Any]) -> Resultado:
    con_rerank = _reranks_de(corridas)
    bien, fallas = [], []
    movimiento = {"sin_cambio": 0, "churn_total": 0, "promovidas_de_afuera": 0}
    scores_vistos: list[int] = []
    sin_puntaje_total = 0
    for corrida, filas in con_rerank:
        problemas = []
        sin_justificación = [
            f for f in filas if not str(f.get("justificacion") or "").strip()
        ]
        if sin_justificación:
            problemas.append(
                f"{len(sin_justificación)} candidata(s) **sin justificación**: las técnicas avanzadas aisladas por medición pide "
                f"*el score 0–10 por candidato **con su justificación***, y el módulo de evaluación lo dice "
                f"como *dejando los dos en el estado*"
            )
        fuera_de_escala = [
            f
            for f in filas
            if f.get("estado") == "puntuado"
            and not isinstance(f.get("score_0_10"), int)
            or (
                f.get("estado") == "puntuado"
                and isinstance(f.get("score_0_10"), int)
                and not 0 <= int(f["score_0_10"]) <= 10
            )
        ]
        if fuera_de_escala:
            problemas.append(
                f"{len(fuera_de_escala)} puntaje(s) fuera de la escala entera 0–10 de "
                f"el diseño de la arquitectura: dos escalas en una columna es lo que prohíbe"
            )
        fusionado = (corrida.estado or {}).get("ranking_fusionado") or []
        if not fusionado:
            problemas.append(
                "no quedó `ranking_fusionado`: la no destrucción de lo ya registrado dice que el reranker **agrega** y no "
                "pisa, y sin el fusionado `C3−C2` y `C4−C3` dejan de ser legibles del "
                "mismo artefacto"
            )
        elif any("score_fusionado" not in f for f in fusionado):
            problemas.append(
                "el `ranking_fusionado` perdió su `score_fusionado`: es el campo que la no destrucción de lo ya registrado "
                "protege explícitamente"
            )
        puntuadas = [f for f in filas if f.get("estado") == "puntuado"]
        sin_puntaje_total += len(filas) - len(puntuadas)
        scores_vistos.extend(
            int(f["score_0_10"]) for f in puntuadas if isinstance(f.get("score_0_10"), int)
        )
        ids_del_rerank = [f.get("artwork_id") for f in filas]
        if corrida.top5 and not corrida.degradaciones:
            if list(corrida.top5) != ids_del_rerank[: len(corrida.top5)]:
                problemas.append(
                    f"el `top5` no son los primeros {len(corrida.top5)} del orden del "
                    f"rerank: {list(corrida.top5)} contra "
                    f"{ids_del_rerank[: len(corrida.top5)]}. la forma declarada del artefacto lo verifica al cerrar y "
                    f"el Módulo C **al leer** (la regla de qué corridas agregan a la tabla)"
                )
            if not set(corrida.top5) <= set(ids_del_rerank):
                problemas.append(
                    "el `top5` tiene obras que el rerank no puntuó: generalizó el "
                    "contrato a *subconjunto de lo que alguna etapa de ranking propuso*, "
                    "y traer una obra de la nada haría que el `%` de ítems sin juzgar de "
                    "la tabla de ablación la contara sin que nadie supiera de dónde salió"
                )
        libro = (corrida.contenido.get("libro_mayor") or {}).get("reranker") or {}
        llamadas = int(libro.get("n_llamadas") or 0)
        intentos = sum(int(f.get("n_intentos") or 0) for f in filas)
        if llamadas and llamadas != intentos:
            problemas.append(
                f"el libro mayor cuenta {llamadas} llamadas del rol `reranker` y las filas "
                f"declaran {intentos} intento(s). **Un candidato por llamada** "
                f"(el diseño de la arquitectura, la cantidad de candidatos a rerankear firme) es de lo que depende `C4 = 2 + N`"
            )
        if problemas:
            fallas.append(
                {"consulta_id": corrida.consulta_id, "problemas": problemas}
            )
            continue
        bien.append(corrida)
        antes = [f["artwork_id"] for f in fusionado][: len(corrida.top5)]
        if list(corrida.top5) == antes:
            movimiento["sin_cambio"] += 1
        movimiento["churn_total"] += len([o for o in corrida.top5 if o not in antes])
        movimiento["promovidas_de_afuera"] += sum(
            1
            for f in filas[: len(corrida.top5)]
            if int(f.get("rango_fusionado") or 0) > len(corrida.top5)
        )
    n = len(con_rerank)
    return Resultado(
        criterio="las técnicas avanzadas aisladas por medición · 4ª técnica — el score 0–10 por candidato **con su justificación**",
        pasa=bool(con_rerank) and not fallas,
        n=len(bien),
        de=n,
        poblacion=(
            f"{n} corridas con `rerank` (las corridas sin él son C0–C3 y **no cuentan**: "
            f"ahí el reranker no está en la cadena)"
        ),
        detalle={
            "candidatos_puntuados": len(scores_vistos),
            "candidatos_sin_puntaje": sin_puntaje_total,
            "escala_usada": (
                f"[{min(scores_vistos)}, {max(scores_vistos)}] de [0, 10] · "
                f"{len(set(scores_vistos))} valores distintos"
                if scores_vistos
                else "ausente"
            ),
            "score_medio": (
                round(sum(scores_vistos) / len(scores_vistos), 2) if scores_vistos else None
            ),
            "top5_sin_cambio_vs_el_fusionado": f"{movimiento['sin_cambio']}/{n}",
            "churn_medio_del_top5": (round(movimiento["churn_total"] / n, 2) if n else None),
            "obras_promovidas_desde_fuera_del_top5": movimiento["promovidas_de_afuera"],
            "nota_del_movimiento": (
                "⚠️ **Cuánto movió el top-5 no es un umbral.** Que el reranker deje el orden "
                "igual es un **resultado** —el fusionado ya estaba bien, o el modelo no "
                "distingue— y convertirlo en criterio daría incentivo a esconderlo (el objetivo de recuperación, "
                "el delta por categoría de pedido). Lo que las técnicas avanzadas aisladas por medición exige es **el artefacto**; esto es insumo del delta "
                "`C4−C3`, y hay que leerlo con el muestreo fijado, registrado y con su variabilidad medida al lado: *si la variabilidad es "
                "comparable al delta, cambia cómo se lee la tabla*."
            ),
            "nota_de_la_escala": (
                "🔑 **Pocos valores distintos sobre muchos candidatos es información, no un "
                "defecto.** Con once valores enteros y N = 20 los empates son el caso normal, "
                "y por eso el orden **desempata por el rango fusionado**: con todos los "
                "puntajes iguales el orden que sale **es** el fusionado, así que *«el "
                "reranker no distinguió nada»* se lee como *«el top-5 no se movió»* en vez de "
                "como una permutación arbitraria."
            ),
        },
        fallas=tuple(fallas),
    )

def p6_llamadas_al_llm(
    corridas: Sequence[Any], *, n_candidatos: int | None = None
) -> Resultado:
    por_configuracion: dict[str, dict] = {}
    for corrida in corridas:
        configuración = (corrida.contenido.get("pipeline") or {}).get("configuracion", "?")
        acumulado = por_configuracion.setdefault(
            configuración,
            {"corridas": 0, "pagadas": 0, "nominales": 0, "planes_reusados": 0,
             "por_rol": {}, "embeddings": 0, "detalle": [], "con_reintento": 0},
        )
        libro = corrida.contenido.get("libro_mayor") or {}
        pagadas = 0
        for rol in ROLES_DEL_LLM:
            llamadas = int((libro.get(rol) or {}).get("n_llamadas") or 0)
            pagadas += llamadas
            if llamadas:
                acumulado["por_rol"][rol] = acumulado["por_rol"].get(rol, 0) + llamadas
        acumulado["embeddings"] += sum(
            int((libro.get(rol) or {}).get("n_llamadas") or 0)
            for rol in ("embedding_texto", "embedding_imagen")
        )
        estado = corrida.estado or {}
        nominales = (
            (1 if estado.get("plan") else 0)
            + len(estado.get("rerank") or ())
            + (1 if estado.get("explicacion") else 0)
        )
        reusado = bool(corrida.plan_reusado)
        acumulado["corridas"] += 1
        acumulado["pagadas"] += pagadas
        acumulado["nominales"] += nominales
        acumulado["planes_reusados"] += 1 if reusado else 0
        acumulado["con_reintento"] += 1 if pagadas > nominales - (1 if reusado else 0) else 0
        acumulado["detalle"].append(nominales)

    fallas = []
    medido: dict[str, Any] = {}
    for configuración, d in sorted(por_configuracion.items()):
        por_consulta = sorted(set(d["detalle"]))
        esperado = LLAMADAS_COMPROMETIDAS.get(configuración)
        if esperado is None and configuración == "C4":
            esperado = (2 + int(n_candidatos)) if n_candidatos else None
        medido[configuración] = {
            "corridas": d["corridas"],
            "llamadas_nominales_por_consulta": (
                por_consulta[0] if len(por_consulta) == 1 else por_consulta
            ),
            "comprometido_por_p6": esperado,
            "intentos_pagados_en_total": d["pagadas"],
            "intentos_pagados_por_consulta": (
                round(d["pagadas"] / d["corridas"], 2) if d["corridas"] else None
            ),
            "corridas_con_reintento_m4_1": d["con_reintento"],
            "planes_reusados_r11": d["planes_reusados"],
            "por_rol": dict(sorted(d["por_rol"].items())),
            "embeddings_que_p6_no_cuenta": d["embeddings"],
        }
        if esperado is not None and por_consulta != [esperado]:
            fallas.append(
                {
                    "configuracion": configuración,
                    "comprometido": esperado,
                    "medido": por_consulta,
                    "problema": (
                        f"el criterio de costo compromete **{esperado}** llamadas al LLM por consulta para "
                        f"{configuración} y las corridas miden {por_consulta} llamadas "
                        f"**nominales** —las que la fila cuesta, leídas del estado—. Un "
                        f"conteo que no cierra es el número publicado que no es"
                    ),
                }
            )
    cerradas = sum(
        1
        for c, d in medido.items()
        if d["comprometido_por_p6"] is not None
        and d["llamadas_nominales_por_consulta"] == d["comprometido_por_p6"]
    )
    con_compromiso = sum(1 for d in medido.values() if d["comprometido_por_p6"] is not None)
    return Resultado(
        criterio="el criterio de costo · llamadas al LLM por consulta y por configuración, **medidas**",
        pasa=bool(con_compromiso) and not fallas,
        n=cerradas,
        de=con_compromiso,
        poblacion=(
            f"{sum(d['corridas'] for d in medido.values())} corridas de "
            f"{sorted(medido)} · del **libro mayor** de cada corrida (los tokens en el artefacto de corrida), o sea del "
            f"`usage` que devolvió la API"
        ),
        detalle={
            "por_configuracion": medido,
            "roles_que_p6_cuenta": list(ROLES_DEL_LLM),
            "nota_del_embedding": (
                "🔴 **Un embedding no es una llamada al LLM**: desde hay **dos** "
                "llamadas a `embedding_texto` por consulta —la rama de texto y el orden de "
                "evidencia por similitud— y el criterio de costo **no las suma**. la tabla de tokens, costo y la figura sí las mide, y están "
                "en el renglón `embeddings_que_p6_no_cuenta` para que el que lea la tabla no "
                "tenga que restarlas a mano."
            ),
            "nota_de_r11": (
                "⚠️ **`llamadas_nominales_por_consulta` es lo que la fila cuesta** —leído del "
                "**estado**: una por etapa con binding y `N` por el reranker— y "
                "`intentos_pagados` es lo que la API facturó (los tokens en el artefacto de corrida). Difieren por **dos** "
                "cosas: el reuso del plan al barrer el peso de fusión —un plan reusado no paga la llamada del planificador: *el "
                "pre-retrieval es una llamada por consulta y por pedido, **no por fila de la "
                "tabla***— y los reintentos."
            ),
            "nota_de_los_reintentos": (
                "🔴 **`el criterio de costo = 2 + N` es el conteo NOMINAL, y lo destapó la primera corrida de "
                "C4**: una consulta pagó **23** llamadas porque el generador devolvió un JSON "
                "que no satisface la regla de que un dato lleva referencia y manda reintentar. Un reintento **es** una "
                "llamada al LLM —se factura y la tabla de tokens, costo y la figura la cuenta— pero **no** es una etapa más de "
                "la cadena: contar intentos haría que el criterio de costo fallara exactamente en el caso que "
                " existe para absorber. Las dos cifras se publican, y "
                "`corridas_con_reintento_m4_1` dice cuántas veces pasó."
            ),
            "nota_de_la_cota": (
                "⚠️ **`C4 = 2 + N` es una cota superior y se alcanza casi siempre.** El "
                "reranker puntúa `min(N, |ranking fusionado|)`, y el fusionado es la unión de "
                "las dos ramas, así que tiene ≥ N filas siempre que la rama visual traiga N "
                "candidatas — que es por construcción. Si bajara, es un dato del corpus y no "
                "un incumplimiento, y por eso el valor medido se publica al lado."
            ),
        },
        fallas=tuple(fallas),
    )

def r12_variabilidad(
    corridas: Sequence[Any], *, rol: str = "reranker"
) -> Resultado:
    por_par: dict[tuple[str, str], list[Any]] = {}
    for c in corridas:
        por_par.setdefault((c.config_id, c.consulta_id), []).append(c)
    pares = {k: sorted(v, key=lambda c: c.ordinal) for k, v in por_par.items() if len(v) >= 2}

    comparaciones, fallas = [], []
    top5_igual = 0
    churn_total = 0
    scores_comparados = scores_distintos = 0
    diferencia_absoluta = 0
    tokens_out = []
    jaccards: list[float] = []
    for (config_id, consulta_id), corridas_del_par in sorted(pares.items()):
        primera, última = corridas_del_par[0], corridas_del_par[-1]
        muestreos = [_muestreo_efectivo_de(c, rol) for c in (primera, última)]
        if not all(muestreos):
            fallas.append(
                {
                    "consulta_id": consulta_id,
                    "problema": (
                        f"una de las dos corridas no registró el **muestreo efectivo** del "
                        f"rol `{rol}`. el muestreo efectivo registrado: *la corrida registra los parámetros que el "
                        f"puerto **efectivamente envió**, no los que el binding pidió* — "
                        f"*registrar «lo configurado» en vez de «lo usado» es la forma "
                        f"elegante de no cumplir el muestreo fijado, registrado y con su variabilidad medida*"
                    ),
                }
            )
            continue
        if muestreos[0] != muestreos[1]:
            fallas.append(
                {
                    "consulta_id": consulta_id,
                    "problema": (
                        f"el muestreo efectivo **difiere** entre los dos ordinales: "
                        f"{muestreos[0]} vs {muestreos[1]}. Entonces lo que se mide no es la "
                        f"variabilidad del modelo sino la de la configuración, y el muestreo fijado, registrado y con su variabilidad medida pide "
                        f"el muestreo **fijado** y registrado"
                    ),
                }
            )
            continue
        churn = len([o for o in última.top5 if o not in set(primera.top5)])
        churn_total += churn
        if list(primera.top5) == list(última.top5):
            top5_igual += 1
        from tp3_rag.modulo_c.metricas import jaccard

        jaccards.append(jaccard(primera.top5, última.top5))
        antes = {
            f["artwork_id"]: f.get("score_0_10")
            for f in ((primera.estado or {}).get("rerank") or [])
        }
        después = {
            f["artwork_id"]: f.get("score_0_10")
            for f in ((última.estado or {}).get("rerank") or [])
        }
        comunes = sorted(set(antes) & set(después))
        cambiados = [
            o
            for o in comunes
            if antes[o] is not None
            and después[o] is not None
            and antes[o] != después[o]
        ]
        scores_comparados += len(comunes)
        scores_distintos += len(cambiados)
        diferencia_absoluta += sum(abs(int(después[o]) - int(antes[o])) for o in cambiados)
        for c in (primera, última):
            tokens_out.append(
                int(((c.contenido.get("libro_mayor") or {}).get(rol) or {}).get(
                    "tokens_out", 0
                ) or 0)
            )
        comparaciones.append(
            {
                "consulta_id": consulta_id,
                "ordinales": [primera.ordinal, última.ordinal],
                "top5_igual": list(primera.top5) == list(última.top5),
                "churn_del_top5": churn,
                "candidatos_comparados": len(comunes),
                "candidatos_con_otro_score": len(cambiados),
            }
        )
    n = len(pares)
    return Resultado(
        criterio=(
            f"el muestreo fijado, registrado y con su variabilidad medida · la variabilidad del `{rol}` **medida**: la misma configuración dos veces, "
            f"con el muestreo fijado y registrado"
        ),
        pasa=bool(pares) and not fallas,
        n=len(comparaciones),
        de=n,
        poblacion=(
            f"{n} par(es) `(config_id, consulta_id)` con **dos o más ordinales**, de "
            f"{len(corridas)} corridas leídas. El ordinal existe para esto (la tabla de identidades)"
        ),
        detalle={
            "top5_identico": f"{top5_igual}/{n}" if n else "ausente",
            "churn_medio_del_top5": round(churn_total / n, 3) if n else None,
            "candidatos_comparados": scores_comparados,
            "candidatos_con_otro_score": scores_distintos,
            "porcentaje_de_scores_que_se_movieron": (
                round(100 * scores_distintos / scores_comparados, 1)
                if scores_comparados
                else None
            ),
            "diferencia_absoluta_media_del_score": (
                round(diferencia_absoluta / scores_distintos, 2) if scores_distintos else None
            ),
            "qs_de_ruido": (
                round(1.0 - sum(jaccards) / len(jaccards), 3) if jaccards else None
            ),
            "nota_del_qs_de_ruido": (
                "🔴 **`QS` tiene un piso de ruido y esta es su medida.** `QS = 1 − "
                "media(Jaccard)` entre los top-5 de los **tres pedidos de una imagen** "
                "(`la evaluación §B`), o sea *cuánto cambia el resultado cuando cambia el pedido*. "
                "Acá el pedido **es el mismo** —la misma consulta corrida dos veces— así que "
                "este número es el `QS` que el sistema produce **sin que el pedido cambie**: "
                "es **variabilidad, no sensibilidad**. ⚠️ El `QS` que el criterio de que el pedido cambia el resultado y la tabla de ablación publican "
                "para esta configuración hay que leerlo **contra este piso**, y eso no estaba "
                "en ningún documento: `la evaluación §B` advierte que *un sistema aleatorio tiene "
                "QS ≈ 1* y no que **un sistema no determinista tiene QS > 0 sin usar el "
                "pedido**. **Va reportado.**"
            ),
            "tokens_out_del_rol_por_corrida": tokens_out,
            "comparaciones": comparaciones,
            "nota_del_umbral": (
                "🔴 **No hay umbral acá, y no puede haberlo.** el muestreo fijado, registrado y con su variabilidad medida pide **medir** la "
                "variabilidad, no acotarla: el muestreo se fija (`temperature = 0`, `seed`) y "
                "la `seed` es *best effort* en la API de chat. Lo que decide cómo se lee la "
                "tabla es la **comparación con el delta**: *si la variabilidad es comparable "
                "al delta `C4−C3`, hay que decirlo fuerte*."
            ),
            "nota_de_m4": (
                "El dato de referencia: con el generador, **la entrada es "
                "determinista y la salida no** —24.705 vs 25.238 tokens de salida en dos "
                "corridas iguales—. Por eso `pasa` mira que el **muestreo** esté fijado y "
                "registrado y que el emparejamiento exista; que los números difieran **es el "
                "dato**, no la falla."
            ),
        },
        fallas=tuple(fallas),
    )

def _muestreo_efectivo_de(corrida, rol: str) -> dict | None:
    plan = corrida.plan or {}
    if rol == "planificador":
        efectivo = (plan.get("binding") or {}).get("muestreo_efectivo")
        if efectivo:
            return dict(efectivo)
    for etapa in (corrida.contenido.get("pipeline") or {}).get("etapas") or ():
        if (etapa.get("binding") or {}).get("rol") != rol:
            continue
        efectivo = etapa.get("muestreo_efectivo")
        return dict(efectivo) if efectivo else None
    return None

def terminos_de(etiquetas: Sequence[str], tabla: Mapping[str, Any]) -> tuple[str, ...]:
    salida: list[str] = []
    for etiqueta in etiquetas:
        for término in (etiqueta, *(tabla.get(etiqueta) or ())):
            if término and término.lower() not in [s.lower() for s in salida]:
                salida.append(término)
    return tuple(salida)

def nombra_la_cultura(texto: str, terminos: Sequence[str]) -> tuple[str, ...]:
    bajo = (texto or "").lower()
    return tuple(t for t in terminos if t.lower() in bajo)

def niega(texto: str, marcadores: Sequence[str]) -> tuple[str, ...]:
    bajo = (texto or "").lower()
    return tuple(m for m in marcadores if m.lower() in bajo)

def cultura_pedida_en_el_plan(
    planes: Sequence[Mapping[str, Any]],
    *,
    culturas_por_consulta: Mapping[str, Sequence[str]],
    terminos_de_cultura: Mapping[str, Any],
    marcadores_de_negacion: Sequence[str] = (),
) -> Resultado:
    por_version: dict[str, dict] = {}
    for plan in planes:
        consulta_id = plan.get("consulta_id") or ""
        etiquetas = culturas_por_consulta.get(consulta_id)
        if not etiquetas:
            continue
        version = str((plan.get("binding") or {}).get("version_del_prompt") or "?")
        acumulado = por_version.setdefault(version, {"de": 0, "nombran": 0, "fallas": []})
        términos = terminos_de(etiquetas, terminos_de_cultura)
        conceptual = plan.get("consulta_conceptual") or ""
        encontrados = nombra_la_cultura(conceptual, términos)
        negaciones = niega(conceptual, marcadores_de_negacion)
        en_visual = nombra_la_cultura(plan.get("consulta_visual") or "", términos)
        acumulado["de"] += 1
        acumulado.setdefault("tambien_en_la_visual", 0)
        acumulado.setdefault("en_negativo", 0)
        if encontrados and not negaciones:
            acumulado["nombran"] += 1
            acumulado["tambien_en_la_visual"] += 1 if en_visual else 0
        else:
            acumulado["en_negativo"] += 1 if negaciones else 0
            acumulado["fallas"].append(
                {
                    "consulta_id": consulta_id,
                    "version_del_prompt": version,
                    "modo": (
                        "la **niega**" if negaciones else "describe **otra** cultura"
                    ),
                    "cultura_pedida": list(etiquetas)[:4],
                    "negacion": list(negaciones),
                    "consulta_conceptual": conceptual[:160],
                }
            )
    versiones = sorted(por_version)
    ultima = versiones[-1] if versiones else None
    actual = por_version.get(ultima or "", {"de": 0, "nombran": 0, "fallas": []})
    return Resultado(
        criterio=(
            "· el plan nombra la **cultura pedida** (no la de la imagen subida) — "
            "antes/después del prompt"
        ),
        pasa=bool(actual["de"]) and actual["nombran"] == actual["de"],
        n=actual["nombran"],
        de=actual["de"],
        poblacion=(
            f"los pedidos **culturales** del set con plan, por versión del prompt. "
            f"Versión medida como *después*: v{ultima}. Las otras dos categorías no "
            f"tienen `cultura_pedida` y **no cuentan**"
        ),
        detalle={
            "por_version": {
                v: f"{d['nombran']}/{d['de']}" for v, d in sorted(por_version.items())
            },
            "tambien_en_la_consulta_visual": {
                v: f"{d.get('tambien_en_la_visual', 0)}/{d['nombran']}"
                for v, d in sorted(por_version.items())
            },
            "la_niegan_en_vez_de_nombrarla": {
                v: d.get("en_negativo", 0) for v, d in sorted(por_version.items())
            },
            "nota_de_la_negacion": (
                "🔴 **Las dos fallas no son la misma, y la negativa es peor.** *"
                "«Describe otra cultura»* es legible como lo que es; *«not related to Indian "
                "art»* **parece correcta y recupera lo contrario**, porque la negación no "
                "existe en el espacio de embeddings. las separa (*«y en uno la "
                "niega»*) y el v2 prohíbe la segunda explícitamente."
            ),
            "nota_del_antes_despues": (
                "🔴 **Es un resultado del informe, no un arreglo**: *«el "
                "pre-retrieval ingenuo suelta la cultura pedida en el 40 % de los pedidos "
                "culturales»* es una observación publicable sobre **query rewriting**, y con "
                "el v2 al lado es un **antes/después medido**. Las dos versiones coexisten "
                "porque la versión del prompt es ingrediente del `plan_id` (la versión del prompt como ingrediente del plan)."
            ),
            "nota_de_la_tabla": (
                "El predicado usa las **etiquetas declaradas** del `SetDeConsultas` "
                "más los gentilicios de `[medicion.terminos_de_cultura]`, que está en el "
                "bloque **no determinante** del `config_id`: ninguna etapa lo lee. "
                "Una entrada que falta **sub-cuenta**, nunca sobre-cuenta."
            ),
        },
        fallas=tuple(actual["fallas"]),
    )

def r2_variantes_coexistiendo(registros: Sequence[Any]) -> Resultado:
    from tp3_rag.nucleo import identidad

    ids = [r.variante_id for r in registros]
    recetas_por_id: dict[str, list[dict]] = {}
    for r in registros:
        recetas_por_id.setdefault(r.variante_id, []).append(r.receta)

    rederivaciones = []
    for r in registros:
        receta = r.receta
        try:
            derivado = identidad.variante_id(
                catalogo_id=receta["catalogo_id"],
                split_id=receta["split_id"],
                criterio_de_inclusion=receta["criterio_de_inclusion"],
                checkpoint_de_clip=receta["checkpoint_de_clip"],
                version_del_preprocesado=receta["version_del_preprocesado"],
                estrategia_de_chunking=receta["estrategia_de_chunking"],
                campos_de_knowledge=receta["campos_de_knowledge"],
                modelo_de_embedding_de_texto=receta["modelo_de_embedding_de_texto"],
                criterio_de_culture=receta["criterio_de_culture"],
            )
        except (KeyError, TypeError) as e:  # pragma: no cover - lo caza `validar()`
            derivado = f"no derivable: {e}"
        rederivaciones.append(
            {
                "variante_id": r.variante_id,
                "re_derivado": derivado,
                "coincide": derivado == r.variante_id,
            }
        )

    raices = {
        r.variante_id: (r.a_json().get("colecciones") or {}).get("raiz")
        for r in registros
    }

    claves = sorted({k for r in registros for k in r.receta})
    ejes_que_difieren = [
        k
        for k in claves
        if len({_canonico(r.receta.get(k)) for r in registros}) > 1
        and not k.startswith("nota")
    ]

    colisiones = tuple(
        {
            "variante_id": vid,
            "n_recetas_distintas": len({_canonico(x) for x in recetas}),
            "causa_declarada": "falta un ingrediente en la tabla de identidades (la regla de que un id determina su contenido, clase 1)",
        }
        for vid, recetas in sorted(recetas_por_id.items())
        if len({_canonico(x) for x in recetas}) > 1
    )
    sin_rederivar = tuple(d for d in rederivaciones if not d["coincide"])
    raices_repetidas = len(raices) != len({v for v in raices.values() if v})

    a = len(registros) >= 2
    b = len(set(ids)) == len(ids)
    c = not sin_rederivar
    d = not raices_repetidas and all(raices.values())
    cumplidas = sum((a, b, c, d))
    return Resultado(
        criterio="la identificación de variantes · variantes de índice coexistiendo, identificables y versionadas (los detectores estructurales)",
        pasa=cumplidas == 4 and not colisiones,
        n=cumplidas,
        de=4,
        poblacion=(
            f"las **{len(registros)}** variantes registradas en `artifacts/indices/` "
            f"(el layout de artefactos: el registro va al repo, los vectores no). Splits: "
            f"{sorted({r.split_id for r in registros})}"
        ),
        detalle={
            "a_coexisten": f"{len(registros)} variantes registradas (pide ≥ 2)",
            "b_ids_distintos": f"{len(set(ids))} ids distintos de {len(ids)} entradas",
            "c_id_re_derivado_de_la_receta_sola": (
                f"{sum(1 for d_ in rederivaciones if d_['coincide'])}/{len(rederivaciones)}"
            ),
            "d_un_directorio_por_variante": (
                f"{len({v for v in raices.values() if v})} raíces distintas de "
                f"{len(raices)} variantes"
            ),
            "variantes": [
                {
                    "variante_id": r.variante_id,
                    "campos_de_knowledge": r.receta.get("campos_de_knowledge"),
                    "chunking": (r.receta.get("estrategia_de_chunking") or {}).get("nombre"),
                    "n_indexadas": r.conteos.get("n_indexadas"),
                    "n_fragmentos": r.conteos.get("n_fragmentos"),
                    "colecciones": raices.get(r.variante_id),
                }
                for r in registros
            ],
            "ejes_de_la_receta_que_difieren": ejes_que_difieren,
            "nota_de_los_ejes": (
                "Es lo que vuelve legible *«identificables»*: dos variantes que difieren "
                "**solo** en un eje declarado son una **ablación**, y la tabla de la evaluación "
                "las cita por el par (config_id, variante_id)."
            ),
            "nota_de_la_re_derivacion": (
                "🔑 Es la **quinta** vez que el proyecto aplica esta prueba —al split, a las "
                "cuatro consultas del piloto, a la variante de y acá a las dos—: "
                "*re-derivar la identidad desde la declaración sola*. Lo que compra es que "
                "el id sea **función** de la receta, y por lo tanto que dos recetas "
                "distintas no puedan compartir id."
            ),
            "nota_del_alcance": (
                "⚠️ **No prueba que los nueve ingredientes de la tabla de identidades estén completos.** Un "
                "eje que no esté ni en la receta ni en el id es invisible acá; lo atrapa "
                "la regla de que un id determina su contenido **al re-construir** (`_exigir_mismo_contenido`), que es por qué "
                "los detectores estructurales lo llama *«el único que atrapa un eje que no está en la lista»*."
            ),
        },
        fallas=colisiones + sin_rederivar,
    )

def r3_cache_de_imagen_reusado(registros: Sequence[Any]) -> Resultado:
    ROL = "embedding_imagen"
    if len(registros) < 2:
        return Resultado(
            criterio=f"el reuso del caché de imagen entre variantes · 0 llamadas nuevas al PuertoEmbeddingImagen en la 2ª variante (los detectores estructurales)",
            pasa=False,
            n=0,
            de=1,
            poblacion=(
                f"{len(registros)} variante(s) registrada(s). **el reuso del caché de imagen entre variantes necesita dos**: lo que "
                f"afirma es que la **segunda** no embebe, y con una no hay segunda. Es por "
                f"esto que el detector no se podía correr antes de"
            ),
            detalle={
                "nota": (
                    "No es que el reuso del caché de imagen entre variantes falle: es que **no es verificable** con esta población, y "
                    "eso se reporta en vez de darlo por bueno. La salida es construir una "
                    "segunda variante — `poetry run tp3-indexar --set "
                    "fragmentacion.campos_de_knowledge=[…]`."
                )
            },
            fallas=({"n_variantes": len(registros), "pide": 2},),
        )

    ordenadas = sorted(
        registros,
        key=lambda r: (r.a_json().get("procedencia") or {}).get("fecha_de_construccion") or "",
    )
    primera, posteriores = ordenadas[0], ordenadas[1:]

    filas = []
    for r in ordenadas:
        libro = r.a_json().get("libro_mayor") or {}
        rol = dict(libro.get(ROL) or {})
        cita = r.cita_al_embebido
        filas.append(
            {
                "variante_id": r.variante_id,
                "construida": (r.a_json().get("procedencia") or {}).get(
                    "fecha_de_construccion"
                ),
                "roles_medidos": sorted(libro),
                "rol_de_imagen_presente": ROL in libro,
                "llamadas_a_embedding_imagen": int(rol.get("n_llamadas", 0)),
                "items_de_embedding_imagen": int(rol.get("n_items", 0)),
                "segundos_de_embedding_imagen": float(rol.get("segundos", 0.0)),
                "embebido_citado": {
                    "ordinal": cita.get("ordinal"),
                    "n_embebidas": cita.get("n_embebidas"),
                    "n_ordinales_disponibles": cita.get("n_ordinales_disponibles"),
                },
            }
        )
    de_las_posteriores = filas[1:]

    llamadas_nuevas = sum(f["llamadas_a_embedding_imagen"] for f in de_las_posteriores)
    ordinales = {f["embebido_citado"]["ordinal"] for f in filas}
    n_embebidas = {f["embebido_citado"]["n_embebidas"] for f in filas}

    a = llamadas_nuevas == 0 and not any(
        f["rol_de_imagen_presente"] for f in de_las_posteriores
    )
    b = len(ordinales) == 1 and len(n_embebidas) == 1
    c = (
        sum(f["items_de_embedding_imagen"] for f in de_las_posteriores) == 0
        and sum(f["segundos_de_embedding_imagen"] for f in de_las_posteriores) == 0.0
    )
    cumplidas = sum((a, b, c))
    return Resultado(
        criterio="el reuso del caché de imagen entre variantes · 0 llamadas nuevas al PuertoEmbeddingImagen en la 2ª variante (los detectores estructurales)",
        pasa=cumplidas == 3,
        n=llamadas_nuevas,
        de=0,
        poblacion=(
            f"las **{len(posteriores)}** variante(s) construida(s) después de "
            f"`{primera.variante_id}`, ordenadas por "
            f"`procedencia.fecha_de_construccion`: "
            f"{[f['variante_id'] for f in de_las_posteriores]}. El libro mayor de cada "
            f"construcción viaja adentro de su `RegistroDeÍndices` (los tokens en el artefacto de corrida)"
        ),
        detalle={
            "llamadas_nuevas_al_PuertoEmbeddingImagen": llamadas_nuevas,
            "a_rol_ausente_del_libro_mayor": (
                f"{sum(1 for f in de_las_posteriores if not f['rol_de_imagen_presente'])}"
                f"/{len(de_las_posteriores)}"
            ),
            "b_mismo_ordinal_del_embebido": (
                f"ordinal {sorted(o for o in ordinales if o is not None)} · "
                f"n_embebidas {sorted(n for n in n_embebidas if n is not None)}"
            ),
            "c_sin_items_ni_segundos_de_imagen": c,
            "por_variante": filas,
            "nota_de_la_topologia": (
                "🔒 **Es topología y no disciplina.** El `ConstructorDeÍndice` **no recibe** "
                "el `PuertoEmbeddingImagen` —*«no es que no lo llame: no lo tiene»*—: recibe "
                "el `CachéDeEmbeddingsDeImagen`, cuya clave **excluye** el criterio de "
                "inclusión, el `variante_id` y el `split_id` (la tabla de identidades, *«la exclusión más "
                "importante del documento»*). Un vector ausente del caché es **error duro**, "
                "no una obra que se re-embebe."
            ),
            "nota_de_a6": (
                "los detectores estructurales nombra el dataset documentado y reproducible como la segunda mitad del detector: *«donde `n` embebidas "
                "deja de diferir de `n` indexadas»*. Leída bien, la evidencia es que la `n` "
                "embebidas **no se movió** entre las dos variantes y las dos citan **el mismo "
                "ordinal**: embeber de nuevo habría producido otro (la regla de qué se versiona)."
            ),
            "nota_de_lo_que_compra": (
                "🔑 Es la restricción que vuelve barata la ablación secundaria: las imágenes "
                "se embeben **una vez** y cada variante nueva cuesta **solo el texto**. Es la "
                "mitad de *«~20 minutos»* del roadmap que **sí** era cierta."
            ),
        },
        fallas=tuple(
            {
                "variante_id": f["variante_id"],
                "llamadas_a_embedding_imagen": f["llamadas_a_embedding_imagen"],
                "roles_medidos": f["roles_medidos"],
                "ordinal_citado": f["embebido_citado"]["ordinal"],
            }
            for f in de_las_posteriores
            if f["rol_de_imagen_presente"] or f["llamadas_a_embedding_imagen"]
        )
        + (
            ()
            if b
            else (
                {
                    "ordinales_citados": sorted(str(o) for o in ordinales),
                    "n_embebidas_citadas": sorted(str(n) for n in n_embebidas),
                    "esperado": "un solo ordinal y una sola `n embebidas` para todas",
                },
            )
        ),
    )

def _canonico(valor):
    import json as _json

    try:
        return _json.dumps(valor, sort_keys=True, ensure_ascii=False)
    except TypeError:  # pragma: no cover - la receta es JSON por construcción
        return repr(valor)

CRITERIO_ESTRICTO: Mapping[str, bool] = {"description_no_vacia": True, "imagen_real": True}
CRITERIO_AMPLIO: Mapping[str, bool] = {"description_no_vacia": False, "imagen_real": True}

CATALOGO_DEL_BANCO = "ba9c0de1a5bada00"

@dataclass(frozen=True)
class VarianteDelBanco:
    criterio: dict
    split_id: str
    variante_id: str
    ids_corpus: tuple[str, ...]
    ids_indexados: tuple[str, ...]
    obras_sin_fragmento: tuple[str, ...]
    n_fragmentos: int
    registrada: bool
    rechazo: str | None = None

    @property
    def igualdad_de_conjuntos(self) -> bool:
        return set(self.ids_indexados) == set(self.ids_corpus)

    def a_json(self) -> dict:
        return {
            "criterio_de_inclusion": self.criterio,
            "split_id": self.split_id,
            "variante_id": self.variante_id,
            "n_corpus_del_manifiesto": len(self.ids_corpus),
            "n_ids_indexados": len(self.ids_indexados),
            "igualdad_de_conjuntos": self.igualdad_de_conjuntos,
            "faltan_en_el_indice": sorted(set(self.ids_corpus) - set(self.ids_indexados)),
            "sobran_en_el_indice": sorted(set(self.ids_indexados) - set(self.ids_corpus)),
            "n_fragmentos": self.n_fragmentos,
            "obras_sin_fragmento": list(self.obras_sin_fragmento),
            "registrada": self.registrada,
            "rechazo": self.rechazo,
        }

def exigir_banco_aislado(almacén):
    from tp3_rag.dominio import ground_truth as gt

    raíz = Path(getattr(almacén, "raíz", "."))
    peligrosas = (gt.RAÍZ, "artifacts", "data/chroma", "data/catalogo")
    encontradas = [p for p in peligrosas if (raíz / p).exists()]
    if encontradas:
        raise ErrorDelProyecto(
            f"🔴 **el banco del corpus parametrizado se niega a correr sobre {raíz}**: ya hay "
            f"{encontradas} ahí.\n"
            f"  el corpus parametrizado pide *construir dos variantes del mismo catálogo con **criterios de "
            f"inclusión distintos*** (los detectores estructurales), y el criterio es **ingrediente del "
            f"`split_id`** (la tabla de identidades, clase 3). O sea que correrlo acá **derivaría otro "
            f"split**, y el `split_id` es el ancla del `SetDeConsultas`, del pool de 379 "
            f"ítems y de cada `AporteDeAnotador`: quedarían **dos splits congelados** y la "
            f"anotación dejaría de tener un ancla única.\n"
            f"  Nada se borraría, y eso es lo engañoso — el daño es que el ancla deja de "
            f"ser una. los detectores estructurales lo dice en su primer párrafo: *un detector que para "
            f"verificarse quema el ground truth sirve menos que nada*.\n"
            f"  El banco corre sobre un almacén **vacío** (un `tmp_path` de `pytest`), con "
            f"un catálogo de fixture, `ÍndiceVectorialEnMemoria` y `EmbeddingTextoFijo`: "
            f"es el *«nivel 2 sobre el índice de fixture, sin tocar anotación»* que los detectores estructurales "
            f"pide."
        )
    return raíz

def rechazo_del_banco(almacén) -> str | None:
    try:
        exigir_banco_aislado(almacén)
    except ErrorDelProyecto as e:
        return str(e)
    return None

def banco_de_dos_criterios(
    *,
    registros: Sequence[Any],
    almacén,
    criterios: Sequence[Mapping[str, bool]] = (CRITERIO_ESTRICTO, CRITERIO_AMPLIO),
    overrides: Mapping[str, Any] | None = None,
    dim: int = 8,
) -> tuple[VarianteDelBanco, ...]:
    import numpy as np

    from tp3_rag.modulo_a.constructor_de_indice import (
        ConstructorDeÍndice,
        derivar_variante_id,
    )
    from tp3_rag.modulo_a.embebedor_de_imagenes import CachéDeEmbeddingsDeImagen
    from tp3_rag.modulo_a.fragmentador import Fragmentador
    from tp3_rag.modulo_a.particionador import Particionador
    from tp3_rag.modulo_a.selector_de_corpus import SelectorDeCorpus
    from tp3_rag.nucleo.configuracion import Configuración
    from tp3_rag.puertos.embedding_imagen import IdentidadDelModelo
    from tp3_rag.puertos.embedding_texto import EmbeddingTextoFijo
    from tp3_rag.puertos.indice_vectorial import ÍndiceVectorialEnMemoria

    exigir_banco_aislado(almacén)
    base = {"fragmentacion.unidad": "caracteres", **dict(overrides or {})}

    salida: list[VarianteDelBanco] = []
    for criterio in criterios:
        cfg = Configuración.cargar(
            overrides={
                **base,
                **{f"ingesta.criterio_de_inclusion.{k}": v for k, v in criterio.items()},
            }
        )
        modelo = IdentidadDelModelo(cfg["embebido.checkpoint"], "1", dim)
        caché = CachéDeEmbeddingsDeImagen(
            almacén.ruta(cfg["rutas.embeddings_imagen"]), modelo
        )
        for n, r in enumerate(registros):
            if r.imagen is None:
                continue
            vector = np.zeros(dim, dtype=np.float32)
            vector[n % dim] = 1.0
            caché.escribir(r.artwork_id, r.imagen.hash_bytes, vector)

        elegibles = SelectorDeCorpus(dict(cfg["ingesta.criterio_de_inclusion"])).seleccionar(
            list(registros)
        )
        split = Particionador(cfg, almacén=almacén).congelar(
            elegibles,
            catalogo_id=CATALOGO_DEL_BANCO,
            version_del_normalizador="1",
            derivar_otro=True,
        )
        manifiesto = split.manifiesto
        fragmentador = Fragmentador(cfg)
        índice = ÍndiceVectorialEnMemoria(
            variante_id=derivar_variante_id(
                cfg,
                catalogo_id=CATALOGO_DEL_BANCO,
                split_id=manifiesto.split_id,
                fragmentador=fragmentador,
            )
        )
        constructor = ConstructorDeÍndice(
            cfg,
            almacén=almacén,
            indice=índice,
            manifiesto=manifiesto,
            catalogo_id=CATALOGO_DEL_BANCO,
            embebedor_de_texto=EmbeddingTextoFijo(dim=dim),
            fragmentador=fragmentador,
        )
        corpus = constructor.verificar_cobertura(list(registros))
        constructor.construir_artworks(corpus)
        por_obra = constructor.fragmentar(corpus)
        rechazo = None
        registrada = False
        try:
            constructor.construir_knowledge(por_obra)
            constructor.registrar(por_obra=por_obra, cita_al_embebido={})
            registrada = True
        except ErrorDelProyecto as e:
            rechazo = str(e)
        ids_del_corpus = tuple(sorted(manifiesto.ids_corpus))
        salida.append(
            VarianteDelBanco(
                criterio=dict(elegibles.criterio),
                split_id=manifiesto.split_id,
                variante_id=constructor.variante_id,
                ids_corpus=ids_del_corpus,
                ids_indexados=tuple(sorted(índice.ids(ARTWORKS))),
                obras_sin_fragmento=tuple(
                    i for i in ids_del_corpus if not por_obra.get(i)
                ),
                n_fragmentos=sum(len(v) for v in por_obra.values()),
                registrada=registrada,
                rechazo=rechazo,
            )
        )
    return tuple(salida)

def r4_corpus_parametrizado(variantes: Sequence[VarianteDelBanco]) -> Resultado:
    de = 3
    if len(variantes) < 2:
        return Resultado(
            criterio="el corpus parametrizado · corpus parametrizado, dos criterios de inclusión (los detectores estructurales)",
            pasa=False,
            n=0,
            de=de,
            poblacion=(
                f"{len(variantes)} variante(s) en el banco. **el corpus parametrizado necesita dos**: lo que "
                f"afirma es que la arquitectura es invariante al **valor** del criterio, y "
                f"con un valor no hay con qué comparar"
            ),
            detalle={"nota": "El banco se arma con `banco_de_dos_criterios`."},
            fallas=({"n_variantes": len(variantes), "pide": 2},),
        )

    fallas: list[dict] = []
    for v in variantes:
        if not v.igualdad_de_conjuntos:
            fallas.append(
                {
                    "clausula": "a",
                    **v.a_json(),
                    "problema": (
                        "`ids indexados` **no es** el lado corpus de su manifiesto "
                        "(los invariantes de la registración inv. 2). Es igualdad de **conjuntos** a propósito: dos "
                        "errores que se compensan dan el mismo conteo"
                    ),
                }
            )
    a = not fallas

    amplia = max(variantes, key=lambda v: len(v.ids_corpus))
    if not amplia.registrada:
        fallas.append(
            {
                "clausula": "b",
                **amplia.a_json(),
                "problema": (
                    "la variante del criterio **amplio** no registra. Si las obras que "
                    "quedaron con 0 fragmentos son justo las que el criterio estricto "
                    "excluía, **alguna pieza está duplicando el criterio de inclusión** —el "
                    "candidato que los detectores estructurales nombra es el `Fragmentador`— y el invariante 3 de "
                    "los invariantes de la registración la rechazó al registrar, que es donde corresponde"
                ),
            }
        )
    b = amplia.registrada

    corpus = {tuple(v.ids_corpus) for v in variantes}
    c = len(corpus) == len(variantes)
    if not c:
        fallas.append(
            {
                "clausula": "c",
                "criterios": [v.criterio for v in variantes],
                "n_corpus": [len(v.ids_corpus) for v in variantes],
                "problema": (
                    "los dos criterios dieron **el mismo** lado corpus, así que el detector "
                    "no midió nada: *un detector que pasa siempre es una preocupación, no "
                    "un control* (los detectores estructurales). El fixture tiene que tener al menos una obra que "
                    "el criterio estricto excluya y el amplio incluya"
                ),
            }
        )

    cumplidas = sum((a, b, c))
    return Resultado(
        criterio="el corpus parametrizado · corpus parametrizado, dos criterios de inclusión (los detectores estructurales)",
        pasa=cumplidas == de,
        n=cumplidas,
        de=de,
        poblacion=(
            f"{len(variantes)} variantes construidas en el banco sobre **el mismo "
            f"catálogo** y con criterios distintos: "
            + " · ".join(
                f"{v.criterio} → {len(v.ids_corpus)} corpus / {v.n_fragmentos} fragmentos"
                for v in variantes
            )
            + ". Índice en memoria y embebedor fijo: **cero llamadas al proveedor**"
        ),
        detalle={
            "a_ids_indexados_es_el_lado_corpus": f"{sum(1 for v in variantes if v.igualdad_de_conjuntos)}/{len(variantes)}",
            "b_la_del_criterio_amplio_registra": amplia.registrada,
            "c_los_criterios_dan_corpus_distintos": c,
            "splits_derivados": sorted({v.split_id for v in variantes}),
            "variantes_derivadas": sorted({v.variante_id for v in variantes}),
            "por_variante": [v.a_json() for v in variantes],
            "nota_del_modo_de_ruptura": (
                "los detectores estructurales nombra el candidato: *«el **Fragmentador** descartando obras sin "
                "texto, que duplicaría el criterio en silencio»*. Con el criterio estricto "
                "la duplicación es **invisible**; con el amplio entra una obra sin "
                "`description` y, si alguna pieza duplica el criterio, sale con 0 "
                "fragmentos y el invariante 3 de los invariantes de la registración la rechaza al registrar."
            ),
            "nota_de_por_que_no_quema_nada": (
                "🔴 El criterio es **ingrediente del `split_id`** (la tabla de identidades), así que este "
                "detector **deriva otro split** por construcción — y el `split_id` es el "
                "ancla del `SetDeConsultas`, del pool y de los juicios. Por eso "
                "`banco_de_dos_criterios` llama a `exigir_banco_aislado`, que **falla** si "
                "bajo la raíz del almacén existe `artifacts/` o `artifacts/ground-truth/`: "
                "con la raíz del repo no hay forma de que corra. los detectores estructurales: *un detector que "
                "para verificarse quema el ground truth sirve menos que nada*."
            ),
            "nota_de_la_tercera_clausula": (
                "⚠️ **(c) no está en los detectores estructurales y hace falta.** Sin ella, un catálogo donde las "
                "dos membresías coincidieran haría pasar el detector **midiendo nada**, que "
                "es el falso positivo que `r2` y `r11` también evitan."
            ),
        },
        fallas=tuple(fallas),
    )

DONDE_CORRE_EL_BANCO = (
    "`banco_de_dos_criterios` sobre un almacén **vacío** (un `tmp_path` de `pytest`), con "
    "un catálogo de fixture, `ÍndiceVectorialEnMemoria` y `EmbeddingTextoFijo`: el "
    "*«nivel 2 sobre el índice de fixture, sin tocar anotación»* que los detectores estructurales pide. ⚠️ **El "
    "banco está escrito y todavía sin correr**: montarlo es construir dos splits, y "
    "construir no es de esta entrega — así que lo que falta queda **declarado**, no simulado."
)

def r4_sin_banco(*, raíz, rechazo: str | None) -> Resultado:
    base = r4_corpus_parametrizado(())
    porque = rechazo or (
        "el guardia del banco **no se opone** a este almacén —no hay `artifacts/`, ni "
        "`artifacts/ground-truth/`, ni `data/chroma/` bajo su raíz—, así que lo que falta "
        "acá **no es el aislamiento: es la población**. Construirla es derivar **otro "
        "`split_id` y otro corpus** (el criterio de inclusión es ingrediente del id, "
        "la tabla de identidades clase 3), y eso es caro y no es de esta entrega (los detectores estructurales, el corpus parametrizado)."
    )
    return Resultado(
        criterio=f"{base.criterio} · **no ejecutable todavía**",
        pasa=False,
        evaluable=False,
        n=0,
        de=base.de,
        poblacion=(
            f"**0 de las 2 variantes que el corpus parametrizado pide**, sobre `{raíz}`. El marcador "
            f"`0/{base.de}` son sus tres cláusulas y **ninguna fue evaluada**: no es un "
            f"`0/3` medido. el corpus parametrizado es el único de los cinco estructurales de los detectores estructurales que pide "
            f"**construir** —dos criterios de inclusión distintos ⇒ dos `split_id`— y este "
            f"entrypoint no construye: *leer artefactos es su único vínculo con el "
            f"pipeline* (la decisión de separar la evaluación)"
        ),
        detalle={
            "ejecutable": False,
            "lo_que_falta": (
                "una segunda variante del **mismo catálogo** con **otro criterio de "
                "inclusión**, o sea otro `split_id` y otro corpus"
            ),
            "lo_que_contesto_el_guardia": porque,
            "donde_si_corre": DONDE_CORRE_EL_BANCO,
            "clausulas_evaluadas": f"0/{base.de}",
            "nota_de_por_que_no_pasa_ni_falla": (
                "🔴 **Pasar vacuamente es peor que no tener el detector** y "
                "fallar sería publicar un veredicto sobre el vacío. los detectores estructurales pide que el "
                "detector sea **ejecutable**; mientras no lo sea, lo ejecutable es **decirlo "
                "con su razón**, que es lo que `r2` y `r3` hacían cuando había una sola "
                "variante y lo que `r11`/`r12` hacen cuando no hay corridas."
            ),
            "nota_del_tercer_estado": (
                "🔴 no evaluable **no es un ✗**. El comando sale con **0** y "
                "`tp3-validar todo` lo lista como no-evaluable con su razón; `pasa` sigue "
                "en `false`, así que no se puede citar como criterio cumplido. Es la cuarta "
                "vez que el proyecto toma esta forma: κ *no medido* ≠ *bajo* (el chequeo), κ "
                "*indefinido* se reporta ausente y nunca 0 ni 1, y un modelo sin "
                "precio da ausente, no cero."
            ),
            "nota_de_la_trampa": (
                "⚠️ **Y la razón por la que esto no se arregla corriéndolo igual:** el "
                "`split_id` es el ancla del `SetDeConsultas`, del pool de 379 ítems y de "
                "cada `AporteDeAnotador`. Derivar otro no borra nada —y eso es lo "
                "engañoso—: el daño es que **el ancla deja de ser una**."
            ),
        },
        fallas=base.fallas,
    )

ROLES_CON_TOKENS: Mapping[str, str] = {
    "planificador": "planificador",
    "reranker": "reranker",
    "generador": "generador",
    "recuperador_textual": "embedding_texto",
    "recolector_de_evidencia": "embedding_texto",
}

ROLES_SIN_TOKENS: Mapping[str, str] = {
    "embedding_imagen": "CLIP corre **local** (el stack declarado): no hay `usage` que pedir",
    "indice": "Chroma es local y no es un modelo",
}

def r9_trazas_aunque_sea_codigo_propio(corridas: Sequence[Any]) -> Resultado:
    alarmas: list[dict] = []
    revisados = 0
    por_rol: dict[str, dict[str, int]] = {}
    reusos = 0
    for corrida in corridas:
        contenido = corrida.a_json() if hasattr(corrida, "a_json") else dict(corrida)
        estado = contenido.get("estado") or {}
        libro = contenido.get("libro_mayor") or {}
        pobladas = set(estado.get("claves_pobladas") or ())
        for etapa, rol in sorted(ROLES_CON_TOKENS.items()):
            if etapa not in pobladas:
                continue
            if etapa == "planificador" and (estado.get("plan") or {}).get("reusado"):
                reusos += 1
                continue
            medido = dict(libro.get(rol) or {})
            tokens = int(medido.get("tokens_in", 0)) + int(medido.get("tokens_out", 0))
            revisados += 1
            acumulado = por_rol.setdefault(rol, {"revisados": 0, "con_tokens": 0})
            acumulado["revisados"] += 1
            if tokens > 0:
                acumulado["con_tokens"] += 1
                continue
            alarmas.append(
                {
                    "corrida_id": contenido.get("corrida_id"),
                    "consulta_id": contenido.get("consulta_id"),
                    "configuracion": (contenido.get("pipeline") or {}).get("configuracion"),
                    "etapa": etapa,
                    "rol": rol,
                    "clave_poblada": (contenido.get("pipeline") or {})
                    .get("claves_por_etapa", {})
                    .get(etapa),
                    "rol_en_el_libro_mayor": rol in libro,
                    "tokens": tokens,
                    "problema": (
                        f"la etapa `{etapa}` **pobló su clave** —o sea que produjo "
                        f"salida— y el rol `{rol}` tiene **0 tokens** en el libro "
                        f"mayor. Es la alarma del libro mayor completo por rol: o la etapa llamó a un SDK "
                        f"**sin pasar por el puerto** —y entonces no hay decorador "
                        f"que la vea (una traza por llamada a puerto)— o el puerto no reportó el `usage`, y "
                        f"la tabla de tokens, costo y la figura y el criterio de costo están contando sobre un libro incompleto"
                    ),
                }
            )
    return Resultado(
        criterio="el libro mayor completo por rol · un rol con 0 tokens y salida no vacía es la alarma (los detectores estructurales)",
        pasa=not alarmas and revisados > 0,
        n=revisados - len(alarmas),
        de=revisados,
        poblacion=(
            f"{revisados} pares (corrida, etapa con modelo tarifado) sobre "
            f"{len(corridas)} corridas, cruzando `estado.claves_pobladas` (la forma declarada del artefacto) contra "
            f"`libro_mayor` (los tokens en el artefacto de corrida/la atribución de cada llamada a su rol). **Excluidos y declarados**: "
            f"{sorted(ROLES_SIN_TOKENS)} (locales) y {reusos} plan(es) reusado(s) "
            f"(la escritura única del plan/el reuso del plan al barrer el peso de fusión)"
        ),
        detalle={
            "por_rol": {k: f"{v['con_tokens']}/{v['revisados']}" for k, v in sorted(por_rol.items())},
            "n_planes_reusados_excluidos": reusos,
            "roles_sin_tokens_por_diseno": dict(ROLES_SIN_TOKENS),
            "nota_del_cruce": (
                "🔑 **Ninguna de las dos mitades sola alcanza.** Un rol ausente del libro "
                "mayor es indistinguible de una etapa omitida (la omisión de etapas en runtime); una etapa que pobló "
                "su clave no dice si hubo span. Lo que vuelve visible el modo de ruptura de "
                "los detectores estructurales —*una etapa que llama a un SDK sin puerto*— es el cruce."
            ),
            "nota_del_plan_reusado": (
                "🐛 **La exclusión que el detector tuvo que aprender midiendo.** Sin ella "
                "daba **848 alarmas** sobre las 1.151 corridas en disco, y todas correctas "
                "del lado del pipeline: la escritura única del plan dice que *un `plan_id` lo produce **una** "
                "corrida y el resto lo reusa*, así que una corrida que reusó el plan **no "
                "llamó al modelo**. Lo que las distingue es `plan.reusado`, que el artefacto "
                "ya declara."
            ),
            "nota_de_p7": (
                "⚠️ **La inyección de falla no produce falsos positivos**, y no por suerte: "
                "con `plan_de_falla` la etapa **no puebla** su clave (la forma declarada del artefacto), así que queda "
                "fuera del denominador. Verificado sobre las corridas con "
                "`generador = «excepcion»`."
            ),
            "nota_de_lo_tautologico": (
                "los detectores estructurales descarta la otra mitad que el tramo 2 proponía —*los tokens por rol "
                "tienen que sumar al total de la corrida*— porque **el total es esa suma**: "
                "`libro_mayor` y `libro_mayor_por_etapa` son dos `GROUP BY` de la misma "
                "lista de spans, no dos acumuladores."
            ),
        },
        fallas=tuple(alarmas),
    )
