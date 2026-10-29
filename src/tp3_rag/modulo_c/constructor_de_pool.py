from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tp3_rag.dominio import pool as dpool
from tp3_rag.dominio.consultas import CATEGORÍAS, ConsultaDelSet, SetDeConsultas
from tp3_rag.dominio.pool import Pool, ÍtemDePool
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.errores import ErrorDeConstrucción
from tp3_rag.nucleo.procedencia import commit_actual

N_CONSULTAS_DEL_PILOTO = len(CATEGORÍAS) + 1

FILAS_DE_LA_TABLA: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4")

CONFIGURACIONES_POR_ESTADO: dict[str, tuple[str, ...]] = {
    "piloto": ("C1",),
    "congelado": FILAS_DE_LA_TABLA,
}

CATEGORÍA_REPETIDA_POR_OMISIÓN = "ambiguo"

@dataclass(frozen=True)
class Regla:
    semilla: int | None = None

    def semilla_efectiva(self, manifiesto) -> int:
        return int(self.semilla if self.semilla is not None else manifiesto.contenido["semilla"])

    def _candidatas(self, set_de_consultas: SetDeConsultas):
        candidatas = [c for c in set_de_consultas.consultas() if not c.es_ejemplo]
        if not candidatas:
            raise ErrorDeConstrucción(
                f"{set_de_consultas.identidad}: no hay ninguna consulta que no sea "
                f"`es_ejemplo`. Un ejemplo **no es dato**: gastar anotación humana sobre un "
                f"pedido que nadie escribió para medir es exactamente lo que el `estado` del "
                f"set existe para impedir."
            )
        return candidatas

@dataclass(frozen=True)
class ReglaDeTodasLasConsultas(Regla):
    def elegir(self, set_de_consultas: SetDeConsultas, *, semilla: int):
        del semilla
        return tuple(sorted(self._candidatas(set_de_consultas), key=lambda c: c.consulta_id))

    def a_json(self, *, semilla: int, elegidas: Sequence[ConsultaDelSet]) -> dict:
        return {
            "fuente": "pooling con metodología TREC",
            "texto": (
                "Para cada consulta: se corren todas las configuraciones de la ablación, se "
                "toma la unión de los top-5 de cada una, se deduplica y se mezcla el orden "
                "ocultando de qué configuración vino cada una."
            ),
            "seleccion_de_consultas": "todas",
            "n_consultas": len(elegidas),
            "orden": "por `consulta_id` ascendente",
            "por_que_ese_orden": (
                "Las consultas quedan **agrupadas** —se mezcla dentro de cada una, como TREC "
                "mezcla dentro del topic— así que el orden entre grupos es contenido del pool "
                "y es parte de lo que la política de trazado compara al importar. Por `consulta_id` y no por el "
                "orden del archivo del set: el archivo se puede reordenar sin cambiar el "
                "artefacto, el id no."
            ),
            "excluye": (
                "las consultas marcadas `es_ejemplo` en el `SetDeConsultas`: un ejemplo no es "
                "dato (el set de consultas / )."
            ),
            "mezcla_dentro_de_la_consulta": (
                'sha256("<semilla>|<consulta_id>|<artwork_id>"), el menor primero '
                "(`identidad.rango_de_seleccion`). No es un PRNG a propósito: el orden "
                "mezclado **es contenido** —es lo que la política de trazado compara al importar— y la "
                "declaración tiene que alcanzar para re-derivarlo."
            ),
            "semilla": semilla,
            "de_donde_sale_la_semilla": (
                "la del `ManifiestoDeSplit`"
                if self.semilla is None
                else "declarada a mano con `--semilla`"
            ),
            "elegidas": [
                {
                    "turno": n,
                    "consulta_id": c.consulta_id,
                    "categoria": c.categoria,
                    "artwork_id": c.artwork_id,
                    "pedido": c.pedido,
                }
                for n, c in enumerate(elegidas, 1)
            ],
        }

@dataclass(frozen=True)
class ReglaDeSelección(Regla):
    categoria_repetida: str = CATEGORÍA_REPETIDA_POR_OMISIÓN
    imagenes_distintas: bool = True

    def __post_init__(self) -> None:
        if self.categoria_repetida not in CATEGORÍAS:
            raise ErrorDeConstrucción(
                f"`categoria_repetida` = {self.categoria_repetida!r} no es una de "
                f"{list(CATEGORÍAS)}. La regla del diseño de la evaluación es *una por categoría más una "
                f"repetida*: la repetida tiene que ser una de las tres."
            )

    def elegir(
        self, set_de_consultas: SetDeConsultas, *, semilla: int
    ) -> tuple[ConsultaDelSet, ...]:
        candidatas = self._candidatas(set_de_consultas)
        por_categoria = {
            cat: sorted(
                (c for c in candidatas if c.categoria == cat),
                key=lambda c: identidad.rango_de_seleccion(semilla, c.consulta_id),
            )
            for cat in CATEGORÍAS
        }
        elegidas: list[ConsultaDelSet] = []
        for turno, categoria in enumerate(self.orden_de_turnos(), 1):
            elegida = self._primera_libre(por_categoria[categoria], elegidas)
            if elegida is None:
                raise ErrorDeConstrucción(
                    f"la regla no pudo completar el turno {turno} "
                    f"(categoría {categoria!r}): quedan "
                    f"{len(por_categoria[categoria])} consultas de esa categoría y ninguna "
                    f"libre"
                    + (
                        " con una **imagen que no se haya usado**. Bajá la restricción con "
                        "`--permitir-imagen-repetida` o elegí otra categoría repetida."
                        if self.imagenes_distintas
                        else "."
                    )
                )
            elegidas.append(elegida)
        return tuple(elegidas)

    def orden_de_turnos(self) -> tuple[str, ...]:
        return tuple(CATEGORÍAS) + (self.categoria_repetida,)

    def _primera_libre(
        self, ordenadas: Sequence[ConsultaDelSet], elegidas: Sequence[ConsultaDelSet]
    ) -> ConsultaDelSet | None:
        ids = {c.consulta_id for c in elegidas}
        imágenes = {c.artwork_id for c in elegidas}
        for c in ordenadas:
            if c.consulta_id in ids:
                continue
            if self.imagenes_distintas and c.artwork_id in imágenes:
                continue
            return c
        return None

    def a_json(self, *, semilla: int, elegidas: Sequence[ConsultaDelSet]) -> dict:
        return {
            "fuente": "la ronda piloto, fijada el 2026-10-11",
            "texto": (
                "El top-5 de v1 (= C1) sobre 4 de las 30 consultas, elegidas una por "
                "categoría más una repetida."
            ),
            "n_consultas": N_CONSULTAS_DEL_PILOTO,
            "orden_de_turnos": list(self.orden_de_turnos()),
            "categoria_repetida": self.categoria_repetida,
            "por_que_esa_categoria_se_repite": (
                "Decisión, declarada: el piloto existe para refinar la rúbrica antes "
                "de gastar las ~540, y la sección que un piloto más veces manda a "
                "reescribir es el **desempate**. La categoría ambigua es la que menos se "
                "defiende sola —«con más vida», «más humano»—, así que repetirla estresa el "
                "desempate dos veces y las otras dos categorías quedan cubiertas igual. Es "
                "un parámetro: `--categoria-repetida` lo cambia y el pool lo declara."
            ),
            "ranking_dentro_de_la_categoria": (
                'sha256("<semilla>|<consulta_id>"), el menor primero '
                "(`identidad.rango_de_seleccion`, el mismo primitivo con el que el "
                "Particionador elige el held-out). No es un PRNG a propósito: la "
                "declaración tiene que alcanzar para re-derivar la selección, y el "
                "algoritmo de `random.sample` cambió entre versiones de CPython."
            ),
            "semilla": semilla,
            "de_donde_sale_la_semilla": (
                "la del `ManifiestoDeSplit`"
                if self.semilla is None
                else "declarada a mano con `--semilla`"
            ),
            "imagenes_distintas": self.imagenes_distintas,
            "por_que_imagenes_distintas": (
                "Los 20 juicios son el banco de pruebas de la rúbrica, y dos pedidos sobre "
                "la **misma** imagen comparten el instrumento visual: si la rúbrica falla "
                "solo con cierto tipo de obra, con 4 imágenes distintas se ve y con 3 puede "
                "no verse. el diseño de la evaluación pide *una categoría repetida*, no una imagen repetida, "
                "así que la restricción no contradice la regla: la desempata."
            ),
            "excluye": (
                "las consultas marcadas `es_ejemplo` en el `SetDeConsultas`: un ejemplo no "
                "es dato (el set de consultas / )."
            ),
            "elegidas": [
                {
                    "turno": n,
                    "consulta_id": c.consulta_id,
                    "categoria": c.categoria,
                    "artwork_id": c.artwork_id,
                    "pedido": c.pedido,
                    "rango": identidad.rango_de_seleccion(semilla, c.consulta_id)[:16],
                }
                for n, c in enumerate(elegidas, 1)
            ],
        }

@dataclass
class ResultadoDeDerivación:
    pool: Pool
    ruta: Path
    ruta_de_procedencia: Path
    ya_estaba: bool
    otros_del_split: tuple[str, ...] = ()
    corridas_descartadas: dict = None  # type: ignore[assignment]

class ConstructorDePool:
    def __init__(self, cfg, *, almacén, rutas, observabilidad=None) -> None:
        self.cfg = cfg
        self.almacén = almacén
        self.rutas = rutas
        self.obs = observabilidad

    def derivar(
        self,
        *,
        designacion: str,
        set_de_consultas: SetDeConsultas,
        manifiesto,
        corridas: Sequence[Any],
        regla: Regla | None = None,
        estado: str = "piloto",
        configuracion_esperada: str | None = None,
        config_id_esperado: str | None = None,
        configuraciones_esperadas: Sequence[str] | None = None,
        config_ids_esperados: Mapping[str, str] | None = None,
    ) -> ResultadoDeDerivación:
        filas = self._filas(estado, configuracion_esperada, configuraciones_esperadas)
        config_ids = self._config_ids(filas, config_id_esperado, config_ids_esperados)
        regla = regla or (
            ReglaDeTodasLasConsultas() if estado == "congelado" else ReglaDeSelección()
        )
        ident = identidad.identidad_del_pool(
            split_id=set_de_consultas.split_id, designacion=designacion
        )
        if set_de_consultas.split_id != manifiesto.split_id:
            raise ErrorDeConstrucción(
                f"el set ancla a `split_id = {set_de_consultas.split_id}` y el manifiesto "
                f"leído es {manifiesto.split_id}. El pool hereda el ancla del set: un pool "
                f"sobre otro split es una orden de trabajo sobre un held-out que no se midió."
            )

        existentes = Pool.existentes(self.almacén, self.rutas.pool)
        if ident in existentes:
            pool = Pool.leer(self.almacén, self.rutas.pool, ident)
            return ResultadoDeDerivación(
                pool=pool,
                ruta=self.almacén.ruta(Pool.ruta_de(self.rutas.pool, ident)),
                ruta_de_procedencia=self.almacén.ruta(
                    Pool.ruta_de_procedencia(self.rutas.pool, ident)
                ),
                ya_estaba=True,
                corridas_descartadas={},
            )

        if not set_de_consultas.es_usable_para_medir:
            raise ErrorDeConstrucción(
                f"{set_de_consultas.identidad}: `estado = {set_de_consultas.estado}` y un "
                f"set que no es `completo` **no es usable para medir**. Un pool "
                f"derivado de una plantilla gastaría anotación humana sobre pedidos que "
                f"nadie escribió para medir."
            )

        if estado == "congelado" and len(filas) < 2:
            raise ErrorDeConstrucción(
                f"se pidió `estado = congelado` con **una sola** configuración de origen "
                f"({list(filas)}). el diseño de la evaluación abre con eso: *\"no se anota todo el "
                f"corpus **ni solo los resultados de una configuración** — eso sesgaría la "
                f"comparación a favor de la configuración anotada\"*. Un pool congelado es la "
                f"unión de los top-5 de **todas** las configuraciones de la ablación. El pool "
                f"del piloto se declara `piloto`, que es lo que lo vuelve no confundible con "
                f"uno definitivo (el relevamiento de cifras )."
            )

        usables, descartadas, por_fila = self._corridas_usables(corridas, filas, config_ids)
        elegidas = regla.elegir(
            set_de_consultas, semilla=regla.semilla_efectiva(manifiesto)
        )
        self._exigir_cobertura(
            elegidas=elegidas, usables=usables, filas=filas, config_ids=config_ids,
            descartadas=descartadas, por_fila=por_fila,
        )
        if estado == "congelado":
            self._exigir_alfa_congelado(elegidas=elegidas, usables=usables, filas=filas)

        configuraciones, origen, items = self._armar_items(
            elegidas=elegidas,
            usables=usables,
            filas=filas,
            semilla=regla.semilla_efectiva(manifiesto),
        )

        procedencia = self._armar_procedencia(
            identidad_del_pool=ident,
            designacion=designacion,
            estado=estado,
            set_de_consultas=set_de_consultas,
            manifiesto=manifiesto,
            regla=regla,
            elegidas=elegidas,
            configuraciones=configuraciones,
            origen=origen,
            items=items,
            descartadas=descartadas,
            por_fila=por_fila,
        )
        pool = Pool(procedencia=procedencia, items=items)
        pool.validar()

        ruta_csv = self.almacén.escribir_texto(
            Pool.ruta_de(self.rutas.pool, ident), pool.a_texto()
        )
        ruta_json = self.almacén.escribir_json(
            Pool.ruta_de_procedencia(self.rutas.pool, ident), procedencia
        )
        return ResultadoDeDerivación(
            pool=pool,
            ruta=ruta_csv,
            ruta_de_procedencia=ruta_json,
            ya_estaba=False,
            otros_del_split=tuple(
                p for p in Pool.del_split(self.almacén, self.rutas.pool, pool.split_id)
                if p != ident
            ),
            corridas_descartadas=descartadas,
        )

    @staticmethod
    def _filas(
        estado: str,
        configuracion_esperada: str | None,
        configuraciones_esperadas: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if configuraciones_esperadas is not None:
            filas = tuple(configuraciones_esperadas)
        elif configuracion_esperada is not None:
            filas = (configuracion_esperada,)
        else:
            filas = CONFIGURACIONES_POR_ESTADO.get(estado, ("C1",))
        if len(set(filas)) != len(filas):
            raise ErrorDeConstrucción(
                f"las configuraciones de origen {list(filas)} tienen una repetida. La unión "
                f"del diseño de la evaluación es sobre configuraciones **distintas**: contar una "
                f"dos veces inflaría su aporte y haría que sus ítems no fueran exclusivos de "
                f"nada."
            )
        if not filas:
            raise ErrorDeConstrucción(
                "no se declaró ninguna configuración de origen. Un pool es la unión de los "
                "top-5 de al menos una."
            )
        return filas

    @staticmethod
    def _config_ids(
        filas: Sequence[str],
        config_id_esperado: str | None,
        config_ids_esperados: Mapping[str, str] | None,
    ) -> dict[str, str]:
        if config_ids_esperados is not None:
            declarados = dict(config_ids_esperados)
        elif config_id_esperado is not None:
            if len(filas) != 1:
                raise ErrorDeConstrucción(
                    f"se pasó un `config_id_esperado` solo y {len(filas)} configuraciones "
                    f"({list(filas)}). Con más de una va `config_ids_esperados`: un id para "
                    f"todas sería la misma configuración cinco veces."
                )
            declarados = {filas[0]: config_id_esperado}
        else:
            from tp3_rag.modulo_c.metricas import config_ids_de_la_tabla

            declarados = config_ids_de_la_tabla()
        faltan = [f for f in filas if not declarados.get(f)]
        if faltan:
            raise ErrorDeConstrucción(
                f"no hay `config_id` declarado para {faltan} (hay para "
                f"{sorted(declarados)}). Cada fila de la tabla de ablación lo saca de **su capa** "
                f"(`config/c0.toml … c4.toml`, `metricas.ARCHIVO_POR_FILA`): sin la capa, el "
                f"pool se armaría por la etiqueta, que es el defecto que cerró — y acá "
                f"lo que se gasta es anotación humana irrecuperable."
            )
        return {f: declarados[f] for f in filas}

    def _corridas_usables(
        self,
        corridas: Sequence[Any],
        filas: Sequence[str],
        config_ids: Mapping[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict, dict]:
        descartadas = {
            "otra_configuracion": 0,
            "otro_config_id": 0,
            "ad_hoc": 0,
            "con_degradacion": 0,
            "fallidas": 0,
            "ordinales_viejos": 0,
            "salida_incompleta": 0,
        }
        de_las_filas = {config_ids[f]: f for f in filas}
        etiquetas = set(filas)
        usables: dict[str, dict[str, Any]] = {f: {} for f in filas}
        por_fila: dict[str, dict[str, int]] = {
            f: dict.fromkeys(descartadas, 0) for f in filas
        }
        for corrida in corridas:
            etiqueta = (corrida.contenido.get("pipeline") or {}).get("configuracion")
            if etiqueta not in etiquetas:
                descartadas["otra_configuracion"] += 1
                continue
            if corrida.config_id != config_ids[etiqueta]:
                descartadas["otro_config_id"] += 1
                por_fila[etiqueta]["otro_config_id"] += 1
                continue
            fila = de_las_filas[corrida.config_id]

            def _descartar(motivo: str, fila: str = fila) -> None:
                descartadas[motivo] += 1
                por_fila[fila][motivo] += 1

            if not corrida.es_del_set:
                _descartar("ad_hoc")
                continue
            if corrida.fallida is not None:
                _descartar("fallidas")
                continue
            if corrida.degradaciones:
                _descartar("con_degradacion")
                continue
            if not self._salida_cubre_el_top5(corrida):
                _descartar("salida_incompleta")
                continue
            previa = usables[fila].get(corrida.consulta_id)
            if previa is not None:
                _descartar("ordinales_viejos")
                if previa.ordinal > corrida.ordinal:
                    continue
            usables[fila][corrida.consulta_id] = corrida
        return usables, descartadas, por_fila

    @staticmethod
    def _exigir_cobertura(
        *,
        elegidas: Sequence[ConsultaDelSet],
        usables: Mapping[str, Mapping[str, Any]],
        filas: Sequence[str],
        config_ids: Mapping[str, str],
        descartadas: Mapping[str, int],
        por_fila: Mapping[str, Mapping[str, int]],
    ) -> None:
        faltan = {
            fila: [c.consulta_id for c in elegidas if c.consulta_id not in usables[fila]]
            for fila in filas
        }
        faltan = {f: v for f, v in faltan.items() if v}
        if not faltan:
            return
        detalle = "\n".join(
            f"    · {f} (config_id {config_ids[f]}): faltan {len(v)} de {len(elegidas)} "
            f"→ {v[:5]}{' …' if len(v) > 5 else ''}  ·  descartadas: "
            f"{ {k: n for k, n in por_fila[f].items() if n} }"
            for f, v in faltan.items()
        )
        raise ErrorDeConstrucción(
            f"no hay corrida **limpia** para todas las consultas elegidas:\n{detalle}\n"
            f"  totales descartados: { {k: n for k, n in descartadas.items() if n} }\n"
            f"  🔑 La regla **no se ajusta a lo que hay en disco**: si se eligiera *entre las "
            f"que corrieron bien*, la selección dejaría de ser la del diseño de la evaluación y el pool "
            f"declararía una regla que no se siguió. Corré `poetry run tp3-responder set "
            f"--config config/<fila>.toml` y volvé."
        )

    @staticmethod
    def _exigir_alfa_congelado(
        *,
        elegidas: Sequence[ConsultaDelSet],
        usables: Mapping[str, Mapping[str, Any]],
        filas: Sequence[str],
    ) -> None:
        provisorias = []
        for fila in filas:
            for consulta in elegidas:
                corrida = usables[fila][consulta.consulta_id]
                resuelta = (corrida.contenido.get("configuracion_resuelta") or {}).get(
                    "consulta"
                ) or {}
                if not resuelta.get("alfa_congelado", False):
                    provisorias.append(
                        f"{fila}/{consulta.consulta_id} → {corrida.corrida_id} "
                        f"(ordinal {corrida.ordinal}, α={resuelta.get('alfa')})"
                    )
        if not provisorias:
            return
        raise ErrorDeConstrucción(
            f"{len(provisorias)} de las {len(filas) * len(elegidas)} corridas elegidas "
            f"declaran `alfa_congelado = false`, y el pool se pidió **congelado**:\n"
            + "\n".join(f"    · {p}" for p in provisorias[:10])
            + (f"\n    … y {len(provisorias) - 10} más" if len(provisorias) > 10 else "")
            + "\n el diseño de la evaluación manda armar el pool con **C0–C4 ya "
            "congeladas**: una corrida pre-congelamiento puede traer otro top-5 —el "
            "reranker no es determinista— y lo que se gastaría sobre él es "
            "anotación humana irrecuperable.\n"
            "  ⚠️ `alfa_congelado` **no está en el `config_id`** a propósito (si estuviera, "
            "congelar habría huerfanado las corridas del propio barrido), así que filtrar por "
            "`config_id` no alcanza y este chequeo es lo que cubre el hueco. Corré la fila de "
            "nuevo con la configuración congelada, o derivá el pool como `piloto`."
        )

    @staticmethod
    def _salida_cubre_el_top5(corrida) -> bool:
        obras = {o["artwork_id"] for o in ((corrida.salida or {}).get("obras") or ())}
        return bool(corrida.top5) and set(corrida.top5) <= obras

    def _armar_items(
        self,
        *,
        elegidas: Sequence[ConsultaDelSet],
        usables: Mapping[str, Mapping[str, Any]],
        filas: Sequence[str],
        semilla: int,
    ) -> tuple[list[dict], list[dict], tuple[ÍtemDePool, ...]]:
        configuraciones: dict[str, dict] = {}
        origen: list[dict] = []
        items: list[ÍtemDePool] = []
        nro_de_fila = 0
        for consulta in elegidas:
            candidatas: dict[str, list[dict]] = {}
            fichas: dict[str, dict] = {}
            for fila in filas:
                corrida = usables[fila][consulta.consulta_id]
                pipeline = corrida.contenido.get("pipeline") or {}
                sellos = corrida.contenido.get("sellos") or {}
                entrada = configuraciones.setdefault(
                    fila,
                    {
                        "configuracion": fila,
                        "config_id": corrida.config_id,
                        "variante_id": sellos.get("variante_id"),
                        "split_id": sellos.get("split_id"),
                        "commit": sellos.get("commit"),
                        "cadena": list(pipeline.get("cadena") or ()),
                        "alfa": self._alfa_de(pipeline),
                        "alfa_congelado": bool(
                            (corrida.contenido.get("configuracion_resuelta") or {})
                            .get("consulta", {})
                            .get("alfa_congelado", False)
                        ),
                        "corridas": [],
                    },
                )
                entrada["corridas"].append(corrida.corrida_id)

                obras = {
                    o["artwork_id"]: o for o in ((corrida.salida or {}).get("obras") or ())
                }
                vistas: set[str] = set()
                for rank, artwork_id in enumerate(corrida.top5, 1):
                    if artwork_id in vistas:
                        continue
                    vistas.add(artwork_id)
                    candidatas.setdefault(artwork_id, []).append(
                        {
                            "consulta_id": consulta.consulta_id,
                            "artwork_id": artwork_id,
                            "configuracion": fila,
                            "config_id": corrida.config_id,
                            "corrida_id": corrida.corrida_id,
                            "rank": rank,
                            "score": obras[artwork_id].get("score"),
                        }
                    )
                    fichas.setdefault(
                        artwork_id, _ficha_de(artwork_id, obras[artwork_id])
                    )
                self._exigir_fichas_consistentes(
                    consulta_id=consulta.consulta_id, fila=fila, obras=obras, fichas=fichas
                )

            mezcladas = sorted(
                candidatas,
                key=lambda aid: identidad.rango_de_seleccion(
                    semilla, f"{consulta.consulta_id}|{aid}"
                ),
            )
            for artwork_id in mezcladas:
                nro_de_fila += 1
                items.append(
                    ÍtemDePool(
                        fila=nro_de_fila,
                        consulta_id=consulta.consulta_id,
                        artwork_id=artwork_id,
                        ficha=fichas[artwork_id],
                    )
                )
                origen.extend(candidatas[artwork_id])
        return [configuraciones[f] for f in filas], origen, tuple(items)

    @staticmethod
    def _exigir_fichas_consistentes(
        *, consulta_id: str, fila: str, obras: Mapping[str, Any], fichas: Mapping[str, dict]
    ) -> None:
        distintas = {
            aid: (fichas[aid], otra)
            for aid, obra in obras.items()
            if aid in fichas and (otra := _ficha_de(aid, obra)) != fichas[aid]
        }
        if not distintas:
            return
        aid, (primera, otra) = next(iter(distintas.items()))
        campos = [k for k in primera if primera[k] != otra[k]]
        raise ErrorDeConstrucción(
            f"{consulta_id}: la ficha de {aid} que publica {fila} no es la que ya había "
            f"copiado otra configuración — difieren en {campos}.\n"
            f"    ya copiada: { {k: primera[k] for k in campos} }\n"
            f"    {fila:<11}: { {k: otra[k] for k in campos} }\n"
            f"  La ficha se copia **por valor** porque la corrida es clase 2 y no va al repo "
            f", así que esta copia es la única que sobrevive: con dos versiones, cuál "
            f"gana lo decidiría el orden de las filas. Es del lado del pool —*dos ramas "
            f"produciendo el mismo tipo con contenidos distintos*— y acá lo que cuelga es el "
            f"juicio humano sobre esa obra."
        )

    @staticmethod
    def _alfa_de(pipeline: Mapping[str, Any]) -> float | None:
        for etapa in pipeline.get("etapas") or ():
            if "alfa" in etapa:
                return etapa["alfa"]
        return None

    def _armar_procedencia(
        self,
        *,
        identidad_del_pool: str,
        designacion: str,
        estado: str,
        set_de_consultas: SetDeConsultas,
        manifiesto,
        regla: Regla,
        elegidas: Sequence[ConsultaDelSet],
        configuraciones: Sequence[dict],
        origen: Sequence[dict],
        items: Sequence[ÍtemDePool],
        descartadas: Mapping[str, int],
        por_fila: Mapping[str, Mapping[str, int]],
    ) -> dict:
        semilla = regla.semilla_efectiva(manifiesto)
        return {
            "artefacto": "Pool",
            "version_del_pool": dpool.VERSIÓN_DEL_POOL,
            "clase_de_identidad": 3,
            "identidad": identidad_del_pool,
            "split_id": set_de_consultas.split_id,
            "designacion": designacion,
            "nota_de_identidad": dpool.nota_de_identidad(),
            "estado": estado,
            "nota_de_estado": dpool.nota_de_estado(estado),
            "configuraciones": list(configuraciones),
            "semilla_de_mezcla": semilla,
            "regla": regla.a_json(semilla=semilla, elegidas=elegidas),
            "origen": list(origen),
            "nota_de_ceguera": dpool.nota_de_ceguera(configuraciones),
            "conteos": _conteos(
                estado=estado, items=items, origen=origen, configuraciones=configuraciones
            ),
            "corridas_descartadas": dict(descartadas),
            "corridas_descartadas_por_configuracion": {
                f: dict(d) for f, d in por_fila.items()
            },
            "nota_de_corridas_descartadas": (
                "`con_degradacion` es la regla de qué corridas agregan a la tabla del lado de la anotación: un top-5 degradado no "
                "es el de la configuración declarada, y acá el costo de dejarlo pasar es "
                "trabajo humano irrecuperable. `ordinales_viejos` es el muestreo fijado, registrado y con su variabilidad medida: hay dos corridas "
                "de la misma consulta a propósito, y se toma el último ordinal **por "
                "(`config_id`, consulta)** —el `corrida_id` elegido está en el origen, así que "
                "*cuál* es citable—. `otra_configuracion` cuenta por etiqueta y es global: una "
                "corrida de una fila que este pool no pidió no se le imputa a ninguna."
            ),
            "cita_al_set_de_consultas": {
                "identidad": set_de_consultas.identidad,
                "autoria": set_de_consultas.autoria,
                "estado": set_de_consultas.estado,
                "nota": (
                    "El **pedido** y la **categoría** de cada consulta NO se copian acá: "
                    "viven en el `SetDeConsultas` y se citan por "
                    "`consulta_id`. Lo que sí se copia por valor es la **ficha de cada "
                    "obra**, porque su único otro hogar es la corrida —clase 2, no va al "
                    "repo y no se reproduce— (la regla de qué se versiona, el argumento de la "
                    "RevisiónDeExplicaciones)."
                ),
            },
            "procedencia": {
                "fecha_de_derivacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": commit_actual(),
                "capas_de_configuracion": list(self.cfg.capas),
                "top_k": int(self.cfg["consulta.top_k"]),
                "manifiesto": {
                    "split_id": manifiesto.split_id,
                    "corpus_indexado": manifiesto.conteos["corpus_indexado"],
                    "held_out": manifiesto.conteos["held_out"],
                },
            },
        }

def _conteos(
    *,
    estado: str,
    items: Sequence[ÍtemDePool],
    origen: Sequence[dict],
    configuraciones: Sequence[dict],
) -> dict:
    nombres = [c["configuracion"] for c in configuraciones]
    pares = frozenset(i.par for i in items)
    por_consulta: dict[str, int] = {}
    for i in items:
        por_consulta[i.consulta_id] = por_consulta.get(i.consulta_id, 0) + 1
    aporte = dpool.aporte_por_configuracion(origen, nombres)
    duplicados = round(len(items) * dpool.FRACCION_DUPLICADA_DE_A5)
    return {
        "n_items": len(items),
        "n_consultas": len(por_consulta),
        "n_obras_distintas": len({i.artwork_id for i in items}),
        "n_juicios_por_anotador": len(items),
        "aporte_por_configuracion": aporte,
        "exclusivos_por_configuracion": dpool.exclusivos_por_configuracion(origen, nombres),
        "suma_de_los_top5_sin_deduplicar": sum(aporte.values()),
        "deduplicados": sum(aporte.values()) - len(items),
        "items_por_consulta": {
            "minimo": min(por_consulta.values(), default=0),
            "maximo": max(por_consulta.values(), default=0),
            "media": round(len(items) / len(por_consulta), 2) if por_consulta else 0,
        },
        "items_del_top5_sin_juzgar_por_configuracion": dpool.items_del_top5_sin_juzgar(
            origen, pares, nombres
        ),
        "duplicado_del_20_por_ciento_de_a5": duplicados,
        "minutos_por_anotador_al_ritmo_de_docs_05": round(
            len(items) * dpool.SEGUNDOS_POR_JUICIO / 60
        ),
        "nota": (
            "`n_items` **es** el presupuesto de anotación por persona, y hacen falta **dos** "
            "personas porque κ necesita dos. "
            + (
                "Son los 20 del piloto: **no cuentan contra los ~540** de la campaña "
                "definitiva y se descartan si la rúbrica cambia."
                if estado != "congelado"
                else "Es la campaña definitiva: `n_items` es el número que reemplaza al "
                "**~540 proyectado** del diseño de la evaluación / los criterios de entrega, que suponía ~18 obras "
                "únicas por consulta. `items_por_consulta.media` es lo que la unión dio de "
                "verdad: cuanto más se solapan los top-5 de C0–C4, menos juicios cuesta el "
                "pool — y el solapamiento es un resultado, no un supuesto."
            )
        ),
        "nota_de_la_union": (
            "`aporte_por_configuracion` suma más que `n_items`: la diferencia **es** la "
            "deduplicación del diseño de la evaluación. `exclusivos_por_configuracion` cuenta los "
            "ítems que trajo **solo** esa configuración, que es la medida de cuánto cambia esa "
            "etapa el conjunto a juzgar. `items_del_top5_sin_juzgar_por_configuracion` es la "
            "*regla innegociable* del diseño de la evaluación computada en el momento en "
            "que tiene que dar **cero**: todo par del top-5 de una configuración de origen está "
            "en el pool por construcción, y se computa igual porque *un invariante que se "
            "cumple por construcción deja de cumplirse el día que la construcción cambia* "
            "por construcción."
        ),
        "nota_del_duplicado": (
            "el diseño de la evaluación manda anotar el **20 %** del pool por "
            "duplicado para el κ del acuerdo entre anotadores. ⚠️ **Es un número, no un subconjunto elegido**: "
            "*cuáles* ítems se duplican es una decisión del protocolo, y hoy "
            "`tp3-anotar exportar` proyecta el pool **entero** por anotador — dos planillas "
            "son el 100 % duplicado, no el 20 %."
        ),
    }

def _ficha_de(artwork_id: str, obra: Mapping[str, Any]) -> dict:
    ficha = obra.get("ficha") or {}
    return {
        "obra": ficha.get("title") or "",
        "artista": ficha.get("artist") or "",
        "cultura": ficha.get("culture") or "",
        "anio": ficha.get("year") if ficha.get("year") is not None else "",
        "museo": artwork_id.split(":")[0],
        "source_url": obra.get("source_url") or "",
    }

