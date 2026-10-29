from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.corrida import VERSIÓN_DE_LA_CORRIDA, AlmacenDeCorridas, Corrida
from tp3_rag.modulo_b.armador import Pipeline
from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA, EstadoDeConsulta
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO
from tp3_rag.nucleo.errores import ErrorDelProyecto

MODOS_DE_FALLA: tuple[str, ...] = ("excepcion", "vacio")

class FallaInyectada(ErrorDelProyecto):
    pass

class EjecutorDeEtapas:
    def __init__(
        self,
        cfg,
        *,
        pipeline: Pipeline,
        almacen_de_corridas: AlmacenDeCorridas,
        observabilidad,
        variante_id: str,
        split_id: str,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.almacen = almacen_de_corridas
        self.obs = observabilidad
        self.variante_id = variante_id
        self.split_id = split_id
        self.commit = _commit()
        for etapa, modo in self.pipeline.plan_de_falla.items():
            if modo not in MODOS_DE_FALLA:
                raise ErrorDelProyecto(
                    f"`consulta.plan_de_falla.{etapa} = {modo!r}` no es uno de "
                    f"{list(MODOS_DE_FALLA)}. El plan de falla **es configuración** "
                    f"y un modo desconocido no puede caer en un default: el criterio de respuesta garantizada reportaría "
                    f"haber inyectado algo distinto de lo que inyectó."
                )

    def correr(self, entrada: EntradaDeConsulta) -> Corrida:
        corrida_id, ordinal, directorio = self.almacen.reservar(
            config_id=self.cfg.config_id,
            consulta_id=entrada.consulta_id,
            commit=self.commit,
        )
        sellos = {
            "corrida_id": corrida_id,
            "config_id": self.cfg.config_id,
            "variante_id": self.variante_id,
            "split_id": self.split_id,
            "commit": self.commit,
        }
        estado = EstadoDeConsulta(entrada=entrada, sellos=sellos)
        t0 = datetime.now(timezone.utc)
        marca = self.obs.marca()

        with self.obs.con_politica_de_trazado(entrada.política_de_trazado):
            with self.obs.nodo(
                f"corrida {self.pipeline.configuracion}",
                tipo="chain",
                entradas={"pedido": entrada.pedido, "consulta_id": entrada.consulta_id},
                metadata={
                    "corrida_id": corrida_id,
                    "config_id": self.cfg.config_id,
                    "variante_id": self.variante_id,
                    "configuracion": self.pipeline.configuracion,
                    "alfa": float(self.cfg["consulta.alfa"]),
                    "alfa_congelado": bool(self.cfg["consulta.alfa_congelado"]),
                    "procedencia": entrada.procedencia,
                },
            ) as nodo_corrida:
                estado = self._correr_cadena(estado)
                nodo_corrida.salidas["render"] = (
                    estado.salida.render() if estado.salida is not None else ""
                )
                nodo_corrida.metadata["degradaciones"] = len(estado.degradaciones)
                nodo_corrida.metadata["fallida"] = bool(estado.fallida)

        et6 = estado.verificar_et6(self.pipeline.cadena, top_k=self.pipeline.top_k)
        contenido = self._sellar(
            estado,
            corrida_id=corrida_id,
            ordinal=ordinal,
            sellos=sellos,
            et6=et6,
            segundos=(datetime.now(timezone.utc) - t0).total_seconds(),
            inicio=t0,
            politica=entrada.política_de_trazado,
            marca=marca,
        )
        self.almacen.escribir(corrida_id, contenido)
        corrida = Corrida(contenido)
        corrida.validar()
        return corrida

    def _correr_cadena(self, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        for etapa in self.pipeline.etapas:
            if estado.fallida:
                estado = estado.degradar(
                    etapa.NOMBRE,
                    causa=f"no se corrió: la corrida ya está marcada fallida en "
                    f"`{estado.fallida['etapa']}`",
                    politica="no corre",
                )
                continue
            with self.obs.nodo(
                etapa.NOMBRE,
                tipo="chain",
                metadata={
                    "etapa": etapa.NOMBRE,
                    "clave": CLAVE_DE_ETAPA[etapa.NOMBRE],
                    "degrada": etapa.DEGRADA,
                },
            ):
                estado = self._correr_etapa(etapa, estado)
        return estado

    def _correr_etapa(self, etapa, estado: EstadoDeConsulta) -> EstadoDeConsulta:
        modo = self.pipeline.plan_de_falla.get(etapa.NOMBRE)
        if modo is not None:
            return estado.degradar(
                etapa.NOMBRE,
                causa=f"falla **inyectada** por `consulta.plan_de_falla` (modo {modo!r})",
                politica=etapa.POLÍTICA,
                planificada=True,
            )
        try:
            nuevo = etapa.correr(estado)
        except Exception as e:
            if not etapa.DEGRADA:
                raise
            return estado.degradar(
                etapa.NOMBRE,
                causa=f"{type(e).__name__}: {e}",
                politica=etapa.POLÍTICA,
            )
        return nuevo

    def _sellar(
        self,
        estado: EstadoDeConsulta,
        *,
        corrida_id: str,
        ordinal: int,
        sellos: Mapping[str, Any],
        et6: Mapping[str, Any],
        segundos: float,
        inicio: datetime,
        politica,
        marca,
    ) -> dict:
        retener_pedido = bool(self.cfg["retencion.pedido"])
        estado_json = estado.a_json(retener_pedido=retener_pedido)
        estado_json["claves_pobladas"] = sorted(estado.etapas_con_clave_poblada())
        return {
            "artefacto": "Corrida",
            "version_de_la_corrida": VERSIÓN_DE_LA_CORRIDA,
            "clase_de_identidad": 2,
            "corrida_id": corrida_id,
            "ordinal": ordinal,
            "config_id": self.cfg.config_id,
            "consulta_id": estado.entrada.consulta_id,
            "nota_de_identidad": (
                "Clase 2: *registro de una ejecución*. La identidad fija **las "
                "entradas, no la salida** —una salida no determinista con la misma entrada "
                "es el dato que el muestreo fijado, registrado y con su variabilidad medida mide—, así que lleva **ordinal** y la regla de que un id determina su contenido no aplica. "
                "El ordinal se **reserva** creando el directorio de forma exclusiva: `1 + n` "
                "es un *leer, después escribir* y el barrido es paralelo."
            ),
            "pipeline": self.pipeline.a_json(),
            "estado": estado_json,
            "invariante_et6": dict(et6),
            "agrega_a_la_tabla": not estado.degradaciones and estado.fallida is None,
            "nota_de_la_tabla": (
                "la tabla de degradación por etapa/la regla de qué corridas agregan a la tabla: una corrida con degradación **no planificada** no agrega a la "
                "tabla C0–C4 —*una C4 que degradó en silencio entraría etiquetada como C4 "
                "comportándose como C3*— y una con degradación **planificada** tampoco: es "
                "el criterio de respuesta garantizada y se reporta aparte. Una etapa **ausente por configuración** no "
                "degrada: no corre, y por eso C0–C3 sí agregan."
            ),
            "libro_mayor": self.obs.libro_mayor(marca),
            "libro_mayor_por_etapa": self.obs.libro_mayor_por_etapa(marca),
            "nota_de_ob2": (
                "*Los tokens se **persisten en el artefacto de corrida**, no solo en "
                "LangSmith*: si el servicio se cae, se pierde la herramienta de "
                "debugging, **no los números** que la tabla de tokens, costo y la figura y el criterio de costo publican. Y el árbol declara "
                "qué se trazó y qué no, que es el registro de decisiones hecho artefacto."
            ),
            "nota_de_a8_vs_bd4": (
                "🔴 **Las dos lecturas del libro mayor, y son proyecciones del mismo dato** "
                "(): la tabla de tokens, costo y la figura pide el costo *por etapa* y la atribución de cada llamada a su rol atribuye *por rol, no al "
                "nombre de la etapa*. En C1–C3 son **1:1**; en C4 el reranker es "
                "**N llamadas de un rol y una etapa** y `embedding_texto` es **un rol en dos "
                "etapas**. `libro_mayor` agrupa por rol —que sigue siendo la atribución de "
                "registro, declarada por el binding— y `libro_mayor_por_etapa` por la etapa "
                "en que el span corrió, **derivada del árbol de trazas y no de un "
                "parámetro**. Los totales coinciden por construcción: son dos `GROUP BY` de "
                "la misma lista de spans, no dos acumuladores — dos acumuladores serían el "
                "**modo de falla 1** sobre la cifra que la tabla de tokens, costo y la figura publica."
            ),
            "arbol_de_trazas": self.obs.arbol_de_trazas(politica, marca),
            "retencion": {
                "imagen": bool(self.cfg["retencion.imagen"]),
                "pedido": retener_pedido,
                "dias": int(self.cfg["retencion.dias"]),
                "nota": (
                    "el registro de decisiones (cierra la política de retención del dato del usuario): **no se persiste nada de un "
                    "tercero**. Aplica al **dato del usuario**, no al ciclo de vida del "
                    "artefacto: esta corrida conserva su `corrida_id`, su ordinal, sus "
                    "rankings, sus métricas y sus degradaciones, y **pierde** la imagen y el "
                    "pedido. Lo que se vacía son campos, y que estaban vaciados queda "
                    "registrado. Borrar la corrida reasignaría el ordinal y un informe que "
                    "cita la corrida #1 apuntaría a otra ejecución."
                ),
            },
            "latencia": {
                "segundos_de_la_corrida": round(segundos, 4),
                "por_etapa": {
                    n.nombre: round(n.segundos, 4)
                    for n in self.obs.nodos(marca)
                    if n.tipo == "chain" and n.nombre in CLAVE_DE_ETAPA
                },
                "nota": (
                    "🔑 **Esto acota el recuadro ⚠️ de la observabilidad.** Ahí se declaraba que *las "
                    "etapas puras no tienen span* —el `Fusionador` y el `RenderizadorDeSalida` "
                    "no llaman a ningún puerto— y que por lo tanto *prometer latencia por "
                    "etapa sobre-promete*. Instrumentando en el **`EjecutorDeEtapas`**, que es "
                    "el lugar que ese mismo recuadro señala, **sí hay latencia para las ocho**. "
                    "la latencia por etapa deja de estar sobre-prometido. **Va reportado.**"
                ),
            },
            "sellos": dict(sellos),
            "fecha": inicio.isoformat(timespec="seconds"),
            "configuracion_resuelta": self.cfg.a_dict(),
            "capas_de_configuracion": list(self.cfg.capas),
            "no_determinan_el_config_id": self.cfg.no_determinantes(),
            "alcance_del_config_id": list(self.cfg.alcance_del_config_id()),
            "nota_de_m9_3": (
                "🔴 **los hallazgos**: *el `config_id` se computa sobre las claves que **la "
                "cadena declarada lee**, no sobre el árbol entero*. había sacado el "
                "layout, el ruteo y el precio con un bloque **global**, y no alcanzó: agregar "
                "`[bindings.reranker]` movió el id de C1–C3 —cadenas que no tienen reranker— "
                "*justo antes de congelar*. El alcance de arriba son "
                "`CLAVES_DE_TODA_CORRIDA` (los ingredientes del `variante_id`, del "
                "`split_id`, la cadena, `top_k`, el plan de falla y la retención) más lo que "
                "declara cada etapa de **esta** cadena en `Etapa.CLAVES_DE_CONFIGURACIÓN`. Y "
                "no es una promesa: el `ArmadorDePipeline` **registra las lecturas** que hace "
                "al construir la cadena y rechaza la que leyó una clave que no declaró — "
                "porque la dirección peligrosa es incluir **de menos**, que deja dos corridas "
                "con el mismo id y contenido distinto."
            ),
            "nota_de_m5_1": (
                "🔴 **los hallazgos**: *toda identidad y toda comparación de contenido "
                "**declaran** su bloque de claves no determinantes*. El `config_id` hashea "
                "la configuración resuelta **menos** las claves de arriba —layout, "
                "parámetros de máquina, ruteo, transporte y el precio declarado—, que son "
                "**procedencia**: viajan enteras en `configuracion_resuelta` y no mueven la "
                "identidad. Antes de esta pasada el `config_id` hasheaba todo, y por eso se "
                "movió **dos veces** por claves que no cambian nada de lo que el pipeline "
                "produce, huerfanando las corridas las dos veces. Declararlo es la "
                "mitad de la decisión: sin esta clave, *«el `config_id` excluye el layout»* "
                "sería una afirmación de documento, y una afirmación de documento no es un "
                "control."
            ),
        }

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
    except Exception:  # pragma: no cover - un árbol sin git
        return None
