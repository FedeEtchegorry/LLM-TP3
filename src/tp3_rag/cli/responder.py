from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.consulta import EntradaDeConsulta
from tp3_rag.dominio.corrida import AlmacenDeCorridas, Corrida
from tp3_rag.dominio.plan import AlmacenDePlanes, PlanDeConsulta, resumen_de_pl3
from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.modulo_b.armador import ArmadorDePipeline
from tp3_rag.modulo_b.ejecutor import EjecutorDeEtapas
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos import trazas as trazas_mod

LÍNEA = "─" * 78

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-responder",
        description=(
            "Módulo B — corre el camino de consulta y sella la corrida (la tabla de identidades, clase "
            "2). En la cadena es la de C1 = v1: imagen + mensaje crudo al text encoder de "
            "CLIP → artworks → evidencia de knowledge por artwork_id → generación citada."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="clave=valor",
        help="override de una clave declarada; una clave desconocida es un error",
    )
    p.add_argument("--variante-id", help="cuál variante, si hay más de una registrada (la identificación de variantes)")
    p.add_argument("--autoria", help="cuál SetDeConsultas, si hay más de uno completo")
    p.add_argument("--silencioso", action="store_true")

    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser(
        "set", help="correr las 30 consultas del SetDeConsultas — es **las 30 consultas de punta a punta**"
    )
    s.add_argument(
        "--consulta", action="append", default=[], metavar="consulta_id",
        help="correr solo estas (para iterar sin gastar las 30)",
    )
    s.add_argument(
        "--categoria", help="correr solo una categoría (visual_cromatico | conceptual_cultural | ambiguo)"
    )

    a = sub.add_parser(
        "adhoc",
        help="una consulta ad-hoc 🔴 su traza NO captura entradas (el registro de decisiones la política de trazado)",
    )
    a.add_argument("--imagen", type=Path, required=True)
    a.add_argument("--pedido", required=True)

    m = sub.add_parser("mostrar", help="leer y mostrar una corrida ya sellada")
    m.add_argument("--corrida", help="el `corrida_id`; si falta, lista las que hay")
    return p.parse_args(argv)

def main(argv=None) -> int:  # noqa: C901 - es un entrypoint: lee, orquesta y muestra
    args = _argumentos(argv)
    overrides = {}
    for par in args.overrides:
        if "=" not in par:
            print(f"--set espera clave=valor, llegó {par!r}", file=sys.stderr)
            return 2
        clave, _, valor = par.partition("=")
        overrides[clave.strip()] = valor

    obs = Observabilidad(silencioso=args.silencioso)
    montaje = None
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
        corridas = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])

        if args.comando == "mostrar":
            return _mostrar(args, corridas, obs)

        credenciales = Credenciales.desde_entorno()
        obs.trazas = trazas_mod.armar(cfg, credenciales)

        ArmadorDePipeline(cfg).validar()

        montaje = mnt.montar(
            cfg,
            almacén,
            variante_id=args.variante_id,
            observabilidad=obs,
            credenciales=credenciales,
        )
        _cabecera(cfg, montaje, obs, credenciales)

        armador = ArmadorDePipeline(
            cfg,
            puerto_de_imagen=montaje.puerto_de_imagen,
            indice=montaje.indice,
            puerto_de_llm=montaje.puerto_de_llm,
            embebedor_de_texto=montaje.embebedor_de_texto,
            almacen_de_planes=AlmacenDePlanes(almacén, cfg["rutas.artefactos_planes"]),
            bytes_de_la_imagen=mnt.bytes_del_corpus(cfg, almacén),
            observabilidad=obs,
        )
        pipeline = armador.armar()
        ejecutor = EjecutorDeEtapas(
            cfg,
            pipeline=pipeline,
            almacen_de_corridas=corridas,
            observabilidad=obs,
            variante_id=montaje.variante_id,
            split_id=montaje.split_id,
        )
        _cadena(pipeline, obs)

        if args.comando == "adhoc":
            resultado = _adhoc(args, ejecutor, obs)
        else:
            resultado = _del_set(args, cfg, almacén, montaje, ejecutor, obs)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if montaje is not None:
            montaje.cerrar()
        if hasattr(obs.trazas, "cerrar"):
            obs.trazas.cerrar()
    return resultado

def _cabecera(cfg, montaje, obs, credenciales) -> None:
    obs.decir(f"variante: {montaje.variante_id} (clase 1 — la identificación de variantes)")
    obs.decir(
        f"split: {montaje.split_id} (clase 3 — congelado)\n"
        f"             corpus {montaje.manifiesto.conteos['corpus_indexado']} · "
        f"held-out {montaje.manifiesto.conteos['held_out']}"
    )
    congelado = bool(cfg["consulta.alfa_congelado"])
    obs.decir(
        f"α: {cfg['consulta.alfa']}   "
        f"{'← **congelado**' if congelado else '← 🔴 **PROVISORIO**'}"
    )
    if not congelado:
        obs.decir(
            " el diseño de la evaluación: el `α` es **uno solo para C1–C4** y sale del barrido\n"
            ", que no puede cerrar antes de que C3 y C4 existan. El tag\n"
            "             `v1` es un **hito de orden**, no la fuente de la fila C1 — que se\n"
            " computa en con el `α` congelado. Cada corrida lo declara."
        )
    else:
        obs.decir(
            " ⚠️ La fila **C1** de la tabla se computa en ** con este `α`**, y\n"
            "             **no sale del tag `v1`**, que corrió con el provisorio (0,5 sin\n"
            "             congelar). `v1` es un hito de orden y **no se re-etiqueta**."
        )
    obs.decir(
        f"N: {cfg['consulta.n_candidatos']} candidatos ← la cantidad de candidatos a rerankear, "
        f"**CERRADO** por (10/20/50 → 20).\n"
        f" En C1–C2 no mueve el top-5; en C3–C4 sí, y en C4 es `el criterio de costo = 2 + N`"
    )
    if "planificador" in tuple(cfg["consulta.etapas"]):
        obs.decir(
            f"planificador: {cfg['bindings.planificador.modelo']} ← el registro de decisiones\n"
            f"             prompt {cfg['bindings.planificador.prompt_id']} "
            f"v{cfg['bindings.planificador.version_del_prompt']} · "
            f"muestreo T={cfg['bindings.planificador.muestreo.temperature']} "
            f"seed={cfg['bindings.planificador.muestreo.seed']}\n"
            f" 🔴 los tres son **ingredientes del `plan_id`**: cambiarlos\n"
            f"             produce otro artefacto y el barrido de `α` vuelve a pagar las 30"
        )
    obs.decir(
        f"generador: {cfg['bindings.generador.modelo']} ← el registro de decisiones "
        f"(prefijo `openai/` obligatorio)\n"
        f"             proveedor {cfg['proveedor.base_url']}  "
        f"(key de `{credenciales.variable_de_la_key}`)"
    )
    obs.decir(
        f"trazas: {cfg['trazas.proveedor']} · proyecto {cfg['trazas.proyecto']!r}   "
        f"← el registro de decisiones decide **qué** se traza, y no es configurable"
    )
    obs.decir(f"evidencia: {cfg['evidencia.n_fragmentos_por_obra']} fragmentos/obra · orden {cfg['evidencia.orden']}")

def _cadena(pipeline, obs) -> None:
    obs.decir(f"\n{LÍNEA}\ncadena · {pipeline.configuracion} (la precondición declarada la validó antes de correr)\n{LÍNEA}")
    for i, etapa in enumerate(pipeline.etapas, start=1):
        from tp3_rag.modulo_b.etapa import CLAVE_DE_ETAPA

        marca = "🔒" if not etapa.DEGRADA else "  "
        obs.decir(f"  {i}. {marca} {etapa.NOMBRE:<26} → {CLAVE_DE_ETAPA[etapa.NOMBRE]}")
    obs.decir(
        "\n 🔒 = **no degrada**: el `recuperador_visual` es el piso y el\n"
        " `renderizador` falla porque sus condiciones son **obligaciones** (las obligaciones de la cita).\n"
        " Las otras dos son omitibles en runtime, y eso es la omisión de etapas en runtime."
    )
    if pipeline.plan_de_falla:
        obs.decir(f"\n plan de falla: {pipeline.plan_de_falla}")
    obs.decir(LÍNEA)

def _adhoc(args, ejecutor, obs) -> int:
    if not args.imagen.exists():
        print(f"no existe {args.imagen}", file=sys.stderr)
        return 2
    entrada = EntradaDeConsulta.ad_hoc(
        datos_de_imagen=args.imagen.read_bytes(), pedido=args.pedido
    )
    política = entrada.política_de_trazado
    obs.decir(
        f"\n🔴 **el registro de decisiones · procedencia `ad_hoc`** → la traza **NO captura entradas**:\n"
        f"   ni el pedido ni la imagen viajan a LangSmith. Lo que viaja es estructura,\n"
        f"   latencia y tokens. `captura_entradas = {política.captura_entradas}`.\n"
        f"   La política se **deriva** de la procedencia: no hay flag ni variable de\n"
        f" entorno que la cambie (los detectores estructurales: *un control que requiere que alguien se\n"
        f"   acuerde no es un control*)."
    )
    corrida = ejecutor.correr(entrada)
    _mostrar_corrida(corrida, obs)
    return 0 if corrida.fallida is None else 1

def _del_set(args, cfg, almacén, montaje, ejecutor, obs) -> int:
    set_ = mnt.set_de_consultas(cfg, almacén, montaje.manifiesto, args.autoria)
    consultas = list(set_.consultas())
    if args.consulta:
        pedidas = set(args.consulta)
        consultas = [c for c in consultas if c.consulta_id in pedidas]
        faltan = sorted(pedidas - {c.consulta_id for c in consultas})
        if faltan:
            print(f"el set {set_.identidad} no tiene {faltan}", file=sys.stderr)
            return 2
    if args.categoria:
        consultas = [c for c in consultas if c.categoria == args.categoria]

    obs.decir(
        f"\nset: {set_.identidad} (clase 3 — {set_.estado})\n"
        f"             {len(set_.consultas())} consultas · {len(consultas)} a correr\n"
        f"             criterio de `culture`: {set_.criterio_de_culture}"
    )
    obs.decir(
        "\n🔴 **el registro de decisiones · procedencia `set`** → estas consultas se trazan\n"
        "   **completas** (entradas, salidas, tokens, latencia por etapa). No son dato de\n"
        "   un tercero: son obras de nuestro propio held-out, versionadas como\n"
        " **instrumento de medición**."
    )

    hechas, fallidas, espontáneas, planificadas = [], [], [], []
    for i, consulta in enumerate(consultas, start=1):
        datos = mnt.bytes_de_la_imagen_del_set(cfg, almacén, consulta)
        entrada = EntradaDeConsulta.del_set(consulta, datos_de_imagen=datos)
        obs.progreso(
            f"{i}/{len(consultas)}  {consulta.consulta_id:<18} [{consulta.categoria}]"
        )
        corrida = ejecutor.correr(entrada)
        hechas.append(corrida)
        if corrida.fallida is not None:
            fallidas.append(corrida)
        if corrida.degradaciones_no_planificadas:
            espontáneas.append(corrida)
        elif corrida.degradaciones:
            planificadas.append(corrida)
    obs.cerrar_progreso()
    _marcador(hechas, fallidas, espontáneas, planificadas, len(consultas), obs)
    return 0 if not fallidas else 1

def _marcador(hechas, fallidas, degradadas, planificadas, total: int, obs) -> None:
    obs.decir(f"\n{LÍNEA}\n**las 30 consultas de punta a punta** · las 30/30 end-to-end\n{LÍNEA}")
    con_cinco = sum(1 for c in hechas if len(c.top5) == 5)
    con_explicacion = sum(
        1
        for c in hechas
        if c.salida and any(o.get("lineas") for o in (c.salida.get("obras") or ()))
    )
    obs.decir(
        f"\n corridas sin error: {len(hechas) - len(fallidas):>3}/{total}\n"
        f" con **5 obras**: {con_cinco:>3}/{total} ← `las 30 consultas de punta a punta` y el criterio de respuesta garantizada\n"
        f" con explicación citada: {con_explicacion:>3}/{total} ← `las 30 consultas de punta a punta`\n"
        f" degradadas **espontáneas**: {len(degradadas):>3}     "
        f"← la regla de qué corridas agregan a la tabla las saca de la tabla C0–C4\n"
        f"  degradadas **planificadas**: {len(planificadas):>3}     "
        f"← el plan de falla de la regla de qué corridas agregan a la tabla: es el criterio de respuesta garantizada y se reporta aparte\n"
        f" fallidas por obligación: {len(fallidas):>3} ← las obligaciones de la cita: el criterio de atribución es umbral duro"
    )
    if planificadas:
        inyectadas = sorted(
            {d["etapa"] for c in planificadas for d in c.degradaciones if d["planificada"]}
        )
        con_cinco_obras = sum(1 for c in planificadas if len(c.top5) == 5)
        obs.decir(
            f"\n el criterio de respuesta garantizada · falla inyectada en {inyectadas}:\n"
            f"    {con_cinco_obras}/{len(planificadas)} devolvieron **5 obras** con su "
            f"fallback declarado.\n"
            f" los criterios de entrega el criterio de respuesta garantizada compromete *30/30 devuelven 5 obras* contra falla\n"
            f"    inyectada en **las etapas post-retrieval que tienen fallback** — el piso\n"
            f" y el renderizador **no** están en esa lista (la asimetría del piso de recuperación, las obligaciones de la cita)."
        )
    descartadas = sum(len(c.afirmaciones_descartadas) for c in hechas)
    obs.decir(
        f" afirmaciones descartadas: {descartadas:>3}      "
        f"← el descarte contado de afirmaciones, y es **lo que `el conteo de referencias colgadas` cuenta**"
    )
    _plan(hechas, obs)
    for c in degradadas:
        for d in c.degradaciones_no_planificadas:
            obs.decir(f"    ⚠️  {c.consulta_id}: [{d['etapa']}] {d['causa'][:120]}")
    for c in fallidas:
        obs.decir(f"    ✗  {c.consulta_id}: {c.fallida}")
    obs.decir(
        f"\n  corridas en artifacts/corridas/ · los validadores leen de ahí:\n"
        f"    poetry run tp3-validar todo"
    )
    _costo(hechas, obs)
    obs.decir(LÍNEA)

def _plan(hechas, obs) -> None:
    planes = [PlanDeConsulta.desde_artefacto(c.plan) for c in hechas if c.plan]
    if not planes:
        return
    pl3 = resumen_de_pl3(planes)
    reusados = sum(1 for c in hechas if c.plan_reusado)
    distintos = len({p.plan_id for p in planes})
    configuración = hechas[0].contenido["pipeline"]["configuracion"]
    obs.decir(f"\n{LÍNEA}\nlas técnicas avanzadas aisladas por medición · el plan como evidencia — las tres cláusulas de las consultas del plan y la escritura única del plan\n{LÍNEA}")
    obs.decir(
        f"\n  población: {len(planes)} planes citados por {len(hechas)} corridas de "
        f"{configuración}\n\n"
        f" las tres cláusulas de las consultas del plan · las **tres** cláusulas, y son tres cosas distintas:\n"
        f" las dos consultas **no vacías**: {pl3['no_vacias']:>3}/{pl3['de']}\n"
        f" **≠ pedido crudo** (*query rewriting*): "
        f"{pl3['distintas_del_pedido_crudo']:>3}/{pl3['de']}\n"
        f" **≠ entre sí** (*query expansion*): "
        f"{pl3['distintas_entre_si']:>3}/{pl3['de']}\n"
        f" las tres a la vez: {pl3['las_tres']:>3}/{pl3['de']}\n"
        f"\n la escritura única del plan · escritura única, reuso registrado:\n"
        f" `plan_id` distintos: {distintos:>3}\n"
        f" **producidos**: {len(planes) - reusados:>3} ← pagaron la llamada "
        f"que la tabla de tokens, costo y la figura cuenta\n"
        f" **reusados**: {reusados:>3} ← el reuso del plan al barrer el peso de fusión: el barrido de `α` no "
        f"re-ejecuta el pre-retrieval"
    )
    if reusados == len(planes) and len(planes):
        obs.decir(
            f"\n 🔑 **Las {reusados} reusaron el plan, y eso es el reuso del plan al barrer el peso de fusión más allá del barrido "
            f"de `α`.**\n"
            f" lo que no entra en la identidad del plan deja la cadena **afuera** del `plan_id`, así que {configuración} "
            f"reusa los planes\n"
            f"  que pagó la configuración anterior: el pre-retrieval es **una llamada por "
            f"consulta\n  y por pedido**, no por fila de la tabla. ⚠️ Y entonces el costo de "
            f"esta corrida\n  **no** es el costo de la fila: la fila lo paga una vez. Va "
            f"dicho en la tabla de la tabla de tokens, costo y la figura."
        )

def _acumular(hechas, clave: str) -> dict[str, dict]:
    acumulado: dict[str, dict] = {}
    for c in hechas:
        for grupo, d in (c.contenido.get(clave) or {}).items():
            acum = acumulado.setdefault(
                grupo, {"n_llamadas": 0, "tokens_in": 0, "tokens_out": 0, "segundos": 0.0}
            )
            for k in acum:
                acum[k] += d.get(k, 0)
    return acumulado

def _costo(hechas, obs) -> None:
    if not hechas:
        return
    por_rol = _acumular(hechas, "libro_mayor")
    por_etapa = _acumular(hechas, "libro_mayor_por_etapa")
    llamadas = {"C0": "0", "C1": "1", "C2": "2", "C3": "2", "C4": "2 + N"}
    configuración = hechas[0].contenido["pipeline"]["configuracion"]
    reusados = sum(1 for c in hechas if c.plan_reusado)
    obs.decir(f"\n{LÍNEA}\nla tabla de tokens, costo y la figura · tokens por rol — medidos, no estimados (los tokens en el artefacto de corrida)\n{LÍNEA}")
    obs.decir(
        f"\n  población: {len(hechas)} consultas de {configuración} · "
        f"{llamadas.get(configuración, '?')} llamada(s) al LLM por consulta "
        f"(el diseño de la arquitectura, el criterio de costo)"
        + (
            f"\n             ⚠️ {reusados} corrida(s) **reusaron** el plan, así que no "
            f"pagaron\n la llamada del planificador: eso es el reuso del plan al barrer el peso de fusión en la tabla "
            f"de costo."
            if reusados
            else ""
        )
        + "\n"
    )
    def _tabla(titulo: str, acumulado: dict[str, dict]) -> tuple[int, int]:
        obs.decir(
            f"  {titulo:<26}{'llamadas':>10}{'tokens in':>12}{'tokens out':>12}"
            f"{'segundos':>10}"
        )
        for grupo, d in sorted(acumulado.items()):
            obs.decir(
                f"  {grupo:<26}{d['n_llamadas']:>10}{d['tokens_in']:>12}"
                f"{d['tokens_out']:>12}{d['segundos']:>10.1f}"
            )
        t_in = sum(d["tokens_in"] for d in acumulado.values())
        t_out = sum(d["tokens_out"] for d in acumulado.values())
        obs.decir(f"  {'TOTAL':<26}{'':>10}{t_in:>12}{t_out:>12}")
        return t_in, t_out

    total_in, total_out = _tabla("por **rol** (la atribución de cada llamada a su rol)", por_rol)
    obs.decir("")
    etapa_in, etapa_out = _tabla("por **etapa** (la tabla de tokens, costo y la figura)", por_etapa)
    obs.decir(
        f"\n  las dos lecturas cierran: {'✓ sí' if (total_in, total_out) == (etapa_in, etapa_out) else '✗ NO — modo de falla 1'}"
        f"  ({total_in} = {etapa_in} in · {total_out} = {etapa_out} out)\n"
        f" 🔴 **la tabla de tokens, costo y la figura pide *por etapa* y la atribución de cada llamada a su rol atribuye *por rol***, y en C1–C3 son **1:1**.\n"
        f"  En C4 dejan de serlo, y no por el reranker —que es N llamadas de **un** rol y\n"
        f"  **una** etapa— sino por `embedding_texto`, que es **un rol en dos etapas**: la\n"
        f" rama de texto y el recolector de evidencia. Las dos tablas son\n"
        f"  **proyecciones del mismo libro mayor**, no dos acumuladores."
    )
    if hechas:
        obs.decir(
            f"\n  por consulta: {total_in // len(hechas)} in · {total_out // len(hechas)} out\n"
            f" el registro de decisiones midió **1.946 tokens de entrada por par de imágenes** con\n"
            f"  `gpt-4.1-mini` (medición a mano, 2 imágenes) — **y esa es exactamente la\n"
            f"  forma de la llamada del reranker**, así que la fila `reranker` de arriba es\n"
            f"  la que se compara contra ese número. Las del planificador y del generador\n"
            f"  llevan **una** imagen y no son comparables con él."
        )
        if "reranker" in por_rol:
            d = por_rol["reranker"]
            llamadas = max(1, d["n_llamadas"])
            obs.decir(
                f"\n 💲 **El costo de C4, y es lo que el criterio de costo pide medir desde la primera "
                f"corrida:**\n"
                f"  reranking: {d['n_llamadas']} llamadas · {d['tokens_in']} tokens in · "
                f"{d['tokens_in'] // llamadas} por llamada\n"
                f" (contra los **1.946** que el registro de decisiones midió a mano con el mismo par de\n"
                f"  imágenes; si esto se despega de ese número, el modelo o el `detail`\n"
                f"  cambiaron y **hay que volver a medir antes de barrer N**).\n"
                f"  ⚠️ **USD: ausente.** `[proveedor]` solo declara el precio de\n"
                f"  `text-embedding-3-small`, y *un modelo sin precio da ausente, no cero*\n"
                f" (, ). Lo que está medido son los **tokens**."
            )
        if "planificador" in por_rol:
            obs.decir(
                "\n 🔑 **El `embedding_texto` no es una llamada al LLM y el criterio de costo no lo cuenta**\n"
                ": el orden de evidencia por defecto cuesta un embedding por\n"
                " consulta y la rama de texto otro, y aparecen en esta tabla porque la tabla de tokens, costo y la figura sí\n"
                " los mide. Las llamadas que el criterio de costo cuenta son las de los roles "
                "`planificador`,\n  `reranker` y `generador`."
            )

def _mostrar(args, corridas, obs) -> int:
    if not args.corrida:
        existentes = corridas.existentes()
        if not existentes:
            print("no hay ninguna corrida todavía", file=sys.stderr)
            return 1
        for cid in existentes:
            obs.decir(cid)
        return 0
    corrida = corridas.leer(args.corrida)
    _mostrar_corrida(corrida, obs)
    return 0

def _mostrar_corrida(corrida: Corrida, obs) -> None:
    obs.decir(
        f"\n{LÍNEA}\nCorrida · {corrida.corrida_id}  (clase 2 — ordinal {corrida.ordinal})\n{LÍNEA}"
    )
    obs.decir(
        f"\n consulta_id: {corrida.consulta_id}\n"
        f" config_id: {corrida.config_id}\n"
        f" cadena: {corrida.cadena}\n"
        f" la forma declarada del artefacto: {'✓' if corrida.contenido['invariante_et6']['cumple_et6'] else '✗'}"
        f"   (top5 verificado contra "
        f"{corrida.contenido['invariante_et6']['top5'].get('contra')})\n"
        f" agrega a la tabla C0–C4: "
        f"{'✓ sí' if corrida.agrega_a_la_tabla else '✗ no (degradó)'}"
    )
    árbol = corrida.contenido["arbol_de_trazas"]
    obs.decir(
        f"\n trazas: {árbol['n_nodos']} nodos · {árbol['por_tipo']}\n"
        f" etapas trazadas: {árbol['etapas_trazadas']}\n"
        f" campos censurados por el registro de decisiones: {árbol['campos_censurados']}"
    )
    if corrida.degradaciones:
        obs.decir("\n  degradaciones:")
        for d in corrida.degradaciones:
            marca = "planificada" if d["planificada"] else "**no planificada**"
            obs.decir(f"    [{d['etapa']}] {marca} · {d['causa'][:140]}")
    if corrida.afirmaciones_descartadas:
        obs.decir(
            f"\n afirmaciones descartadas (el descarte contado de afirmaciones, lo que el conteo de referencias colgadas cuenta): "
            f"{len(corrida.afirmaciones_descartadas)}"
        )
    if corrida.fallida:
        obs.decir(f"\n  🔴 **fallida** por obligación: {corrida.fallida}")
    if corrida.salida:
        obs.decir(f"\n{LÍNEA}\nSalida citada\n{LÍNEA}\n")
        obs.decir(corrida.salida["render"])
    obs.decir(LÍNEA)

if __name__ == "__main__":
    raise SystemExit(main())
