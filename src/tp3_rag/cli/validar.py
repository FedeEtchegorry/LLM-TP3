from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.corrida import AlmacenDeCorridas
from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.modulo_c import metricas as met
from tp3_rag.modulo_c import validadores as val
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.cliente_http import ClienteHTTP

LÍNEA = "─" * 78
N_SELF_RETRIEVAL = 20

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-validar",
        description=(
            "ValidadoresEjecutables: el sanity check de self-retrieval con α = 1, y "
            "trazabilidad, referencias colgadas, atribución e idioma leídos de las "
            "corridas sin llamar al LLM."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="clave=valor"
    )
    p.add_argument("--variante-id", help="cuál variante, si hay más de una registrada (la identificación de variantes)")
    p.add_argument("--autoria", help="cuál SetDeConsultas, si hay más de uno completo")
    p.add_argument(
        "--n", type=int, default=N_SELF_RETRIEVAL,
        help=f"obras del self-retrieval (el sanity check de self-retrieval compromete {N_SELF_RETRIEVAL})",
    )
    p.add_argument(
        "--sin-red", action="store_true",
        help=(
            "no resolver los `source_url` del criterio de trazabilidad. **No los da por buenos**: los reporta como "
            "*no verificados* y el criterio de trazabilidad no pasa"
        ),
    )
    p.add_argument("--silencioso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)
    for nombre, ayuda in (
        ("self-retrieval", "el sanity check: 20/20 con α = 1. Carga CLIP; no llama al LLM"),
        (
            "criterios",
            "trazabilidad, referencias colgadas, atribución e idioma sobre las "
            "corridas, más las dos cláusulas del plan si tienen plan. Sin llamar al LLM",
        ),
        ("todo", "los dos, que es lo que hay que correr antes de etiquetar `v1`"),
    ):
        sub.add_parser(nombre, help=ayuda)
    r11 = sub.add_parser(
        "r11", help="el reuso del plan al barrer el peso de fusión · barrido de α sobre una consulta: 1 plan_id / 5 corrida_id"
    )
    r11.add_argument(
        "--consulta", required=True, metavar="consulta_id",
        help="la consulta sobre la que se barrió `α`",
    )
    r11.add_argument(
        "--alfas", type=int, default=5,
        help="cuántos valores de `α` tiene el grid (el diseño de la evaluación: 0/0.25/0.5/0.75/1 → 5)",
    )
    sub.add_parser(
        "cultura",
        help=(
            " · ¿el plan nombra la **cultura pedida** o la de la imagen subida? "
            "**Por versión del prompt**: es el antes/después del informe"
        ),
    )
    r12 = sub.add_parser(
        "r12",
        help=(
            "🔴 el muestreo fijado, registrado y con su variabilidad medida · la variabilidad medida: la misma configuración **dos veces**, con el "
            "muestreo fijado y registrado. **Antes de interpretar cualquier delta**"
        ),
    )
    r12.add_argument(
        "--rol",
        default="reranker",
        help="qué rol se compara (reranker | generador | planificador). Default: reranker",
    )
    sub.add_parser(
        "p6",
        help=(
            "🔴 el criterio de costo · las llamadas al LLM por consulta y por configuración, medidas del libro "
            "mayor. **Las cinco filas juntas**: C0 = 0, C1 = 1, C2 = C3 = 2, C4 = 2 + N"
        ),
    )
    anclaje = sub.add_parser(
        "anclaje",
        help=(
            "🔴 · cuánto anclaje a la imagen cuesta cada `α` del barrido. **NO es "
            "el sanity check de self-retrieval**, que va pinchado en α = 1. Local: cero llamadas al proxy"
        ),
    )
    anclaje.add_argument(
        "--alfas",
        default="0,0.25,0.5,0.75,1",
        help="el grid del diseño de la evaluación, separado por comas",
    )
    sub.add_parser(
        "variantes",
        help=(
            "🔴 la identificación de variantes + el reuso del caché de imagen entre variantes · las variantes coexistiendo e identificables, y **0 llamadas "
            "nuevas al PuertoEmbeddingImagen** en la segunda. Lee `artifacts/indices/`: "
            "sin Chroma, sin CLIP y sin la key"
        ),
    )
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = _argumentos(argv)
    overrides = {}
    for par in args.overrides:
        if "=" not in par:
            print(f"--set espera clave=valor, llegó {par!r}", file=sys.stderr)
            return 2
        clave, _, valor = par.partition("=")
        overrides[clave.strip()] = valor

    obs = Observabilidad(silencioso=args.silencioso)
    resultados = []
    montaje = None
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)

        if args.comando in ("self-retrieval", "todo", "anclaje"):
            montaje = mnt.montar(
                cfg,
                almacén,
                variante_id=args.variante_id,
                observabilidad=obs,
                credenciales=Credenciales.desde_entorno(),
                con_clip=True,
                con_llm=False,
                con_embeddings_de_texto=False,
            )
            obs.decir(
                f"variante: {montaje.variante_id}\n"
                f"split: {montaje.split_id} · corpus "
                f"{montaje.manifiesto.conteos['corpus_indexado']} · held-out "
                f"{montaje.manifiesto.conteos['held_out']}"
            )
            comunes = dict(
                puerto_de_imagen=montaje.puerto_de_imagen,
                indice=montaje.indice,
                manifiesto=montaje.manifiesto,
                bytes_de=mnt.bytes_del_corpus(cfg, almacén),
                n=args.n,
            )
            if args.comando == "anclaje":
                for alfa in [float(a) for a in str(args.alfas).split(",") if a.strip()]:
                    resultados.append(val.anclaje_a_la_imagen(**comunes, alfa=alfa))
            else:
                resultados.append(val.self_retrieval(**comunes, alfa=1.0))

        if args.comando == "r11":
            resultados.append(_r11(args, cfg, almacén, obs))

        if args.comando == "cultura":
            resultados.append(_cultura(args, cfg, almacén, obs))

        if args.comando == "r12":
            resultados.append(_r12(args, cfg, almacén, obs))

        if args.comando == "p6":
            resultados.append(_p6(args, cfg, almacén, obs))

        if args.comando == "variantes":
            resultados.extend(_variantes(args, cfg, almacén, obs))

        if args.comando in ("criterios", "todo"):
            resultados.extend(_criterios(args, cfg, almacén, obs, montaje))
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if montaje is not None:
            montaje.cerrar()

    _mostrar(resultados, obs)
    return 0 if all(r.pasa for r in resultados) else 1

def _r11(args, cfg, almacén, obs):
    corridas_de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    todas = []
    for cid in corridas_de.existentes():
        corrida = corridas_de.leer(cid)
        if corrida.consulta_id == args.consulta and corrida.plan:
            todas.append(corrida)
    if not todas:
        raise ErrorDelProyecto(
            f"no hay corridas **con plan** para la consulta {args.consulta!r} en "
            f"{cfg['rutas.artefactos_corridas']}. el reuso del plan al barrer el peso de fusión se verifica sobre el barrido ya "
            f"corrido; producilo con cinco invocaciones del entrypoint, una por valor del "
            f"grid del diseño de la evaluación:\n"
            f"    for a in 0 0.25 0.5 0.75 1; do \\\n"
            f"      poetry run tp3-responder --config config/c2.toml \\\n"
            f"        --set consulta.alfa=$a set --consulta {args.consulta}; done"
        )
    obs.decir(
        f"corridas: {len(todas)} de `{args.consulta}` con plan, de "
        f"{len({c.config_id for c in todas})} `config_id` distintos"
    )
    return val.r11_un_plan_cinco_corridas(
        todas, consulta_id=args.consulta, alfas_esperados=int(args.alfas)
    )

def _r12(args, cfg, almacén, obs):
    corridas = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"]).del_config(
        cfg.config_id
    )
    if not corridas:
        raise ErrorDelProyecto(
            f"no hay corridas con `config_id = {cfg.config_id}` en "
            f"{cfg['rutas.artefactos_corridas']}. el muestreo fijado, registrado y con su variabilidad medida se mide sobre **dos** corridas de la "
            f"misma configuración y la misma consulta, así que hay que producirlas:\n"
            f"    poetry run tp3-responder --config config/c4.toml set\n"
            f"    poetry run tp3-responder --config config/c4.toml set   # el ordinal 2"
        )
    ordinales = sorted({c.ordinal for c in corridas})
    obs.decir(
        f"corridas: {len(corridas)} de `config_id` {cfg.config_id} "
        f"({cfg['consulta.configuracion']}) · ordinales {ordinales}\n"
        f" ⚠️ **sin quedarse con el último**: el muestreo fijado, registrado y con su variabilidad medida compara los dos, y es lo "
        f"contrario\n             de lo que hace `criterios` (ahí promediarlos contaría dos "
        f"veces los mismos ítems)"
    )
    return val.r12_variabilidad(corridas, rol=args.rol)

def _p6(args, cfg, almacén, obs):
    lectura = met.corridas_de_la_tabla(
        AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    )
    corridas = lectura["corridas"]
    if not corridas:
        raise ErrorDelProyecto(
            f"no hay ninguna corrida de las **cinco filas declaradas** en "
            f"{cfg['rutas.artefactos_corridas']} (hay {lectura['en_disco']} en disco, de "
            f"otros `config_id`). el criterio de costo se mide **del libro mayor de las corridas**, "
            f"o sea del `usage` que devolvió la API: primero hay que correrlas."
        )
    obs.decir(
        f"corridas: {len(corridas)} de las **cinco filas declaradas**, de "
        f"{lectura['en_disco']} en disco · {lectura['por_fila']}\n"
        f"             ⚠️ se agrupa por **`config_id`** y no por la etiqueta: "
        f"{lectura['huerfanas']} de un hito\n             anterior y {lectura['de_p7']} de "
        f"el criterio de respuesta garantizada quedaron afuera (**modo de falla 3**)\n"
        + (
            f"             ⚠️ y {lectura['ad_hoc']} **ad-hoc** con el `config_id` de una fila: "
            f"no son de las 30 del set ()\n"
            if lectura["ad_hoc"]
            else ""
        )
        + f"N: {cfg['consulta.n_candidatos']} candidatos ← la cantidad de candidatos a rerankear, "
        f"**CERRADO** por (10/20/50 → 20). Es la `N` de `C4 = 2 + N`"
    )
    if lectura["faltan"]:
        obs.decir(
            f" 🔴 **sin corridas de {lectura['faltan']}**: el criterio de costo compromete las "
            f"**cinco** filas,\n             así que lo que esto publica es parcial y lo dice."
        )
    return val.p6_llamadas_al_llm(
        corridas, n_candidatos=int(cfg["consulta.n_candidatos"])
    )

def _variantes(args, cfg, almacén, obs):
    from tp3_rag.dominio.variante import RegistroDeÍndices

    raíz = cfg["rutas.artefactos_indices"]
    existentes = RegistroDeÍndices.existentes(almacén, raíz)
    if not existentes:
        raise ErrorDelProyecto(
            f"no hay ninguna variante registrada en {raíz}. la identificación de variantes y el reuso del caché de imagen entre variantes se verifican "
            f"**sobre el registro** (el layout de artefactos: va al repo), así que primero hay que "
            f"construir: `poetry run tp3-indexar`."
        )
    registros = [RegistroDeÍndices.leer(almacén, raíz, v) for v in existentes]
    obs.decir(
        f"variantes: {len(registros)} registradas en {raíz}\n"
        f"             {[r.variante_id for r in registros]}\n"
        f" ⚠️ **no se filtra por `--variante-id`**: la identificación de variantes pide varias "
        f"coexistiendo y\n el reuso del caché de imagen entre variantes pide *la segunda*. Fijar una haría que los dos "
        f"pasaran siempre."
    )
    return [
        val.r2_variantes_coexistiendo(registros),
        val.r3_cache_de_imagen_reusado(registros),
    ]

def _cultura(args, cfg, almacén, obs):
    from tp3_rag.dominio.plan import AlmacenDePlanes

    manifiesto = mnt.manifiesto_del_registro(
        cfg, almacén, mnt.registro_de_indices(cfg, almacén, args.variante_id)
    )
    set_ = mnt.set_de_consultas(cfg, almacén, manifiesto, args.autoria)
    culturas = {
        c.consulta_id: tuple(c.cultura_pedida.etiquetas)
        for c in set_.consultas()
        if c.cultura_pedida
    }
    planes = AlmacenDePlanes(almacén, cfg["rutas.artefactos_planes"])
    contenidos = [p.contenido_del_artefacto() for p in planes.todos()]
    if not contenidos:
        raise ErrorDelProyecto(
            f"no hay ningún plan en {cfg['rutas.artefactos_planes']}. El antes/después del "
            f"prompt se mide **sobre los planes**, que son el artefacto que las técnicas avanzadas aisladas por medición pide como "
            f"evidencia: producilos con `poetry run tp3-responder --config config/c2.toml "
            f"set --categoria conceptual_cultural`."
        )
    obs.decir(
        f"set: {set_.identidad} ({len(culturas)} pedidos culturales de "
        f"{len(set_.consultas())})\n"
        f"planes: {len(contenidos)} en {cfg['rutas.artefactos_planes']} "
        f"(**todas las versiones del prompt**: la versión es ingrediente del `plan_id`)"
    )
    return val.cultura_pedida_en_el_plan(
        contenidos,
        culturas_por_consulta=culturas,
        terminos_de_cultura={
            k: list(v) for k, v in dict(cfg["medicion.terminos_de_cultura"]).items()
        },
        marcadores_de_negacion=list(cfg["medicion.marcadores_de_negacion"]),
    )

def _criterios(args, cfg, almacén, obs, montaje):
    corridas_de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    todas = []
    for cid in corridas_de.existentes():
        corrida = corridas_de.leer(cid)
        if corrida.config_id == cfg.config_id:
            todas.append(corrida)
    if not todas:
        raise ErrorDelProyecto(
            f"no hay corridas con `config_id = {cfg.config_id}` en "
            f"{cfg['rutas.artefactos_corridas']}. el criterio de trazabilidad, `el conteo de referencias colgadas`, el criterio de atribución y el criterio de idioma se verifican "
            f"**contra el artefacto** y no corriendo el pipeline (el módulo de evaluación: *leen almacenes; "
            f"no recalculan nada*), así que primero hace falta la corrida: "
            f"`poetry run tp3-responder set`. ⚠️ Y el `config_id` tiene que ser **el mismo**: "
            f"validar corridas de otra configuración es publicar un número de una fila sobre "
            f"otra."
        )
    por_consulta: dict[str, object] = {}
    for corrida in todas:
        previa = por_consulta.get(corrida.consulta_id)
        if previa is None or corrida.ordinal > previa.ordinal:
            por_consulta[corrida.consulta_id] = corrida
    corridas = tuple(por_consulta.values())
    obs.decir(
        f"corridas: {len(corridas)} (último ordinal por consulta, de {len(todas)} en "
        f"disco) · config_id {cfg.config_id}"
    )

    manifiesto = (
        montaje.manifiesto
        if montaje is not None
        else mnt.manifiesto_del_registro(
            cfg, almacén, mnt.registro_de_indices(cfg, almacén, args.variante_id)
        )
    )
    cliente = None
    if not args.sin_red:
        cliente = ClienteHTTP.desde_configuración(cfg, Credenciales.desde_entorno())
    return val.todos_sobre_corridas(
        corridas,
        ids_del_corpus=manifiesto.ids_corpus,
        exigen_atribucion={
            k: bool(v) for k, v in dict(cfg["salida.exigen_atribucion"]).items()
        },
        cliente_http=cliente,
        n_candidatos=int(cfg["consulta.n_candidatos"]),
    )

def _mostrar(resultados, obs) -> None:
    obs.decir(f"\n{LÍNEA}\nValidadoresEjecutables · los criterios como aserciones\n{LÍNEA}\n")
    for r in resultados:
        marca = "✓" if r.pasa else "✗"
        obs.decir(f"  {marca}  {r.criterio}")
        obs.decir(f"       {r.marcador}   población: {r.poblacion}")
        for clave, valor in r.detalle.items():
            if clave.startswith("nota"):
                continue
            obs.decir(f"       {clave}: {valor}")
        for falla in r.fallas[:5]:
            obs.decir(f"       ✗ {falla}")
        if len(r.fallas) > 5:
            obs.decir(f"       … y {len(r.fallas) - 5} más")
        obs.decir("")
    pasan = sum(1 for r in resultados if r.pasa)
    obs.decir(f"{LÍNEA}\n  {pasan}/{len(resultados)} criterios pasan\n{LÍNEA}")

if __name__ == "__main__":
    raise SystemExit(main())
