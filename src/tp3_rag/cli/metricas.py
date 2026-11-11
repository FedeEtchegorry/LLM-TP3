from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.corrida import AlmacenDeCorridas
from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.modulo_c import metricas as met
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.indice_vectorial import ARTWORKS

LÍNEA = "─" * 78

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-metricas",
        description=(
            "MétricasDeRanking (el módulo de evaluación): QS, cobertura cultural y el delta por categoría. "
            "Lee `artifacts/corridas/` y **no construye el pipeline ni pide la key**. "
            "⚠️ nDCG@5 y P@5 necesitan el pool y no están acá."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="clave=valor"
    )
    p.add_argument("--variante-id", help="cuál variante, si hay más de una (la identificación de variantes)")
    p.add_argument("--autoria", help="cuál SetDeConsultas, si hay más de uno completo")
    p.add_argument("--silencioso", action="store_true")
    sub = p.add_subparsers(dest="comando", required=True)
    sub.add_parser("qs", help="QS por imagen y total — `la evaluación §B`, sin anotación")
    sub.add_parser(
        "cobertura",
        help="cobertura cultural del top-5 en los pedidos culturales — la cobertura cultural del resultado",
    )
    o2 = sub.add_parser(
        "o2",
        help=(
            "🔑 el delta por categoría entre `--config` (B) y `--contra` (A). Para "
            "`C3−C2`: `--config config/c3.toml --contra config/c2.toml`"
        ),
    )
    o2.add_argument(
        "--contra",
        type=Path,
        required=True,
        help="la configuración **base** del delta (la A de `B − A`)",
    )
    costo = sub.add_parser(
        "costo",
        help=(
            "🔴 la tabla de tokens, costo y la figura · el costo **medido** por configuración (las dos lecturas: por rol y por "
            "etapa) y la **proyección para N ∈ {10, 20, 50}**, que es lo que necesita"
        ),
    )
    costo.add_argument(
        "--consultas",
        type=int,
        default=30,
        help="sobre cuántas consultas se proyecta (las 30 del set)",
    )
    n = sub.add_parser(
        "n",
        help=(
            "🔴 el barrido de `N` —la cantidad de candidatos a rerankear— con los **tres** "
            "valores derivados de una sola corrida de N = 50: QS, cobertura, churn, "
            "latencia y costo por N"
        ),
    )
    n.add_argument(
        "--valores",
        default="10,20,50",
        help="el grid de `la evaluación § Barridos`. El mayor tiene que ser el que se corrió",
    )
    generacion = sub.add_parser(
        "generacion",
        help=(
            "🔴 el **estudio secundario de preprocesamiento** del lado de la generación — "
            "los dos brazos de **campos de `knowledge`** o los dos de **chunking** "
            "(⚠️ ciego contra por campo): referencias por obra, de qué `section` salen, "
            "citas byte a byte, licencias mezcladas, tokens de evidencia por obra y el %% "
            "de ítems sin juzgar. **Cero llamadas al proxy**"
        ),
    )
    generacion.add_argument(
        "--contra",
        type=Path,
        required=True,
        help=(
            "la capa del brazo **base** (la A de `B − A`). Los dos brazos son "
            "`config/c3.toml` y lo único que los separa va por `--set`: "
            "`fragmentacion.campos_de_knowledge` para el eje de campos, "
            "`fragmentacion.estrategia` para el de chunking. **Uno de los dos, no los "
            "dos**: un delta sobre dos variables no dice de cuál es"
        ),
    )
    generacion.add_argument(
        "--pool",
        help=(
            "el `pool_id` contra el que se cuenta la **séptima columna de la tabla de ablación**. Si falta, "
            "el pool **congelado** del split"
        ),
    )
    tabla = sub.add_parser(
        "tabla",
        help=(
            "🔴  · la tabla de ablación · **la tabla C0–C4 completa** (nDCG@5, P@5, P@5 estricta, QS, "
            "groundedness, latencia p95, USD/consulta) con el desglose por categoría y el "
            "%% sin juzgar — y **publicada DOS VECES**: sobre los 379 ítems del consenso y "
            "sobre los 246 de acuerdo (el desempate). **Cero llamadas al proxy**"
        ),
    )
    tabla.add_argument(
        "--pool",
        help="el `pool_id` del ground truth. Si falta, el pool **congelado** del split",
    )
    tabla.add_argument(
        "--rubrica", default="v1", help="el `rubrica_id` del consenso (κ es **por rúbrica**)"
    )
    tabla.add_argument(
        "--por",
        default="ndcg_sobre_todas",
        choices=("ndcg_sobre_todas", "ndcg_sobre_definidas", "p5", "p5_estricta", "mrr"),
        help=(
            "con qué métrica se ordenan las filas para comparar las dos poblaciones. El "
            "default es el promedio que la **decisión del acuerdo entre anotadores** declara (*«c05-ambiguo no "
            "aporta a nDCG@5 y sí al denominador»*)"
        ),
    )
    tabla.add_argument(
        "--json",
        type=Path,
        help="dónde dejar la tabla entera como JSON (por omisión, no se escribe nada)",
    )
    tabla.add_argument(
        "--fila",
        action="append",
        default=[],
        metavar="FILA=CONFIG_ID",
        help=(
            "apuntar una fila a otro `config_id` (repetible). Es lo que reproduce la tabla "
            "del diseño anterior: `--fila C3=ce0910b50e3569e9 --fila C4=e30ca8c5003c614c` "
            "son las corridas con RRF"
        ),
    )
    tabla.add_argument(
        "--juicios-extra",
        type=Path,
        action="append",
        default=[],
        help=(
            "🔴 pooling incremental · un `AporteDeAnotador` (CSV) con juicios de pares que "
            "el consenso NO tiene. Se suman **solo a la población (a) completa** y solo "
            "para pares sin grado; la (b) restringida no los ve (no pasaron por acuerdo). "
            "Repetible. ⚠️ Extender el GT mueve el IDCG: las cinco filas se recomputan "
            "sobre la misma base"
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
    indice = None
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
        registro = mnt.registro_de_indices(cfg, almacén, args.variante_id)
        manifiesto = mnt.manifiesto_del_registro(cfg, almacén, registro)
        set_ = mnt.set_de_consultas(cfg, almacén, manifiesto, args.autoria)
        obs.decir(
            f"variante   : {registro.variante_id}\n"
            f"set        : {set_.identidad}  ({len(set_.consultas())} consultas)\n"
            f"criterio de `culture`: {set_.criterio_de_culture}   ← **va con el número** "
            f"(criterio declarado)"
        )
        if args.comando == "costo":
            return _costo(args, cfg, almacén, obs)
        if args.comando == "qs":
            return _qs(cfg, almacén, set_, obs)
        if args.comando == "generacion":
            return _generacion(args, cfg, almacén, set_, obs)
        if args.comando == "tabla":
            return _tabla(args, cfg, almacén, set_, obs)
        if args.comando == "cobertura":
            indice = mnt.abrir_indice(cfg, almacén, registro.variante_id)
            return _cobertura(cfg, almacén, set_, indice, obs)
        if args.comando == "n":
            indice = mnt.abrir_indice(cfg, almacén, registro.variante_id)
            return _n(args, cfg, almacén, set_, indice, obs)
        indice = mnt.abrir_indice(cfg, almacén, registro.variante_id)
        return _o2(args, cfg, almacén, set_, indice, obs)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if indice is not None:
            indice.cerrar()

def _corridas_que_agregan(cfg, almacén, obs, *, etiqueta: str):
    todas = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"]).del_config(
        cfg.config_id
    )
    if not todas:
        raise ErrorDelProyecto(
            f"no hay corridas con `config_id = {cfg.config_id}` ({etiqueta}) en "
            f"{cfg['rutas.artefactos_corridas']}. Producilas con `poetry run tp3-responder "
            f"--config …{mnt.flag_de_variante(cfg, almacén, None)} set`."
        )
    del_set = [c for c in todas if c.es_del_set]
    ad_hoc = len(todas) - len(del_set)
    por_consulta: dict[str, object] = {}
    for c in del_set:
        previa = por_consulta.get(c.consulta_id)
        if previa is None or c.ordinal > previa.ordinal:
            por_consulta[c.consulta_id] = c
    reparto = met.separar_por_9_5(tuple(por_consulta.values()))
    obs.decir(
        f"\n{etiqueta:<4} config_id {cfg.config_id} · {len(todas)} corridas en disco → "
        f"{len(por_consulta)} (del set, último ordinal)"
        + (f" · {ad_hoc} **ad-hoc** fuera de la población" if ad_hoc else "")
        + f"\n la regla de qué corridas agregan a la tabla: {len(reparto.agregan)} agregan · "
        f"{len(reparto.planificadas)} con degradación planificada (el criterio de respuesta garantizada) · "
        f"{len(reparto.espontaneas)} espontánea · {len(reparto.fallidas)} fallidas"
    )
    for causa in reparto.a_json()["causas_espontaneas"][:5]:
        obs.decir(f"     ⚠️  espontánea: {causa}")
    if not reparto.agregan:
        raise ErrorDelProyecto(
            f"ninguna corrida de {etiqueta} agrega a la tabla (la regla de qué corridas agregan a la tabla). Las que hay "
            f"degradaron, y *una C4 que degradó en silencio entraría etiquetada como C4 "
            f"comportándose como C3*: medir sobre ellas publicaría una fila que dice lo "
            f"que no es."
        )
    return reparto.agregan, reparto

def _indices_del_set(set_):
    categoria, imagen, culturas = {}, {}, {}
    for c in set_.consultas():
        categoria[c.consulta_id] = c.categoria
        imagen[c.consulta_id] = c.artwork_id
        if c.cultura_pedida:
            culturas[c.consulta_id] = tuple(c.cultura_pedida.etiquetas)
    return categoria, imagen, culturas

def _costo(args, cfg, almacén, obs) -> int:
    de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    _proxy(cfg, almacén, de, obs)
    lectura = met.corridas_de_la_tabla(de)
    corridas = lectura["corridas"]
    if not corridas:
        raise ErrorDelProyecto(
            f"no hay ninguna corrida de las **cinco filas declaradas** en "
            f"{cfg['rutas.artefactos_corridas']} (hay {lectura['en_disco']} en disco, de "
            f"otros `config_id`). El costo lo **emite el pipeline** (el dataset documentado y reproducible, los tokens en el artefacto de corrida): sale del "
            f"`usage` que devolvió la API y está adentro de cada corrida."
        )
    costos = met.costo_por_configuracion(
        corridas,
        precios_in=cfg.get("proveedor.precio_usd_por_millon_de_tokens") or {},
        precios_out=cfg.get("proveedor.precio_usd_por_millon_de_tokens_de_salida") or {},
    )
    obs.decir(
        f"\n{LÍNEA}\nla tabla de tokens, costo y la figura · el costo **medido**, por configuración (los tokens en el artefacto de corrida)\n{LÍNEA}\n"
    )
    obs.decir(
        f"  población: {len(corridas)} corridas de las **cinco filas declaradas**, de "
        f"{lectura['en_disco']} en disco\n"
        f"             {lectura['por_fila']}\n"
        f"             `config_id` por fila: {lectura['config_ids']}\n"
        f"             ⚠️ dos filtros, y los dos cambian la población:\n"
        f"             · **por `config_id`**, no por la etiqueta: {lectura['huerfanas']} "
        f"de un pipeline anterior\n y {lectura['de_p7']} del criterio de respuesta garantizada (plan de falla "
        f"declarado, que la regla de qué corridas agregan a la tabla manda reportar\n aparte) —la etiqueta "
        f"`C1` sobrevive a los cambios\n               de configuración y la identidad no, y "
        f"`n = 120` en una fila de 30 consultas\n               es una población que no "
        f"existe (**modo de falla 3**);\n"
        f"             · **último ordinal por consulta**: con el muestreo fijado, registrado y con su variabilidad medida hay dos corridas de la "
        f"misma\n               consulta a propósito, y sumarlas duplicaría el costo por "
        f"consulta;\n"
        f"             · 🔴 **y la procedencia**: {lectura['ad_hoc']} ad-hoc con el "
        f"`config_id` de una fila\n               declarada, que **no son de las 30 del "
        f"set**. Entraban, y una fila de 30\n               consultas publicaba `n = 31`.\n"
        + (
            f"             🔴 **sin corridas de {lectura['faltan']}**: la tabla de ablación pide **5 filas**, "
            f"así que esto es parcial.\n"
            if lectura["faltan"]
            else ""
        )
    )
    obs.decir(
        f"  {'config':<8}{'n':>4}{'N':>4}{'llamadas LLM':>14}{'tokens in':>12}"
        f"{'tokens out':>12}{'in/consulta':>13}"
    )
    for c in costos:
        obs.decir(
            f"  {c.configuracion:<8}{c.n_corridas:>4}{str(c.n_candidatos or '—'):>4}"
            f"{c.llamadas_al_llm:>14}{c.tokens_in:>12}{c.tokens_out:>12}"
            f"{(c.tokens_in // c.n_corridas if c.n_corridas else 0):>13}"
        )
    obs.decir(
        f"\n  🔴 el reuso del plan al barrer el peso de fusión · lo mismo **por consulta servida** —con el plan adentro—, que es "
        f"la\n columna de la tabla de la tabla de ablación. Difiere de la de arriba cuando hay planes "
        f"reusados:\n\n"
        f"  {'config':<8}{'n':>4}{'llam/cons':>11}{'in/consulta':>13}"
        f"{'planes reus.':>14}{'USD/consulta':>14}"
    )
    for c in costos:
        usd = c.usd_por_consulta
        obs.decir(
            f"  {c.configuracion:<8}{c.n_corridas:>4}"
            f"{(c.llamadas_al_llm_con_el_plan / c.n_corridas if c.n_corridas else 0):>11.2f}"
            f"{str(c.tokens_in_por_consulta_con_el_plan or '—'):>13}"
            f"{c.planes_reusados:>14}"
            f"{(f'{usd:.4f}' if usd is not None else 'ausente'):>14}"
        )
    obs.decir(
        "\n  🔴 **Los tokens son MEDIDOS; el USD es DERIVADO de un precio DECLARADO.**\n"
        "  Precio de lista de OpenAI leído el **2026-11-03** en\n"
        "  `https://developers.openai.com/api/docs/pricing` — **no** es la tarifa del proxy\n"
        "  de ITBA, que nadie puede leer (`/key/info` bloqueado para virtual keys).\n"
        "  *Un modelo sin precio da **ausente**, no cero* sigue vigente.\n"
        "  ⚠️ **Dos cifras del gasto total, y no se promedian**: derivar de los\n"
        "  artefactos a precio de lista da **USD 12,43** y el dashboard facturó **10,0654**.\n"
        "  El proxy factura un **19 % por debajo de lista** (base: lista; al revés son 23 %),\n"
        "  y la hipótesis es el descuento por **entrada cacheada** (0,10 contra 0,40\n"
        "  en `gpt-4.1-mini`, y el reranker repite el mismo prompt N veces por consulta);\n"
        "  **no es verificable con nuestros artefactos**: el libro mayor guarda\n"
        "  `prompt_tokens` y no `cached_tokens`."
    )
    for c in costos:
        if not c.por_rol.get("reranker"):
            continue
        obs.decir(f"\n{LÍNEA}\nla cantidad de candidatos a rerankear · la proyección de N, para \n{LÍNEA}\n")
        proyección = met.proyeccion_de_n(c, n_consultas=int(args.consultas))
        obs.decir(
            f"  medido con **N = {proyección['medido_con_n']}** · "
            f"{proyección['llamadas_del_reranker_medidas']} llamadas de reranking · "
            f"**{proyección['tokens_in_por_llamada_del_reranker']} tokens in por llamada**\n"
            f"  (contra los **1.946** que el registro de decisiones midió a mano con el mismo par de "
            f"imágenes)\n"
        )
        obs.decir(
            f"  {'N':>4}{'llamadas/consulta':>20}{'in/consulta':>14}"
            f"{'in sobre las ' + str(args.consultas):>18}{'reranking':>12}{'':>4}"
        )
        for n, d in proyección["por_n"].items():
            marca = "← **medido**" if d["medido"] else "   proyectado"
            obs.decir(
                f"  {n:>4}{d['llamadas_al_llm_por_consulta']:>20}"
                f"{d['tokens_in_por_consulta']:>14}{d['tokens_in_sobre_las_30']:>18}"
                f"{d['llamadas_de_reranking_sobre_las_30']:>12}  {marca}"
            )
        obs.decir(f"\n  {proyección['nota_del_supuesto']}")
        obs.decir(f"\n  {proyección['nota_de_d13']}")
    obs.decir(LÍNEA)
    return 0

def _proxy(cfg, almacén, de, obs) -> None:
    import json

    extra = []
    raíz = almacén.ruta(cfg["rutas.artefactos_indices"])
    if raíz.exists():
        for archivo in sorted(raíz.glob("*.json")):
            try:
                extra.append(json.loads(archivo.read_text(encoding="utf-8")))
            except ValueError:  # pragma: no cover - un artefacto ilegible
                continue
    gasto = met.llamadas_al_proxy(de, artefactos_extra=extra)
    precio = 0.0010
    obs.decir(
        f"\n{LÍNEA}\n💲 **Llamadas al proxy** — la unidad que el dashboard cobra\n{LÍNEA}\n"
    )
    obs.decir(
        f"  población: **todas** las {gasto['n_corridas_leidas']} corridas en disco "
        f"(huérfanas, del criterio de respuesta garantizada, ad-hoc y brazos\n de barrido incluidos) + "
        f"{len(extra)} artefacto(s) de índice con libro mayor.\n"
        f"             ⚠️ **No es la población de la tabla de la tabla de ablación**, que son solo las cinco\n"
        f"             filas declaradas: sumar esa subestimaría el gasto, y es la dirección\n"
        f"             cómoda.\n"
        f"             Excluidos por locales: {', '.join(gasto['roles_locales_excluidos'])}.\n"
    )
    for rol, n in gasto["por_rol"].items():
        obs.decir(f"  {rol:<22}{n:>8}")
    obs.decir(
        f"  {'TOTAL':<22}{gasto['total']:>8}   "
        f"({gasto['por_fuente']['corridas']} de corridas · "
        f"{gasto['por_fuente']['otros_artefactos']} de otros artefactos)\n"
        f"\n  a ~**USD {precio:.4f}** por request → **USD {gasto['total'] * precio:.2f}**\n"
        f"  ⚠️ El **precio** es declarado y no verificable (`/key/info` está bloqueado para\n"
        f"  virtual keys): lo **medido** son los requests. Es la misma distinción que la tabla de tokens, costo y la figura\n"
        f"  hace entre tokens y USD.\n"
        f"  🔑 Y `embedding_texto` **cuenta acá y no en el criterio de costo**: el criterio de costo compromete *llamadas al\n"
        f"  **LLM***, y un embedding no lo es — pero el dashboard cobra el request."
    )
    obs.decir(LÍNEA)

def _qs(cfg, almacén, set_, obs) -> int:
    corridas, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta="esta")
    _, imagen, _ = _indices_del_set(set_)
    por_imagen = met.qs_por_imagen(corridas, imagen_de=imagen.get)
    definidas = [v for v in por_imagen.values() if v is not None]
    obs.decir(f"\n{LÍNEA}\n**QS** · sensibilidad al pedido (`la evaluación §B`)\n{LÍNEA}\n")
    for img, valor in por_imagen.items():
        obs.decir(
            f"  {img:<16} {'ausente (1 pedido)' if valor is None else f'{valor:.3f}'}"
        )
    obs.decir(
        f"\n  población: {len(por_imagen)} imágenes con corrida que agrega · "
        f"{len(definidas)} con ≥ 2 pedidos\n"
        f"  **QS media: {sum(definidas) / len(definidas):.3f}**"
        if definidas
        else "\n  QS **ausente**: ninguna imagen tiene dos pedidos"
    )
    obs.decir(
        "\n  ⚠️ **`QS` alta sola no significa nada**: *un sistema aleatorio tiene QS ≈ 1*\n"
        "  (`la evaluación §B`). Se reporta **siempre junto a nDCG@5**, que necesita el pool y\n"
        "  **no está medido todavía**. Lo que esta cifra sí dice es si el pedido mueve el\n"
        "  resultado, que es la premisa del proyecto."
    )
    obs.decir(LÍNEA)
    return 0

def _cobertura_de(cfg, almacén, set_, indice, obs, *, etiqueta: str):
    corridas, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta=etiqueta)
    categoria, _, culturas = _indices_del_set(set_)
    obras = sorted({o for c in corridas for o in c.top5})
    metadata = indice.metadatas(ARTWORKS, obras)
    faltan = [o for o in obras if o not in metadata]
    if faltan:
        obs.decir(
            f"     ⚠️  {len(faltan)} obras del top-5 no están en `artworks` de esta "
            f"variante ({faltan[:3]}): se cuentan como **sin `culture`**, no como *no es "
            f"de la cultura pedida*."
        )
    return corridas, met.cobertura_cultural(
        corridas,
        culturas_por_consulta=culturas,
        categoria_de=categoria,
        metadata_por_obra=metadata,
    )

def _cobertura(cfg, almacén, set_, indice, obs) -> int:
    _, cobertura = _cobertura_de(cfg, almacén, set_, indice, obs, etiqueta="esta")
    obs.decir(
        f"\n{LÍNEA}\nla cobertura cultural del resultado · obras de la cultura pedida en el top-5 "
        f"(`la evaluación §C`)\n{LÍNEA}\n"
    )
    for c in cobertura:
        obs.decir(
            f"  {c.consulta_id:<18} {c.n_en_el_top5}/{c.de}   pedida: "
            f"{', '.join(c.cultura_pedida[:3])}{'…' if len(c.cultura_pedida) > 3 else ''}\n"
            f"  {'':<18} top-5: {', '.join(c.culturas_del_top5[:6]) or '—'}"
            + (f"  ({c.sin_culture} sin `culture`)" if c.sin_culture else "")
        )
    total = sum(c.n_en_el_top5 for c in cobertura)
    obs.decir(
        f"\n  población: {len(cobertura)} pedidos **culturales** × 5 obras = "
        f"{len(cobertura) * 5}\n"
        f"  **{total}/{len(cobertura) * 5}** obras del top-5 son de la cultura pedida\n"
        f"  criterio de `culture`: el que indexó la variante (la vista derivada de culture), declarado arriba\n\n"
        f"  ⚠️ **No es un criterio y no puede serlo** (el criterio de cobertura): *el valor de `n` depende de\n"
        f"  el criterio de normalización de culture, que sigue abierto, **así que no puede ser el umbral**; el umbral\n"
        f"  es que las tres cosas estén y sean consistentes*. Esto es la (b)."
    )
    obs.decir(LÍNEA)
    return 0

def _n(args, cfg, almacén, set_, indice, obs) -> int:
    corridas, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta="C4")
    categoria, imagen, culturas = _indices_del_set(set_)
    valores = [int(v) for v in str(args.valores).split(",") if v.strip()]
    corrido = int(cfg["consulta.n_candidatos"])
    if max(valores) != corrido:
        raise ErrorDelProyecto(
            f"se pidió derivar hasta N = {max(valores)} y esta configuración corrió con "
            f"`consulta.n_candidatos = {corrido}`. **Solo se puede derivar hacia abajo**: el "
            f"anidado toma *los primeros N del ranking fusionado* de una corrida que puntuó "
            f"{corrido} candidatas, y pedir más sería inventar puntajes que nadie pagó. "
            f"Corré `--set consulta.n_candidatos={max(valores)}` o bajá el grid."
        )
    obras = sorted(
        {f["artwork_id"] for c in corridas for f in (c.estado.get("rerank") or ())}
    )
    barrido = met.barrido_anidado_de_n(
        corridas,
        imagen_de=imagen.get,
        culturas_por_consulta=culturas,
        categoria_de=categoria,
        metadata_por_obra=indice.metadatas(ARTWORKS, obras),
        valores=valores,
    )
    obs.decir(
        f"\n{LÍNEA}\n🔴 la cantidad de candidatos a rerankear · el barrido de N, **anidado**\n{LÍNEA}\n"
    )
    medido = barrido["por_llamada_medido"]
    obs.decir(
        f"  población: {barrido['n_consultas']} consultas de C4 que agregan, corridas con "
        f"**N = {barrido['medido_con_n']}**\n"
        f"             medido por llamada de reranking: **{medido['segundos']} s** · "
        f"**{medido['tokens_in']} tokens in** ({medido['llamadas']} llamadas)\n"
        + (
            f"             ⚠️ salteadas por candidata sin puntaje: "
            f"{list(barrido['consultas_salteadas'])}\n"
            if barrido["consultas_salteadas"]
            else ""
        )
    )
    obs.decir(
        f"  {'N':>4}{'QS':>8}{'cobertura':>12}{'churn vs 50':>13}{'llam/consulta':>15}"
        f"{'s de rerank':>13}{'tokens in':>12}{'':>4}"
    )
    for n, fila in barrido["por_n"].items():
        marca = "← **medido**" if fila["medido"] else "   derivado"
        obs.decir(
            f"  {n:>4}{(fila['qs'] if fila['qs'] is not None else 0):>8.3f}"
            f"{(fila['cobertura'] or '—'):>12}"
            f"{(fila['churn_medio_contra_n_mayor'] if fila['churn_medio_contra_n_mayor'] is not None else 0):>12.2f}/5"
            f"{fila['llamadas_al_llm_por_consulta']:>15}"
            f"{fila['segundos_de_reranking_por_consulta']:>13.1f}"
            f"{fila['tokens_in_por_consulta_de_reranking']:>12}  {marca}"
        )
    obs.decir(
        f"\n  **Ahorro del anidado**: "
        f"{barrido['medido_con_n'] * barrido['n_consultas']} llamadas de reranking en vez "
        f"de {sum(barrido['por_n']) * barrido['n_consultas']} "
        f"(−{barrido['ahorro_en_llamadas_de_reranking']}).\n"
    )
    obs.decir(f"  {barrido['nota_del_anidado']}\n")
    obs.decir(f"  {barrido['nota_de_la_latencia']}")
    obs.decir(LÍNEA)
    return 0

def _medidor_declarado(cfg):
    from tp3_rag.modulo_a.fragmentador import MEDIDORES

    ruta = cfg.get("rutas.tokenizadores")
    return MEDIDORES[cfg["fragmentacion.unidad"]](
        cfg["fragmentacion.tokenizador"], Path(ruta) if ruta else None
    )

def _variantes(cfg, almacén, gen_a, gen_b, obs) -> None:
    from tp3_rag.dominio.variante import RegistroDeÍndices

    raíz = cfg["rutas.artefactos_indices"]
    obs.decir(f"\n{LÍNEA}\n**Los dos índices** · lo que cada brazo produjo\n{LÍNEA}\n")
    obs.decir(
        f"  {'brazo':<8}{'variante':<20}{'chunking':<22}{'n indexadas':>12}"
        f"{'n fragmentos':>14}{'frag/obra':>11}"
    )
    registros = {}
    for etiqueta, gen in (("base", gen_a), ("ciego" if gen_b.estrategia_de_chunking == "ciego" else "ampl.", gen_b)):
        try:
            registro = RegistroDeÍndices.leer(almacén, raíz, gen.variante_id)
        except ErrorDelProyecto:
            obs.decir(f"  {etiqueta:<8}{gen.variante_id:<20}← **sin registro en disco**")
            continue
        registros[etiqueta] = registro
        n_i = registro.conteos["n_indexadas"]
        n_f = registro.conteos["n_fragmentos"]
        obs.decir(
            f"  {etiqueta:<8}{gen.variante_id:<20}"
            f"{registro.receta['estrategia_de_chunking']['nombre']:<22}{n_i:>12}{n_f:>14}"
            f"{(n_f / n_i if n_i else 0):>11.2f}"
        )
    obs.decir(
        "\n  ⚠️ **La cifra heredada, verificada.** `la evaluación § Volumen` dice *«ciego ≈ 1,00\n"
        "  vs por campo 2,39»* **para el conjunto de tres campos**, que es el de arriba.\n"
        "  Lo medido es lo que la tabla dice: si el ciego no da 1,00 exacto es porque la\n"
        "  **concatenación** supera los 500 tokens más seguido que cada campo por\n"
        "  separado — se midió 0,9 % de los **campos** y la suma de los tres se pasa\n"
        "  más veces. Es la misma lección en el otro brazo: la cifra no viaja."
    )
    for etiqueta, registro in registros.items():
        informe = registro.conteos.get("concatenacion_del_brazo_ciego")
        if not informe:
            continue
        obs.decir(
            f"\n  🔑 **{etiqueta}** · del índice entero, no solo de lo citado:\n"
            f"     fragmentos que **mezclan licencias**: "
            f"{informe['n_fragmentos_que_mezclan_licencias']}/{informe['n_fragmentos']} = "
            f"**{informe['pct_que_mezclan_licencias']} %**\n"
            f"     combinaciones: {informe['combinaciones_de_licencias']}\n"
            f"     campos por fragmento: {informe['campos_por_fragmento']} · "
            f"concatenaciones subdivididas: {informe['n_concatenaciones_subdivididas']} "
            f"({informe['pct_subdivididas']} %)"
        )
    obs.decir(LÍNEA)

def _volumen_de_evidencia(gen_a, gen_b, comun_a, comun_b, cfg, obs) -> None:
    obs.decir(
        f"\n{LÍNEA}\n🔴  · el confundidor · **¿los dos brazos ven el mismo "
        f"texto?**\n{LÍNEA}\n"
    )
    obs.decir(
        f"  presupuesto de evidencia: `evidencia.n_fragmentos_por_obra = "
        f"{cfg['evidencia.n_fragmentos_por_obra']}` — **fijo en los dos brazos**\n"
    )
    obs.decir(
        f"  {'':<40}{'base':>14}{'otro brazo':>14}{'razón':>10}"
    )
    filas = [
        (
            "fragmentos de evidencia por obra",
            gen_a.fragmentos_por_obra,
            gen_b.fragmentos_por_obra,
        ),
        (
            "🔑 **tokens** de evidencia por obra",
            gen_a.tokens_de_evidencia_por_obra,
            gen_b.tokens_de_evidencia_por_obra,
        ),
        (
            "caracteres de evidencia por obra",
            gen_a.caracteres_de_evidencia_por_obra,
            gen_b.caracteres_de_evidencia_por_obra,
        ),
        (
            "tokens por obra · solo pares comunes",
            comun_a.tokens_de_evidencia_por_obra,
            comun_b.tokens_de_evidencia_por_obra,
        ),
    ]
    for nombre, a, b in filas:
        razón = f"{b / a:.2f}×" if (a and b) else "—"
        obs.decir(
            f"  {nombre:<40}{('—' if a is None else f'{a:,.1f}'):>14}"
            f"{('—' if b is None else f'{b:,.1f}'):>14}{razón:>10}"
        )
    obs.decir(
        f"\n  población de las dos columnas: {gen_a.n_obras} y {gen_b.n_obras} obras del "
        f"top-5 ({gen_a.n_corridas} y {gen_b.n_corridas} consultas).\n"
        f"  Los tokens se cuentan con el **medidor declarado** "
        f"(`{cfg['fragmentacion.unidad']}`/`{cfg['fragmentacion.tokenizador']}`), el mismo\n"
        f"  con el que se cortaron los fragmentos, así que son comparables contra el\n"
        f"  `tamano_objetivo` de {cfg['fragmentacion.tamano_objetivo']}.\n"
    )

    tope = int(cfg["evidencia.n_fragmentos_por_obra"])
    obs.decir(f"  obras por cantidad de fragmentos recibidos (tope = {tope}):")
    for etiqueta, gen in (("base ", gen_a), ("otro ", gen_b)):
        d = gen.obras_por_cantidad_de_fragmentos
        al_tope = sum(n for k, n in d.items() if k >= tope)
        obs.decir(
            f"    {etiqueta}: {dict(sorted(d.items()))}   → con el tope justo: {al_tope}/"
            f"{gen.n_obras} "
            f"({100 * al_tope / gen.n_obras if gen.n_obras else 0:.1f} %)"
        )
    obs.decir(
        f"\n  ⚠️ **Y «con el tope justo» NO es «el tope recortó».** Una obra que recibió\n"
        f"  {tope} puede tener exactamente {tope} fragmentos en el índice, y entonces el\n"
        f"  recolector le dio **todos**. Lo que dice si el presupuesto **desplazó** es la\n"
        f"  fila de tokens: si los dos brazos entregan el mismo texto, no desplazó."
    )
    obs.decir(
        "\n  🔴 **Cómo se lee, y es lo que el 4/11 necesita.** Si la razón de tokens por\n"
        "  obra no es ≈ 1,00, los dos brazos **no reciben la misma cantidad de texto** y\n"
        "  cualquier diferencia de *groundedness* tiene **dos** causas posibles: los\n"
        "  bordes y el volumen. Con las dos cifras al lado se puede decir cuál;\n"
        "  sin ellas, no — y publicar el delta solo sería el **modo de falla 3**.\n"
        "  🔑 **La fila que cierra la pregunta es la de los pares comunes**: ahí los dos\n"
        "  brazos tienen **las mismas obras**, así que una razón de tokens ≈ 1,00 dice que\n"
        "  reciben **el mismo texto cortado distinto** — que es exactamente lo que el eje\n"
        "  quiere aislar— y una razón lejos de 1,00 dice que el delta es de volumen.\n"
        "  ⚠️ **No se arregla bajando el `n` del brazo ciego**: eso cambia la\n"
        "  configuración congelada. Se **mide y se declara**."
    )
    obs.decir(LÍNEA)

def _licencias(gen_a, gen_b, obs) -> None:
    obs.decir(
        f"\n{LÍNEA}\n🔑  · cuántos fragmentos **mezclan licencias**, y cuántos "
        f"de los citados\n{LÍNEA}\n"
    )
    obs.decir(f"  {'':<40}{'base':>14}{'otro brazo':>14}")
    obs.decir(
        f"  {'evidencia ofrecida que mezcla':<40}"
        f"{gen_a.n_fragmentos_que_mezclan_licencias:>5}/{gen_a.n_fragmentos_de_evidencia:<8}"
        f"{gen_b.n_fragmentos_que_mezclan_licencias:>5}/{gen_b.n_fragmentos_de_evidencia:<8}"
    )
    obs.decir(
        f"  {'🔑 **referencias citadas** que mezclan':<40}"
        f"{gen_a.n_referencias_que_mezclan_licencias:>5}/{gen_a.n_referencias:<8}"
        f"{gen_b.n_referencias_que_mezclan_licencias:>5}/{gen_b.n_referencias:<8}"
    )
    obs.decir(
        f"\n  combinaciones de licencias en la **evidencia**:\n"
        f"    base : {gen_a.evidencia_por_combinacion_de_licencias}\n"
        f"    otro : {gen_b.evidencia_por_combinacion_de_licencias}\n"
        f"  combinaciones en las **citas publicadas**:\n"
        f"    base : {gen_a.referencias_por_combinacion_de_licencias}\n"
        f"    otro : {gen_b.referencias_por_combinacion_de_licencias}"
    )
    ejemplos = gen_b.ejemplos_de_mezcla_citada or gen_a.ejemplos_de_mezcla_citada
    if ejemplos:
        e = ejemplos[0]
        obs.decir(
            f"\n  **El ejemplo concreto** — *uno sirve para el informe más que una "
            f"tabla*:\n"
            f"    consulta {e['consulta_id']} · obra {e['artwork_id']}\n"
            f"    {e['fragment_id']}\n"
            f"    campos {e['campos']} · licencias {e['licencias']} · `license` publicado: "
            f"{e['license_publicado']}\n"
            f"    atribución renderizada:\n      {e['atribucion']}\n"
            f"    texto citado: {e['texto_citado'][:240]!r}"
        )
    obs.decir(
        "\n  🔴 **Y el chunking por campo no puede producir ni uno**: no es que midamos 0,\n"
        "  es que un fragmento de un solo campo tiene **una** licencia por construcción.\n"
        "  Es un argumento **nuevo** a favor del desvío ⚠️ declarado del chunking: el desvío\n"
        "  no solo evita mezclar análisis con bibliografía, **también evita mezclar\n"
        "  licencias** — y una cita sin atribución *es una infracción*, no un defecto de\n"
        "  calidad (los modos de falla del proyecto)."
    )
    obs.decir(LÍNEA)

def _secciones(gen_a, gen_b, nuevos, obs, *, titulo: str) -> None:
    obs.decir(f"\n{LÍNEA}\n🔑 **De qué `section` salen** — {titulo}\n{LÍNEA}\n")
    uso_a, uso_b = gen_a.tasa_de_uso_por_seccion(), gen_b.tasa_de_uso_por_seccion()
    secciones = sorted(
        set(gen_a.evidencia_por_seccion)
        | set(gen_b.evidencia_por_seccion)
        | set(gen_a.referencias_por_seccion)
        | set(gen_b.referencias_por_seccion)
    )
    obs.decir(
        f"  {'section':<24}{'evid. base':>11}{'refs base':>11}{'uso':>8}"
        f"{'evid. ampl':>12}{'refs ampl':>11}{'uso':>8}{'  obras con evid.':>18}"
    )
    for seccion in secciones:
        ua, ub = uso_a.get(seccion), uso_b.get(seccion)
        marca = " ←nuevo" if seccion in nuevos else ""
        obras = (
            f"{gen_a.obras_con_evidencia_por_seccion.get(seccion, 0)}"
            f" → {gen_b.obras_con_evidencia_por_seccion.get(seccion, 0)}"
        )
        obs.decir(
            f"  {seccion:<24}{gen_a.evidencia_por_seccion.get(seccion, 0):>11}"
            f"{gen_a.referencias_por_seccion.get(seccion, 0):>11}"
            f"{('—' if ua is None else f'{ua:.2f}'):>8}"
            f"{gen_b.evidencia_por_seccion.get(seccion, 0):>12}"
            f"{gen_b.referencias_por_seccion.get(seccion, 0):>11}"
            f"{('—' if ub is None else f'{ub:.2f}'):>8}{obras:>18}{marca}"
        )
    obs.decir(
        f"  {'TOTAL':<24}{gen_a.n_fragmentos_de_evidencia:>11}{gen_a.n_referencias:>11}"
        f"{'':>8}{gen_b.n_fragmentos_de_evidencia:>12}{gen_b.n_referencias:>11}"
        f"{'':>8}{gen_a.n_obras:>8} → {gen_b.n_obras}"
    )
    if not nuevos:
        obs.decir(
            "\n  ⚠️ **Los dos brazos declaran los mismos campos**, así que no hay campos\n"
            "  nuevos que contar: este eje no cambia **qué** se ofrece sino **cómo está\n"
            "  cortado**. En el brazo ciego un fragmento puede abarcar varios campos, y el\n"
            "  `section` de la tabla es **el primero** de la lista — por eso la columna de\n"
            "  evidencia se concentra ahí y hay que leerla junto al bloque de licencias."
        )
        return
    refs_nuevas = sum(gen_b.referencias_por_seccion.get(c, 0) for c in nuevos)
    evid_nueva = sum(gen_b.evidencia_por_seccion.get(c, 0) for c in nuevos)
    pct_refs = 100 * refs_nuevas / gen_b.n_referencias if gen_b.n_referencias else 0.0
    pct_evid = (
        100 * evid_nueva / gen_b.n_fragmentos_de_evidencia
        if gen_b.n_fragmentos_de_evidencia
        else 0.0
    )
    obs.decir(
        f"\n  🔑 **Los campos nuevos ({', '.join(nuevos)}) en el brazo ampliado:**\n"
        f"     evidencia **ofrecida** : {evid_nueva:>5}/{gen_b.n_fragmentos_de_evidencia} "
        f"({pct_evid:.1f} %)\n"
        f"     referencias **citadas**: {refs_nuevas:>5}/{gen_b.n_referencias} "
        f"({pct_refs:.1f} %)\n"
        f"  Si el generador casi no los cita **teniéndolos disponibles**, eso decide "
        f"la ablación de campos de la colección de texto solo."
    )

def _generacion(args, cfg, almacén, set_, obs) -> int:  # noqa: C901 - es un reporte
    from tp3_rag.dominio import ground_truth as gt
    from tp3_rag.dominio.pool import Pool

    cfg_a = Configuración.cargar(archivo=args.contra)
    if cfg_a.config_id == cfg.config_id:
        raise ErrorDelProyecto(
            f"los dos brazos tienen el mismo `config_id` ({cfg.config_id}), así que **no son "
            f"dos brazos**. `fragmentacion` determina el `config_id` —*no entra al bloque "
            f"no determinante*—, así que el brazo ampliado se declara con "
            f"`--set fragmentacion.campos_de_knowledge=[…]` sobre `--config`. Sin eso, las "
            f"corridas de las dos variantes caerían bajo la misma identidad — *dos corridas "
            f"sobre dos corpus distintos con el mismo `config_id`*, que la regla de que un id determina su contenido declara "
            f"error duro."
        )
    corridas_a, _ = _corridas_que_agregan(cfg_a, almacén, obs, etiqueta="base")
    corridas_b, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta="ampl")
    medidor = _medidor_declarado(cfg)
    gen_a = met.generacion_medida(corridas_a, medidor=medidor)
    gen_b = met.generacion_medida(corridas_b, medidor=medidor)

    nuevos = tuple(
        c for c in gen_b.campos_de_knowledge if c not in gen_a.campos_de_knowledge
    )
    cambia_el_chunking = gen_a.estrategia_de_chunking != gen_b.estrategia_de_chunking
    eje = (
        "**eje de chunking** chunking/"
        if cambia_el_chunking
        else "**eje de campos de `knowledge`** (la ablación de campos de la colección de texto)"
    )
    obs.decir(f"\n{LÍNEA}\n🔴 El **estudio secundario** · {eje}\n{LÍNEA}\n")
    if cambia_el_chunking and nuevos:
        obs.decir(
            "  🔴 **Los dos brazos difieren en los DOS ejes a la vez**, así que esto no\n"
            "  aísla ninguno: la evaluación define el estudio como chunking **×** campos, y un\n"
            "  delta sobre dos variables no dice de cuál es. Corré un brazo que cambie\n"
            "  **solo** `fragmentacion.estrategia` o **solo** `campos_de_knowledge`.\n"
        )
    elif cambia_el_chunking:
        obs.decir(
            "  🔑 **Es el eje que vuelve el desvío del chunking un desvío *mitigado*** — *«el chunking\n"
            "  ciego queda como brazo de la ablación, así que se decide con números»*. Los\n"
            "  dos brazos declaran **los mismos campos** y el mismo tamaño objetivo: la\n"
            "  única diferencia es **dónde caen los bordes**.\n"
            "  ⚠️ Y hay un confundidor que **se mide y se declara**, no se esconde: ver el\n"
            "  bloque de más abajo.\n"
        )
    else:
        obs.decir(
            "  ⚠️ **El otro eje del estudio es el chunking**, y tiene su propio par de\n"
            "  brazos desde que la variante ciega existe. Esta tabla\n"
            "  mide **campos de `knowledge`**: para el otro eje, corré los dos brazos que\n"
            "  difieren solo en `fragmentacion.estrategia`.\n"
        )
    for etiqueta, gen in (("base ", gen_a), ("ampl.", gen_b)):
        obs.decir(
            f"  {etiqueta} {gen.configuracion} · config_id {gen.config_id} · variante "
            f"{gen.variante_id}\n"
            f"        chunking: {gen.estrategia_de_chunking}\n"
            f"        campos  : {', '.join(gen.campos_de_knowledge)}"
        )
    _variantes(cfg, almacén, gen_a, gen_b, obs)
    if gen_a.variante_id == gen_b.variante_id:
        obs.decir(
            "\n  🔴 **Los dos brazos leen la MISMA variante**, así que esto no es la "
            "ablación de\n la ablación de campos de la colección de texto: el eje son **dos índices**, no dos "
            "configuraciones sobre uno. Pasá\n  `--variante-id` y verificá que las corridas "
            "del brazo ampliado se hayan producido\n  contra la variante ampliada."
        )

    obs.decir(f"\n{LÍNEA}\n**`la evaluación §D`** · lo que se mide sin juez y sin humano\n{LÍNEA}\n")
    filas = (
        ("corridas que agregan (la regla de qué corridas agregan a la tabla)", gen_a.n_corridas, gen_b.n_corridas),
        ("obras del top-5", gen_a.n_obras, gen_b.n_obras),
        ("obras con explicación", gen_a.n_explicaciones, gen_b.n_explicaciones),
        ("obras con abstención", gen_a.n_abstenciones, gen_b.n_abstenciones),
        (
            "fragmentos de evidencia",
            gen_a.n_fragmentos_de_evidencia,
            gen_b.n_fragmentos_de_evidencia,
        ),
        ("referencias `[n]` publicadas", gen_a.n_referencias, gen_b.n_referencias),
        (
            "obras con ≥ 1 referencia",
            gen_a.n_obras_con_referencia,
            gen_b.n_obras_con_referencia,
        ),
        (
            "🔴 referencias **colgadas** (el conteo de referencias colgadas)",
            gen_a.n_referencias_colgadas,
            gen_b.n_referencias_colgadas,
        ),
        (
            "afirmaciones descartadas (el descarte contado de afirmaciones)",
            gen_a.n_afirmaciones_descartadas,
            gen_b.n_afirmaciones_descartadas,
        ),
        (
            "🔴 citas **byte a byte** (el criterio de idioma)",
            f"{gen_a.n_citas_byte_a_byte}/{gen_a.n_referencias}",
            f"{gen_b.n_citas_byte_a_byte}/{gen_b.n_referencias}",
        ),
    )
    obs.decir(f"  {'':<38}{'base':>12}{'ampliado':>12}")
    for nombre, a, b in filas:
        obs.decir(f"  {nombre:<38}{str(a):>12}{str(b):>12}")
    obs.decir(
        f"  {'referencias por obra':<38}{gen_a.referencias_por_obra:>12.3f}"
        f"{gen_b.referencias_por_obra:>12.3f}\n"
        f"  {'fragmentos de evidencia por obra':<38}{gen_a.fragmentos_por_obra:>12.3f}"
        f"{gen_b.fragmentos_por_obra:>12.3f}"
    )
    obs.decir(
        "\n  ⚠️ **`evidencia` no cambia de tamaño por diseño**: el recolector trae\n"
        f"  `evidencia.n_fragmentos_por_obra = {cfg['evidencia.n_fragmentos_por_obra']}` "
        "fragmentos por obra en los dos brazos. Lo que cambia es\n  **de qué campo salen** "
        "esos tres, que es justo lo que la hipótesis predice."
    )

    comunes = met.pares_comunes(corridas_a, corridas_b)
    _secciones(gen_a, gen_b, nuevos, obs, titulo="todo el top-5 de cada brazo")
    obs.decir(
        f"\n  ⚠️ **Esa tabla mezcla dos efectos**, porque el eje **también mueve el "
        f"top-5**:\n  el desplazamiento dentro de la evidencia de una obra, y el cambio de "
        f"**qué obras**\n  se recuperaron. La de abajo aísla el primero restringiendo a los "
        f"**{len(comunes)} pares\n  `(consulta, obra)` que los dos brazos devolvieron** — "
        f"*un delta sobre poblaciones\n  distintas no es un delta*."
    )
    comun_a = met.generacion_medida(corridas_a, pares=comunes, medidor=medidor)
    comun_b = met.generacion_medida(corridas_b, pares=comunes, medidor=medidor)
    _secciones(
        comun_a,
        comun_b,
        nuevos,
        obs,
        titulo=f"solo los {len(comunes)} pares comunes — **la cifra que decide**",
    )
    _volumen_de_evidencia(gen_a, gen_b, comun_a, comun_b, cfg, obs)
    _licencias(gen_a, gen_b, obs)

    categoria, _, _ = _indices_del_set(set_)
    deltas = met.delta_por_categoria(corridas_a, corridas_b, categoria_de=categoria)
    consultas_comunes = len(
        {c.consulta_id for c in corridas_a} & {c.consulta_id for c in corridas_b}
    )
    obs.decir(
        f"\n{LÍNEA}\n⚠️ **Y el eje también mueve el retrieval** — la otra mitad de "
        f"la ablación de campos de la colección de texto\n{LÍNEA}\n"
    )
    obs.decir(
        f"  población: {consultas_comunes} consultas emparejadas por `consulta_id`\n\n"
        f"  {'categoría':<24}{'n':>4}{'churn del top-5':>18}"
    )
    for d in deltas:
        obs.decir(f"  {d.categoria:<24}{d.n_consultas:>4}{d.churn_medio:>17.2f}/5")
    obs.decir(
        "\n  La rama de texto busca **sobre `knowledge`**, así que agregar campos cambia\n"
        "  **qué obras se recuperan**, no solo con qué se las cita. El churn dice *dónde*\n"
        "  actúa; **no dice si mejora** — eso es nDCG@5 y necesita los juicios.\n"
        "  🔑 **Y el piso de ruido de C3 es `0,000` exacto**: sobre C3 toda\n"
        "  diferencia medida es señal, que es la razón por la que este estudio corre acá y\n"
        "  no sobre C4."
    )

    raíz_del_pool = gt.rutas(cfg).pool
    if args.pool:
        pool_id = args.pool
    else:
        candidatos = [
            p
            for p in Pool.del_split(almacén, raíz_del_pool, set_.split_id)
            if Pool.leer(almacén, raíz_del_pool, p).es_congelado
        ]
        if len(candidatos) != 1:
            raise ErrorDelProyecto(
                f"hay {len(candidatos)} pools **congelados** del split {set_.split_id} en "
                f"{raíz_del_pool} ({candidatos}). La séptima columna de la tabla de ablación se cuenta "
                f"contra **uno**: pasá `--pool`."
            )
        pool_id = candidatos[0]
    pool = Pool.leer(almacén, raíz_del_pool, pool_id)
    obs.decir(
        f"\n{LÍNEA}\n⚠️ **La séptima columna de la tabla de ablación** · `%` de ítems sin juzgar en el "
        f"top-5\n{LÍNEA}\n"
    )
    obs.decir(
        f"  pool: {pool.pool_id} ({pool.estado}) · {len(pool.pares)} pares juzgables · "
        f"formado por {list(pool.nombres_de_configuracion())}\n"
    )
    for etiqueta, corridas in (("base", corridas_a), ("ampliado", corridas_b)):
        sin_juzgar = met.items_del_top5_sin_juzgar(corridas, pool=pool)
        d = sin_juzgar.a_json()
        obs.decir(
            f"  **{etiqueta}** · {sin_juzgar.configuracion} / {sin_juzgar.config_id}\n"
            f"    sin juzgar: {sin_juzgar.n_sin_juzgar}/{sin_juzgar.n_items} = "
            f"**{sin_juzgar.pct_sin_juzgar} %**  "
            f"({d['n_obras_distintas_sin_juzgar']} obras distintas)\n"
            f"    {d['lectura']}\n"
        )
    obs.decir(
        "  🔑 **El mismo número se lee de dos formas y el informe del Ej. 4 tiene que\n"
        "  decir cuál publica.** Para las cinco filas que formaron el pool da **0 por\n"
        "  construcción** y es una tautología; para una **sexta** configuración el número\n"
        "  es real, y si es alto su nDCG@5 estaría subestimado por la convención de TREC."
    )
    obs.decir(LÍNEA)
    return 0

def _o2(args, cfg, almacén, set_, indice, obs) -> int:
    cfg_a = Configuración.cargar(archivo=args.contra)
    etiqueta_a = cfg_a["consulta.configuracion"]
    etiqueta_b = cfg["consulta.configuracion"]
    corridas_a, cobertura_a = _cobertura_de(
        cfg_a, almacén, set_, indice, obs, etiqueta=etiqueta_a
    )
    corridas_b, cobertura_b = _cobertura_de(
        cfg, almacén, set_, indice, obs, etiqueta=etiqueta_b
    )
    categoria, _, _ = _indices_del_set(set_)
    deltas = met.delta_por_categoria(
        corridas_a,
        corridas_b,
        categoria_de=categoria,
        cobertura_a=cobertura_a,
        cobertura_b=cobertura_b,
    )
    comunes = len(
        {c.consulta_id for c in corridas_a} & {c.consulta_id for c in corridas_b}
    )
    obs.decir(
        f"\n{LÍNEA}\nel delta por categoría de pedido · el delta **{etiqueta_b} − {etiqueta_a}** por categoría\n{LÍNEA}\n"
    )
    obs.decir(
        f"  población: {comunes} consultas con corrida **que agrega** en las dos\n"
        f"             configuraciones ({len(corridas_a)} en {etiqueta_a}, "
        f"{len(corridas_b)} en {etiqueta_b}). Emparejadas por `consulta_id`:\n"
        f"             un delta sobre poblaciones distintas no es un delta.\n"
    )
    obs.decir(
        f"  {'categoría':<22}{'n':>4}{'churn top-5':>14}"
        f"{'cobertura ' + etiqueta_a:>16}{'cobertura ' + etiqueta_b:>16}{'Δ':>8}"
    )
    for d in deltas:
        cob_a = "—" if d.cobertura_a is None else f"{d.cobertura_a:.2f}"
        cob_b = "—" if d.cobertura_b is None else f"{d.cobertura_b:.2f}"
        delta = "—" if d.delta_de_cobertura is None else f"{d.delta_de_cobertura:+.2f}"
        obs.decir(
            f"  {d.categoria:<22}{d.n_consultas:>4}{d.churn_medio:>13.2f}/5"
            f"{cob_a:>16}{cob_b:>16}{delta:>8}"
        )
    obs.decir(
        "\n  **churn top-5** = cuántas de las 5 obras cambiaron respecto de "
        f"{etiqueta_a}.\n"
        "  Dice **dónde actúa** el cambio, no si mejora.\n"
        "  **cobertura** = obras del top-5 de la cultura pedida (la cobertura cultural del resultado), y **solo existe\n"
        "  para los pedidos culturales**: las otras dos categorías no tienen cultura pedida\n"
        ", así que ahí la columna es `—` y eso no es un cero.\n"
    )
    obs.decir(
        "  🔴 **Lo que este delta NO es, y hay que decirlo antes de leerlo.** el delta por categoría de pedido dice que\n"
        "  la rama de texto *aporta*, y aportar es **calidad de ranking** = **nDCG@5**, que\n"
        "  necesita los juicios: el pool está en `piloto` con 20 ítems (la evaluación), así\n"
        "  que **la mitad de calidad del delta por categoría de pedido no es computable con este pool**. Lo que hay\n"
        "  acá es churn (dónde), cobertura cultural (el único proxy de calidad sin\n"
        "  anotación) y QS. La tabla completa es.\n\n"
        "  🔑 **Y el signo se reporta igual.** *Que la rama de texto **no** aporte es un\n"
        "  resultado válido y publicable* (el delta por categoría de pedido): *convertirlo en criterio nos\n"
        "  daría incentivo a esconderlo*."
    )
    obs.decir(LÍNEA)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

def _pool_congelado(args, cfg, almacén, set_):
    from tp3_rag.dominio import ground_truth as gt
    from tp3_rag.dominio.pool import Pool

    raíz = gt.rutas(cfg).pool
    if args.pool:
        return Pool.leer(almacén, raíz, args.pool)
    candidatos = [
        p
        for p in Pool.del_split(almacén, raíz, set_.split_id)
        if Pool.leer(almacén, raíz, p).es_congelado
    ]
    if len(candidatos) != 1:
        raise ErrorDelProyecto(
            f"hay {len(candidatos)} pools **congelados** del split {set_.split_id} en {raíz} "
            f"({candidatos}). La tabla de la tabla de ablación se mide contra **uno**: pasá `--pool`."
        )
    return Pool.leer(almacén, raíz, candidatos[0])

def _p95(valores) -> float | None:
    if not valores:
        return None
    ordenados = sorted(valores)
    if len(ordenados) == 1:
        return ordenados[0]
    posición = 0.95 * (len(ordenados) - 1)
    bajo = int(posición)
    alto = min(bajo + 1, len(ordenados) - 1)
    return ordenados[bajo] + (posición - bajo) * (ordenados[alto] - ordenados[bajo])

def _filas_de_la_tabla(cfg, almacén, obs, filas=None):
    from tp3_rag.dominio.corrida import AlmacenDeCorridas

    de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    lectura = met.corridas_de_la_tabla(de, filas=filas)
    de_la_tabla = {v: k for k, v in lectura["config_ids"].items()}
    por_fila: dict[str, list] = {}
    for corrida in lectura["corridas"]:
        por_fila.setdefault(de_la_tabla[corrida.config_id], []).append(corrida)
    salida, repartos = {}, {}
    for fila in sorted(por_fila):
        reparto = met.separar_por_9_5(tuple(por_fila[fila]))
        repartos[fila] = reparto
        if not reparto.agregan:
            obs.decir(
                f"  ⚠️  **{fila} no agrega ninguna corrida** (la regla de qué corridas agregan a la tabla): queda fuera de la "
                f"tabla y se reporta aparte con su causa."
            )
            continue
        salida[fila] = tuple(sorted(reparto.agregan, key=lambda c: c.consulta_id))
    return salida, lectura, repartos

def _columnas_sin_ground_truth(corridas, *, imagen_de, precios_in=None, precios_out=None):
    por_imagen = met.qs_por_imagen(corridas, imagen_de=imagen_de)
    definidas = [v for v in por_imagen.values() if v is not None]
    latencias = [
        float((c.contenido.get("latencia") or {}).get("segundos_de_la_corrida") or 0.0)
        for c in corridas
    ]
    tokens_in = tokens_out = llamadas = 0
    from tp3_rag.modulo_c.validadores import ROLES_DEL_LLM

    for c in corridas:
        for rol, valores in (c.contenido.get("libro_mayor") or {}).items():
            tokens_in += int(valores.get("tokens_in", 0) or 0)
            tokens_out += int(valores.get("tokens_out", 0) or 0)
            if rol in ROLES_DEL_LLM:
                llamadas += int(valores.get("n_llamadas", 0) or 0)
    costos = met.costo_por_configuracion(
        corridas, precios_in=precios_in or {}, precios_out=precios_out or {}
    )
    if len(costos) != 1:
        raise ErrorDelProyecto(
            f"una fila de la tabla de ablación tiene **{len(costos)}** etiquetas de configuración "
            f"({[c.configuracion for c in costos]}) y el costo de la fila sería el de una "
            f"de ellas. `corridas_de_la_tabla` filtra por `config_id`, así que llegar acá "
            f"significa que la población de la fila no es la que la tabla declara (**modo "
            f"de falla 3**)."
        )
    costo = costos[0]
    try:
        gen = met.generacion_medida(tuple(corridas))
        groundedness = {
            "n_referencias": gen.n_referencias,
            "n_colgadas": gen.n_referencias_colgadas,
            "n_citas_byte_a_byte": gen.n_citas_byte_a_byte,
            "pct_byte_a_byte": (
                round(100 * gen.n_citas_byte_a_byte / gen.n_referencias, 1)
                if gen.n_referencias
                else None
            ),
            "n_obras_con_referencia": gen.n_obras_con_referencia,
            "n_obras": gen.n_obras,
            "n_abstenciones": gen.n_abstenciones,
            "n_afirmaciones_descartadas_por_sa5": gen.n_afirmaciones_descartadas,
        }
    except ErrorDelProyecto:  # pragma: no cover - una fila sin salida renderizada
        groundedness = None
    return {
        "qs": round(sum(definidas) / len(definidas), 3) if definidas else None,
        "imagenes_con_qs": len(definidas),
        "latencia_p50_s": (
            round(sorted(latencias)[len(latencias) // 2], 2) if latencias else None
        ),
        "latencia_p95_s": round(_p95(latencias), 2) if latencias else None,
        "llamadas_al_llm": llamadas,
        "llamadas_al_llm_por_consulta": round(llamadas / len(corridas), 2) if corridas else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_in_por_consulta": tokens_in // len(corridas) if corridas else None,
        "llamadas_al_llm_por_consulta_con_el_plan": (
            round(costo.llamadas_al_llm_con_el_plan / len(corridas), 2) if corridas else None
        ),
        "tokens_in_por_consulta_con_el_plan": costo.tokens_in_por_consulta_con_el_plan,
        "planes_reusados_r11": costo.planes_reusados,
        "usd_por_consulta": (
            round(costo.usd_por_consulta, 6) if costo.usd_por_consulta is not None else None
        ),
        "usd_por_rol": costo.usd_por_rol,
        "nota_del_usd": costo.a_json()["nota_del_usd"],
        "nota_del_plan_r11": costo.a_json()["nota_del_plan_r11"],
        "groundedness_automatica": groundedness,
    }

def _imprimir_tabla(obs, filas, columnas, *, titulo: str, nota: str) -> None:
    obs.decir(f"\n{LÍNEA}\n{titulo}\n{LÍNEA}\n")
    obs.decir(
        f"  {'config':<5} {'n':>3} {'nDCG@5':>8} {'(def)':>8} {'P@5':>6} {'P@5e':>6} "
        f"{'MRR':>6} {'QS':>6} {'p95 s':>7} {'USD/q':>7} {'%s/j':>6}"
    )
    for nombre, fila in filas.items():
        c = columnas[nombre]
        def f(v, n=3):
            return "ausente" if v is None else f"{v:.{n}f}"
        obs.decir(
            f"  {nombre:<5} {fila.n_consultas:>3} "
            f"{f(fila.ndcg_sobre_todas):>8} {f(fila.ndcg_sobre_definidas):>8} "
            f"{f(fila.p5):>6} {f(fila.p5_estricta):>6} {f(fila.mrr):>6} "
            f"{f(c['qs']):>6} {f(c['latencia_p95_s'], 1):>7} "
            f"{f(c['usd_por_consulta'], 4):>7} {fila.pct_sin_grado:>5.1f}%"
        )
    obs.decir(f"\n{nota}")

def _juicios_extra(almacén, rutas_extra, completa) -> dict | None:
    if not rutas_extra:
        return None
    from tp3_rag.dominio import ground_truth as gtmod

    grados: dict[tuple[str, str], int] = {}
    ignorados = 0
    archivos = []
    for ruta in rutas_extra:
        _, filas = gtmod.partir_csv(almacén.leer_texto(ruta), ruta=ruta)
        archivos.append(str(ruta))
        for fila in filas:
            crudo = (fila.get("grado") or "").strip()
            if not crudo:
                continue
            par = ((fila.get("consulta_id") or "").strip(), (fila.get("artwork_id") or "").strip())
            if par in completa.grados:
                ignorados += 1
                continue
            grados[par] = int(crudo)
    return {
        "archivos": archivos,
        "grados": grados,
        "n_pares_agregados": len(grados),
        "n_pares_ya_en_el_consenso_ignorados": ignorados,
        "nota": (
            "Pooling incremental (la evaluación): los pares nuevos se suman **solo a la "
            "población (a)**. La (b) restringida no los incluye porque no pasaron por "
            "acuerdo de dos anotadores. Extender el GT mueve el IDCG, así que las cinco "
            "filas se recomputaron sobre esta misma base y **no se comparan** con la "
            "tabla publicada sobre los 379."
        ),
    }

def _tabla(args, cfg, almacén, set_, obs) -> int:  # noqa: C901 - es *la* tabla
    from tp3_rag.dominio import ground_truth as gtmod

    pool = _pool_congelado(args, cfg, almacén, set_)
    rutas = gtmod.rutas(cfg)
    completa = met.leer_consenso(
        almacén, rutas.acuerdo, pool_id=pool.pool_id, rubrica_id=args.rubrica
    )
    restringida = met.leer_consenso(
        almacén,
        rutas.acuerdo,
        pool_id=pool.pool_id,
        rubrica_id=args.rubrica,
        resoluciones=("acuerdo",),
        poblacion=met.POBLACIÓN_RESTRINGIDA,
    )
    extra = _juicios_extra(almacén, args.juicios_extra, completa)
    if extra:
        import dataclasses

        completa = dataclasses.replace(
            completa,
            grados={**completa.grados, **extra["grados"]},
            archivo=f"{completa.archivo} + {', '.join(extra['archivos'])}",
        )
    categoria, imagen, _ = _indices_del_set(set_)

    obs.decir(f"\n{LÍNEA}\n🔴 la tabla de ablación · la tabla C0–C4 — *el titular del Ejercicio 4*\n{LÍNEA}\n")
    obs.decir(
        f"  pool        : {pool.pool_id} ({pool.estado}) · {len(pool.items)} ítems · "
        f"formado por {list(pool.nombres_de_configuracion())}\n"
        f"  consenso    : {completa.archivo}\n"
        f"                {completa.n_filas_del_consenso} filas · "
        f"{dict(sorted(completa.por_resolucion.items()))} · "
        f"{completa.n_sin_grado} sin grado\n"
        f"  rúbrica     : {args.rubrica}\n"
        f"  **población (a) completa**   : {completa.n} pares con grado "
        f"(acuerdo + desempate)\n"
        f"  **población (b) restringida**: {restringida.n} pares con grado "
        f"(solo acuerdo de los dos primeros)"
    )

    filas = {}
    for par in args.fila:
        if "=" not in par:
            raise ErrorDelProyecto(f"`--fila {par}`: se espera `FILA=CONFIG_ID`.")
        fila, config_id = par.split("=", 1)
        filas[fila.strip()] = config_id.strip()
    por_fila, lectura, repartos = _filas_de_la_tabla(cfg, almacén, obs, filas=filas)
    obs.decir(
        f"\n  población de las corridas: {sum(len(v) for v in por_fila.values())} de "
        f"{lectura['en_disco']} en disco · {lectura['config_ids']}\n"
        f"  fuera: huérfanas {lectura['huerfanas']} · del criterio de respuesta garantizada {lectura['de_p7']} · "
        f"ad-hoc {lectura['ad_hoc']}\n"
        f"  la regla de qué corridas agregan a la tabla: "
        + " · ".join(
            f"{k} {len(v.agregan)}↑/{len(v.espontaneas)}✗/{len(v.fallidas)}☠"
            for k, v in sorted(repartos.items())
        )
    )

    columnas = {
        nombre: _columnas_sin_ground_truth(
            corridas,
            imagen_de=imagen.get,
            precios_in=cfg.get("proveedor.precio_usd_por_millon_de_tokens") or {},
            precios_out=cfg.get("proveedor.precio_usd_por_millon_de_tokens_de_salida")
            or {},
        )
        for nombre, corridas in por_fila.items()
    }

    filas_a = {
        nombre: met.fila_de_ranking(
            corridas, gt=completa, categoria_de=categoria, politica="cero"
        )
        for nombre, corridas in por_fila.items()
    }
    _imprimir_tabla(
        obs,
        filas_a,
        columnas,
        titulo="**(a)** los **379** ítems del consenso — con los 133 del desempate",
        nota=(
            "  `nDCG@5` es el promedio **sobre todas** las consultas (la decisión del acuerdo entre anotadores:\n"
            "  *«c05-ambiguo no aporta a nDCG@5 y sí al denominador»*); `(def)` es el mismo\n"
            "  numerador **sobre las que tienen `IDCG > 0`**. `%s/j` es el `%` de ítems del\n"
            "  top-5 sin grado — **0 por construcción** acá: las cinco filas\n"
            "  formaron el pool, así que el 0 **no dice que la comparación se sostenga: dice\n"
            "  que la pregunta no aplica**."
        ),
    )

    filas_b = {}
    for politica in met.POLÍTICAS_DE_SIN_GRADO:
        filas_b[politica] = {
            nombre: met.fila_de_ranking(
                corridas, gt=restringida, categoria_de=categoria, politica=politica
            )
            for nombre, corridas in por_fila.items()
        }
    _imprimir_tabla(
        obs,
        filas_b["condensada"],
        columnas,
        titulo=(
            "**(b)** los **246** de acuerdo — *lista condensada*: el ítem excluido **sale "
            "del ranking**"
        ),
        nota=(
            "  🔑 **Acá el `%s/j` NO es 0 y ese número es el mecanismo**: es la fracción del\n"
            "  top-5 de cada fila que dependía del desempate. Una fila con más ítems\n"
            "  disputados es una fila cuyo lugar en la tabla lo decidió el tercer anotador.\n"
            "  ⚠️ **Y la política importa:** un ítem excluido acá **sí se juzgó** —no es un\n"
            "  ítem sin juzgar—, así que contarlo 0 mediría la decisión del protocolo y no\n"
            "  el sistema. La tabla de abajo lo cuenta 0 igual, como control."
        ),
    )
    _imprimir_tabla(
        obs,
        filas_b["cero"],
        columnas,
        titulo=(
            "**(b′)** los mismos 246, con la convención de TREC — el excluido cuenta **0**"
        ),
        nota=(
            "  Es el control de la política: si el orden coincide con el de (b), la\n"
            "  conclusión no depende de qué se hace con el ítem excluido."
        ),
    )

    obs.decir(f"\n{LÍNEA}\n🔴 **el desempate · la sensibilidad** — *lo que se publica es esto*\n{LÍNEA}\n")
    comparaciones = {}
    for etiqueta, filas in (("condensada", filas_b["condensada"]), ("cero", filas_b["cero"])):
        comparación = met.comparar_poblaciones(
            list(filas_a.values()), list(filas.values()), por=args.por
        )
        comparaciones[etiqueta] = comparación
        obs.decir(
            f"  (b) con política `{etiqueta}` · métrica `{args.por}`\n"
            f"    (a) completa    : {' > '.join(comparación['orden_en_la_completa'])}\n"
            f"    (b) restringida : {' > '.join(comparación['orden_en_la_restringida'])}\n"
            f"    → **el orden {'CAMBIA' if comparación['el_orden_cambia'] else 'NO cambia'}**\n"
        )
        for nombre, d in comparación["por_fila"].items():
            obs.decir(
                f"      {nombre}  {d['completa']} → {d['restringida']}  "
                f"(Δ {d['delta']:+.4f})  ·  {d['pct_sin_grado_en_la_restringida']} % del "
                f"top-5 dependía del desempate"
                if d["delta"] is not None
                else f"      {nombre}  ausente"
            )
        obs.decir(f"\n  {comparación['lectura']}\n")
    obs.decir(f"  {comparaciones['condensada']['nota']}")

    obs.decir(
        f"\n{LÍNEA}\nla tabla de ablación · el desglose por las 3 categorías de pedido\n"
        f"*«ese desglose es donde está la historia interesante, no en el promedio»* "
        f"(la evaluación)\n{LÍNEA}"
    )
    for etiqueta, filas in (("(a) 379", filas_a), ("(b) 246", filas_b["condensada"])):
        obs.decir(f"\n  ── {etiqueta} " + "─" * 56)
        obs.decir(
            f"  {'cat':<21} " + " ".join(f"{n:>13}" for n in sorted(filas))
        )
        for cat in met.CATEGORÍAS:
            celdas = []
            for nombre in sorted(filas):
                sub = filas[nombre].por_categoria().get(cat)
                celdas.append(
                    "          —"
                    if sub is None
                    else f"{(sub.ndcg_sobre_todas or 0):.3f}/{sub.p5:.2f}"
                )
            obs.decir(f"  {cat:<21} " + " ".join(f"{c:>13}" for c in celdas))
        obs.decir("  (cada celda: nDCG@5 sobre todas / P@5)")

    _columnas_5_a_7(obs, columnas)
    piso = _piso_de_ruido(obs, cfg, almacén, lectura["config_ids"])
    _al_lado_de_la_tabla(obs, por_fila, columnas, completa, restringida)

    if args.json:
        destino = almacén.escribir_json(
            args.json,
            {
                "artefacto": "TablaDeA2",
                "es_una_lectura": True,
                "pool_id": pool.pool_id,
                "rubrica_id": args.rubrica,
                "juicios_extra": (
                    {k: v for k, v in extra.items() if k != "grados"} if extra else None
                ),
                "poblaciones": {
                    "completa": completa.a_json(),
                    "restringida": restringida.a_json(),
                },
                "config_ids": lectura["config_ids"],
                "columnas_sin_ground_truth": columnas,
                "tabla_a_completa": {k: v.a_json() for k, v in filas_a.items()},
                "tabla_b_restringida_condensada": {
                    k: v.a_json() for k, v in filas_b["condensada"].items()
                },
                "tabla_b_restringida_cero": {
                    k: v.a_json() for k, v in filas_b["cero"].items()
                },
                "por_categoria_completa": {
                    k: {c: f.a_json() for c, f in v.por_categoria().items()}
                    for k, v in filas_a.items()
                },
                "por_categoria_restringida": {
                    k: {c: f.a_json() for c, f in v.por_categoria().items()}
                    for k, v in filas_b["condensada"].items()
                },
                "sensibilidad_td_1": comparaciones,
                "piso_de_ruido_m9_1": piso,
            },
        )
        obs.decir(f"\n  JSON: {destino}")
    obs.decir(LÍNEA)
    return 0

def _piso_de_ruido(obs, cfg, almacén, config_ids) -> dict:
    from tp3_rag.dominio.corrida import AlmacenDeCorridas

    de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    piso = met.piso_de_ruido(de, config_ids=config_ids)
    obs.decir(
        f"\n{LÍNEA}\n🔴  · el piso de ruido, **medido por fila**\n{LÍNEA}\n"
    )
    obs.decir(
        f"  {'fila':<6} {'pares':>6} {'ordinales':>11} {'top5 igual':>11} "
        f"{'churn/5':>8} {'piso de QS':>11}  estado"
    )
    for fila, d in piso["por_fila"].items():
        obs.decir(
            f"  {fila:<6} {d['n_pares_con_dos_ordinales']:>6} "
            f"{str(d['ordinales']):>11} {str(d['top5_identico'] or '—'):>11} "
            + (f"{d['churn_medio_de_5']:>8.3f}" if d["churn_medio_de_5"] is not None else f"{'—':>8}")
            + (f"{d['piso_de_qs']:>11.4f}" if d["piso_de_qs"] is not None else f"{'ausente':>11}")
            + f"  {d['estado']}"
        )
    if piso["de_otros_config_id"]:
        obs.decir(
            "\n  ⚠️ **Y el 0,249 heredado está medido sobre OTRO `config_id`** que el de la\n"
            "     fila publicada: las corridas con dos ordinales de C4 son de un pipeline\n"
            "     anterior. Se reporta aparte en vez de imputárselo a la fila:"
        )
        for config_id, d in piso["de_otros_config_id"].items():
            obs.decir(
                f"       {config_id}  (etiqueta {d['fila_que_declara_la_etiqueta'] or '?'})  "
                f"{d['n_pares_con_dos_ordinales']} pares · top5 igual "
                f"{d['top5_identico']} · churn {d['churn_medio_de_5']:.3f}/5 · "
                f"piso de QS **{d['piso_de_qs']:.4f}**"
            )
    obs.decir(f"\n  {piso['nota']}\n\n  {piso['nota_de_la_inclusion']}")
    return piso

def _al_lado_de_la_tabla(obs, por_fila, columnas, completa, restringida) -> None:
    obs.decir(f"\n{LÍNEA}\n🔴 Lo que va **al lado de la tabla**, en la misma página\n{LÍNEA}\n")
    obs.decir(
        "  **1 · El piso de ruido**. Dos corridas de la **misma** configuración\n"
        "    mueven **0,8 obras de 5** del top-5, así que **toda diferencia de ±0,8 obras\n"
        "    entre dos filas es indistinguible del ruido**. `QS` se publica con su piso al\n"
        "    lado: **0,249** para C4 (medido sobre 30 pares de ordinales) y **0,000 exacto**\n"
        "    para C3 (30/30 top-5 idénticos). ⚠️ El piso de C4 está medido sobre **otro\n"
        "    `config_id`** que el de la fila publicada: las corridas con dos ordinales de C4\n"
        "    son de un pipeline anterior.\n"
    )
    obs.decir(
        "  **2 · El κ por categoría** (el acuerdo entre anotadores, estrato limpio de 269 ítems y 22 consultas):\n"
        "    `conceptual_cultural` **0,810** · `visual_cromatico` **0,249** · `ambiguo`\n"
        "    **0,228**. 🔴 **La fila `ambiguo` de esta tabla descansa sobre un κ de 0,228**,\n"
        "    que es un factor 3,5 por debajo de la categoría donde la rúbrica se refinó.\n"
    )
    obs.decir(
        "  **3 · Una consulta con varianza cero**: `c05-ambiguo` salió **15/15 en grado 0**.\n"
        "    **No aporta a `nDCG@5` y sí al denominador** (decisión del acuerdo entre anotadores, punto 4), y por\n"
        "    eso la tabla publica los dos promedios: `nDCG@5` y `(def)`.\n"
    )
    obs.decir(
        "  **4 · ** · **0 corridas en dos cubetas, medido**: el reparto de la regla de qué corridas agregan a la tabla de\n"
        "    arriba suma todas las corridas en disco y `fallidas` es **0** — o sea que el\n"
        "    defecto de `separar_por_9_5` (una corrida fallida **y** degradada se contaría\n"
        "    dos veces) es **latente**, no activo. No se arregla acá: es cambio de\n"
        "    comportamiento y decisión del grupo.\n"
    )
    obs.decir(
        "  5 · Las secciones · la tabla por `section` **no es comparable entre brazos**: en el\n"
        "    brazo ciego un fragmento abarca varios campos y el `section` publicado es **el\n"
        "    primero**, así que `did_you_know` cae de 25 citas a 1 **y no es que no se cite**.\n"
        "    Vale para `tp3-metricas generacion`, no para esta tabla — va dicho igual.\n"
    )
    obs.decir(
        f"  **Y la séptima columna de la tabla de ablación**: el `%` de ítems sin juzgar es **0 en las cinco\n"
        f"    filas por construcción**, porque las cinco formaron el pool. El pool\n"
        f"    lo declara medido: {dict(sorted(completa.por_resolucion.items()))} sobre "
        f"{completa.n_filas_del_consenso} filas.\n"
        f"    ⚠️ **Un lector va a leer ese 0 como evidencia de que la comparación se\n"
        f"    sostiene, cuando es una tautología.** Donde la regla muerde es en una **sexta**\n"
        f"    configuración — y en la población (b), donde el `%` mide **otra cosa**: cuánto\n"
        f"    del top-5 dependía del desempate."
    )

def _columnas_5_a_7(obs, columnas) -> None:
    obs.decir(
        f"\n{LÍNEA}\nla tabla de ablación · las columnas 5 a 7 — *groundedness*, latencia y costo\n{LÍNEA}\n"
    )
    obs.decir(
        f"  {'config':<6} {'refs':>5} {'colgadas':>9} {'byte a byte':>12} "
        f"{'abst':>5} {'el descarte contado de afirmaciones':>5} {'p50 s':>7} {'p95 s':>7} {'llam/q':>7} {'tok in/q':>9} "
        f"{'USD/q':>8}"
    )
    for nombre, c in sorted(columnas.items()):
        g = c["groundedness_automatica"] or {}
        pct = g.get("pct_byte_a_byte")
        usd = c["usd_por_consulta"]
        usd_txt = f"{usd:.4f}" if usd is not None else "ausente"
        obs.decir(
            f"  {nombre:<6} {g.get('n_referencias', 0):>5} "
            f"{g.get('n_colgadas', 0):>9} "
            f"{(f'{pct:.1f} %' if pct is not None else 'ausente'):>12} "
            f"{g.get('n_abstenciones', 0):>5} "
            f"{g.get('n_afirmaciones_descartadas_por_sa5', 0):>5} "
            f"{c['latencia_p50_s']:>7} {c['latencia_p95_s']:>7} "
            f"{c['llamadas_al_llm_por_consulta_con_el_plan']:>7} "
            f"{c['tokens_in_por_consulta_con_el_plan']:>9} "
            f"{usd_txt:>8}"
        )
    obs.decir(
        "\n  🔴 **La columna *groundedness* de la tabla de ablación está publicada A MEDIAS, y va dicho.**\n"
        "    Lo que hay acá es la mitad **automática** —el conteo de referencias colgadas: referencias colgadas, con\n"
        "    umbral duro 0; y la cita **byte a byte** contra el fragmento de la evidencia—.\n"
        "    La mitad **fina** (*«% de afirmaciones factuales respaldadas por el fragmento\n"
        "    citado»*, `la evaluación §D`) necesita **un humano** y es la muestra fija de **25**\n"
        "    explicaciones del criterio de honestidad, que **no está hecha**. Publicar la automática\n"
        "    con el nombre de la otra sería el modo de falla 3: una cifra con una población\n"
        "    que no es la que la afirmación usa.\n"
        "    ⚠️ el descarte contado de afirmaciones es `n_afirmaciones_descartadas_por_sa5` y **no cuenta la afirmación mal formada registrada y contada**: una\n"
        "    afirmación **mal formada** no tiene `ref` y no tiene dónde contarse.\n"
        "    Lo nuevo entra **al lado**, no renombrando esa clave (ocho tests la\n"
        "    afirman y el pool está construido encima).\n"
        "\n  🔴 **`USD/consulta` ya NO sale ausente — se leyó el precio el 2026-11-03.**\n"
        "     dejó la condición escrita —*el día que alguien lea el precio con su\n"
        "    fecha y su URL se anota en `[proveedor]` y en ningún otro lado*— y se cumplió:\n"
        "    `https://developers.openai.com/api/docs/pricing`, leída el 2026-11-03.\n"
        "    **No hizo falta re-correr nada**: los tokens ya estaban medidos en los libros\n"
        "    mayores. La regla sigue intacta: *un modelo sin precio da **ausente**, no cero*.\n"
        "    ⚠️ **Sigue sin copiarse de LangSmith**: el precio se declara acá, donde\n"
        "    viaja adentro de la corrida como procedencia, en vez de en una tabla del\n"
        "    servicio **que se actualiza sola cuando el proveedor cambia precios**.\n"
        "    ⚠️ **Y es DERIVADO de un precio de lista, no de la tarifa del proxy**, que nadie\n"
        "    puede leer. Derivar el total da **USD 12,43** contra los **10,0654** que facturó\n"
        "    el dashboard: **las dos conviven y no se promedian**.\n"
        "\n  🔴 **`llam/q` y `tok in/q` llevan el plan adentro desde el 2026-11-03 (el reuso del plan al barrer el peso de fusión).**\n"
        "    Un plan **reusado** no paga el planificador —cinco brazos de un barrido\n"
        "    comparten un plan—, así que leer solo el libro mayor publicaba **1,0**\n"
        "    llamadas/consulta en `C2` cuando el criterio de costo compromete **2**, y **3.604** tokens de\n"
        "    entrada cuando son **5.629** (un 56 % menos). El caché del plan es un artefacto\n"
        "    de **nuestra ablación**, no de producción: una consulta de un usuario paga su\n"
        "    planificador siempre. La lectura **pagada** sigue publicándose en la tabla de\n"
        "    la tabla de tokens, costo y la figura, y las dos van juntas igual que en el criterio de costo.\n"
        "    ⚠️ El reintento de sigue adentro de la pagada — C1 pagó **31** sobre 30\n"
        "    consultas — y `2 + N` sigue siendo **cota superior**: el reranker puntúa\n"
        "    `min(N, |fusionado|)`.\n"
        "    🔑 **Y los embeddings que el criterio de costo no cuenta y la tabla de tokens, costo y la figura sí van dichos acá**: el rol\n"
        "    `embedding_texto` **no es una llamada al LLM** y el dashboard del proxy\n"
        "    **cobra el request igual** — por eso `tp3-metricas costo` los suma y esta\n"
        "    columna no. Son dos cifras con dos significados."
    )
