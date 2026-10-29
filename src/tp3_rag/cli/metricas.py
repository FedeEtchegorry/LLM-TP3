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
            "MétricasDeRanking: QS, cobertura cultural y el delta por categoría. "
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
    sub.add_parser("qs", help="QS por imagen y total — el diseño de la evaluación, sin anotación")
    sub.add_parser(
        "cobertura",
        help="cobertura cultural del top-5 en los pedidos culturales — `la cobertura cultural del resultado`",
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
            "🔴 · la cantidad de candidatos a rerankear · los **tres** valores de N derivados de una sola corrida "
            "de N = 50: QS, cobertura, churn, latencia y costo por N"
        ),
    )
    n.add_argument(
        "--valores",
        default="10,20,50",
        help="el grid del diseño de la evaluación. El mayor tiene que ser el que se corrió",
    )
    generacion = sub.add_parser(
        "generacion",
        help=(
            "🔴 · la ablación de campos de la colección de texto · los dos brazos de campos de `knowledge` del lado de "
            "la **generación**: referencias por obra, de qué `section` salen, citas byte a "
            "byte y el `%%` de ítems sin juzgar. **Cero llamadas al proxy**"
        ),
    )
    generacion.add_argument(
        "--contra",
        type=Path,
        required=True,
        help=(
            "la capa del brazo **base** (la A de `B − A`). Para el estudio secundario los "
            "dos brazos son `config/c3.toml` y lo único que los separa es "
            "`fragmentacion.campos_de_knowledge`, que va por `--set` sobre `--config`"
        ),
    )
    generacion.add_argument(
        "--pool",
        help=(
            "el `pool_id` contra el que se cuenta la **séptima columna de la tabla de ablación**. Si falta, "
            "el pool **congelado** del split (`MP-3`)"
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
            f"variante: {registro.variante_id}\n"
            f"set: {set_.identidad} ({len(set_.consultas())} consultas)\n"
            f"criterio de `culture`: {set_.criterio_de_culture}   ← **va con el número** "
            f"(criterio declarado)"
        )
        if args.comando == "costo":
            return _costo(args, cfg, almacén, obs)
        if args.comando == "qs":
            return _qs(cfg, almacén, set_, obs)
        if args.comando == "generacion":
            return _generacion(args, cfg, almacén, set_, obs)
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
            f"--config … set`."
        )
    del_set = [c for c in todas if c.es_del_set]
    ad_hoc = len(todas) - len(del_set)
    por_consulta: dict[str, object] = {}
    for c in del_set:
        previa = por_consulta.get(c.consulta_id)
        if previa is None or c.ordinal > previa.ordinal:
            por_consulta[c.consulta_id] = c
    reparto = met.separar_por_degradacion(tuple(por_consulta.values()))
    obs.decir(
        f"\n{etiqueta:<4} config_id {cfg.config_id} · {len(todas)} corridas en disco → "
        f"{len(por_consulta)} (del set, último ordinal)"
        + (f" · {ad_hoc} **ad-hoc** fuera de la población" if ad_hoc else "")
        + f"\n **la regla de qué corridas agregan a la tabla**: {len(reparto.agregan)} agregan · "
        f"{len(reparto.planificadas)} con degradación planificada · "
        f"{len(reparto.espontaneas)} espontánea · {len(reparto.fallidas)} fallidas"
    )
    for causa in reparto.a_json()["causas_espontaneas"][:5]:
        obs.decir(f"     ⚠️  espontánea: {causa}")
    if not reparto.agregan:
        raise ErrorDelProyecto(
            f"ninguna corrida de {etiqueta} agrega a la tabla. Las que hay "
            f"degradaron, y *una C4 que degradó en silencio entraría etiquetada como C4 "
            f"comportándose como C3*: medir sobre ellas publicaría una fila que dice lo "
            f"que no."
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
    costos = met.costo_por_configuracion(corridas)
    obs.decir(
        f"\n{LÍNEA}\nla tabla de tokens, costo y la figura · el costo **medido**, por configuración\n{LÍNEA}\n"
    )
    obs.decir(
        f"  población: {len(corridas)} corridas de las **cinco filas declaradas**, de "
        f"{lectura['en_disco']} en disco\n"
        f"             {lectura['por_fila']}\n"
        f"             `config_id` por fila: {lectura['config_ids']}\n"
        f"             ⚠️ dos filtros, y los dos cambian la población:\n"
        f"             · **por `config_id`**, no por la etiqueta: {lectura['huerfanas']} "
        f"de un hito anterior\n y {lectura['de_p7']} del criterio de respuesta garantizada (plan de falla "
        f"declarado, que la regla de qué corridas agregan a la tabla manda reportar\n aparte) —la etiqueta "
        f"`C1` sobrevive a los cambios\n               de configuración y la identidad no, y "
        f"`n = 120` en una fila de 30 consultas\n               es una población que no "
        f"existe (**modo de falla 3**);\n"
        f" · **último ordinal por consulta**: con el muestreo fijado, registrado y con su variabilidad medida hay dos corridas de la "
        f"misma\n               consulta a propósito, y sumarlas duplicaría el costo por "
        f"consulta;\n"
        f" · 🔴 **y la procedencia** (): {lectura['ad_hoc']} ad-hoc con el "
        f"`config_id` de una fila\n               declarada, que **no son de las 30 del "
        f"set**. Entraban, y una fila de 30\n               consultas publicaba `n = 31`.\n"
        + (
            f" 🔴 **sin corridas de {lectura['faltan']}**: la tabla de ablación pide **5 filas**, "
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
        "\n ⚠️ **En tokens y no en USD, y no es una omisión** (, ):\n"
        "  `[proveedor]` declara el precio de `openai/text-embedding-3-small` y de ningún\n"
        "  otro modelo, y *un modelo sin precio da **ausente**, no cero*. El día que alguien\n"
        "  lea el precio de `gpt-4.1-mini` **con su fecha y su URL**, se anota ahí y esta\n"
        "  tabla publica USD sin tocar nada más."
    )
    for c in costos:
        if not c.por_rol.get("reranker"):
            continue
        obs.decir(f"\n{LÍNEA}\n**la cantidad de candidatos a rerankear** · la proyección de N, para \n{LÍNEA}\n")
        proyección = met.proyeccion_de_n(c, n_consultas=int(args.consultas))
        obs.decir(
            f"  medido con **N = {proyección['medido_con_n']}** · "
            f"{proyección['llamadas_del_reranker_medidas']} llamadas de reranking · "
            f"**{proyección['tokens_in_por_llamada_del_reranker']} tokens in por llamada**\n"
            f" (contra los **1.946** que el registro de decisiones midió a mano con el mismo par de "
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
        f" ⚠️ **No es la población de la tabla de la tabla de ablación**, que son solo las cinco\n"
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
        f" virtual keys): lo **medido** son los requests. Es la misma distinción que la tabla de tokens, costo y la figura\n"
        f" hace entre tokens y USD.\n"
        f" 🔑 Y `embedding_texto` **cuenta acá y no en el criterio de costo**: el criterio de costo compromete *llamadas al\n"
        f" **LLM***, y un embedding no lo es — pero el dashboard cobra el request."
    )
    obs.decir(LÍNEA)

def _qs(cfg, almacén, set_, obs) -> int:
    corridas, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta="esta")
    _, imagen, _ = _indices_del_set(set_)
    por_imagen = met.qs_por_imagen(corridas, imagen_de=imagen.get)
    definidas = [v for v in por_imagen.values() if v is not None]
    obs.decir(f"\n{LÍNEA}\n**QS** · sensibilidad al pedido\n{LÍNEA}\n")
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
        ". Se reporta **siempre junto a nDCG@5**, que necesita el pool y\n"
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
        f"\n{LÍNEA}\n**la cobertura cultural del resultado** · obras de la cultura pedida en el top-5 "
        f"\n{LÍNEA}\n"
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
        f" criterio de `culture`: el que indexó la variante, declarado arriba\n\n"
        f" ⚠️ **No es un criterio y no puede serlo**: *el valor de `n` depende de\n"
        f" el criterio de normalización de culture, que sigue abierto, **así que no puede ser el umbral**; el umbral\n"
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
        f"\n{LÍNEA}\n🔴 **la cantidad de candidatos a rerankear** · el barrido de N, **anidado**\n{LÍNEA}\n"
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
    refs_nuevas = sum(gen_b.referencias_por_seccion.get(c, 0) for c in nuevos)
    evid_nueva = sum(gen_b.evidencia_por_seccion.get(c, 0) for c in nuevos)
    pct_refs = 100 * refs_nuevas / gen_b.n_referencias if gen_b.n_referencias else 0.0
    pct_evid = (
        100 * evid_nueva / gen_b.n_fragmentos_de_evidencia
        if gen_b.n_fragmentos_de_evidencia
        else 0.0
    )
    obs.decir(
        f"\n  🔑 **Los campos nuevos ({', '.join(nuevos) or '—'}) en el brazo ampliado:**\n"
        f" evidencia **ofrecida**: {evid_nueva:>5}/{gen_b.n_fragmentos_de_evidencia} "
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
            f"dos brazos**. `fragmentacion` determina el `config_id` (: *no entra al "
            f"bloque no determinante*), así que el brazo ampliado se declara con "
            f"`--set fragmentacion.campos_de_knowledge=[…]` sobre `--config`. Sin eso, las "
            f"corridas de las dos variantes caerían bajo la misma identidad — *dos corridas "
            f"sobre dos corpus distintos con el mismo `config_id`*, que la regla de que un id determina su contenido declara "
            f"error duro."
        )
    corridas_a, _ = _corridas_que_agregan(cfg_a, almacén, obs, etiqueta="base")
    corridas_b, _ = _corridas_que_agregan(cfg, almacén, obs, etiqueta="ampl")
    gen_a = met.generacion_medida(corridas_a)
    gen_b = met.generacion_medida(corridas_b)

    obs.decir(
        f"\n{LÍNEA}\n🔴 **la ablación de campos de la colección de texto** · el estudio secundario, **eje de campos de "
        f"`knowledge`**\n{LÍNEA}\n"
    )
    obs.decir(
        "  🔴 **El estudio queda con UN eje de dos, y el que falta tiene fecha.**\n"
        " el diseño de la evaluación lo define como **chunking × campos de `knowledge`**. El eje de\n"
        " **chunking** —el que vuelve *mitigado* al desvío /— **no está\n"
        "  implementado**: `ESTRATEGIAS = (\"por_campo_semantico\",)`. Construirlo es\n"
        " los hallazgos y es una decisión del grupo antes del 4/11. Lo que esta\n"
        "  tabla mide es el **otro** eje, el que sí tiene sus dos variantes construidas.\n"
    )
    for etiqueta, gen in (("base ", gen_a), ("ampl.", gen_b)):
        obs.decir(
            f"  {etiqueta} {gen.configuracion} · config_id {gen.config_id} · variante "
            f"{gen.variante_id}\n"
            f"        campos: {', '.join(gen.campos_de_knowledge)}"
        )
    if gen_a.variante_id == gen_b.variante_id:
        obs.decir(
            "\n  🔴 **Los dos brazos leen la MISMA variante**, así que esto no es la "
            "ablación de\n la ablación de campos de la colección de texto: el eje son **dos índices**, no dos "
            "configuraciones sobre uno. Pasá\n  `--variante-id` y verificá que las corridas "
            "del brazo ampliado se hayan producido\n  contra la variante ampliada."
        )

    obs.decir(f"\n{LÍNEA}\n**el diseño de la evaluación** · lo que se mide sin juez y sin humano\n{LÍNEA}\n")
    filas = (
        ("corridas que agregan", gen_a.n_corridas, gen_b.n_corridas),
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
            "🔴 referencias **colgadas**",
            gen_a.n_referencias_colgadas,
            gen_b.n_referencias_colgadas,
        ),
        (
            "afirmaciones descartadas",
            gen_a.n_afirmaciones_descartadas,
            gen_b.n_afirmaciones_descartadas,
        ),
        (
            "🔴 citas **byte a byte**",
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

    nuevos = tuple(
        c for c in gen_b.campos_de_knowledge if c not in gen_a.campos_de_knowledge
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
    _secciones(
        met.generacion_medida(corridas_a, pares=comunes),
        met.generacion_medida(corridas_b, pares=comunes),
        nuevos,
        obs,
        titulo=f"solo los {len(comunes)} pares comunes — **la cifra que decide**",
    )

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
        " 🔑 **Y el piso de ruido de C3 es `0,000` exacto** (, ): sobre C3 toda\n"
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
        "  🔑 **`MP-3`: el mismo número se lee de dos formas y el informe del Ej. 4 tiene\n"
        "  que decir cuál publica.** Para las cinco filas que formaron el pool da **0 por\n"
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
        " **cobertura** = obras del top-5 de la cultura pedida, y **solo existe\n"
        "  para los pedidos culturales**: las otras dos categorías no tienen cultura pedida\n"
        ", así que ahí la columna es `—` y eso no es un cero.\n"
    )
    obs.decir(
        " 🔴 **Lo que este delta NO, y hay que decirlo antes de leerlo.** el delta por categoría de pedido dice que\n"
        "  la rama de texto *aporta*, y aportar es **calidad de ranking** = **nDCG@5**, que\n"
        " necesita los juicios: el pool está en `piloto` con 20 ítems (el diseño de la evaluación, ), así\n"
        " que **la mitad de calidad del delta por categoría de pedido no es computable en esta sesión**. Lo que hay\n"
        "  acá es churn (dónde), cobertura cultural (el único proxy de calidad sin\n"
        " anotación) y QS. La tabla completa.\n\n"
        "  🔑 **Y el signo se reporta igual.** *Que la rama de texto **no** aporte es un\n"
        " resultado válido y publicable* (los criterios de entrega): *convertirlo en criterio nos\n"
        "  daría incentivo a esconderlo*."
    )
    obs.decir(LÍNEA)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
