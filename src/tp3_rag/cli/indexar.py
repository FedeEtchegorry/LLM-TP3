from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.catalogo import CatálogoCanónico
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.dominio.variante import RegistroDeÍndices
from tp3_rag.modulo_a.constructor_de_indice import ConstructorDeÍndice, derivar_variante_id
from tp3_rag.modulo_a.fragmentador import Fragmentador, informe_de_secciones
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.indice_vectorial import ARTWORKS, KNOWLEDGE, ÍndiceVectorialChroma

PASOS = ("artworks", "fragmentos", "knowledge", "registro")

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-indexar",
        description=(
            "ConstructorDeÍndice: `artworks` desde el caché (el reuso del caché de imagen entre variantes) + `knowledge` desde los "
            "fragmentos, y la registración como transacción (los invariantes de la registración, los cuatro invariantes)."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], metavar="clave=valor",
        help="override de una clave declarada; una clave desconocida es un error",
    )
    p.add_argument("--catalogo-id", help="cuál catálogo, si hay más de uno en disco")
    p.add_argument("--split-id", help="cuál split, si hay más de uno congelado")
    p.add_argument(
        "--hasta", choices=PASOS, default="registro",
        help=(
            "hasta qué paso llegar. `fragmentos` es todo lo que se puede hacer **sin** "
            "`LLM_API_KEY`; registrar sin `knowledge` es imposible por el invariante 3"
        ),
    )
    p.add_argument(
        "--informe-de-secciones", action="store_true",
        help=(
            "mide el `n` de fragmentos de **todas** las secciones del catálogo, no solo "
            "las configuradas: es el insumo de la ablación de campos de la colección de texto, que sigue abierto"
        ),
    )
    p.add_argument("--mostrar", action="store_true", help="las variantes ya registradas")
    p.add_argument("--silencioso", action="store_true")
    return p.parse_args(argv)

def main(argv=None) -> int:  # noqa: C901 - es un entrypoint: lee, orquesta y muestra
    args = _argumentos(argv)
    overrides = {}
    for par in args.set:
        if "=" not in par:
            print(f"--set espera clave=valor, llegó {par!r}", file=sys.stderr)
            return 2
        clave, _, valor = par.partition("=")
        overrides[clave.strip()] = valor

    obs = Observabilidad(silencioso=args.silencioso)
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)

        if args.mostrar:
            return _mostrar_registradas(cfg, almacén, obs)

        base = (
            Path(cfg["rutas.catalogo"]) / args.catalogo_id
            if args.catalogo_id
            else CatálogoCanónico.único(almacén, cfg["rutas.catalogo"])
        )
        obs.decir(f"catálogo: {base} (re-verificando el `catalogo_id`…)")
        catálogo = CatálogoCanónico.cargar(almacén, base)
        obs.decir(
            f"             {len(catálogo.registros)} registros · "
            f"catalogo_id = {catálogo.catalogo_id} ✓"
        )

        manifiesto = _manifiesto(cfg, almacén, args.split_id)
        obs.decir(
            f"split: {manifiesto.split_id} (clase 3 — congelado)\n"
            f"             corpus {manifiesto.conteos['corpus_indexado']} · "
            f"held-out {manifiesto.conteos['held_out']} · "
            f"elegibles {manifiesto.conteos['elegibles_con_texto_e_imagen']}"
        )

        fragmentador = Fragmentador(cfg)
        constructor, indice = _armar(cfg, almacén, manifiesto, catálogo, fragmentador, obs)
        obs.decir(
            f"variante: {constructor.variante_id} (clase 1 — la identificación de variantes)\n"
            f"             chunking {fragmentador.estrategia.nombre} · "
            f"{fragmentador.estrategia.tamano_objetivo} "
            f"{fragmentador.estrategia.unidad} (solapamiento "
            f"{fragmentador.estrategia.solapamiento})\n"
            f"             campos de knowledge: {list(fragmentador.campos_de_knowledge)}"
            f" ← la ablación de campos de la colección de texto, abierto\n"
            f"             colecciones en {indice.raíz}"
        )

        obs.decir("\ncobertura: `ids del lado corpus ⊆ ids del catálogo`…")
        corpus = constructor.verificar_cobertura(catálogo.registros)
        obs.decir(f"             ✓ las {len(corpus)} del corpus están en el catálogo")

        obs.decir(
            f"\nartworks: leyendo el caché de embeddings (el reuso del caché de imagen entre variantes: 0 llamadas a CLIP)…"
        )
        n_indexadas = constructor.construir_artworks(corpus)
        obs.decir(f" ✓ {n_indexadas} vectores · norma 1 verificada (la verificación de la norma de los vectores)")

        obs.decir("\nfragmentos: chunking por campo semántico (⚠️ / el registro de decisiones la elección del índice vectorial)…")
        por_obra = constructor.fragmentar(corpus)
        n_fragmentos = sum(len(v) for v in por_obra.values())
        obs.decir(f"             ✓ {n_fragmentos} fragmentos de {len(corpus)} obras")

        cita = constructor.cita_al_embebido()
        _a6(obs, constructor, n_indexadas, cita)
        _secciones(obs, fragmentador, por_obra)

        if args.informe_de_secciones:
            _informe_de_p2(obs, cfg, corpus)

        if args.hasta in ("artworks", "fragmentos"):
            _por_que_para_aca(obs, args.hasta, constructor, n_indexadas, n_fragmentos)
            return 0

        obs.decir(
            "\npre-vuelo: invariantes 1, 3 y 4 de los invariantes de la registración **antes** de gastar la API…"
        )
        constructor.verificar_invariantes(por_obra=por_obra, materializado=True)
        obs.decir("             ✓ los cuatro pasan contra lo ya materializado")

        credenciales = Credenciales.desde_entorno()
        credenciales.exigir_openai()
        from tp3_rag.puertos.embedding_texto import EmbeddingTextoOpenAI

        constructor.embebedor_de_texto = obs.instrumentar(
            EmbeddingTextoOpenAI(
                api_key=credenciales.openai_api_key,
                base_url=cfg["proveedor.base_url"],
                modelo=cfg["embebido_texto.modelo"],
                dim=cfg["embebido_texto.dim"],
                limite_de_tokens=cfg["embebido_texto.limite_de_tokens"],
                reintentos=cfg["http.reintentos"],
                backoff_base=cfg["http.backoff_base"],
                medidor=fragmentador.medidor,
            ),
            rol="embedding_texto",
        )
        obs.decir(
            f"\nknowledge: {cfg['embebido_texto.modelo']} · {n_fragmentos} fragmentos\n"
            f"             proveedor {cfg['proveedor.base_url']}  "
            f"(key de `{credenciales.variable_de_la_key}`)…"
        )
        n = constructor.construir_knowledge(por_obra)
        obs.decir(f"             ✓ {n} fragmentos en `knowledge`")
        _costo(obs, constructor, por_obra)

        if args.hasta == "knowledge":
            _por_que_para_aca(obs, args.hasta, constructor, n_indexadas, n_fragmentos)
            return 0

        obs.decir("\nregistro: los cuatro invariantes de los invariantes de la registración, juntos…")
        resultado = constructor.registrar(por_obra=por_obra, cita_al_embebido=cita)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _mostrar_registro(resultado, obs)
    return 0

def _armar(cfg, almacén, manifiesto, catálogo, fragmentador, obs):
    variante_id = derivar_variante_id(
        cfg,
        catalogo_id=catálogo.catalogo_id,
        split_id=manifiesto.split_id,
        fragmentador=fragmentador,
    )
    indice = ÍndiceVectorialChroma(
        almacén.ruta(cfg["rutas.chroma"]),
        variante_id=variante_id,
        metrica=cfg["indice.metrica"],
        batch_de_upsert=cfg["indice.batch_de_upsert"],
    )
    constructor = ConstructorDeÍndice(
        cfg,
        almacén=almacén,
        indice=indice,
        manifiesto=manifiesto,
        catalogo_id=catálogo.catalogo_id,
        observabilidad=obs,
        fragmentador=fragmentador,
    )
    return constructor, indice

def _manifiesto(cfg, almacén, split_id: str | None) -> ManifiestoDeSplit:
    raíz = cfg["rutas.artefactos_splits"]
    if split_id:
        return ManifiestoDeSplit.leer(almacén, raíz, split_id)
    existentes = ManifiestoDeSplit.existentes(almacén, raíz)
    if not existentes:
        raise ErrorDeArtefacto(
            "no hay ningún split congelado. Corré `poetry run tp3-split` primero: la "
            "membresía del corpus la fija el manifiesto y no el catálogo (el corpus parametrizado, los invariantes de la registración)."
        )
    if len(existentes) > 1:
        raise ErrorDeArtefacto(
            f"hay {len(existentes)} splits congelados: {list(existentes)}. Pasá "
            f"`--split-id`: con más de uno, cuál se indexa es una decisión y no un "
            f"`mtime`. Y los juicios entran por `split_id` (la tabla de identidades, clase 3)."
        )
    return ManifiestoDeSplit.leer(almacén, raíz, existentes[0])

def _a6(obs, constructor, n_indexadas: int, cita: dict) -> None:
    línea = "─" * 78
    corpus = set(constructor.manifiesto.ids_corpus)
    indexados = set(constructor.indice.ids(ARTWORKS))
    n_embebidas = cita.get("n_embebidas")

    obs.decir(f"\n{línea}\nA6 · las dos cifras, que son **dos números separados**\n{línea}")
    obs.decir(
        f"\n n embebidas: {n_embebidas if n_embebidas is not None else '—':>6}"
        f"   ← del **universo con imagen** · reporte del embebido ordinal "
        f"{cita.get('ordinal')}\n"
        f" n indexadas: {n_indexadas:>6} ← del **corpus** · este RegistroDeÍndices\n"
        f"  diferencia: "
        f"{(n_embebidas - n_indexadas) if n_embebidas is not None else '—':>6}"
        f"   ← **declarada esperada**: obras con imagen que no pasan el\n"
        f" criterio de texto, más el held-out (la exclusión del held-out de toda colección)"
    )
    obs.decir(
        f"\n invariante 2 de los invariantes de la registración · **igualdad de conjuntos**, no de conteos\n"
        f" |ids indexados|: {len(indexados):>6}\n"
        f"    |lado corpus del manifiesto|: {len(corpus):>6}\n"
        f" indexados ∖ corpus: {len(indexados - corpus):>6}\n"
        f" corpus ∖ indexados: {len(corpus - indexados):>6}\n"
        f" son el mismo conjunto: "
        f"{'✓ sí' if indexados == corpus else '✗ NO'}"
    )
    held = set(constructor.manifiesto.ids_held_out)
    obs.decir(
        f"\n invariante 1 de los invariantes de la registración · contra el manifiesto, no contra un flag\n"
        f" held-out ∩ indexados: {len(held & indexados):>6}   "
        f"{'✓' if not (held & indexados) else '✗'}"
    )
    obs.decir(línea)

def _costo(obs, constructor, por_obra) -> None:
    línea = "─" * 78
    m = constructor.medición_del_embebido_de_texto(por_obra)
    p, d = m["poblacion"], m["medido"]
    usd = m["costo_usd"]
    obs.decir(
        f"\n{línea}\nCosto del embebido de texto · **medido**, no proyectado "
        f"(la tabla de tokens, costo y la figura · los tokens en el artefacto de corrida)\n{línea}"
    )
    obs.decir(
        f"\n población: {p['n_fragmentos']} fragmentos de "
        f"{p['n_obras_del_corpus']} obras **del corpus**\n"
        f"                  campos {p['campos_de_knowledge']} · modelo {p['modelo']}\n"
        f"\n n llamadas: {d['n_llamadas']:>10} ← batches al puerto (: plural)\n"
        f" n fragmentos: {d['n_fragmentos_embebidos']:>10}\n"
        f"  tokens de entrada: {d['tokens_in']:>7}   ← del `usage` de la API, no estimados\n"
        f" tokens/fragmento: {d['tokens_por_fragmento']:>7}\n"
        f" segundos de API: {d['segundos_de_api']:>7}\n"
        f" resultados: {d['resultados']}"
    )
    obs.decir(
        f"\n precio USD/1M tokens: "
        f"{m['precio_usd_por_millon_de_tokens'] if m['precio_usd_por_millon_de_tokens'] else '—'}"
        f"   ← **declarado** (`proveedor.precio_usd_por_millon_de_tokens`),\n"
        f"                         no verificable por API: `/key/info` está bloqueado\n"
        f"                         para virtual keys\n"
        f" **costo USD**: {usd if usd is not None else '—  (sin precio declarado)'}"
        f"   ← derivado declarado: tokens medidos × precio"
    )
    obs.decir(
        "\n la ablación de campos de la colección de texto proyectaba 0,012 USD para esta misma población. La cifra de\n"
        "  arriba la **emite el pipeline** y queda adentro del `RegistroDeÍndices`\n"
        ", que es lo que hace comparable una con otra sin transcribir."
    )
    obs.decir(línea)

def _secciones(obs, fragmentador, por_obra) -> None:
    línea = "─" * 78
    conteos = fragmentador.conteos_por_seccion()
    unidad = fragmentador.estrategia.unidad
    objetivo = fragmentador.estrategia.tamano_objetivo

    obs.decir(f"\n{línea}\nFragmentos de la variante · por `section`\n{línea}\n")
    obs.decir(
        f"  {'section':<16}{'obras':>7}{'frag':>7}{'frag/obra':>11}"
        f"{'subdiv':>8}{'%':>7}{'prom':>8}{'mayor':>8}"
    )
    for section, c in conteos.items():
        obs.decir(
            f"  {section:<16}{c['n_obras_con_el_campo']:>7}{c['n_fragmentos']:>7}"
            f"{c['fragmentos_por_obra']:>11.2f}{c['n_campos_subdivididos']:>8}"
            f"{c['pct_subdivididos']:>6.1f}%{c['unidades_promedio']:>8.1f}"
            f"{c['unidades_del_mayor']:>8}"
        )
    total = sum(c["n_fragmentos"] for c in conteos.values())
    subdiv = sum(c["n_campos_subdivididos"] for c in conteos.values())
    campos = sum(c["n_obras_con_el_campo"] for c in conteos.values())
    obs.decir(
        f"  {'TOTAL':<16}{campos:>7}{total:>7}"
        f"{'':>11}{subdiv:>8}"
        f"{(100 * subdiv / campos if campos else 0):>6.1f}%"
    )
    obs.decir(
        f"\n  `prom` y `mayor` están en **{unidad}**; el objetivo es {objetivo}. "
        f"`subdiv` es\n  cuántos campos lo superaron y el `RecursiveCharacterTextSplitter` "
        f"🎓 tuvo que\n entrar — que es la sospecha del stack declarado que nadie había "
        f"medido."
    )

    sin = [i for i, f in por_obra.items() if not f]
    obs.decir(
        f"\n obras del corpus con 0 fragmentos: {len(sin)}   "
        f"{'✓ todas con fragmento' if not sin else '✗ la variante NO se registra'}"
    )
    obs.decir(línea)

def _informe_de_p2(obs, cfg, corpus) -> None:
    línea = "═" * 78
    informe = informe_de_secciones(cfg, corpus)
    obs.decir(
        f"\n{línea}\nInsumo de la ablación de campos de la colección de texto · todas las secciones del catálogo\n"
        f"(**abierto**: esto no lo cierra, le pone números)\n{línea}"
    )
    obs.decir(
        f"\n configuradas: {informe['secciones_configuradas']}\n"
        f" candidatas ⚠️: {informe['candidatas_de_p2']}\n"
        f" medido sobre: {informe['n_obras']} obras del corpus indexado\n"
    )
    for ámbito, conteos in [("TOTAL", informe["total"])] + [
        (museo, d["secciones"]) for museo, d in informe["por_museo"].items()
    ]:
        n_obras = (
            informe["n_obras"]
            if ámbito == "TOTAL"
            else informe["por_museo"][ámbito]["n_obras"]
        )
        obs.decir(f"  {ámbito}  ({n_obras} obras)")
        obs.decir(
            f"    {'section':<16}{'obras':>7}{'cobertura':>11}{'frag':>8}"
            f"{'frag/obra':>11}{'subdiv':>8}{'prom':>8}{'mayor':>8}"
        )
        for section, c in conteos.items():
            cobertura = 100 * c["n_obras_con_el_campo"] / n_obras if n_obras else 0
            obs.decir(
                f"    {section:<16}{c['n_obras_con_el_campo']:>7}{cobertura:>10.1f}%"
                f"{c['n_fragmentos']:>8}{c['fragmentos_por_obra']:>11.2f}"
                f"{c['n_campos_subdivididos']:>8}{c['unidades_promedio']:>8.1f}"
                f"{c['unidades_del_mayor']:>8}"
            )
        obs.decir("")
    obs.decir(línea)

def _por_que_para_aca(obs, hasta: str, constructor, n_indexadas, n_fragmentos) -> None:
    línea = "─" * 78
    obs.decir(f"\n{línea}\n`--hasta {hasta}` · lo que queda sin hacer, **dicho**\n{línea}")
    obs.decir(
        f"\n hecho: artworks ({n_indexadas} vectores desde el caché) · "
        f"fragmentos ({n_fragmentos})\n"
        f"  **sin hacer**: `knowledge` y la **registración**\n"
    )
    obs.decir(
        "  La variante **no está registrada**, y no es un paso que falte correr: el\n"
        " invariante 3 de los invariantes de la registración —*toda obra indexada en `artworks` tiene ≥ 1\n"
        "  fragmento en `knowledge`*— **es parte de la transacción de registración**.\n"
        "  Sin `knowledge` no se puede verificar, y una variante que no pasa los cuatro\n"
        "  invariantes no se registra: *si uno falla, no se escribe nada*.\n"
    )
    obs.decir(
        f"  Cuando aparezca la `LLM_API_KEY`:\n"
        f"    poetry run tp3-indexar\n\n"
        f"  variante_id (ya derivado, no cambia al registrar): "
        f"{constructor.variante_id}\n"
        f" colecciones: {constructor.indice.raíz}\n"
        f" artworks: {constructor.indice.contar(ARTWORKS)} vectores\n"
        f"    knowledge: "
        + (
            f"{constructor.indice.contar(KNOWLEDGE)} vectores"
            if constructor.indice.existe(KNOWLEDGE)
            else "**no existe todavía** — y el contrato de los puertos la vuelve error duro al leerla,\n"
            "                así que nadie la puede consultar creyendo que está vacía"
        )
    )
    obs.decir(línea)

def _mostrar_registro(resultado, obs: Observabilidad) -> None:
    línea = "─" * 78
    estado = (
        "ya estaba registrada · se leyó y se verificó, no se reescribió"
        if resultado.ya_estaba_registrada
        else "REGISTRADA ahora"
    )
    obs.decir(
        f"\n{línea}\nRegistroDeÍndices · {resultado.variante_id}  (clase 1 — {estado})\n{línea}"
    )
    inv = resultado.registro.a_json()["invariantes_de_9_6"]
    obs.decir(
        f"\n los cuatro invariantes de los invariantes de la registración · **se verifican juntos**\n"
        f" 1 · held-out fuera de toda colección (la exclusión del held-out de toda colección) ✓\n"
        f" 2 · ids indexados = lado corpus (conjuntos) + el dataset documentado y reproducible ✓\n"
        f" 3 · toda obra indexada con ≥ 1 fragmento (al menos un fragmento por obra indexada) ✓\n"
        f"    4 · receta ⊇ ingredientes del variante_id        ✓"
    )
    obs.decir(
        f"\n n indexadas: {inv['n_indexadas']:>6} ← dueño: este artefacto\n"
        f" n fragmentos: {inv['n_fragmentos']:>6} ← dueño: este artefacto\n"
        f" n embebidas: "
        f"{inv['2_igualdad_de_conjuntos_y_a6']['n_embebidas_citada'] or '—':>6}   "
        f"← **citada**, no copiada (reporte del embebido)"
    )
    obs.decir(f"\n registro: {resultado.ruta_del_registro}")
    obs.decir(línea)

def _mostrar_registradas(cfg, almacén, obs: Observabilidad) -> int:
    raíz = cfg["rutas.artefactos_indices"]
    existentes = RegistroDeÍndices.existentes(almacén, raíz)
    if not existentes:
        print(
            "no hay ninguna variante registrada todavía. `tp3-indexar --hasta fragmentos` "
            "hace todo lo que no necesita la `LLM_API_KEY`.",
            file=sys.stderr,
        )
        return 1
    for vid in existentes:
        registro = RegistroDeÍndices.leer(almacén, raíz, vid)
        receta = registro.receta
        obs.decir(
            f"\n{vid}  split={registro.split_id}  catálogo={receta['catalogo_id']}\n"
            f" chunking: {receta['estrategia_de_chunking']['nombre']} · "
            f"{receta['estrategia_de_chunking']['tamano_objetivo']} "
            f"{receta['estrategia_de_chunking']['unidad']}\n"
            f" campos: {receta['campos_de_knowledge']}\n"
            f" conteos: n_indexadas={registro.conteos['n_indexadas']} · "
            f"n_fragmentos={registro.conteos['n_fragmentos']}\n"
            f" embebido: ordinal {registro.cita_al_embebido.get('ordinal')} "
            f"(citado, no copiado)"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
