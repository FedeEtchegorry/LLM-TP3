from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio import ground_truth as gt
from tp3_rag.dominio import rubrica as drubrica
from tp3_rag.dominio.consultas import SetDeConsultas
from tp3_rag.dominio.corrida import AlmacenDeCorridas
from tp3_rag.dominio.juicios import GRADOS, SIGNIFICADO_DE_LA_ESCALA, AporteDeAnotador
from tp3_rag.dominio.pool import ESTADOS, Pool
from tp3_rag.dominio.rubrica import Rúbrica
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.dominio.variante import RegistroDeÍndices
from tp3_rag.modulo_c.acuerdo import UMBRAL, Acuerdo
from tp3_rag.modulo_c.constructor_de_pool import (
    CATEGORÍA_REPETIDA_POR_OMISIÓN,
    CONFIGURACIONES_POR_ESTADO,
    ConstructorDePool,
    ReglaDeSelección,
    ReglaDeTodasLasConsultas,
)
from tp3_rag.modulo_c.planilla import PlanillaDeAnotación
from tp3_rag.nucleo import identidad
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDeArtefacto, ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

LÍNEA = "─" * 78

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-anotar",
        description=(
            "El camino de anotación: Rúbrica, Pool, PlanillaDeAnotación y κ. "
            "Los tres artefactos son clase 3 y se escriben una vez. El TEXTO de la "
            "rúbrica y los juicios los escribe el grupo; esto propone, valida y congela."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], metavar="clave=valor", dest="overrides",
        help="override de una clave declarada; una clave desconocida es un error",
    )
    p.add_argument("--split-id", help="cuál split, si hay más de uno congelado")
    p.add_argument("--autoria", help="cuál `SetDeConsultas`, si hay más de uno del split")
    p.add_argument("--silencioso", action="store_true")

    sub = p.add_subparsers(dest="comando", required=True)

    rp = sub.add_parser(
        "rubrica-plantilla",
        help="🔴 el ESQUELETO de la rúbrica para que el grupo llene. No la escribe",
    )
    rp.add_argument("--designacion", default="piloto", help="el `rubrica_id`")
    rp.add_argument("--refina", help="si refina otra rúbrica (el acuerdo entre anotadores con κ < 0.6)")
    rp.add_argument("--salida", type=Path, help="dónde escribirla (por omisión, stdout)")
    rp.add_argument(
        "--escribir", action="store_true",
        help="escribirla en artifacts/ground-truth/rubricas/<designacion>.md",
    )

    r = sub.add_parser("rubrica", help="leer y validar las rúbricas congeladas")
    r.add_argument("--rubrica-id", help="cuál, si hay más de una")

    pd = sub.add_parser(
        "pool-derivar",
        help=(
            "derivar un pool: el PROVISORIO de la ronda piloto o el "
            "CONGELADO con --estado congelado"
        ),
    )
    pd.add_argument("--designacion", default="piloto", help="la mitad declarada del `pool_id`")
    pd.add_argument(
        "--estado", default="piloto", choices=list(ESTADOS),
        help=(
            "🔴 el protocolo, no una etiqueta: `piloto` es C1 sobre 4 consultas y "
            "`congelado` es la unión de los top-5 de C0–C4 sobre todas"
        ),
    )
    pd.add_argument(
        "--categoria-repetida", default=CATEGORÍA_REPETIDA_POR_OMISIÓN,
        help="solo en `piloto`: la regla es *una por categoría más una repetida*",
    )
    pd.add_argument(
        "--semilla", type=int,
        help="la de la mezcla y la selección (por omisión, la del split)",
    )
    pd.add_argument(
        "--permitir-imagen-repetida", action="store_true",
        help="solo en `piloto`: aflojar que las 4 consultas sean de 4 imágenes distintas",
    )
    pd.add_argument(
        "--configuracion", action="append", dest="configuraciones", metavar="CX",
        help=(
            "qué filas de la tabla forman la unión; repetible. Por omisión, las del estado: "
            f"{dict(CONFIGURACIONES_POR_ESTADO)}"
        ),
    )

    pm = sub.add_parser("pool", help="mostrar un pool derivado")
    pm.add_argument("--designacion", help="cuál, si hay más de uno del split")

    co = sub.add_parser("controles", help=" y sueltos — los mismos que corren al exportar")
    co.add_argument("--designacion", default="piloto")

    ex = sub.add_parser("exportar", help="la planilla de UNA persona (corre y antes)")
    ex.add_argument("--designacion", default="piloto", help="de qué pool")
    ex.add_argument("--rubrica", required=True, help="el `rubrica_id` con el que se va a juzgar")
    ex.add_argument("--anotador", required=True, help="la etiqueta de quién anota")
    ex.add_argument("--salida", type=Path, help="dónde (por omisión, planillas/…csv)")

    im = sub.add_parser("importar", help="🔴 la política de trazado · validar que lo importado sea lo exportado")
    im.add_argument("--planilla", type=Path, required=True)

    ka = sub.add_parser("kappa", help="κ de Cohen del piloto — se reporta pase lo que pase (el acuerdo entre anotadores)")
    ka.add_argument("--designacion", default="piloto")
    ka.add_argument("--rubrica", required=True)
    ka.add_argument(
        "--anotador", action="append", default=[],
        help="la pareja del κ (κ de Cohen es entre DOS). Por omisión, los dos que haya",
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
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        almacén = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
        rutas = gt.rutas(cfg)

        if args.comando == "rubrica-plantilla":
            return _rubrica_plantilla(args, almacén, obs, rutas)
        if args.comando == "rubrica":
            return _rubrica(args, almacén, obs, rutas)
        if args.comando == "pool-derivar":
            return _pool_derivar(args, cfg, almacén, obs, rutas)
        if args.comando == "pool":
            return _pool_mostrar(args, cfg, almacén, obs, rutas)
        if args.comando == "controles":
            return _controles(args, cfg, almacén, obs, rutas)
        if args.comando == "exportar":
            return _exportar(args, cfg, almacén, obs, rutas)
        if args.comando == "importar":
            return _importar(args, cfg, almacén, obs, rutas)
        if args.comando == "kappa":
            return _kappa(args, cfg, almacén, obs, rutas)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0  # pragma: no cover - argparse exige un subcomando

def _manifiesto(args, cfg, almacén) -> ManifiestoDeSplit:
    raíz = cfg["rutas.artefactos_splits"]
    if args.split_id:
        return ManifiestoDeSplit.leer(almacén, raíz, args.split_id)
    existentes = ManifiestoDeSplit.existentes(almacén, raíz)
    if not existentes:
        raise ErrorDeArtefacto(
            "no hay ningún split congelado. Todo el ground truth **ancla a un split** "
            ": corré `poetry run tp3-split` primero."
        )
    if len(existentes) > 1:
        raise ErrorDeArtefacto(
            f"hay {len(existentes)} splits congelados: {list(existentes)}. Pasá "
            f"`--split-id`: *los juicios de un split no son los de otro* "
            f", y elegir por `mtime` sería tomar esa decisión por "
            f"fecha de modificación."
        )
    return ManifiestoDeSplit.leer(almacén, raíz, existentes[0])

def _set_de_consultas(args, almacén, rutas, manifiesto) -> SetDeConsultas:
    if args.autoria:
        ident = identidad.identidad_del_set_de_consultas(
            split_id=manifiesto.split_id, autoria=args.autoria
        )
    else:
        del_split = SetDeConsultas.del_split(almacén, rutas.consultas, manifiesto.split_id)
        usables = []
        for i in del_split:
            s = SetDeConsultas.leer(almacén, rutas.consultas, i)
            if s.es_usable_para_medir:
                usables.append(i)
        if not usables:
            raise ErrorDeArtefacto(
                f"no hay ningún `SetDeConsultas` **completo** para el split "
                f"{manifiesto.split_id} (hay {list(del_split)}). Un set `plantilla` no es "
                f"usable para medir: autoralo con `tp3-consultas autorar`."
            )
        if len(usables) > 1:
            raise ErrorDeArtefacto(
                f"hay {len(usables)} sets completos del split: {usables}. Pasá `--autoria`: "
                f"los ~540 juicios entran por los `consulta_id` de **uno**."
            )
        ident = usables[0]
    return SetDeConsultas.leer(almacén, rutas.consultas, ident)

def _registros_de_variantes(cfg, almacén) -> dict:
    raíz = cfg["rutas.artefactos_indices"]
    return {
        v: RegistroDeÍndices.leer(almacén, raíz, v)
        for v in RegistroDeÍndices.existentes(almacén, raíz)
    }

def _pool(args, almacén, rutas, manifiesto, *, designacion=None) -> Pool:
    designación = designacion or getattr(args, "designacion", None)
    if designación:
        ident = identidad.identidad_del_pool(
            split_id=manifiesto.split_id, designacion=designación
        )
    else:
        del_split = Pool.del_split(almacén, rutas.pool, manifiesto.split_id)
        if not del_split:
            raise ErrorDeArtefacto(
                f"no hay ningún pool para el split {manifiesto.split_id}. Derivá el del "
                f"piloto con `poetry run tp3-anotar pool-derivar --designacion piloto`."
            )
        if len(del_split) > 1:
            raise ErrorDeArtefacto(
                f"hay {len(del_split)} pools del split: {list(del_split)}. Pasá "
                f"`--designacion`: el pool `piloto` y el `definitivo` son **dos artefactos "
                f"con dos ids** a propósito (la tabla de identidades, la verificación de la norma de los vectores)."
            )
        ident = del_split[0]
    return Pool.leer(almacén, rutas.pool, ident)

def _rubrica_plantilla(args, almacén, obs, rutas) -> int:
    texto = drubrica.plantilla(designacion=args.designacion, refina=args.refina)
    destino = args.salida
    if args.escribir and destino is None:
        destino = Rúbrica.ruta_de(rutas.rubricas, args.designacion)
    if destino is None:
        print(texto)
        return 0
    if almacén.existe(destino):
        raise ErrorDeArtefacto(
            f"ya existe {almacén.ruta(destino)} y **no se sobreescribe**: la `Rúbrica` es "
            f"clase 3 y lo que hay adentro puede ser el texto que el grupo "
            f"escribió. Si querés otra, usá otra `--designacion` — refinar es **derivar "
            f"otra**."
        )
    ruta = almacén.escribir_texto(destino, texto)
    obs.decir(
        f"\n{LÍNEA}\nRúbrica `{args.designacion}` — **PLANTILLA** (estado = plantilla)\n{LÍNEA}\n"
        f"\n  {ruta}\n"
        f"\n  🔴 **El texto lo escribe el grupo.** Lo que hay ahí son las cinco secciones "
        f"que\n"
        f" la tabla de identidades dice que una rúbrica tiene —definiciones de {list(GRADOS)}, ejemplos y\n"
        f"  desempate— con cada hueco marcado `{drubrica.MARCA}`, más lo único que ya está\n"
        f" decidido: los tres nombres de la escala del diseño de la evaluación.\n"
        f"\n  Mientras quede un `{drubrica.MARCA}`, `estado` sigue en `plantilla` y "
        f"**exportar\n  una planilla falla**: gastar 20 juicios bajo un instrumento a medio "
        f"escribir es\n  justo lo que el piloto existe para no hacer con los ~540.\n"
        f"\n  Al terminar: poné `estado = \"completa\"` y validá con\n"
        f"      poetry run tp3-anotar rubrica\n"
        f"\n ✅ Va al repo, **diffeable a propósito**: el diff entre dos rúbricas "
        f"es\n  la evidencia de qué cambió el piloto. Agregala por ruta explícita — este "
        f"repo\n  **no usa `git add -A`**.\n{LÍNEA}"
    )
    return 0

def _rubrica(args, almacén, obs, rutas) -> int:
    existentes = [args.rubrica_id] if args.rubrica_id else list(
        Rúbrica.existentes(almacén, rutas.rubricas)
    )
    if not existentes:
        print(
            f"no hay ninguna `Rúbrica` todavía.\n"
            f"  Sacá el esqueleto con `poetry run tp3-anotar rubrica-plantilla --escribir`.\n"
            f"  🔴 El texto lo escribe el grupo: es el instrumento con el que se juzgan los "
            f"~540.",
            file=sys.stderr,
        )
        return 1
    for rubrica_id in existentes:
        r = Rúbrica.leer(almacén, rutas.rubricas, rubrica_id)
        marca = "✓" if r.es_usable_para_anotar else "⚠️"
        obs.decir(
            f"{marca} {r.rubrica_id:<14} estado = {r.estado:<10} "
            f"hash = {r.hash_del_texto}"
            + (f"  · refina `{r.refina}`" if r.refina else "")
        )
        obs.decir(
            f"     secciones: "
            + " · ".join(
                f"{k} ({len(v.splitlines())} líneas)" for k, v in sorted(r.secciones().items())
            )
        )
        if not r.es_usable_para_anotar:
            obs.decir(
                f"     ⚠️  `estado = {r.estado}`: **no es usable para anotar**. Quedan "
                f"{r.texto.count(drubrica.MARCA)} marcas `{drubrica.MARCA}`."
            )
    return 0

def _pool_derivar(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    set_de_consultas = _set_de_consultas(args, almacén, rutas, manifiesto)
    corridas_de = AlmacenDeCorridas(almacén, cfg["rutas.artefactos_corridas"])
    corridas = []
    for cid in corridas_de.existentes():
        try:
            corridas.append(corridas_de.leer(cid))
        except ErrorDeArtefacto as e:
            obs.decir(f"⚠️  corrida {cid} rechazada al leer: {e}")
    if not corridas:
        raise ErrorDeArtefacto(
            f"no hay corridas en {cfg['rutas.artefactos_corridas']}. El pool del piloto es "
            f"**el top-5 de C1**, y eso se lee de las corridas: "
            f"`artifacts/corridas/` está gitignoreado y es **regenerable** con "
            f"`poetry run tp3-responder set`."
        )
    obs.decir(
        f"split: {manifiesto.split_id} ·  "
        f"{manifiesto.conteos['corpus_indexado']} corpus + "
        f"{manifiesto.conteos['held_out']} held-out\n"
        f"set: {set_de_consultas.identidad} · estado "
        f"{set_de_consultas.estado}\n"
        f"corridas: {len(corridas)} en disco"
    )

    regla = (
        ReglaDeTodasLasConsultas(semilla=args.semilla)
        if args.estado == "congelado"
        else ReglaDeSelección(
            categoria_repetida=args.categoria_repetida,
            imagenes_distintas=not args.permitir_imagen_repetida,
            semilla=args.semilla,
        )
    )
    resultado = ConstructorDePool(
        cfg, almacén=almacén, rutas=rutas, observabilidad=obs
    ).derivar(
        designacion=args.designacion,
        set_de_consultas=set_de_consultas,
        manifiesto=manifiesto,
        corridas=corridas,
        regla=regla,
        estado=args.estado,
        configuraciones_esperadas=args.configuraciones,
    )
    _mostrar_pool(resultado.pool, obs, ruta=resultado.ruta,
                  ruta_json=resultado.ruta_de_procedencia, ya_estaba=resultado.ya_estaba)
    if not resultado.ya_estaba:
        obs.decir(
            f"  descartadas: {resultado.corridas_descartadas}\n"
            f" ✅ Los dos archivos **van al repo**: es el material más caro del\n"
            f"     proyecto. Agregalos por ruta explícita — **no `git add -A`**.\n"
            f"\n  Y lo que sigue:\n"
            f"     1. el grupo llena la rúbrica   (tp3-anotar rubrica-plantilla --escribir)\n"
            f"     2. una planilla por persona    (tp3-anotar exportar --anotador …)\n"
            f"     3. **dos** personas anotan — κ necesita dos\n"
            f"     4. tp3-anotar importar, y después tp3-anotar kappa\n{LÍNEA}"
        )
    return 0

def _pool_mostrar(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    _mostrar_pool(
        pool, obs,
        ruta=almacén.ruta(Pool.ruta_de(rutas.pool, pool.pool_id)),
        ruta_json=almacén.ruta(Pool.ruta_de_procedencia(rutas.pool, pool.pool_id)),
        ya_estaba=True,
    )
    return 0

FILAS_QUE_SE_LISTAN = 20

def _mostrar_pool(pool: Pool, obs, *, ruta, ruta_json, ya_estaba: bool) -> None:
    estado = "ya estaba derivado · se leyó, no se reescribió" if ya_estaba else "DERIVADO ahora"
    obs.decir(f"\n{LÍNEA}\nPool · {pool.pool_id}  (clase 3 — {estado})\n{LÍNEA}")
    regla = pool.regla
    obs.decir(
        f"\n split_id: {pool.split_id}\n"
        f" designación: {pool.designacion} ← la mitad declarada de la identidad\n"
        + (
            f" 🔴 estado: {pool.estado} ← la campaña definitiva: una vez anotado **no "
            f"se toca**\n"
            if pool.es_congelado
            else f" 🔴 estado: {pool.estado} ← **NO es el pool congelado** "
            f"(el diseño de la evaluación, )\n"
        )
        + "  configs:\n"
        + "\n".join(
            f"    {c['configuracion']}  config_id {c['config_id']}  variante "
            f"{c['variante_id']}  α={c.get('alfa')}  "
            + ("α congelado" if c.get("alfa_congelado") else "⚠️ α **provisorio**")
            + f"  ·  {len(c.get('corridas') or ())} corridas"
            for c in pool.configuraciones
        )
        + f"\n semilla: {pool.semilla_de_mezcla} ({regla['de_donde_sale_la_semilla']})"
    )
    obs.decir(
        f"\n  La regla (declarada adentro, {regla['fuente']}):\n"
        f"    «{regla['texto']}»"
    )
    if "orden_de_turnos" in regla:
        obs.decir(
            f" turnos: {' → '.join(regla['orden_de_turnos'])}\n"
            f" repetida: {regla['categoria_repetida']}\n"
            f" ranking: {regla['ranking_dentro_de_la_categoria'].splitlines()[0]}\n"
            f"    imágenes distintas: {regla['imagenes_distintas']}"
        )
        obs.decir("\n  Las 4 consultas que salieron, y por qué esas:")
        for e in regla["elegidas"]:
            obs.decir(
                f"    {e['turno']}. {e['consulta_id']:<16}{e['categoria']:<22}"
                f"{e['artwork_id']:<14}rango {e['rango'][:8]}…  «{e['pedido']}»"
            )
    else:
        obs.decir(
            f" selección: {regla['seleccion_de_consultas']} "
            f"({regla['n_consultas']} consultas, {regla['orden']})\n"
            f" mezcla: {regla['mezcla_dentro_de_la_consulta'].splitlines()[0]}\n"
            f" excluye: {regla['excluye']}"
        )

    c = pool.procedencia["conteos"]
    obs.decir(
        f"\n  {c['n_items']} ítems  ·  {c['n_consultas']} consultas  ·  "
        f"{c['n_obras_distintas']} obras distintas\n"
        f"  **{c['n_juicios_por_anotador']} juicios por persona** "
        f"(~{c.get('minutos_por_anotador_al_ritmo_de_docs_05', '?')} min a 15 s por juicio), "
        f"y hacen falta dos (κ necesita dos)\n"
        f"  de los que **{c.get('duplicado_del_20_por_ciento_de_a5', '?')}** son el 20 % "
        f"duplicado del acuerdo entre anotadores"
    )
    if "aporte_por_configuracion" in c:
        obs.decir(
            f"\n La cuenta de la unión:\n"
            f"    {'fila':<6}{'aporta':>8}{'exclusivos':>12}{'sin juzgar':>12}\n"
            + "\n".join(
                f"    {f:<6}{c['aporte_por_configuracion'][f]:>8}"
                f"{c['exclusivos_por_configuracion'][f]:>12}"
                f"{c['items_del_top5_sin_juzgar_por_configuracion'][f]:>12}"
                for f in pool.nombres_de_configuracion()
            )
            + f"\n    {'suma':<6}{c['suma_de_los_top5_sin_deduplicar']:>8}\n"
            f"    − {c['deduplicados']} deduplicados = **{c['n_items']} a juzgar**  ·  "
            f"ítems por consulta: {c['items_por_consulta']['minimo']}–"
            f"{c['items_por_consulta']['maximo']} (media "
            f"{c['items_por_consulta']['media']})\n"
            f"    🔑 `exclusivos` es cuántos ítems trajo **solo** esa configuración: es "
            f"cuánto\n"
            f"       cambia esa etapa el conjunto a juzgar. `sin juzgar` es la *regla\n"
            f" innegociable* del diseño de la evaluación y tiene que dar 0 acá por construcción."
        )
    obs.decir(f"\n  {'fila':<6}{'consulta_id':<17}{'artwork_id':<14}{'obra':<42}origen")
    for n, fila in enumerate(pool.filas_para_mostrar(), 1):
        if n > FILAS_QUE_SE_LISTAN:
            obs.decir(
                f"  … y {len(pool.items) - FILAS_QUE_SE_LISTAN} filas más: la lista completa "
                f"está en los dos archivos de abajo (`tp3-anotar pool` la vuelve a imprimir)."
            )
            break
        obs.decir(f"  {fila[0]:<6}{fila[1]:<17}{fila[2]:<14}{fila[3]:<42}{fila[4]}")
    obs.decir(
        f"\n  ⚠️ La columna `origen` **no está en el CSV**: el pool ciego no lleva ni la\n"
        f"     configuración ni el `rank`. Viven en el `.json`, que es donde las métricas\n"
        f"     los leen y donde el anotador no mira.\n"
        f"\n pool (ciego): {ruta}\n"
        f" procedencia: {ruta_json}\n{LÍNEA}"
    )

def _controles(args, cfg, almacén, obs, rutas) -> int:
    from tp3_rag.modulo_c import validadores as val

    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    resultados = (
        val.e10_puerta_de_la_anotacion(
            pool=pool,
            manifiesto=manifiesto,
            registros_de_variantes=_registros_de_variantes(cfg, almacén),
        ),
        val.e5_pares_juzgados_en_algun_pool(
            aportes=AporteDeAnotador.todos(almacén, rutas.juicios, split_id=pool.split_id),
            pools=[Pool.leer(almacén, rutas.pool, p)
                   for p in Pool.del_split(almacén, rutas.pool, pool.split_id)],
            split_id=pool.split_id,
        ),
    )
    _mostrar_controles(resultados, obs)
    return 0 if all(r.pasa for r in resultados) else 1

def _mostrar_controles(resultados, obs) -> None:
    obs.decir(
        f"\n{LÍNEA}\nControles del ground truth · se corren **antes de exportar**\n{LÍNEA}\n"
    )
    for r in resultados:
        obs.decir(f"  {'✓' if r.pasa else '✗'}  {r.criterio}")
        obs.decir(f"       {r.marcador}   población: {r.poblacion}")
        for clave, valor in r.detalle.items():
            if clave.startswith("nota"):
                continue
            obs.decir(f"       {clave}: {valor}")
        for falla in r.fallas[:5]:
            obs.decir(f"       ✗ {falla}")
        obs.decir("")
    obs.decir(
        f" El momento importa: este es el último punto en que todavía no se gastó\n"
        f"  en que todavía no se gastó anotación**.\n{LÍNEA}"
    )

def _exportar(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    set_de_consultas = _set_de_consultas(args, almacén, rutas, manifiesto)
    pool = _pool(args, almacén, rutas, manifiesto)
    rubrica = Rúbrica.leer(almacén, rutas.rubricas, args.rubrica)

    resultado = PlanillaDeAnotación(
        cfg, almacén=almacén, rutas=rutas, observabilidad=obs
    ).exportar(
        pool=pool,
        rubrica=rubrica,
        anotador=args.anotador,
        set_de_consultas=set_de_consultas,
        manifiesto=manifiesto,
        registros_de_variantes=_registros_de_variantes(cfg, almacén),
        salida=args.salida,
    )
    _mostrar_controles(resultado.controles, obs)
    obs.decir(
        f"{LÍNEA}\nPlanilla de `{resultado.anotador}` · {resultado.n_filas} juicios\n{LÍNEA}\n"
        f"\n planilla: {resultado.ruta}\n"
        f" cómo anotar: {resultado.ruta_de_instrucciones}\n"
        f" rúbrica: {Rúbrica.ruta_de(rutas.rubricas, resultado.rubrica_id)} "
        f"(hash {rubrica.hash_del_texto})\n"
        f" pool: {resultado.pool_id} (estado {pool.estado})\n"
        f"\n  Las dos columnas que se llenan: **grado** y **nota**.\n"
        + "\n".join(
            f"      {g} = {SIGNIFICADO_DE_LA_ESCALA[g]}" for g in sorted(GRADOS, reverse=True)
        )
        + f"\n\n  🔴 No agregar ni borrar filas, no tocar los ids, **no reordenar**: los tres\n"
        f" son error de import. Si ya reordenaste, ordená por `fila`.\n"
        f"\n  Cuando termine:\n"
        f"      poetry run tp3-anotar importar --planilla {resultado.ruta}\n"
        f"  Si pasa, imprime {resultado.n_filas}/{resultado.n_filas} y congela los juicios. "
        f"**Eso es cómo se sabe que terminó.**\n"
        f"\n  ❌ `planillas/` **no va al repo**: es un borrador. Lo irreproducible es el\n"
        f"     aporte que el import congela en `{rutas.juicios}/`."
    )
    if resultado.imagenes_faltantes:
        obs.decir(
            f"\n  ⚠️  {len(resultado.imagenes_faltantes)} archivo(s) de imagen no están en "
            f"disco (p. ej.\n      {resultado.imagenes_faltantes[0]}). `data/imagenes/` es "
            f"regenerable y no versionado:\n      corré `tp3-ingesta` o mirá la obra por "
            f"`source_url`."
        )
    obs.decir(LÍNEA)
    return 0

def _importar(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    set_de_consultas = _set_de_consultas(args, almacén, rutas, manifiesto)
    planilla = PlanillaDeAnotación(cfg, almacén=almacén, rutas=rutas, observabilidad=obs)
    resultado = planilla.importar(
        planilla=args.planilla,
        set_de_consultas=set_de_consultas,
        buscar_pool=lambda pool_id: Pool.leer(almacén, rutas.pool, pool_id),
        buscar_rubrica=lambda rubrica_id: Rúbrica.leer(almacén, rutas.rubricas, rubrica_id),
    )
    aporte = resultado.aporte
    estado = "ya estaba congelado · se leyó, no se reescribió" if resultado.ya_estaba else (
        "CONGELADO ahora"
    )
    obs.decir(
        f"\n{LÍNEA}\nAporteDeAnotador · {aporte.identidad}  (clase 3 — {estado})\n{LÍNEA}\n"
        f"\n  ✓ **{resultado.n_juicios}/{resultado.n_juicios}** juicios  ·  "
        f"la política de trazado verificado: lo importado **es** lo exportado\n"
        f"\n split_id: {aporte.split_id}\n"
        f" rubrica_id: {aporte.rubrica_id} ← 🔑: sin esto, el piloto y la campaña\n"
        f"                            definitiva comparten nombre de archivo\n"
        f" anotador: {aporte.anotador}\n"
        f" pool_id: {aporte.pool_id} ← procedencia, **no** ingrediente\n"
        f"\n  distribución: "
        + " · ".join(f"{k} = {v}" for k, v in sorted(resultado.distribucion.items()))
        + f"\n\n juicios: {resultado.ruta}\n"
        f" ✅ Va al repo: es **irreproducible**. Agregalo por ruta explícita.\n"
        f"\n  Cuando haya **dos** aportes de la misma rúbrica:\n"
        f"      poetry run tp3-anotar kappa --rubrica {aporte.rubrica_id}\n{LÍNEA}"
    )
    return 0

def _kappa(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    rubrica = Rúbrica.leer(almacén, rutas.rubricas, args.rubrica)
    pareja = tuple(args.anotador) if args.anotador else None
    if pareja is not None and len(pareja) != 2:
        raise ErrorDeArtefacto(
            f"se pasaron {len(pareja)} `--anotador` y **κ de Cohen es entre dos**. El "
            f"tercero no promedia: sus juicios son el **desempate** y entran al consenso "
            f"(el diseño de la evaluación, el corte del alcance)."
        )
    resultado = Acuerdo(cfg, almacén=almacén, rutas=rutas, observabilidad=obs).calcular(
        pool=pool,
        rubrica=rubrica,
        aportes=AporteDeAnotador.todos(almacén, rutas.juicios, split_id=pool.split_id),
        pareja=pareja,
    )
    k = resultado.kappa
    obs.decir(
        f"\n{LÍNEA}\nκ de Cohen · pool {resultado.pool_id} · rúbrica "
        f"{resultado.rubrica_id}\n{LÍNEA}\n"
        f"\n anotadores: {k.anotadores[0]} vs {k.anotadores[1]}\n"
        f" población: {k.poblacion}\n"
        f"\n  acuerdo observado      po = {k.acuerdo_observado:.3f}\n"
        f"  acuerdo esperado        pe = {k.acuerdo_esperado:.3f}\n"
        f"  **κ = "
        + ("ausente" if k.kappa is None else f"{k.kappa:.3f}")
        + f"** (umbral del acuerdo entre anotadores: {UMBRAL})"
    )
    if k.matriz:
        obs.decir(f"\n  matriz de confusión ({k.anotadores[0]} ↓ / {k.anotadores[1]} →)")
        obs.decir("        " + "".join(f"{f'g{g}':>6}" for g in GRADOS))
        for g1 in GRADOS:
            obs.decir(
                f"    g{g1}  "
                + "".join(f"{k.matriz[f'grado_{g1}'][f'grado_{g2}']:>6}" for g2 in GRADOS)
            )
    for quién, pares in k.solo_uno.items():
        if pares:
            obs.decir(
                f"\n  ⚠️  {len(pares)} par(es) que juzgó solo {quién} **no entran al κ** y se "
                f"declaran: un número sobre una población que no es la que la decisión usa "
                f"es el modo de falla 3 del proyecto."
            )
    por_resolución: dict[str, int] = {}
    for f in resultado.consenso:
        por_resolución[f.como_se_resolvio] = por_resolución.get(f.como_se_resolvio, 0) + 1
    obs.decir(f"\n  consenso: {por_resolución}")
    if por_resolución.get("sin_consenso"):
        obs.decir(
            f"     {por_resolución['sin_consenso']} par(es) **sin consenso**: los dos "
            f"difieren y no hay un tercer aporte.\n"
            f" el diseño de la evaluación: el desempate del tercer integrante es **una anotación "
            f"más**, no una\n"
            f"     corrección del consenso. Importá su planilla y el consenso se recalcula."
        )
    obs.decir(f"\n  {k.veredicto.replace('{rubrica}', resultado.rubrica_id)}")
    obs.decir(
        f"\n consenso: {resultado.ruta_del_consenso}\n"
        f" κ: {resultado.ruta_del_kappa}\n"
        f" ✅ Los dos van al repo: son la lectura que el acuerdo entre anotadores publica.\n"
        f"\n ⚠️ Este κ es el **del piloto** (20 juicios) y el del acuerdo entre anotadores es otro (~108 "
        f"duplicados\n     del pool definitivo). **No se suman ni se promedian** "
        f".\n{LÍNEA}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
