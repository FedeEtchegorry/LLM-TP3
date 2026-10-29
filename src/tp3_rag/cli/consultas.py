from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tp3_rag.dominio.catalogo import CatálogoCanónico
from tp3_rag.dominio.consultas import CATEGORÍAS, SetDeConsultas
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.modulo_c.candidatas import CUPO, PropuestaDeCandidatas
from tp3_rag.modulo_c.set_de_consultas import (
    AutorDelSetDeConsultas,
    Declaración,
    verificar_hash_contra_el_catalogo,
)
from tp3_rag.modulo_c.vocabulario_de_culture import VocabularioDeCulture
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

LÍNEA = "─" * 78

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-consultas",
        description=(
            "SetDeConsultas — 30 consultas = 10 imágenes del held-out × 3 "
            "pedidos. Clase 3: se autora una vez y se congela. Las 10 imágenes y los "
            "30 pedidos los elige el grupo; esta herramienta propone, valida y congela."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], metavar="clave=valor",
        help="override de una clave declarada; una clave desconocida es un error",
    )
    p.add_argument("--catalogo-id", help="cuál catálogo, si hay más de uno en disco")
    p.add_argument("--split-id", help="cuál split, si hay más de uno congelado")
    p.add_argument("--silencioso", action="store_true")

    sub = p.add_subparsers(dest="comando", required=True)

    c = sub.add_parser(
        "candidatas",
        help="🟡 PROPUESTA de ~30 candidatas del held-out, estratificada y determinista",
    )
    c.add_argument("--cupo", type=int, default=CUPO)
    c.add_argument(
        "--incluir", action="append", default=[], metavar="artwork_id",
        help="forzar una obra del held-out (p. ej. para rescatar un valor de la cola)",
    )
    c.add_argument("--salida", type=Path, help="escribir la propuesta como JSON")

    v = sub.add_parser(
        "culturas",
        help="el vocabulario de `culture` DEL CORPUS con su n, y los grupos propuestos",
    )
    v.add_argument("--tope", type=int, default=40, help="cuántas etiquetas mostrar")
    v.add_argument("--todas", action="store_true", help="mostrar las etiquetas con n = 1")
    v.add_argument("--salida", type=Path, help="escribir el vocabulario como JSON")

    pl = sub.add_parser("plantilla", help="emitir el TOML de declaración a llenar")
    pl.add_argument("--autoria", default="revision-1")
    pl.add_argument("--estado", default="completo", help="plantilla | completo")
    pl.add_argument(
        "--imagen", action="append", default=[], metavar="artwork_id",
        help="los artwork_id elegidos; si no se pasan, quedan como REEMPLAZAR",
    )
    pl.add_argument("--revisadas", type=int, help="cuántas imágenes se revisaron")
    pl.add_argument("--salida", type=Path, help="dónde escribirla (por omisión, stdout)")

    a = sub.add_parser("autorar", help="validar y CONGELAR el set desde una declaración")
    a.add_argument("--declaracion", type=Path, required=True)
    a.add_argument(
        "--derivar-otro", action="store_true",
        help=(
            "🔴 autorar OTRO set habiendo uno congelado para el mismo split. Cuesta la "
            "comparabilidad de los juicios del anterior (la tabla de identidades, clase 3)"
        ),
    )

    for nombre, ayuda in (
        ("validar", "leer y validar los sets congelados (incluidos los bytes, )"),
        ("mostrar", "mostrar un set congelado"),
    ):
        s = sub.add_parser(nombre, help=ayuda)
        s.add_argument("--autoria", help="cuál set, si hay más de uno para el split")
        if nombre == "validar":
            s.add_argument(
                "--contra-el-catalogo", action="store_true",
                help="además, verificar el `imagen_hash` contra el catálogo (procedencia)",
            )
    return p.parse_args(argv)

def main(argv=None) -> int:
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
        raíz = cfg["rutas.ground_truth_consultas"]

        if args.comando == "plantilla":
            return _plantilla(args, cfg, almacén, obs, raíz)
        if args.comando in ("validar", "mostrar"):
            return _leer(args, cfg, almacén, obs, raíz)

        manifiesto, catálogo = _artefactos(args, cfg, almacén, obs)
        if args.comando == "candidatas":
            return _candidatas(args, cfg, obs, manifiesto, catálogo)
        if args.comando == "culturas":
            return _culturas(args, cfg, obs, manifiesto, catálogo)
        if args.comando == "autorar":
            return _autorar(args, cfg, almacén, obs, manifiesto, catálogo)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0  # pragma: no cover - argparse exige un subcomando

def _artefactos(args, cfg, almacén, obs):
    manifiesto = _manifiesto(args, cfg, almacén)
    obs.decir(
        f"split: {manifiesto.split_id} · {manifiesto.conteos['corpus_indexado']} "
        f"corpus + {manifiesto.conteos['held_out']} held-out  (clase 3, congelado)"
    )
    base = (
        Path(cfg["rutas.catalogo"]) / args.catalogo_id
        if args.catalogo_id
        else CatálogoCanónico.único(almacén, cfg["rutas.catalogo"])
    )
    catálogo = CatálogoCanónico.cargar(almacén, base)
    obs.decir(
        f"catálogo: {catálogo.catalogo_id} ✓ · {len(catálogo.registros)} registros "
        f"(re-verificado)"
    )
    faltantes = manifiesto.cobertura_contra({r.artwork_id for r in catálogo.registros})
    if faltantes:
        obs.decir(
            f"⚠️  el catálogo no cubre {len(faltantes)} ids del lado corpus del "
            f"manifiesto (p. ej. {list(faltantes[:3])}): el `n` de las `cultura_pedida` "
            f"saldría sobre una población más chica que la declarada."
        )
    return manifiesto, catálogo

def _manifiesto(args, cfg, almacén) -> ManifiestoDeSplit:
    from tp3_rag.nucleo.errores import ErrorDeArtefacto

    raíz = cfg["rutas.artefactos_splits"]
    if args.split_id:
        return ManifiestoDeSplit.leer(almacén, raíz, args.split_id)
    existentes = ManifiestoDeSplit.existentes(almacén, raíz)
    if not existentes:
        raise ErrorDeArtefacto(
            "no hay ningún split congelado. El `SetDeConsultas` **ancla a un split** "
            ": corré `poetry run tp3-split` primero."
        )
    if len(existentes) > 1:
        raise ErrorDeArtefacto(
            f"hay {len(existentes)} splits congelados: {list(existentes)}. Pasá "
            f"`--split-id`: el set **ancla a un split** y elegir por `mtime` "
            f"sería tomar una decisión de alcance por fecha de modificación."
        )
    return ManifiestoDeSplit.leer(almacén, raíz, existentes[0])

def _candidatas(args, cfg, obs, manifiesto, catálogo) -> int:
    held = set(manifiesto.ids_held_out)
    registros = [r for r in catálogo.registros if r.artwork_id in held]
    propuesta = PropuestaDeCandidatas(cfg).proponer(
        registros, cupo=args.cupo, incluir=args.incluir
    )

    obs.decir(
        f"\n{LÍNEA}\nLista corta de candidatas  —  🟡 **PROPUESTA, no decisión**\n{LÍNEA}\n"
        f"\n  {len(propuesta.candidatas)} de las {len(registros)} del held-out.\n"
        f"  **Las 10 las elige el grupo, y los `atributos` también**: paisaje, retrato,\n"
        f"  interior, bodegón, abstracta, paleta fría/cálida **no existen en el esquema\n"
        f" canónico**. Lo que esta lista estratifica es lo que la\n"
        f"  metadata sí dice."
    )
    c = propuesta.criterio
    obs.decir(
        f"\n criterio: {c['metodo']}\n"
        f"             ejes = {' → '.join(c['ejes'])}\n"
        f"             dentro de cada valor: {c['seleccion_dentro_del_valor']} "
        f"(semilla {c['semilla']}, la del split)\n"
        f"             criterio de culture = {c['criterio_de_culture']}"
    )

    obs.decir(f"\n  {'artwork_id':<14}{'museo':<6}{'periodo':<12}{'cultura':<18}"
              f"{'tecnica':<26}{'art':<4}título")
    for cand in propuesta.candidatas:
        título = (cand.title or "—")[:38]
        obs.decir(
            f"  {cand.artwork_id:<14}{cand.museo:<6}{cand.periodo:<12}"
            f"{cand.cultura[:17]:<18}{cand.tecnica:<26}"
            f"{'sí' if cand.tiene_artista else 'no':<4}{título}"
        )

    obs.decir("\n  cobertura de los ejes (valores cubiertos / valores del held-out)")
    for eje, datos in propuesta.cobertura.items():
        obs.decir(
            f"    {eje:<16}{datos['valores_cubiertos']:>3} / "
            f"{datos['valores_del_held_out']:<4}  {datos['reparto']}"
        )

    obs.decir("\n  qué quedó afuera, y por qué (la cola: valores de 1–2 obras)")
    for eje in propuesta.criterio["ejes"]:
        d = propuesta.afuera[eje]
        muestra = list(d["valores"])[:6]
        obs.decir(
            f"    {eje:<16}{d['n_valores_sin_cubrir']:>3} valores · "
            f"{d['n_obras_en_esos_valores']:>3} obras   {muestra}"
        )
    obs.decir(f"\n  {propuesta.afuera['_nota']}")
    obs.decir(
        "\n  Las imágenes están en `data/imagenes/<artwork_id>.jpg` (con `_` en vez de "
        "`:`).\n  Miralas, elegí 10, y después:  "
        "`tp3-consultas plantilla --imagen <id> --imagen <id> …`"
    )
    obs.decir(LÍNEA)

    if args.salida:
        args.salida.write_text(
            json.dumps(propuesta.a_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        obs.decir(f"  propuesta escrita en {args.salida}")
    return 0

def _culturas(args, cfg, obs, manifiesto, catálogo) -> int:
    criterio = cfg["normalizacion.culture.criterio"]
    vocabulario = VocabularioDeCulture.del_corpus_del_split(
        catálogo, manifiesto, criterio=criterio
    )
    dist = vocabulario.distribucion()

    obs.decir(
        f"\n{LÍNEA}\nVocabulario de `culture` — la distribución de culture del corpus y el insumo del criterio de normalización de culture\n{LÍNEA}\n"
        f"\n población: {dist['poblacion']} · n = {dist['n_poblacion']}\n"
        f" criterio: {dist['criterio']} (el criterio de normalización de culture sigue **abierto**)\n"
        f" etiquetas: {dist['n_etiquetas_distintas']} distintas ·  "
        f"{dist['n_obras_sin_culture']} obras sin `culture`\n"
        f"\n 🔴 La población es **el corpus**, que es la que la cobertura cultural del resultado usa. Las cifras de\n"
        f" el criterio de normalización de culture están medidas sobre el **universo** (5.915) y son 2–3× más\n"
        f"  grandes. Las dos son correctas; solo una es la que la decisión necesita."
    )

    etiquetas = list(dist["etiquetas"].items())
    mostradas = etiquetas if args.todas else etiquetas[: args.tope]
    obs.decir(f"\n  {'etiqueta':<38}{'n':>6}")
    for etiqueta, n in mostradas:
        obs.decir(f"    {etiqueta:<36}{n:>6}")
    if len(mostradas) < len(etiquetas):
        cola = sum(n for _, n in etiquetas[args.tope :])
        obs.decir(
            f"    … {len(etiquetas) - len(mostradas)} etiquetas más, {cola} obras "
            f"(`--todas` las muestra)"
        )

    obs.decir(
        f"\n{LÍNEA}\nGrupos de sinónimos — 🟡 **PROPUESTA**, insumo del criterio de normalización de culture\n{LÍNEA}\n"
        f"\n  **Qué etiquetas forman cada `cultura_pedida` lo decide el grupo.** Esto es\n"
        f"  el trabajo de mirar las {dist['n_etiquetas_distintas']} etiquetas y marcar "
        f"dónde el vocabulario parte\n  una cultura en sinónimos, con el `n` de cada "
        f"unión medido sobre este corpus."
    )
    for grupo in vocabulario.grupos_propuestos():
        obs.decir(
            f"\n  ── {grupo['nombre']}  (`{grupo['clave']}`)\n"
            f" n de la unión: {grupo['n']}"
            + (
                f"   (suma de etiquetas {sum(grupo['n_por_etiqueta'].values())}, "
                f"solapamiento {grupo['solapamiento']})"
                if grupo["solapamiento"]
                else ""
            )
        )
        obs.decir(
            "     etiquetas: "
            + ", ".join(f"{e} ({n})" for e, n in grupo["n_por_etiqueta"].items())
        )
        if grupo["ausentes"]:
            obs.decir(f" ⚠️ ausentes: {grupo['ausentes']}")
        if grupo["dudosas"]:
            obs.decir(
                f"     dudosas ({grupo['n_dudosas']} obras) — **las decide una persona**:"
            )
            for etiqueta, razón in grupo["dudosas"].items():
                obs.decir(f"       · {etiqueta} ({vocabulario.n_de(etiqueta)}) — {razón}")
        obs.decir(f" por qué: {grupo['por_que']}")

    obs.decir(
        f"\n{LÍNEA}\n  Para usar un grupo en un pedido cultural, va así en la declaración:\n"
        f'    cultura_pedida = {{ nombre = "la India", etiquetas = ["Mughal India", "India", …] }}\n'
        f" El `n` **no se escribe**: lo computa `autorar` contra el corpus, y una\n"
        f"  etiqueta que no existe es **error duro**, no un `n = 0`.\n{LÍNEA}"
    )

    if args.salida:
        args.salida.write_text(
            json.dumps(
                {**dist, "grupos_propuestos": vocabulario.grupos_propuestos()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        obs.decir(f"  vocabulario escrito en {args.salida}")
    return 0

def _plantilla(args, cfg, almacén, obs, raíz) -> int:
    split_id = args.split_id or (
        ManifiestoDeSplit.existentes(almacén, cfg["rutas.artefactos_splits"]) or ["REEMPLAZAR"]
    )[0]
    texto = AutorDelSetDeConsultas(cfg, almacén=almacén).plantilla(
        split_id=split_id,
        autoria=args.autoria,
        artwork_ids=args.imagen,
        n_imagenes_revisadas=args.revisadas,
        estado=args.estado,
    )
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(texto, encoding="utf-8")
        obs.decir(
            f"plantilla escrita en {args.salida}\n"
            f" Llenala y congelala:\n"
            f"    poetry run tp3-consultas autorar --declaracion {args.salida}"
        )
    else:
        print(texto)
    return 0

def _autorar(args, cfg, almacén, obs, manifiesto, catálogo) -> int:
    declaración = Declaración.desde_toml(args.declaracion)
    resultado = AutorDelSetDeConsultas(cfg, almacén=almacén, observabilidad=obs).autorar(
        declaración,
        catálogo=catálogo,
        manifiesto=manifiesto,
        derivar_otro=args.derivar_otro,
    )
    _mostrar(resultado.set_de_consultas, obs, ruta=resultado.ruta,
             ya_estaba=resultado.ya_estaba_congelado)
    if not resultado.ya_estaba_congelado:
        n = len(resultado.rutas_de_imagenes)
        obs.decir(
            f"\n 🔴 · {n} {'imagen' if n == 1 else 'imágenes'} **versionada"
            f"{'' if n == 1 else 's'}** en {cfg['rutas.ground_truth_consultas']}/imagenes/\n"
            f"     Van al repo: son parte del instrumento de medición, no del corpus "
            f"(el set de consultas, la regla de qué se versiona).\n"
            f"     Agregalas por ruta explícita — este repo **no usa `git add -A`**."
        )
    return 0

def _leer(args, cfg, almacén, obs, raíz) -> int:
    existentes = SetDeConsultas.existentes(almacén, raíz)
    if args.autoria:
        split_id = args.split_id or (
            ManifiestoDeSplit.existentes(almacén, cfg["rutas.artefactos_splits"]) or [""]
        )[0]
        from tp3_rag.nucleo import identidad

        existentes = [
            identidad.identidad_del_set_de_consultas(split_id=split_id, autoria=args.autoria)
        ]
    if not existentes:
        print(
            "no hay ningún `SetDeConsultas` congelado todavía.\n"
            "  Sacá la plantilla con `poetry run tp3-consultas plantilla`.",
            file=sys.stderr,
        )
        return 1

    catálogo = None
    if args.comando == "validar" and getattr(args, "contra_el_catalogo", False):
        base = (
            Path(cfg["rutas.catalogo"]) / args.catalogo_id
            if args.catalogo_id
            else CatálogoCanónico.único(almacén, cfg["rutas.catalogo"])
        )
        catálogo = CatálogoCanónico.cargar(almacén, base)

    for ident in existentes:
        set_ = SetDeConsultas.leer(almacén, raíz, ident)
        if catálogo is not None:
            verificar_hash_contra_el_catalogo(set_, catálogo)
        if args.comando == "validar":
            obs.decir(
                f"✓ {ident}  ·  estado = {set_.estado}  ·  "
                f"{set_.contenido['conteos']['n_imagenes']} imágenes / "
                f"{set_.contenido['conteos']['n_consultas']} consultas  ·  "
                f"bytes verificados ({len(set_.imagenes())} archivos)"
                + ("  ·  `imagen_hash` ✓ contra el catálogo" if catálogo else "")
            )
            if not set_.es_usable_para_medir:
                obs.decir(
                    f"  ⚠️  `estado = {set_.estado}`: **no es usable para medir**. "
                    f"el diseño de la evaluación compromete 30 consultas y el pipeline base completo y andando mide *30/30*."
                )
        else:
            _mostrar(set_, obs, ruta=almacén.ruta(SetDeConsultas.ruta_de(raíz, ident)),
                     ya_estaba=True)
    return 0

def _mostrar(set_: SetDeConsultas, obs: Observabilidad, *, ruta, ya_estaba: bool) -> None:
    estado = (
        "ya estaba congelado · se leyó, no se reescribió" if ya_estaba else "CONGELADO ahora"
    )
    obs.decir(f"\n{LÍNEA}\nSetDeConsultas · {set_.identidad}  (clase 3 — {estado})\n{LÍNEA}")

    c = set_.contenido["conteos"]
    obs.decir(
        f"\n split_id: {set_.split_id} ← el ancla\n"
        f" autoría: {set_.autoria} ← la mitad declarada de la identidad\n"
        f" estado: {set_.estado}"
        + ("" if set_.es_usable_para_medir else "   ⚠️ no es usable para medir")
        + f"\n criterio de `culture`: {set_.criterio_de_culture}   "
        f"← declarado adentro, **no** en el id ()\n"
        f" revisadas: {set_.n_imagenes_revisadas} de "
        f"{set_.contenido['revision'].get('n_held_out', '—')} del held-out, para elegir "
        f"{c['n_imagenes']}"
    )
    obs.decir(
        f"\n  {c['n_imagenes']} {'imagen' if c['n_imagenes'] == 1 else 'imágenes'} × 3 "
        f"pedidos = {c['n_consultas']} consultas  ·  "
        f"{c['n_pedidos_culturales']} culturales\n"
        f" por categoría (lo que la tabla de ablación desglosa y el delta por categoría de pedido usa): {c['por_categoria']}"
    )

    for img in set_.imagenes():
        p = img.procedencia or {}
        marca = "  [EJEMPLO]" if img.es_ejemplo else ""
        obs.decir(
            f"\n  ── {img.artwork_id}{marca}  ·  {p.get('title') or '—'}"
            f"  ({p.get('artist') or 'sin artista'}, {p.get('date_display') or '—'})\n"
            f" atributos (): {list(img.atributos)}\n"
            f" imagen_hash: {img.imagen_hash[:30]}… · {img.archivo}"
        )
        for consulta in img.consultas:
            obs.decir(f"     · {consulta.consulta_id:<16}{consulta.categoria:<22}"
                      f"«{consulta.pedido}»")
            if consulta.cultura_pedida:
                cp = consulta.cultura_pedida
                obs.decir(
                    f"       cultura_pedida: {cp.nombre} = {list(cp.etiquetas)}\n"
                    f" la cobertura cultural del resultado · n = {cp.n} obras de {cp.n_poblacion} "
                    f"({cp.poblacion})\n"
                    f"       desglose: {cp.n_por_etiqueta}"
                )

    faltan = [cat for cat in CATEGORÍAS if not set_.por_categoria().get(cat)]
    if faltan:  # pragma: no cover - `validar` lo vuelve inalcanzable
        obs.decir(f"\n  ⚠️ sin ninguna consulta de categoría {faltan}")

    pr = set_.contenido.get("procedencia", {})
    obs.decir(
        f"\n  procedencia (declarada adentro, NO en la identidad)\n"
        f"    catalogo_id = {pr.get('catalogo_id')}  ·  "
        f"commit = {(pr.get('commit') or '—')[:12]}\n"
        f"    autorado   = {pr.get('fecha_de_autoria')}\n"
        f"\n set: {ruta}\n{LÍNEA}"
    )

if __name__ == "__main__":
    raise SystemExit(main())
