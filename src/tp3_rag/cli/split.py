from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.catalogo import CatálogoCanónico
from tp3_rag.dominio.split import ManifiestoDeSplit
from tp3_rag.modulo_a.normalizador import VERSIÓN_DEL_NORMALIZADOR
from tp3_rag.modulo_a.particionador import Particionador
from tp3_rag.modulo_a.selector_de_corpus import SelectorDeCorpus
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-split",
        description=(
            "SelectorDeCorpus → elegibles con texto e imagen · Particionador → "
            "corpus indexado + held-out · ManifiestoDeSplit congelado (el dataset documentado y reproducible, la exclusión del held-out de toda colección)."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument(
        "--set", action="append", default=[], metavar="clave=valor",
        help="override de una clave declarada; una clave desconocida es un error",
    )
    p.add_argument("--catalogo-id", help="cuál catálogo, si hay más de uno en disco")
    p.add_argument(
        "--mostrar", action="store_true",
        help="no deriva nada: muestra el manifiesto que ya está congelado",
    )
    p.add_argument(
        "--derivar-otro", action="store_true",
        help=(
            "derivar OTRO split habiendo uno congelado. Cuesta la anotación del "
            "anterior: el SetDeConsultas se ancla por `split_id`"
        ),
    )
    p.add_argument("--silencioso", action="store_true")
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

        if args.mostrar:
            existentes = ManifiestoDeSplit.existentes(almacén, cfg["rutas.artefactos_splits"])
            if not existentes:
                print("no hay ningún split congelado todavía", file=sys.stderr)
                return 1
            for sid in existentes:
                _mostrar(
                    ManifiestoDeSplit.leer(almacén, cfg["rutas.artefactos_splits"], sid),
                    obs,
                    ruta=almacén.ruta(
                        ManifiestoDeSplit.ruta_de(cfg["rutas.artefactos_splits"], sid)
                    ),
                    ya_estaba=True,
                )
            return 0

        base = (
            Path(cfg["rutas.catalogo"]) / args.catalogo_id
            if args.catalogo_id
            else CatálogoCanónico.único(almacén, cfg["rutas.catalogo"])
        )
        obs.decir(f"catálogo: {base} (re-verificando el `catalogo_id`…)")
        catálogo = CatálogoCanónico.cargar(almacén, base)
        obs.decir(
            f"            {len(catálogo.registros)} registros · "
            f"catalogo_id = {catálogo.catalogo_id} ✓"
        )

        selector = SelectorDeCorpus(cfg["ingesta.criterio_de_inclusion"])
        elegibles = selector.seleccionar(catálogo.registros)
        obs.decir(
            f"\nSelectorDeCorpus · criterio = {selector.criterio}\n"
            f"            {elegibles.n_universo} universo normalizado → "
            f"{len(elegibles.registros)} elegibles con texto e imagen (pre-split)\n"
            f"            descartadas por condición: "
            f"{elegibles.descartadas_por_condicion}"
        )

        resultado = Particionador(cfg, almacén=almacén, observabilidad=obs).congelar(
            elegibles,
            catalogo_id=catálogo.catalogo_id,
            version_del_normalizador=VERSIÓN_DEL_NORMALIZADOR,
            derivar_otro=args.derivar_otro,
        )
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _mostrar(
        resultado.manifiesto, obs, ruta=resultado.ruta, ya_estaba=resultado.ya_estaba_congelado
    )
    return 0

def _mostrar(manifiesto: ManifiestoDeSplit, obs: Observabilidad, *, ruta, ya_estaba: bool) -> None:
    m = manifiesto.a_json()
    línea = "─" * 78
    estado = "ya estaba congelado · se leyó, no se reescribió" if ya_estaba else "CONGELADO ahora"
    obs.decir(f"\n{línea}\nManifiestoDeSplit · {manifiesto.split_id}  (clase 3 — {estado})\n{línea}")

    c = m["conteos"]
    obs.decir(
        f"\n  los TRES conteos (este artefacto es su dueño)\n"
        f" elegibles con texto e imagen: {c['elegibles_con_texto_e_imagen']:>5}\n"
        f" corpus indexado: {c['corpus_indexado']:>5} ← la cifra publicable\n"
        f" held-out: {c['held_out']:>5} ← no entra a ninguna colección"
    )

    e = m["definicion_de_estrato"]
    tramos = " · ".join(t["nombre"] for t in e["tramos"])
    obs.decir(
        f"\n semilla: {m['semilla']} proporción: {m['proporcion_held_out']}\n"
        f" criterio: {m['criterio_de_inclusion']}\n"
        f" estrato: {' × '.join(e['ejes'])} sobre `{e['campo_de_periodo']}`\n"
        f"             tramos: {tramos} · {e['etiqueta_sin_periodo']}\n"
        f" reparto: {m['procedencia']['metodo_de_reparto']}\n"
        f"  selección: {m['procedencia']['metodo_de_seleccion']}"
    )

    obs.decir("\n  held-out por estrato")
    obs.decir(f"    {'estrato':<24}{'elegibles':>10}{'corpus':>8}{'held-out':>10}{'%':>7}")
    for clave, d in m["estratos"].items():
        obs.decir(
            f"    {clave:<24}{d['n_elegibles']:>10}{d['n_corpus']:>8}"
            f"{d['n_held_out']:>10}{d['pct_held_out']:>6.1f}%"
        )

    dist = m["distribucion_del_held_out"]
    for eje in e["ejes"]:
        obs.decir(f"\n  distribución del held-out por {eje}")
        for valor, d in dist[f"por_{eje}"].items():
            obs.decir(
                f"    {valor:<24}{d['n_elegibles']:>10}{'':>8}{d['n_held_out']:>10}"
                f"{d['pct']:>6.1f}%"
            )

    obs.decir(
        f"\n  procedencia (declarada adentro, NO en la identidad)\n"
        f"    catalogo_id = {m['procedencia']['catalogo_id']}  ·  "
        f"commit = {(m['procedencia']['commit'] or '—')[:12]}\n"
        f"    derivado   = {m['procedencia']['fecha_de_derivacion']}"
    )
    obs.decir(f"\n manifiesto: {ruta}")
    obs.decir(línea)

if __name__ == "__main__":
    raise SystemExit(main())
