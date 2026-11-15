"""Medición del prefiltro de pedidos vagos.

    validar   el prefiltro sobre 12 pedidos fuera del set (gasta)
    sugerir   el prefiltro sobre las 30 consultas (gasta)
    correr    la fila final con el pedido sugerido en las consultas vagas (gasta)
    evaluar   cada consulta aclarada contra la fila final (no gasta)
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path

import cascada as c

sys.path.insert(0, str(c.RAIZ / "src"))
from tp3_rag.modulo_b.aclarador import VERSIÓN_DEL_PROMPT  # noqa: E402
from tp3_rag.modulo_b.confianza import confianza_del_top5  # noqa: E402

VARIANTE_ID = "c3274567371ff42b"
CONFIG_FINAL = c.RAIZ / "config" / "analisis" / "c4-filtro-metadata.toml"
FINAL = "C4·filtro100·metadata·rerank20"
VALIDACION = Path(__file__).resolve().parent / "pedidos-de-validacion.json"

V = f"v{VERSIÓN_DEL_PROMPT}"
ACLARACIONES = c.SALIDA / f"aclaraciones-{V}.json"
CORRIDAS_ACLARADAS = c.RAIZ / "artifacts" / f"corridas-aclaradas-{V}"
OVERRIDES = {
    "rutas.artefactos_corridas": f"artifacts/corridas-aclaradas-{V}",
    "rutas.artefactos_planes": "artifacts/planes-aclarados",
}


def _montar(config, overrides):
    from tp3_rag.modulo_b import montaje as mnt
    from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
    from tp3_rag.nucleo.credenciales import Credenciales
    from tp3_rag.nucleo.observabilidad import Observabilidad
    from tp3_rag.puertos import trazas as trazas_mod
    from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos

    cfg = Configuración.cargar(archivo=config, overrides=overrides)
    almacen = AlmacenamientoEnArchivos(RAÍZ_DEL_REPO)
    credenciales = Credenciales.desde_entorno()
    obs = Observabilidad(silencioso=True)
    obs.trazas = trazas_mod.armar(cfg, credenciales)
    montaje = mnt.montar(cfg, almacen, variante_id=VARIANTE_ID, observabilidad=obs, credenciales=credenciales)
    return cfg, almacen, obs, montaje


def _cerrar(montaje, obs) -> None:
    montaje.cerrar()
    if hasattr(obs.trazas, "cerrar"):
        obs.trazas.cerrar()


def sugerir(args) -> None:
    from tp3_rag.dominio.binding import BindingDelAclarador
    from tp3_rag.modulo_b import aclarador, montaje as mnt

    cfg, almacen, obs, montaje = _montar(CONFIG_FINAL, OVERRIDES)
    try:
        binding = BindingDelAclarador.desde_configuración(cfg, puerto=montaje.puerto_de_llm, observabilidad=obs)
        set_ = mnt.set_de_consultas(cfg, almacen, montaje.manifiesto, None)
        salida = {}
        for consulta in set_.consultas():
            datos = mnt.bytes_de_la_imagen_del_set(cfg, almacen, consulta)
            a = aclarador.aclarar(binding, pedido=consulta.pedido, datos_de_imagen=datos)
            salida[consulta.consulta_id] = {"categoria": c.categoria(consulta.consulta_id), "pedido": consulta.pedido, **a.a_json()}
            print(f"{consulta.consulta_id:<16} {a.nivel:<9} {consulta.pedido!r} -> {a.sugerencia!r}")
    finally:
        _cerrar(montaje, obs)
    c.SALIDA.mkdir(parents=True, exist_ok=True)
    ACLARACIONES.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    _resumen_del_prefiltro(salida)
    print(f"Escrito en {ACLARACIONES.relative_to(c.RAIZ)}")


def validar(args) -> None:
    from tp3_rag.dominio.binding import BindingDelAclarador
    from tp3_rag.modulo_b import aclarador, montaje as mnt

    casos = json.loads(VALIDACION.read_text(encoding="utf-8"))["pedidos"]
    cfg, almacen, obs, montaje = _montar(CONFIG_FINAL, OVERRIDES)
    try:
        binding = BindingDelAclarador.desde_configuración(cfg, puerto=montaje.puerto_de_llm, observabilidad=obs)
        por_id = {q.consulta_id: q for q in mnt.set_de_consultas(cfg, almacen, montaje.manifiesto, None).consultas()}
        salida = []
        for caso in casos:
            datos = mnt.bytes_de_la_imagen_del_set(cfg, almacen, por_id[f"{caso['imagen']}-ambiguo"])
            a = aclarador.aclarar(binding, pedido=caso["pedido"], datos_de_imagen=datos)
            salida.append({**caso, **a.a_json()})
            print(f"esperado {caso['esperado']:<9} obtuvo {a.nivel:<9} {caso['pedido']!r} -> {a.sugerencia!r}")
    finally:
        _cerrar(montaje, obs)
    destino = c.SALIDA / f"validacion-{V}.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    frena = [x for x in salida if x["nivel"] == "vago"]
    vagos = [x for x in salida if x["esperado"] == "vago"]
    otros = [x for x in salida if x["esperado"] != "vago"]
    print(f"  vagos que frena: {sum(x['nivel'] == 'vago' for x in vagos)} de {len(vagos)}")
    print(f"  concretos y medios que frena sin necesidad: {sum(x['nivel'] == 'vago' for x in otros)} de {len(otros)}")
    print(f"  nivel exacto: {sum(x['nivel'] == x['esperado'] for x in salida)} de {len(salida)}")
    print(f"  frena en total: {len(frena)}. Escrito en {destino.relative_to(c.RAIZ)}")


def _resumen_del_prefiltro(aclaraciones: dict) -> None:
    por_categoria = collections.Counter((a["categoria"], a["nivel"] == "vago") for a in aclaraciones.values())
    for cat in c.CATEGORIAS:
        print(f"  {cat}: vago {por_categoria[(cat, True)]} de {por_categoria[(cat, True)] + por_categoria[(cat, False)]}")
    tp = por_categoria[("ambiguo", True)]
    avisadas = sum(n for (_, vago), n in por_categoria.items() if vago)
    print(f"  contra las 10 consultas ambiguas: precisión {tp}/{avisadas}, cobertura {tp}/10")
    tokens = sum(a["uso"].get("tokens_in", 0) for a in aclaraciones.values()), sum(a["uso"].get("tokens_out", 0) for a in aclaraciones.values())
    print(f"  tokens del prefiltro en las {len(aclaraciones)} consultas: {tokens[0]} de entrada y {tokens[1]} de salida")


def _aclaraciones() -> dict:
    if not ACLARACIONES.exists():
        sys.exit(f"Falta {ACLARACIONES.relative_to(c.RAIZ)}. Correr `aclarar.py sugerir` primero.")
    return json.loads(ACLARACIONES.read_text(encoding="utf-8"))


def correr(args) -> None:
    from tp3_rag.dominio.consulta import EntradaDeConsulta
    from tp3_rag.dominio.corrida import AlmacenDeCorridas
    from tp3_rag.dominio.plan import AlmacenDePlanes
    from tp3_rag.modulo_b import montaje as mnt
    from tp3_rag.modulo_b.armador import ArmadorDePipeline
    from tp3_rag.modulo_b.ejecutor import EjecutorDeEtapas

    aclaraciones = _aclaraciones()
    if args.consulta:
        objetivo = list(args.consulta)
    else:
        objetivo = [q for q, a in aclaraciones.items() if a["nivel"] == "vago"]
    cfg, almacen, obs, montaje = _montar(args.config, OVERRIDES)
    try:
        pipeline = ArmadorDePipeline(
            cfg,
            puerto_de_imagen=montaje.puerto_de_imagen,
            indice=montaje.indice,
            puerto_de_llm=montaje.puerto_de_llm,
            embebedor_de_texto=montaje.embebedor_de_texto,
            almacen_de_planes=AlmacenDePlanes(almacen, cfg["rutas.artefactos_planes"]),
            bytes_de_la_imagen=mnt.bytes_del_corpus(cfg, almacen),
            observabilidad=obs,
        ).armar()
        ejecutor = EjecutorDeEtapas(
            cfg,
            pipeline=pipeline,
            almacen_de_corridas=AlmacenDeCorridas(almacen, cfg["rutas.artefactos_corridas"]),
            observabilidad=obs,
            variante_id=montaje.variante_id,
            split_id=montaje.split_id,
        )
        set_ = mnt.set_de_consultas(cfg, almacen, montaje.manifiesto, None)
        for i, consulta in enumerate((q for q in set_.consultas() if q.consulta_id in objetivo), start=1):
            nuevo = aclaraciones[consulta.consulta_id]["sugerencia"]
            datos = mnt.bytes_de_la_imagen_del_set(cfg, almacen, consulta)
            entrada = dataclasses.replace(EntradaDeConsulta.del_set(consulta, datos_de_imagen=datos), pedido=nuevo)
            corrida = ejecutor.correr(entrada)
            print(f"{i}/{len(objetivo)} {consulta.consulta_id:<16} fallida={corrida.fallida is not None} {nuevo!r}")
    finally:
        _cerrar(montaje, obs)


def _cargar_aclaradas() -> dict[str, dict]:
    """El último ordinal de cada consulta aclarada."""
    por_consulta: dict[str, dict] = {}
    for archivo in sorted(CORRIDAS_ACLARADAS.glob("*/corrida.json")):
        corrida = json.loads(archivo.read_text(encoding="utf-8"))
        estado = corrida["estado"]
        if not estado.get("top5"):
            continue
        estado["_ordinal"] = int(corrida.get("ordinal", 1))
        estado["_pedido"] = estado["entrada"]["pedido"]
        previo = por_consulta.get(corrida["consulta_id"])
        if previo is None or estado["_ordinal"] > previo["_ordinal"]:
            por_consulta[corrida["consulta_id"]] = estado
    return por_consulta


def _fila(ev, q, estado) -> dict:
    top5 = estado["top5"]
    conf = confianza_del_top5(estado.get("rerank"), top5)
    return {
        "ndcg": ev.ndcg(q, top5),
        "obras_de_grado_2": sum(1 for a in top5 if ev.gt.get((q, a), 0) >= 2),
        "confianza": conf.puntaje_medio if conf else None,
        "sin_juzgar": ev.sin_juzgar(q, top5),
    }


def evaluar(args) -> None:
    ev = c.Evaluador(c.cargar_juicios([*c.JUICIOS_BASE, *sorted(c.SALIDA.glob("juicios-*.csv")), *map(str, args.juicios_extra)]))
    final = c.cargar_corridas()[FINAL]
    aclaradas = _cargar_aclaradas()
    aclaraciones = _aclaraciones()
    if not aclaradas:
        sys.exit("No hay corridas aclaradas. Correr `aclarar.py correr` primero.")

    filas, pares = [], set()
    for q in sorted(aclaradas):
        antes, despues = _fila(ev, q, final[q]), _fila(ev, q, aclaradas[q])
        pares.update((q, a) for a in despues["sin_juzgar"])
        filas.append(
            {
                "consulta": q,
                "categoria": c.categoria(q),
                "pedido": aclaraciones[q]["pedido"],
                "pedido_aclarado": aclaradas[q]["_pedido"],
                "ndcg_antes": antes["ndcg"],
                "ndcg_despues": despues["ndcg"],
                "grado2_antes": antes["obras_de_grado_2"],
                "grado2_despues": despues["obras_de_grado_2"],
                "confianza_antes": antes["confianza"],
                "confianza_despues": despues["confianza"],
                "sin_juzgar_despues": len(despues["sin_juzgar"]),
            }
        )
    deltas = [f["ndcg_despues"] - f["ndcg_antes"] for f in filas]
    resumen = {
        "consultas_aclaradas": len(filas),
        "ndcg_antes": round(c.media(f["ndcg_antes"] for f in filas), 3),
        "ndcg_despues": round(c.media(f["ndcg_despues"] for f in filas), 3),
        "delta_ndcg": round(c.media(deltas), 3),
        "ic95_del_delta": c.ic95(deltas),
        "mejoran": sum(d > 0.005 for d in deltas),
        "empeoran": sum(d < -0.005 for d in deltas),
        "grado2_por_top5_antes": round(c.media(f["grado2_antes"] for f in filas), 2),
        "grado2_por_top5_despues": round(c.media(f["grado2_despues"] for f in filas), 2),
        "pares_sin_juzgar_despues": len(pares),
        "aviso": "con pares sin juzgar, el 'despues' es una cota inferior (un par sin juicio cuenta como no relevante)",
    }
    destino = c.SALIDA / f"aclaradas-{V}.json"
    destino.write_text(json.dumps({"resumen": resumen, "por_consulta": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    (c.SALIDA / f"pares-sin-juzgar-aclaradas-{V}.json").write_text(json.dumps(sorted([list(p) for p in pares]), ensure_ascii=False), encoding="utf-8")

    print(f"{'consulta':<16}{'nDCG antes':>11}{'despues':>9}{'grado2':>8}{'->':>3}{'':<3}{'confianza':>10}{'->':>4}{'s/j':>5}")
    for f in filas:
        ca = f"{f['confianza_antes']:.1f}" if f["confianza_antes"] is not None else "-"
        cd = f"{f['confianza_despues']:.1f}" if f["confianza_despues"] is not None else "-"
        print(f"{f['consulta']:<16}{f['ndcg_antes']:>11.3f}{f['ndcg_despues']:>9.3f}{f['grado2_antes']:>8}{f['grado2_despues']:>5}{ca:>13}{cd:>6}{f['sin_juzgar_despues']:>5}")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"Escrito en {destino.relative_to(c.RAIZ)} y pares-sin-juzgar-aclaradas-{V}.json ({len(pares)} pares)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("sugerir").set_defaults(fn=sugerir)
    sub.add_parser("validar").set_defaults(fn=validar)
    p = sub.add_parser("correr")
    p.add_argument("--config", type=Path, default=CONFIG_FINAL)
    p.add_argument("--consulta", action="append", help="correr solo estas, aunque el prefiltro no las marque vagas")
    p.set_defaults(fn=correr)
    p = sub.add_parser("evaluar")
    p.add_argument("--juicios-extra", action="append", default=[], help="CSV con consulta_id, artwork_id y grado (repetible)")
    p.set_defaults(fn=evaluar)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
