from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.dominio.catalogo import CatálogoCanónico
from tp3_rag.modulo_a.embebedor_de_imagenes import EmbebedorDeImágenes
from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO, Configuración
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad
from tp3_rag.puertos.almacenamiento import AlmacenamientoEnArchivos
from tp3_rag.puertos.embedding_imagen import EmbeddingImagenCLIP

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-embeber",
        description=(
            "EmbebedorDeImágenes: CLIP ViT-B/32 local sobre el universo con imagen → "
            "CachéDeEmbeddingsDeImagen + artefacto del embebido (el dataset documentado y reproducible, )."
        ),
    )
    p.add_argument("--config", type=Path, help="capa sobre config/default.toml")
    p.add_argument("--set", action="append", default=[], metavar="clave=valor")
    p.add_argument("--catalogo-id", help="cuál catálogo, si hay más de uno en disco")
    p.add_argument("--limite", type=int, help="embeber solo las primeras n (para medir)")
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
        base = (
            Path(cfg["rutas.catalogo"]) / args.catalogo_id
            if args.catalogo_id
            else CatálogoCanónico.único(almacén, cfg["rutas.catalogo"])
        )
        obs.decir(f"catálogo: {base} (re-verificando el `catalogo_id`…)")
        catálogo = CatálogoCanónico.cargar(almacén, base)
        obs.decir(f"             catalogo_id = {catálogo.catalogo_id} ✓\n")

        obs.decir("cargando CLIP…")
        puerto = obs.instrumentar(
            EmbeddingImagenCLIP(
                checkpoint=cfg["embebido.checkpoint"],
                version_del_preprocesado=cfg["embebido.version_del_preprocesado"],
                threads=cfg["embebido.threads"],
            ),
            rol="embedding_imagen",
        )

        resultado = EmbebedorDeImágenes(
            cfg, puerto=puerto, almacén=almacén, observabilidad=obs
        ).correr(catálogo.registros, catalogo_id=catálogo.catalogo_id, limite=args.limite)
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _resumen(resultado, obs)
    return 0

def _resumen(resultado, obs: Observabilidad) -> None:
    a = resultado.artefacto
    e = a["embebido"]
    línea = "─" * 78
    obs.decir(
        f"\n{línea}\nReporteDeEmbebido · {a['catalogo_id']}-{a['ordinal']} (clase 2 — )\n{línea}"
    )
    obs.decir(
        f"\n n embebidas: {e['n_embebidas']:>5} ← la mitad del dataset documentado y reproducible que es del universo\n"
        f" calculadas acá: {e['n_calculadas_en_esta_corrida']:>5}\n"
        f" desde el caché: {e['n_desde_cache']:>5} ← el reuso del caché de imagen entre variantes: no se recalcula\n"
        f" universo con imagen: {e['n_universo_con_imagen']:>5}\n"
        f" fallas: {e['n_fallas']:>5}   "
        f"(el conteo de conversiones de imagen espera 0 de formato: {a['a9c_fallas_de_embedding_por_formato']})\n"
    )
    obs.decir(
        f" throughput: {e['img_por_segundo']:.1f} img/s   "
        f"( midió 22,0 sobre 100 · batch={e['batch']} · threads={e['threads']})\n"
        f" embebido: {e['segundos_de_embebido'] / 60:.1f} min de CPU   "
        f"pared: {e['segundos_de_pared'] / 60:.1f} min"
    )
    obs.decir(f"\n  dim={a['modelo']['dim']} · checkpoint={a['modelo']['checkpoint']}")
    obs.decir(f" artefacto: {resultado.ruta}")
    obs.decir(línea)

if __name__ == "__main__":
    raise SystemExit(main())
