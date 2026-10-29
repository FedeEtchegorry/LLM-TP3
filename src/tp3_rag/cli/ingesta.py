from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tp3_rag.modulo_a.ingesta import Ingesta
from tp3_rag.nucleo.configuracion import Configuración
from tp3_rag.nucleo.credenciales import Credenciales
from tp3_rag.nucleo.errores import ErrorDelProyecto
from tp3_rag.nucleo.observabilidad import Observabilidad

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="tp3-ingesta",
        description=(
            "Ingesta del universo elegible de AIC y CMA → CatálogoCanónico + "
            "AlmacénDeImágenes + ReporteDeIngesta (el dataset documentado y reproducible, el pre y postprocesamiento medido)."
        ),
    )
    p.add_argument("--config", type=Path, help="capa de configuración sobre config/default.toml")
    p.add_argument(
        "--limite-n",
        type=int,
        help="obras por museo. 0 o ausente = universo completo",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="clave=valor",
        help="override de una clave declarada; una clave desconocida es un error",
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
    if args.limite_n is not None:
        overrides["ingesta.limite_n"] = args.limite_n

    obs = Observabilidad(silencioso=args.silencioso)
    try:
        cfg = Configuración.cargar(archivo=args.config, overrides=overrides)
        credenciales = Credenciales.desde_entorno()
        resultado = Ingesta(cfg, credenciales, observabilidad=obs).correr()
    except ErrorDelProyecto as e:
        print(f"\n✗ {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _resumen(resultado, obs, cfg)
    return 0

def _resumen(resultado, obs: Observabilidad, cfg) -> None:
    r = resultado.reporte.a_json()
    línea = "─" * 72
    obs.decir(f"\n{línea}\nReporteDeIngesta · {r['catalogo_id']}-{r['ordinal']}\n{línea}")

    for museo, u in r["universo"]["por_museo"].items():
        pag = u["paginacion"]
        obs.decir(
            f"  {museo}  n={u['n_normalizadas']:>5}  "
            f"declaradas={pag['n_total_declarada']:>5}  "
            f"no_alcanzadas={pag['n_no_alcanzadas']}"
            f"{'  (truncado por límite_n)' if pag['truncado_por_limite'] else ''}"
        )
        obs.decir(
            f"         con imagen real={u['n_con_imagen_real']}  "
            f"sin imagen real={u['n_sin_imagen_real']}  "
            f"{u['bytes_descargados'] / 1e6:.1f} MB"
            + (f"  motivos={u['motivos_sin_imagen']}" if u["motivos_sin_imagen"] else "")
        )

    a9a = r["a9a_cobertura_de_normalizacion"]
    obs.decir(f"\n la cobertura de la normalización cobertura de normalización — denominador: {a9a['denominador']}")
    obs.decir("  " + " " * 16 + "".join(f"{c[:12]:>13}" for c in a9a["campos"]))
    for ámbito, datos in [("total", a9a["total"])] + sorted(a9a["por_museo"].items()):
        obs.decir(
            f"  {ámbito:<5} (n={datos['n_universo_normalizado']:>5})"
            + "".join(f"{datos[c]['pct']:>12.1f}%" for c in a9a["campos"])
        )

    b = r["a9b_culture"]
    obs.decir(
        f"\n valores distintos de culture · criterio={b['criterio']!r}: "
        f"{b['total']['distintos_antes']} distintos antes → "
        f"{b['total']['distintos_despues']} después "
        f"({b['total']['colapsados']} colapsados)"
    )

    c = r["a9c_imagen"]
    obs.decir(
        f" conversiones de imagen · preparadas={c['n_con_imagen_preparada']}  "
        f"grises={c['grises']}  alfa/paleta={c['alfa_o_paleta']}  "
        f"convertidas a RGB={c['convertidas_a_rgb']}  "
        f"fallas de formato={c['fallas_de_conversion_de_formato']}  "
        f"fallas de embedding=(no medible)"
    )

    obs.decir("\n  cobertura por campo de texto (post-render, o sea: tiene texto de verdad)")
    for museo, datos in r["cobertura_por_campo_de_texto"].items():
        partes = " ".join(
            f"{s}={v['pct']:.1f}%" for s, v in datos["secciones"].items()
        )
        obs.decir(f"    {museo} (n={datos['n_universo_normalizado']}): {partes}")

    obs.decir("\n licencias por section (la licencia declarada y no asumida, insumo del Ej. 3)")
    for museo, secciones in r["licencias_por_seccion"].items():
        for section, lics in secciones.items():
            obs.decir(f"    {museo}/{section}: {lics}")

    _top_de_culture(resultado, obs, cfg)

    obs.decir(f"\n catálogo: {resultado.ruta_del_catálogo}")
    obs.decir(f" reporte: {resultado.ruta_del_reporte}")
    obs.decir(f" segundos: {r['segundos']}")
    obs.decir(línea)

def _top_de_culture(resultado, obs: Observabilidad, cfg, n: int = 12) -> None:
    from collections import Counter

    from tp3_rag.modulo_a.normalizador_de_culture import NormalizadorDeCulture

    norm = NormalizadorDeCulture(cfg["normalizacion.culture.criterio"])
    for museo in sorted({r.museum for r in resultado.registros}):
        cuenta: Counter = Counter()
        for registro in resultado.registros:
            if registro.museum == museo:
                cuenta.update(norm.normalizar(registro.culture_raw))
        top = ", ".join(f"{v}={k}" for v, k in cuenta.most_common(n))
        obs.decir(f"\n  culture normalizada, top {n} de {museo}")
        obs.decir(f"    {top}")

if __name__ == "__main__":
    raise SystemExit(main())
