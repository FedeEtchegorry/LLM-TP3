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
from tp3_rag.modulo_b import montaje as mnt
from tp3_rag.modulo_c.acuerdo import UMBRAL, Acuerdo
from tp3_rag.modulo_c.constructor_de_pool import (
    CATEGORÍA_REPETIDA_POR_OMISIÓN,
    CONFIGURACIONES_POR_ESTADO,
    ConstructorDePool,
    ReglaDeSelección,
    ReglaDeTodasLasConsultas,
)
from tp3_rag.modulo_c.planilla import (
    ALCANCE_COMPLETA,
    ALCANCES,
    PlanillaDeAnotación,
)
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
            "El camino de anotación (el módulo de evaluación): Rúbrica, Pool, PlanillaDeAnotación y κ. "
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
    rp.add_argument("--designacion", default="piloto", help="el `rubrica_id` (clase 3)")
    rp.add_argument("--refina", help="si refina otra rúbrica (el acuerdo entre anotadores con κ < 0.6)")
    rp.add_argument("--salida", type=Path, help="dónde escribirla (por omisión, stdout)")
    rp.add_argument(
        "--escribir", action="store_true",
        help="escribirla en artifacts/ground-truth/rubricas/<designacion>.md",
    )

    r = sub.add_parser("rubrica", help="leer y validar las rúbricas congeladas")
    r.add_argument("--rubrica-id", help="cuál, si hay más de una")

    da = sub.add_parser(
        "decision-de-a5",
        help=(
            "🔴 el ESQUELETO de la decisión del acuerdo entre anotadores para una rúbrica con κ < umbral. "
            "No la decide: trae la medición adentro y el grupo elige una de las dos salidas"
        ),
    )
    da.add_argument("--rubrica", required=True, help="el `rubrica_id` cuyo κ está bajo")
    da.add_argument("--salida", type=Path, help="dónde escribirla (por omisión, stdout)")
    da.add_argument(
        "--escribir", action="store_true",
        help="escribirla al lado de la rúbrica, en artifacts/ground-truth/rubricas/",
    )

    pd = sub.add_parser(
        "pool-derivar",
        help=(
            "derivar un pool: el PROVISORIO del piloto (la evaluación § La ronda piloto) o el "
            "CONGELADO con --estado congelado (la evaluación § Pooling)"
        ),
    )
    pd.add_argument("--designacion", default="piloto", help="la mitad declarada del `pool_id`")
    pd.add_argument(
        "--estado", default="piloto", choices=list(ESTADOS),
        help=(
            "🔴 el protocolo, no una etiqueta: `piloto` es C1 sobre 4 consultas y "
            "`congelado` es la unión de los top-5 de C0–C4 sobre todas (la evaluación)"
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

    co = sub.add_parser(
        "controles",
        help=(
            "los dos controles del ground truth, sueltos —todo par juzgado estuvo en "
            "algún pool de ese split, y ninguna obra del held-out entra por la puerta "
            "de la anotación—: los mismos que corren al exportar"
        ),
    )
    co.add_argument("--designacion", default="piloto")

    ex = sub.add_parser(
        "exportar", help="la planilla de UNA persona (corre los dos controles antes)"
    )
    ex.add_argument("--designacion", default="piloto", help="de qué pool")
    ex.add_argument("--rubrica", required=True, help="el `rubrica_id` con el que se va a juzgar")
    ex.add_argument("--anotador", required=True, help="la etiqueta de quién anota")
    ex.add_argument("--salida", type=Path, help="dónde (por omisión, planillas/…csv)")
    ex.add_argument(
        "--alcance", default=ALCANCE_COMPLETA, choices=list(ALCANCES),
        help=(
            "🔴 qué filas le tocan. `completa` es el pool entero; `desempate` es el "
            "TERCER anotador del tercer anotador, — solo los pares sin consenso, que los declara "
            "el consenso y no esta bandera. El alcance **viaja en la planilla**"
        ),
    )

    im = sub.add_parser("importar", help="🔴 la política de trazado · validar que lo importado sea lo exportado")
    im.add_argument("--planilla", type=Path, required=True)

    ka = sub.add_parser(
        "kappa", help="κ de Cohen — se reporta pase lo que pase (el acuerdo entre anotadores). Escribe κ y consenso"
    )
    ka.add_argument(
        "--designacion",
        help=(
            "🔴 de qué pool. Sin default: con un solo pool del split se deriva y "
            "se dice cuál; con más de uno hay que elegir, porque el `pool_id` va en el "
            "NOMBRE del artefacto que el acuerdo entre anotadores publica"
        ),
    )
    ka.add_argument("--rubrica", required=True)
    ka.add_argument(
        "--anotador", action="append", default=[],
        help="la pareja del κ (κ de Cohen es entre DOS). Por omisión, los dos que haya",
    )

    jz = sub.add_parser(
        "juez",
        help=(
            "🔴 💲 GASTA · correr el LLM-judge sobre el pool: un grado 0/1/2 por "
            "par, con la MISMA rúbrica que los anotadores. Una llamada por ítem (379 en el "
            "pool congelado) y deja el `AporteDelJuez` (clase 2, con ordinal)"
        ),
    )
    jz.add_argument("--designacion", help="de qué pool. Sin default: se declara o se deriva")
    jz.add_argument("--rubrica", required=True, help="el `rubrica_id` que el juez aplica")
    jz.add_argument(
        "--limite",
        type=int,
        help=(
            "🔴 correr solo los primeros `n` ítems — **el piloto antes de gastar los 379**. "
            "El alcance parcial **queda declarado en el aporte**: un aporte parcial que no "
            "dijera que es parcial sería una población que no existe"
        ),
    )
    jz.add_argument(
        "--confirmar-gasto",
        action="store_true",
        help=(
            "🔴 **obligatorio para correr sin `--limite`**. Sin esto, el comando dice "
            "cuántas llamadas haría y sale con 0 **sin llamar a nada**: *el nombre de un "
            "flag no te dice si gasta*"
        ),
    )

    jk = sub.add_parser(
        "juez-kappa",
        help=(
            "🔴 mitad 2 · los **TRES** κ del juez: contra el consenso "
            "(estratificado por configuración de origen), contra `ia-codex` y contra "
            "`ia-claude`. **Es una lectura: no gasta**"
        ),
    )
    jk.add_argument("--designacion", help="de qué pool. Sin default: se declara o se deriva")
    jk.add_argument("--rubrica", required=True, help="el `rubrica_id` del consenso")
    jk.add_argument(
        "--aporte",
        help="cuál `AporteDelJuez` (su identidad). Por omisión, el de **ordinal más alto**",
    )
    jk.add_argument("--json", type=Path, help="dónde dejar los tres κ como JSON")
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
        if args.comando == "decision-de-a5":
            return _decision_de_a5(args, almacén, obs, rutas)
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
        if args.comando == "juez":
            return _juez(args, cfg, almacén, obs, rutas)
        if args.comando == "juez-kappa":
            return _juez_kappa(args, cfg, almacén, obs, rutas)
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
            "(el manifiesto de split): corré `poetry run tp3-split` primero."
        )
    if len(existentes) > 1:
        raise ErrorDeArtefacto(
            f"hay {len(existentes)} splits congelados: {list(existentes)}. Pasá "
            f"`--split-id`: *los juicios de un split no son los de otro* "
            f"(la frontera entre el juicio y el split), y elegir por `mtime` sería tomar esa decisión por "
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
            f"clase 3 (la regla de que un id determina su contenido) y lo que hay adentro puede ser el texto que el grupo "
            f"escribió. Si querés otra, usá otra `--designacion` — refinar es **derivar "
            f"otra** (el acuerdo entre anotadores)."
        )
    ruta = almacén.escribir_texto(destino, texto)
    obs.decir(
        f"\n{LÍNEA}\nRúbrica `{args.designacion}` — **PLANTILLA** (estado = plantilla)\n{LÍNEA}\n"
        f"\n  {ruta}\n"
        f"\n  🔴 **El texto lo escribe el grupo.** Lo que hay ahí son las cinco secciones "
        f"que\n"
        f"  la tabla de identidades dice que una rúbrica tiene —definiciones de {list(GRADOS)}, ejemplos y\n"
        f"  desempate— con cada hueco marcado `{drubrica.MARCA}`, más lo único que ya está\n"
        f"  decidido: los tres nombres de la escala de la evaluación.\n"
        f"\n  Mientras quede un `{drubrica.MARCA}`, `estado` sigue en `plantilla` y "
        f"**exportar\n  una planilla falla**: gastar 20 juicios bajo un instrumento a medio "
        f"escribir es\n  justo lo que el piloto existe para no hacer con los ~540.\n"
        f"\n  Al terminar: poné `estado = \"completa\"` y validá con\n"
        f"      poetry run tp3-anotar rubrica\n"
        f"\n  ✅ Va al repo (la regla de qué se versiona), **diffeable a propósito**: el diff entre dos rúbricas "
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
        _mostrar_kappa_de_la_rubrica(r, almacén, obs, rutas)
    return 0

def _mostrar_kappa_de_la_rubrica(r: Rúbrica, almacén, obs, rutas) -> None:
    from tp3_rag.modulo_c import acuerdo as ac

    kappas = ac.kappas_de_la_rubrica(almacén, rutas.acuerdo, r.rubrica_id)
    if not kappas:
        obs.decir(
            f"     κ (el acuerdo entre anotadores): **no medido** — ningún `kappa-…json` cita esta rúbrica.\n"
            f"                 No bloquea exportar (no medido ≠ *bajo*), y "
            f"tampoco dice nada."
        )
        return
    for k in kappas:
        if k.es_ausente:
            obs.decir(
                f"     κ (el acuerdo entre anotadores): **ausente** sobre {k.pool_id} ({k.n} pares) — "
                f"no es 0 ni 1, **no está definido**.\n"
                f"                 {(k.razon_de_ausencia or '').splitlines()[0]}"
            )
        elif k.esta_bajo_el_umbral:
            obs.decir(
                f"     κ (el acuerdo entre anotadores): 🔴 **{k.kappa:.3f} < {k.umbral}** sobre {k.pool_id} "
                f"({k.n} pares) — la rúbrica es **ambigua por medición**."
            )
        else:
            obs.decir(
                f"     κ (el acuerdo entre anotadores): ✓ {k.kappa:.3f} ≥ {k.umbral} sobre {k.pool_id} "
                f"({k.n} pares)."
            )
    if not any(k.esta_bajo_el_umbral for k in kappas):
        return
    decisión = drubrica.DecisiónDeA5.leer_si_existe(almacén, rutas.rubricas, r.rubrica_id)
    if decisión is None:
        obs.decir(
            f"                 🔴 **sin decisión del acuerdo entre anotadores declarada → `exportar` falla** "
            f"(el chequeo).\n"
            f"                 Las dos salidas del acuerdo entre anotadores: refinar (otra `rubrica_id`) o "
            f"declarar la limitación:\n"
            f"                     poetry run tp3-anotar decision-de-a5 --rubrica "
            f"{r.rubrica_id} --escribir"
        )
        return
    problemas = decisión.problemas_contra(kappas=kappas, hash_del_texto=r.hash_del_texto)
    obs.decir(
        f"                 decisión del acuerdo entre anotadores: `salida = {decisión.salida}` · "
        f"{decisión.decidida_por} · {decisión.fecha}"
    )
    if problemas:
        obs.decir("                 🔴 **y no cubre esta medición → `exportar` falla**:")
        for p in problemas:
            obs.decir(f"                     · {p}")
    else:
        obs.decir(
            f"                 ✓ cubre la medición: `exportar` pasa y **la limitación "
            f"viaja en el `-como-anotar.md`**."
        )

def _decision_de_a5(args, almacén, obs, rutas) -> int:
    from tp3_rag.modulo_c import acuerdo as ac

    r = Rúbrica.leer(almacén, rutas.rubricas, args.rubrica)
    kappas = ac.kappas_de_la_rubrica(almacén, rutas.acuerdo, r.rubrica_id)
    bajos = [k for k in kappas if k.esta_bajo_el_umbral]
    if not bajos:
        raise ErrorDeArtefacto(
            f"la rúbrica {r.rubrica_id!r} **no tiene un κ medido bajo el umbral**, así que "
            f"no hay nada que decidir: el acuerdo entre anotadores pide elegir una de las dos salidas *si* "
            f"κ < umbral.\n"
            f"  Lo que hay: "
            + (
                "ninguna medición (`tp3-anotar kappa --rubrica "
                f"{r.rubrica_id}` todavía no corrió)."
                if not kappas
                else ", ".join(
                    f"{k.pool_id} → "
                    + ("ausente (no definido)" if k.es_ausente else f"κ = {k.kappa:.3f}")
                    for k in kappas
                )
            )
            + "\n  ⚠️ Declarar una decisión del acuerdo entre anotadores sin un κ bajo que la justifique sería "
            "dejar el chequeo desactivado de antemano."
        )
    peor = min(bajos, key=lambda k: k.kappa)
    texto = drubrica.plantilla_de_decision_de_a5(
        rubrica_id=r.rubrica_id,
        kappa=peor.kappa,
        pool_id=peor.pool_id,
        hash_del_texto=r.hash_del_texto,
        umbral=peor.umbral,
    )
    destino = args.salida
    if args.escribir and destino is None:
        destino = drubrica.DecisiónDeA5.ruta_de(rutas.rubricas, r.rubrica_id)
    if destino is None:
        print(texto)
        return 0
    if almacén.existe(destino):
        raise ErrorDeArtefacto(
            f"ya existe {almacén.ruta(destino)} y **no se sobreescribe**: es clase 3 y lo "
            f"que hay adentro puede ser la decisión que el grupo tomó. Editala a mano, o "
            f"mirá qué dice con `poetry run tp3-anotar rubrica --rubrica-id {r.rubrica_id}`."
        )
    ruta = almacén.escribir_texto(destino, texto)
    obs.decir(
        f"\n{LÍNEA}\nDecisión del acuerdo entre anotadores · rúbrica `{r.rubrica_id}` — **PLANTILLA**\n{LÍNEA}\n"
        f"\n  {ruta}\n"
        f"\n  La medición ya está adentro y **no se transcribe a mano**:\n"
        f"      κ = {peor.kappa:.3f} < {peor.umbral}  ·  pool {peor.pool_id}  ·  hash del "
        f"texto {r.hash_del_texto}\n"
        f"\n  🔴 **Lo que falta es del grupo**: cuál de las dos salidas del acuerdo entre anotadores, si es la "
        f"limitación,\n  **el párrafo que va al informe**. Mientras quede un "
        f"`{drubrica.MARCA}`, la decisión no valida\n  y `exportar` sigue fallando.\n"
        f"\n  Las dos salidas, y no son simétricas:\n"
        f"    · `refinada`             → derivar otra `rubrica_id` con `refina = "
        f"\"{r.rubrica_id}\"`. **No**\n"
        f"                                autoriza a exportar con esta: se ejerce con "
        f"`--rubrica <la nueva>`.\n"
        f"    · `limitacion_declarada` → autoriza, y la limitación **viaja adentro del "
        f"`-como-anotar.md`**\n"
        f"                                de cada planilla: la decisión queda en el "
        f"artefacto y no en la\n                                cabeza de quien corre el "
        f"comando.\n"
        f"\n  ✅ Va al repo (la regla de qué se versiona), al lado de la rúbrica. Agregalo por ruta explícita — "
        f"**no `git add -A`**.\n{LÍNEA}"
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
            f"**el top-5 de C1** (la evaluación), y eso se lee de las corridas: "
            f"`artifacts/corridas/` está gitignoreado y es **regenerable** con "
            f"`poetry run tp3-responder"
            f"{mnt.flag_de_variante(cfg, almacén, None)} set`."
        )
    obs.decir(
        f"split      : {manifiesto.split_id}  ·  "
        f"{manifiesto.conteos['corpus_indexado']} corpus + "
        f"{manifiesto.conteos['held_out']} held-out\n"
        f"set        : {set_de_consultas.identidad}  ·  estado "
        f"{set_de_consultas.estado}\n"
        f"corridas   : {len(corridas)} en disco"
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
            f"  ✅ Los dos archivos **van al repo** (la regla de qué se versiona): es el material más caro del\n"
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
        f"\n  split_id   : {pool.split_id}\n"
        f"  designación: {pool.designacion}    ← la mitad declarada de la identidad (la tabla de identidades)\n"
        + (
            f"  🔴 estado  : {pool.estado}    ← la campaña definitiva: una vez anotado **no "
            f"se toca**\n"
            if pool.es_congelado
            else f"  🔴 estado  : {pool.estado}    ← **NO es el pool congelado**: "
            f"los juicios que salgan de acá no son la campaña definitiva\n"
        )
        + "  configs    :\n"
        + "\n".join(
            f"    {c['configuracion']}  config_id {c['config_id']}  variante "
            f"{c['variante_id']}  α={c.get('alfa')}  "
            + ("α congelado" if c.get("alfa_congelado") else "⚠️ α **provisorio**")
            + f"  ·  {len(c.get('corridas') or ())} corridas"
            for c in pool.configuraciones
        )
        + f"\n  semilla    : {pool.semilla_de_mezcla}  ({regla['de_donde_sale_la_semilla']})"
    )
    obs.decir(
        f"\n  La regla (declarada adentro, {regla['fuente']}):\n"
        f"    «{regla['texto']}»"
    )
    if "orden_de_turnos" in regla:
        obs.decir(
            f"    turnos         : {' → '.join(regla['orden_de_turnos'])}\n"
            f"    repetida       : {regla['categoria_repetida']}\n"
            f"    ranking        : {regla['ranking_dentro_de_la_categoria'].splitlines()[0]}\n"
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
            f"    selección      : {regla['seleccion_de_consultas']} "
            f"({regla['n_consultas']} consultas, {regla['orden']})\n"
            f"    mezcla         : {regla['mezcla_dentro_de_la_consulta'].splitlines()[0]}\n"
            f"    excluye        : {regla['excluye']}"
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
            f"\n La cuenta de la unión (la evaluación § Pooling):\n"
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
            f"       innegociable* de la evaluación y tiene que dar 0 acá por construcción."
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
        f"\n  pool (ciego) : {ruta}\n"
        f"  procedencia  : {ruta_json}\n{LÍNEA}"
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
        f"  El momento es la mitad del arreglo (el módulo de evaluación, el criterio de inclusión del corpus/la política de trazado): **este es el último punto\n"
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
        alcance=args.alcance,
    )
    _mostrar_controles(resultado.controles, obs)
    v = resultado.veredicto_de_a5
    if v is not None and v.hay_limitacion_declarada:
        peor = min((k for k in v.kappas if k.esta_bajo_el_umbral), key=lambda k: k.kappa)
        obs.decir(
            f"  🔴 esta rúbrica tiene κ = {peor.kappa:.3f} < {peor.umbral} y se "
            f"exporta porque\n"
            f"      hay una **decisión declarada** (`salida = {v.decision.salida}`, "
            f"{v.decision.decidida_por}, {v.decision.fecha}).\n"
            f"      El `-como-anotar.md` lleva **un puntero** —que existe y dónde "
            f"está—, no una copia\n"
            f"      (la limitación es correcta en el artefacto): la limitación es correcta **en el artefacto** y "
            f"contaminante en el briefing.\n"
            f"      📄 {drubrica.DecisiónDeA5.ruta_de(rutas.rubricas, resultado.rubrica_id)}\n"
        )
    obs.decir(
        f"{LÍNEA}\nPlanilla de `{resultado.anotador}` · {resultado.n_filas} juicios\n{LÍNEA}\n"
        f"\n  planilla     : {resultado.ruta}\n"
        f"  cómo anotar  : {resultado.ruta_de_instrucciones}\n"
        f"  rúbrica      : {Rúbrica.ruta_de(rutas.rubricas, resultado.rubrica_id)} "
        f"(hash {rubrica.hash_del_texto})\n"
        f"  pool         : {resultado.pool_id}  (estado {pool.estado})\n"
        + (
            f"  🔴 alcance   : **{resultado.alcance.tipo}** — {resultado.alcance.n} de "
            f"{resultado.alcance.n_del_pool} filas a llenar\n"
            f"                 {resultado.alcance.declarado_por}\n"
            if resultado.alcance is not None and not resultado.alcance.es_completa
            else f"  alcance      : {resultado.n_filas} filas, el pool entero\n"
        )
        + f"\n  Las dos columnas que se llenan: **grado** y **nota**.\n"
        + "\n".join(
            f"      {g} = {SIGNIFICADO_DE_LA_ESCALA[g]}" for g in sorted(GRADOS, reverse=True)
        )
        + f"\n\n  🔴 No agregar ni borrar filas, no tocar los ids, **no reordenar**: los tres\n"
        f"     son error de import (la política de trazado). Si ya reordenaste, ordená por `fila`.\n"
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
    alcance = resultado.alcance
    parcial = alcance is not None and not alcance.es_completa
    obs.decir(
        f"\n{LÍNEA}\nAporteDeAnotador · {aporte.identidad}  (clase 3 — {estado})\n{LÍNEA}\n"
        + (
            f"\n  ✓ **{resultado.n_juicios}/{alcance.n}** juicios del alcance "
            f"**{alcance.tipo}**  ·  la política de trazado verificado: lo importado **es** lo exportado\n"
            f"\n  🔴 alcance : **{alcance.tipo}** — {alcance.n} de {alcance.n_del_pool} "
            f"ítems del pool, y las otras\n"
            f"               {alcance.n_del_pool - alcance.n} están vacías **a propósito** "
            f"(La política de trazado compromete lo\n"
            f"               importado es lo exportado*, **no** *todas las filas tienen "
            f"grado*)\n"
            f"               declarado por: {alcance.declarado_por}\n"
            f"               pareja del κ : {list(alcance.pareja_del_kappa)} — este aporte "
            f"**no entra al κ**,\n"
            f"               entra al **consenso** (la evaluación, el desempate del tercero)\n"
            if parcial
            else f"\n  ✓ **{resultado.n_juicios}/{resultado.n_juicios}** juicios  ·  "
            f"la política de trazado verificado: lo importado **es** lo exportado\n"
        )
        + f"\n  split_id   : {aporte.split_id}\n"
        f"  rubrica_id: {aporte.rubrica_id}    ← 🔑 sin esto, el piloto y la campaña\n"
        f"                            definitiva comparten nombre de archivo\n"
        f"  anotador   : {aporte.anotador}\n"
        f"  pool_id: {aporte.pool_id}    ← procedencia, **no** ingrediente (la tabla de identidades)\n"
        f"\n  distribución: "
        + " · ".join(f"{k} = {v}" for k, v in sorted(resultado.distribucion.items()))
        + f"\n\n  juicios : {resultado.ruta}\n"
        f"  ✅ Va al repo (la regla de qué se versiona): es **irreproducible**. Agregalo por ruta explícita.\n"
        + (
            f"\n  Y ahora el consenso se **recalcula** con tu desempate:\n"
            if parcial
            else f"\n  Cuando haya **dos** aportes de la misma rúbrica:\n"
        )
        + f"      poetry run tp3-anotar kappa --designacion "
        f"{identidad.partir_identidad_del_pool(aporte.pool_id)[1]} "
        f"--rubrica {aporte.rubrica_id}"
        + (
            " \\\n          "
            + " ".join(f"--anotador {n}" for n in alcance.pareja_del_kappa)
            + "\n      (la pareja va explícita: con tu aporte hay **tres** y κ de Cohen es "
            "entre dos;\n       la que se pasa es la que el κ ya declaró, no una elección "
            "nueva)"
            if parcial and alcance.pareja_del_kappa
            else ""
        )
        + f"\n{LÍNEA}"
    )
    return 0

def _kappa(args, cfg, almacén, obs, rutas) -> int:
    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    obs.decir(
        f"pool       : {pool.pool_id}  ·  "
        + (
            "**declarado** con `--designacion`"
            if args.designacion
            else "**derivado**: es el único pool del split, y se dice cuál se eligió "
            ""
        )
        + f"\n             ⚠️ el `pool_id` va en el **nombre** de los dos artefactos que "
        f"esto escribe,\n así que la población **no** tiene default"
    )
    rubrica = Rúbrica.leer(almacén, rutas.rubricas, args.rubrica)
    pareja = tuple(args.anotador) if args.anotador else None
    if pareja is not None and len(pareja) != 2:
        raise ErrorDeArtefacto(
            f"se pasaron {len(pareja)} `--anotador` y **κ de Cohen es entre dos**. El "
            f"tercero no promedia: sus juicios son el **desempate** y entran al consenso "
            f"(la evaluación, el desempate del tercero)."
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
        f"\n  anotadores : {k.anotadores[0]} vs {k.anotadores[1]}\n"
        f"  población  : {k.poblacion}\n"
        f"\n  acuerdo observado      po = {k.acuerdo_observado:.3f}\n"
        f"  acuerdo esperado        pe = {k.acuerdo_esperado:.3f}\n"
        f"  **κ = "
        + ("ausente" if k.kappa is None else f"{k.kappa:.3f}")
        + f"**   (umbral del acuerdo entre anotadores: {UMBRAL})"
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
            f"     la evaluación (el desempate del tercero): el desempate del tercer integrante es **una anotación "
            f"más**, no una\n"
            f"     corrección del consenso. Importá su planilla y el consenso se recalcula:\n"
            f"         poetry run tp3-anotar exportar --designacion {pool.designacion} "
            f"--rubrica {resultado.rubrica_id} \\\n"
            f"             --anotador <el tercero> --alcance desempate"
        )
    obs.decir(f"\n  {k.veredicto.replace('{rubrica}', resultado.rubrica_id)}")
    obs.decir(
        f"\n  consenso : {resultado.ruta_del_consenso}"
        + ("   ·  **ya estaba** (mismo contenido: no se reescribió)"
           if resultado.consenso_ya_estaba else "   ·  escrito")
        + f"\n  κ        : {resultado.ruta_del_kappa}"
        + ("   ·  **ya estaba** (mismo contenido: no se reescribió)"
           if resultado.kappa_ya_estaba else "   ·  escrito")
        + (
            "\n  🔑 Lo que se compara es **el contenido**, no si el comando corrió. "
            "Si\n     entra un desempate, el κ se recomputa y los dos se reescriben — que es "
            "el desempate del tercero."
            if resultado.ya_estaba
            else ""
        )
        + f"\n  ✅ Los dos van al repo (la regla de qué se versiona): son la lectura que el acuerdo entre anotadores publica.\n"
        f"\n  ⚠️ Este κ es el de **{pool.designacion}** sobre {k.n} pares "
        f"({len(pool.items)} ítems en el pool,\n estado {pool.estado}). la evaluación pide "
        f"no confundirlo con los κ de **otras rondas**:\n     **no se suman ni se "
        f"promedian**, y cada uno se cita con su población al lado.\n{LÍNEA}"
    )
    return 0

def _juez(args, cfg, almacén, obs, rutas) -> int:
    from tp3_rag.modulo_b import montaje as mnt
    from tp3_rag.modulo_c.juez import BindingDelJuez, Juez
    from tp3_rag.nucleo.credenciales import Credenciales

    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    set_ = _set_de_consultas(args, almacén, rutas, manifiesto)
    rubrica = Rúbrica.leer(almacén, rutas.rubricas, args.rubrica)
    binding_pelado = BindingDelJuez.desde_configuración(cfg)
    n = args.limite or len(pool.items)

    obs.decir(
        f"{LÍNEA}\n🔴 **El juez** · `la evaluación § Validación del LLM-judge`\n{LÍNEA}\n"
        f"  pool        : {pool.pool_id} ({pool.estado}) · {len(pool.items)} ítems\n"
        f"  rúbrica     : {rubrica.rubrica_id} (estado {rubrica.estado}, hash "
        f"{rubrica.hash_del_texto})\n"
        f"  set         : {set_.identidad}\n"
        f"  modelo: **{binding_pelado.modelo}**  ← que el juez no sea el reranker / mitigación 3: el reranker es "
        f"`{cfg['bindings.reranker.modelo']}`\n"
        f"  binding_juez: {binding_pelado.binding_juez}  (modelo + prompt + muestreo, la tabla de identidades)\n"
        f"  alcance     : **{n} ítem(s)**"
        + ("" if args.limite is None else " — ⚠️ **PARCIAL**, y queda declarado en el aporte")
    )
    obs.decir(
        f"\n  💲 **Esto va a hacer {n} llamada(s) al proxy**, cada una con **dos imágenes**\n"
        f"     (la original de la consulta + la de la obra) y **la rúbrica entera** adentro\n"
        f"     del mensaje de sistema ({len(rubrica.texto)} caracteres). A ~USD 0,0010 por\n"
        f"     request son **~USD {n * 0.001:.2f}**. ⚠️ El precio es **declarado y no\n"
        f"     verificable** (`/key/info` está bloqueado para virtual keys): lo **medido**\n"
        f"     son los requests y los tokens, y la verdad de facturación es el dashboard\n"
        f"     del proxy. **Las dos cifras no se promedian.**"
    )
    if args.limite is None and not args.confirmar_gasto:
        obs.decir(
            f"\n  🔴 **No se llamó a nada.** Para correr el pool completo hace falta\n"
            f"     `--confirmar-gasto`; para un piloto, `--limite 5`. El grupo pidió "
            f"**gastar lo\n     menos posible aunque no haya tope**, así que el default de "
            f"este comando es\n     **no gastar**.\n{LÍNEA}"
        )
        return 0

    credenciales = Credenciales.desde_entorno(RAÍZ_DEL_REPO)
    from tp3_rag.puertos.llm import LLMOpenAI

    puerto = LLMOpenAI(
        api_key=credenciales.exigir_llm(),
        base_url=cfg["proveedor.base_url"],
        timeout_segundos=float(cfg["http.timeout_segundos"]),
        backoff_base=float(cfg["http.backoff_base"]),
    )
    binding = BindingDelJuez.desde_configuración(cfg, puerto=puerto, observabilidad=obs)
    juez = Juez(
        cfg,
        almacén=almacén,
        rutas=rutas,
        binding=binding,
        bytes_de_la_imagen_de_consulta=lambda c: mnt.bytes_de_la_imagen_del_set(
            cfg, almacén, c
        ),
        bytes_de_la_imagen_de_la_obra=mnt.bytes_del_corpus(cfg, almacén),
        observabilidad=obs,
    )

    def al_juzgar(i, total, juicio):
        if i % 25 == 0 or i == total or not juicio.juzgado:
            obs.decir(
                f"     {i:>4}/{total}  {juicio.consulta_id:<18} {juicio.artwork_id:<14} "
                + ("grado " + str(juicio.grado) if juicio.juzgado else "**SIN GRADO**")
            )

    obs.decir("\n  corriendo…\n")
    resultado = juez.juzgar(
        pool=pool, rubrica=rubrica, set_de_consultas=set_, limite=args.limite,
        al_juzgar=al_juzgar,
    )

    aporte = resultado.aporte
    obs.decir(
        f"\n{LÍNEA}\n**`AporteDelJuez`** · clase 2 con ordinal (la tabla de identidades)\n{LÍNEA}\n"
        f"  identidad : {aporte.identidad}\n"
        f"  archivo   : {resultado.ruta}\n"
        f"  juicios   : {len(aporte.juicios)}  ·  por grado {aporte.por_grado()}  ·  "
        f"**{aporte.n_sin_grado} sin grado**\n"
        f"  medido: {resultado.n_llamadas} llamada(s) (cuenta **intentos**) · "
        f"{resultado.tokens_in} tokens in · {resultado.tokens_out} out · "
        f"{resultado.segundos:.0f} s\n"
        f"\n  🔴 **Dos corridas del juez son DOS aportes** (ordinal {aporte.ordinal}): el juez "
        f"es *el* caso de\n     salida no determinista con entrada fija, así que el κ puede "
        f"decir **de cuál** salió.\n"
        f"  🔑 Y está en `juicios-juez/`, **no** en `juicios/`: otro tipo y otra carpeta, que "
        f"es\n la mitigación 1 de la evaluación hecha estructura — un aporte del juez en "
        f"`juicios/`\n     habría entrado al **desempate del consenso** sin que nada lo "
        f"acusara.\n"
        f"\n  Ahora: `poetry run tp3-anotar juez-kappa --designacion {pool.designacion} "
        f"--rubrica {rubrica.rubrica_id}`\n{LÍNEA}"
    )
    return 0

def _juez_kappa(args, cfg, almacén, obs, rutas) -> int:
    import json as _json

    from tp3_rag.modulo_c import metricas as met
    from tp3_rag.modulo_c.juez import AporteDelJuez, kappas_del_juez

    manifiesto = _manifiesto(args, cfg, almacén)
    pool = _pool(args, almacén, rutas, manifiesto)
    set_ = _set_de_consultas(args, almacén, rutas, manifiesto)
    aportes = AporteDelJuez.todos(almacén, rutas.juicios_juez, split_id=pool.split_id)
    if not aportes:
        raise ErrorDeArtefacto(
            f"no hay ningún `AporteDelJuez` del split {pool.split_id} en "
            f"{rutas.juicios_juez}. Corré `tp3-anotar juez --rubrica {args.rubrica} "
            f"--confirmar-gasto` primero — **y eso gasta**: una llamada por ítem del pool."
        )
    if args.aporte:
        elegido = next((a for a in aportes if a.identidad == args.aporte), None)
        if elegido is None:
            raise ErrorDeArtefacto(
                f"no hay un `AporteDelJuez` con identidad {args.aporte!r}. Los que hay: "
                f"{[a.identidad for a in aportes]}."
            )
    else:
        elegido = max(aportes, key=lambda a: a.ordinal)
    del_mismo = [a for a in aportes if a.binding_juez == elegido.binding_juez]

    gt_completo = met.leer_consenso(
        almacén, rutas.acuerdo, pool_id=pool.pool_id, rubrica_id=args.rubrica
    )
    humanos = {
        a.anotador: {j.par: j.grado for j in a.juicios}
        for a in AporteDeAnotador.todos(
            almacén, rutas.juicios, split_id=pool.split_id, rubrica_id=args.rubrica
        )
    }
    origen_por_par: dict[tuple[str, str], tuple[str, ...]] = {}
    for fila in pool.procedencia.get("origen") or ():
        par = (fila["consulta_id"], fila["artwork_id"])
        origen_por_par[par] = tuple(
            sorted(set(origen_por_par.get(par, ())) | {fila["configuracion"]})
        )

    obs.decir(
        f"{LÍNEA}\n🔴 **el acuerdo entre anotadores mitad 2** · los TRES κ del juez\n{LÍNEA}\n"
        f"  aporte      : {elegido.identidad}  (ordinal {elegido.ordinal} de "
        f"{len(del_mismo)} con este `binding_juez`)\n"
        f"  modelo      : **{elegido.modelo}**  ·  rúbrica {elegido.rubrica_id} "
        f"(hash {elegido.hash_de_la_rubrica})\n"
        f"  juicios     : {len(elegido.juicios)} · por grado {elegido.por_grado()} · "
        f"{elegido.n_sin_grado} sin grado\n"
        f"  consenso    : {gt_completo.n} pares con grado "
        f"({dict(sorted(gt_completo.por_resolucion.items()))})\n"
        f"  anotadores  : {sorted(humanos)}"
    )
    if elegido.n_sin_grado:
        causas: dict[str, int] = {}
        for j in elegido.juicios:
            if j.juzgado:
                continue
            marca = "presupuesto agotado (`budget_exceeded`)" if "budget" in j.nota.lower() else (
                "imagen faltante" if "imagen" in j.nota.lower() else "otra"
            )
            causas[marca] = causas.get(marca, 0) + 1
        cubiertas = sorted({j.consulta_id for j in elegido.juicios if j.juzgado})
        obs.decir(
            f"\n  🔴 **{elegido.n_sin_grado} de {len(elegido.juicios)} pares quedaron SIN "
            f"GRADO**, así que el κ de abajo es sobre\n     **{len(elegido.pares)} pares y "
            f"{len(cubiertas)} de 30 consultas**, no sobre el pool entero. Causas: "
            f"{causas}.\n"
            f"     ⚠️ **Y el recorte NO es aleatorio**: el juez recorre el pool en el orden "
            f"de sus ítems,\n     así que lo que quedó es un **prefijo por consulta** "
            f"({cubiertas[0]} … {cubiertas[-1]}).\n"
            f"     Las tres categorías están representadas y eso se puede verificar; "
            f"**la mitad de las\n     10 imágenes no está**. El número va con esta "
            f"población al lado o no es citable."
        )
    if elegido.hash_de_la_rubrica and elegido.hash_de_la_rubrica != (
        Rúbrica.leer(almacén, rutas.rubricas, args.rubrica).hash_del_texto
    ):
        raise ErrorDeArtefacto(
            f"el aporte del juez citó la rúbrica {args.rubrica!r} con hash "
            f"{elegido.hash_de_la_rubrica} y el texto en disco tiene otro. **El texto de la "
            f"rúbrica se movió entre el juicio y la medición**, así que el κ saldría igual y "
            f"no significaría nada: κ es siempre sobre una rúbrica."
        )

    tres = kappas_del_juez(
        elegido,
        consenso=gt_completo.grados,
        aportes_humanos=humanos,
        origen_por_par=origen_por_par,
        rubrica_id=args.rubrica,
        categoria_de={c.consulta_id: c.categoria for c in set_.consultas()},
    )

    def _linea(nombre, k):
        return (
            f"  {nombre:<28} n = {k.n:>3}   po = {k.acuerdo_observado:.3f}   "
            f"pe = {k.acuerdo_esperado:.3f}   "
            + ("κ = **ausente**" if k.kappa is None else f"**κ = {k.kappa:.3f}**")
            + ("" if k.kappa is None else ("  ✅ ≥ 0,6" if k.kappa >= UMBRAL else "  ✗ < 0,6"))
        )

    k_consenso = tres["contra_el_consenso"]
    obs.decir("\n  **1 · contra el CONSENSO** — lo que el acuerdo entre anotadores compromete\n")
    obs.decir(_linea("juez vs consenso", k_consenso))
    obs.decir(
        "\n  matriz de confusión (juez ↓ / consenso →)\n"
        + "        "
        + "".join(f"{f'g{g}':>6}" for g in GRADOS)
    )
    for g1 in GRADOS:
        obs.decir(
            f"    g{g1}  "
            + "".join(
                f"{(k_consenso.matriz.get(f'grado_{g1}') or {}).get(f'grado_{g2}', 0):>6}"
                for g2 in GRADOS
            )
        )

    obs.decir(
        "\n  **estratificado por configuración de origen** — mitigación 2 de la evaluación\n"
    )
    for configuración, k in sorted(tres["por_configuracion_de_origen"].items()):
        obs.decir(_linea(f"juez vs consenso · {configuración}", k))
    obs.decir(f"\n  {tres['nota_de_los_estratos']}\n\n  {tres['nota_de_la_mitigacion_2']}")

    if tres["por_categoria"]:
        obs.decir(
            f"\n{LÍNEA}\n🔴 **El κ del juez por CATEGORÍA** — contra el κ humano de la "
            f"decisión del acuerdo entre anotadores\n{LÍNEA}\n"
        )
        obs.decir(
            f"  {'categoría':<22} {'n':>4} {'κ juez vs consenso':>20} "
            f"{'κ humano (limpio)':>19}"
        )
        HUMANO = {
            "conceptual_cultural": "0,810 (n=83)",
            "visual_cromatico": "0,249 (n=86)",
            "ambiguo": "0,228 (n=100)",
        }
        for categoria, d in sorted(tres["por_categoria"].items()):
            k = d["contra_el_consenso"]
            obs.decir(
                f"  {categoria:<22} {k.n:>4} "
                + (f"{'ausente':>20}" if k.kappa is None else f"{k.kappa:>20.3f}")
                + f" {HUMANO.get(categoria, '—'):>19}"
            )
            for nombre, kk in sorted(d["contra_cada_anotador"].items()):
                obs.decir(
                    f"    └ vs {nombre:<16} {kk.n:>4} "
                    + ("ausente" if kk.kappa is None else f"{kk.kappa:.3f}")
                )
        obs.decir(f"\n  {tres['nota_del_kappa_por_categoria']}")
    obs.decir(
        f"\n{LÍNEA}\n🔑 **2 y 3 · contra CADA anotador por separado** — el efecto proveedor\n"
        f"{LÍNEA}\n"
    )
    for nombre, k in sorted(tres["contra_cada_anotador"].items()):
        obs.decir(_linea(f"juez vs {nombre}", k))
    de_openai = tres["contra_cada_anotador"].get("ia-codex")
    de_otro = tres["contra_cada_anotador"].get("ia-claude")
    if de_openai is not None and de_otro is not None and None not in (
        de_openai.kappa, de_otro.kappa
    ):
        delta = de_openai.kappa - de_otro.kappa
        obs.decir(
            f"\n  🔑 **El juez es OpenAI (`{elegido.modelo}`) y `ia-codex` también; "
            f"`ia-claude` es de otra familia.**\n"
            f"     κ(juez, ia-codex) − κ(juez, ia-claude) = **{delta:+.3f}**\n"
            + (
                "     → **concuerda MÁS con el anotador de su propio proveedor**: el efecto "
                "proveedor\n       está **medido directamente**, y eso vuelve interpretable "
                "el 71 %/22 % del desempate.\n"
                if delta > 0.05
                else (
                    "     → **concuerda MÁS con el anotador del OTRO proveedor**, que es lo "
                    "contrario de\n       lo que la hipótesis del efecto proveedor predice. "
                    "Es un resultado y va dicho.\n"
                    if delta < -0.05
                    else "     → **no hay diferencia apreciable** (|Δ| ≤ 0,05): el efecto "
                    "proveedor **queda\n       acotado**, y eso también es un resultado "
                    "(la evaluación: *un κ bajo honestamente\n reportado es un hallazgo*).\n"
                )
            )
            + "     ⚠️ **Y es un Δ entre dos κ sobre dos poblaciones distintas** (los pares "
            "que el juez\n       y cada anotador juzgaron): las `n` van al lado y no se "
            "promedian."
        )
    obs.decir(
        f"\n  ⚠️ **Qué NO prueba esto.** El κ del juez **no sostiene la tabla C0–C4**: "
        f"la evaluación\n mitigación 1 y los modos de falla del proyecto lo declaran modo de falla **no "
        f"aceptable**\n     (*«sostener la afirmación v1→v2 con el LLM-judge en vez de con "
        f"anotación»*). Lo que\n     el juez agrega es **otra frontera de proveedor medida**, "
        f"que es lo que el desempate dejó\n abierto — y el valor del juez ya dice que su valor *depende del "
        f"modelo, no de nosotros*.\n{LÍNEA}"
    )

    if args.json:
        destino = almacén.escribir_json(
            args.json,
            {
                "artefacto": "KappaDelJuez",
                "es_una_lectura": True,
                "aporte_del_juez": elegido.a_json(),
                "pool_id": pool.pool_id,
                "rubrica_id": args.rubrica,
                "contra_el_consenso": k_consenso.a_json(),
                "por_configuracion_de_origen": {
                    c: k.a_json() for c, k in tres["por_configuracion_de_origen"].items()
                },
                "contra_cada_anotador": {
                    n: k.a_json() for n, k in tres["contra_cada_anotador"].items()
                },
                "por_categoria": {
                    c: {
                        "contra_el_consenso": d["contra_el_consenso"].a_json(),
                        "contra_cada_anotador": {
                            n: k.a_json() for n, k in d["contra_cada_anotador"].items()
                        },
                    }
                    for c, d in tres["por_categoria"].items()
                },
                "nota_del_kappa_por_categoria": tres["nota_del_kappa_por_categoria"],
                "nota_de_los_estratos": tres["nota_de_los_estratos"],
                "nota_de_la_mitigacion_2": tres["nota_de_la_mitigacion_2"],
                "nota_de_la_mitigacion_1": (
                    "🔴 El κ del juez **no sostiene la tabla C0–C4** (la evaluación mitigación 1, "
                    "los modos de falla del proyecto: modo de falla **no aceptable**). Lo que agrega es otra "
                    "frontera de proveedor medida."
                ),
            },
        )
        obs.decir(f"  JSON: {destino}")
    del _json
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
