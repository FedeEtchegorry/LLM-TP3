from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RAÍZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAÍZ / "src"))

from tp3_rag.nucleo.configuracion import Configuración  # noqa: E402
from tp3_rag.nucleo.credenciales import Credenciales  # noqa: E402
from tp3_rag.nucleo.identidad import clave_de_cache_http  # noqa: E402
from tp3_rag.puertos.cliente_http import ClienteHTTP  # noqa: E402
from tp3_rag.puertos.museos.aic import AdaptadorAIC  # noqa: E402
from tp3_rag.puertos.museos.cma import AdaptadorCMA  # noqa: E402

AQUÍ = Path(__file__).parent
CACHE_FIXTURE = AQUÍ / "http-cache"
N = 8

def _copiar(cache_real: Path, url: str, nombres: list[str]) -> bool:
    clave = clave_de_cache_http(url, nombres)
    origen = cache_real / clave[:2]
    destino = CACHE_FIXTURE / clave[:2]
    destino.mkdir(parents=True, exist_ok=True)
    copiados = 0
    for sufijo in (".bin", ".json"):
        f = origen / f"{clave}{sufijo}"
        if f.exists():
            shutil.copy2(f, destino / f.name)
            copiados += 1
    return copiados == 2

def main() -> int:
    cfg = Configuración.cargar(overrides={"ingesta.limite_n": N})
    credenciales = Credenciales.desde_entorno()
    cliente = ClienteHTTP.desde_configuración(cfg, credenciales)
    cache_real = cliente.cache_dir

    if CACHE_FIXTURE.exists():
        shutil.rmtree(CACHE_FIXTURE)
    CACHE_FIXTURE.mkdir(parents=True)

    urls: list[str] = []
    crudos: dict[str, list] = {}

    for museo, fábrica in (("aic", AdaptadorAIC), ("cma", AdaptadorCMA)):
        adaptador = fábrica(cliente, cfg)
        ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", N)
        crudos[museo] = [adaptador.traer(i) for i in ids]
        if museo == "aic":
            urls.append(adaptador._query(list(cfg["ingesta.tipos"]), "libre", 0))
        else:
            urls.append(adaptador._url(list(cfg["ingesta.tipos"]), "libre", 0, N))
        for i in ids[:2]:
            from tp3_rag.dominio.urls import url_de_imagen

            u = url_de_imagen(museo, adaptador.traer(i), cfg["ingesta.urls"])
            if u:
                urls.append(u)

    cma = AdaptadorCMA(cliente, cfg)
    corot = cma.traer(124078)
    crudos["cma"].append(corot)
    print(f"  cma:124078 · culture = {corot.get('culture')}")

    sin_description = [c for c in crudos["aic"] if not c.get("description")]
    print(f"  AIC sin `description` en el fixture: {len(sin_description)}")
    if not sin_description:
        print("  ⚠️ ninguno de los primeros AIC vino sin `description`; ampliá N")

    faltantes = [u for u in urls if not _copiar(cache_real, u, sorted(cliente.headers_de(u)))]
    if faltantes:
        print(f"  ⚠️ no se pudieron copiar {len(faltantes)} entradas: {faltantes}")

    for museo, registros in crudos.items():
        (AQUÍ / f"crudos-{museo}.json").write_text(
            json.dumps(registros, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    n_bin = len(list(CACHE_FIXTURE.rglob("*.bin")))
    peso = sum(f.stat().st_size for f in CACHE_FIXTURE.rglob("*")) / 1e6
    print(f"  fixture: {n_bin} respuestas · {peso:.1f} MB · crudos: "
          f"{ {m: len(v) for m, v in crudos.items()} }")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
