"""Arma una planilla HTML para juzgar a mano los pares (consulta, obra) sin juicio.

    python analisis/cascada.py     # deja artifacts/analisis/pares-sin-juzgar.json
    python analisis/planilla.py    # escribe artifacts/analisis/planilla-juicios.html

La planilla se abre en el navegador desde esa carpeta (usa las imágenes de data/imagenes),
es ciega a la configuración que recuperó cada obra y exporta un CSV con las columnas
consulta_id, artwork_id, grado, nota, anotador, que `cascada.py --juicios-extra` lee tal cual.
"""
from __future__ import annotations

import glob
import html
import json
import random
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "artifacts" / "analisis"


def consultas() -> dict[str, dict]:
    """consulta_id -> {imagen, pedido, categoria}, leído de consultas-g2.toml sin dependencias."""
    texto = (RAIZ / "consultas-g2.toml").read_text(encoding="utf-8")
    salida, imagen = {}, None
    for bloque in re.split(r"\n(?=\[\[)", texto):
        campos = dict(re.findall(r'^(\w+)\s*=\s*"([^"]*)"', bloque, flags=re.M))
        if bloque.startswith("[[imagenes]]"):
            imagen = campos.get("artwork_id")
        elif bloque.startswith("[[imagenes.consultas]]") and "consulta_id" in campos:
            salida[campos["consulta_id"]] = {"imagen": imagen, "pedido": campos.get("pedido", ""), "categoria": campos.get("categoria", "")}
    return salida


def fichas() -> dict[str, dict]:
    salida: dict[str, dict] = {}
    for archivo in glob.glob(str(RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        with open(archivo, encoding="utf-8") as f:
            estado = json.load(f)["estado"]
        for obra in ((estado.get("salida") or {}).get("obras") or []):
            salida.setdefault(obra["artwork_id"], {**(obra.get("ficha") or {}), "source_url": obra.get("source_url")})
    return salida


def ruta_de_imagen(artwork_id: str) -> str:
    return "../../data/imagenes/" + artwork_id.replace(":", "_") + ".jpg"


def main() -> None:
    pares = json.loads((SALIDA / "pares-sin-juzgar.json").read_text(encoding="utf-8"))
    cons, fich = consultas(), fichas()
    random.Random(7369).shuffle(pares)  # ciego al orden del ranking
    por_consulta: dict[str, list[str]] = {}
    for consulta, obra in pares:
        por_consulta.setdefault(consulta, []).append(obra)

    e = html.escape
    secciones = []
    for consulta in sorted(por_consulta):
        c = cons[consulta]
        tarjetas = []
        for obra in por_consulta[consulta]:
            f = fich.get(obra, {})
            datos = " · ".join(str(f[k]) for k in ("artist", "culture", "date_display", "medium") if f.get(k))
            enlace = f'<a href="{e(f["source_url"])}" target="_blank" rel="noopener">ver en el museo</a>' if f.get("source_url") else ""
            clave = f"{consulta}|{obra}"
            radios = "".join(
                f'<label class="g g{g}"><input type="radio" name="{e(clave)}" value="{g}"> {g}</label>' for g in (0, 1, 2)
            )
            tarjetas.append(
                f'<article class="obra" data-clave="{e(clave)}">'
                f'<img loading="lazy" src="{e(ruta_de_imagen(obra))}" alt="">'
                f'<h3>{e(str(f.get("title") or obra))}</h3><p class="datos">{e(datos)}</p><p class="datos">{e(obra)} {enlace}</p>'
                f'<div class="grados">{radios}</div>'
                f'<input class="nota" type="text" placeholder="nota (opcional; si dudás, el grado es 1)"></article>'
            )
        secciones.append(
            f'<section><header class="consulta"><img src="{e(ruta_de_imagen(c["imagen"]))}" alt="">'
            f'<div><p class="id">{e(consulta)} · {e(c["categoria"])}</p><p class="pedido">«{e(c["pedido"])}»</p>'
            f'<p class="datos">Imagen original: {e(c["imagen"])}. ¿La obra de abajo cumple el pedido respecto de esta imagen?</p></div></header>'
            f'<div class="obras">{"".join(tarjetas)}</div></section>'
        )

    pagina = PLANTILLA.replace("{{SECCIONES}}", "\n".join(secciones)).replace("{{N}}", str(len(pares)))
    (SALIDA / "planilla-juicios.html").write_text(pagina, encoding="utf-8")
    faltan = [o for _, o in pares if not (RAIZ / "data" / "imagenes" / (o.replace(":", "_") + ".jpg")).exists()]
    print(f"{len(pares)} pares en {len(por_consulta)} consultas → artifacts/analisis/planilla-juicios.html" + (f" · ⚠ sin imagen local: {faltan}" if faltan else ""))


PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planilla de juicios</title>
<style>
:root{--fondo:#f6f5f2;--tinta:#1d1d1f;--suave:#6b6b70;--borde:#d9d7d0;--tarjeta:#fff;--ok:#1f7a4d}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--fondo);color:var(--tinta)}
.barra{position:sticky;top:0;z-index:2;background:var(--tarjeta);border-bottom:1px solid var(--borde);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.barra input{padding:6px 8px;border:1px solid var(--borde);border-radius:6px;font:inherit}
.barra button{padding:7px 14px;border:0;border-radius:6px;background:var(--tinta);color:#fff;font:inherit;cursor:pointer}
main{max-width:1200px;margin:0 auto;padding:16px}
.rubrica{background:var(--tarjeta);border:1px solid var(--borde);border-radius:8px;padding:12px 16px;margin-bottom:20px}
.rubrica li{margin:4px 0}
section{margin-bottom:36px}
.consulta{display:flex;gap:16px;align-items:flex-start;background:var(--tarjeta);border:1px solid var(--borde);border-radius:8px;padding:12px;margin-bottom:12px}
.consulta img{width:220px;max-height:260px;object-fit:contain;background:#eee;border-radius:4px}
.id{margin:0;color:var(--suave);font-size:13px}.pedido{margin:4px 0 8px;font-size:20px;font-weight:600}
.datos{margin:2px 0;color:var(--suave);font-size:13px}
.obras{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.obra{background:var(--tarjeta);border:2px solid var(--borde);border-radius:8px;padding:10px}
.obra.hecha{border-color:var(--ok)}
.obra img{width:100%;height:240px;object-fit:contain;background:#eee;border-radius:4px}
.obra h3{margin:8px 0 2px;font-size:15px}
.grados{display:flex;gap:6px;margin:8px 0}
.g{flex:1;text-align:center;padding:6px;border:1px solid var(--borde);border-radius:6px;cursor:pointer}
.nota{width:100%;padding:6px;border:1px solid var(--borde);border-radius:6px;font:inherit;font-size:13px}
</style></head><body>
<div class="barra"><strong>Planilla de juicios</strong>
<label>Anotador: <input id="anotador" placeholder="tu nombre"></label>
<span id="progreso">0 de {{N}}</span>
<button id="descargar">Descargar CSV</button></div>
<main>
<div class="rubrica"><strong>Rúbrica v1, resumida</strong> (la completa está en <code>artifacts/ground-truth/rubricas/v1.md</code>)
<ul><li><strong>2</strong>: cumple claramente el pedido respecto de la imagen original.</li>
<li><strong>1</strong>: cumple parcialmente o es defendible. Si dudás entre dos grados, es 1.</li>
<li><strong>0</strong>: no cumple. También es 0 si la obra es la misma imagen original.</li></ul>
Juzgá cada obra por separado, sin compararla con las otras de la misma consulta. Cada persona usa su propia copia y no mira los grados de la otra.</div>
{{SECCIONES}}
</main>
<script>
const obras=[...document.querySelectorAll('.obra')], anotador=document.getElementById('anotador');
const CLAVE='planilla-juicios-v1';
let estado={};
try{estado=JSON.parse(localStorage.getItem(CLAVE)||'{}')}catch(e){}
function guardar(){try{localStorage.setItem(CLAVE,JSON.stringify(estado))}catch(e){}}
function pintar(){let n=0;obras.forEach(o=>{const h=estado[o.dataset.clave]&&estado[o.dataset.clave].grado!==undefined;o.classList.toggle('hecha',!!h);if(h)n++});
 document.getElementById('progreso').textContent=n+' de '+obras.length}
anotador.value=estado.__anotador||'';
anotador.addEventListener('input',()=>{estado.__anotador=anotador.value;guardar()});
obras.forEach(o=>{const k=o.dataset.clave,v=estado[k]||{};
 o.querySelectorAll('input[type=radio]').forEach(r=>{if(String(v.grado)===r.value)r.checked=true;
  r.addEventListener('change',()=>{estado[k]=Object.assign(estado[k]||{},{grado:Number(r.value)});guardar();pintar()})});
 const nota=o.querySelector('.nota');nota.value=v.nota||'';
 nota.addEventListener('input',()=>{estado[k]=Object.assign(estado[k]||{},{nota:nota.value});guardar()})});
pintar();
document.getElementById('descargar').addEventListener('click',()=>{
 const q=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
 const filas=['consulta_id,artwork_id,grado,nota,anotador'];
 obras.forEach(o=>{const v=estado[o.dataset.clave];if(v&&v.grado!==undefined){const p=o.dataset.clave.split('|');
  filas.push([p[0],p[1],v.grado,q(v.nota),q(anotador.value)].join(','))}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([filas.join('\\n')],{type:'text/csv'}));
 a.download='juicios-'+(anotador.value||'anotador').replace(/\\W+/g,'-')+'.csv';a.click()});
</script></body></html>
"""

if __name__ == "__main__":
    main()
