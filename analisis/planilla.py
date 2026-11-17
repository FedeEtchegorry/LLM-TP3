"""Planilla HTML para juzgar los pares del set de 30 que todavía no tienen grado. No gasta.

    python analisis/cascada.py      # actualiza pares-sin-juzgar.json
    python analisis/planilla.py

Escribe `artifacts/analisis/planilla-juicios.html` con los pares de `pares-sin-juzgar.json`:
la imagen original, el pedido y cada obra, sin decir qué configuración la recuperó y en un
orden mezclado con semilla fija. Cada persona usa su copia y descarga
`juicios-<nombre>.csv`, que cascada.py lee solo desde artifacts/analisis.
"""
from __future__ import annotations

import collections
import html
import json
import random
import re

import cascada as c
import planilla_mixta as pm

SALIDA = c.SALIDA / "planilla-juicios.html"
DECLARACION = c.RAIZ / "consultas-g2.toml"

RUBRICA = (
    '<div class="rubrica"><strong>Rúbrica v1, resumida</strong> (la completa está en '
    "<code>artifacts/ground-truth/rubricas/v1.md</code>)"
    "<ul><li><strong>2</strong>: cumple claramente el pedido respecto de la imagen original.</li>"
    "<li><strong>1</strong>: cumple parcialmente o es defendible. Si dudás entre dos grados, es 1.</li>"
    "<li><strong>0</strong>: no cumple. También es 0 si la obra es la misma imagen original.</li></ul>"
    "Juzgá cada obra por separado, sin compararla con las otras de la misma consulta. Cada persona "
    "usa su propia copia y no mira los grados de la otra. El avance queda guardado en este navegador.</div>"
)

GUION = """
const obras=[...document.querySelectorAll('.obra')], anotador=document.getElementById('anotador');
const CLAVE='planilla-juicios-v2';
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
 a.download='juicios-pendientes-'+(anotador.value||'anotador').replace(/\\W+/g,'-')+'.csv';a.click()});
"""


def _consultas() -> dict[str, dict]:
    consultas, imagen, actual = {}, None, {}
    for linea in DECLARACION.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*(artwork_id|consulta_id|categoria|pedido)\s*=\s*"(.*)"\s*$', linea)
        if not m:
            continue
        clave, valor = m.groups()
        if clave == "artwork_id":
            imagen = valor
        elif clave == "consulta_id":
            actual = consultas.setdefault(valor, {"imagen": imagen})
        else:
            actual[clave] = valor
    return consultas


def _obra(consulta: str, artwork_id: str, r: dict) -> str:
    e = html.escape
    clave = f"{consulta}|{artwork_id}"
    datos = " · ".join(str(x) for x in (r.get("artist"), (r.get("culture_raw") or [None])[0], r.get("date_display"), r.get("medium")) if x)
    radios = "".join(
        f'<label class="g g{g}"><input type="radio" name="{e(clave)}" value="{g}"> {g}</label>' for g in (0, 1, 2)
    )
    return (
        f'<article class="obra" data-clave="{e(clave)}"><img loading="lazy" src="{e(pm._imagen(artwork_id))}" alt="">'
        f'<h3>{e(r.get("title") or artwork_id)}</h3><p class="datos">{e(datos)}</p>'
        f'<p class="datos">{e(artwork_id)} <a href="{e(r.get("source_url") or "#")}" target="_blank" rel="noopener">ver en el museo</a></p>'
        f'<div class="grados">{radios}</div><input class="nota" placeholder="nota (opcional)"></article>'
    )


def main() -> None:
    e = html.escape
    pares = json.loads((c.SALIDA / "pares-sin-juzgar.json").read_text(encoding="utf-8"))
    por_consulta: dict[str, list[str]] = collections.defaultdict(list)
    for consulta, artwork_id in pares:
        if artwork_id not in por_consulta[consulta]:
            por_consulta[consulta].append(artwork_id)
    mezcla = random.Random(7369)
    catalogo, consultas = pm._catalogo(), _consultas()
    secciones = []
    for consulta in sorted(por_consulta):
        obras = sorted(por_consulta[consulta])
        mezcla.shuffle(obras)
        d = consultas[consulta]
        secciones.append(
            f'<section><header class="consulta"><img src="{e(pm._imagen(d["imagen"]))}" alt=""><div>'
            f'<p class="id">{e(consulta)} · {e(d.get("categoria", ""))} · {len(obras)} obras</p>'
            f'<p class="pedido">«{e(d["pedido"])}»</p>'
            f'<p class="datos">Imagen original: {e(d["imagen"])}. ¿La obra de abajo cumple el pedido respecto de esta imagen?</p>'
            f'</div></header><div class="obras">{"".join(_obra(consulta, o, catalogo.get(o, {})) for o in obras)}</div></section>'
        )
    total = sum(len(v) for v in por_consulta.values())
    estilo = pm.ESTILO + ".grados{display:flex;gap:6px;margin:8px 0}.g{flex:1;text-align:center;padding:6px;border:1px solid var(--borde);border-radius:6px;cursor:pointer}"
    pagina = (
        '<!doctype html>\n<html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Planilla de juicios</title><style>{estilo}</style></head><body>"
        '<div class="barra"><strong>Juicios pendientes del set de 30</strong>'
        '<label>Anotador: <input id="anotador" placeholder="tu nombre"></label>'
        f'<span id="progreso">0 de {total}</span><button id="descargar">Descargar CSV</button></div><main>'
        f'{RUBRICA}{"".join(secciones)}</main><script>{GUION}</script></body></html>\n'
    )
    SALIDA.write_text(pagina, encoding="utf-8")
    print(f"{SALIDA.relative_to(c.RAIZ)}: {len(por_consulta)} consultas, {total} pares")


if __name__ == "__main__":
    main()
