"""Planilla HTML para juzgar las consultas mixtas. No gasta.

    python analisis/planilla_mixta.py

Lee el top-5 de todas las corridas `cNN-mixto` y escribe
`artifacts/analisis/planilla-juicios-mixtas.html`. Cada persona abre su copia, marca A, B y C
por obra (rúbrica: docs/rubrica-mixta.md) y descarga un CSV `juicios-mixtas-<nombre>.csv`.
El grado sale solo: A + B, y en el control (sin restricción) A + C.
La planilla no dice qué configuración recuperó cada obra y las mezcla con semilla fija.
"""
from __future__ import annotations

import glob
import html
import json
import random
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "artifacts" / "analisis" / "planilla-juicios-mixtas.html"
DECLARACION = RAIZ / "consultas-g2-mixto.toml"

# Los tres componentes de cada pedido, leídos del pedido declarado. B = None es el control.
COMPONENTES = {
    "c01-mixto": ("tonos más cálidos", "óleo y siglo XIX (1801–1900)", "más calma"),
    "c02-mixto": ("luz dorada", "pintura holandesa", "más serena"),
    "c03-mixto": ("tonos más fríos", "pintura italiana del barroco (aprox. 1600–1750)", "más imponente"),
    "c04-mixto": ("colores más suaves", "óleo y siglo XX (1901–2000)", "más ordenada"),
    "c05-mixto": ("paleta más luminosa", "pintura francesa del siglo XVIII (1701–1800)", "más cercano"),
    "c06-mixto": ("colores más apagados", None, "más humano"),
    "c07-mixto": ("tonos más fríos", "China, dinastía Ming (1368–1644)", "más austero"),
    "c08-mixto": ("colores más apagados", "Japón, época Edo (1603–1868)", "más caótica"),
    "c09-mixto": ("más color", "tinta, Japón, período Edo (1603–1868)", "más delicado"),
    "c10-mixto": ("paleta más oscura", "arte mogol del siglo XVII (1601–1700)", "más generosa"),
}

ESTILO = """
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
.comp{margin:3px 0;font-size:14px}.comp b{display:inline-block;width:1.3em}
.obras{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.obra{background:var(--tarjeta);border:2px solid var(--borde);border-radius:8px;padding:10px}
.obra.hecha{border-color:var(--ok)}
.obra img{width:100%;height:240px;object-fit:contain;background:#eee;border-radius:4px}
.obra h3{margin:8px 0 2px;font-size:15px}
.meta{margin:6px 0;padding:6px 8px;background:var(--fondo);border-radius:6px;font-size:13px}
.meta div{margin:1px 0}
.fila{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px}
.fila span{flex:1}.fila label{padding:3px 9px;border:1px solid var(--borde);border-radius:6px;cursor:pointer}
.opc{color:var(--suave)}
.grado{margin:6px 0;font-weight:600}
.nota{width:100%;padding:6px;border:1px solid var(--borde);border-radius:6px;font:inherit;font-size:13px}
"""

GUION = """
const obras=[...document.querySelectorAll('.obra')], anotador=document.getElementById('anotador');
const CLAVE='planilla-juicios-mixtas-v1';
let estado={};
try{estado=JSON.parse(localStorage.getItem(CLAVE)||'{}')}catch(e){}
function guardar(){try{localStorage.setItem(CLAVE,JSON.stringify(estado))}catch(e){}}
function grado(o,v){if(!v||v.a===undefined)return null;const s=o.dataset.control?v.c:v.b;if(s===undefined)return null;return v.a+s}
function pintar(){let n=0;obras.forEach(o=>{const g=grado(o,estado[o.dataset.clave]);o.classList.toggle('hecha',g!==null);
 o.querySelector('.grado').textContent='Grado: '+(g===null?'—':g);if(g!==null)n++});
 document.getElementById('progreso').textContent=n+' de '+obras.length}
anotador.value=estado.__anotador||'';
anotador.addEventListener('input',()=>{estado.__anotador=anotador.value;guardar()});
obras.forEach(o=>{const k=o.dataset.clave,v=estado[k]||{};
 o.querySelectorAll('input[type=radio]').forEach(r=>{const c=r.dataset.comp;if(String(v[c])===r.value)r.checked=true;
  r.addEventListener('change',()=>{estado[k]=Object.assign(estado[k]||{},{[c]:Number(r.value)});guardar();pintar()})});
 const nota=o.querySelector('.nota');nota.value=v.nota||'';
 nota.addEventListener('input',()=>{estado[k]=Object.assign(estado[k]||{},{nota:nota.value});guardar()})});
pintar();
document.getElementById('descargar').addEventListener('click',()=>{
 const q=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
 const x=n=>n===undefined?'':n;
 const filas=['consulta_id,artwork_id,grado,nota,anotador,cambio_visual,restriccion,adjetivo'];
 obras.forEach(o=>{const v=estado[o.dataset.clave],g=grado(o,v);if(g!==null){const p=o.dataset.clave.split('|');
  filas.push([p[0],p[1],g,q(v.nota),q(anotador.value),x(v.a),x(v.b),x(v.c)].join(','))}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([filas.join('\\n')],{type:'text/csv'}));
 a.download='juicios-mixtas-'+(anotador.value||'anotador').replace(/\\W+/g,'-')+'.csv';a.click()});
"""


def _imagen(artwork_id: str) -> str:
    return f"../../data/imagenes/{artwork_id.replace(':', '_')}.jpg"


def _catalogo() -> dict[str, dict]:
    registros = {}
    for archivo in glob.glob(str(RAIZ / "data" / "catalogo" / "*" / "registros.jsonl")):
        with open(archivo, encoding="utf-8") as f:
            for linea in f:
                r = json.loads(linea)
                registros[r["artwork_id"]] = r
    return registros


def _consultas() -> dict[str, dict]:
    """{consulta_id: pedido e imagen}, leído línea por línea para no depender de la versión de Python."""
    consultas, imagen, consulta = {}, None, None
    for linea in DECLARACION.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*(artwork_id|consulta_id|pedido)\s*=\s*"(.*)"\s*$', linea)
        if not m:
            continue
        clave, valor = m.groups()
        if clave == "artwork_id":
            imagen = valor
        elif clave == "consulta_id":
            consulta = valor
        else:
            consultas[consulta] = {"pedido": valor, "imagen": imagen}
    return consultas


def _pares() -> dict[str, list[str]]:
    obras: dict[str, set[str]] = {}
    for archivo in glob.glob(str(RAIZ / "artifacts" / "corridas" / "*" / "corrida.json")):
        with open(archivo, encoding="utf-8") as f:
            corrida = json.load(f)
        if corrida["consulta_id"].endswith("-mixto"):
            obras.setdefault(corrida["consulta_id"], set()).update(corrida["estado"].get("top5") or [])
    mezcla = random.Random(7369)
    salida = {}
    for consulta in sorted(obras):
        lista = sorted(obras[consulta])
        mezcla.shuffle(lista)
        salida[consulta] = lista
    return salida


def _fila(clave: str, comp: str, texto: str, opcional: bool = False) -> str:
    e = html.escape
    return (
        f'<div class="fila{" opc" if opcional else ""}"><span>{texto}</span>'
        f'<label><input type="radio" name="{e(clave)}|{comp}" data-comp="{comp}" value="1"> sí</label>'
        f'<label><input type="radio" name="{e(clave)}|{comp}" data-comp="{comp}" value="0"> no</label></div>'
    )


def _obra(consulta: str, artwork_id: str, r: dict, control: bool) -> str:
    e = html.escape
    clave = f"{consulta}|{artwork_id}"
    cultura = ", ".join(r.get("culture_raw") or []) or "sin dato"
    filas = [_fila(clave, "a", "<b>A</b> cambio visual")]
    if control:
        filas.append(_fila(clave, "c", "<b>C</b> adjetivo"))
    else:
        filas.append(_fila(clave, "b", "<b>B</b> restricción (por metadata)"))
        filas.append(_fila(clave, "c", "<b>C</b> adjetivo (no suma)", opcional=True))
    marca = ' data-control="1"' if control else ""
    return (
        f'<article class="obra" data-clave="{e(clave)}"{marca}>'
        f'<img loading="lazy" src="{e(_imagen(artwork_id))}" alt="">'
        f'<h3>{e(r.get("title") or artwork_id)}</h3>'
        f'<p class="datos">{e(r.get("artist") or "autor sin dato")} · {e(artwork_id)} '
        f'<a href="{e(r.get("source_url") or "#")}" target="_blank" rel="noopener">ver en el museo</a></p>'
        f'<div class="meta"><div><strong>culture:</strong> {e(cultura)}</div>'
        f'<div><strong>medium:</strong> {e(r.get("medium") or "sin dato")}</div>'
        f'<div><strong>year:</strong> {e(str(r.get("year") or "sin dato"))} ({e(r.get("date_display") or "sin fecha")})</div>'
        f'<div><strong>style:</strong> {e(r.get("style") or "sin dato")}</div></div>'
        f'{"".join(filas)}<p class="grado">Grado: —</p>'
        f'<input class="nota" placeholder="nota (opcional)"></article>'
    )


def main() -> None:
    e = html.escape
    catalogo, consultas, pares = _catalogo(), _consultas(), _pares()
    total = sum(len(v) for v in pares.values())
    secciones = []
    for consulta, obras in pares.items():
        a, b, c = COMPONENTES[consulta]
        datos = consultas[consulta]
        control = b is None
        restriccion = f"<p class='comp'><b>B</b> Restricción: {e(b)}. Se verifica en la metadata, no a ojo.</p>" if b else \
            "<p class='comp'><b>B</b> Sin restricción: es la consulta de control. El grado es A + C.</p>"
        secciones.append(
            f'<section><header class="consulta"><img src="{e(_imagen(datos["imagen"]))}" alt=""><div>'
            f'<p class="id">{e(consulta)} · {len(obras)} obras</p><p class="pedido">«{e(datos["pedido"])}»</p>'
            f"<p class='comp'><b>A</b> Cambio visual: {e(a)}, respecto de esta imagen.</p>{restriccion}"
            f"<p class='comp'><b>C</b> Adjetivo: {e(c)}.{'' if control else ' No suma al grado.'}</p>"
            f'<p class="datos">Imagen original: {e(datos["imagen"])}.</p></div></header>'
            f'<div class="obras">{"".join(_obra(consulta, o, catalogo.get(o, {}), control) for o in obras)}</div></section>'
        )
    pagina = (
        '<!doctype html>\n<html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Planilla de juicios: consultas mixtas</title><style>{ESTILO}</style></head><body>"
        '<div class="barra"><strong>Juicios de consultas mixtas</strong>'
        '<label>Anotador: <input id="anotador" placeholder="tu nombre"></label>'
        f'<span id="progreso">0 de {total}</span><button id="descargar">Descargar CSV</button></div><main>'
        '<div class="rubrica"><strong>Rúbrica mixta, resumida</strong> (la completa está en <code>docs/rubrica-mixta.md</code>)'
        "<ul><li><strong>A, cambio visual</strong>: la obra tiene el cambio de color o luz pedido respecto de la imagen original. Se mira la obra al lado de la original.</li>"
        "<li><strong>B, restricción</strong>: la obra es de la cultura, técnica o época que el pedido nombra. Se verifica en <em>culture</em>, <em>medium</em> y <em>year</em>. Si falta el dato, es «no» y se anota.</li>"
        "<li><strong>C, adjetivo vago</strong>: es subjetivo y no suma al grado, salvo en el control (c06), donde reemplaza a B.</li>"
        "<li><strong>Grado</strong> = A + B: 2 si cumple las dos, 1 si cumple una, 0 si ninguna. Se calcula solo.</li></ul>"
        "Juzgá cada obra por separado. Cada persona usa su propia copia y no mira los grados de la otra. "
        "El avance queda guardado en este navegador.</div>"
        f'{"".join(secciones)}</main><script>{GUION}</script></body></html>\n'
    )
    SALIDA.write_text(pagina, encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)}: {len(pares)} consultas, {total} pares")


if __name__ == "__main__":
    main()
