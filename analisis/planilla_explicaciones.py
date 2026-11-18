"""Planilla HTML para validar a mano la revisión de explicaciones hecha por el asistente. No gasta.

    python analisis/planilla_explicaciones.py

Elige 22 de las 51 afirmaciones de `claude-c1-5x5-v2.json`:
  - las 9 en las que las dos revisiones con IA (m13b y claude v2) no coinciden, para desempatarlas;
  - 13 más al azar (semilla fija) entre las que coinciden, para estimar el acuerdo sin sesgo.
La planilla no muestra ningún veredicto y mezcla el orden. Escribe
`artifacts/analisis/planilla-explicaciones.html`; quien la complete descarga
`juicios-explicaciones-<nombre>.csv` y lo deja en artifacts/analisis/ para `acuerdo_explicaciones.py`.
"""
from __future__ import annotations

import html
import json
import random
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REV = RAIZ / "artifacts" / "ground-truth" / "revision-explicaciones"
SALIDA = RAIZ / "artifacts" / "analisis" / "planilla-explicaciones.html"
SEMILLA = 7369
N_AL_AZAR = 13


def _misma(a_nuevo, a_viejo) -> bool:
    nuevo_bien = a_nuevo["clasificacion"] == "bien"
    viejo_bien = a_viejo["veredicto_clasificacion"] == "bien"
    cita_v = {"no_sostiene": "no"}.get(a_viejo["veredicto_cita"], a_viejo["veredicto_cita"])
    obs_n = None if a_nuevo["observacion"] == "no_visual" else a_nuevo["observacion"]
    obs_v = {"falsa_a_la_vista": "falsa"}.get(a_viejo["veredicto_observacion"], a_viejo["veredicto_observacion"])
    return nuevo_bien == viejo_bien and a_nuevo["cita"] == cita_v and obs_n == obs_v


def elegir():
    nuevo = json.loads((REV / "claude-c1-5x5-v2.json").read_text(encoding="utf-8"))
    viejo = json.loads((REV / "m13b-c1-5x5.json").read_text(encoding="utf-8"))
    previas = {(o["artwork_id"], a["n"]): a for o in viejo["obras"] for a in o["afirmaciones"]}
    evidencia = {o["artwork_id"]: o["evidencia_disponible"] for o in viejo["obras"]}
    items, coinciden = [], []
    for o in nuevo["obras"]:
        for a in o["afirmaciones"]:
            it = {"obra": o, "af": a, "evidencia": evidencia[o["artwork_id"]], "grupo": "desacuerdo"}
            if _misma(a, previas[(o["artwork_id"], a["n"])]):
                it["grupo"] = "azar"
                coinciden.append(it)
            else:
                items.append(it)
    azar = random.Random(SEMILLA)
    items += azar.sample(coinciden, N_AL_AZAR)
    azar.shuffle(items)
    return items


ESTILO = """
body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b1a17;background:#faf8f4}
h1{font-size:22px}.rubrica{background:#fff;border:1px solid #d6cdbd;border-radius:8px;padding:12px 16px;font-size:14px;line-height:1.5}
.item{background:#fff;border:1px solid #d6cdbd;border-radius:8px;margin:18px 0;padding:14px 16px;display:grid;grid-template-columns:280px 1fr;gap:18px}
.item img{width:100%;border-radius:4px;cursor:zoom-in}.meta{font-size:12px;color:#52514e}
.afirm{font-size:16px;background:#f3ece4;border-radius:6px;padding:10px 12px;margin:8px 0}
.tipo{font-weight:700;text-transform:uppercase;font-size:12px;letter-spacing:.04em}
.frag{font-size:13px;background:#eef1f2;border-left:3px solid #7f9c96;padding:8px 10px;margin:6px 0;white-space:pre-wrap}
details{font-size:13px;margin:6px 0}fieldset{border:none;padding:0;margin:10px 0}legend{font-weight:600;font-size:14px;margin-bottom:4px}
label{display:inline-block;margin:3px 10px 3px 0;font-size:14px;cursor:pointer}textarea{width:100%;font:inherit;font-size:13px}
.hecho{border-color:#7f9c96;box-shadow:0 0 0 2px #7f9c9655}.barra{position:sticky;top:0;background:#faf8f4;padding:10px 0;border-bottom:1px solid #d6cdbd;z-index:2}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid #8a6a2f;background:#8a6a2f;color:#fff;cursor:pointer}
@media(max-width:760px){.item{grid-template-columns:1fr}}
"""

RUBRICA = """<div class="rubrica"><strong>Qué hay que hacer.</strong> Cada tarjeta es una afirmación de una explicación generada por el sistema.
El generador marca cada afirmación como <b>dato</b> (viene de un texto del museo y lleva cita) u <b>observación</b> (lo que el modelo ve en la imagen).
Juzgá cada tarjeta por separado, sin buscar otras fuentes.
<ul>
<li><b>Clasificación.</b> ¿El tipo es el correcto? Si está mal, elegí por qué: <i>A</i> algo visual o propio presentado como dato citado;
<i>B</i> algo que dice el texto o la ficha del museo presentado como observación propia; <i>C</i> conocimiento general del modelo presentado como observación. Si mezcla cosas, juzgá por lo central.</li>
<li><b>Cita</b> (solo datos). Leé el fragmento citado: ¿dice lo que afirma? <i>Sí</i> / <i>En parte</i> (alguna parte no está) / <i>No</i>.</li>
<li><b>Lo visual</b> (solo observaciones). Mirá la obra (clic para agrandar): <i>Defendible</i> / <i>Discutible</i> (se puede sostener pero la impresión dominante dice otra cosa) / <i>Falsa a la vista</i> / <i>No afirma nada visible</i>.</li>
</ul>El avance queda guardado en este navegador. Al terminar, descargá el CSV y dejalo en <code>artifacts/analisis/</code>.</div>"""


def _radio(nombre, opciones):
    return "".join(f'<label><input type="radio" name="{nombre}" value="{v}"> {html.escape(t)}</label>' for v, t in opciones)


def tarjeta(i, it):
    o, a = it["obra"], it["af"]
    clave = f'{o["artwork_id"]}#{a["n"]}'
    img = f'../../data/imagenes/{o["artwork_id"].replace(":", "_")}.jpg'
    ficha = " · ".join(str(v) for k, v in o["ficha"].items() if k in ("title", "artist", "culture", "date_display", "medium") and v)
    citados = [e for e in it["evidencia"] if e["fragment_id"] in a["refs"]]
    otros = [e for e in it["evidencia"] if e["fragment_id"] not in a["refs"]]
    frag = "".join(f'<div class="frag"><b>{html.escape(e["fragment_id"])}</b>\n{html.escape(e["texto"])}</div>' for e in citados)
    resto = "".join(f'<div class="frag"><b>{html.escape(e["fragment_id"])}</b>\n{html.escape(e["texto"])}</div>' for e in otros)
    preguntas = '<fieldset><legend>Clasificación</legend>' + _radio(f"c{i}", [("bien", "Bien"), ("A", "Mal: A"), ("B", "Mal: B"), ("C", "Mal: C")]) + "</fieldset>"
    if a["tipo"] == "dato":
        preguntas += '<fieldset><legend>¿La cita sostiene el dato?</legend>' + _radio(f"t{i}", [("sostiene", "Sí"), ("parcial", "En parte"), ("no", "No")]) + "</fieldset>"
    else:
        preguntas += '<fieldset><legend>¿Lo visual es defendible?</legend>' + _radio(f"o{i}", [("defendible", "Defendible"), ("discutible", "Discutible"), ("falsa", "Falsa a la vista"), ("no_visual", "No afirma nada visible")]) + "</fieldset>"
    return f"""<div class="item" data-clave="{clave}" data-tipo="{a['tipo']}" data-i="{i}">
<div><a href="{img}" target="_blank"><img src="{img}" alt="obra"></a><div class="meta">{html.escape(ficha)}</div></div>
<div><div class="meta">Tarjeta {i + 1} · pedido del usuario: «{html.escape(o['pedido'] or '')}»</div>
<div class="afirm"><span class="tipo">{a['tipo']}</span><br>{html.escape(a['texto'])}</div>
{('<div class="meta">Fragmento citado:</div>' + frag) if frag else '<div class="meta">Sin cita.</div>'}
<details><summary>Otros textos del museo que tenía el generador para esta obra</summary>{resto or '—'}</details>
{preguntas}<textarea name="m{i}" rows="1" placeholder="Comentario (opcional)"></textarea></div></div>"""


GUION = """
const CLAVE='planilla-explicaciones-v1';let estado={};try{estado=JSON.parse(localStorage.getItem(CLAVE)||'{}')}catch(e){}
const items=[...document.querySelectorAll('.item')],quien=document.getElementById('quien');
function guardar(){try{localStorage.setItem(CLAVE,JSON.stringify(estado))}catch(e){}}
function marcar(it){const e=estado[it.dataset.clave]||{};const listo=e.c&&(it.dataset.tipo==='dato'?e.t:e.o);it.classList.toggle('hecho',!!listo);}
function avance(){const n=items.filter(i=>i.classList.contains('hecho')).length;document.getElementById('avance').textContent=n+' de '+items.length;}
quien.value=estado.__quien||'';quien.oninput=()=>{estado.__quien=quien.value;guardar()};
items.forEach(it=>{const k=it.dataset.clave,i=it.dataset.i,e=estado[k]||{};
 for(const [campo,pre] of [['c','c'],['t','t'],['o','o']]){it.querySelectorAll(`input[name=${pre}${i}]`).forEach(r=>{if(e[campo]===r.value)r.checked=true;
  r.onchange=()=>{(estado[k]=estado[k]||{})[campo]=r.value;guardar();marcar(it);avance()}})}
 const t=it.querySelector('textarea');t.value=e.m||'';t.oninput=()=>{(estado[k]=estado[k]||{}).m=t.value;guardar()};marcar(it)});
avance();
document.getElementById('bajar').onclick=()=>{const q=(quien.value||'').trim();if(!q){alert('Escribí tu nombre');return}
 const filas=[['revisor','artwork_id','n','tipo','clasificacion','cita','observacion','comentario']];
 items.forEach(it=>{const [a,n]=it.dataset.clave.split('#');const e=estado[it.dataset.clave]||{};
  filas.push([q,a,n,it.dataset.tipo,e.c||'',e.t||'',e.o||'',(e.m||'').replace(/[\\n\\r]+/g,' ')])});
 const csv=filas.map(f=>f.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\n');
 const b=new Blob([csv],{type:'text/csv'}),l=document.createElement('a');l.href=URL.createObjectURL(b);
 l.download='juicios-explicaciones-'+q.toLowerCase().replace(/[^a-z0-9]+/g,'-')+'.csv';l.click()};
"""


def main() -> None:
    items = elegir()
    cuerpo = "".join(tarjeta(i, it) for i, it in enumerate(items))
    pagina = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión de explicaciones</title><style>{ESTILO}</style></head><body>
<h1>Revisión de explicaciones: validación a mano</h1>{RUBRICA}
<div class="barra">Tu nombre: <input id="quien" placeholder="nombre"> · Completadas: <b id="avance"></b> · <button id="bajar">Descargar CSV</button></div>
{cuerpo}<script>{GUION}</script></body></html>"""
    SALIDA.write_text(pagina, encoding="utf-8")
    print(f"{len(items)} afirmaciones ({sum(i['grupo'] == 'desacuerdo' for i in items)} en desacuerdo, "
          f"{sum(i['grupo'] == 'azar' for i in items)} al azar) → {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
