from __future__ import annotations

import io

import pytest
from PIL import Image

from tp3_rag.modulo_a.preparador_de_imagen import PreparadorDeImagen
from tp3_rag.nucleo.identidad import hash_de_bytes

pytestmark = pytest.mark.nivel1

class _ClienteDeBytes:
    def __init__(self, por_url):
        self.por_url = por_url

    def get(self, url):
        from tp3_rag.puertos.cliente_http import RespuestaHTTP

        valor = self.por_url[url]
        if isinstance(valor, Exception):
            raise valor
        return RespuestaHTTP(url, 200, valor, {}, True)

def _png(modo: str, tamaño=(4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new(modo, tamaño).save(buf, format="PNG")
    return buf.getvalue()

@pytest.mark.parametrize(
    "modo,gris,alfa",
    [
        ("RGB", False, False),
        ("L", True, False),
        ("LA", True, True),
        ("RGBA", False, True),
        ("P", False, True),
        ("1", True, False),
    ],
)
def test_a9c_detecta_grises_y_alfa_y_convierte_a_rgb(tmp_path, modo, gris, alfa):
    url = f"https://x/{modo}.png"
    prep = PreparadorDeImagen(_ClienteDeBytes({url: _png(modo)}), tmp_path)
    r = prep.preparar(f"aic:{modo}", url)
    assert r.ok
    assert r.modo_original == modo
    assert r.es_gris is gris
    assert r.tiene_alfa is alfa
    assert r.convertida_a_rgb is True

def test_el_hash_es_de_los_bytes_que_sirvio_el_museo(tmp_path):
    datos = _png("L")
    url = "https://x/g.png"
    r = PreparadorDeImagen(_ClienteDeBytes({url: datos}), tmp_path).preparar("aic:1", url)
    assert r.hash_bytes == hash_de_bytes(datos)
    assert (tmp_path / "aic_1.jpg").read_bytes() == datos

def test_el_path_lleva_artwork_id_y_no_el_hash(tmp_path):
    url = "https://x/a.png"
    r = PreparadorDeImagen(_ClienteDeBytes({url: _png("RGB")}), tmp_path).preparar("cma:124078", url)
    assert r.ruta.endswith("cma_124078.jpg")
    assert "sha256" not in r.ruta

def test_sin_url_de_imagen_se_descarta_y_se_cuenta(tmp_path):
    r = PreparadorDeImagen(_ClienteDeBytes({}), tmp_path).preparar("aic:1", None)
    assert not r.ok and r.motivo == "sin_url_de_imagen"
    assert r.hash_bytes is None

def test_un_404_se_descarta_y_se_cuenta(tmp_path):
    from tp3_rag.puertos.cliente_http import RecursoNoEncontrado

    url = "https://x/no.jpg"
    prep = PreparadorDeImagen(_ClienteDeBytes({url: RecursoNoEncontrado("404")}), tmp_path)
    r = prep.preparar("aic:1", url)
    assert not r.ok and r.motivo == "http_404"

def test_un_archivo_que_no_es_imagen_cuenta_como_falla_de_formato(tmp_path):
    url = "https://x/roto.jpg"
    prep = PreparadorDeImagen(_ClienteDeBytes({url: b"no soy un jpeg"}), tmp_path)
    r = prep.preparar("aic:1", url)
    assert not r.ok and r.motivo.startswith("falla_de_formato")

def test_un_cuerpo_vacio_no_pasa_como_imagen_valida(tmp_path):
    url = "https://x/vacio.jpg"
    r = PreparadorDeImagen(_ClienteDeBytes({url: b""}), tmp_path).preparar("aic:1", url)
    assert not r.ok and r.motivo == "cuerpo_vacio"

def test_a9c_agrega_los_conteos_que_el_criterio_pide(tmp_path):
    urls = {f"https://x/{m}.png": _png(m) for m in ("RGB", "L", "RGBA")}
    urls["https://x/roto.jpg"] = b"basura"
    prep = PreparadorDeImagen(_ClienteDeBytes(urls), tmp_path)
    preparadas = [
        prep.preparar("aic:1", "https://x/RGB.png"),
        prep.preparar("aic:2", "https://x/L.png"),
        prep.preparar("aic:3", "https://x/RGBA.png"),
        prep.preparar("aic:4", "https://x/roto.jpg"),
        prep.preparar("aic:5", None),
    ]
    a9c = PreparadorDeImagen.a9c(preparadas)
    assert a9c["n_con_imagen_preparada"] == 3
    assert a9c["grises"] == 1
    assert a9c["alfa_o_paleta"] == 1
    assert a9c["convertidas_a_rgb"] == 3
    assert a9c["fallas_de_conversion_de_formato"] == 1
    assert a9c["motivos_de_descarte"]["sin_url_de_imagen"] == 1
    assert a9c["modos_originales"] == {"L": 1, "RGB": 1, "RGBA": 1}

def test_a9c_declara_que_la_mitad_de_embedding_no_es_medible_en_m1(tmp_path):
    a9c = PreparadorDeImagen.a9c([])
    assert a9c["fallas_de_embedding_por_formato"] is None
    assert "EmbebedorDeImágenes no corre" in a9c["nota"]

def test_las_imagenes_reales_del_fixture_se_preparan_sin_red(cliente_sin_red, cfg, tmp_path):
    from tp3_rag.dominio.urls import url_de_imagen
    from tp3_rag.puertos.cliente_http import CacheMiss
    from tp3_rag.puertos.museos.cma import AdaptadorCMA

    adaptador = AdaptadorCMA(cliente_sin_red, cfg)
    ids = adaptador.listar(list(cfg["ingesta.tipos"]), "libre", 8)
    prep = PreparadorDeImagen(cliente_sin_red, tmp_path)

    preparadas = 0
    for id_del_museo in ids:
        url = url_de_imagen("cma", adaptador.traer(id_del_museo), cfg["ingesta.urls"])
        try:
            r = prep.preparar(f"cma:{id_del_museo}", url)
        except CacheMiss:
            continue
        if r.ok:
            preparadas += 1
            assert r.hash_bytes.startswith("sha256:")
            assert r.convertida_a_rgb and r.ancho and r.alto
    assert preparadas >= 1, "el fixture tiene que traer al menos una imagen de CMA"
