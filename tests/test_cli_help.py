from __future__ import annotations

import contextlib
import importlib
import io
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

pytestmark = pytest.mark.nivel1


def _entrypoints():
    pyproject = tomllib.loads(
        (RAÍZ_DEL_REPO / "pyproject.toml").read_text(encoding="utf-8")
    )
    declarados = pyproject["project"]["scripts"]
    return sorted(
        (nombre, destino.split(":")[0]) for nombre, destino in declarados.items()
    )


@pytest.mark.parametrize("nombre,modulo", _entrypoints())
def test_el_help_de_cada_entrypoint_no_levanta(nombre, modulo):
    cli = importlib.import_module(modulo)
    salida = io.StringIO()
    with pytest.raises(SystemExit) as e:
        with contextlib.redirect_stdout(salida):
            cli.main(["--help"])
    assert e.value.code in (0, None)
    ayuda = salida.getvalue()
    assert nombre in ayuda
