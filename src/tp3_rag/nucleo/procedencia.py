from __future__ import annotations

import subprocess
from pathlib import Path

from tp3_rag.nucleo.configuracion import RAÍZ_DEL_REPO

def commit_actual() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAÍZ_DEL_REPO,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None

def ruta_publicable(ruta: str | Path | None) -> str | None:
    if ruta is None:
        return None
    try:
        return str(Path(ruta).resolve().relative_to(RAÍZ_DEL_REPO))
    except ValueError:
        return Path(ruta).name
