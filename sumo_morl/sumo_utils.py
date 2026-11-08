"""SUMO-val kapcsolatos apró segédfüggvények (bináris keresés, indítás)."""
from __future__ import annotations

import os
import shutil


def sumo_home() -> str | None:
    """Visszaadja a SUMO telepítési könyvtárát (``SUMO_HOME``), ha megtalálható."""
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    try:  # pip install eclipse-sumo
        import sumo  # type: ignore

        return sumo.SUMO_HOME
    except Exception:
        return None


def find_sumo_binary(name: str = "sumo") -> str:
    """Megkeresi a ``sumo`` / ``sumo-gui`` / ``netconvert`` futtatható fájlt."""
    home = sumo_home()
    if home:
        for cand in (os.path.join(home, "bin", name), os.path.join(home, "bin", name + ".exe")):
            if os.path.isfile(cand):
                return cand
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Nem található a(z) '{name}' program. Telepítsd a SUMO-t "
        "(pl. `pip install eclipse-sumo`), vagy állítsd be a SUMO_HOME változót."
    )
