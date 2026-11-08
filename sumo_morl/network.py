"""
A T-csomópont hálózatának (``.net.xml``) előállítása.

A hallgatóknak ehhez nem kell hozzányúlniuk: a csomag tartalmaz egy kész
hálózatot (``sumo_morl/data/t_junction.net.xml``). Ez a modul csak azért van
itt, hogy látható legyen, *hogyan* készült, és hogy szükség esetén (pl. más
ágszakasz-hossz vagy sebesség) újra lehessen generálni.

A csomópont felépítése (jobbkezes forgalom, észak felfelé)::

                  (nincs északi ág)
        W ─────────────── C ─────────────── E
     (nyugat)             │              (kelet)
                          │
                          S (dél)

* **Nyugati ág (W_C)**: 1 sáv, egyenesen keletre + jobbra délre (közös sáv).
* **Keleti ág (E_C)**: 2 sáv – ez a "jobb oldali" ág, amely ketté van véve:
    - 0. sáv (jobb oldali): egyenesen nyugatra,
    - 1. sáv (bal oldali): balra kanyarodó, délre.
* **Déli ág (S_C)**: 1 sáv, balra nyugatra + jobbra keletre (közös sáv).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

from .sumo_utils import find_sumo_binary

# Ágak és mozgások nevei, a csomag többi része ezekre hivatkozik.
APPROACHES = ("W", "E", "S")
INCOMING_EDGES = {"W": "W_C", "E": "E_C", "S": "S_C"}
OUTGOING_EDGES = {"W": "C_W", "E": "C_E", "S": "C_S"}
TLS_ID = "C"

#: (honnan, hova) -> (bejövő él, sáv index)
MOVEMENTS = {
    ("W", "E"): ("W_C", 0),
    ("W", "S"): ("W_C", 0),
    ("E", "W"): ("E_C", 0),
    ("E", "S"): ("E_C", 1),
    ("S", "W"): ("S_C", 0),
    ("S", "E"): ("S_C", 0),
}


@dataclass
class NetworkSpec:
    """A T-csomópont geometriai paraméterei."""

    approach_length: float = 250.0   # ágak hossza [m]
    speed: float = 13.89             # megengedett sebesség [m/s] (50 km/h)


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_network(out_file: str, spec: NetworkSpec | None = None) -> str:
    """Legenerálja a T-csomópont ``.net.xml`` fájlját ``netconvert``-tel.

    Args:
        out_file: a kimeneti ``.net.xml`` útvonala.
        spec: geometriai paraméterek (alapértelmezés: :class:`NetworkSpec`).

    Returns:
        A létrehozott fájl útvonala.
    """
    spec = spec or NetworkSpec()
    L, v = spec.approach_length, spec.speed

    nodes = f"""<nodes>
    <node id="C" x="0" y="0" type="traffic_light"/>
    <node id="W" x="{-L}" y="0" type="priority"/>
    <node id="E" x="{L}" y="0" type="priority"/>
    <node id="S" x="0" y="{-L}" type="priority"/>
</nodes>
"""
    edges = f"""<edges>
    <edge id="W_C" from="W" to="C" numLanes="1" speed="{v}" priority="2"/>
    <edge id="C_W" from="C" to="W" numLanes="1" speed="{v}" priority="2"/>
    <edge id="E_C" from="E" to="C" numLanes="2" speed="{v}" priority="2"/>
    <edge id="C_E" from="C" to="E" numLanes="1" speed="{v}" priority="2"/>
    <edge id="S_C" from="S" to="C" numLanes="1" speed="{v}" priority="1"/>
    <edge id="C_S" from="C" to="S" numLanes="1" speed="{v}" priority="1"/>
</edges>
"""
    cons = """<connections>
    <connection from="W_C" to="C_E" fromLane="0" toLane="0"/>
    <connection from="W_C" to="C_S" fromLane="0" toLane="0"/>
    <connection from="E_C" to="C_W" fromLane="0" toLane="0"/>
    <connection from="E_C" to="C_S" fromLane="1" toLane="0"/>
    <connection from="S_C" to="C_W" fromLane="0" toLane="0"/>
    <connection from="S_C" to="C_E" fromLane="0" toLane="0"/>
</connections>
"""
    out_file = os.path.abspath(out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        n, e, c = (os.path.join(tmp, x) for x in ("t.nod.xml", "t.edg.xml", "t.con.xml"))
        _write(n, nodes)
        _write(e, edges)
        _write(c, cons)
        cmd = [
            find_sumo_binary("netconvert"),
            "--node-files", n, "--edge-files", e, "--connection-files", c,
            "--output-file", out_file,
            "--no-turnarounds", "true",
            "--tls.default-type", "static",
            "--junctions.corner-detail", "8",
            "--no-warnings", "true",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return out_file


if __name__ == "__main__":  # python -m sumo_morl.network
    here = os.path.join(os.path.dirname(__file__), "data", "t_junction.net.xml")
    print(build_network(here))
