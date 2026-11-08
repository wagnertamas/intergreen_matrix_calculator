"""
Forgalmi igény (járműáramok) megadása és SUMO route fájl generálása.

Az igényt irányonként (honnan → hova) adjuk meg jármű/óra egységben. A
:meth:`Demand.random` minden epizódhoz véletlen, de reális igényt sorsol, így
az ágens változatos forgalmi helyzeteken tanul.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .network import INCOMING_EDGES, MOVEMENTS, OUTGOING_EDGES

Movement = Tuple[str, str]


@dataclass
class Demand:
    """Forgalmi igény.

    Attributes:
        flows: ``{(honnan, hova): jármű/óra}``, pl. ``{("W", "E"): 500}``.
        profile: időbeli szorzók (egyenlő hosszú szakaszokra osztva az
            epizódot), pl. ``(0.5, 1.0, 1.5, 1.0)`` egy csúcsórás hullámhoz.
            ``(1.0,)`` = állandó igény.
    """

    flows: Dict[Movement, float]
    profile: Sequence[float] = (1.0,)

    def __post_init__(self):
        for m in self.flows:
            if m not in MOVEMENTS:
                raise ValueError(f"Ismeretlen mozgás: {m}. Lehetséges: {list(MOVEMENTS)}")

    @property
    def total(self) -> float:
        return float(sum(self.flows.values()))

    def approach_volume(self, approach: str) -> float:
        return float(sum(v for (a, _), v in self.flows.items() if a == approach))

    @classmethod
    def constant(cls, main: float = 500, side: float = 200, left_share: float = 0.3,
                 right_share: float = 0.2) -> "Demand":
        """Egyszerű, állandó igény a főirányon (ny–k) és a mellékirányon (dél).

        Args:
            main: a főirány egy irányának forgalma [jármű/óra].
            side: a déli mellékág forgalma [jármű/óra].
            left_share: a keletről balra (délre) kanyarodók aránya.
            right_share: a nyugatról jobbra (délre) kanyarodók aránya.
        """
        return cls({
            ("W", "E"): main * (1 - right_share), ("W", "S"): main * right_share,
            ("E", "W"): main * (1 - left_share), ("E", "S"): main * left_share,
            ("S", "W"): side * 0.5, ("S", "E"): side * 0.5,
        })

    @classmethod
    def random(cls, rng: np.random.Generator, main_range=(200, 700), side_range=(80, 400),
               turn_range=(0.1, 0.45), profile: Optional[Sequence[float]] = None) -> "Demand":
        """Véletlen igény sorsolása (epizódonként más-más)."""
        mw, me = rng.uniform(*main_range, size=2)
        side = rng.uniform(*side_range)
        rs, ls, ss = rng.uniform(*turn_range, size=3)
        flows = {
            ("W", "E"): mw * (1 - rs), ("W", "S"): mw * rs,
            ("E", "W"): me * (1 - ls), ("E", "S"): me * ls,
            ("S", "W"): side * (1 - ss), ("S", "E"): side * ss,
        }
        return cls(flows, tuple(profile) if profile is not None else (1.0,))

    def scaled(self, factor: float) -> "Demand":
        return Demand({m: v * factor for m, v in self.flows.items()}, self.profile)

    def write_routes(self, out_file: str, duration: int, seed: int = 0) -> str:
        """SUMO ``.rou.xml`` írása ``<flow>`` elemekkel (Poisson-szerű érkezés)."""
        root = ET.Element("routes")
        ET.SubElement(root, "vType", id="car", vClass="passenger", length="4.5", minGap="2.5",
                      accel="2.6", decel="4.5", sigma="0.5", maxSpeed="16.67")
        for (a, b) in self.flows:
            ET.SubElement(root, "route", id=f"r_{a}{b}", edges=f"{INCOMING_EDGES[a]} {OUTGOING_EDGES[b]}")
        n_seg = len(self.profile)
        seg = duration / n_seg
        elems = []
        for k, mult in enumerate(self.profile):
            for (a, b), vph in self.flows.items():
                rate = vph * mult
                if rate <= 0:
                    continue
                elems.append((k * seg, dict(
                    id=f"f_{a}{b}_{k}", type="car", route=f"r_{a}{b}", begin=f"{k * seg:.0f}",
                    end=f"{(k + 1) * seg:.0f}", period=f"exp({rate / 3600.0:.6f})",
                    departLane="best", departSpeed="max")))
        for _, attrs in sorted(elems, key=lambda e: e[0]):
            ET.SubElement(root, "flow", **attrs)
        ET.indent(root)
        ET.ElementTree(root).write(out_file, encoding="utf-8", xml_declaration=True)
        return out_file

    def __str__(self) -> str:
        parts = ", ".join(f"{a}->{b}: {v:.0f}" for (a, b), v in self.flows.items())
        return f"Demand({parts} jármű/óra, profil={tuple(self.profile)})"
