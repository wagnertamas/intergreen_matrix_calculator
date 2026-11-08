"""
Célfüggvények (a vektor-jutalom komponensei).

Minden célfüggvény egy függvény, amely egy vezérlési lépés hurokdetektor-adataiból
(:class:`~sumo_morl.detectors.DetectorSnapshot`) és néhány, a vezérlő által
amúgy is ismert adatból (pl. volt-e fázisváltás) **egyetlen számot** ad vissza.
A konvenció: **nagyobb = jobb** (a költségeket negatív előjellel adjuk vissza).

A jutalmak a lépés másodperceire összegzett mennyiségek, így a lépés hosszától
függetlenül összeadhatók (az epizód összjutalma = időintegrál). A ``scale``
tényező csak arra szolgál, hogy a tipikus értékek ~1 nagyságrendűek legyenek.

Saját célfüggvény hozzáadása::

    from sumo_morl import register_objective

    @register_objective("stopline_occupancy", scale=0.1,
                        doc="A stopvonali hurkok foglaltsága (álló járművek)")
    def stopline_occupancy(snap, ctx):
        mask = snap.select(kind="stopline")
        return -float(snap.stopline_occupied_seconds[mask].sum())

    env = TJunctionMOEnv(objectives=("queue", "stopline_occupancy"))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .detectors import DetectorSnapshot

ObjectiveFn = Callable[[DetectorSnapshot, Dict], float]


@dataclass
class Objective:
    name: str
    fn: ObjectiveFn
    scale: float
    doc: str

    def __call__(self, snap: DetectorSnapshot, ctx: Dict) -> float:
        return float(self.fn(snap, ctx)) * self.scale


OBJECTIVES: Dict[str, Objective] = {}


def register_objective(name: str, scale: float = 1.0, doc: str = ""):
    """Dekorátor új célfüggvény regisztrálásához (lásd a modul leírását)."""

    def deco(fn: ObjectiveFn) -> ObjectiveFn:
        OBJECTIVES[name] = Objective(name, fn, scale, doc or (fn.__doc__ or "").strip())
        return fn

    return deco


def get_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise KeyError(f"Ismeretlen célfüggvény: '{name}'. Elérhetők: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]


def list_objectives() -> str:
    """Az elérhető célfüggvények listája rövid leírással."""
    return "\n".join(f"  {o.name:<12} (skála {o.scale:g}): {o.doc}" for o in OBJECTIVES.values())


# ---------------------------------------------------------------- beépítettek
@register_objective("queue", scale=0.01,
                    doc="-(becsült járműszám az ágakon, másodpercekre összegezve) "
                        "[jármű*s]; a teljes késleltetés közelítése (min.)")
def _queue(snap: DetectorSnapshot, ctx: Dict) -> float:
    return -sum(snap.approach_vehicle_seconds.values())


@register_objective("throughput", scale=0.1,
                    doc="A stopvonali hurkokon áthaladt járművek száma [jármű] (max.)")
def _throughput(snap: DetectorSnapshot, ctx: Dict) -> float:
    return float(snap.count[snap.select(kind="stopline")].sum())


@register_objective("fairness", scale=0.02,
                    doc="-(a legrosszabb ág becsült járműszáma, másodpercekre összegezve) "
                        "[jármű*s]; méltányosság az ágak között (min.)")
def _fairness(snap: DetectorSnapshot, ctx: Dict) -> float:
    return -max(snap.approach_vehicle_seconds.values()) if snap.approach_vehicle_seconds else 0.0


@register_objective("switches", scale=1.0,
                    doc="-1 minden fázisváltásnál; stabil, kevés váltásos vezérlés (min.)")
def _switches(snap: DetectorSnapshot, ctx: Dict) -> float:
    return -1.0 if ctx.get("phase_changed") else 0.0


@register_objective("stopline_occupancy", scale=0.1,
                    doc="-(a stopvonali hurkok foglalt másodpercei) [s]; a stopvonalnál "
                        "álló/lassú járművek közelítése (min.)")
def _stopline_occ(snap: DetectorSnapshot, ctx: Dict) -> float:
    return -float(snap.stopline_occupied_seconds[snap.select(kind="stopline")].sum())
