"""
A T-csomópont forgatókönyv: hálózat + jelzésterv + detektorok egy helyen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .detectors import DetectorInfo, detector_layout
from .intergreen import SignalPlan, build_signal_plan
from .network import APPROACHES, INCOMING_EDGES, TLS_ID

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_NET = os.path.join(DATA_DIR, "t_junction.net.xml")

#: Olvasható nevek a jelzőcsoportokhoz (bejövő sáv -> név).
GROUP_NAMES = {
    "W_C_0": "Ny_egyenes+jobb",
    "E_C_0": "K_egyenes",
    "E_C_1": "K_balra",
    "S_C_0": "D_bal+jobb",
}

#: A fázisok (jelzőcsoportok sávjai szerint) és neveik. Ezek pontosan a
#: konfliktusmentes klikkek, amelyeket a közbensőidő-mátrixból kapunk; itt csak
#: rögzítjük a sorrendjüket, hogy a fázisok sorszáma stabil legyen.
PHASE_LANES = [
    ["E_C_0", "W_C_0"],
    ["E_C_0", "E_C_1"],
    ["S_C_0"],
]
PHASE_NAMES = [
    "Főirány: nyugat-kelet egyenes (+ nyugatról jobbra)",
    "Keleti ág: egyenes + balra kanyarodó",
    "Mellékirány: déli ág (balra + jobbra)",
]


@dataclass
class TJunctionScenario:
    """Mindent tartalmaz, ami a szimulációhoz kell.

    Attributes:
        net_file: hálózat fájl.
        plan: a jelzésterv (fázisok, közbenső idők, átmenetek).
        detectors: a hurokdetektorok listája.
    """

    net_file: str = DEFAULT_NET
    yellow: int = 3
    red_yellow: int = 2
    plan: SignalPlan = field(init=False)
    detectors: List[DetectorInfo] = field(init=False)
    tls_id: str = TLS_ID
    approaches: tuple = APPROACHES

    def __post_init__(self):
        auto = build_signal_plan(self.net_file, self.tls_id, yellow=self.yellow,
                                 red_yellow=self.red_yellow, group_names=GROUP_NAMES)
        lane_to_group = {g.lane_id: i for i, g in enumerate(auto.groups)}
        phases = [sorted(lane_to_group[l] for l in lanes) for lanes in PHASE_LANES]
        if sorted(map(tuple, phases)) != sorted(map(tuple, auto.phases)):
            raise RuntimeError(f"A rögzített fázisok ({phases}) eltérnek a klikkektől ({auto.phases}).")
        self.plan = build_signal_plan(self.net_file, self.tls_id, phases=phases, yellow=self.yellow,
                                      red_yellow=self.red_yellow, group_names=GROUP_NAMES,
                                      phase_names=PHASE_NAMES)
        self.detectors = detector_layout(self.net_file, INCOMING_EDGES)

    def describe(self) -> str:
        out = [self.plan.describe(), "", "Hurokdetektorok:"]
        for d in self.detectors:
            out.append(f"  {d.id:<18} ág={d.approach} sáv={d.lane_id:<6} típus={d.kind:<8} poz={d.pos:6.1f} m")
        return "\n".join(out)
