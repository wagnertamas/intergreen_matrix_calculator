"""
Egyszerű, nem tanuló vezérlők összehasonlítási alapnak (baseline).

Mindegyik csak a hurokdetektorok adatait és a vezérlő saját állapotát
használja (``info``), ahogy egy valódi forgalomirányító berendezés is.

Használat::

    policy = FixedTimePolicy(env, green_times=[30, 10, 15])
    obs, info = env.reset()
    policy.reset()
    action = policy(obs, info)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class Policy:
    """Alap osztály: ``policy(obs, info) -> akció``."""

    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

    def reset(self) -> None:
        pass

    def __call__(self, obs, info) -> int:  # pragma: no cover - absztrakt
        raise NotImplementedError


class RandomPolicy(Policy):
    """Véletlen fázisválasztás."""

    def __init__(self, env, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, info) -> int:
        return int(self.rng.integers(self.env.n_phases))


class FixedTimePolicy(Policy):
    """Fix idejű (ciklikus) vezérlés: minden fázis adott ideig zöld.

    Args:
        green_times: zöldidők fázisonként [s]. Mivel a döntések
            ``decision_interval`` lépésekben jönnek, a tényleges zöldidő ennek
            többszörösére kerekedik fel.
    """

    def __init__(self, env, green_times: Sequence[float] = (30, 15, 15)):
        super().__init__(env)
        if len(green_times) != self.env.n_phases:
            raise ValueError(f"{self.env.n_phases} zöldidő kell (fázisonként egy).")
        self.green_times = list(green_times)

    def __call__(self, obs, info) -> int:
        p = info["phase"]
        if info["time_in_phase"] >= self.green_times[p]:
            return (p + 1) % self.env.n_phases
        return p


class ActuatedPolicy(Policy):
    """Járműkövető (forgalomfüggő) vezérlés időrés-figyeléssel.

    A zöld addig hosszabbodik, amíg az aktuális fázis sávjainak előjelző
    (``advance``) hurkain ``gap`` másodpercen belül volt érzékelés, de legfeljebb
    ``max_green`` másodpercig. Utána a következő olyan fázisra vált, amelynek a
    sávjaiban van igény (stopvonali vagy előjelző hurok foglalt / számlált).
    """

    def __init__(self, env, max_green: float = 45, gap: float = 3.0):
        super().__init__(env)
        self.max_green = max_green
        self.gap = gap

    def _lane_demand(self, snap, lane: str) -> bool:
        m = snap.select(lane_id=lane) & (snap.select(kind="stopline") | snap.select(kind="advance"))
        return bool((snap.occupancy[m] > 0).any() or (snap.count[m] > 0).any()
                    or (snap.gap[m] < self.gap).any())

    def __call__(self, obs, info) -> int:
        snap = info["detectors"]
        p = info["phase"]
        n = self.env.n_phases
        lanes = self.env.plan.phase_lanes(p)
        adv = np.zeros(len(snap.detectors), dtype=bool)
        for l in lanes:
            adv |= snap.select(lane_id=l, kind="advance")
        still_coming = bool((snap.gap[adv] < self.gap).any())
        if still_coming and info["time_in_phase"] < self.max_green:
            return p
        for k in range(1, n + 1):
            q = (p + k) % n
            if q == p:
                return p
            if any(self._lane_demand(snap, l) for l in self.env.plan.phase_lanes(q)):
                return q
        return p


class LongestQueuePolicy(Policy):
    """Mohó vezérlő: a legnagyobb becsült igényű fázist választja.

    Az igény a fázis sávjainak stopvonali + előjelző hurok foglaltsága, valamint
    a sávok ágainak bemenet–kimenet járműszám-becslése alapján számolódik.
    """

    def __init__(self, env, min_hold: float = 10.0):
        super().__init__(env)
        self.min_hold = min_hold

    def phase_scores(self, snap) -> np.ndarray:
        scores = np.zeros(self.env.n_phases)
        for p in range(self.env.n_phases):
            for l in self.env.plan.phase_lanes(p):
                m = snap.select(lane_id=l) & ~snap.select(kind="entry")
                scores[p] += snap.occupancy[m].sum()
                app = next(d.approach for d in snap.detectors if d.lane_id == l)
                n_lanes = len({d.lane_id for d in snap.detectors if d.approach == app})
                scores[p] += 0.1 * snap.approach_vehicles[app] / n_lanes
        return scores

    def __call__(self, obs, info) -> int:
        if info["time_in_phase"] < self.min_hold:
            return info["phase"]
        return int(np.argmax(self.phase_scores(info["detectors"])))
