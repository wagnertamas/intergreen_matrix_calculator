"""
Többcélú (MORL) Gymnasium környezet a T-csomópont jelzőlámpájához.

A környezet az MO-Gymnasium konvenciót követi: a ``step`` jutalma egy
``numpy`` **vektor** (célfüggvényenként egy elem), és van ``reward_space``
attribútuma. Skalarizáló wrapperrel (:class:`sumo_morl.morl.LinearScalarization`)
bármely egycélú RL könyvtárral (pl. Stable-Baselines3) használható.

Vezérlés (fázis alapú)
----------------------
* Akció: a következő fázis sorszáma (``Discrete(n_phases)``).
* Ha az akció = aktuális fázis -> a zöld ``decision_interval`` másodperccel
  meghosszabbodik.
* Ha más fázist választ -> lefut az átmenet (sárga / piros / piros-sárga), amit
  a közbenső idők határoznak meg, majd az új fázis ``min_green`` másodpercig
  biztosan zöld marad. Csak ezután jön a következő döntés.

Így az ágens **nem tud** szabálytalan jelzésképet előállítani, és nem kell
foglalkoznia a közbenső időkkel.

Megfigyelés (csak hurokdetektorok + a vezérlő saját állapota)
------------------------------------------------------------
Lásd :meth:`TJunctionMOEnv.observation_names`.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .detectors import DetectorReader, DetectorSnapshot, FEATURES, write_detector_file
from .network import INCOMING_EDGES
from .objectives import get_objective
from .scenario import TJunctionScenario
from .sumo_utils import find_sumo_binary, sumo_home
from .traffic import Demand

DEFAULT_OBJECTIVES = ("queue", "fairness", "switches")


def _import_traci():
    home = sumo_home()
    if home and os.path.join(home, "tools") not in sys.path:
        sys.path.append(os.path.join(home, "tools"))
    try:
        import traci  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        raise ImportError("A 'traci' csomag nem található. Telepítsd a függőségeket a csomag "
                          "gyökérkönyvtárában: pip install -e .") from e
    return traci


class TJunctionMOEnv(gym.Env):
    """Többcélú jelzőlámpa-vezérlési környezet egy T-csomópontra.

    Args:
        objectives: a célfüggvények nevei (lásd :func:`sumo_morl.list_objectives`).
            A jutalomvektor ebben a sorrendben tartalmazza őket.
        demand: forgalmi igény. ``None`` -> minden epizódban véletlen
            (:meth:`Demand.random`); megadható egy :class:`Demand` objektum vagy
            egy függvény ``rng -> Demand``.
        episode_length: az epizód hossza szimulált másodpercben.
        decision_interval: zöldhosszabbítás egy "maradok" akcióra [s].
        min_green: minimális zöldidő váltás után [s].
        max_green: ha nem ``None``, ennél hosszabb zöld után a környezet
            kikényszeríti a következő fázisra váltást [s].
        warmup: ennyi másodpercig fix idejű vezérlés fut az epizód elején, hogy
            a csomópont ne üresen induljon [s].
        gui: ``sumo-gui`` indítása (megjelenítés).
        seed: véletlenmag.
        ground_truth: ha igaz, az ``info["ground_truth"]`` alatt SUMO-ból
            kiolvasott "valós" mutatók is megjelennek (várakozási idő, CO2, ...).
            Ezeket **csak kiértékelésre** használd, a megfigyelésben és a
            jutalomban nem szerepelnek!
        scenario: egyedi forgatókönyv (haladóknak).
        sumo_args: további parancssori kapcsolók a SUMO-nak.
        record_signals: ha igaz, az ``env.signal_log`` másodpercenként rögzíti a
            jelzésképet (``[(idő, állapot), ...]``), pl. jelzésterv-ábrához.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        objectives: Sequence[str] = DEFAULT_OBJECTIVES,
        demand: Union[None, Demand, Any] = None,
        episode_length: int = 3600,
        decision_interval: int = 5,
        min_green: int = 5,
        max_green: Optional[int] = None,
        warmup: int = 0,
        gui: bool = False,
        seed: Optional[int] = None,
        ground_truth: bool = True,
        scenario: Optional[TJunctionScenario] = None,
        sumo_args: Sequence[str] = (),
        render_mode: Optional[str] = None,
        record_signals: bool = False,
    ):
        super().__init__()
        self.scenario = scenario or TJunctionScenario()
        self.plan = self.scenario.plan
        self.objective_names = tuple(objectives)
        self._objectives = [get_objective(o) for o in self.objective_names]
        self.demand_spec = demand
        self.episode_length = int(episode_length)
        self.decision_interval = int(decision_interval)
        self.min_green = int(min_green)
        self.max_green = max_green
        self.warmup = int(warmup)
        self.gui = gui or render_mode == "human"
        self.render_mode = render_mode
        self.ground_truth = ground_truth
        self.sumo_args = list(sumo_args)
        self.record_signals = record_signals
        self.signal_log: List[Tuple[int, str]] = []

        self.detectors = self.scenario.detectors
        self.approaches = list(self.scenario.approaches)
        self._reader = DetectorReader(self.detectors, self.approaches)
        self._approach_capacity = self._compute_capacity()

        n_ph = self.plan.n_phases
        self.action_space = spaces.Discrete(n_ph)
        self._obs_names = self._build_obs_names()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(self._obs_names),), dtype=np.float32)
        self.reward_space = spaces.Box(-np.inf, np.inf, shape=(len(self._objectives),), dtype=np.float32)
        self.reward_dim = len(self._objectives)

        self._workdir = tempfile.mkdtemp(prefix="sumo_morl_")
        self._det_file = write_detector_file(self.detectors, os.path.join(self._workdir, "det.add.xml"))
        self._route_file = os.path.join(self._workdir, "routes.rou.xml")
        self._label = f"sumo_morl_{uuid.uuid4().hex[:8]}"
        self._traci = _import_traci()
        self._conn = None
        self._rng = np.random.default_rng(seed)
        self._episode = 0
        self.current_demand: Optional[Demand] = None

    # ================================================================ tulajdonságok
    @property
    def n_phases(self) -> int:
        return self.plan.n_phases

    @property
    def phase_names(self) -> List[str]:
        return list(self.plan.phase_names)

    def observation_names(self) -> List[str]:
        """A megfigyelésvektor elemeinek nevei (ugyanabban a sorrendben)."""
        return list(self._obs_names)

    def _build_obs_names(self) -> List[str]:
        names = []
        for d in self.detectors:
            for f in FEATURES:
                names.append(f"{d.id}.{f}")
        for a in self.approaches:
            names.append(f"approach_{a}.vehicles")
        for p in range(self.plan.n_phases):
            names.append(f"phase_{p}")
        names.append("time_in_phase")
        return names

    def _compute_capacity(self) -> Dict[str, float]:
        cap = {a: 0.0 for a in self.approaches}
        lanes = {}
        for d in self.detectors:
            lanes.setdefault(d.lane_id, {})[d.kind] = d
        for lane in lanes.values():
            length = lane["stopline"].pos - lane["entry"].pos
            cap[lane["entry"].approach] += length / 7.0  # ~7 m / jármű sorban
        return cap

    # ================================================================ SUMO
    def _start_sumo(self, seed: int) -> None:
        binary = find_sumo_binary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-n", self.scenario.net_file, "-r", self._route_file, "-a", self._det_file,
            "--step-length", "1", "--seed", str(seed), "--no-step-log", "true",
            "--no-warnings", "true", "--time-to-teleport", "-1",
            "--waiting-time-memory", "3600", "--duration-log.disable", "true",
            "--end", str(self.episode_length + self.warmup + 3600),
        ]
        if self.gui:
            cmd += ["--start", "true", "--quit-on-end", "true", "--delay", "50"]
        cmd += self.sumo_args
        with contextlib.redirect_stdout(io.StringIO()):  # "Retrying..." üzenetek elnyelése
            self._traci.start(cmd, label=self._label)
        self._conn = self._traci.getConnection(self._label)
        self._reader.subscribe(self._conn)

    def _close_sumo(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _make_demand(self) -> Demand:
        spec = self.demand_spec
        if spec is None:
            return Demand.random(self._rng)
        if isinstance(spec, Demand):
            return spec
        if callable(spec):
            return spec(self._rng)
        raise TypeError("demand: None, Demand vagy függvény (rng -> Demand) lehet.")

    # ================================================================ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        self._close_sumo()
        self.current_demand = options.get("demand") or self._make_demand()
        sumo_seed = int(self._rng.integers(0, 2**31 - 1))
        self.current_demand.write_routes(self._route_file, self.episode_length + self.warmup, sumo_seed)
        self._start_sumo(sumo_seed)

        self._reader.reset()
        self.signal_log: List[Tuple[int, str]] = []
        self._phase = 0
        self._time_in_phase = 0
        self._sim_time = 0
        self._switches = 0
        self._gt_reset()
        self._set_state(self.plan.green_state(self._phase))

        # Bemelegítés fix idejű vezérléssel (a jutalomba nem számít bele).
        if self.warmup > 0:
            t0 = 0
            while t0 < self.warmup:
                target = (self._phase + 1) % self.n_phases if self._time_in_phase >= 20 else self._phase
                if target != self._phase:
                    t0 += self._run_transition(target)
                else:
                    t0 += self._run_green(1)
            snap = self._reader.snapshot()
            self._gt_reset()
        else:
            snap = self._empty_snapshot()
        self._episode_time0 = self._sim_time
        self._last_snapshot = snap
        obs = self._observation(snap)
        info = self._info(snap, np.zeros(self.reward_dim, dtype=np.float32), False, 0)
        return obs, info

    def step(self, action: int):
        if self._conn is None:
            raise RuntimeError("Előbb hívd meg a reset()-et!")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Érvénytelen akció: {action}")
        forced = (self.max_green is not None and action == self._phase
                  and self._time_in_phase >= self.max_green)
        if forced:
            action = (self._phase + 1) % self.n_phases

        changed = action != self._phase
        t_start = self._sim_time
        if changed:
            self._run_transition(action)
            self._run_green(self.min_green)
            self._switches += 1
        else:
            self._run_green(self.decision_interval)

        snap = self._reader.snapshot()
        self._last_snapshot = snap
        ctx = {"phase_changed": changed, "phase": self._phase, "duration": snap.duration,
               "time": self._sim_time, "forced": forced}
        reward = np.array([o(snap, ctx) for o in self._objectives], dtype=np.float32)
        truncated = (self._sim_time - self._episode_time0) >= self.episode_length
        obs = self._observation(snap)
        info = self._info(snap, reward, changed, self._sim_time - t_start, forced=forced)
        return obs, reward, False, truncated, info

    def close(self):
        self._close_sumo()
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def render(self):  # a sumo-gui maga rajzol
        return None

    # ================================================================ belső léptetés
    def _set_state(self, state: str) -> None:
        self._conn.trafficlight.setRedYellowGreenState(self.plan.tls_id, state)
        self._state = state

    def _sim_step(self) -> None:
        self._conn.simulationStep()
        self._sim_time += 1
        if self.record_signals:
            self.signal_log.append((self._sim_time, self._state))
        self._reader.sample(self._conn)
        if self.ground_truth:
            self._gt_sample()

    def _run_green(self, seconds: int) -> int:
        self._set_state(self.plan.green_state(self._phase))
        for _ in range(seconds):
            self._sim_step()
            self._time_in_phase += 1
        return seconds

    def _run_transition(self, target: int) -> int:
        total = 0
        for state, dur in self.plan.transition(self._phase, target):
            self._set_state(state)
            for _ in range(int(dur)):
                self._sim_step()
            total += int(dur)
        self._phase = target
        self._time_in_phase = 0
        return total

    # ================================================================ megfigyelés
    def _empty_snapshot(self) -> DetectorSnapshot:
        n = len(self.detectors)
        return DetectorSnapshot(0.0, self.detectors, np.zeros(n), np.zeros(n),
                                np.array([d.speed_limit for d in self.detectors]), np.zeros(n),
                                {a: 0.0 for a in self.approaches}, {a: 0.0 for a in self.approaches},
                                np.zeros(n))

    def _observation(self, snap: DetectorSnapshot) -> np.ndarray:
        dur = max(snap.duration, 1.0)
        feats = []
        for i, d in enumerate(self.detectors):
            flow = snap.count[i] / dur * 3600.0 / 1800.0       # ~ telítési forgalomhoz viszonyítva
            feats += [min(flow, 1.0), snap.occupancy[i], min(snap.speed[i] / d.speed_limit, 1.0),
                      min(snap.gap[i] / 60.0, 1.0)]
        for a in self.approaches:
            feats.append(min(snap.approach_vehicles[a] / max(self._approach_capacity[a], 1.0), 1.0))
        onehot = [0.0] * self.n_phases
        onehot[self._phase] = 1.0
        feats += onehot
        feats.append(min(self._time_in_phase / 120.0, 1.0))
        return np.clip(np.asarray(feats, dtype=np.float32), 0.0, 1.0)

    def _info(self, snap, reward, changed, elapsed, forced=False) -> Dict[str, Any]:
        info = {
            "time": self._sim_time - getattr(self, "_episode_time0", 0),
            "elapsed": elapsed,
            "phase": self._phase,
            "phase_name": self.phase_names[self._phase] if self.phase_names else str(self._phase),
            "time_in_phase": self._time_in_phase,
            "phase_changed": changed,
            "forced_switch": forced,
            "switches": self._switches,
            "objectives": {n: float(r) for n, r in zip(self.objective_names, reward)},
            "detectors": snap,
            "demand": self.current_demand,
        }
        if self.ground_truth:
            info["ground_truth"] = self._gt_summary()
        return info

    # ================================================================ "valós" mutatók
    def _gt_reset(self) -> None:
        self._gt = {"halting_vehicle_seconds": 0.0, "co2_g": 0.0, "arrived": 0, "departed": 0,
                    "travel_time_sum": 0.0}
        self._depart_time: Dict[str, int] = {}

    def _gt_sample(self) -> None:
        c = self._conn
        halting = 0
        co2 = 0.0
        for e in list(INCOMING_EDGES.values()):
            halting += c.edge.getLastStepHaltingNumber(e)
            co2 += c.edge.getCO2Emission(e)
        self._gt["halting_vehicle_seconds"] += halting
        self._gt["co2_g"] += co2 / 1000.0
        for v in c.simulation.getDepartedIDList():
            self._depart_time[v] = self._sim_time
        self._gt["departed"] += c.simulation.getDepartedNumber()
        for v in c.simulation.getArrivedIDList():
            t0 = self._depart_time.pop(v, None)
            if t0 is not None:
                self._gt["travel_time_sum"] += self._sim_time - t0
                self._gt["arrived"] += 1
        self._gt["waiting_to_enter"] = len(c.simulation.getPendingVehicles())

    def _gt_summary(self) -> Dict[str, float]:
        g = dict(self._gt)
        g["mean_travel_time"] = g["travel_time_sum"] / g["arrived"] if g["arrived"] else float("nan")
        g["mean_delay_per_vehicle"] = (g["halting_vehicle_seconds"] / g["departed"]
                                       if g["departed"] else float("nan"))
        return g
