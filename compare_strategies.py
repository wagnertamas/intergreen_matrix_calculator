#!/usr/bin/env python3
"""
Stratégia-összehasonlító — R1C1_C kereszteződés

Összehasonlítja egymással:
  • Sima (fixed) lámpaprogram
  • SUMO actuated vezérlés
  • Minden betöltött neurális háló (SB3 DQN/QRDQN .zip)

Tesztelési protokoll:
  • Csak R1C1_C kereszteződés
  • FLOW_LEVELS fix forgalmi szint (csak TARGET_JID), EPISODES_PER_LEVEL epizód szintenként
  • Egységes route fájl → minden stratégia pontosan ugyanazt a forgalmat látja

Kimenet:
  eval_comparison/
    ├── results.csv            ← nyers metrikák minden futáshoz
    ├── comparison_bars.png    ← átlagok + szórás sávdiagramon
    └── comparison_box.png     ← flow-szintenkénti boxplot

Használat:
    python compare_strategies.py
    python compare_strategies.py --plot-only     # meglévő results.csv újrarajzolása
    python compare_strategies.py --models-dir /saját/modell/könyvtár
    python compare_strategies.py --add-zip /path/to/model.zip --add-zip /path2/model2.zip
"""

import os, sys, json, random, argparse, time, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURÁCIÓ
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
NET_FILE     = os.path.join(DATA_DIR, "mega_catalogue_v2.net.xml")
LOGIC_FILE   = os.path.join(SCRIPT_DIR, "traffic_lights.json")
DETECTOR_FILE= os.path.join(DATA_DIR, "detectors.add.xml")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "eval_comparison")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models", "test")

TARGET_JID   = "R1C1_C"           # Tesztelt kereszteződés
FLOW_LEVELS  = [150,300,450,600,750,900]    # Forgalmi szintek [jármű/óra/sáv] — csak TARGET_JID
EPISODES_PER_LEVEL = 2            # Epizódok / forgalmi szint (statisztika)
DURATION     = 1200               # Szimulációs idő [sec]
WARMUP       = 120                # Bemelegítő lépések [sec]
DELTA_TIME   = 5                  # Lépésméret [sec]
MIN_GREEN    = 5                  # Minimum zöld idő [sec]

# libsumo / traci: macOS-en traci a biztonságos
USE_LIBSUMO  = False              # True = libsumo (gyorsabb, Linux szerveren)

# ─────────────────────────────────────────────────────────────────────────────
# TRACI INICIALIZÁLÁS
# ─────────────────────────────────────────────────────────────────────────────
_env_override = os.environ.get('USE_LIBSUMO')
_use_libsumo  = (int(_env_override) == 1) if _env_override is not None else USE_LIBSUMO

traci = None
sumolib = None

def _import_sumo():
    global traci, sumolib
    if traci is not None:
        return
    if _use_libsumo:
        import libsumo as _traci
    else:
        import traci as _traci
    import sumolib as _sumolib
    traci  = _traci
    sumolib = _sumolib

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE GENERÁLÁS
# ─────────────────────────────────────────────────────────────────────────────
def generate_route_file(flow: int, ep: int) -> str:
    """
    Route fájl generálása fix forgalommal — kizárólag TARGET_JID sávjaira.
    Pontosan `flow` jármű/óra/sáv kerül minden incoming sávra, más junction
    nem kap forgalmat (csak R1C1_C-t mérjük, a többi felesleges overhead).
    Seed = flow * 10000 + ep → determinisztikus, de epizódonként különböző
    indulási ütemezés (variancia becsléshez).
    """
    _import_sumo()
    fname = os.path.join(SCRIPT_DIR, f"_cmp_{TARGET_JID}_flow{flow}_ep{ep}.rou.xml")
    if os.path.exists(fname):
        return fname

    rng = random.Random(flow * 10000 + ep)   # determinisztikus seed

    net = sumolib.net.readNet(NET_FILE)

    node = net.getNode(TARGET_JID)
    if node is None:
        raise ValueError(f"Junction '{TARGET_JID}' nem található a hálózatban.")

    all_trips = []
    for inc_edge in node.getIncoming():
        eid = inc_edge.getID()
        if eid.startswith(':'):
            continue
        for lane in inc_edge.getLanes():
            lane_idx = lane.getIndex()
            targets  = []
            for conn in lane.getOutgoing():
                to_edge = conn.getToLane().getEdge()
                teid = to_edge.getID()
                if not teid.startswith(':'):
                    targets.append(teid)
            if not targets:
                continue
            num_veh = int(flow * (DURATION / 3600.0))
            if num_veh <= 0:
                continue
            avg_gap = DURATION / max(num_veh, 1)
            for i in range(num_veh):
                dep = i * avg_gap + rng.uniform(0, avg_gap * 0.3)
                dep = min(dep, DURATION - 1.0)
                all_trips.append((dep, eid, lane_idx, rng.choice(targets)))

    all_trips.sort(key=lambda x: x[0])
    with open(fname, 'w') as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" accel="0.8" decel="4.5" sigma="0.5" '
                'length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
        for idx, (dep, fe, li, te) in enumerate(all_trips):
            f.write(f'  <trip id="v_{idx}" type="car" depart="{dep:.2f}" '
                    f'from="{fe}" to="{te}" departLane="{li}"/>\n')
        f.write('</routes>\n')
    return fname


def cleanup_route_files():
    for f in Path(SCRIPT_DIR).glob(f"_cmp_{TARGET_JID}_*.rou.xml"):
        try:
            f.unlink()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# SUMO START / STOP
# ─────────────────────────────────────────────────────────────────────────────
_traci_running = False

def _start_sumo(route_file: str):
    global _traci_running
    _import_sumo()
    sumo_args = [
        "-n", NET_FILE, "-r", route_file, "-a", DETECTOR_FILE,
        "--no-step-log", "true", "--ignore-route-errors", "true",
        "--no-warnings", "true", "--xml-validation", "never", "--random", "true",
    ]
    if _use_libsumo:
        if not _traci_running:
            traci.start(["sumo"] + sumo_args)
        else:
            try:
                traci.load(sumo_args)
            except Exception:
                traci.start(["sumo"] + sumo_args)
    else:
        if _traci_running:
            try:
                traci.close()
            except Exception:
                pass
        traci.start(["sumo"] + sumo_args)
    _traci_running = True


def _stop_sumo():
    global _traci_running
    if _traci_running:
        try:
            traci.close()
        except Exception:
            pass
        _traci_running = False

# ─────────────────────────────────────────────────────────────────────────────
# ACTUATED PROGRAM BEÁLLÍTÁSA
# ─────────────────────────────────────────────────────────────────────────────
def _setup_actuated(jid: str):
    programs = traci.trafficlight.getAllProgramLogics(jid)
    if not programs:
        return
    orig = programs[0]
    act_phases = []
    for ph in orig.phases:
        min_dur = max(5.0, ph.duration * 0.5)
        max_dur = max(min_dur + 5.0, ph.duration * 1.5)
        act_phases.append(traci.trafficlight.Phase(ph.duration, ph.state, min_dur, max_dur))
    act_logic = traci.trafficlight.Logic(
        "actuated_cmp", 3, 0, act_phases,
        {"detector-gap": "2.0", "passing-time": "1.5", "max-gap": "3.0"},
    )
    traci.trafficlight.setProgramLogic(jid, act_logic)
    traci.trafficlight.setProgram(jid, "actuated_cmp")

# ─────────────────────────────────────────────────────────────────────────────
# NN MEGFIGYELÉS SEGÉDOSZTÁLY
# ─────────────────────────────────────────────────────────────────────────────
class _LightAgent:
    """
    Minimál TrafficAgent — az RL env TrafficAgentjével ekvivalens,
    de csak az összehasonlítóhoz szükséges részeket tartalmazza.
    """
    def __init__(self, jid: str, logic_data: dict, detectors: list):
        self.jid           = jid
        self.logic_phases  = {int(k): v for k, v in logic_data['logic_phases'].items()}
        self.transitions   = logic_data.get('transitions', {})
        self.num_phases    = len(self.logic_phases)
        self.phase_registry = {p['index']: p for p in logic_data['phases']}
        self.detectors     = list(detectors)
        self.min_green_const = max(1, MIN_GREEN // DELTA_TIME)

        self.current_logic_idx = 0
        self.target_logic_idx  = 0
        self.is_transitioning  = False
        self.transition_queue  = []
        self.transition_cursor = 0
        self.transition_step   = 0
        self.next_logic_cache  = 0
        self.min_green_timer   = 0
        self.phase_timer       = 0
        self._reset_obs()
        self._reset_episode()

    # ── Obs akkumulátorok: reset minden delta_time ablak elején ──────────────
    def _reset_obs(self):
        """Per-lépés obs reset. Minden delta_time ablak elején hívandó."""
        self.steps_measured = 0
        self.det_occ  = {d: 0.0 for d in self.detectors}
        self.det_flow = {d: 0   for d in self.detectors}

    # ── Epizód-szintű metrika akkumulátorok: csak epizód elején resetelni ───
    def _reset_episode(self):
        """Epizód metrika reset. Csak egyszer hívandó az epizód legelején."""
        self.ep_speed       = 0.0
        self.ep_speed_lanes = 0
        self.ep_halt        = 0
        self.ep_veh         = 0
        self.ep_tp          = 0
        self.ep_co2         = 0.0

    # Backward compat: régi hívók is működjenek
    def _reset_metrics(self):
        self._reset_obs()
        self._reset_episode()

    def is_ready(self):
        return (not self.is_transitioning) and (self.min_green_timer <= 0)

    def set_target(self, idx: int):
        if self.is_ready() and idx in self.logic_phases:
            self.target_logic_idx = idx

    def _apply_phase(self):
        sidx = self.logic_phases.get(self.current_logic_idx)
        if sidx is not None and sidx in self.phase_registry:
            try:
                traci.trafficlight.setRedYellowGreenState(
                    self.jid, self.phase_registry[sidx]['state'])
            except Exception:
                pass

    def update(self):
        if self.is_transitioning:
            if self.transition_step > 0:
                self.transition_step -= 1
            else:
                if self.transition_cursor < len(self.transition_queue):
                    sidx = self.transition_queue[self.transition_cursor]
                    if sidx in self.phase_registry:
                        p = self.phase_registry[sidx]
                        try:
                            traci.trafficlight.setRedYellowGreenState(self.jid, p['state'])
                        except Exception:
                            pass
                        self.transition_step = max(0, int(p['duration']) - 1)
                    self.transition_cursor += 1
                else:
                    self.is_transitioning = False
                    self.current_logic_idx = self.next_logic_cache
                    self._apply_phase()
                    self.min_green_timer = self.min_green_const
                    self.phase_timer = 0
        else:
            if self.min_green_timer > 0:
                self.min_green_timer -= 1
                self._apply_phase()
                self.phase_timer += 1
            else:
                if self.target_logic_idx != self.current_logic_idx:
                    key = f"{self.current_logic_idx}->{self.target_logic_idx}"
                    self.transition_queue  = self.transitions.get(key, [])
                    self.is_transitioning  = True
                    self.transition_cursor = 0
                    self.transition_step   = 0
                    self.next_logic_cache  = self.target_logic_idx
                    self.phase_timer = 0
                else:
                    self._apply_phase()
                    self.phase_timer += 1

    def collect(self, incoming_lanes: list):
        if self.is_transitioning:
            return
        # ── Obs akkumulátorok (per delta_time ablak) ─────────────────────────
        self.steps_measured += 1
        for det in self.detectors:
            self.det_occ[det]  += traci.inductionloop.getLastStepOccupancy(det)
            self.det_flow[det] += traci.inductionloop.getLastStepVehicleNumber(det)
        # ── Epizód metrikák ──────────────────────────────────────────────────
        for lane_id in incoming_lanes:
            try:
                veh  = traci.lane.getLastStepVehicleNumber(lane_id)
                halt = traci.lane.getLastStepHaltingNumber(lane_id)
                self.ep_halt += halt
                self.ep_veh  += veh
                if veh > 0:
                    self.ep_speed       += traci.lane.getLastStepMeanSpeed(lane_id)
                    self.ep_speed_lanes += 1
                self.ep_co2 += traci.lane.getCO2Emission(lane_id)
            except Exception:
                pass
        # Throughput: detektor flow kumulált összege az epizódban
        self.ep_tp = sum(self.det_flow[d] for d in self.detectors)

    def get_obs(self) -> dict:
        """Megfigyelés — pontosan ugyanúgy mint az edzésnél (SumoRLEnvironment).

        A modell a delta_time ablak alatt akkumulált átlagokat kapta edzéskor:
          occ[i]  = (Σ occupancy / steps_measured) / 100
          flow[i] = Σ vehicle_count / steps_measured
        Ezért itt is ezeket adjuk vissza, nem pillanatfelvételt.
        Ha steps_measured == 0 (pl. átmenet alatt), nullákat adunk.
        """
        n = len(self.detectors)
        occ  = np.zeros(n, dtype=np.float32)
        flow = np.zeros(n, dtype=np.float32)
        if self.steps_measured > 0:
            for i, d in enumerate(self.detectors):
                occ[i]  = (self.det_occ[d]  / self.steps_measured) / 100.0
                flow[i] = (self.det_flow[d] / self.steps_measured)
        return {
            "phase":       np.array([self.current_logic_idx], dtype=np.int32),
            "phase_timer": np.array([min(self.phase_timer / 120.0, 1.0)], dtype=np.float32),
            "occupancy":   occ,
            "flow":        flow,
        }

    def metrics(self) -> dict:
        avg_spd     = self.ep_speed / max(self.ep_speed_lanes, 1)
        halt_r      = self.ep_halt  / max(self.ep_veh, 1)
        tp          = float(self.ep_tp)
        total_co2_g = self.ep_co2 / 1000.0   # mg → g
        return {"avg_speed": avg_spd, "throughput": tp,
                "halt_ratio": halt_r, "total_co2_g": total_co2_g}


# ─────────────────────────────────────────────────────────────────────────────
# SB3 MODEL BETÖLTÉS
# ─────────────────────────────────────────────────────────────────────────────
def load_sb3_model(zip_path: str):
    """
    Betölti az SB3 modellt CPU-ra; automatikusan megpróbálja DQN / QRDQN / PPO / A2C sorrendben.
    """
    import zipfile
    # Melyik algoritmust használt?
    algo_name = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            if 'data' in z.namelist():
                meta = json.loads(z.read('data'))
                algo_name = meta.get('policy_class', {}).get('__type', '')
    except Exception:
        pass

    candidates = []
    try:
        from stable_baselines3 import DQN, PPO, A2C
        candidates += [('DQN', DQN), ('PPO', PPO), ('A2C', A2C)]
    except ImportError:
        pass
    try:
        from sb3_contrib import QRDQN
        candidates.insert(1, ('QRDQN', QRDQN))
    except ImportError:
        pass

    if not candidates:
        raise ImportError("stable_baselines3 nincs telepítve.")

    # Ha tudjuk az algo nevét, azzal kezdjük
    if algo_name:
        candidates.sort(key=lambda c: (0 if algo_name.lower().find(c[0].lower()) >= 0 else 1))

    last_err = None
    for name, AlgoClass in candidates:
        try:
            model = AlgoClass.load(zip_path, device='cpu')
            return model, name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Nem sikerült betölteni: {zip_path}\nUtolsó hiba: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# EPIZÓD FUTTATÁSA
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(strategy: str, route_file: str,
                model=None, junction_dets: dict = None,
                junction_lanes: dict = None, logic_data: dict = None) -> dict:
    """
    Lefuttat egy teljes szimulációt egy adott stratégiával.
    Visszatér a mért metrikák szótárával (TARGET_JID-re vonatkozó).

    strategy: "fixed" | "actuated" | "nn:<run_name>"
    model:    SB3 model (csak nn esetén szükséges)
    """
    _start_sumo(route_file)
    all_loops = traci.inductionloop.getIDList()
    jid = TARGET_JID

    # Detektorok és sávok meghatározása (ha már ismert, újrahasználjuk)
    if junction_dets is None or jid not in junction_dets:
        controlled = traci.trafficlight.getControlledLinks(jid)
        incoming_lanes = set()
        for group in controlled:
            for link in group:
                if link:
                    incoming_lanes.add(link[0])
        dets = sorted([l for l in all_loops
                       if traci.inductionloop.getLaneID(l) in incoming_lanes])
    else:
        dets = junction_dets[jid]
        incoming_lanes = set(junction_lanes.get(jid, []))

    with open(LOGIC_FILE) as f:
        ldata = json.load(f)

    agent = _LightAgent(jid, ldata[jid], dets)

    # Actuated mód beállítása (traci-n keresztül)
    if strategy == "actuated":
        _setup_actuated(jid)

    # Warmup
    for _ in range(WARMUP // DELTA_TIME * DELTA_TIME):
        traci.simulationStep()

    # Fő szimuláció
    # Loop sorrend = SumoRLEnvironment.step() sorrendje:
    #   1. reset  2. delta_time lépés + collect  3. obs a gyűjtött átlagokból
    #   4. predict → set_target a KÖVETKEZŐ periódushoz
    total_steps = (DURATION - WARMUP) // DELTA_TIME
    _model_keys = (set(model.policy.observation_space.spaces.keys())
                   if strategy == "nn" and model is not None else set())

    agent._reset_episode()   # Epizód metrikák: egyszer, itt

    for _ in range(total_steps):
        agent._reset_obs()   # Obs akkumulátorok: minden delta_time ablak elején

        # Szimuláció lépések + mérés ebben az ablakban
        for _ in range(DELTA_TIME):
            if strategy == "nn":
                agent.update()
            traci.simulationStep()
            agent.collect(list(incoming_lanes))

        # Ablak lefutott → obs a most gyűjtött átlagokból (= edzéskori megfigyelés)
        if strategy == "nn" and agent.is_ready() and model is not None:
            obs = agent.get_obs()
            obs_filtered = {k: v for k, v in obs.items() if k in _model_keys}
            obs_batch    = {k: v.reshape(1, *v.shape) for k, v in obs_filtered.items()}
            action, _    = model.predict(obs_batch, deterministic=True)
            agent.set_target(int(action[0]))

    result = agent.metrics()   # Epizód összesítő
    result['strategy'] = strategy
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODELLEK FELFEDEZÉSE
# ─────────────────────────────────────────────────────────────────────────────
def discover_models(extra_zips: list = None) -> list:
    """
    Visszaadja az összes R1C1_C model zip-et:
    [(run_name, zip_path), ...]
    Sorrend: alapértelmezett modell könyvtár, majd extra_zips.
    """
    found = []
    models_dir = Path(MODELS_DIR)
    if models_dir.exists():
        # final/dqn_agent_R1C1_C.zip és best/best_model_R1C1_C.zip
        for pattern in [f"*/final/dqn_agent_{TARGET_JID}.zip",
                         f"*/best/best_model_{TARGET_JID}.zip",
                         f"*/final/*{TARGET_JID}*.zip"]:
            for p in sorted(models_dir.glob(pattern)):
                run_name = p.parent.parent.name
                ckpt = "best" if "best" in p.parent.name else "final"
                label = f"{run_name[:30]} [{ckpt}]"
                found.append((label, str(p)))

    # baseline wandb exportok
    bl_dir = Path(SCRIPT_DIR) / "baseline_reward_final_log" / "wandb"
    if bl_dir.exists():
        for p in sorted(bl_dir.glob(f"*/files/dqn_agent_{TARGET_JID}.zip")):
            run_name = p.parent.parent.name[:30]
            found.append((f"baseline/{run_name}", str(p)))

    # User-supplied extra zips
    if extra_zips:
        for z in extra_zips:
            zp = Path(z)
            if zp.exists():
                found.append((zp.stem, str(zp)))

    # Deduplikálás (path alapján)
    seen = set()
    unique = []
    for name, path in found:
        if path not in seen:
            seen.add(path)
            unique.append((name, path))
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# FŐ ÖSSZEHASONLÍTÁS
# ─────────────────────────────────────────────────────────────────────────────
def run_all_comparisons(extra_zips: list = None,
                        strategies_filter: list = None) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Baseline stratégiák
    baselines = [("fixed", None), ("actuated", None)]

    # NN modellek
    nn_models = []
    all_zips = discover_models(extra_zips)
    print(f"\nFelfedezett modellek: {len(all_zips)} db")
    for label, zpath in all_zips:
        try:
            model, algo = load_sb3_model(zpath)
            nn_models.append((label, model, algo))
            print(f"  ✓ [{algo}] {label}")
        except Exception as e:
            print(f"  ✗ {label}: {e}")

    all_strategies = baselines + [("nn", m, lbl, algo)
                                   for lbl, m, algo in nn_models]
    if strategies_filter:
        all_strategies = [s for s in all_strategies
                          if any(f in (s[0] if isinstance(s, tuple) else s)
                                 for f in strategies_filter)]

    records = []
    total = len(FLOW_LEVELS) * EPISODES_PER_LEVEL * len(all_strategies)
    done  = 0

    for flow in FLOW_LEVELS:
        for ep in range(EPISODES_PER_LEVEL):
            route_file = generate_route_file(flow, ep)
            print(f"\n── Flow={flow} veh/h | Epizód {ep+1}/{EPISODES_PER_LEVEL} ──")

            for strat_entry in all_strategies:
                if len(strat_entry) == 2:
                    strategy_key, model = strat_entry
                    label = strategy_key
                    algo  = "—"
                elif len(strat_entry) == 4:
                    strategy_key, model, label, algo = strat_entry
                    strategy_key = "nn"
                else:
                    continue

                if strategies_filter and label not in strategies_filter \
                        and strategy_key not in strategies_filter:
                    done += 1
                    continue

                try:
                    t0 = time.time()
                    result = run_episode(strategy_key, route_file, model=model)
                    elapsed = time.time() - t0

                    rec = {
                        "strategy": label,
                        "algo":     algo,
                        "flow":     flow,
                        "episode":  ep,
                        "avg_speed":   result['avg_speed'],
                        "throughput":  result['throughput'],
                        "halt_ratio":  result['halt_ratio'],
                        "elapsed_s":   round(elapsed, 1),
                    }
                    records.append(rec)
                    done += 1
                    print(f"  [{done:3d}/{total}] {label:<40s} "
                          f"spd={result['avg_speed']:.2f}m/s  "
                          f"tp={result['throughput']:.0f}  "
                          f"halt={result['halt_ratio']:.3f}  "
                          f"({elapsed:.0f}s)")
                except Exception as e:
                    import traceback
                    print(f"  HIBA [{label}]: {e}")
                    traceback.print_exc()
                    done += 1

    _stop_sumo()
    cleanup_route_files()

    df = pd.DataFrame(records)
    if df.empty:
        print("Nincs adat — valami hiba történt a futtatás során.")
        return df

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Eredmények mentve: {csv_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÁBRÁK
# ─────────────────────────────────────────────────────────────────────────────
def _strategy_order(df: pd.DataFrame) -> list:
    """Sorrend: fixed, actuated, majd NN-ek avg_speed alapján csökkenő sorban."""
    base = [s for s in ["fixed", "actuated"] if s in df['strategy'].values]
    nn = [s for s in df['strategy'].unique() if s not in base]
    nn_sorted = sorted(nn, key=lambda s: df[df['strategy'] == s]['avg_speed'].mean(),
                        reverse=True)
    return base + nn_sorted


def _palette(strategies: list) -> dict:
    fixed_colors = {"fixed": "#4A90D9", "actuated": "#27AE60"}
    cmap = cm.get_cmap('tab20', max(1, len(strategies) - 2))
    nn_strats = [s for s in strategies if s not in fixed_colors]
    colors = dict(fixed_colors)
    for i, s in enumerate(nn_strats):
        colors[s] = matplotlib.colors.to_hex(cmap(i))
    return colors


def plot_results(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    if df.empty:
        print("Nincs adat a plothoz.")
        return

    order   = _strategy_order(df)
    palette = _palette(order)

    metrics = [
        ("avg_speed",  "Átl. sebesség [m/s]",   True),
        ("throughput", "Throughput [járm]",       True),
        ("halt_ratio", "Megállási arány [0–1]",   False),
    ]

    # ── 1. Sávdiagram (átlag ± szórás, minden flow-szinten egybe)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Stratégia-összehasonlítás — {TARGET_JID}  "
                 f"(flow: {FLOW_LEVELS} veh/h, {EPISODES_PER_LEVEL} ep/szint)",
                 fontsize=13, y=1.01)

    for ax, (col, ylabel, higher_better) in zip(axes, metrics):
        means = df.groupby('strategy')[col].mean()
        stds  = df.groupby('strategy')[col].std(ddof=0).fillna(0)
        x     = np.arange(len(order))
        bars  = ax.bar(x, [means.get(s, 0) for s in order],
                       yerr=[stds.get(s, 0) for s in order],
                       capsize=4, error_kw=dict(lw=1.2),
                       color=[palette[s] for s in order],
                       edgecolor='#333', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s if len(s) <= 22 else s[:20] + '…' for s in order],
            rotation=40, ha='right', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        arrow = "↑ jobb" if higher_better else "↓ jobb"
        ax.set_title(f"{col}  ({arrow})", fontsize=10)
        ax.grid(axis='y', lw=0.5, alpha=0.5)
        # Kiemelés: legjobb stratégia
        best_val = max(means.get(s, 0) for s in order) if higher_better \
                   else min(means.get(s, np.inf) for s in order)
        for s, bar in zip(order, bars):
            v = means.get(s, 0)
            if abs(v - best_val) < 1e-9:
                bar.set_edgecolor('#E74C3C')
                bar.set_linewidth(2.5)

    fig.tight_layout()
    out = os.path.join(output_dir, "comparison_bars.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")

    # ── 2. Boxplot — flow-szintenként
    n_flows = len(df['flow'].unique())
    fig, axes = plt.subplots(n_flows, 3,
                             figsize=(16, 4 * n_flows),
                             sharey='col')
    if n_flows == 1:
        axes = [axes]
    fig.suptitle(f"Flow-szintenkénti boxplot — {TARGET_JID}", fontsize=13, y=1.01)

    for row, flow in enumerate(sorted(df['flow'].unique())):
        sub = df[df['flow'] == flow]
        for col_idx, (col, ylabel, _) in enumerate(metrics):
            ax = axes[row][col_idx]
            data = [sub[sub['strategy'] == s][col].values for s in order]
            bp   = ax.boxplot(data, patch_artist=True, notch=False,
                               medianprops=dict(color='black', lw=2))
            for patch, s in zip(bp['boxes'], order):
                patch.set_facecolor(palette[s])
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(
                [s if len(s) <= 18 else s[:16] + '…' for s in order],
                rotation=40, ha='right', fontsize=7)
            if col_idx == 0:
                ax.set_ylabel(f"Flow {flow}\n{ylabel}", fontsize=8)
            else:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis='y', lw=0.5, alpha=0.4)

    fig.tight_layout()
    out = os.path.join(output_dir, "comparison_box.png")
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")

    # ── 3. Összesítő táblázat kiírás
    print("\n─── Összesítő (átlag minden flow-szinten) ───")
    summary = df.groupby('strategy')[['avg_speed','throughput','halt_ratio']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    print(summary.sort_values('avg_speed_mean', ascending=False).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Összehasonlítja a forgalomirányítási stratégiákat R1C1_C-n")
    parser.add_argument('--plot-only', action='store_true',
                        help='Csak az ábrák generálása (meglévő results.csv alapján)')
    parser.add_argument('--models-dir', default='',
                        help='Egyéni modell könyvtár (alapértelmezett: models/sumo-rl-independent)')
    parser.add_argument('--add-zip', action='append', default=[],
                        dest='extra_zips', metavar='ZIP',
                        help='Extra model zip (többször megadható)')
    parser.add_argument('--strategies', nargs='+', default=[],
                        help='Szűrés stratégia nevekre (üres = mind)')
    parser.add_argument('--output-dir', default='',
                        help='Kimeneti könyvtár (alap: eval_comparison)')
    args = parser.parse_args()

    global MODELS_DIR, OUTPUT_DIR
    if args.models_dir:
        MODELS_DIR = args.models_dir
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.plot_only:
        csv_path = os.path.join(OUTPUT_DIR, "results.csv")
        if not os.path.exists(csv_path):
            print(f"Nincs results.csv: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        plot_results(df)
        return

    print(f"═══ Stratégia-összehasonlítás indítása ═══")
    print(f"  Junction:   {TARGET_JID}")
    print(f"  Flow:       {FLOW_LEVELS} veh/h")
    print(f"  Epizódok:   {EPISODES_PER_LEVEL} / szint")
    print(f"  Időtartam:  {DURATION}s + {WARMUP}s warmup")
    print(f"  libsumo:    {'igen' if _use_libsumo else 'nem (traci)'}")
    print(f"  Modellek:   {MODELS_DIR}")

    df = run_all_comparisons(
        extra_zips=args.extra_zips or None,
        strategies_filter=args.strategies or None,
    )
    if not df.empty:
        print("\nÁbrák generálása...")
        plot_results(df)


if __name__ == '__main__':
    main()
