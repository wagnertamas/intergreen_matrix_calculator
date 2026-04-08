#!/usr/bin/env python3
"""
Per-junction metrika gyűjtés és normalizációs paraméter kalibráció.

A metric_collection_test.py szimulációs logikáját használja (teljes hálózat,
minden junction-re forgalom + random jelzőlámpa), de az elemzést
JUNCTION-ÖNKÉNT végzi el.

Kimenet:
  metric_pca_per_junction/
    ├── junction_reward_params.json   ← A LÉNYEG: {global: {...}, per_junction: {jid: {MU_SPEED, STD_SPEED, MU_THROUGHPUT, STD_THROUGHPUT}}}
    ├── junction_comparison.png       ← Heatmap + bar chart összehasonlítás
    ├── junction_reward_ranges.png    ← Per-junction reward eloszlás (saját vs globális params)
    ├── per_junction_summary.csv      ← Részletes tábla
    └── <jid>_flow<N>_ep<M>.csv      ← Nyers step-szintű adatok (mint korábban)

Használat:
    python metric_collection_per_junction.py
    python metric_collection_per_junction.py --skip-simulation   # csak elemzés (ha CSV-k már megvannak)
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
NET_FILE = os.path.join(DATA_DIR, "mega_catalogue_v2.net.xml")
LOGIC_FILE = os.path.join(DATA_DIR, "traffic_lights.json")
DETECTOR_FILE = os.path.join(DATA_DIR, "detectors.add.xml")

# --- Config ---
FLOW_MIN = 100
FLOW_MAX_LEVELS = [200,300, 400, 500, 600]
DURATION = 1800       # 1 óra szimuláció
DELTA_TIME = 5        # lépésméret (sec)
WARMUP = 100          # warmup (sec)
EPISODES_PER_LEVEL = 0  # ismétlések forgalmi szintenként (random kontroll)
ACTUATED_EPISODES = 8   # ismétlések forgalmi szintenként (actuated kontroll)
MAX_REALISTIC_TT = 1000  # travel time szűrő

# --- Globális referencia (AvgSpeed + Throughput, log-tanh normalizáció) ---
# Ezek az értékek az analyze_per_junction() futtatásakor frissülnek.
GLOBAL_PARAMS = {
    'MU_SPEED': None,       # log(AvgSpeed) median — kalibrációnál töltődik
    'STD_SPEED': None,      # log(AvgSpeed) IQR-based STD
    'MU_THROUGHPUT': None,  # log(Throughput) median
    'STD_THROUGHPUT': None, # log(Throughput) IQR-based STD
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "metric_pca_per_junction")

# --- Junction comparison plot variáns ---
# Lehetséges értékek: "plain" | "halt" | "triplet" | "" (üres = auto: legjobb IQR alapján)
PLOT_REWARD_VARIANT = "plain"

# --- Reward step curve (reward_step_curve.png) ---
# Ha meg van adva, csak ezt a junction-t plotolja LOKÁLIS normalizációs paraméterekkel.
# Ha üres → minden junction, globális paraméterekkel.
REWARD_CURVE_JUNCTION = "R1C1_C"


def run_simulation(flow_max, episode_idx, output_dir, control_mode="random", use_gui=False):
    """
    Egy szimulációs epizód futtatása.
    PONTOSAN UGYANAZ mint metric_collection_test.py — teljes hálózaton fut,
    minden junction-re gyűjt metrikát.

    control_mode:
        "random"   — random jelzőlámpa fázisválasztás (eredeti viselkedés)
        "actuated" — SUMO beépített actuated kontroller (reálisabb baseline)
    use_gui:
        False — libsumo (headless, gyors)
        True  — sumo-gui (vizuális, socket alapú traci)
    """

    if use_gui:
        import traci
        import sumolib
    else:
        import libsumo as traci
        import sumolib

    route_file = os.path.join(SCRIPT_DIR, f"_metric_pj_{flow_max}_{episode_idx}_{control_mode}.rou.xml")

    # --- Forgalom generálás (lane-szintű) ---
    net = sumolib.net.readNet(NET_FILE)
    with open(LOGIC_FILE) as f:
        logic = json.load(f)

    junction_ids = list(logic.keys())

    lane_routes = {}
    for jid in junction_ids:
        node = net.getNode(jid)
        if node is None:
            continue
        for inc_edge in node.getIncoming():
            eid = inc_edge.getID()
            if eid.startswith(':'):
                continue
            for lane in inc_edge.getLanes():
                lane_id = lane.getID()
                lane_idx = lane.getIndex()
                targets = set()
                for conn in lane.getOutgoing():
                    to_lane = conn.getToLane()
                    to_edge = to_lane.getEdge()
                    to_eid = to_edge.getID()
                    if not to_eid.startswith(':'):
                        targets.add(to_eid)
                if targets:
                    lane_routes[lane_id] = (eid, lane_idx, list(targets))

    all_trips = []
    for lane_id, (edge_id, lane_idx, to_edges) in lane_routes.items():
        flow = random.randint(FLOW_MIN, flow_max)
        num_veh = int(flow * (DURATION / 3600.0))
        if num_veh <= 0:
            continue
        avg_gap = DURATION / num_veh
        for i in range(num_veh):
            depart = i * avg_gap + random.uniform(0, avg_gap * 0.5)
            depart = min(depart, DURATION - 1.0)
            to_edge = random.choice(to_edges)
            all_trips.append((depart, edge_id, lane_idx, to_edge))

    all_trips.sort(key=lambda x: x[0])

    with open(route_file, 'w') as f:
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" '
                'length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
        for idx, (depart, from_e, lane_idx, to_e) in enumerate(all_trips):
            f.write(f'    <trip id="veh_{idx}" type="car" depart="{depart:.2f}" '
                    f'from="{from_e}" to="{to_e}" departLane="{lane_idx}" />\n')
        f.write('</routes>\n')

    # --- SUMO indítás ---
    sumo_args = ["-n", NET_FILE, "-r", route_file, "-a", DETECTOR_FILE,
        "--no-step-log", "true", "--ignore-route-errors", "true",
        "--no-warnings", "true", "--xml-validation", "never", "--random", "true"]

    if use_gui:
        sumo_bin = "sumo-gui"
        traci.start([sumo_bin] + sumo_args)
    else:
        traci.load(sumo_args)

    # --- Junction → bejövő lane-ek és detektorok ---
    junction_lanes = {}
    junction_dets = {}

    all_loops = traci.inductionloop.getIDList()

    for jid in junction_ids:
        controlled = traci.trafficlight.getControlledLinks(jid)
        incoming_lanes = set()
        for link_group in controlled:
            for link in link_group:
                if link:
                    incoming_lanes.add(link[0])
        junction_lanes[jid] = sorted(incoming_lanes)

        dets = []
        for loop in all_loops:
            lane_id = traci.inductionloop.getLaneID(loop)
            if lane_id in incoming_lanes:
                dets.append(loop)
        junction_dets[jid] = sorted(dets)

    # --- Metrikák ---
    metrics = {jid: [] for jid in junction_ids}

    # --- Warmup ---
    for _ in range(WARMUP):
        traci.simulationStep()

    # --- Jelzőlámpa vezérlők inicializálása (random módhoz) ---
    # Ugyanolyan logika mint sumo_rl_environment.py: előre definiált fázisok,
    # átmenetekkel és minimum zöld idővel (MIN_GREEN_TIME lépés).
    MIN_GREEN_TIME = max(1, 5 // DELTA_TIME)  # 5 mp -> lépésben

    class TLController:
        """Egyszerűsített TrafficLightAgent — ugyanolyan fázislogika mint az RL env-ben."""
        def __init__(self, jid, logic_data):
            self.jid = jid
            self.logic_phases   = {int(k): v for k, v in logic_data['logic_phases'].items()}
            self.transitions    = logic_data['transitions']
            # Fázis adatok: index -> {state, duration}
            self.phase_data     = {p['index']: p for p in logic_data['phases']}
            self.num_phases     = len(self.logic_phases)

            self.current_logic_idx   = 0
            self.target_logic_idx    = 0
            self.is_transitioning    = False
            self.transition_queue    = []
            self.transition_cursor   = 0
            self.transition_step_timer = 0
            self.next_logic_idx_cache = 0
            self.min_green_timer     = MIN_GREEN_TIME

            # Kezdeti SUMO fázis beállítása
            self._apply_current_phase()

        def is_ready(self):
            return (not self.is_transitioning) and (self.min_green_timer <= 0)

        def set_target(self, idx):
            if self.is_ready() and idx in self.logic_phases:
                self.target_logic_idx = idx

        def update(self):
            if self.is_transitioning:
                if self.transition_step_timer > 0:
                    self.transition_step_timer -= 1
                else:
                    if self.transition_cursor < len(self.transition_queue):
                        sumo_idx = self.transition_queue[self.transition_cursor]
                        pd = self.phase_data.get(sumo_idx)
                        if pd:
                            try:
                                traci.trafficlight.setRedYellowGreenState(self.jid, pd['state'])
                            except Exception:
                                pass
                            # Duration lépésben (DELTA_TIME mp / lépés)
                            dur_steps = max(0, int(pd.get('duration', DELTA_TIME) / DELTA_TIME) - 1)
                            self.transition_step_timer = dur_steps
                        self.transition_cursor += 1
                    else:
                        self.is_transitioning    = False
                        self.current_logic_idx   = self.next_logic_idx_cache
                        self._apply_current_phase()
                        self.min_green_timer     = MIN_GREEN_TIME
            else:
                if self.min_green_timer > 0:
                    self.min_green_timer -= 1
                    self._apply_current_phase()
                else:
                    if self.target_logic_idx != self.current_logic_idx:
                        self._start_transition(self.target_logic_idx)
                    else:
                        self._apply_current_phase()

        def _apply_current_phase(self):
            sumo_idx = self.logic_phases.get(self.current_logic_idx)
            if sumo_idx is not None:
                pd = self.phase_data.get(sumo_idx)
                if pd:
                    try:
                        traci.trafficlight.setRedYellowGreenState(self.jid, pd['state'])
                    except Exception:
                        pass

        def _start_transition(self, next_idx):
            key = f"{self.current_logic_idx}->{next_idx}"
            self.transition_queue     = self.transitions.get(key, [])
            self.is_transitioning     = True
            self.transition_cursor    = 0
            self.transition_step_timer = 0
            self.next_logic_idx_cache = next_idx

    # Vezérlők létrehozása (csak random módhoz kell)
    tl_controllers = {}
    if control_mode == "random":
        for jid in junction_ids:
            if jid in logic:
                tl_controllers[jid] = TLController(jid, logic[jid])

    # --- Actuated mód beállítása ---
    if control_mode == "actuated":
        for jid in junction_ids:
            programs = traci.trafficlight.getAllProgramLogics(jid)
            if programs:
                orig = programs[0]
                # Actuated program: min/max zöld idő a fázisonkénti duration alapján
                act_phases = []
                for ph in orig.phases:
                    # Actuated fázisok: minDur = duration*0.5, maxDur = duration*1.5
                    min_dur = max(5.0, ph.duration * 0.5)
                    max_dur = max(min_dur + 5.0, ph.duration * 1.5)
                    act_phases.append(traci.trafficlight.Phase(
                        ph.duration, ph.state, min_dur, max_dur
                    ))
                act_logic = traci.trafficlight.Logic(
                    "actuated_calib",  # programID
                    3,                 # type: 3 = actuated
                    0,                 # currentPhaseIndex
                    act_phases,        # phases
                    {                  # parameters
                        "detector-gap": "2.0",
                        "passing-time": "1.5",
                        "max-gap": "3.0",
                    }
                )
                traci.trafficlight.setProgramLogic(jid, act_logic)
                traci.trafficlight.setProgram(jid, "actuated_calib")
        print(f"      [ACTUATED] Jelzolampa programok atallitva actuated modra")

    # Előző fázis tárolás (state-hez)
    prev_phase = {jid: 0 for jid in junction_ids}

    total_steps = (DURATION - WARMUP) // DELTA_TIME

    for step_i in range(total_steps):
        # Jelzőlámpa vezérlés: random VAGY actuated (nincs beavatkozás)
        current_phase = {}
        if control_mode == "random":
            for jid in junction_ids:
                ctrl = tl_controllers.get(jid)
                if ctrl:
                    # Csak akkor adunk új akciót, ha a kontroller készen áll
                    # (nem transitioning és letelt a min green time)
                    if ctrl.is_ready():
                        ctrl.set_target(random.randint(0, ctrl.num_phases - 1))
                    ctrl.update()
                    current_phase[jid] = ctrl.current_logic_idx
                else:
                    current_phase[jid] = 0
        else:
            # Actuated: a SUMO maga kezeli, csak kiolvassuk az aktuális fázist
            for jid in junction_ids:
                current_phase[jid] = traci.trafficlight.getPhase(jid)

        # Akkumulálás delta_time lépésen
        acc = {jid: {
            'tt': 0.0, 'tt_raw': 0.0, 'waiting': 0.0, 'co2': 0.0,
            'veh': 0, 'speed': 0.0, 'halted': 0, 'occ': 0.0,
            'valid_tt': 0, 'valid_tt_raw': 0, 'valid_speed': 0,
            'steps': 0,
            # Új metrikák
            'lane_speeds': [],       # per-sub-step lane átlagsebességek (SpeedStd-hez)
            'throughput': 0,         # detektoron áthaladó járművek (kumulált darabszám)
            # Detektor-szintű state akkumulátorok
            'det_occ': defaultdict(float),   # per-detektor occupancy
            'det_veh': defaultdict(int),     # per-detektor vehicle count
            'det_speed': defaultdict(float), # per-detektor speed
            'det_speed_valid': defaultdict(int),
        } for jid in junction_ids}

        for dt_step in range(DELTA_TIME):
            traci.simulationStep()

            for jid in junction_ids:
                lanes = junction_lanes[jid]
                dets = junction_dets[jid]
                acc[jid]['steps'] += 1

                sub_step_speeds = []  # sávsebességek ebben az 1s sub-step-ben

                for lane in lanes:
                    tt = traci.lane.getTraveltime(lane)
                    if 0 < tt < MAX_REALISTIC_TT:
                        acc[jid]['tt'] += tt
                        acc[jid]['valid_tt'] += 1
                    if tt > 0:
                        acc[jid]['tt_raw'] += tt
                        acc[jid]['valid_tt_raw'] += 1

                    acc[jid]['waiting'] += traci.lane.getWaitingTime(lane)
                    acc[jid]['co2'] += traci.lane.getCO2Emission(lane)
                    acc[jid]['veh'] += traci.lane.getLastStepVehicleNumber(lane)

                    speed = traci.lane.getLastStepMeanSpeed(lane)
                    if speed >= 0:
                        acc[jid]['speed'] += speed
                        acc[jid]['valid_speed'] += 1
                        sub_step_speeds.append(speed)

                    acc[jid]['halted'] += traci.lane.getLastStepHaltingNumber(lane)

                # Sub-step sávsebességeket gyűjtjük a SpeedStd-hez
                acc[jid]['lane_speeds'].extend(sub_step_speeds)

                # Detektor-szintű state + throughput gyűjtés
                for det in dets:
                    acc[jid]['det_occ'][det] += traci.inductionloop.getLastStepOccupancy(det)
                    det_veh_count = traci.inductionloop.getLastStepVehicleNumber(det)
                    acc[jid]['det_veh'][det] += det_veh_count
                    acc[jid]['throughput'] += det_veh_count  # áthaladó járművek
                    det_speed = traci.inductionloop.getLastStepMeanSpeed(det)
                    if det_speed >= 0:
                        acc[jid]['det_speed'][det] += det_speed
                        acc[jid]['det_speed_valid'][det] += 1
                    # Occupancy összes (régi logika kompatibilitás)
                    acc[jid]['occ'] += traci.inductionloop.getLastStepOccupancy(det)

        # Átlagolás
        for jid in junction_ids:
            a = acc[jid]
            s = a['steps']
            if s == 0:
                continue

            avg_waiting_total = a['waiting'] / s
            avg_co2_total = a['co2'] / s
            avg_tt_total = a['tt'] / s if a['valid_tt'] > 0 else 0.0
            avg_tt_raw_total = a['tt_raw'] / s if a['valid_tt_raw'] > 0 else 0.0

            avg_veh = a['veh'] / s
            avg_waiting_per_veh = a['waiting'] / a['veh'] if a['veh'] > 0 else 0.0
            avg_co2_per_veh = a['co2'] / a['veh'] if a['veh'] > 0 else 0.0
            avg_speed = a['speed'] / a['valid_speed'] if a['valid_speed'] > 0 else 0.0
            avg_halted = a['halted'] / s
            avg_occ = (a['occ'] / s) / max(len(junction_dets[jid]), 1)

            # --- Új közlekedési metrikák ---
            # Throughput: járművek / DELTA_TIME sec (detektorokon áthaladó)
            # Összesen ennyi jármű haladt át DELTA_TIME másodperc alatt
            throughput = a['throughput']

            # SpeedStd: sávsebesség szórás a DELTA_TIME-on belül
            # A lane_speeds lista tartalmazza az összes (sub-step × lane) sebességet
            # Magas szórás = stop-and-go dinamika (van aki áll, van aki megy)
            speed_std = float(np.std(a['lane_speeds'])) if len(a['lane_speeds']) > 1 else 0.0

            # HaltRatio: álló járművek aránya (0-1 tartomány)
            # halted / total_veh, mindkettő kumulált a DELTA_TIME-on
            halt_ratio = a['halted'] / a['veh'] if a['veh'] > 0 else 0.0

            # Per-detektor state értékek
            dets = junction_dets[jid]
            det_occ_vals = []
            det_veh_vals = []
            det_speed_vals = []
            for det in dets:
                det_occ_vals.append(a['det_occ'][det] / s)
                det_veh_vals.append(a['det_veh'][det] / s)
                sv = a['det_speed_valid'][det]
                det_speed_vals.append(a['det_speed'][det] / sv if sv > 0 else 0.0)

            row = {
                'step': step_i,
                'control_mode': control_mode,
                'phase': current_phase[jid],
                'prev_phase': prev_phase[jid],
                # --- Reward metrikák (sáv-alapú, mint eddig) ---
                'TotalTravelTime': avg_tt_total,
                'AvgTravelTime': avg_tt_total / max(a['valid_tt']/s, 1) if a['valid_tt'] > 0 else 0.0,
                'TotalTravelTime_Raw': avg_tt_raw_total,
                'AvgTravelTime_Raw': avg_tt_raw_total / max(a['valid_tt_raw']/s, 1) if a['valid_tt_raw'] > 0 else 0.0,
                'TotalWaitingTime': avg_waiting_total,
                'AvgWaitingTime': avg_waiting_per_veh,
                'TotalCO2': avg_co2_total,
                'AvgCO2': avg_co2_per_veh,
                'VehCount': avg_veh,
                'AvgSpeed': avg_speed,
                'AvgOccupancy': avg_occ,
                'QueueLength': avg_halted,
                # --- Új közlekedési metrikák ---
                'Throughput': throughput,
                'SpeedStd': speed_std,
                'HaltRatio': halt_ratio,
                # --- State metrikák (detektor-szintű, aggregált) ---
                'det_occ_mean': np.mean(det_occ_vals) if det_occ_vals else 0.0,
                'det_occ_max': max(det_occ_vals) if det_occ_vals else 0.0,
                'det_veh_sum': sum(det_veh_vals),
                'det_veh_mean': np.mean(det_veh_vals) if det_veh_vals else 0.0,
                'det_speed_mean': np.mean(det_speed_vals) if det_speed_vals else 0.0,
                'det_speed_min': min(det_speed_vals) if det_speed_vals else 0.0,
                'n_detectors': len(dets),
            }
            # Per-detektor értékek (d0_occ, d0_veh, d0_speed, d1_occ, ...)
            for di, det in enumerate(dets):
                row[f'd{di}_occ'] = det_occ_vals[di]
                row[f'd{di}_veh'] = det_veh_vals[di]
                row[f'd{di}_speed'] = det_speed_vals[di]

            metrics[jid].append(row)

        # Fázis frissítés
        for jid in junction_ids:
            prev_phase[jid] = current_phase[jid]

        if (step_i + 1) % 50 == 0:
            print(f"    Step {step_i+1}/{total_steps} [{control_mode}]")

    traci.close()

    if os.path.exists(route_file):
        os.remove(route_file)

    # CSV mentés
    for jid in junction_ids:
        df = pd.DataFrame(metrics[jid])
        csv_path = os.path.join(output_dir, f"{jid}_flow{flow_max}_ep{episode_idx}_{control_mode}.csv")
        df.to_csv(csv_path, index=False)

    return metrics


def calc_reward_log_tanh(speed, throughput, mu_s, std_s, mu_t, std_t):
    """Log-tanh reward: AvgSpeed + Throughput kombináció (reward hacking védelem NÉLKÜL).
    AvgSpeed: magasabb = jobb → R = (1 + tanh) / 2
    Throughput: magasabb = jobb → R = (1 + tanh) / 2"""
    z_s = (np.log(speed + 1e-5) - mu_s) / (std_s + 1e-9)
    r_speed = (1.0 + np.tanh(z_s)) / 2.0

    z_t = (np.log(throughput + 1e-5) - mu_t) / (std_t + 1e-9)
    r_throughput = (1.0 + np.tanh(z_t)) / 2.0

    return (r_speed + r_throughput) / 2.0


def calc_reward_log_tanh_with_halt(speed, throughput, halt_ratio,
                                    mu_s, std_s, mu_t, std_t,
                                    mu_h=-1.20, std_h=1.10):
    """Log-tanh reward: AvgSpeed + Throughput + Halt büntetés (reward hacking védelem).
    Ugyanaz mint sumo_rl_environment.py _compute_reward() speed_throughput módja:
        base = (r_speed + r_throughput) / 2
        return base * (1 - r_halt * 0.8)   ← max 80% büntetés ha teljes a dugó
    """
    z_s = (np.log(speed + 1e-5) - mu_s) / (std_s + 1e-9)
    r_speed = (1.0 + np.tanh(z_s)) / 2.0

    z_t = (np.log(throughput + 1e-5) - mu_t) / (std_t + 1e-9)
    r_throughput = (1.0 + np.tanh(z_t)) / 2.0

    z_h = (np.log(halt_ratio + 1e-5) - mu_h) / (std_h + 1e-9)
    r_halt = (1.0 + np.tanh(z_h)) / 2.0

    base = (r_speed + r_throughput) / 2.0
    return base * (1.0 - r_halt * 0.8)


def calc_reward_triplet(m1_vals, m2_vals, halt_vals,
                         mu1, std1, mu2, std2,
                         mu_h=-1.20, std_h=1.10,
                         higher_is_better_m1=True,
                         higher_is_better_m2=True):
    """Általános 3 metrikás additív reward — a 3. mindig HaltRatio.

    Minden metrikát log-tanh normalizál [0,1]-re.
    "Higher is better" metrikáknál: r = (1+tanh(z))/2
    "Lower is better" metrikáknál (pl. CO2): r = (1-tanh(z))/2 = 1 - r_raw

    HaltRatio mindig "lower is better" → r_halt invertálva adódik össze:
        reward = (r_m1 + r_m2 + (1 - r_halt)) / 3
    """
    def _norm(vals, mu, std, higher_better):
        z = (np.log(np.clip(vals, 1e-5, None) + 1e-5) - mu) / (std + 1e-9)
        r = (1.0 + np.tanh(z)) / 2.0
        return r if higher_better else (1.0 - r)

    r1   = _norm(m1_vals,   mu1,  std1,  higher_is_better_m1)
    r2   = _norm(m2_vals,   mu2,  std2,  higher_is_better_m2)
    r_h  = _norm(halt_vals, mu_h, std_h, True)   # r_halt: magasabb halt = rosszabb
    return (r1 + r2 + (1.0 - r_h)) / 3.0


# Pre-definiált triplet kombinációk (csak HaltRatio lehet a 3.)
# Minden bejegyzés: (m1_name, m2_name, m1_higher_better, m2_higher_better)
# Pre-definiált triplet kombinációk (3. mindig HaltRatio)
# (m1_name, m2_name, m1_higher_is_better, m2_higher_is_better, readable_label)
TRIPLET_DEFS = [
    ('AvgSpeed',   'Throughput', True,  True,  'Speed+TP+Halt'),
    ('AvgSpeed',   'SpeedStd',   True,  True,  'Speed+Std+Halt'),
    ('Throughput', 'SpeedStd',   True,  True,  'TP+Std+Halt'),
    ('AvgSpeed',   'TotalCO2',   True,  False, 'Speed+CO2+Halt'),
    ('Throughput', 'TotalCO2',   True,  False, 'TP+CO2+Halt'),
]


def analyze_per_junction(output_dir):
    """
    Per-junction elemzés: AvgSpeed + Throughput log-tanh normalizáció.

    Minden junction-re kiszámolja a lokális MU/STD paramétereket (median/IQR
    alapú robusztus becslés), majd összehasonlítja a globális paraméterekkel.

    Reward: calc_reward_log_tanh(AvgSpeed, Throughput, mu_s, std_s, mu_t, std_t)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("  PER-JUNCTION NORMALIZACIOS PARAMETER KALIBRACIO")
    print("  Reward: AvgSpeed + Throughput | Normalizacio: log-tanh")
    print("=" * 80)

    # --- CSV-k betöltése ---
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]
    if not csv_files:
        print("[ERROR] Nincsenek CSV fajlok! Futtasd elobb a szimulaciot.")
        return

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        # Ellenőrzés: szükséges oszlopok megléte
        if 'AvgSpeed' not in df.columns or 'Throughput' not in df.columns:
            print(f"  [SKIP] {csv_file} — hiányzó AvgSpeed/Throughput oszlop")
            continue
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        jid = csv_file.split('_flow')[0]
        df['junction'] = jid
        if 'control_mode' not in df.columns:
            if '_random.csv' in csv_file:
                df['control_mode'] = 'random'
            elif '_actuated.csv' in csv_file:
                df['control_mode'] = 'actuated'
            else:
                df['control_mode'] = 'random'
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    junction_ids = sorted(full_df['junction'].unique())

    print(f"\nOsszes adatpont: {len(full_df)}")
    print(f"Junction-ok szama: {len(junction_ids)}")
    print(f"Junction-ok: {', '.join(junction_ids)}")
    if 'control_mode' in full_df.columns:
        mode_counts = full_df['control_mode'].value_counts()
        print(f"Control mode eloszlas:")
        for mode, cnt in mode_counts.items():
            pct = 100.0 * cnt / len(full_df)
            print(f"  {mode}: {cnt} adatpont ({pct:.1f}%)")

    # --- Robusztus paraméter becslés (median/IQR) ---
    def robust_log_params(values, min_std=0.1):
        """Log-transzformált median és IQR-alapú STD becslés."""
        pos = values[values > 0]
        if len(pos) < 10:
            return None, None
        log_vals = np.log(pos + 1e-5)
        mu = float(np.median(log_vals))
        iqr = float(np.percentile(log_vals, 75) - np.percentile(log_vals, 25))
        std = max(iqr / 1.3489, min_std)  # IQR/1.3489 ≈ STD normális eloszlásnál
        return mu, std

    # --- Globális paraméterek kiszámítása (minden junction aggregálva) ---
    valid_global = full_df[full_df['VehCount'] > 0].copy()
    glob_mu_s, glob_std_s = robust_log_params(valid_global['AvgSpeed'].values)
    glob_mu_t, glob_std_t = robust_log_params(valid_global['Throughput'].values)

    if glob_mu_s is None or glob_mu_t is None:
        print("[ERROR] Nem eleg globalis adat a parameterbecsleshez!")
        return

    # GLOBAL_PARAMS frissítése
    GLOBAL_PARAMS['MU_SPEED'] = round(glob_mu_s, 6)
    GLOBAL_PARAMS['STD_SPEED'] = round(glob_std_s, 6)
    GLOBAL_PARAMS['MU_THROUGHPUT'] = round(glob_mu_t, 6)
    GLOBAL_PARAMS['STD_THROUGHPUT'] = round(glob_std_t, 6)

    print(f"\n  Globalis parameterek (log-space, median/IQR):")
    print(f"    MU_SPEED:      {glob_mu_s:.6f}")
    print(f"    STD_SPEED:     {glob_std_s:.6f}")
    print(f"    MU_THROUGHPUT: {glob_mu_t:.6f}")
    print(f"    STD_THROUGHPUT:{glob_std_t:.6f}")

    # --- Per-junction paraméterek ---
    junction_params = {}
    summary_rows = []

    print(f"\n{'Junction':15} {'MU_SPD':>10} {'STD_SPD':>10} {'MU_THR':>10} {'STD_THR':>10} "
          f"{'dMU_S':>8} {'dMU_T':>8} {'N':>8}")
    print("-" * 95)

    for jid in junction_ids:
        jdf = full_df[full_df['junction'] == jid].copy()
        jdf_valid = jdf[jdf['VehCount'] > 0].copy()

        if len(jdf_valid) == 0:
            print(f"{jid:15} {'SKIP — no data':>50}")
            continue

        # AvgSpeed paraméterek
        mu_s, std_s = robust_log_params(jdf_valid['AvgSpeed'].values)
        if mu_s is None:
            mu_s, std_s = glob_mu_s, glob_std_s

        # Throughput paraméterek
        mu_t, std_t = robust_log_params(jdf_valid['Throughput'].values)
        if mu_t is None:
            mu_t, std_t = glob_mu_t, glob_std_t

        junction_params[jid] = {
            'MU_SPEED': round(mu_s, 6),
            'STD_SPEED': round(std_s, 6),
            'MU_THROUGHPUT': round(mu_t, 6),
            'STD_THROUGHPUT': round(std_t, 6),
        }

        d_mu_s = mu_s - glob_mu_s
        d_mu_t = mu_t - glob_mu_t

        print(f"{jid:15} {mu_s:10.4f} {std_s:10.4f} {mu_t:10.4f} {std_t:10.4f} "
              f"{d_mu_s:+8.4f} {d_mu_t:+8.4f} {len(jdf_valid):8d}")

        # --- Reward összehasonlítás: global vs local params ---
        speed_vals = jdf_valid['AvgSpeed'].values
        thr_vals = jdf_valid['Throughput'].values
        mask_pos = (speed_vals > 0) & (thr_vals > 0)
        spd_pos = speed_vals[mask_pos]
        thr_pos = thr_vals[mask_pos]

        if len(spd_pos) > 10:
            # HaltRatio ugyanolyan maszkkal
            halt_pos = jdf_valid['HaltRatio'].values[mask_pos] if 'HaltRatio' in jdf_valid.columns else np.zeros(len(spd_pos))

            # --- Variáns 1: plain (reward hacking védelem NÉLKÜL) ---
            rewards_global_plain = calc_reward_log_tanh(
                spd_pos, thr_pos, glob_mu_s, glob_std_s, glob_mu_t, glob_std_t)
            rewards_local_plain = calc_reward_log_tanh(
                spd_pos, thr_pos, mu_s, std_s, mu_t, std_t)

            # --- Variáns 2: halt-protected multiplikatív (reward hacking védelemmel) ---
            rewards_global_halt = calc_reward_log_tanh_with_halt(
                spd_pos, thr_pos, halt_pos, glob_mu_s, glob_std_s, glob_mu_t, glob_std_t)
            rewards_local_halt = calc_reward_log_tanh_with_halt(
                spd_pos, thr_pos, halt_pos, mu_s, std_s, mu_t, std_t)

            # --- Variáns 3: AvgSpeed + Throughput + HaltRatio additív triplet ---
            rewards_global_triplet = calc_reward_triplet(
                spd_pos, thr_pos, halt_pos,
                glob_mu_s, glob_std_s, glob_mu_t, glob_std_t)
            rewards_local_triplet = calc_reward_triplet(
                spd_pos, thr_pos, halt_pos,
                mu_s, std_s, mu_t, std_t)

            def _stats(arr):
                return {
                    'mean':  float(np.mean(arr)),
                    'std':   float(np.std(arr)),
                    'iqr':   float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                    'range': float(np.percentile(arr, 90) - np.percentile(arr, 10)),
                }

            sg  = _stats(rewards_global_plain)
            sl  = _stats(rewards_local_plain)
            sgh = _stats(rewards_global_halt)
            slh = _stats(rewards_local_halt)
            sgt = _stats(rewards_global_triplet)
            slt = _stats(rewards_local_triplet)

            summary_rows.append({
                'junction': jid,
                'MU_SPEED': mu_s, 'STD_SPEED': std_s,
                'MU_THROUGHPUT': mu_t, 'STD_THROUGHPUT': std_t,
                'dMU_SPEED': d_mu_s, 'dMU_THROUGHPUT': d_mu_t,
                'N_valid': len(jdf_valid),
                'median_speed': float(np.median(spd_pos)),
                'median_throughput': float(np.median(thr_pos)),
                # Variáns 1: Plain
                'reward_global_mean':  sg['mean'],  'reward_global_std':  sg['std'],
                'reward_global_iqr':   sg['iqr'],   'reward_global_range': sg['range'],
                'reward_local_mean':   sl['mean'],  'reward_local_std':   sl['std'],
                'reward_local_iqr':    sl['iqr'],   'reward_local_range':  sl['range'],
                # Variáns 2: Halt-protected (multiplikatív)
                'reward_global_halt_mean':  sgh['mean'], 'reward_global_halt_std':  sgh['std'],
                'reward_global_halt_iqr':   sgh['iqr'],  'reward_global_halt_range': sgh['range'],
                'reward_local_halt_mean':   slh['mean'], 'reward_local_halt_std':   slh['std'],
                'reward_local_halt_iqr':    slh['iqr'],  'reward_local_halt_range':  slh['range'],
                # Variáns 3: Triplet additív (AvgSpeed + Throughput + HaltRatio)
                'reward_global_triplet_mean':  sgt['mean'], 'reward_global_triplet_std':  sgt['std'],
                'reward_global_triplet_iqr':   sgt['iqr'],  'reward_global_triplet_range': sgt['range'],
                'reward_local_triplet_mean':   slt['mean'], 'reward_local_triplet_std':   slt['std'],
                'reward_local_triplet_iqr':    slt['iqr'],  'reward_local_triplet_range':  slt['range'],
            })

    # --- JSON kiírás ---
    # A JSON formátum: globális + per-junction paraméterek
    output_json = {
        'reward_function': 'AvgSpeed + Throughput, log-tanh normalization',
        'global': {
            'MU_SPEED': round(glob_mu_s, 6),
            'STD_SPEED': round(glob_std_s, 6),
            'MU_THROUGHPUT': round(glob_mu_t, 6),
            'STD_THROUGHPUT': round(glob_std_t, 6),
        },
        'per_junction': junction_params,
    }
    json_path = os.path.join(DATA_DIR, "junction_reward_params.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"  junction_reward_params.json mentve: {json_path}")
    print(f"  {len(junction_params)} junction parameterei")
    print(f"{'=' * 80}")

    json_copy = os.path.join(output_dir, "junction_reward_params.json")
    with open(json_copy, 'w') as f:
        json.dump(output_json, f, indent=2)

    # --- Summary CSV ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, 'per_junction_summary.csv'), index=False)
        print(f"  per_junction_summary.csv mentve")

    # --- Ábra: junction_comparison.png — kiválasztott variáns × Global/Local + MU ---
    if summary_rows:
        sdf = pd.DataFrame(summary_rows).set_index('junction')
        jids = sdf.index.tolist()
        x = np.arange(len(jids))

        # Variáns definíciók: (label, range_col, iqr_col, color_glob, color_loc, hatch)
        VARIANT_DEFS = {
            'plain':   {
                'glob': ('Global — Plain',       'reward_global_range',           'reward_global_iqr',           '#3498db', '', 0.90),
                'loc':  ('Local — Plain',         'reward_local_range',            'reward_local_iqr',            '#e74c3c', '', 0.90),
                'imp_loc': 'reward_local_range', 'imp_glob': 'reward_global_range',
                'title_suffix': 'Plain (Speed+TP)/2',
            },
            'halt':    {
                'glob': ('Global — Halt×mult',   'reward_global_halt_range',      'reward_global_halt_iqr',      '#3498db', '//', 0.55),
                'loc':  ('Local — Halt×mult',    'reward_local_halt_range',       'reward_local_halt_iqr',       '#e74c3c', '//', 0.55),
                'imp_loc': 'reward_local_halt_range', 'imp_glob': 'reward_global_halt_range',
                'title_suffix': 'Halt×mult (base × (1 − halt×0.8))',
            },
            'triplet': {
                'glob': ('Global — Triplet+halt', 'reward_global_triplet_range',  'reward_global_triplet_iqr',   '#8e44ad', '', 0.85),
                'loc':  ('Local — Triplet+halt',  'reward_local_triplet_range',   'reward_local_triplet_iqr',    '#d35400', '', 0.85),
                'imp_loc': 'reward_local_triplet_range', 'imp_glob': 'reward_global_triplet_range',
                'title_suffix': 'Triplet-additív (Speed+TP+Halt)/3',
            },
        }

        # Variáns kiválasztása (PLOT_REWARD_VARIANT config, vagy auto: legjobb local IQR)
        iqr_cols = {
            'plain':   'reward_local_iqr',
            'halt':    'reward_local_halt_iqr',
            'triplet': 'reward_local_triplet_iqr',
        }
        if PLOT_REWARD_VARIANT and PLOT_REWARD_VARIANT in VARIANT_DEFS:
            selected_variant = PLOT_REWARD_VARIANT
        else:
            # Auto-select: legjobb mean IQR lokálisan
            available = [v for v in iqr_cols if iqr_cols[v] in sdf.columns]
            if available:
                selected_variant = max(available, key=lambda v: sdf[iqr_cols[v]].mean())
            else:
                selected_variant = 'plain'
        print(f"  junction_comparison.png variáns: {selected_variant}")

        vd = VARIANT_DEFS[selected_variant]
        glob_v = vd['glob']   # (label, range_col, iqr_col, color, hatch, alpha)
        loc_v  = vd['loc']

        width = 0.30
        offsets = [-0.5, 0.5]  # global, local

        fig, axes = plt.subplots(4, 1, figsize=(max(18, len(jids) * 1.1), 24))
        fig.suptitle(
            f'Per-Junction Normalization: {vd["title_suffix"]}  |  Global vs Local',
            fontsize=13, fontweight='bold', y=0.995)
        fig.subplots_adjust(top=0.965, hspace=0.55)

        plot_variants = [glob_v, loc_v]

        # Panel 1: MU eltérések
        ax = axes[0]
        bar_w = 0.35
        ax.bar(x - bar_w/2, sdf['dMU_SPEED'].values, bar_w,
               label=r'$\Delta$ MU_SPEED', color='#2980b9', alpha=0.85)
        ax.bar(x + bar_w/2, sdf['dMU_THROUGHPUT'].values, bar_w,
               label=r'$\Delta$ MU_THROUGHPUT', color='#e67e22', alpha=0.85)
        ax.axhline(0, color='black', ls='-', lw=1)
        ax.set_xticks(x); ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Deviation from Global (log-space)')
        ax.set_title('MU Parameter Deviation (Local − Global)', fontsize=11, pad=6)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

        # Panel 2: Reward Range (p90-p10)
        ax = axes[1]
        for off, (label, rng_col, _, color, hatch, alpha) in zip(offsets, plot_variants):
            ax.bar(x + off*width, sdf[rng_col].values, width,
                   label=label, color=color, hatch=hatch, alpha=alpha)
        ax.set_xticks(x); ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reward Range (p90 − p10)')
        ax.set_title(f'Reward Dynamic Range — {selected_variant}', fontsize=11, pad=6)
        ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3, axis='y')

        # Panel 3: Reward IQR
        ax = axes[2]
        for off, (label, _, iqr_col, color, hatch, alpha) in zip(offsets, plot_variants):
            ax.bar(x + off*width, sdf[iqr_col].values, width,
                   label=label, color=color, hatch=hatch, alpha=alpha)
        ax.set_xticks(x); ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reward IQR (p75 − p25)')
        ax.set_title(f'Reward IQR — {selected_variant}', fontsize=11, pad=6)
        ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3, axis='y')

        # Panel 4: Local improvement %
        ax = axes[3]
        loc_col  = vd['imp_loc']
        glob_col = vd['imp_glob']
        imp = ((sdf[loc_col] / sdf[glob_col].clip(lower=1e-6)) - 1) * 100
        c_imp = ['#2ecc71' if v > 0 else '#c0392b' for v in imp.values]
        ax.bar(x, imp.values, 0.55, color=c_imp, alpha=0.85, label=selected_variant)
        for i, v in enumerate(imp.values):
            ax.text(i, v + (0.5 if v >= 0 else -2), f'{v:+.0f}%',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=6, fontweight='bold')
        ax.axhline(0, color='black', ls='-', lw=1)
        ax.set_xticks(x); ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Local vs Global improvement (%)')
        ax.set_title(f'Local Normalization Improvement — {selected_variant}', fontsize=11, pad=6)
        ax.grid(True, alpha=0.3, axis='y')

        fig.savefig(os.path.join(output_dir, 'junction_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  junction_comparison.png mentve ({selected_variant} variáns)")

    # --- Összefoglaló statisztikák ---
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        print(f"\n{'=' * 80}")
        print("  OSSZEFOGLALO STATISZTIKAK")
        print(f"{'=' * 80}")

        print(f"\n  MU_SPEED elteresek a globalitol:")
        print(f"    Min:  {sdf['dMU_SPEED'].min():+.4f}  ({sdf.loc[sdf['dMU_SPEED'].idxmin(), 'junction']})")
        print(f"    Max:  {sdf['dMU_SPEED'].max():+.4f}  ({sdf.loc[sdf['dMU_SPEED'].idxmax(), 'junction']})")
        print(f"    Mean: {sdf['dMU_SPEED'].mean():+.4f}")
        print(f"    Std:  {sdf['dMU_SPEED'].std():.4f}")

        print(f"\n  MU_THROUGHPUT elteresek a globalitol:")
        print(f"    Min:  {sdf['dMU_THROUGHPUT'].min():+.4f}  ({sdf.loc[sdf['dMU_THROUGHPUT'].idxmin(), 'junction']})")
        print(f"    Max:  {sdf['dMU_THROUGHPUT'].max():+.4f}  ({sdf.loc[sdf['dMU_THROUGHPUT'].idxmax(), 'junction']})")
        print(f"    Mean: {sdf['dMU_THROUGHPUT'].mean():+.4f}")
        print(f"    Std:  {sdf['dMU_THROUGHPUT'].std():.4f}")

        for variant, rng_col, iqr_col in [
            ('Plain',              'reward_local_range',          'reward_local_iqr'),
            ('Halt×mult',          'reward_local_halt_range',     'reward_local_halt_iqr'),
            ('Triplet+halt (add)', 'reward_local_triplet_range',  'reward_local_triplet_iqr'),
        ]:
            glob_rng = rng_col.replace('local', 'global')
            glob_iqr = iqr_col.replace('local', 'global')
            print(f"\n  [{variant}] Reward range javulas (local vs global):")
            improved = (sdf[rng_col] > sdf[glob_rng]).sum()
            total = len(sdf)
            print(f"    Javult: {improved}/{total} junction")
            print(f"    Atlagos global range: {sdf[glob_rng].mean():.4f}")
            print(f"    Atlagos local range:  {sdf[rng_col].mean():.4f}")
            pct = (sdf[rng_col].mean() / sdf[glob_rng].mean() - 1) * 100
            print(f"    Valtozas: {pct:+.1f}%")

            print(f"\n  [{variant}] Reward IQR javulas (local vs global):")
            improved_iqr = (sdf[iqr_col] > sdf[glob_iqr]).sum()
            print(f"    Javult: {improved_iqr}/{total} junction")
            print(f"    Atlagos global IQR: {sdf[glob_iqr].mean():.4f}")
            print(f"    Atlagos local IQR:  {sdf[iqr_col].mean():.4f}")
            pct_iqr = (sdf[iqr_col].mean() / sdf[glob_iqr].mean() - 1) * 100
            print(f"    Valtozas: {pct_iqr:+.1f}%")



def generate_reward_step_curve(output_dir, junction_filter=None):
    """
    WandB-stílusú step-szintű reward görbe (Speed+TP, log-tanh normálás).

    Ha junction_filter meg van adva (pl. "R1C1_C"), csak azt a junction-t plotolja
    LOKÁLIS normalizációs paraméterekkel.
    Ha junction_filter üres/None → minden junction, globális paraméterekkel.

    Önállóan is futtatható: python metric_collection_per_junction.py --reward-curve-only
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if junction_filter is None:
        junction_filter = REWARD_CURVE_JUNCTION  # config default

    csv_files = sorted([f for f in os.listdir(output_dir)
                        if f.endswith('.csv') and '_flow' in f and '_ep' in f])
    if not csv_files:
        print("  [SKIP] Nincs CSV fájl a step curve generálásához.")
        return

    # Paraméterek betöltése
    params_path = os.path.join(DATA_DIR, "junction_reward_params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(output_dir, "junction_reward_params.json")
    junction_params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            junction_params = json.load(f)

    glob_p   = junction_params.get('global', {})
    local_p  = junction_params.get('per_junction', {})

    # Ha junction_filter meg van adva, lokális paramétereket használunk
    if junction_filter and junction_filter in local_p:
        params = local_p[junction_filter]
        param_label = f"lokális ({junction_filter})"
    else:
        params = glob_p
        param_label = "globális"

    mu_s  = params.get('MU_SPEED',      glob_p.get('MU_SPEED',      2.0))
    std_s = params.get('STD_SPEED',     glob_p.get('STD_SPEED',     1.0))
    mu_t  = params.get('MU_THROUGHPUT', glob_p.get('MU_THROUGHPUT', 3.0))
    std_t = params.get('STD_THROUGHPUT',glob_p.get('STD_THROUGHPUT',1.0))

    # CSV beolvasás
    all_data = []
    for csv_file in csv_files:
        jid = csv_file.split('_flow')[0]
        if junction_filter and jid != junction_filter:
            continue
        try:
            df = pd.read_csv(os.path.join(output_dir, csv_file))
        except Exception:
            continue
        if 'AvgSpeed' not in df.columns or 'Throughput' not in df.columns:
            continue
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        ep = int(csv_file.split('_ep')[1].split('_')[0]) if '_ep' in csv_file else 0
        mode = 'actuated' if '_actuated.csv' in csv_file else 'random'
        df['junction']     = jid
        df['flow_level']   = flow_level
        df['episode']      = ep
        df['control_mode'] = df['control_mode'] if 'control_mode' in df.columns else mode
        df['run_id']       = f"{jid}_flow{flow_level}_ep{ep}_{mode}"
        all_data.append(df)

    if not all_data:
        print(f"  [SKIP] Nincs adat junction_filter='{junction_filter}' esetén.")
        return

    full = pd.concat(all_data, ignore_index=True)
    full = full[full['VehCount'] > 0].copy()

    epsilon = 1e-5
    spd = full['AvgSpeed'].values
    thr = full['Throughput'].values
    r_s = (1 + np.tanh((np.log(spd.clip(min=epsilon)) - mu_s)  / (std_s  + 1e-9))) / 2
    r_t = (1 + np.tanh((np.log(thr.clip(min=epsilon)) - mu_t)  / (std_t  + 1e-9))) / 2
    full['reward_plain_step'] = (r_s + r_t) / 2

    run_ids     = sorted(full['run_id'].unique())
    n_runs      = len(run_ids)
    flow_levels = sorted(full['flow_level'].unique())
    flow_norm   = {fl: i / max(len(flow_levels) - 1, 1) for i, fl in enumerate(flow_levels)}
    cmap        = plt.cm.get_cmap('RdYlBu_r')
    EMA_ALPHA   = 0.05

    fig, ax = plt.subplots(figsize=(16, 6))
    all_smoothed = []

    for run_id in run_ids:
        rd = full[full['run_id'] == run_id].sort_values('step')
        if len(rd) < 5:
            continue
        steps  = rd['step'].values
        reward = rd['reward_plain_step'].values
        color  = cmap(flow_norm[rd['flow_level'].iloc[0]])

        ax.plot(steps, reward, color=color, alpha=0.18, linewidth=0.7)

        ema = np.zeros_like(reward, dtype=float)
        ema[0] = reward[0]
        for i in range(1, len(reward)):
            ema[i] = EMA_ALPHA * reward[i] + (1 - EMA_ALPHA) * ema[i - 1]
        ax.plot(steps, ema, color=color, alpha=0.55, linewidth=1.2)
        all_smoothed.append((steps, ema))

    if all_smoothed:
        max_step     = max(s[-1] for s, _ in all_smoothed)
        common_steps = np.arange(0, int(max_step) + 1)
        interp_rows  = [np.interp(common_steps, s, e) for s, e in all_smoothed if len(s) > 1]
        if interp_rows:
            gm = np.mean(interp_rows, axis=0)
            ema_g = np.zeros_like(gm)
            ema_g[0] = gm[0]
            for i in range(1, len(gm)):
                ema_g[i] = EMA_ALPHA * gm[i] + (1 - EMA_ALPHA) * ema_g[i - 1]
            ax.plot(common_steps, ema_g, color='black', linewidth=2.5,
                    label=f'Átlag (n={len(interp_rows)} run)', zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=min(flow_levels), vmax=max(flow_levels)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label('Flow level (veh/h)', fontsize=10)

    title_jid = junction_filter if junction_filter else 'összes junction'
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Normalized reward  [0 – 1]', fontsize=12)
    ax.set_title(
        f'Step-szintű reward — Speed + Throughput, log-tanh  |  {title_jid}  |  {param_label} params\n'
        f'μ_s={mu_s:.3f}  σ_s={std_s:.3f}  |  μ_tp={mu_t:.3f}  σ_tp={std_t:.3f}  |  '
        f'{n_runs} run  |  EMA α={EMA_ALPHA}',
        fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'reward_step_curve.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  reward_step_curve.png mentve  ({n_runs} run, {title_jid}, {param_label})")


def reward_selection_analysis(output_dir):
    """
    Reward metrika és normalizáció kiválasztás — tudományos módszertan.

    Szűrőkritériumok (bármelyik FAIL → kiesik):
      1. MONOTONITÁS: Spearman(flow_level, reward) szignifikánsan negatív
      2. REDUNDANCIA: |Pearson(metrika_A, metrika_B)| < 0.85
      3. KOMPRESSZIÓ: reward IQR > 0.10

    Rangsorolás az átmenő jelöltek között:
      - One-way ANOVA η² (eta-squared): a flow_level által magyarázott
        variancia aránya. Alacsonyabb η² = a forgalmi szint kevésbé
        dominálja a reward-ot → az ágens akciójának hatása jobban látszik.

    Normalizációs módszerek összehasonlítása:
      - Ugyanezek a kritériumok, de módszerenként alkalmazva.
    """
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import itertools

    print("\n" + "=" * 100)
    print("  REWARD METRIKA ÉS NORMALIZÁCIÓ KIVÁLASZTÁS")
    print("=" * 100)

    # --- CSV-k betöltése ---
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]
    if not csv_files:
        print("[ERROR] Nincsenek CSV fajlok!")
        return
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        jid = csv_file.split('_flow')[0]
        df['junction'] = jid
        # Epizód száma a fájlnévből (pl. R1C1_C_flow1000_ep2_actuated.csv → ep=2)
        ep = int(csv_file.split('_ep')[1].split('_')[0]) if '_ep' in csv_file else 0
        df['episode'] = ep
        # Kontrol mód (ha nincs oszlop)
        if 'control_mode' not in df.columns:
            df['control_mode'] = 'actuated' if '_actuated.csv' in csv_file else 'random'
        # Egyedi run azonosító
        df['run_id'] = f"{jid}_flow{flow_level}_ep{ep}_{df['control_mode'].iloc[0]}"
        all_data.append(df)
    full_df = pd.concat(all_data, ignore_index=True)
    junction_ids = sorted(full_df['junction'].unique())

    params_path = os.path.join(DATA_DIR, "junction_reward_params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(output_dir, "junction_reward_params.json")
    with open(params_path) as f:
        junction_params = json.load(f)

    # =====================================================================
    # HELPER: normalizációs függvények
    # =====================================================================
    # =================================================================
    # NORMALIZÁCIÓS MÓDSZEREK
    #
    # Két kategória:
    #   A) PARAMETRIKUS CDF: feltételezünk egy eloszlást, illesztjük,
    #      R = 1 - CDF(x). Statisztikailag megalapozott.
    #      Referencia: D'Agostino & Stephens (1986) "Goodness-of-fit techniques"
    #
    #   B) NEM-PARAMETRIKUS: nincs eloszlási feltételezés.
    #      - Empirikus CDF: R = 1 - F̂(x), ahol F̂ a tapasztalati CDF
    #        Referencia: van der Vaart (1998) "Asymptotic Statistics"
    #      - Min-max: R = 1 - (x - min) / (max - min)
    #        Referencia: Han, Kamber & Pei (2011) "Data Mining" ch. 3
    #
    #   C) TRANSZFORMÁCIÓ + SQUASH: transzformálunk (log, sqrt, Box-Cox)
    #      majd sigmoid/tanh-al [0,1]-be nyomjuk.
    #      - Box-Cox: Box & Cox (1964) JRSS-B, 26(2), 211-252
    #      - Log-sigmoid: gyakori RL reward shaping, pl. Ng et al. (1999)
    #
    # Mind a mu, std paramétert kapja (log-térben), és vals-t (nyers).
    # =================================================================

    # --- A) Parametrikus CDF ---
    def normalize_lognormal_cdf(vals, mu, std):
        """Lognormal CDF: R = 1 - Phi((log(x) - mu) / std)
        Feltételezés: X ~ LogNormal(mu, std)
        Ref: limpert2001 'Log-normal distributions across the sciences'"""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return 1.0 - stats.norm.cdf(z)

    def normalize_gamma_cdf(vals, mu, std):
        """Gamma CDF: R = 1 - F_gamma(x; alpha, beta)
        Illesztés: method of moments, alpha = (mean/std)^2, beta = mean/std^2
        Ref: Forbes et al. (2011) 'Statistical Distributions'"""
        v = vals.clip(min=1e-5)
        m, s = np.mean(v), np.std(v)
        if s < 1e-9:
            return np.full_like(vals, 0.5)
        alpha = (m / s) ** 2
        beta = s ** 2 / m  # scale parameter
        return 1.0 - stats.gamma.cdf(v, a=alpha, scale=beta)

    def normalize_weibull_cdf(vals, mu, std):
        """Weibull CDF: R = exp(-(x/lambda)^k)
        Illesztés: method of moments approximáció
        Ref: Rinne (2008) 'The Weibull Distribution'"""
        v = vals.clip(min=1e-5)
        m, s = np.mean(v), np.std(v)
        if s < 1e-9 or m < 1e-9:
            return np.full_like(vals, 0.5)
        cv = s / m  # coefficient of variation
        # k approximáció: Justus et al. (1978)
        k = max(0.5, (cv) ** (-1.086))
        from math import lgamma
        lam = m / (np.exp(lgamma(1 + 1.0/k)) if k > 0 else 1.0)
        lam = max(lam, 1e-5)
        return np.exp(-np.power(v / lam, k))

    # --- B) Nem-parametrikus ---
    def normalize_empirical_cdf(vals, mu, std):
        """Empirikus CDF: R = 1 - rank(x) / N
        Nem feltételez semmilyen eloszlást.
        Ref: van der Vaart (1998) 'Asymptotic Statistics'"""
        from scipy.stats import rankdata
        ranks = rankdata(vals, method='average')
        return 1.0 - ranks / (len(vals) + 1)  # +1: Hazen plotting position

    def normalize_minmax(vals, mu, std):
        """Min-max: R = 1 - (x - min) / (max - min)
        Ref: Han, Kamber & Pei (2011) 'Data Mining' ch. 3"""
        v_min, v_max = np.min(vals), np.max(vals)
        if v_max - v_min < 1e-9:
            return np.full_like(vals, 0.5)
        return 1.0 - (vals - v_min) / (v_max - v_min)

    # --- C) Transzformáció + squash ---
    def normalize_log_sigmoid(vals, mu, std):
        """Log-sigmoid: R = 1 - sigma((log(x) - mu) / std)
        Log transzformáció + logisztikus squash.
        Ref: reward shaping irodalom, pl. Ng et al. (1999) ICML"""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    def normalize_boxcox_sigmoid(vals, mu, std):
        """Box-Cox + sigmoid: optimális lambda keresés, majd z-score + sigmoid.
        Ref: Box & Cox (1964) JRSS-B 26(2):211-252"""
        v = vals.clip(min=1e-5)
        # Box-Cox lambda keresés (scipy)
        try:
            transformed, lam = stats.boxcox(v)
        except Exception:
            # Fallback: log ha boxcox nem sikerül
            transformed = np.log(v + 1e-5)
        mu_bc = np.mean(transformed)
        std_bc = np.std(transformed)
        z = (transformed - mu_bc) / (std_bc + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    def normalize_log_tanh(vals, mu, std):
        """Log-tanh: R = (1 - tanh((log(x) - mu) / std)) / 2
        Steeper transition mint sigmoid — gyorsabban szaturál."""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return (1.0 - np.tanh(z)) / 2.0

    # --- D) RL-specifikus módszerek ---
    def normalize_zscore(vals, mu, std):
        """Z-score standardizálás + sigmoid squash.
        z = (x - mean) / std, majd sigmoid([0,1]-be).
        Az RL-ben a running mean/std-vel frissített verzió az alap
        (Welford-algoritmus), itt a batch statisztikát használjuk.
        Ref: Welford (1962), Sutton & Barto (2018) ch. 2.4"""
        v = vals.clip(min=1e-5)
        m, s = np.mean(v), np.std(v)
        z = (v - m) / (s + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    def normalize_popart(vals, mu, std):
        """PopArt: Preserving Outputs Precisely, while Adaptively
        Rescaling Targets.
        Adaptív normalizáció ami a reward skálát követi.
        Batch esetben: z = (x - mu) / sigma, klippelve [-5, 5],
        majd lineárisan [0, 1]-re.
        Ref: van Hasselt et al. (2016) 'Learning values across many
        orders of magnitude' arXiv:1602.07714"""
        v = vals.clip(min=1e-5)
        m, s = np.mean(v), np.std(v)
        z = (v - m) / (s + 1e-9)
        z_clipped = np.clip(z, -5.0, 5.0)
        # Lineáris leképezés [-5, 5] → [0, 1], invertálva (alacsonyabb = jobb)
        return 1.0 - (z_clipped + 5.0) / 10.0

    def normalize_percentile_clip(vals, mu, std):
        """Percentilis-alapú klippelés: robusztus min-max.
        A szélső 2.5%-ot levágja (p2.5, p97.5), majd lineárisan [0,1]-re.
        Robusztus kiugró értékekre.
        Ref: Gyakori practice RL-ben, pl. OpenAI Baselines
        VecNormalize (Dhariwal et al. 2017)"""
        p_low = np.percentile(vals, 2.5)
        p_high = np.percentile(vals, 97.5)
        if p_high - p_low < 1e-9:
            return np.full_like(vals, 0.5)
        clipped = np.clip(vals, p_low, p_high)
        return 1.0 - (clipped - p_low) / (p_high - p_low)

    NORM_METHODS = {
        # Parametrikus CDF
        'lognormal-cdf':  normalize_lognormal_cdf,
        'gamma-cdf':      normalize_gamma_cdf,
        'weibull-cdf':    normalize_weibull_cdf,
        # Nem-parametrikus
        'empirical-cdf':  normalize_empirical_cdf,
        'min-max':        normalize_minmax,
        # Transzformáció + squash
        'log-sigmoid':    normalize_log_sigmoid,
        'boxcox-sigmoid': normalize_boxcox_sigmoid,
        'log-tanh':       normalize_log_tanh,
        # RL-specifikus
        'zscore-sigmoid': normalize_zscore,
        'popart':         normalize_popart,
        'percentile-clip': normalize_percentile_clip,
    }

    # =====================================================================
    # 1. KORRELÁCIÓ — REDUNDANCIA SZŰRÉS
    # =====================================================================
    print("\n" + "=" * 100)
    print("  1. KORRELÁCIÓ — REDUNDANCIA SZŰRÉS")
    print("     Pearson |r| > 0.85 → redundáns, nem érdemes együtt használni")
    print("=" * 100)

    # Egyedi metrikák amiket vizsgálunk (nem per-vehicle, mert azok veszélyesek)
    candidate_metrics_all = ['TotalWaitingTime', 'TotalCO2', 'AvgSpeed', 'QueueLength',
                             'TotalTravelTime', 'AvgOccupancy',
                             'Throughput', 'SpeedStd', 'HaltRatio']
    # Csak azokat tartjuk meg, amik ténylegesen vannak az adatban
    candidate_metrics = [m for m in candidate_metrics_all if m in full_df.columns]

    # Globális korreláció (log-transzformált, VehCount > 0 szűrés)
    df_valid = full_df[full_df['VehCount'] > 0].copy()
    epsilon = 1e-5

    # --- Kombinált reward oszlopok hozzáadása (mindkét variáns) ---
    glob = junction_params.get('global', {})
    _mu_s  = glob.get('MU_SPEED', 0.617003)
    _std_s = glob.get('STD_SPEED', 0.951352)
    _mu_t  = glob.get('MU_THROUGHPUT', 2.995733)
    _std_t = glob.get('STD_THROUGHPUT', 0.814450)

    if 'AvgSpeed' in df_valid.columns and 'Throughput' in df_valid.columns:
        spd  = df_valid['AvgSpeed'].values
        thr  = df_valid['Throughput'].values
        halt = df_valid['HaltRatio'].values if 'HaltRatio' in df_valid.columns else np.zeros(len(spd))

        # 2-metrikás variánsok
        df_valid['reward_plain'] = calc_reward_log_tanh(spd, thr, _mu_s, _std_s, _mu_t, _std_t)
        df_valid['reward_halt']  = calc_reward_log_tanh_with_halt(spd, thr, halt, _mu_s, _std_s, _mu_t, _std_t)
        new_cols = ['reward_plain', 'reward_halt']

        # 3-metrikás (triplet) variánsok — TRIPLET_DEFS alapján, 3. mindig HaltRatio
        def _log_params(vals):
            lv = np.log(np.clip(vals[vals > 0], 1e-5, None) + 1e-5)
            mu = float(np.median(lv)) if len(lv) > 0 else 0.0
            q75, q25 = np.percentile(lv, [75, 25]) if len(lv) > 0 else (1, -1)
            std = max(0.1, (q75 - q25) / 1.349)
            return mu, std

        for (m1_name, m2_name, m1_hib, m2_hib, label) in TRIPLET_DEFS:
            if m1_name not in df_valid.columns or m2_name not in df_valid.columns:
                continue
            m1_vals = df_valid[m1_name].values
            m2_vals = df_valid[m2_name].values
            mu1, std1 = _log_params(m1_vals)
            mu2, std2 = _log_params(m2_vals)

            col_name = f'triplet_{label}'   # pl. "triplet_Speed+TP+Halt"
            df_valid[col_name] = calc_reward_triplet(
                m1_vals, m2_vals, halt,
                mu1, std1, mu2, std2,
                higher_is_better_m1=m1_hib,
                higher_is_better_m2=m2_hib)
            new_cols.append(col_name)

        # Hozzáadjuk a jelöltek listájához
        candidate_metrics_all.extend(new_cols)
        candidate_metrics = [m for m in candidate_metrics_all if m in df_valid.columns]

    corr_data = {}
    for col in candidate_metrics:
        vals = df_valid[col].values
        # AvgSpeed invertálva (magasabb = jobb, de reward-ban alacsonyabb kell)
        if col == 'AvgSpeed':
            corr_data[col] = np.log(vals.clip(min=epsilon) + epsilon)
        else:
            corr_data[col] = np.log(vals.clip(min=epsilon) + epsilon)
    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr(method='pearson')

    print(f"\n  Pearson korreláció (log-transzformált, {len(df_valid)} adatpont):\n")
    # Header
    header = f"{'':22}"
    for col in candidate_metrics:
        header += f" {col[:10]:>12}"
    print(header)
    print("  " + "-" * (20 + 13 * len(candidate_metrics)))
    for row_col in candidate_metrics:
        line = f"  {row_col:20}"
        for col_col in candidate_metrics:
            r = corr_matrix.loc[row_col, col_col]
            marker = " *" if abs(r) > 0.85 and row_col != col_col else "  "
            line += f" {r:+8.3f}{marker}"
        print(line)

    print(f"\n  * = |r| > 0.85 (redundáns pár)")

    # Redundáns párok listája
    redundant_pairs = set()
    for i, m1 in enumerate(candidate_metrics):
        for m2 in candidate_metrics[i+1:]:
            r = abs(corr_matrix.loc[m1, m2])
            if r > 0.85:
                redundant_pairs.add((m1, m2))
                print(f"    REDUNDÁNS: {m1} ↔ {m2} (|r| = {r:.3f})")

    # =====================================================================
    # 2. MONOTONITÁS — FIZIKAI KONZISZTENCIA
    # =====================================================================
    print("\n" + "=" * 100)
    print("  2. MONOTONITÁS — FIZIKAI KONZISZTENCIA")
    print("     Spearman(flow_level, metrika): pozitív = több forgalom → nagyobb érték")
    print("     Reward-nál: negatív Spearman kell (több forgalom → alacsonyabb reward)")
    print("=" * 100)

    # Egyedi metrikák monotonitása (nyers, nem reward)
    print(f"\n  Nyers metrika monotonitás (flow_level vs metrika átlag):")
    print(f"  {'Metrika':20} {'Spearman_r':>10} {'p-value':>10} {'Irány':>12}")
    print("  " + "-" * 55)

    metric_directions = {}  # 'up' = nő flow-val (wait, co2, queue), 'down' = csökken (speed)
    for col in candidate_metrics:
        # Flow level → átlagos metrika érték (junction-ökön átlagolva)
        flow_means = df_valid.groupby('flow_level')[col].mean()
        rho, pval = stats.spearmanr(flow_means.index, flow_means.values)
        direction = "NŐ ↑" if rho > 0 else "CSÖKKEN ↓"
        metric_directions[col] = 'up' if rho > 0 else 'down'
        print(f"  {col:20} {rho:+10.3f} {pval:10.2e} {direction:>12}")

    # =====================================================================
    # 3. METRIKA KOMBINÁCIÓK TESZTELÉSE
    # =====================================================================
    print("\n" + "=" * 100)
    print("  3. METRIKA KOMBINÁCIÓK — SZŰRÉS ÉS RANGSOROLÁS")
    print("=" * 100)

    # Kombók generálása: 1-es és 2-es kombinációk
    all_combos = []
    for r in [1, 2]:
        for combo in itertools.combinations(candidate_metrics, r):
            # 2-es kombónál: redundancia szűrés
            if r == 2:
                if (combo[0], combo[1]) in redundant_pairs or (combo[1], combo[0]) in redundant_pairs:
                    continue
            all_combos.append(combo)

    print(f"\n  Vizsgált kombinációk: {len(all_combos)}")
    print(f"  (2-es kombókból kiszűrve a redundáns párok: |r| > 0.85)\n")

    combo_results = []

    for combo in all_combos:
        combo_name = " + ".join(combo)

        # Per-junction reward számítás és aggregálás
        all_rewards = []
        all_flow_levels = []
        combo_monotonicity_per_junction = []
        combo_iqr_per_junction = []
        combo_eta2_per_junction = []

        for jid in junction_ids:
            jdf = df_valid[df_valid['junction'] == jid].copy()
            if len(jdf) < 50:
                continue

            # Reward: minden metrikára log-sigmoid lokális params, átlag
            reward_components = []
            for metric in combo:
                vals = jdf[metric].values
                vals_pos = vals.clip(min=1e-5)

                # Lokális mu/std számítás
                log_v = np.log(vals_pos + 1e-5)
                mu = np.mean(log_v)
                std = np.std(log_v)

                if metric in ('AvgSpeed', 'Throughput'):
                    # Speed/Throughput invertálva: magasabb = jobb = magasabb reward
                    # R = sigmoid((log(x) - mu) / std) — NEM 1-sigmoid!
                    z = (log_v - mu) / (std + 1e-9)
                    r_component = 1.0 / (1.0 + np.exp(-z))
                else:
                    # Waiting, CO2, Queue, TT, Occ, SpeedStd, HaltRatio: alacsonyabb = jobb
                    z = (log_v - mu) / (std + 1e-9)
                    r_component = 1.0 - 1.0 / (1.0 + np.exp(-z))

                reward_components.append(r_component)

            rewards = np.mean(reward_components, axis=0)
            flow_levels = jdf['flow_level'].values

            all_rewards.extend(rewards)
            all_flow_levels.extend(flow_levels)

            # Per-junction monotonitás
            # Lokális pozíció-indexelés (iloc-alapú, nem label-alapú)
            jdf_reset = jdf.reset_index(drop=True)
            flow_groups = jdf_reset.groupby('flow_level').apply(
                lambda g: np.mean(rewards[g.index.values])
                if len(g) > 0 else 0.5
            )
            if len(flow_groups) > 2:
                rho_j, _ = stats.spearmanr(flow_groups.index, flow_groups.values)
                combo_monotonicity_per_junction.append(rho_j)

            # Per-junction IQR
            iqr = float(np.percentile(rewards, 75) - np.percentile(rewards, 25))
            combo_iqr_per_junction.append(iqr)

            # Per-junction ANOVA η² (eta-squared)
            # η² = SS_between / SS_total
            groups = [rewards[flow_levels == fl] for fl in sorted(set(flow_levels)) if np.sum(flow_levels == fl) > 0]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) > 1:
                grand_mean = np.mean(rewards)
                ss_total = np.sum((rewards - grand_mean) ** 2)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                eta2 = ss_between / ss_total if ss_total > 0 else 1.0
                combo_eta2_per_junction.append(eta2)

        if not combo_monotonicity_per_junction:
            continue

        all_rewards = np.array(all_rewards)
        all_flow_levels = np.array(all_flow_levels)

        # Aggregált statisztikák
        avg_monotonicity = np.mean(combo_monotonicity_per_junction)
        avg_iqr = np.mean(combo_iqr_per_junction)
        avg_eta2 = np.mean(combo_eta2_per_junction) if combo_eta2_per_junction else 1.0
        mono_pass_pct = np.mean([r < -0.5 for r in combo_monotonicity_per_junction]) * 100
        iqr_pass_pct = np.mean([q > 0.10 for q in combo_iqr_per_junction]) * 100

        # Szűrők
        mono_ok = avg_monotonicity < -0.5
        iqr_ok = avg_iqr > 0.10

        combo_results.append({
            'combo': combo_name,
            'combo_tuple': combo,
            'n_metrics': len(combo),
            'avg_monotonicity': avg_monotonicity,
            'mono_pass_pct': mono_pass_pct,
            'mono_ok': mono_ok,
            'avg_iqr': avg_iqr,
            'iqr_pass_pct': iqr_pass_pct,
            'iqr_ok': iqr_ok,
            'avg_eta2': avg_eta2,
            'eta2_within': 1.0 - avg_eta2,  # within-flow proportion
            'passed': mono_ok and iqr_ok,
            'is_current': False,
        })

    # --- Eredmények kiírása ---
    print(f"\n  {'Kombináció':40} {'Mono_r':>8} {'Mono%':>6} {'IQR':>6} {'IQR%':>6} "
          f"{'η²':>6} {'1-η²':>6} {'Szűrő':>8}")
    print("  " + "-" * 100)

    # Rendezés: átmenők first, azon belül eta2 ascending (alacsonyabb = jobb)
    combo_results.sort(key=lambda x: (not x['passed'], x['avg_eta2']))

    for cr in combo_results:
        marker = ""
        status = "✓ PASS" if cr['passed'] else "✗ FAIL"
        mono_flag = "" if cr['mono_ok'] else " [MONO!]"
        iqr_flag = "" if cr['iqr_ok'] else " [IQR!]"

        print(f"  {cr['combo']:40} {cr['avg_monotonicity']:+8.3f} {cr['mono_pass_pct']:5.0f}% "
              f"{cr['avg_iqr']:6.3f} {cr['iqr_pass_pct']:5.0f}% "
              f"{cr['avg_eta2']:6.3f} {cr['eta2_within']:6.3f} "
              f"{status}{mono_flag}{iqr_flag}{marker}")

    # --- Átmenők rangsorolása ---
    passed = [cr for cr in combo_results if cr['passed']]
    failed = [cr for cr in combo_results if not cr['passed']]

    print(f"\n  Átment a szűrőn: {len(passed)}/{len(combo_results)}")
    print(f"  Kiesett:          {len(failed)}/{len(combo_results)}")

    if passed:
        print(f"\n  --- RANGSOR (alacsonyabb η² = jobb: az ágens hatása jobban látszik) ---")
        print(f"  {'#':>3} {'Kombináció':40} {'η²':>8} {'1-η²':>8} {'Mono_r':>8} {'IQR':>6}")
        print("  " + "-" * 80)
        for i, cr in enumerate(passed):
            marker = ""
            print(f"  {i+1:3} {cr['combo']:40} {cr['avg_eta2']:8.4f} {cr['eta2_within']:8.4f} "
                  f"{cr['avg_monotonicity']:+8.3f} {cr['avg_iqr']:6.3f}{marker}")

    # =====================================================================
    # 4. NORMALIZÁCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA
    # =====================================================================
    print("\n" + "=" * 100)
    print("  4. NORMALIZÁCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA")
    print("     A legjobb metrika-kombó(ka)t teszteljük különböző normalizációkkal")
    print("=" * 100)

    # Top 5 kombó η² alapján (preference-mentes: passed-ok előre, azon belül η² szerint;
    # ha nincs elég passed, akkor failed-ek is kerülnek a listába, hogy mindig 5 legyen)
    test_combos = []
    top5 = combo_results[:5]  # combo_results már rendezve: passed first, then η² asc
    test_combos = [cr['combo_tuple'] for cr in top5]

    norm_results = []

    for combo in test_combos:
        combo_name = " + ".join(combo)

        for method_name, norm_fn in NORM_METHODS.items():
            method_mono = []
            method_iqr = []
            method_eta2 = []

            for jid in junction_ids:
                jdf = df_valid[df_valid['junction'] == jid].copy()
                if len(jdf) < 50:
                    continue

                reward_components = []
                for metric in combo:
                    vals = jdf[metric].values.clip(min=1e-5)
                    log_v = np.log(vals + 1e-5)
                    mu = np.mean(log_v)
                    std_v = np.std(log_v)

                    r_component = norm_fn(vals, mu, std_v)

                    if metric in ('AvgSpeed', 'Throughput'):
                        r_component = 1.0 - r_component  # Invert for speed/throughput

                    reward_components.append(r_component)

                rewards = np.mean(reward_components, axis=0)
                flow_levels = jdf['flow_level'].values

                # Monotonitás
                flow_means = pd.Series(rewards, index=jdf.index).groupby(flow_levels).mean()
                if len(flow_means) > 2:
                    rho, _ = stats.spearmanr(flow_means.index, flow_means.values)
                    method_mono.append(rho)

                # IQR
                iqr = float(np.percentile(rewards, 75) - np.percentile(rewards, 25))
                method_iqr.append(iqr)

                # η²
                groups = [rewards[flow_levels == fl] for fl in sorted(set(flow_levels)) if np.sum(flow_levels == fl) > 0]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) > 1:
                    grand_mean = np.mean(rewards)
                    ss_total = np.sum((rewards - grand_mean) ** 2)
                    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                    eta2 = ss_between / ss_total if ss_total > 0 else 1.0
                    method_eta2.append(eta2)

            if method_mono:
                norm_results.append({
                    'combo': combo_name,
                    'method': method_name,
                    'avg_mono': np.mean(method_mono),
                    'avg_iqr': np.mean(method_iqr),
                    'avg_eta2': np.mean(method_eta2) if method_eta2 else 1.0,
                    'mono_ok': np.mean(method_mono) < -0.5,
                    'iqr_ok': np.mean(method_iqr) > 0.10,
                })

    if norm_results:
        print(f"\n  {'Kombináció':35} {'Módszer':18} {'Mono_r':>8} {'IQR':>6} {'η²':>8} {'1-η²':>8} {'Szűrő':>8}")
        print("  " + "-" * 105)
        norm_results.sort(key=lambda x: (x['combo'], x['avg_eta2']))
        for nr in norm_results:
            status = "✓" if nr['mono_ok'] and nr['iqr_ok'] else "✗"
            mono_flag = "" if nr['mono_ok'] else " [MONO!]"
            iqr_flag = "" if nr['iqr_ok'] else " [IQR!]"
            print(f"  {nr['combo']:35} {nr['method']:18} {nr['avg_mono']:+8.3f} {nr['avg_iqr']:6.3f} "
                  f"{nr['avg_eta2']:8.4f} {1-nr['avg_eta2']:8.4f} {status:>8}{mono_flag}{iqr_flag}")

    # =====================================================================
    # 5. STATE ↔ REWARD KONZISZTENCIA
    # =====================================================================
    print("\n" + "=" * 100)
    print("  5. STATE ↔ REWARD KONZISZTENCIA")
    print("     Az ágens state-je (detektor-szintű) mennyire predikálja a reward-ot?")
    print("     Ha a korreláció gyenge → az ágens a state-ből nem tudja megtanulni a reward-ot.")
    print("=" * 100)

    # State változók: detektor-szintű aggregáltak + fázis
    state_cols = ['det_occ_mean', 'det_occ_max',  'det_veh_sum','det_speed_mean', 'det_speed_min']
    #['det_occ_mean', 'det_occ_max', 'det_veh_sum', 'det_veh_mean','det_speed_mean', 'det_speed_min', 'phase', 'prev_phase']

    # Ellenőrzés: vannak-e detektor oszlopok az adatban?
    has_det_data = all(col in full_df.columns for col in state_cols[:6])

    if has_det_data:
        # Reward jelöltek: a szűrőn átmenők + jelenlegi
        triplet_cols = [c for c in df_valid.columns if c.startswith('triplet_')]
        # Rendezés: TRIPLET_DEFS sorrendjében
        defined_order = [f'triplet_{label}' for (_, _, _, _, label) in TRIPLET_DEFS]
        triplet_cols = [c for c in defined_order if c in triplet_cols]
        reward_metrics_to_test = ['AvgSpeed', 'Throughput', 'reward_plain', 'reward_halt'] + triplet_cols
        reward_metrics_to_test = [r for r in reward_metrics_to_test if r in df_valid.columns]

        print(f"\n  State változók: {', '.join(state_cols)}")
        print(f"  Reward jelöltek: {', '.join(reward_metrics_to_test)}")

        # --- 5a. State ↔ Reward korreláció ---
        print(f"\n  --- Pearson korreláció: |state| vs |reward metrika| ---")
        print(f"  {'':20}", end="")
        for sc in state_cols[:6]:
            print(f" {sc[:10]:>11}", end="")
        print()
        print("  " + "-" * (20 + 12 * 6))

        state_reward_corr = {}
        for rm in reward_metrics_to_test:
            line = f"  {rm:20}"
            corrs = []
            for sc in state_cols[:6]:
                valid = df_valid[[sc, rm]].dropna()
                valid = valid[(valid[sc] != 0) | (valid[rm] != 0)]  # legalább az egyik nem 0
                if len(valid) > 50:
                    r, p = stats.pearsonr(valid[sc].values, valid[rm].values)
                    corrs.append(abs(r))
                    marker = "**" if abs(r) > 0.5 else "* " if abs(r) > 0.3 else "  "
                    line += f" {r:+9.3f}{marker}"
                else:
                    corrs.append(0.0)
                    line += f" {'N/A':>11}"
            print(line)
            state_reward_corr[rm] = np.mean(corrs)

        print(f"\n  ** = |r| > 0.5 (erős),  * = |r| > 0.3 (közepes)")
        print(f"\n  Átlagos |korreláció| a state változókkal:")
        for rm in sorted(state_reward_corr.keys(), key=lambda x: -state_reward_corr[x]):
            strength = "ERŐS" if state_reward_corr[rm] > 0.4 else "KÖZEPES" if state_reward_corr[rm] > 0.25 else "GYENGE"
            print(f"    {rm:25} avg|r| = {state_reward_corr[rm]:.3f}  ({strength})")

        # --- 5b. State redundancia (state változók egymás között) ---
        print(f"\n  --- 5b. State változók függetlensége ---")
        state_only = state_cols[:6]
        state_df = df_valid[state_only].dropna()
        if len(state_df) > 50:
            # Globális Pearson korreláció
            state_corr = state_df.corr(method='pearson')
            print(f"\n  GLOBÁLIS Pearson korreláció:")
            print(f"  {'':18}", end="")
            for sc in state_only:
                print(f" {sc[:9]:>10}", end="")
            print()
            print("  " + "-" * (18 + 11 * len(state_only)))
            for sc1 in state_only:
                line = f"  {sc1:18}"
                for sc2 in state_only:
                    r = state_corr.loc[sc1, sc2]
                    marker = " *" if abs(r) > 0.85 and sc1 != sc2 else "  "
                    line += f" {r:+8.3f}{marker}"
                print(line)

            print(f"\n  * = |r| > 0.85 (redundáns)")
            for i, s1 in enumerate(state_only):
                for s2 in state_only[i+1:]:
                    r = abs(state_corr.loc[s1, s2])
                    if r > 0.85:
                        print(f"    REDUNDÁNS: {s1} ↔ {s2} (|r| = {r:.3f})")

            # Parciális korreláció — flow_level kiszűrése
            # r_partial(A,B|C) = (r_AB - r_AC * r_BC) / sqrt((1 - r_AC²)(1 - r_BC²))
            print(f"\n  PARCIÁLIS korreláció (flow_level kiszűrve):")
            print(f"  (Ha a globális magas de a parciális alacsony → csak a forgalom mozgatja együtt)")
            if 'flow_level' in df_valid.columns:
                state_with_flow = df_valid[state_only + ['flow_level']].dropna()
                flow_corrs = {}
                for sc in state_only:
                    r_fc, _ = stats.pearsonr(state_with_flow['flow_level'].values,
                                             state_with_flow[sc].values)
                    flow_corrs[sc] = r_fc

                print(f"  {'':18}", end="")
                for sc in state_only:
                    print(f" {sc[:9]:>10}", end="")
                print()
                print("  " + "-" * (18 + 11 * len(state_only)))

                partial_corr = {}
                for sc1 in state_only:
                    line = f"  {sc1:18}"
                    partial_corr[sc1] = {}
                    for sc2 in state_only:
                        if sc1 == sc2:
                            line += f" {'1.000':>10}"
                            partial_corr[sc1][sc2] = 1.0
                            continue
                        r_ab = state_corr.loc[sc1, sc2]
                        r_ac = flow_corrs[sc1]
                        r_bc = flow_corrs[sc2]
                        denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
                        if denom > 1e-9:
                            r_partial = (r_ab - r_ac * r_bc) / denom
                        else:
                            r_partial = r_ab
                        partial_corr[sc1][sc2] = r_partial
                        # Jelölés: ha globálisan magas de parciálisan alacsony → ↓
                        if abs(r_ab) > 0.85 and abs(r_partial) < 0.5:
                            marker = " ↓"  # flow_level okozta a korrelációt
                        elif abs(r_partial) > 0.85:
                            marker = " *"  # valódi redundancia
                        else:
                            marker = "  "
                        line += f" {r_partial:+8.3f}{marker}"
                    print(line)

                print(f"\n  * = |r_partial| > 0.85 (valódi redundancia, flow szűrése után is fennáll)")
                print(f"  ↓ = globálisan |r| > 0.85, de parciálisan |r| < 0.5 (CSAK a forgalom mozgatta)")

            # VIF (Variance Inflation Factor) — multikollinearitás mérése
            # VIF_j = 1 / (1 - R²_j), ahol R²_j = az j-edik változó R² értéke
            # a többi változóból regresszálva
            print(f"\n  VIF (Variance Inflation Factor):")
            print(f"  VIF = 1/(1-R²): VIF > 10 → erős multikollinearitás, VIF > 5 → közepes")
            from numpy.linalg import lstsq

            state_arr = state_df.values
            state_means = state_arr.mean(axis=0)
            state_stds = state_arr.std(axis=0)
            state_stds[state_stds == 0] = 1.0
            state_norm = (state_arr - state_means) / state_stds

            print(f"  {'State változó':20} {'VIF':>8} {'Értékelés':>15}")
            print("  " + "-" * 45)
            vif_results = {}
            for j, sc in enumerate(state_only):
                # y = j-edik oszlop, X = többi oszlop
                y = state_norm[:, j]
                X = np.delete(state_norm, j, axis=1)
                X = np.column_stack([X, np.ones(len(X))])  # intercept
                coeffs, residuals, _, _ = lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 0.9999 else 9999.0
                vif_results[sc] = vif
                verdict = "ERŐS MULTIKOL" if vif > 10 else "KÖZEPES" if vif > 5 else "OK"
                print(f"  {sc:20} {vif:8.2f} {verdict:>15}")

            # Javaslat state-re a VIF alapján
            print(f"\n  Javasolt state változók (VIF < 10):")
            good_states = [sc for sc, v in vif_results.items() if v < 10]
            high_vif_states = [sc for sc, v in vif_results.items() if v >= 10]
            for sc in good_states:
                print(f"    ✓ {sc} (VIF = {vif_results[sc]:.2f})")
            for sc in high_vif_states:
                print(f"    ✗ {sc} (VIF = {vif_results[sc]:.2f}) — redundáns, elhagyható")

        # --- 5c. Mutual Information approximáció ---
        # MI ≈ -0.5 * log(1 - r²) a normális eloszlás esetén (Gaussi MI)
        print(f"\n  --- Gaussi Mutual Information: state → reward metrika ---")
        print(f"  MI ≈ -0.5 * ln(1 - r²)  [nats]")
        print(f"  {'Reward metrika':25} {'Σ MI (összes state)':>20} {'Max MI (legjobb state)':>25}")
        print("  " + "-" * 75)

        mi_scores = {}
        for rm in reward_metrics_to_test:
            total_mi = 0.0
            max_mi = 0.0
            max_mi_state = ""
            for sc in state_cols[:6]:
                valid = df_valid[[sc, rm]].dropna()
                valid = valid[(valid[sc] != 0) | (valid[rm] != 0)]
                if len(valid) > 50:
                    r, _ = stats.pearsonr(valid[sc].values, valid[rm].values)
                    r2 = min(r**2, 0.9999)  # numerikus stabilitás
                    mi = -0.5 * np.log(1 - r2)
                    total_mi += mi
                    if mi > max_mi:
                        max_mi = mi
                        max_mi_state = sc
            mi_scores[rm] = total_mi
            print(f"  {rm:25} {total_mi:20.4f} {max_mi:10.4f} ({max_mi_state})")

        # --- 5d. Javaslat ---
        print(f"\n  --- STATE↔REWARD JAVASLAT ---")
        # Rendezés MI szerint
        sorted_mi = sorted(mi_scores.items(), key=lambda x: -x[1])
        print(f"  A state legtöbb információt hordoz ezekről a reward metrikákról:")
        for i, (rm, mi) in enumerate(sorted_mi[:5]):
            flag = ""
            print(f"    {i+1}. {rm:25} MI = {mi:.4f}{flag}")

    else:
        print("\n  [SKIP] Nincs detektor-szintű adat a CSV-kben.")
        print("  Futtasd újra a szimulációt (--skip-simulation NÉLKÜL)!")

    # =====================================================================
    # 6. ÁBRÁK
    # =====================================================================

    # Ábra 1: Korreláció mátrix — csak egyedi (nem aggregált) metrikák
    # Kiszűrjük a reward_* és triplet_* aggregált oszlopokat, csak nyers változókat mutatunk
    plot_metrics = [m for m in candidate_metrics
                    if not m.startswith('reward_') and not m.startswith('triplet_')]
    plot_corr = corr_matrix.loc[plot_metrics, plot_metrics]
    n_met = len(plot_metrics)
    fig_sz = max(8, n_met * 1.2)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))
    im = ax.imshow(plot_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_met))
    ax.set_xticklabels(plot_metrics, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_met))
    ax.set_yticklabels(plot_metrics, fontsize=9)
    for i in range(n_met):
        for j in range(n_met):
            val = plot_corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            weight = 'bold' if abs(val) > 0.85 and i != j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Metric Correlation Matrix (log-transformed)\n* Bold = redundant (|r| > 0.85)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_correlation_matrix.png'), dpi=200)
    plt.close()
    print(f"\n  reward_correlation_matrix.png mentve ({n_met} egyedi metrika, aggregált kiszűrve)")

    # Ábra 2: Top 20 kombináció — η² + monotonitás + IQR
    if combo_results:
        sorted_cr = sorted(combo_results, key=lambda x: (not x['passed'], x['avg_eta2']))
        show_cr = sorted_cr[:20]
        names = [cr['combo'] for cr in show_cr]
        n_show = len(names)

        fig, axes = plt.subplots(1, 3, figsize=(24, max(8, n_show * 0.45)))
        colors = ['#2ecc71' if cr['passed'] else '#e74c3c' for cr in show_cr]
        y_pos = np.arange(n_show)

        ax = axes[0]
        ax.barh(y_pos, [cr['avg_eta2'] for cr in show_cr], color=colors, alpha=0.8)
        for i, cr in enumerate(show_cr):
            ax.text(cr['avg_eta2'] + 0.01, i, f"{cr['avg_eta2']:.3f}", va='center', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('η² (lower = better)')
        ax.set_title('ANOVA η²', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[1]
        ax.barh(y_pos, [cr['avg_monotonicity'] for cr in show_cr], color=colors, alpha=0.8)
        ax.axvline(-0.5, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([], fontsize=9)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Monotonicity', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[2]
        ax.barh(y_pos, [cr['avg_iqr'] for cr in show_cr], color=colors, alpha=0.8)
        ax.axvline(0.10, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([], fontsize=9)
        ax.set_xlabel('IQR')
        ax.set_title('Reward IQR', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(f'Top {n_show} Metric Combinations (green=PASS, red=FAIL)', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_combo_selection.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_combo_selection.png mentve")

    # Ábra 2b: ÖSSZES kombináció
    if combo_results:
        sorted_cr_all = sorted(combo_results, key=lambda x: (not x['passed'], x['avg_eta2']))
        names_all = [cr['combo'] for cr in sorted_cr_all]
        n_all = len(names_all)

        fig, axes = plt.subplots(1, 3, figsize=(26, max(10, n_all * 0.4)))
        colors_all = ['#2ecc71' if cr['passed'] else '#e74c3c' for cr in sorted_cr_all]
        y_all = np.arange(n_all)

        ax = axes[0]
        ax.barh(y_all, [cr['avg_eta2'] for cr in sorted_cr_all], color=colors_all, alpha=0.8)
        for i, cr in enumerate(sorted_cr_all):
            eta_val = cr['avg_eta2']
            if not np.isnan(eta_val):
                ax.text(eta_val + 0.01, i, f"{eta_val:.3f}", va='center', fontsize=7)
        ax.set_yticks(y_all)
        ax.set_yticklabels(names_all, fontsize=7)
        ax.set_xlabel('η² (lower = better)')
        ax.set_title('ANOVA η²', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[1]
        mono_plot = [v if not np.isnan(v) else 0.0 for v in [cr['avg_monotonicity'] for cr in sorted_cr_all]]
        ax.barh(y_all, mono_plot, color=colors_all, alpha=0.8)
        ax.axvline(-0.5, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_all)
        ax.set_yticklabels([], fontsize=7)
        ax.set_xlabel('Spearman ρ')
        ax.set_title('Monotonicity', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[2]
        iqr_all = [cr['avg_iqr'] if not np.isnan(cr['avg_iqr']) else 0.0 for cr in sorted_cr_all]
        ax.barh(y_all, iqr_all, color=colors_all, alpha=0.8)
        ax.axvline(0.10, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_all)
        ax.set_yticklabels([], fontsize=7)
        ax.set_xlabel('IQR')
        ax.set_title('Reward IQR', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(f'All {n_all} Metric Combinations (green=PASS, red=FAIL)', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_combo_selection_full.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_combo_selection_full.png mentve")

    # Ábra 3: Normalizáció összehasonlítás — vertikális layout (top 5 kombó, η² szerint rendezve)
    if norm_results:
        # Kombók sorrendje: η² szerint (legkisebb felül = legjobb)
        combo_eta2_order = {}
        for nr in norm_results:
            combo_eta2_order.setdefault(nr['combo'], []).append(nr['avg_eta2'])
        combos_ordered = sorted(combo_eta2_order.keys(),
                                key=lambda c: np.mean(combo_eta2_order[c]))
        n_combos = len(combos_ordered)

        panel_h = 3.8   # magasabb panel = több hely a feliratnak
        fig_h = max(10, panel_h * n_combos + 1.5)
        fig, axes = plt.subplots(n_combos, 1, figsize=(12, fig_h))
        if n_combos == 1:
            axes = [axes]

        for idx, combo_name in enumerate(combos_ordered):
            ax = axes[idx]
            c_results = [nr for nr in norm_results if nr['combo'] == combo_name]
            c_results.sort(key=lambda x: x['avg_eta2'])
            methods = [nr['method'] for nr in c_results]
            eta2s = [nr['avg_eta2'] for nr in c_results]
            colors_n = ['#2ecc71' if nr['mono_ok'] and nr['iqr_ok'] else '#e74c3c' for nr in c_results]

            y = np.arange(len(methods))
            ax.barh(y, eta2s, color=colors_n, alpha=0.8, height=0.55)
            for j, e in enumerate(eta2s):
                ax.text(e + 0.002, j, f'{e:.4f}', va='center', fontsize=9)
            ax.set_yticks(y)
            ax.set_yticklabels(methods, fontsize=10)
            ax.set_xlabel('η² (lower = better)', fontsize=10)
            ax.set_title(f'#{idx+1}  {combo_name}', fontsize=12, fontweight='bold', pad=5)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle('Normalization Methods Comparison — Top 5 Metric Combos',
                     fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(output_dir, 'reward_normalization_comparison.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_normalization_comparison.png mentve ({n_combos} kombó)")

    # Ábra 4: State ↔ Reward heatmap + MI
    if has_det_data and state_reward_corr:
        n_rewards = len(reward_metrics_to_test)
        n_states = len(state_cols[:6])
        fig, axes = plt.subplots(1, 2, figsize=(20, max(3, n_rewards * 0.8)),
                                 gridspec_kw={'width_ratios': [1.3, 1]})

        # 4a. State ↔ Reward korreláció heatmap
        ax = axes[0]
        sr_matrix = []
        for rm in reward_metrics_to_test:
            row_vals = []
            for sc in state_cols[:6]:
                valid = df_valid[[sc, rm]].dropna()
                valid = valid[(valid[sc] != 0) | (valid[rm] != 0)]
                if len(valid) > 50:
                    r, _ = stats.pearsonr(valid[sc].values, valid[rm].values)
                    row_vals.append(r)
                else:
                    row_vals.append(0.0)
            sr_matrix.append(row_vals)
        sr_matrix = np.array(sr_matrix)
        im = ax.imshow(sr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_states))
        ax.set_xticklabels([s.replace('det_', '') for s in state_cols[:6]],
                           rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(n_rewards))
        ax.set_yticklabels(reward_metrics_to_test, fontsize=10)
        for i in range(n_rewards):
            for j in range(n_states):
                val = sr_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold' if abs(val) > 0.7 else 'normal')
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title('State → Reward Correlation (Pearson r)', fontsize=13)

        # 4b. MI bar chart
        ax = axes[1]
        sorted_mi_items = sorted(mi_scores.items(), key=lambda x: -x[1])
        mi_names = [x[0] for x in sorted_mi_items]
        mi_vals = [x[1] for x in sorted_mi_items]
        colors_mi = ['#3498db' for _ in mi_names]
        y = np.arange(len(mi_names))
        bars = ax.barh(y, mi_vals, color=colors_mi, alpha=0.8, height=0.6)
        for i, v in enumerate(mi_vals):
            ax.text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(mi_names, fontsize=10)
        ax.set_xlabel('Gaussian Mutual Information (nats)', fontsize=10)
        ax.set_title('State → Reward: Total MI', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_state_consistency.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_state_consistency.png mentve")

    # =====================================================================
    # ÖSSZEFOGLALÓ
    # =====================================================================
    print(f"\n{'=' * 100}")
    print("  ÖSSZEFOGLALÓ ÉS AJÁNLÁS")
    print(f"{'=' * 100}")

    if passed:
        best = passed[0]
        print(f"\n  Legjobb metrika-kombináció: {best['combo']}")
        print(f"    η² = {best['avg_eta2']:.4f} (a flow_level a variancia {best['avg_eta2']*100:.1f}%-át magyarázza)")
        print(f"    1-η² = {best['eta2_within']:.4f} (a variancia {best['eta2_within']*100:.1f}%-a NEM a forgalmi szintből jön)")
        print(f"    Monotonitás = {best['avg_monotonicity']:+.3f}")
        print(f"    IQR = {best['avg_iqr']:.3f}")

        # Top 5 listázása
        for i, cr in enumerate(passed[:5]):
            if i == 0:
                continue  # már kiírva fent
            print(f"\n  #{i+1}: {cr['combo']}")
            print(f"    η² = {cr['avg_eta2']:.4f}, Mono = {cr['avg_monotonicity']:+.3f}, IQR = {cr['avg_iqr']:.3f}")

    if norm_results:
        # Best normalization for top combo
        best_combo_name = passed[0]['combo'] if passed else "TotalWaitingTime + TotalCO2"
        best_norms = [nr for nr in norm_results if nr['combo'] == best_combo_name and nr['mono_ok'] and nr['iqr_ok']]
        if best_norms:
            best_norms.sort(key=lambda x: x['avg_eta2'])
            bn = best_norms[0]
            print(f"\n  Legjobb normalizáció ({best_combo_name}):")
            print(f"    Módszer: {bn['method']}")
            print(f"    η² = {bn['avg_eta2']:.4f}")

    # CSV mentés
    combo_df = pd.DataFrame([{k: v for k, v in cr.items() if k != 'combo_tuple'} for cr in combo_results])
    combo_df.to_csv(os.path.join(output_dir, 'reward_selection_results.csv'), index=False)
    print(f"\n  reward_selection_results.csv mentve")

    if norm_results:
        norm_df = pd.DataFrame(norm_results)
        norm_df.to_csv(os.path.join(output_dir, 'reward_normalization_results.csv'), index=False)
        print(f"  reward_normalization_results.csv mentve")

    # =========================================================================
    # Ábra 5: WandB-stílusú step-szintű reward görbe — önálló függvénybe kiemelve
    # =========================================================================
    generate_reward_step_curve(output_dir)

    # State↔Reward összefoglaló
    if has_det_data and state_reward_corr:
        print(f"\n  State↔Reward konzisztencia:")
        best_sr = max(state_reward_corr.items(), key=lambda x: x[1])
        worst_sr = min(state_reward_corr.items(), key=lambda x: x[1])
        print(f"    Legerősebb state↔reward: {best_sr[0]} (avg|r| = {best_sr[1]:.3f})")
        print(f"    Leggyengébb state↔reward: {worst_sr[0]} (avg|r| = {worst_sr[1]:.3f})")
        if best_sr[0] in ('TotalWaitingTime', 'TotalCO2'):
            print(f"    ✓ A jelenlegi reward metrika erős state-kapcsolattal bír")
        else:
            print(f"    ⚠ A state-ből jobban predikálható: {best_sr[0]}")


def main():
    parser = argparse.ArgumentParser(description='Per-junction metric collection and normalization calibration')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip simulation, only run analysis on existing CSVs')
    parser.add_argument('--gui', action='store_true',
                        help='Use sumo-gui instead of libsumo (vizuális mód, Windows-on is működik)')
    parser.add_argument('--reward-curve-only', action='store_true',
                        help='Csak a reward_step_curve.png generálása (szimuláció + analízis nélkül)')
    parser.add_argument('--reward-curve-junction', type=str, default='',
                        help='Csak ezt a junction-t plotolja lokális paraméterekkel (pl. R1C1_C). '
                             'Ha üres, a REWARD_CURVE_JUNCTION config értéke érvényes.')
    args = parser.parse_args()
    use_gui = args.gui

    # Gyors mód: csak a reward step curve regenerálása
    if args.reward_curve_only:
        jid = args.reward_curve_junction or REWARD_CURVE_JUNCTION or None
        print(f"[reward-curve-only] junction={jid or 'összes'}")
        generate_reward_step_curve(OUTPUT_DIR, junction_filter=jid)
        return

    # Ellenőrzés
    for name, path in [("net", NET_FILE), ("logic", LOGIC_FILE), ("det", DETECTOR_FILE)]:
        if not os.path.exists(path):
            print(f"[ERROR] Hianyo fajl ({name}): {path}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.skip_simulation:
        print("=" * 80)
        print("  PER-JUNCTION METRIKA GYUJTES")
        print("=" * 80)
        print(f"  Net: {NET_FILE}")
        print(f"  Flow szintek (max): {FLOW_MAX_LEVELS}")
        print(f"  Epizodok szintenkent: {EPISODES_PER_LEVEL} random + {ACTUATED_EPISODES} actuated")
        print(f"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s")
        print(f"  Kimenet: {OUTPUT_DIR}")
        print("=" * 80)

        total_random = len(FLOW_MAX_LEVELS) * EPISODES_PER_LEVEL
        total_actuated = len(FLOW_MAX_LEVELS) * ACTUATED_EPISODES
        total_sims = total_random + total_actuated
        sim_count = 0

        # --- Random epizódok ---
        print(f"\n  [1/2] RANDOM kontroll: {total_random} epizod")
        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(EPISODES_PER_LEVEL):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{EPISODES_PER_LEVEL} | RANDOM ---")
                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode="random", use_gui=use_gui)

        # --- Actuated epizódok ---
        print(f"\n  [2/2] ACTUATED kontroll: {total_actuated} epizod")
        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(ACTUATED_EPISODES):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{ACTUATED_EPISODES} | ACTUATED ---")
                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode="actuated", use_gui=use_gui)

        print(f"\n  Szimulacio kesz! ({total_sims} epizod: {total_random} random + {total_actuated} actuated)")
    else:
        print("  --skip-simulation: Csak elemzes a meglevo CSV-kre")

    # Per-junction kalibráció
    analyze_per_junction(OUTPUT_DIR)

    # Reward metrika és normalizáció kiválasztás
    reward_selection_analysis(OUTPUT_DIR)


if __name__ == "__main__":
    main()

