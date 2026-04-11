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
import re
import sys
import json
import random
import argparse
import platform
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
EPISODES_PER_LEVEL = 4  # ismétlések forgalmi szintenként (random kontroll)
ACTUATED_EPISODES = 8   # ismétlések forgalmi szintenként (actuated kontroll)
MIXED_EPISODES    = 4   # ismétlések forgalmi szintenként (mixed: 80% actuated + 20% random)
MIXED_EPSILON     = 0.20  # random akció valószínűsége mixed módban (ϵ-greedy)
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

# --- libsumo vs traci ---
# libsumo: gyorsabb (beágyazott), de egyes rendszereken instabil lehet.
# Felülírható: USE_LIBSUMO=0 python ... ha traci kellene.
USE_LIBSUMO = True

# --- Junction comparison plot variáns ---
# Lehetséges értékek: "plain" | "halt" | "triplet" | "" (üres = auto: legjobb IQR alapján)
PLOT_REWARD_VARIANT = ""

# --- Reward step curve (reward_step_curve.png) ---
# Ha meg van adva, csak ezt a junction-t plotolja LOKÁLIS normalizációs paraméterekkel.
# Ha üres → minden junction, globális paraméterekkel.
REWARD_CURVE_JUNCTION = "R1C1_C"
# Reward mód: "speed_throughput" | "wait_triplet_tpstdhalt" | "wait_haltratio" | "halt_ratio" | "co2_speedstd"
REWARD_CURVE_MODE = "wait_haltratio"

_TRACI_IS_RUNNING = False

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

    global _TRACI_IS_RUNNING
    # USE_LIBSUMO config alapján döntünk (macOS-en False, Linuxon True)
    # env var felülírja: USE_LIBSUMO=0 / USE_LIBSUMO=1
    _env_override = os.environ.get('USE_LIBSUMO')
    use_libsumo = (int(_env_override) == 1) if _env_override is not None else USE_LIBSUMO
    if use_gui or not use_libsumo:
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

    if use_gui or not use_libsumo:
        sumo_bin = "sumo-gui" if use_gui else "sumo"
        if _TRACI_IS_RUNNING:
            try: traci.close()
            except: pass
        traci.start([sumo_bin] + sumo_args)
        _TRACI_IS_RUNNING = True
    else:
        sumo_bin = "sumo"
        if not _TRACI_IS_RUNNING:
            traci.start([sumo_bin] + sumo_args)
            _TRACI_IS_RUNNING = True
        else:
            try:
                traci.load(sumo_args)
            except Exception:
                traci.start([sumo_bin] + sumo_args)
            _TRACI_IS_RUNNING = True

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
    for step in range(WARMUP):
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

    # Vezérlők létrehozása (random és mixed módhoz)
    tl_controllers = {}
    if control_mode == "random" or control_mode.startswith("mixed"):
        for jid in junction_ids:
            if jid in logic:
                tl_controllers[jid] = TLController(jid, logic[jid])

    # --- Actuated mód beállítása ---
    # NEM a SUMO bepített programot hasznlja, hanem ugyanazt a traffic_lights.json
    # alap fzis + tmeneti logikt, mint az RL gens  igazsgos sszehasonlts!
    # Dntslogika: occupancy-alap (ha az aktulis fzis dtektori kihasznltak
    # s < MAX_GREEN_STEPS, marad; egybknt a legterheltebb fzisra vlt).
    MAX_GREEN_STEPS  = max(1, 60 // DELTA_TIME)   # 60 mp max zld id  lpesben
    OCC_THRESHOLD    = 0.15                        # 15% feletti occupancy  marad a fzis

    class TLActuatedController(TLController):
        """TLController + occupancy-alap fzisvalts logika.
        Ugyanazokat a JSON fzisokat s tmeneteket hasznlja mint az RL."""
        def __init__(self, jid, logic_data, all_dets):
            super().__init__(jid, logic_data)
            # Detek: mely detektorok tartoznak ehhez a junction-hoz
            self.dets = all_dets
            # Fzis  detektor trkp: logic_idx  (kzeltleg) az adott fzishoz
            # tartoz detektorok. Egyszerstett: az sszes dtektort figyelembe vesszk
            # mivel a junction_dets m r el van ksztve nm kthet fzisonknt.
            self.green_timer = 0  # hny lpse tart m mr az aktulis zld fzis

        def decide_next_phase(self):
            """Legmagasabb tlagos occupancy-j fzis kivlasztsa.
            Ha az aktulis fzis mg elg terhelt  s nem rte el a MAX-ot  marad."""
            if self.green_timer < MAX_GREEN_STEPS:
                # Check if current phase detectors are still busy
                cur_occ = self._phase_occupancy(self.current_logic_idx)
                if cur_occ >= OCC_THRESHOLD:
                    return self.current_logic_idx  # marad
            # Vlts: a legterheltebb (legmagasabb occ) fzisra
            best_idx  = self.current_logic_idx
            best_occ  = -1.0
            for idx in range(self.num_phases):
                occ = self._phase_occupancy(idx)
                if occ > best_occ:
                    best_occ = occ
                    best_idx = idx
            return best_idx

        def _phase_occupancy(self, logic_idx):
            """tlagos pillanati occupancy az sszes dtektoron (kzelts)."""
            if not self.dets:
                return 0.0
            total = 0.0
            for det in self.dets:
                try:
                    total += traci.inductionloop.getLastStepOccupancy(det)
                except Exception:
                    pass
            return total / len(self.dets)

        def step(self):
            """Egy szimulcis lpst hajt vgre  beleertve a dntslogikt."""
            if self.is_ready():
                next_ph = self.decide_next_phase()
                if next_ph != self.current_logic_idx:
                    self.set_target(next_ph)
                    self.green_timer = 0
                else:
                    self.green_timer += 1
            self.update()

    if control_mode == "actuated":
        for jid in junction_ids:
            if jid in logic:
                tl_controllers[jid] = TLActuatedController(
                    jid, logic[jid], junction_dets.get(jid, [])
                )
        print(f"      [ACTUATED] JSON-alap fzislogika (+ tmenetek) bekapcsolva")

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
        elif control_mode.startswith("mixed"):
            # Mixed (ϵ-greedy): MIXED_EPSILON valószínűséggel random akció,
            # egyébként megtartja az aktuális fázist (actuated-szerű viselkedés).
            try:
                _eps_rand = int(control_mode.split('_')[1]) / 100.0
            except Exception:
                _eps_rand = MIXED_EPSILON
            for jid in junction_ids:
                ctrl = tl_controllers.get(jid)
                if ctrl:
                    if ctrl.is_ready() and random.random() < _eps_rand:
                        ctrl.set_target(random.randint(0, ctrl.num_phases - 1))
                    # else: ne tegyünk semmit — a fázis marad (actuated-viselkedés)
                    ctrl.update()
                    current_phase[jid] = ctrl.current_logic_idx
                else:
                    current_phase[jid] = 0
        else:
            # Actuated: TLActuatedController  ugyanolyan JSON fzisok mint az RL
            for jid in junction_ids:
                ctrl = tl_controllers.get(jid)
                if ctrl:
                    ctrl.step()
                    current_phase[jid] = ctrl.current_logic_idx
                else:
                    current_phase[jid] = 0

        # Akkumulálás delta_time lépésen
        acc = {jid: {
            'tt': 0.0, 'tt_raw': 0.0, 'waiting': 0.0, 'co2': 0.0,
            'veh': 0, 'speed': 0.0, 'halted': 0, 'occ': 0.0,
            'valid_tt': 0, 'valid_tt_raw': 0, 'valid_speed': 0,
            'steps': 0,
            # Új metrikák
            'lane_speeds': [],       # per-sub-step lane átlagsebességek (SpeedStd-hez)
            'throughput': 0,         # MOZGÓ járművek (= det_veh - det_halting), mint training
            # Detektor-szintű state akkumulátorok
            'det_occ': defaultdict(float),   # per-detektor occupancy
            'det_veh': defaultdict(int),     # per-detektor vehicle count
            'det_halting': defaultdict(int), # per-detektor halting count (throughput korrekció)
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
                    # Halting becslés: ha sebesség < 0.1 m/s → álló
                    det_spd = traci.inductionloop.getLastStepMeanSpeed(det)
                    det_halt = det_veh_count if (det_spd >= 0 and det_spd < 0.1) else 0
                    acc[jid]['det_halting'][det] += det_halt
                    # Throughput = mozgó járművek (mint training get_throughput_metric())
                    acc[jid]['throughput'] += max(0, det_veh_count - det_halt)
                    if det_spd >= 0:
                        acc[jid]['det_speed'][det] += det_spd
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

    if use_gui or not use_libsumo:
        try:
            traci.close()
        except:
            pass

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
    Megegyezik a sumo_rl_environment.py _compute_reward() speed_throughput módjával:
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

    # --- Extra globális paraméterek: wait_triplet_tpstdhalt módhoz ---
    _g_wait = valid_global['TotalWaitingTime'].values if 'TotalWaitingTime' in valid_global.columns else np.array([])
    _g_std  = valid_global['SpeedStd'].values         if 'SpeedStd'         in valid_global.columns else np.array([])
    _g_halt = valid_global['HaltRatio'].values        if 'HaltRatio'        in valid_global.columns else np.array([])
    glob_mu_w,  glob_std_w  = robust_log_params(_g_wait)  or (6.189495, 2.666679)
    glob_mu_ss, glob_std_ss = robust_log_params(_g_std)   or (0.767463, 0.671022)
    glob_mu_h,  glob_std_h  = robust_log_params(_g_halt)  or (-0.445295, 0.510940)
    if glob_mu_w  is None: glob_mu_w,  glob_std_w  = 6.189495, 2.666679
    if glob_mu_ss is None: glob_mu_ss, glob_std_ss = 0.767463, 0.671022
    if glob_mu_h  is None: glob_mu_h,  glob_std_h  = -0.445295, 0.510940

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

        # Extra paraméterek a wait_triplet_tpstdhalt módhoz
        _jw   = jdf_valid['TotalWaitingTime'].values if 'TotalWaitingTime' in jdf_valid.columns else np.array([])
        _jss  = jdf_valid['SpeedStd'].values         if 'SpeedStd'         in jdf_valid.columns else np.array([])
        _jh   = jdf_valid['HaltRatio'].values        if 'HaltRatio'        in jdf_valid.columns else np.array([])
        mu_w,  std_w  = robust_log_params(_jw)  if len(_jw) > 5  else (None, None)
        mu_ss, std_ss = robust_log_params(_jss) if len(_jss) > 5 else (None, None)
        mu_h,  std_h  = robust_log_params(_jh)  if len(_jh) > 5  else (None, None)
        if mu_w  is None: mu_w,  std_w  = glob_mu_w,  glob_std_w
        if mu_ss is None: mu_ss, std_ss = glob_mu_ss, glob_std_ss
        if mu_h  is None: mu_h,  std_h  = glob_mu_h,  glob_std_h

        junction_params[jid] = {
            'MU_SPEED':       round(mu_s, 6),
            'STD_SPEED':      round(std_s, 6),
            'MU_THROUGHPUT':  round(mu_t, 6),
            'STD_THROUGHPUT': round(std_t, 6),
            'MU_WAIT':        round(mu_w, 6),
            'STD_WAIT':       round(std_w, 6),
            'MU_SPEEDSTD':    round(mu_ss, 6),
            'STD_SPEEDSTD':   round(std_ss, 6),
            'MU_HALT':        round(mu_h, 6),
            'STD_HALT':       round(std_h, 6),
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
        'reward_function': 'multi-mode log-tanh normalization',
        'global': {
            'MU_SPEED':       round(glob_mu_s, 6),
            'STD_SPEED':      round(glob_std_s, 6),
            'MU_THROUGHPUT':  round(glob_mu_t, 6),
            'STD_THROUGHPUT': round(glob_std_t, 6),
            'MU_WAIT':        round(glob_mu_w, 6),
            'STD_WAIT':       round(glob_std_w, 6),
            'MU_SPEEDSTD':    round(glob_mu_ss, 6),
            'STD_SPEEDSTD':   round(glob_std_ss, 6),
            'MU_HALT':        round(glob_mu_h, 6),
            'STD_HALT':       round(glob_std_h, 6),
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



def generate_reward_step_curve(output_dir, junction_filter=None, reward_mode=None):
    """
    WandB-stílusú step-szintű reward görbe.

    Ha junction_filter meg van adva (pl. "R1C1_C"), csak azt a junction-t plotolja
    LOKÁLIS normalizációs paraméterekkel.
    Ha junction_filter üres/None → minden junction, globális paraméterekkel.
    reward_mode: "speed_throughput" | "wait_triplet_tpstdhalt" | "wait_haltratio" | "halt_ratio" | "co2_speedstd"

    Önállóan is futtatható: python metric_collection_per_junction.py --reward-curve-only
    """
    if reward_mode is None:
        reward_mode = REWARD_CURVE_MODE
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

    mu_s     = params.get('MU_SPEED',       glob_p.get('MU_SPEED',       2.0))
    std_s    = params.get('STD_SPEED',      glob_p.get('STD_SPEED',      1.0))
    mu_t     = params.get('MU_THROUGHPUT',  glob_p.get('MU_THROUGHPUT',  3.0))
    std_t    = params.get('STD_THROUGHPUT', glob_p.get('STD_THROUGHPUT', 1.0))
    mu_wait  = params.get('MU_WAIT',        glob_p.get('MU_WAIT',        6.0))
    std_wait = params.get('STD_WAIT',       glob_p.get('STD_WAIT',       2.0))
    mu_sstd  = params.get('MU_SPEEDSTD',    glob_p.get('MU_SPEEDSTD',    0.5))
    std_sstd = params.get('STD_SPEEDSTD',   glob_p.get('STD_SPEEDSTD',   0.5))
    mu_halt  = params.get('MU_HALT',        glob_p.get('MU_HALT',        0.0))
    std_halt = params.get('STD_HALT',       glob_p.get('STD_HALT',       0.5))

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
        if 'AvgSpeed' not in df.columns:
            continue
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        ep = int(csv_file.split('_ep')[1].split('_')[0]) if '_ep' in csv_file else 0
        if '_actuated.csv' in csv_file:
            mode = 'actuated'
        elif '_mixed' in csv_file:
            _m = csv_file.split('_ep')[1].split('_', 1)
            mode = 'mixed_' + _m[1].replace('.csv', '') if len(_m) > 1 else 'mixed'
        else:
            mode = 'random'
        df['junction']     = jid
        df['flow_level']   = flow_level
        df['episode']      = ep
        df['control_mode'] = df['control_mode'] if 'control_mode' in df.columns else mode
        df['run_id']       = f"{jid}_flow{flow_level}_ep{ep}_{df['control_mode'].iloc[0]}"
        all_data.append(df)

    if not all_data:
        print(f"  [SKIP] Nincs adat junction_filter='{junction_filter}' esetén.")
        return

    full = pd.concat(all_data, ignore_index=True)
    full = full[full['VehCount'] > 0].copy()

    epsilon = 1e-5

    def _logtanh_pos(arr, mu, std):  # magasabb = jobb
        return (1 + np.tanh((np.log(arr.clip(min=epsilon)) - mu) / (std + 1e-9))) / 2

    def _logtanh_neg(arr, mu, std):  # alacsonyabb = jobb (pl. waiting, halt)
        return (1 - np.tanh((np.log(arr.clip(min=epsilon)) - mu) / (std + 1e-9))) / 2

    if reward_mode == 'wait_triplet_tpstdhalt':
        wait = full['AvgWaitingTime'].values if 'AvgWaitingTime' in full.columns else full['AvgSpeed'].values * 0
        thr  = full['Throughput'].values    if 'Throughput'     in full.columns else np.ones(len(full))
        sstd = full['SpeedStd'].values      if 'SpeedStd'       in full.columns else np.ones(len(full))
        halt = full['HaltRatio'].values     if 'HaltRatio'      in full.columns else np.zeros(len(full))
        r_wait  = _logtanh_neg(wait + epsilon, mu_wait,  std_wait)
        r_tp    = _logtanh_pos(thr,            mu_t,     std_t)
        r_sstd  = _logtanh_pos(sstd + epsilon, mu_sstd,  std_sstd)
        r_halt  = _logtanh_pos(np.abs(halt) + epsilon, mu_halt, std_halt)
        r_trip  = (r_tp + r_sstd + (1 - r_halt)) / 3
        full['reward_plain_step'] = (r_wait + r_trip) / 2
        reward_label = f'TotalWaitingTime + (TP+SpeedStd+Halt)/3  |  μ_wait={mu_wait:.2f}  σ_wait={std_wait:.2f}'
    elif reward_mode == 'wait_haltratio':
        # Megegyezik a sumo_rl_environment.py wait_haltratio móddal:
        # reward = (r_wait_inv + r_halt_inv) / 2
        wait = full['TotalWaitingTime'].values if 'TotalWaitingTime' in full.columns else np.zeros(len(full))
        halt = full['HaltRatio'].values        if 'HaltRatio'        in full.columns else np.zeros(len(full))
        r_wait_inv = _logtanh_neg(wait + epsilon, mu_wait, std_wait)
        r_halt_inv = _logtanh_neg(np.abs(halt) + epsilon, mu_halt, std_halt)
        full['reward_plain_step'] = (r_wait_inv + r_halt_inv) / 2
        reward_label = f'TotalWaitingTime + HaltRatio  |  μ_wait={mu_wait:.2f}  σ_wait={std_wait:.2f}  |  μ_halt={mu_halt:.2f}  σ_halt={std_halt:.2f}'
    elif reward_mode == 'halt_ratio':
        halt = full['HaltRatio'].values if 'HaltRatio' in full.columns else np.zeros(len(full))
        full['reward_plain_step'] = _logtanh_neg(np.abs(halt) + epsilon, mu_halt, std_halt)
        reward_label = f'HaltRatio  |  μ={mu_halt:.2f}  σ={std_halt:.2f}'
    elif reward_mode == 'co2_speedstd':
        co2  = full['AvgCO2'].values   if 'AvgCO2'   in full.columns else np.ones(len(full))
        sstd = full['SpeedStd'].values if 'SpeedStd' in full.columns else np.ones(len(full))
        r_co2  = _logtanh_neg(co2 + epsilon,  mu_halt,  std_halt)   # MU_CO2 nincs külön itt, fallback
        r_sstd = _logtanh_pos(sstd + epsilon, mu_sstd,  std_sstd)
        full['reward_plain_step'] = (r_co2 + r_sstd) / 2
        reward_label = f'CO2 + SpeedStd'
    else:  # speed_throughput (default)
        spd = full['AvgSpeed'].values
        thr = full['Throughput'].values if 'Throughput' in full.columns else np.ones(len(full))
        r_s = _logtanh_pos(spd, mu_s, std_s)
        r_t = _logtanh_pos(thr, mu_t, std_t)
        full['reward_plain_step'] = (r_s + r_t) / 2
        reward_label = f'Speed + Throughput  |  μ_s={mu_s:.3f}  σ_s={std_s:.3f}  |  μ_tp={mu_t:.3f}  σ_tp={std_t:.3f}'

    run_ids     = sorted(full['run_id'].unique())
    n_runs      = len(run_ids)
    flow_levels = sorted(full['flow_level'].unique())
    flow_norm   = {fl: i / max(len(flow_levels) - 1, 1) for i, fl in enumerate(flow_levels)}
    cmap        = plt.cm.get_cmap('RdYlBu_r')
    EMA_ALPHA   = 0.05

    # Vezérlési módok csoportosítása (random / mixed / actuated)
    def _mode_group(cm):
        if cm == 'actuated': return 'actuated'
        if str(cm).startswith('mixed'): return 'mixed'
        return 'random'
    full['mode_group'] = full['control_mode'].apply(_mode_group)
    mode_order  = ['random', 'mixed', 'actuated']
    mode_colors = {'random': '#e74c3c', 'mixed': '#f39c12', 'actuated': '#2980b9'}
    mode_titles = {
        'random':   'Random kontroll  (ϵ=1.0)',
        'mixed':    f'Mixed kontroll  (ϵ={MIXED_EPSILON})',
        'actuated': 'Actuated kontroll  (ϵ=0.0)',
    }
    present_modes = [m for m in mode_order if m in full['mode_group'].values]
    n_panels = max(len(present_modes), 1)

    title_jid = junction_filter if junction_filter else 'összes junction'
    fig, axes = plt.subplots(1, n_panels, figsize=(max(16, 14 * n_panels), 6),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, mg in zip(axes, present_modes):
        sub = full[full['mode_group'] == mg]
        sub_run_ids = sorted(sub['run_id'].unique())
        all_smoothed = []
        base_color = mode_colors.get(mg, '#555555')

        for run_id in sub_run_ids:
            rd = sub[sub['run_id'] == run_id].sort_values('step')
            if len(rd) < 5:
                continue
            steps  = rd['step'].values
            reward = rd['reward_plain_step'].values
            color  = cmap(flow_norm.get(rd['flow_level'].iloc[0], 0.5))

            ax.plot(steps, reward, color=color, alpha=0.15, linewidth=0.6)

            ema = np.zeros_like(reward, dtype=float)
            ema[0] = reward[0]
            for i in range(1, len(reward)):
                ema[i] = EMA_ALPHA * reward[i] + (1 - EMA_ALPHA) * ema[i - 1]
            ax.plot(steps, ema, color=color, alpha=0.5, linewidth=1.0)
            all_smoothed.append((steps, ema))

        if all_smoothed:
            max_step     = max(s[-1] for s, _ in all_smoothed)
            common_steps = np.arange(0, int(max_step) + 1)
            interp_rows  = [np.interp(common_steps, s, e)
                            for s, e in all_smoothed if len(s) > 1]
            if interp_rows:
                gm = np.mean(interp_rows, axis=0)
                ema_g = np.zeros_like(gm)
                ema_g[0] = gm[0]
                for i in range(1, len(gm)):
                    ema_g[i] = EMA_ALPHA * gm[i] + (1 - EMA_ALPHA) * ema_g[i - 1]
                ax.plot(common_steps, ema_g, color=base_color, linewidth=2.8,
                        label=f'Átlag (n={len(interp_rows)} run)', zorder=10)

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=min(flow_levels), vmax=max(flow_levels)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
        cbar.set_label('Flow level (veh/h)', fontsize=9)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Normalized reward  [0 – 1]', fontsize=11)
        ax.set_title(mode_titles.get(mg, mg), fontsize=13,
                     color=base_color, fontweight='bold')
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'Step-szintű reward — {reward_label}  |  {title_jid}  |  {param_label} params\n'
        f'{n_runs} run  |  EMA α={EMA_ALPHA}',
        fontsize=11)

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
            if '_actuated.csv' in csv_file:
                df['control_mode'] = 'actuated'
            elif '_mixed' in csv_file:
                _cm = csv_file.split('_ep')[1].split('_', 1)
                df['control_mode'] = 'mixed_' + _cm[1].replace('.csv', '') if len(_cm) > 1 else 'mixed'
            else:
                df['control_mode'] = 'random'
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

    # State-korreláció szűrő: a reward legyen összefüggésben az ágens állapotterével.
    # Ha avg|r(reward, state_col)| < küszöb → a reward nem tartalmaz tanulható gradienst.
    # (ref: Mnih et al. 2015 DQN: reward must correlate with observable state)
    _sc_filter_cols = ['det_occ_mean', 'det_occ_max', 'det_veh_sum',
                       'det_speed_mean', 'det_speed_min']
    _sc_filter_avail = [c for c in _sc_filter_cols if c in df_valid.columns]
    _SC_CORR_THRESHOLD = 0.20   # avg|r| < 0.20 → GYENGE / kuka
    if _sc_filter_avail:
        print(f"  State-korreláció szűrő AKTÍV: {_sc_filter_avail}")
        print(f"  Küszöb: avg|r(reward, state)| >= {_SC_CORR_THRESHOLD}  (ref: Mnih et al. 2015 DQN)")
    else:
        print("  [State-korreláció szűrő SKIP] Nincs detektor state adat.")

    # Anti-starvation teszt (Varaiya 2013, TRC-36; Wei et al. 2019, KDD PressLight):
    # A reward érzékeny-e az INAKTÍV irányok torlódására?
    # Módszer: phase-kondicionált Pearson r(reward, max_inactive_occ | phase=p).
    # Ha r < -0.3 → reward bünteti a starvation-t → PASS.
    # (Referencia: "max pressure" elvén alapuló phase-fairness validáció)
    _per_det_occ_pat = re.compile(r'^d\d+_occ$')
    antistary_det_cols = sorted([c for c in df_valid.columns if _per_det_occ_pat.match(c)])
    antistary_available = bool(antistary_det_cols) and 'phase' in df_valid.columns
    if antistary_available:
        print(f"  Anti-starvation teszt AKTÍV: {antistary_det_cols}  (ref: Varaiya 2013 TRC-36)")
    else:
        print("  [Anti-starvation teszt SKIP] Nincs 'phase' v. per-det occ adat.")

    combo_results = []

    for combo in all_combos:
        combo_name = " + ".join(combo)

        # Per-junction reward számítás és aggregálás
        all_rewards = []
        all_flow_levels = []
        combo_monotonicity_per_junction = []
        combo_iqr_per_junction = []
        combo_eta2_per_junction = []
        combo_antistary_per_junction = []
        combo_state_corr_per_junction = []

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

            # State-korreláció: avg|Pearson r(rewards, state_col)| per junction
            # ref: Mnih et al. 2015 DQN — reward ↔ observable state gradiens
            if _sc_filter_avail:
                sc_rs = []
                for _sc in _sc_filter_avail:
                    if _sc in jdf_reset.columns:
                        _sc_vals = jdf_reset[_sc].values
                        if _sc_vals.std() > 1e-6 and rewards.std() > 1e-6:
                            _r_sc, _ = stats.pearsonr(_sc_vals, rewards)
                            sc_rs.append(abs(_r_sc))
                if sc_rs:
                    combo_state_corr_per_junction.append(float(np.mean(sc_rs)))

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

            # Anti-starvation: phase-kondicionált corr(reward, max_inactive_occ)
            # Ref: Varaiya (2013) TRC-36; Wei et al. (2019) KDD PressLight
            if antistary_available:
                det_cols_j = [c for c in antistary_det_cols if c in jdf_reset.columns]
                if det_cols_j and len(det_cols_j) > 1:
                    phase_corrs_j = []
                    for ph in jdf_reset['phase'].unique():
                        ph_mask = jdf_reset['phase'].values == ph
                        if ph_mask.sum() < 20:
                            continue
                        # Aktív detektor: legalacsonyabb átlag-occ ennél a fázisnál
                        ph_occ_means = jdf_reset.loc[ph_mask, det_cols_j].mean()
                        active_d = ph_occ_means.idxmin()
                        inactive_ds = [d for d in det_cols_j if d != active_d]
                        max_inact = jdf_reset.loc[ph_mask, inactive_ds].max(axis=1).values
                        ph_rewards = rewards[ph_mask]
                        if max_inact.std() < 1e-6 or len(ph_rewards) < 20:
                            continue
                        rho_as, _ = stats.pearsonr(max_inact, ph_rewards)
                        phase_corrs_j.append(rho_as)
                    if phase_corrs_j:
                        combo_antistary_per_junction.append(float(np.mean(phase_corrs_j)))

        if not combo_monotonicity_per_junction:
            continue

        all_rewards = np.array(all_rewards)
        all_flow_levels = np.array(all_flow_levels)

        # Aggregált statisztikák
        avg_monotonicity = np.mean(combo_monotonicity_per_junction)
        avg_iqr = np.mean(combo_iqr_per_junction)
        avg_eta2 = np.mean(combo_eta2_per_junction) if combo_eta2_per_junction else 1.0

        # State-korreláció aggregálás
        if combo_state_corr_per_junction:
            avg_state_r = float(np.mean(combo_state_corr_per_junction))
        else:
            avg_state_r = float('nan')
        state_corr_ok = (not np.isnan(avg_state_r)) and (avg_state_r >= _SC_CORR_THRESHOLD)
        mono_pass_pct = np.mean([r < -0.5 for r in combo_monotonicity_per_junction]) * 100
        iqr_pass_pct = np.mean([q > 0.10 for q in combo_iqr_per_junction]) * 100

        # Szűrők
        mono_ok = avg_monotonicity < -0.5
        iqr_ok = avg_iqr > 0.10

        # Anti-starvation szűrő
        if combo_antistary_per_junction:
            avg_antistary_r = float(np.mean(combo_antistary_per_junction))
        else:
            avg_antistary_r = float('nan')
        antistary_ok = (not np.isnan(avg_antistary_r)) and (avg_antistary_r < -0.3)

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
            'avg_antistary_r': avg_antistary_r,
            'antistary_ok': antistary_ok,
            'avg_state_r': avg_state_r,
            'state_corr_ok': state_corr_ok,
            'passed': (mono_ok and iqr_ok
                       and (antistary_ok or np.isnan(avg_antistary_r))
                       and (state_corr_ok or np.isnan(avg_state_r))),
            'is_current': False,
        })

    # --- Eredmények kiírása ---
    print(f"\n  {'Kombináció':40} {'Mono_r':>8} {'Mono%':>6} {'IQR':>6} {'IQR%':>6} "
          f"{'η²':>6} {'1-η²':>6} {'AS_r':>6} {'SC_r':>6} {'Szűrő':>8}")
    print("  " + "-" * 120)

    # Rendezés: átmenők first, azon belül eta2 ascending (alacsonyabb = jobb)
    combo_results.sort(key=lambda x: (not x['passed'], x['avg_eta2']))

    for cr in combo_results:
        status = "✓ PASS" if cr['passed'] else "✗ FAIL"
        mono_flag = "" if cr['mono_ok'] else " [MONO!]"
        iqr_flag = "" if cr['iqr_ok'] else " [IQR!]"
        as_flag = "" if (cr['antistary_ok'] or np.isnan(cr['avg_antistary_r'])) else " [AS!]"
        sc_flag = "" if (cr['state_corr_ok'] or np.isnan(cr['avg_state_r'])) else " [SC!]"
        as_str = f"{cr['avg_antistary_r']:+6.3f}" if not np.isnan(cr['avg_antistary_r']) else '   N/A'
        sc_str = f"{cr['avg_state_r']:6.3f}" if not np.isnan(cr['avg_state_r']) else '   N/A'

        print(f"  {cr['combo']:40} {cr['avg_monotonicity']:+8.3f} {cr['mono_pass_pct']:5.0f}% "
              f"{cr['avg_iqr']:6.3f} {cr['iqr_pass_pct']:5.0f}% "
              f"{cr['avg_eta2']:6.3f} {cr['eta2_within']:6.3f} "
              f"{as_str} {sc_str} {status}{mono_flag}{iqr_flag}{as_flag}{sc_flag}")

    # --- Átmenők rangsorolása ---
    passed = [cr for cr in combo_results if cr['passed']]
    failed = [cr for cr in combo_results if not cr['passed']]

    print(f"\n  Átment a szűrőn: {len(passed)}/{len(combo_results)}")
    print(f"  Kiesett:          {len(failed)}/{len(combo_results)}")
    if antistary_available:
        n_as_fail = sum(1 for cr in combo_results
                        if not cr['antistary_ok'] and not np.isnan(cr['avg_antistary_r']))
        print(f"  Ebből [AS!] kiesett: {n_as_fail}  (phase-kondicionált anti-starvation, Varaiya 2013 TRC-36)")
    if _sc_filter_avail:
        n_sc_fail = sum(1 for cr in combo_results
                        if not cr['state_corr_ok'] and not np.isnan(cr['avg_state_r']))
        print(f"  Ebből [SC!] kiesett: {n_sc_fail}  (avg|r(reward,state)| < {_SC_CORR_THRESHOLD}, Mnih et al. 2015)")

    if passed:
        print(f"\n  --- RANGSOR (alacsonyabb η² = jobb: az ágens hatása jobban látszik) ---")
        print(f"  {'#':>3} {'Kombináció':40} {'η²':>8} {'1-η²':>8} {'Mono_r':>8} {'IQR':>6} {'AS_r':>6} {'SC_r':>6}")
        print("  " + "-" * 98)
        for i, cr in enumerate(passed):
            as_str = f"{cr['avg_antistary_r']:+6.3f}" if not np.isnan(cr['avg_antistary_r']) else '   N/A'
            sc_str = f"{cr['avg_state_r']:6.3f}" if not np.isnan(cr['avg_state_r']) else '   N/A'
            print(f"  {i+1:3} {cr['combo']:40} {cr['avg_eta2']:8.4f} {cr['eta2_within']:8.4f} "
                  f"{cr['avg_monotonicity']:+8.3f} {cr['avg_iqr']:6.3f} {as_str} {sc_str}")

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

    # State változók: aggregált + per-detektor occupancy (d0_occ, d1_occ, ...)
    agg_state_cols = ['det_occ_mean', 'det_occ_max', 'det_veh_sum', 'det_speed_mean', 'det_speed_min']
    per_det_occ_cols = sorted([c for c in full_df.columns
                                if c.startswith('d') and c.endswith('_occ')
                                and c[1:-4].isdigit()],
                               key=lambda c: int(c[1:-4]))
    state_cols = agg_state_cols + per_det_occ_cols
    # FIGYELEM: det_veh_sum <-> Throughput cirkulárisan összefügg (azonos forrásadat)!

    # Ellenőrzés: vannak-e detektor oszlopok az adatban?
    has_det_data = all(col in full_df.columns for col in agg_state_cols)

    # --- Helper: state↔reward elemzés egy df szeletre ---
    def _state_reward_section(df_slice, sc_list, agg_sc, per_det_cols, reward_cols, label):
        """Pearson + MI táblázat kiírása egy adott df szeletre."""
        print(f"\n{'─' * 100}")
        print(f"  [{label}]  n={len(df_slice)} sor,  {len(sc_list)} state oszlop")
        print(f"{'─' * 100}")

        if per_det_cols:
            print(f"  [i] Per-detektor occ: {', '.join(per_det_cols)}")
            print(f"  [!] det_veh_sum <-> Throughput: cirkuláris — MI mesterségesen magas!")

        # --- Pearson korreláció ---
        print(f"\n  Pearson r  (** |r|>0.5  * |r|>0.3)")
        header = f"  {'':22}"
        for sc in sc_list:
            header += f" {sc[:9]:>10}"
        print(header)
        print("  " + "─" * (22 + 11 * len(sc_list)))

        corr_results = {}
        for rm in reward_cols:
            if rm not in df_slice.columns:
                continue
            line = f"  {rm:22}"
            corrs = []
            for sc in sc_list:
                if sc not in df_slice.columns:
                    line += f" {'N/A':>10}"
                    corrs.append(0.0)
                    continue
                sub = df_slice[[sc, rm]].dropna()
                sub = sub[(sub[sc] != 0) | (sub[rm] != 0)]
                if len(sub) > 30:
                    r, _ = stats.pearsonr(sub[sc].values, sub[rm].values)
                    corrs.append(abs(r))
                    mk = "**" if abs(r) > 0.5 else "* " if abs(r) > 0.3 else "  "
                    line += f" {r:+8.3f}{mk}"
                else:
                    corrs.append(0.0)
                    line += f" {'N/A':>10}"
            print(line)
            corr_results[rm] = np.mean(corrs)

        # --- MI ---
        print(f"\n  Gaussi MI  (MI ≈ -0.5·ln(1-r²) [nats])")
        print(f"  {'Reward metrika':25} {'Σ MI':>10} {'Max MI':>10} {'legjobb state':>18}")
        print("  " + "─" * 68)
        mi_results = {}
        for rm in reward_cols:
            if rm not in df_slice.columns:
                continue
            total_mi, max_mi, best_sc = 0.0, 0.0, ""
            for sc in sc_list:
                if sc not in df_slice.columns:
                    continue
                sub = df_slice[[sc, rm]].dropna()
                sub = sub[(sub[sc] != 0) | (sub[rm] != 0)]
                if len(sub) > 30:
                    r, _ = stats.pearsonr(sub[sc].values, sub[rm].values)
                    r2 = min(r**2, 0.9999)
                    mi = -0.5 * np.log(1 - r2)
                    total_mi += mi
                    if mi > max_mi:
                        max_mi, best_sc = mi, sc
            mi_results[rm] = total_mi
            circ = " [CIRK!]" if rm == 'Throughput' and 'det_veh_sum' in sc_list else ""
            print(f"  {rm:25} {total_mi:10.4f} {max_mi:10.4f} {best_sc:>18}{circ}")

        # --- Összefoglaló rangsor ---
        print(f"\n  avg|r| rangsor:")
        for rm, avg_r in sorted(corr_results.items(), key=lambda x: -x[1]):
            strength = "ERŐS" if avg_r > 0.4 else "KÖZEPES" if avg_r > 0.25 else "GYENGE"
            print(f"    {rm:28} avg|r|={avg_r:.3f}  ({strength})")

        return corr_results, mi_results

    # --- Reward metrika lista (common) ---
    triplet_cols_5 = [f'triplet_{label}' for (_, _, _, _, label) in TRIPLET_DEFS
                      if f'triplet_{label}' in df_valid.columns]
    reward_metrics_5 = [r for r in
                        ['AvgSpeed', 'Throughput', 'reward_plain', 'reward_halt'] + triplet_cols_5
                        if r in df_valid.columns]

    print("\n" + "=" * 100)
    print("  5. STATE ↔ REWARD KONZISZTENCIA")
    print("     Az ágens state-je (detektor-szintű) mennyire predikálja a reward-ot?")
    print("     Ha a korreláció gyenge → az ágens a state-ből nem tudja megtanulni a reward-ot.")
    print("=" * 100)

    # --- Per-junction elemzés ---
    junction_corr_summary = {}   # jid → {rm: avg_r}
    if has_det_data and 'junction' in df_valid.columns:
        for jid in sorted(df_valid['junction'].unique()):
            jdf = df_valid[df_valid['junction'] == jid].copy()
            # Per-detektor occ oszlopok amik ebben a junction-ben ténylegesen vannak
            j_per_det = [c for c in per_det_occ_cols if c in jdf.columns
                         and jdf[c].notna().any() and jdf[c].abs().sum() > 0]
            j_state_cols = agg_state_cols + j_per_det
            corr_res, mi_res = _state_reward_section(
                jdf, j_state_cols, agg_state_cols, j_per_det, reward_metrics_5,
                label=f"Junction: {jid}")
            junction_corr_summary[jid] = corr_res

    # --- Globális elemzés (minden junction együtt) ---
    print("\n" + "─" * 100)
    print("  GLOBÁLIS ÖSSZESÍTÉS (minden junction együtt):")
    state_reward_corr = {}
    mi_scores = {}
    if has_det_data:
        state_reward_corr, mi_scores = _state_reward_section(
            df_valid, state_cols, agg_state_cols, per_det_occ_cols, reward_metrics_5,
            label="GLOBÁLIS")

    # --- Kereszt-összehasonlítás: reward × junction heatmap szövegesen ---
    if junction_corr_summary and reward_metrics_5:
        print(f"\n  JUNCTION × REWARD avg|r| összehasonlítás:")
        jids = sorted(junction_corr_summary.keys())
        # csak reward_plain és reward_halt + esetleg triplet-ek
        show_rewards = [r for r in reward_metrics_5 if r not in ('AvgSpeed', 'Throughput')]
        if show_rewards:
            header_line = f"  {'Reward':28}"
            for jid in jids:
                header_line += f" {jid[:10]:>12}"
            header_line += f" {'GLOBÁLIS':>12}"
            print(header_line)
            print("  " + "─" * (28 + 13 * (len(jids) + 1)))
            for rm in show_rewards:
                line = f"  {rm:28}"
                for jid in jids:
                    val = junction_corr_summary[jid].get(rm, 0.0)
                    line += f" {val:12.3f}"
                g_val = state_reward_corr.get(rm, 0.0)
                line += f" {g_val:12.3f}"
                print(line)

    # --- 5b. Aggregált state redundancia + VIF (globális) ---
    if has_det_data:
        state_only = agg_state_cols
        state_df = df_valid[state_only].dropna()
        if len(state_df) > 50:
            print(f"\n{'─' * 100}")
            print(f"  5b. Aggregált state változók redundanciája (globális, VIF):")
            print(f"{'─' * 100}")
            from numpy.linalg import lstsq as _lstsq
            state_arr = state_df.values
            state_means = state_arr.mean(axis=0)
            state_stds  = state_arr.std(axis=0)
            state_stds[state_stds == 0] = 1.0
            state_norm = (state_arr - state_means) / state_stds
            print(f"  {'State változó':20} {'VIF':>8} {'Értékelés':>15}")
            print("  " + "─" * 46)
            for j, sc in enumerate(state_only):
                y = state_norm[:, j]
                X = np.delete(state_norm, j, axis=1)
                X = np.column_stack([X, np.ones(len(X))])
                coeffs, _, _, _ = _lstsq(X, y, rcond=None)
                y_pred = X @ coeffs
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 0.9999 else 9999.0
                verdict = "ERŐS MULTIKOL" if vif > 10 else "KÖZEPES" if vif > 5 else "OK"
                print(f"  {sc:20} {vif:8.2f} {verdict:>15}")
    else:
        print("\n  [SKIP] Nincs detektor-szintű adat a CSV-kben.")
        print("  Futtasd újra a szimulációt (--skip-simulation NÉLKÜL)!")

    # =====================================================================
    # 6. ÁBRÁK
    # =====================================================================

    # Ábra 1: Korreláció mátrix — módok szerint bontva (random / mixed / actuated)
    plot_metrics = [m for m in candidate_metrics
                    if not m.startswith('reward_') and not m.startswith('triplet_')]
    n_met = len(plot_metrics)
    _mc_order  = ['random', 'mixed', 'actuated']
    _mc_colors = {'random': '#e74c3c', 'mixed': '#f39c12', 'actuated': '#2980b9'}
    _mc_titles = {'random': 'Random (ϵ=1.0)', 'mixed': f'Mixed (ϵ={MIXED_EPSILON})', 'actuated': 'Actuated (ϵ=0.0)'}
    def _mg_c(cm): return 'actuated' if cm == 'actuated' else ('mixed' if str(cm).startswith('mixed') else 'random')
    if 'mode_group' not in df_valid.columns:
        df_valid['mode_group'] = df_valid['control_mode'].apply(_mg_c)
    present_mc = [m for m in _mc_order if m in df_valid['mode_group'].values]
    n_mc = max(len(present_mc), 1)
    fig_sq = max(8, n_met * 1.1)
    fig, axes_mc = plt.subplots(1, n_mc, figsize=(fig_sq * n_mc, fig_sq * 0.88))
    if n_mc == 1:
        axes_mc = [axes_mc]
    for ax_mc, mg_mc in zip(axes_mc, present_mc):
        sub_mc = df_valid[df_valid['mode_group'] == mg_mc]
        cd_mc = {}
        for col in plot_metrics:
            if col in sub_mc.columns:
                v = sub_mc[col].values
                if len(v[v > 0]) > 5:
                    cd_mc[col] = np.log(np.clip(v, epsilon, None) + epsilon)
        cd_cols_mc = [c for c in plot_metrics if c in cd_mc]
        if cd_cols_mc:
            cm_sub = pd.DataFrame({c: cd_mc[c] for c in cd_cols_mc}).corr(method='pearson')
            vals_mc = cm_sub.values
        else:
            cd_cols_mc = plot_metrics[:1] if plot_metrics else []
            vals_mc = np.eye(len(cd_cols_mc))
        n_c = len(cd_cols_mc)
        im = ax_mc.imshow(vals_mc, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_mc.set_xticks(range(n_c))
        ax_mc.set_xticklabels(cd_cols_mc, rotation=45, ha='right', fontsize=8)
        ax_mc.set_yticks(range(n_c))
        ax_mc.set_yticklabels(cd_cols_mc, fontsize=8)
        for i in range(n_c):
            for j in range(n_c):
                val = vals_mc[i, j]
                col_t = 'white' if abs(val) > 0.6 else 'black'
                w = 'bold' if abs(val) > 0.85 and i != j else 'normal'
                ax_mc.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=7, color=col_t, fontweight=w)
        fig.colorbar(im, ax=ax_mc, shrink=0.75)
        ax_mc.set_title(
            f'{_mc_titles.get(mg_mc, mg_mc)}  (n={len(sub_mc)})'
            f'\nMetric Correlation Matrix  |  bold = redundant (|r|>0.85)',
            fontsize=10, color=_mc_colors.get(mg_mc, 'black'), fontweight='bold')
    fig.suptitle('Metric Correlation Matrix (log-transformed)  |  per Control Mode',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_correlation_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  reward_correlation_matrix.png mentve ({n_met} egyedi metrika, {n_mc} mód)")

    # Ábra 2: Top 20 kombináció — η² + monotonitás + IQR
    if combo_results:
        sorted_cr = sorted(combo_results, key=lambda x: (not x['passed'], x['avg_eta2']))
        show_cr = sorted_cr[:20]
        names = [cr['combo'] for cr in show_cr]
        n_show = len(names)

        fig, axes = plt.subplots(1, 5, figsize=(40, max(8, n_show * 0.45)))
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

        ax = axes[3]
        as_vals = [cr['avg_antistary_r'] if not np.isnan(cr['avg_antistary_r']) else 0.0
                   for cr in show_cr]
        as_colors = ['#2ecc71' if cr['antistary_ok'] else
                     ('#95a5a6' if np.isnan(cr['avg_antistary_r']) else '#e74c3c')
                     for cr in show_cr]
        ax.barh(y_pos, as_vals, color=as_colors, alpha=0.85)
        ax.axvline(-0.3, color='red', ls='--', lw=2, alpha=0.7)
        for i, v in enumerate(as_vals):
            ax.text(v - 0.01 if v < 0 else v + 0.01, i,
                    f'{v:+.2f}', va='center', ha='right' if v < 0 else 'left', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([], fontsize=9)
        ax.set_xlabel('r(reward, max_inactive_occ | phase)  [Varaiya 2013]')
        ax.set_title('Anti-starvation\n(< −0.3 = PASS)', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[4]
        sc_vals = [cr['avg_state_r'] if not np.isnan(cr['avg_state_r']) else 0.0
                   for cr in show_cr]
        sc_colors = ['#2ecc71' if cr['state_corr_ok'] else
                     ('#95a5a6' if np.isnan(cr['avg_state_r']) else '#e74c3c')
                     for cr in show_cr]
        ax.barh(y_pos, sc_vals, color=sc_colors, alpha=0.85)
        ax.axvline(0.20, color='red', ls='--', lw=2, alpha=0.7)
        for i, v in enumerate(sc_vals):
            ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([], fontsize=9)
        ax.set_xlabel('avg|r(reward, state)| per junction  [Mnih 2015]')
        ax.set_title('State Correlation\n(\u2265 0.20 = PASS)', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(f'Top {n_show} Metric Combinations (green=PASS, red=FAIL)\n'
                     'Filters: Monotonicity [MONO] | IQR | Anti-starvation [AS] (Varaiya 2013 TRC-36) | State-Corr [SC] (Mnih 2015)',
                     fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_combo_selection.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_combo_selection.png mentve")

    # Ábra 2b: ÖSSZES kombináció
    if combo_results:
        sorted_cr_all = sorted(combo_results, key=lambda x: (not x['passed'], x['avg_eta2']))
        names_all = [cr['combo'] for cr in sorted_cr_all]
        n_all = len(names_all)

        fig, axes = plt.subplots(1, 5, figsize=(42, max(10, n_all * 0.4)))
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

        ax = axes[3]
        as_all = [cr['avg_antistary_r'] if not np.isnan(cr['avg_antistary_r']) else 0.0
                  for cr in sorted_cr_all]
        as_colors_all = ['#2ecc71' if cr['antistary_ok'] else
                         ('#95a5a6' if np.isnan(cr['avg_antistary_r']) else '#e74c3c')
                         for cr in sorted_cr_all]
        ax.barh(y_all, as_all, color=as_colors_all, alpha=0.85)
        ax.axvline(-0.3, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_all)
        ax.set_yticklabels([], fontsize=7)
        ax.set_xlabel('r(reward, max_inactive_occ | phase)  [Varaiya 2013]')
        ax.set_title('Anti-starvation\n(< −0.3 = PASS)', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        ax = axes[4]
        sc_all = [cr['avg_state_r'] if not np.isnan(cr['avg_state_r']) else 0.0
                  for cr in sorted_cr_all]
        sc_colors_all = ['#2ecc71' if cr['state_corr_ok'] else
                         ('#95a5a6' if np.isnan(cr['avg_state_r']) else '#e74c3c')
                         for cr in sorted_cr_all]
        ax.barh(y_all, sc_all, color=sc_colors_all, alpha=0.85)
        ax.axvline(0.20, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_yticks(y_all)
        ax.set_yticklabels([], fontsize=7)
        ax.set_xlabel('avg|r(reward, state)| per junction  [Mnih 2015]')
        ax.set_title('State Correlation\n(\u2265 0.20 = PASS)', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        fig.suptitle(f'All {n_all} Metric Combinations (green=PASS, red=FAIL)\n'
                     'Filters: Monotonicity [MONO] | IQR | Anti-starvation [AS] (Varaiya 2013 TRC-36) | State-Corr [SC] (Mnih 2015)',
                     fontsize=12)
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

    # Ábra 4 előtt: 3-as és 4-es kombók state-korreláció alapú rangsorolása (R1C1_C)
    _combo_jid = 'R1C1_C'
    _combo_df  = df_valid[df_valid['junction'] == _combo_jid].copy() if 'junction' in df_valid.columns else df_valid
    _combo_sc  = agg_state_cols

    combo_search_metrics = [m for m in
        ['TotalWaitingTime', 'TotalCO2', 'QueueLength', 'TotalTravelTime',
         'HaltRatio', 'SpeedStd', 'AvgOccupancy', 'AvgSpeed', 'Throughput']
        if m in _combo_df.columns]

    if has_det_data and len(_combo_df) > 0 and len(combo_search_metrics) >= 3:
        print(f"\n{'=' * 100}")
        print(f"  5e. 3-as és 4-es kombók state-korreláció rangsor [{_combo_jid}]")
        print(f"      Minden kombóra: reward = átlag(per-metrika log-tanh normált érték)")
        print(f"      Rang: avg|Pearson r| a reward és az 5 aggregált state változó között")
        print(f"{'=' * 100}")

        # Előre kiszámoljuk az r-értékeket minden nyers metrikára × state oszlopra
        # hogy ne kelljen N^k × 5 Pearson hívás
        base_r = {}   # metric → np.array([r_sc0, r_sc1, ...]) per state col
        for m in combo_search_metrics:
            rs = []
            for sc in _combo_sc:
                sub = _combo_df[[sc, m]].dropna()
                sub = sub[(sub[sc] != 0) | (sub[m] != 0)]
                if len(sub) > 30:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        r, _ = stats.pearsonr(sub[sc].values, sub[m].values)
                else:
                    r = 0.0
                rs.append(r)
            base_r[m] = np.array(rs)  # shape: (5,)

        # Metrika iránya: "magasabb = jobb" (speed) vs "alacsonyabb = jobb" (co2, wait stb.)
        higher_better = {'AvgSpeed', 'Throughput'}

        # Kombináció reward-jának korrelációja a state-tel:
        # reward = mean(r_m1, r_m2, ...) ahol r_mi = tanh-normált metrika _combo_df-ből
        # A korreláció közelíthető a komponens r-ek átlagával (lineáris közelítés).
        # De pontosabb: a reward oszlopot ténylegesen kiszámolni és pearsonr-t hívni.
        # A gyors változat: az avg|r|-t a komponens r-ek átlagaként becsüljük.

        all_combo_scores = []

        for combo_size in [3, 4]:
            for combo in itertools.combinations(combo_search_metrics, combo_size):
                # Gyors becslés: avg|r| ≈ mean(|r_mi_sc| for m in combo, sc in state_cols)
                # Ez linearitást feltételez, de elég a rangsoroláshoz
                component_abs_r = np.array([np.abs(base_r[m]) for m in combo])  # (k, 5)
                # A reward i-edik fázisban: átlag a komponensek irányított r-jein
                # Irány: "lower is better" metrikánál a reward inverze a metrikának
                signed_r = []
                for m in combo:
                    if m in higher_better:
                        signed_r.append(base_r[m])     # pozitív r = magas metrika = magas reward → OK
                    else:
                        signed_r.append(-base_r[m])    # negatív r = magas metrika = alacsony reward → flip
                avg_signed_r = np.mean(signed_r, axis=0)   # (5,)  — a reward-state r becslése
                avg_abs_r = float(np.mean(np.abs(avg_signed_r)))
                max_abs_r = float(np.max(np.abs(avg_signed_r)))

                all_combo_scores.append({
                    'combo': ' + '.join(combo),
                    'size': combo_size,
                    'avg_abs_r': avg_abs_r,
                    'max_abs_r': max_abs_r,
                    'r_per_state': avg_signed_r.tolist(),
                })

        all_combo_scores.sort(key=lambda x: -x['avg_abs_r'])

        print(f"\n  Top 20 kombó (avg|r| a reward és az 5 state változó között, [{_combo_jid}]):")
        print(f"  {'#':>3}  {'Kombináció':55} {'avg|r|':>8} {'max|r|':>8}  "
              + '  '.join(f"{sc[:8]:>8}" for sc in _combo_sc))
        print("  " + "─" * (3 + 2 + 55 + 9 + 9 + 2 + 10 * len(_combo_sc)))

        for i, entry in enumerate(all_combo_scores[:20]):
            r_str = '  '.join(f"{v:+8.3f}" for v in entry['r_per_state'])
            print(f"  {i+1:3}  {entry['combo']:55} {entry['avg_abs_r']:8.3f} "
                  f"{entry['max_abs_r']:8.3f}  {r_str}")

        print(f"\n  Összes vizsgált kombó: {len(all_combo_scores)} "
              f"(3-as: {sum(1 for x in all_combo_scores if x['size']==3)}, "
              f"4-es: {sum(1 for x in all_combo_scores if x['size']==4)})")

    # Ábra 4: State ↔ Reward heatmap + MI  — R1C1_C, csak 5 aggregált state változó
    _plot_jid = 'R1C1_C'
    _plot_sc  = agg_state_cols  # csak az 5 aggregált
    _plot_df_all = df_valid[df_valid['junction'] == _plot_jid].copy() if 'junction' in df_valid.columns else df_valid.copy()
    if has_det_data and len(_plot_df_all) > 0:
        # Nyers metrikák + normalizált reward jelöltek
        raw_metrics = [m for m in ['TotalWaitingTime', 'TotalCO2', 'QueueLength',
                                    'TotalTravelTime', 'HaltRatio', 'SpeedStd', 'AvgOccupancy']
                       if m in _plot_df_all.columns]
        reward_metrics_to_test = raw_metrics + reward_metrics_5
        n_rewards = len(reward_metrics_to_test)
        n_states  = len(_plot_sc)

        # Módok csoportosítása
        _sm_order  = ['random', 'mixed', 'actuated']
        _sm_colors = {'random': '#e74c3c', 'mixed': '#f39c12', 'actuated': '#2980b9'}
        _sm_titles = {'random': 'Random (ϵ=1.0)', 'mixed': f'Mixed (ϵ={MIXED_EPSILON})', 'actuated': 'Actuated (ϵ=0.0)'}
        def _mg_s(cm): return 'actuated' if cm == 'actuated' else ('mixed' if str(cm).startswith('mixed') else 'random')
        if 'mode_group' not in _plot_df_all.columns:
            _plot_df_all['mode_group'] = _plot_df_all['control_mode'].apply(_mg_s)
        present_sm = [m for m in _sm_order if m in _plot_df_all['mode_group'].values]
        n_sm = max(len(present_sm), 1)

        fig, axes_sc = plt.subplots(n_sm, 2,
                                    figsize=(14, max(5, n_rewards * 0.75) * n_sm),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
        if n_sm == 1:
            axes_sc = [axes_sc]

        for row_sc, mg_sc in enumerate(present_sm):
            _plot_df = _plot_df_all[_plot_df_all['mode_group'] == mg_sc]
            _sm_color = _sm_colors.get(mg_sc, '#555555')
            ax_heat, ax_mi = axes_sc[row_sc]

            # Heatmap
            sr_matrix = []
            for rm in reward_metrics_to_test:
                row_vals = []
                for sc in _plot_sc:
                    sub = _plot_df[[sc, rm]].dropna() if rm in _plot_df.columns else pd.DataFrame()
                    sub = sub[(sub[sc] != 0) | (sub[rm] != 0)] if len(sub) > 0 else sub
                    if len(sub) > 30:
                        r, _ = stats.pearsonr(sub[sc].values, sub[rm].values)
                        row_vals.append(r)
                    else:
                        row_vals.append(0.0)
                sr_matrix.append(row_vals)
            sr_matrix = np.array(sr_matrix)
            im = ax_heat.imshow(sr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax_heat.set_xticks(range(n_states))
            ax_heat.set_xticklabels([s.replace('det_', '') for s in _plot_sc],
                                    rotation=45, ha='right', fontsize=10)
            ax_heat.set_yticks(range(n_rewards))
            ax_heat.set_yticklabels(reward_metrics_to_test, fontsize=10)
            for i in range(n_rewards):
                for j in range(n_states):
                    val = sr_matrix[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=9, color=color,
                                fontweight='bold' if abs(val) > 0.7 else 'normal')
            fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
            ax_heat.set_title(
                f'{_sm_titles.get(mg_sc, mg_sc)}  (n={len(_plot_df)})\n'
                f'State → Reward Correlation (Pearson r)  [{_plot_jid}]',
                fontsize=11, color=_sm_color, fontweight='bold')

            # MI bar chart
            mi_sc = {}
            for rm in reward_metrics_to_test:
                if rm not in _plot_df.columns:
                    mi_sc[rm] = 0.0
                    continue
                total_mi = 0.0
                for sc in _plot_sc:
                    sub = _plot_df[[sc, rm]].dropna()
                    sub = sub[(sub[sc] != 0) | (sub[rm] != 0)]
                    if len(sub) > 30:
                        r, _ = stats.pearsonr(sub[sc].values, sub[rm].values)
                        r2 = min(r**2, 0.9999)
                        total_mi += -0.5 * np.log(1 - r2)
                mi_sc[rm] = total_mi
            sorted_mi = sorted(mi_sc.items(), key=lambda x: -x[1])
            mi_names = [x[0] for x in sorted_mi]
            mi_vals  = [x[1] for x in sorted_mi]
            y = np.arange(len(mi_names))
            ax_mi.barh(y, mi_vals, color=_sm_color, alpha=0.80, height=0.6)
            for i, v in enumerate(mi_vals):
                ax_mi.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
            ax_mi.set_yticks(y)
            ax_mi.set_yticklabels(mi_names, fontsize=10)
            ax_mi.set_xlabel('Gaussian Mutual Information (nats)', fontsize=10)
            ax_mi.set_title(
                f'{_sm_titles.get(mg_sc, mg_sc)}\nState → Reward: Total MI  [{_plot_jid}]',
                fontsize=11, color=_sm_color, fontweight='bold')
            ax_mi.invert_yaxis()
            ax_mi.grid(True, alpha=0.3, axis='x')

        fig.suptitle(
            f'State ↔ Reward Konzisztencia  [{_plot_jid}, {len(present_sm)} mód]',
            fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_state_consistency.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_state_consistency.png mentve ({_plot_jid}, {len(present_sm)} mód)")

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
    generate_reward_step_curve(output_dir, reward_mode=REWARD_CURVE_MODE)

    # State↔Reward összefoglaló
    if has_det_data and state_reward_corr:
        print(f"\n  State↔Reward konzisztencia (globális):")
        best_sr = max(state_reward_corr.items(), key=lambda x: x[1])
        worst_sr = min(state_reward_corr.items(), key=lambda x: x[1])
        print(f"    Legerősebb state↔reward: {best_sr[0]} (avg|r| = {best_sr[1]:.3f})")
        print(f"    Leggyengébb state↔reward: {worst_sr[0]} (avg|r| = {worst_sr[1]:.3f})")
        if junction_corr_summary:
            print(f"\n  Junction-onkénti avg|r| a célreward-okra:")
            for rm in [r for r in reward_metrics_5 if 'reward' in r or 'triplet' in r]:
                vals = [(jid, junction_corr_summary[jid].get(rm, 0.0))
                        for jid in sorted(junction_corr_summary)]
                line = f"    {rm:28}"
                for jid, v in vals:
                    flag = "↑" if v > 0.4 else ("·" if v > 0.25 else "↓")
                    line += f"  {jid}: {v:.3f}{flag}"
                print(line)


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
    parser.add_argument('--reward-curve-mode', type=str, default='',
                        help='Reward mód az ábrához: speed_throughput | wait_triplet_tpstdhalt | wait_haltratio | halt_ratio | co2_speedstd. '
                             'Ha üres, a REWARD_CURVE_MODE config értéke érvényes.')
    args = parser.parse_args()
    use_gui = args.gui

    # Gyors mód: csak a reward step curve regenerálása
    if args.reward_curve_only:
        jid  = args.reward_curve_junction or REWARD_CURVE_JUNCTION or None
        rmod = args.reward_curve_mode      or REWARD_CURVE_MODE     or 'speed_throughput'
        print(f"[reward-curve-only] junction={jid or 'összes'}, reward_mode={rmod}")
        generate_reward_step_curve(OUTPUT_DIR, junction_filter=jid, reward_mode=rmod)
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
        print(f"  Epizodok szintenkent: {EPISODES_PER_LEVEL} random + {MIXED_EPISODES} mixed + {ACTUATED_EPISODES} actuated")
        print(f"  Mixed epsilon: {MIXED_EPSILON}  (random akció valószínűsége)")
        print(f"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s")
        print(f"  Kimenet: {OUTPUT_DIR}")
        print("=" * 80)

        total_random   = len(FLOW_MAX_LEVELS) * EPISODES_PER_LEVEL
        total_mixed    = len(FLOW_MAX_LEVELS) * MIXED_EPISODES
        total_actuated = len(FLOW_MAX_LEVELS) * ACTUATED_EPISODES
        total_sims     = total_random + total_mixed + total_actuated
        sim_count      = 0

        # --- Random epizódok ---
        print(f"\n  [1/3] RANDOM kontroll: {total_random} epizod")
        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(EPISODES_PER_LEVEL):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{EPISODES_PER_LEVEL} | RANDOM ---")
                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode="random", use_gui=use_gui)

        # --- Mixed epizódok (ϵ-greedy) ---
        _mixed_mode = f"mixed_{int(MIXED_EPSILON * 100)}"
        print(f"\n  [2/3] MIXED kontroll ({_mixed_mode}): {total_mixed} epizod")
        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(MIXED_EPISODES):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{MIXED_EPISODES} | MIXED (ϵ={MIXED_EPSILON}) ---")
                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=_mixed_mode, use_gui=use_gui)

        # --- Actuated epizódok ---
        print(f"\n  [3/3] ACTUATED kontroll: {total_actuated} epizod")
        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(ACTUATED_EPISODES):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{ACTUATED_EPISODES} | ACTUATED ---")
                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode="actuated", use_gui=use_gui)

        print(f"\n  Szimulacio kesz! ({total_sims} epizod: "
              f"{total_random} random + {total_mixed} mixed + {total_actuated} actuated)")
    else:
        print("  --skip-simulation: Csak elemzes a meglevo CSV-kre")

    # Per-junction kalibráció
    analyze_per_junction(OUTPUT_DIR)

    # Reward metrika és normalizáció kiválasztás
    reward_selection_analysis(OUTPUT_DIR)


if __name__ == "__main__":
    main()

