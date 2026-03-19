#!/usr/bin/env python3
"""
Metrika gyűjtés az OSM junction-re (2632893078) — normalizációs konstansok összehasonlításhoz.

Ez a teszt UGYANAZT a metrika gyűjtési logikát használja, mint a metric_collection_test.py
(ami a mega_catalogue 21 junction-ös hálózatra futott, 527K mintát gyűjtött),
de az OSM single-junction hálózaton fut.

Cél: megvizsgálni, hogy a mega_catalogue-ból származó MU_WAIT, STD_WAIT, MU_CO2, STD_CO2
normalizációs konstansok mennyire illeszkednek az OSM junction-re.

Használat:
    python metric_collection_osm.py

Kimenet:
    metric_pca_osm/ mappa — CSV-k, PCA ábra, normalizációs összehasonlítás
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "maps")
NET_FILE = os.path.join(MAPS_DIR, "osm.net.xml")
LOGIC_FILE = os.path.join(MAPS_DIR, "traffic_lights.json")
DETECTOR_FILE = os.path.join(MAPS_DIR, "osm.detectors.add.xml")

# --- Config ---
FLOW_MIN = 100
FLOW_MAX_LEVELS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
DURATION = 3600       # 1 óra szimuláció
DELTA_TIME = 5        # lépésméret (sec)
WARMUP = 100          # warmup (sec)
EPISODES_PER_LEVEL = 3  # ismétlések forgalmi szintenként
MAX_REALISTIC_TT = 1000  # travel time szűrő

# --- Mega catalogue normalizációs konstansok (referenciaértékek) ---
MEGA_CATALOGUE_PARAMS = {
    'MU_WAIT': 4.584800,
    'STD_WAIT': 1.824900,
    'MU_CO2': 10.870600,
    'STD_CO2': 0.962900,
}


def run_simulation(flow_max, episode_idx, output_dir):
    """Egy szimulációs epizód futtatása és metrika gyűjtés."""

    import traci
    import sumolib

    route_file = os.path.join(SCRIPT_DIR, f"_metric_osm_{flow_max}_{episode_idx}.rou.xml")

    # --- Forgalom generálás ---
    net = sumolib.net.readNet(NET_FILE)
    with open(LOGIC_FILE) as f:
        logic = json.load(f)

    junction_ids = list(logic.keys())
    print(f"  Junction-ök: {junction_ids}")

    lane_routes = {}
    for jid in junction_ids:
        node = net.getNode(jid)
        if node is None:
            print(f"  WARNING: junction {jid} nem található a net fájlban")
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

    print(f"  Sávok forgalommal: {len(lane_routes)}")

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
    print(f"  Összes jármű: {len(all_trips)}")

    with open(route_file, 'w') as f:
        f.write('<routes>\n')
        f.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" '
                'length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
        for idx, (depart, from_e, lane_idx, to_e) in enumerate(all_trips):
            f.write(f'    <trip id="veh_{idx}" type="car" depart="{depart:.2f}" '
                    f'from="{from_e}" to="{to_e}" departLane="{lane_idx}" />\n')
        f.write('</routes>\n')

    # --- SUMO indítás ---
    traci.start(["sumo", "-n", NET_FILE, "-r", route_file, "-a", DETECTOR_FILE,
        "--no-step-log", "true", "--ignore-route-errors", "true",
        "--no-warnings", "true", "--xml-validation", "never", "--random", "true"])

    # --- Junction → bejövő lane-ek és detektorok feltérképezése ---
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

    for jid in junction_ids:
        print(f"  {jid}: {len(junction_lanes[jid])} incoming lanes, {len(junction_dets[jid])} detectors")

    # --- Metrikák ---
    metrics = {jid: [] for jid in junction_ids}

    # --- Warmup ---
    for _ in range(WARMUP):
        traci.simulationStep()

    # --- Fő ciklus ---
    junction_phases = {}
    for jid in junction_ids:
        programs = traci.trafficlight.getAllProgramLogics(jid)
        if programs:
            num_phases = len(programs[0].phases)
            junction_phases[jid] = num_phases
        else:
            junction_phases[jid] = 1

    total_steps = (DURATION - WARMUP) // DELTA_TIME

    for step_i in range(total_steps):
        # Random fázis (mint a PCA tesztben — nem tanulunk, csak mérünk)
        for jid in junction_ids:
            phase = random.randint(0, junction_phases[jid] - 1)
            traci.trafficlight.setPhase(jid, phase)

        # Akkumulálás delta_time lépésen
        acc = {jid: {
            'tt': 0.0, 'tt_raw': 0.0, 'waiting': 0.0, 'co2': 0.0,
            'veh': 0, 'speed': 0.0, 'halted': 0, 'occ': 0.0,
            'valid_tt': 0, 'valid_tt_raw': 0, 'valid_speed': 0,
            'steps': 0
        } for jid in junction_ids}

        for dt_step in range(DELTA_TIME):
            traci.simulationStep()

            for jid in junction_ids:
                lanes = junction_lanes[jid]
                dets = junction_dets[jid]
                acc[jid]['steps'] += 1

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

                    acc[jid]['halted'] += traci.lane.getLastStepHaltingNumber(lane)

                for det in dets:
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

            metrics[jid].append({
                'step': step_i,
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
            })

        if (step_i + 1) % 50 == 0:
            print(f"    Step {step_i+1}/{total_steps}")

    traci.close()

    if os.path.exists(route_file):
        os.remove(route_file)

    for jid in junction_ids:
        df = pd.DataFrame(metrics[jid])
        csv_path = os.path.join(output_dir, f"{jid}_flow{flow_max}_ep{episode_idx}.csv")
        df.to_csv(csv_path, index=False)

    return metrics


def compare_normalization(output_dir):
    """Összehasonlítja az OSM junction normalizációs paramétereit a mega catalogue-val."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("NORMALIZACIOS PARAMETEREK OSSZEHASONLITAS")
    print("=" * 70)

    # Összes CSV betöltése
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        df['junction'] = csv_file.split('_flow')[0]
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    mask = full_df['VehCount'] > 0
    df_valid = full_df[mask].copy()

    print(f"\nOsszes adatpont: {len(full_df)}")
    print(f"Ervenyes adatpont (VehCount > 0): {len(df_valid)}")

    # --- OSM normalizációs paraméterek kiszámítása ---
    osm_params = {}
    for col in ['TotalWaitingTime', 'TotalCO2']:
        vals = df_valid[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            log_vals = np.log(vals + 1e-5)
            mu = np.mean(log_vals)
            std = np.std(log_vals)
            osm_params[col] = {'mu': mu, 'std': std, 'median': np.median(vals),
                               'p5': np.percentile(vals, 5), 'p95': np.percentile(vals, 95)}

    # --- Összehasonlítás ---
    print(f"\n{'':20} {'MEGA CATALOGUE':>20} {'OSM JUNCTION':>20} {'ELTERES':>12}")
    print("-" * 75)

    mega = MEGA_CATALOGUE_PARAMS
    if 'TotalWaitingTime' in osm_params:
        osm_w = osm_params['TotalWaitingTime']
        print(f"{'MU_WAIT':20} {mega['MU_WAIT']:20.4f} {osm_w['mu']:20.4f} {osm_w['mu'] - mega['MU_WAIT']:+12.4f}")
        print(f"{'STD_WAIT':20} {mega['STD_WAIT']:20.4f} {osm_w['std']:20.4f} {osm_w['std'] - mega['STD_WAIT']:+12.4f}")
        print(f"{'  median (raw)':20} {'120.20':>20} {osm_w['median']:20.2f}")
        print(f"{'  p5-p95 (raw)':20} {'3.2 - 1340.4':>20} {osm_w['p5']:.1f} - {osm_w['p95']:.1f}")

    print()
    if 'TotalCO2' in osm_params:
        osm_c = osm_params['TotalCO2']
        print(f"{'MU_CO2':20} {mega['MU_CO2']:20.4f} {osm_c['mu']:20.4f} {osm_c['mu'] - mega['MU_CO2']:+12.4f}")
        print(f"{'STD_CO2':20} {mega['STD_CO2']:20.4f} {osm_c['std']:20.4f} {osm_c['std'] - mega['STD_CO2']:+12.4f}")
        print(f"{'  median (raw)':20} {'70129':>20} {osm_c['median']:20.2f}")
        print(f"{'  p5-p95 (raw)':20} {'8383 - 173545':>20} {osm_c['p5']:.0f} - {osm_c['p95']:.0f}")

    # --- Reward szimuláció: mit ad a sigmoid a mega vs osm paraméterekkel? ---
    print(f"\n\n{'=' * 70}")
    print("REWARD SZIMULACIO — Milyen reward-ot kapna az agens?")
    print("=" * 70)

    if 'TotalWaitingTime' in osm_params and 'TotalCO2' in osm_params:
        # Tipikus értékek az OSM junction-ön
        test_cases = [
            ("Alacsony forgalom", osm_params['TotalWaitingTime']['p5'], osm_params['TotalCO2']['p5']),
            ("Median forgalom", osm_params['TotalWaitingTime']['median'], osm_params['TotalCO2']['median']),
            ("Magas forgalom", osm_params['TotalWaitingTime']['p95'], osm_params['TotalCO2']['p95']),
        ]

        def calc_reward(wait, co2, mu_w, std_w, mu_c, std_c):
            z_wait = (np.log(wait + 1e-5) - mu_w) / (std_w + 1e-9)
            z_co2 = (np.log(co2 + 1e-5) - mu_c) / (std_c + 1e-9)
            r_wait = 1.0 - 1.0 / (1.0 + np.exp(-z_wait))
            r_co2 = 1.0 - 1.0 / (1.0 + np.exp(-z_co2))
            return (r_wait + r_co2) / 2.0

        osm_w = osm_params['TotalWaitingTime']
        osm_c = osm_params['TotalCO2']

        print(f"\n{'Scenario':20} {'Wait':>10} {'CO2':>12} {'R(mega)':>10} {'R(osm)':>10} {'Diff':>8}")
        print("-" * 75)
        for name, wait, co2 in test_cases:
            r_mega = calc_reward(wait, co2, mega['MU_WAIT'], mega['STD_WAIT'], mega['MU_CO2'], mega['STD_CO2'])
            r_osm = calc_reward(wait, co2, osm_w['mu'], osm_w['std'], osm_c['mu'], osm_c['std'])
            print(f"{name:20} {wait:10.1f} {co2:12.0f} {r_mega:10.4f} {r_osm:10.4f} {r_osm - r_mega:+8.4f}")

        # Reward eloszlás összehasonlítás
        all_waits = df_valid['TotalWaitingTime'].values
        all_co2s = df_valid['TotalCO2'].values
        mask_pos = (all_waits > 0) & (all_co2s > 0)
        all_waits = all_waits[mask_pos]
        all_co2s = all_co2s[mask_pos]

        rewards_mega = np.array([calc_reward(w, c, mega['MU_WAIT'], mega['STD_WAIT'], mega['MU_CO2'], mega['STD_CO2'])
                                  for w, c in zip(all_waits, all_co2s)])
        rewards_osm = np.array([calc_reward(w, c, osm_w['mu'], osm_w['std'], osm_c['mu'], osm_c['std'])
                                 for w, c in zip(all_waits, all_co2s)])

        print(f"\n{'Reward eloszlas':20} {'MEGA params':>20} {'OSM params':>20}")
        print("-" * 65)
        print(f"{'  Mean':20} {np.mean(rewards_mega):20.4f} {np.mean(rewards_osm):20.4f}")
        print(f"{'  Std':20} {np.std(rewards_mega):20.4f} {np.std(rewards_osm):20.4f}")
        print(f"{'  Min':20} {np.min(rewards_mega):20.4f} {np.min(rewards_osm):20.4f}")
        print(f"{'  Max':20} {np.max(rewards_mega):20.4f} {np.max(rewards_osm):20.4f}")
        print(f"{'  p10':20} {np.percentile(rewards_mega, 10):20.4f} {np.percentile(rewards_osm, 10):20.4f}")
        print(f"{'  p90':20} {np.percentile(rewards_mega, 90):20.4f} {np.percentile(rewards_osm, 90):20.4f}")
        print(f"{'  Range (p90-p10)':20} {np.percentile(rewards_mega, 90) - np.percentile(rewards_mega, 10):20.4f} {np.percentile(rewards_osm, 90) - np.percentile(rewards_osm, 10):20.4f}")

        # --- Ábra: reward eloszlás összehasonlítás ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Reward hisztogram
        axes[0, 0].hist(rewards_mega, bins=50, alpha=0.6, label='MEGA params', color='blue', density=True)
        axes[0, 0].hist(rewards_osm, bins=50, alpha=0.6, label='OSM params', color='red', density=True)
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Reward Distribution: MEGA vs OSM params')
        axes[0, 0].legend()
        axes[0, 0].axvline(np.mean(rewards_mega), color='blue', ls='--', label=f'mean(mega)={np.mean(rewards_mega):.3f}')
        axes[0, 0].axvline(np.mean(rewards_osm), color='red', ls='--', label=f'mean(osm)={np.mean(rewards_osm):.3f}')

        # 2. Log waiting time distribution
        log_waits = np.log(all_waits + 1e-5)
        axes[0, 1].hist(log_waits, bins=50, alpha=0.6, color='green', density=True)
        axes[0, 1].axvline(mega['MU_WAIT'], color='blue', ls='--', lw=2, label=f"MEGA MU={mega['MU_WAIT']:.2f}")
        axes[0, 1].axvline(osm_w['mu'], color='red', ls='--', lw=2, label=f"OSM MU={osm_w['mu']:.2f}")
        axes[0, 1].set_xlabel('log(TotalWaitingTime)')
        axes[0, 1].set_title('Log WaitingTime Distribution vs MU')
        axes[0, 1].legend()

        # 3. Log CO2 distribution
        log_co2s = np.log(all_co2s + 1e-5)
        axes[1, 0].hist(log_co2s, bins=50, alpha=0.6, color='orange', density=True)
        axes[1, 0].axvline(mega['MU_CO2'], color='blue', ls='--', lw=2, label=f"MEGA MU={mega['MU_CO2']:.2f}")
        axes[1, 0].axvline(osm_c['mu'], color='red', ls='--', lw=2, label=f"OSM MU={osm_c['mu']:.2f}")
        axes[1, 0].set_xlabel('log(TotalCO2)')
        axes[1, 0].set_title('Log CO2 Distribution vs MU')
        axes[1, 0].legend()

        # 4. Reward vs flow level
        flow_levels = df_valid['flow_level'].values[mask_pos]
        if len(flow_levels) == len(rewards_mega):
            unique_flows = sorted(set(flow_levels))
            mega_means = [np.mean(rewards_mega[flow_levels == fl]) for fl in unique_flows]
            osm_means = [np.mean(rewards_osm[flow_levels == fl]) for fl in unique_flows]
            axes[1, 1].plot(unique_flows, mega_means, 'bo-', label='MEGA params', lw=2)
            axes[1, 1].plot(unique_flows, osm_means, 'ro-', label='OSM params', lw=2)
            axes[1, 1].set_xlabel('Flow Level (veh/h/lane)')
            axes[1, 1].set_ylabel('Mean Reward')
            axes[1, 1].set_title('Mean Reward by Traffic Level')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'normalization_comparison.png'), dpi=200)
        plt.close()
        print(f"\nAbra mentve: normalization_comparison.png")

    # --- OSM paraméterek kiírása Python konstansokként ---
    print(f"\n{'=' * 70}")
    print("PYTHON KONSTANSOK (ha cserélni akarod a sumo_rl_environment.py-ben):")
    print("=" * 70)
    if 'TotalWaitingTime' in osm_params:
        print(f"MU_WAIT = {osm_params['TotalWaitingTime']['mu']:.6f}")
        print(f"STD_WAIT = {osm_params['TotalWaitingTime']['std']:.6f}")
    if 'TotalCO2' in osm_params:
        print(f"MU_CO2 = {osm_params['TotalCO2']['mu']:.6f}")
        print(f"STD_CO2 = {osm_params['TotalCO2']['std']:.6f}")


def main():
    # Ellenőrzés
    for name, path in [("net", NET_FILE), ("logic", LOGIC_FILE), ("det", DETECTOR_FILE)]:
        if not os.path.exists(path):
            print(f"[ERROR] Hiányzó fájl ({name}): {path}")
            sys.exit(1)

    output_dir = os.path.join(SCRIPT_DIR, "metric_pca_osm")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  METRIKA GYUJTES — OSM JUNCTION (normalizacio osszehasonlitas)")
    print("=" * 70)
    print(f"  Net: {NET_FILE}")
    print(f"  Logic: {LOGIC_FILE}")
    print(f"  Detectors: {DETECTOR_FILE}")
    print(f"  Flow szintek (max): {FLOW_MAX_LEVELS}")
    print(f"  Epizodok szintenkent: {EPISODES_PER_LEVEL}")
    print(f"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s")
    print("=" * 70)

    summary_rows = []

    for flow_max in FLOW_MAX_LEVELS:
        for ep in range(EPISODES_PER_LEVEL):
            print(f"\n--- Flow max: {flow_max}/h | Epizod {ep+1}/{EPISODES_PER_LEVEL} ---")
            metrics = run_simulation(flow_max, ep, output_dir)

            for jid, data in metrics.items():
                if data:
                    df = pd.DataFrame(data)
                    summary_rows.append({
                        'flow_max': flow_max,
                        'episode': ep,
                        'junction': jid,
                        'mean_TotalWaitingTime': df['TotalWaitingTime'].mean(),
                        'mean_TotalCO2': df['TotalCO2'].mean(),
                        'mean_AvgWaitingTime': df['AvgWaitingTime'].mean(),
                        'mean_AvgCO2': df['AvgCO2'].mean(),
                        'mean_VehCount': df['VehCount'].mean(),
                        'mean_AvgSpeed': df['AvgSpeed'].mean(),
                        'mean_QueueLength': df['QueueLength'].mean(),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'episode_summary.csv'), index=False)
    print(f"\nEpisode summary mentve: {os.path.join(output_dir, 'episode_summary.csv')}")

    # Normalizációs összehasonlítás
    compare_normalization(output_dir)


if __name__ == "__main__":
    main()
