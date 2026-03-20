#!/usr/bin/env python3
"""
Per-junction metrika gyűjtés és normalizációs paraméter kalibráció.

A metric_collection_test.py szimulációs logikáját használja (teljes hálózat,
minden junction-re forgalom + random jelzőlámpa), de az elemzést
JUNCTION-ÖNKÉNT végzi el.

Kimenet:
  metric_pca_per_junction/
    ├── junction_reward_params.json   ← A LÉNYEGAlak: {jid: {MU_WAIT, STD_WAIT, MU_CO2, STD_CO2}}
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
FLOW_MAX_LEVELS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
DURATION = 3600       # 1 óra szimuláció
DELTA_TIME = 5        # lépésméret (sec)
WARMUP = 100          # warmup (sec)
EPISODES_PER_LEVEL = 3  # ismétlések forgalmi szintenként
MAX_REALISTIC_TT = 1000  # travel time szűrő

# --- Globális referencia (a korábbi 21-junction aggregált PCA-ból) ---
GLOBAL_PARAMS = {
    'MU_WAIT': 4.584800,
    'STD_WAIT': 1.824900,
    'MU_CO2': 10.870600,
    'STD_CO2': 0.962900,
}

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "metric_pca_per_junction")


def run_simulation(flow_max, episode_idx, output_dir):
    """
    Egy szimulációs epizód futtatása.
    PONTOSAN UGYANAZ mint metric_collection_test.py — teljes hálózaton fut,
    minden junction-re gyűjt metrikát.
    """

    import traci
    import sumolib

    route_file = os.path.join(SCRIPT_DIR, f"_metric_pj_{flow_max}_{episode_idx}.rou.xml")

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
    traci.start(["sumo", "-n", NET_FILE, "-r", route_file, "-a", DETECTOR_FILE,
        "--no-step-log", "true", "--ignore-route-errors", "true",
        "--no-warnings", "true", "--xml-validation", "never", "--random", "true"])

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

    # --- Jelzőlámpa fázis infó ---
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
        # Random jelzőlámpa akció
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

    # CSV mentés
    for jid in junction_ids:
        df = pd.DataFrame(metrics[jid])
        csv_path = os.path.join(output_dir, f"{jid}_flow{flow_max}_ep{episode_idx}.csv")
        df.to_csv(csv_path, index=False)

    return metrics


def calc_reward(wait, co2, mu_w, std_w, mu_c, std_c):
    """Log-sigmoid reward számítás (azonos a sumo_rl_environment.py logikával)."""
    z_wait = (np.log(wait + 1e-5) - mu_w) / (std_w + 1e-9)
    z_co2 = (np.log(co2 + 1e-5) - mu_c) / (std_c + 1e-9)
    r_wait = 1.0 - 1.0 / (1.0 + np.exp(-z_wait))
    r_co2 = 1.0 - 1.0 / (1.0 + np.exp(-z_co2))
    return (r_wait + r_co2) / 2.0


def analyze_per_junction(output_dir):
    """
    Per-junction elemzés: MU/STD számítás, összehasonlítás a globálissal,
    junction_reward_params.json kiírása.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("  PER-JUNCTION NORMALIZACIOS PARAMETER KALIBRACIO")
    print("=" * 80)

    # --- CSV-k betöltése ---
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]
    if not csv_files:
        print("[ERROR] Nincsenek CSV fajlok! Futtasd elobb a szimulaciot.")
        return

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        # Junction ID: mindent a _flow előtt
        jid = csv_file.split('_flow')[0]
        df['junction'] = jid
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    junction_ids = sorted(full_df['junction'].unique())

    print(f"\nOsszes adatpont: {len(full_df)}")
    print(f"Junction-ok szama: {len(junction_ids)}")
    print(f"Junction-ok: {', '.join(junction_ids)}")

    # --- Per-junction paraméterek ---
    junction_params = {}
    summary_rows = []

    print(f"\n{'Junction':15} {'MU_WAIT':>10} {'STD_WAIT':>10} {'MU_CO2':>10} {'STD_CO2':>10} "
          f"{'dMU_W':>8} {'dMU_C':>8} {'N':>8}")
    print("-" * 95)

    for jid in junction_ids:
        jdf = full_df[full_df['junction'] == jid].copy()
        jdf_valid = jdf[jdf['VehCount'] > 0].copy()

        if len(jdf_valid) == 0:
            print(f"{jid:15} {'SKIP — no data':>50}")
            continue

        # TotalWaitingTime paraméterek
        wait_vals = jdf_valid['TotalWaitingTime'].values
        wait_pos = wait_vals[wait_vals > 0]
        if len(wait_pos) > 10:
            log_wait = np.log(wait_pos + 1e-5)
            mu_wait = float(np.mean(log_wait))
            std_wait = float(np.std(log_wait))
        else:
            mu_wait = GLOBAL_PARAMS['MU_WAIT']
            std_wait = GLOBAL_PARAMS['STD_WAIT']

        # TotalCO2 paraméterek
        co2_vals = jdf_valid['TotalCO2'].values
        co2_pos = co2_vals[co2_vals > 0]
        if len(co2_pos) > 10:
            log_co2 = np.log(co2_pos + 1e-5)
            mu_co2 = float(np.mean(log_co2))
            std_co2 = float(np.std(log_co2))
        else:
            mu_co2 = GLOBAL_PARAMS['MU_CO2']
            std_co2 = GLOBAL_PARAMS['STD_CO2']

        junction_params[jid] = {
            'MU_WAIT': round(mu_wait, 6),
            'STD_WAIT': round(std_wait, 6),
            'MU_CO2': round(mu_co2, 6),
            'STD_CO2': round(std_co2, 6),
        }

        d_mu_w = mu_wait - GLOBAL_PARAMS['MU_WAIT']
        d_mu_c = mu_co2 - GLOBAL_PARAMS['MU_CO2']

        print(f"{jid:15} {mu_wait:10.4f} {std_wait:10.4f} {mu_co2:10.4f} {std_co2:10.4f} "
              f"{d_mu_w:+8.4f} {d_mu_c:+8.4f} {len(jdf_valid):8d}")

        # --- Reward range összehasonlítás ---
        mask_pos = (jdf_valid['TotalWaitingTime'].values > 0) & (jdf_valid['TotalCO2'].values > 0)
        w_pos = jdf_valid['TotalWaitingTime'].values[mask_pos]
        c_pos = jdf_valid['TotalCO2'].values[mask_pos]

        if len(w_pos) > 10:
            rewards_global = np.array([calc_reward(w, c,
                GLOBAL_PARAMS['MU_WAIT'], GLOBAL_PARAMS['STD_WAIT'],
                GLOBAL_PARAMS['MU_CO2'], GLOBAL_PARAMS['STD_CO2'])
                for w, c in zip(w_pos, c_pos)])
            rewards_local = np.array([calc_reward(w, c, mu_wait, std_wait, mu_co2, std_co2)
                for w, c in zip(w_pos, c_pos)])

            summary_rows.append({
                'junction': jid,
                'MU_WAIT': mu_wait, 'STD_WAIT': std_wait,
                'MU_CO2': mu_co2, 'STD_CO2': std_co2,
                'dMU_WAIT': d_mu_w, 'dMU_CO2': d_mu_c,
                'N_valid': len(jdf_valid),
                'median_wait': float(np.median(wait_pos)) if len(wait_pos) > 0 else 0,
                'median_co2': float(np.median(co2_pos)) if len(co2_pos) > 0 else 0,
                'reward_global_mean': float(np.mean(rewards_global)),
                'reward_global_std': float(np.std(rewards_global)),
                'reward_global_range': float(np.percentile(rewards_global, 90) - np.percentile(rewards_global, 10)),
                'reward_local_mean': float(np.mean(rewards_local)),
                'reward_local_std': float(np.std(rewards_local)),
                'reward_local_range': float(np.percentile(rewards_local, 90) - np.percentile(rewards_local, 10)),
            })

    # --- JSON kiírás ---
    json_path = os.path.join(DATA_DIR, "junction_reward_params.json")
    with open(json_path, 'w') as f:
        json.dump(junction_params, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"  junction_reward_params.json mentve: {json_path}")
    print(f"  {len(junction_params)} junction parameterei")
    print(f"{'=' * 80}")

    # Másolat az output mappába is
    json_copy = os.path.join(output_dir, "junction_reward_params.json")
    with open(json_copy, 'w') as f:
        json.dump(junction_params, f, indent=2)

    # --- Summary CSV ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, 'per_junction_summary.csv'), index=False)
        print(f"  per_junction_summary.csv mentve")

    # --- Ábra 1: Heatmap — MU/STD eltérés a globálistól ---
    if summary_rows:
        sdf = pd.DataFrame(summary_rows).set_index('junction')

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Per-Junction Normalization Parameters vs Global', fontsize=16, y=1.02)

        # 1a. MU_WAIT bar chart
        ax = axes[0, 0]
        jids = sdf.index.tolist()
        x = np.arange(len(jids))
        ax.bar(x, sdf['MU_WAIT'].values, color='steelblue', alpha=0.8)
        ax.axhline(GLOBAL_PARAMS['MU_WAIT'], color='red', ls='--', lw=2, label=f"Global: {GLOBAL_PARAMS['MU_WAIT']:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MU_WAIT')
        ax.set_title('MU_WAIT per Junction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1b. MU_CO2 bar chart
        ax = axes[0, 1]
        ax.bar(x, sdf['MU_CO2'].values, color='darkorange', alpha=0.8)
        ax.axhline(GLOBAL_PARAMS['MU_CO2'], color='red', ls='--', lw=2, label=f"Global: {GLOBAL_PARAMS['MU_CO2']:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MU_CO2')
        ax.set_title('MU_CO2 per Junction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1c. Reward range comparison
        ax = axes[1, 0]
        width = 0.35
        ax.bar(x - width/2, sdf['reward_global_range'].values, width, label='Global params', color='blue', alpha=0.7)
        ax.bar(x + width/2, sdf['reward_local_range'].values, width, label='Local params', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reward Range (p90 - p10)')
        ax.set_title('Reward Range: Global vs Local Normalization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 1d. Eltérés heatmap
        ax = axes[1, 1]
        delta_data = sdf[['dMU_WAIT', 'dMU_CO2']].copy()
        delta_data.columns = ['Delta MU_WAIT', 'Delta MU_CO2']
        im = ax.imshow(delta_data.values.T, cmap='RdBu_r', aspect='auto',
                       vmin=-max(abs(delta_data.values.min()), abs(delta_data.values.max())),
                       vmax=max(abs(delta_data.values.min()), abs(delta_data.values.max())))
        ax.set_xticks(range(len(jids)))
        ax.set_xticklabels(jids, rotation=45, ha='right', fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Delta MU_WAIT', 'Delta MU_CO2'])
        ax.set_title('Deviation from Global Parameters')
        # Annotáció
        for i in range(2):
            for j in range(len(jids)):
                val = delta_data.values[j, i]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'junction_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  junction_comparison.png mentve")

    # --- Ábra 2: Per-junction reward eloszlás ---
    if summary_rows and len(junction_ids) > 0:
        n_junctions = len(junction_ids)
        n_cols = 5
        n_rows = (n_junctions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Reward Distribution per Junction (Global vs Local params)', fontsize=14)

        for idx, jid in enumerate(junction_ids):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            jdf = full_df[full_df['junction'] == jid].copy()
            mask_pos = (jdf['TotalWaitingTime'].values > 0) & (jdf['TotalCO2'].values > 0)
            w_pos = jdf['TotalWaitingTime'].values[mask_pos]
            c_pos = jdf['TotalCO2'].values[mask_pos]

            if len(w_pos) < 10:
                ax.set_title(f'{jid}\n(insufficient data)', fontsize=9)
                continue

            p = junction_params[jid]
            r_global = np.array([calc_reward(w, c,
                GLOBAL_PARAMS['MU_WAIT'], GLOBAL_PARAMS['STD_WAIT'],
                GLOBAL_PARAMS['MU_CO2'], GLOBAL_PARAMS['STD_CO2'])
                for w, c in zip(w_pos, c_pos)])
            r_local = np.array([calc_reward(w, c,
                p['MU_WAIT'], p['STD_WAIT'], p['MU_CO2'], p['STD_CO2'])
                for w, c in zip(w_pos, c_pos)])

            ax.hist(r_global, bins=30, alpha=0.5, label='Global', color='blue', density=True)
            ax.hist(r_local, bins=30, alpha=0.5, label='Local', color='red', density=True)
            g_range = np.percentile(r_global, 90) - np.percentile(r_global, 10)
            l_range = np.percentile(r_local, 90) - np.percentile(r_local, 10)
            ax.set_title(f'{jid}\nG_rng={g_range:.2f} L_rng={l_range:.2f}', fontsize=9)
            ax.set_xlim(0, 1)
            if idx == 0:
                ax.legend(fontsize=7)

        # Üres subplot-ok elrejtése
        for idx in range(len(junction_ids), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'junction_reward_ranges.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  junction_reward_ranges.png mentve")

    # --- Összefoglaló statisztikák ---
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        print(f"\n{'=' * 80}")
        print("  OSSZEFOGLALO STATISZTIKAK")
        print(f"{'=' * 80}")

        print(f"\n  MU_WAIT elteresek a globalitol:")
        print(f"    Min:  {sdf['dMU_WAIT'].min():+.4f}  ({sdf.loc[sdf['dMU_WAIT'].idxmin(), 'junction']})")
        print(f"    Max:  {sdf['dMU_WAIT'].max():+.4f}  ({sdf.loc[sdf['dMU_WAIT'].idxmax(), 'junction']})")
        print(f"    Mean: {sdf['dMU_WAIT'].mean():+.4f}")
        print(f"    Std:  {sdf['dMU_WAIT'].std():.4f}")

        print(f"\n  MU_CO2 elteresek a globalitol:")
        print(f"    Min:  {sdf['dMU_CO2'].min():+.4f}  ({sdf.loc[sdf['dMU_CO2'].idxmin(), 'junction']})")
        print(f"    Max:  {sdf['dMU_CO2'].max():+.4f}  ({sdf.loc[sdf['dMU_CO2'].idxmax(), 'junction']})")
        print(f"    Mean: {sdf['dMU_CO2'].mean():+.4f}")
        print(f"    Std:  {sdf['dMU_CO2'].std():.4f}")

        print(f"\n  Reward range javulas (local vs global):")
        improved = (sdf['reward_local_range'] > sdf['reward_global_range']).sum()
        total = len(sdf)
        print(f"    Javult: {improved}/{total} junction")
        print(f"    Atlagos global range: {sdf['reward_global_range'].mean():.4f}")
        print(f"    Atlagos local range:  {sdf['reward_local_range'].mean():.4f}")
        pct = (sdf['reward_local_range'].mean() / sdf['reward_global_range'].mean() - 1) * 100
        print(f"    Valtozas: {pct:+.1f}%")



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
    def normalize_log_sigmoid(vals, mu, std):
        """Log-sigmoid: R = 1 - sigmoid((log(x) - mu) / std)"""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    def normalize_log_tanh(vals, mu, std):
        """Log-tanh: R = (1 - tanh((log(x) - mu) / std)) / 2"""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return (1.0 - np.tanh(z)) / 2.0

    def normalize_lognormal_cdf(vals, mu, std):
        """Lognormal CDF: R = 1 - Phi((log(x) - mu) / std)"""
        z = (np.log(vals + 1e-5) - mu) / (std + 1e-9)
        return 1.0 - stats.norm.cdf(z)

    def normalize_linear_sigmoid(vals, mu, std):
        """Linear sigmoid (no log): R = 1 - sigmoid((x - exp(mu)) / (exp(mu)*std))"""
        center = np.exp(mu)
        scale = center * std
        z = (vals - center) / (scale + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    def normalize_sqrt_sigmoid(vals, mu_log, std_log):
        """Sqrt-sigmoid: R = 1 - sigmoid((sqrt(x) - mu_sqrt) / std_sqrt)"""
        sqrt_vals = np.sqrt(vals + 1e-5)
        mu_s = np.mean(sqrt_vals)
        std_s = np.std(sqrt_vals)
        z = (sqrt_vals - mu_s) / (std_s + 1e-9)
        return 1.0 - 1.0 / (1.0 + np.exp(-z))

    NORM_METHODS = {
        'log-sigmoid':    normalize_log_sigmoid,
        'log-tanh':       normalize_log_tanh,
        'lognormal-cdf':  normalize_lognormal_cdf,
        'linear-sigmoid': normalize_linear_sigmoid,
        'sqrt-sigmoid':   normalize_sqrt_sigmoid,
    }

    # =====================================================================
    # 1. KORRELÁCIÓ — REDUNDANCIA SZŰRÉS
    # =====================================================================
    print("\n" + "=" * 100)
    print("  1. KORRELÁCIÓ — REDUNDANCIA SZŰRÉS")
    print("     Pearson |r| > 0.85 → redundáns, nem érdemes együtt használni")
    print("=" * 100)

    # Egyedi metrikák amiket vizsgálunk (nem per-vehicle, mert azok veszélyesek)
    candidate_metrics = ['TotalWaitingTime', 'TotalCO2', 'AvgSpeed', 'QueueLength',
                         'TotalTravelTime', 'AvgOccupancy']

    # Globális korreláció (log-transzformált, VehCount > 0 szűrés)
    df_valid = full_df[full_df['VehCount'] > 0].copy()
    epsilon = 1e-5

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
    header = f"{'':20}"
    for col in candidate_metrics:
        header += f" {col[:8]:>9}"
    print(header)
    print("-" * (20 + 10 * len(candidate_metrics)))
    for row_col in candidate_metrics:
        line = f"  {row_col:18}"
        for col_col in candidate_metrics:
            r = corr_matrix.loc[row_col, col_col]
            marker = " *" if abs(r) > 0.85 and row_col != col_col else "  "
            line += f" {r:+7.3f}{marker}"
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

    # Jelölés a jelenlegi módszerhez
    current_combo = ('TotalWaitingTime', 'TotalCO2')

    print(f"\n  Vizsgált kombinációk: {len(all_combos)}")
    print(f"  (2-es kombókból kiszűrve a redundáns párok: |r| > 0.85)\n")

    combo_results = []

    for combo in all_combos:
        combo_name = " + ".join(combo)
        is_current = combo == current_combo or (len(combo) == 2 and combo[::-1] == current_combo)

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

                if metric == 'AvgSpeed':
                    # Speed invertálva: magasabb speed = jobb = magasabb reward
                    # R = sigmoid((log(x) - mu) / std) — NEM 1-sigmoid!
                    z = (log_v - mu) / (std + 1e-9)
                    r_component = 1.0 / (1.0 + np.exp(-z))
                else:
                    # Waiting, CO2, Queue, TT, Occ: alacsonyabb = jobb
                    z = (log_v - mu) / (std + 1e-9)
                    r_component = 1.0 - 1.0 / (1.0 + np.exp(-z))

                reward_components.append(r_component)

            rewards = np.mean(reward_components, axis=0)
            flow_levels = jdf['flow_level'].values

            all_rewards.extend(rewards)
            all_flow_levels.extend(flow_levels)

            # Per-junction monotonitás
            flow_groups = jdf.groupby('flow_level').apply(
                lambda g: np.mean([
                    np.mean(reward_components[i][g.index - jdf.index[0]])
                    if len(g) > 0 else 0.5
                    for i, _ in enumerate(combo)
                ]) if len(g) > 0 else 0.5
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
            'is_current': is_current,
        })

    # --- Eredmények kiírása ---
    print(f"\n  {'Kombináció':35} {'Mono_r':>8} {'Mono%':>6} {'IQR':>6} {'IQR%':>6} "
          f"{'η²':>6} {'1-η²':>6} {'Szűrő':>8}")
    print("  " + "-" * 95)

    # Rendezés: átmenők first, azon belül eta2 ascending (alacsonyabb = jobb)
    combo_results.sort(key=lambda x: (not x['passed'], x['avg_eta2']))

    for cr in combo_results:
        marker = " ← JELENLEGI" if cr['is_current'] else ""
        status = "✓ PASS" if cr['passed'] else "✗ FAIL"
        mono_flag = "" if cr['mono_ok'] else " [MONO!]"
        iqr_flag = "" if cr['iqr_ok'] else " [IQR!]"

        print(f"  {cr['combo']:35} {cr['avg_monotonicity']:+8.3f} {cr['mono_pass_pct']:5.0f}% "
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
        print(f"  {'#':>3} {'Kombináció':35} {'η²':>8} {'1-η²':>8} {'Mono_r':>8} {'IQR':>6}")
        print("  " + "-" * 75)
        for i, cr in enumerate(passed):
            marker = " ← JELENLEGI" if cr['is_current'] else ""
            print(f"  {i+1:3} {cr['combo']:35} {cr['avg_eta2']:8.4f} {cr['eta2_within']:8.4f} "
                  f"{cr['avg_monotonicity']:+8.3f} {cr['avg_iqr']:6.3f}{marker}")

    # =====================================================================
    # 4. NORMALIZÁCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA
    # =====================================================================
    print("\n" + "=" * 100)
    print("  4. NORMALIZÁCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA")
    print("     A legjobb metrika-kombó(ka)t teszteljük különböző normalizációkkal")
    print("=" * 100)

    # Top 3 átmenő kombó + jelenlegi
    test_combos = []
    if passed:
        test_combos = [cr['combo_tuple'] for cr in passed[:3]]
    current_in_passed = any(cr['is_current'] for cr in passed)
    if not current_in_passed:
        test_combos.append(current_combo)

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

                    if method_name in ['linear-sigmoid', 'sqrt-sigmoid']:
                        r_component = norm_fn(vals, mu, std_v)
                    else:
                        r_component = norm_fn(vals, mu, std_v)

                    if metric == 'AvgSpeed':
                        r_component = 1.0 - r_component  # Invert for speed

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
        print(f"\n  {'Kombináció':30} {'Módszer':18} {'Mono_r':>8} {'IQR':>6} {'η²':>8} {'1-η²':>8} {'Szűrő':>8}")
        print("  " + "-" * 100)
        norm_results.sort(key=lambda x: (x['combo'], x['avg_eta2']))
        for nr in norm_results:
            status = "✓" if nr['mono_ok'] and nr['iqr_ok'] else "✗"
            print(f"  {nr['combo']:30} {nr['method']:18} {nr['avg_mono']:+8.3f} {nr['avg_iqr']:6.3f} "
                  f"{nr['avg_eta2']:8.4f} {1-nr['avg_eta2']:8.4f} {status:>8}")

    # =====================================================================
    # 5. ÁBRÁK
    # =====================================================================

    # Ábra 1: Korreláció mátrix
    fig, ax = plt.subplots(figsize=(10, 8))
    import matplotlib.colors as mcolors
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(candidate_metrics)))
    ax.set_xticklabels(candidate_metrics, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(candidate_metrics)))
    ax.set_yticklabels(candidate_metrics, fontsize=10)
    for i in range(len(candidate_metrics)):
        for j in range(len(candidate_metrics)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            weight = 'bold' if abs(val) > 0.85 and i != j else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Metric Correlation Matrix (log-transformed)\n* Bold = redundant (|r| > 0.85)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_correlation_matrix.png'), dpi=200)
    plt.close()
    print(f"\n  reward_correlation_matrix.png mentve")

    # Ábra 2: Kombináció szűrés eredménye
    if combo_results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Rendezés: passed first, eta2 ascending
        sorted_cr = sorted(combo_results, key=lambda x: (not x['passed'], x['avg_eta2']))
        names = [cr['combo'] for cr in sorted_cr]
        colors = ['#2ecc71' if cr['passed'] else '#e74c3c' for cr in sorted_cr]
        current_idx = next((i for i, cr in enumerate(sorted_cr) if cr['is_current']), None)

        y_pos = np.arange(len(names))

        # 2a. η² (lower = better)
        ax = axes[0]
        bars = ax.barh(y_pos, [cr['avg_eta2'] for cr in sorted_cr], color=colors, alpha=0.8)
        if current_idx is not None:
            bars[current_idx].set_edgecolor('blue')
            bars[current_idx].set_linewidth(3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('η² (lower = agent effect more visible)')
        ax.set_title('ANOVA η²\n(flow_level explained variance)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # 2b. Monotonitás
        ax = axes[1]
        mono_vals = [cr['avg_monotonicity'] for cr in sorted_cr]
        bars = ax.barh(y_pos, mono_vals, color=colors, alpha=0.8)
        if current_idx is not None:
            bars[current_idx].set_edgecolor('blue')
            bars[current_idx].set_linewidth(3)
        ax.axvline(-0.5, color='red', ls='--', lw=2, label='Threshold (-0.5)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Spearman ρ (more negative = correct)')
        ax.set_title('Monotonicity\n(flow ↑ → reward ↓)')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

        # 2c. IQR
        ax = axes[2]
        iqr_vals = [cr['avg_iqr'] for cr in sorted_cr]
        bars = ax.barh(y_pos, iqr_vals, color=colors, alpha=0.8)
        if current_idx is not None:
            bars[current_idx].set_edgecolor('blue')
            bars[current_idx].set_linewidth(3)
        ax.axvline(0.10, color='red', ls='--', lw=2, label='Threshold (0.10)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('IQR (higher = more differentiation)')
        ax.set_title('Reward IQR\n(compression check)')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Metric Combination Selection — Green = PASS, Red = FAIL, Blue border = current',
                      fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_combo_selection.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_combo_selection.png mentve")

    # Ábra 3: Normalizáció összehasonlítás a top kombó(k)ra
    if norm_results:
        combos_in_results = sorted(set(nr['combo'] for nr in norm_results))
        n_combos = len(combos_in_results)
        fig, axes = plt.subplots(1, n_combos, figsize=(7 * n_combos, 6))
        if n_combos == 1:
            axes = [axes]

        for idx, combo_name in enumerate(combos_in_results):
            ax = axes[idx]
            c_results = [nr for nr in norm_results if nr['combo'] == combo_name]
            c_results.sort(key=lambda x: x['avg_eta2'])
            methods = [nr['method'] for nr in c_results]
            eta2s = [nr['avg_eta2'] for nr in c_results]
            colors_n = ['#2ecc71' if nr['mono_ok'] and nr['iqr_ok'] else '#e74c3c' for nr in c_results]

            y = np.arange(len(methods))
            ax.barh(y, eta2s, color=colors_n, alpha=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(methods, fontsize=10)
            ax.set_xlabel('η²')
            ax.set_title(f'{combo_name}')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Normalization Methods — η² by Metric Combination', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'reward_normalization_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  reward_normalization_comparison.png mentve")

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

        current_entry = next((cr for cr in combo_results if cr['is_current']), None)
        if current_entry:
            print(f"\n  Jelenlegi (TotalWait + TotalCO2):")
            print(f"    η² = {current_entry['avg_eta2']:.4f}")
            print(f"    Monotonitás = {current_entry['avg_monotonicity']:+.3f}")
            print(f"    IQR = {current_entry['avg_iqr']:.3f}")
            print(f"    Szűrő: {'✓ PASS' if current_entry['passed'] else '✗ FAIL'}")

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


def main():
    parser = argparse.ArgumentParser(description='Per-junction metric collection and normalization calibration')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip simulation, only run analysis on existing CSVs')
    args = parser.parse_args()

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
        print(f"  Epizodok szintenkent: {EPISODES_PER_LEVEL}")
        print(f"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s")
        print(f"  Kimenet: {OUTPUT_DIR}")
        print("=" * 80)

        total_sims = len(FLOW_MAX_LEVELS) * EPISODES_PER_LEVEL
        sim_count = 0

        for flow_max in FLOW_MAX_LEVELS:
            for ep in range(EPISODES_PER_LEVEL):
                sim_count += 1
                print(f"\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | "
                      f"Epizod {ep+1}/{EPISODES_PER_LEVEL} ---")
                run_simulation(flow_max, ep, OUTPUT_DIR)

        print(f"\n  Szimulacio kesz! ({total_sims} epizod)")
    else:
        print("  --skip-simulation: Csak elemzes a meglevo CSV-kre")

    # Per-junction kalibráció
    analyze_per_junction(OUTPUT_DIR)

    # Reward metrika és normalizáció kiválasztás
    reward_selection_analysis(OUTPUT_DIR)


if __name__ == "__main__":
    main()
