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


def check_fit_quality(output_dir):
    """
    Illeszkedés-ellenőrzés: mennyire felel meg a log-sigmoid normalizáció az adatoknak.

    Vizsgálatok:
      1. Log-normalitás teszt (Anderson-Darling + ferdeség/csúcsosság)
      2. Sigmoid lefedettség (reward hány %-a esik a hasznos [0.15-0.85] zónába)
      3. QQ-plotok junction-önként
      4. Összefoglaló: melyik junction problémás
    """
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("  ILLESZKEDÉS-ELLENŐRZÉS (GOODNESS-OF-FIT)")
    print("=" * 80)

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

    # --- Junction reward params betöltése ---
    params_path = os.path.join(DATA_DIR, "junction_reward_params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(output_dir, "junction_reward_params.json")
    with open(params_path) as f:
        junction_params = json.load(f)

    # =====================================================================
    # 1. LOG-NORMALITÁS TESZT
    # =====================================================================
    print(f"\n--- 1. LOG-NORMALITAS TESZT (Anderson-Darling) ---")
    print(f"  H0: log(metrika) normális eloszlást követ")
    print(f"  Ha a statisztika < kritikus érték (5%), elfogadjuk H0-t\n")

    print(f"{'Junction':15} {'Metrika':20} {'AD stat':>10} {'Crit(5%)':>10} {'Ferdeség':>10} "
          f"{'Csúcsosság':>10} {'Verdict':>10}")
    print("-" * 100)

    fit_results = []

    for jid in junction_ids:
        jdf = full_df[full_df['junction'] == jid]
        jdf_valid = jdf[jdf['VehCount'] > 0]

        for metric_name, col in [('TotalWaitingTime', 'TotalWaitingTime'), ('TotalCO2', 'TotalCO2')]:
            vals = jdf_valid[col].values
            vals_pos = vals[vals > 0]

            if len(vals_pos) < 50:
                continue

            log_vals = np.log(vals_pos + 1e-5)

            # Anderson-Darling teszt
            ad_result = stats.anderson(log_vals, dist='norm')
            ad_stat = ad_result.statistic
            # 5%-os szignifikanciaszint (index 2)
            ad_crit_5pct = ad_result.critical_values[2]
            passed = ad_stat < ad_crit_5pct

            # Ferdeség és csúcsosság
            skewness = float(stats.skew(log_vals))
            kurtosis = float(stats.kurtosis(log_vals))  # excess kurtosis (norm = 0)

            verdict = "OK" if passed else "FAIL"
            # Enyhe ferdeség/csúcsosság is elfogadható a sigmoid-hoz
            if not passed and abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
                verdict = "MARGINAL"

            print(f"{jid:15} {metric_name:20} {ad_stat:10.3f} {ad_crit_5pct:10.3f} "
                  f"{skewness:+10.3f} {kurtosis:+10.3f} {verdict:>10}")

            fit_results.append({
                'junction': jid,
                'metric': metric_name,
                'ad_statistic': ad_stat,
                'ad_critical_5pct': ad_crit_5pct,
                'ad_passed': passed,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'verdict': verdict,
                'n_samples': len(vals_pos),
            })

    # =====================================================================
    # 2. SIGMOID LEFEDETTSÉG
    # =====================================================================
    print(f"\n--- 2. SIGMOID LEFEDETTSEG ---")
    print(f"  Hasznos zóna: reward ∈ [0.15, 0.85] (ahol a sigmoid gradiens informatív)")
    print(f"  Telített zóna: reward < 0.15 vagy > 0.85 (ahol a gradiens ≈ 0)\n")

    print(f"{'Junction':15} {'Hasznos%':>10} {'Telített%':>10} {'<0.15%':>8} {'>0.85%':>8} "
          f"{'R_mean':>8} {'R_std':>8} {'Verdict':>10}")
    print("-" * 95)

    coverage_results = []

    for jid in junction_ids:
        jdf = full_df[full_df['junction'] == jid]
        mask_pos = (jdf['TotalWaitingTime'].values > 0) & (jdf['TotalCO2'].values > 0)
        w_pos = jdf['TotalWaitingTime'].values[mask_pos]
        c_pos = jdf['TotalCO2'].values[mask_pos]

        if len(w_pos) < 10:
            continue

        p = junction_params[jid]
        rewards = np.array([calc_reward(w, c, p['MU_WAIT'], p['STD_WAIT'], p['MU_CO2'], p['STD_CO2'])
                            for w, c in zip(w_pos, c_pos)])

        useful_pct = np.mean((rewards >= 0.15) & (rewards <= 0.85)) * 100
        saturated_low = np.mean(rewards < 0.15) * 100
        saturated_high = np.mean(rewards > 0.85) * 100
        saturated_total = saturated_low + saturated_high
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)

        # Verdict: jó ha >60% hasznos, problémás ha <40%
        if useful_pct >= 60:
            verdict = "JO"
        elif useful_pct >= 40:
            verdict = "ELFOGADHATO"
        else:
            verdict = "GYENGE"

        print(f"{jid:15} {useful_pct:10.1f} {saturated_total:10.1f} {saturated_low:8.1f} "
              f"{saturated_high:8.1f} {r_mean:8.3f} {r_std:8.3f} {verdict:>10}")

        coverage_results.append({
            'junction': jid,
            'useful_pct': useful_pct,
            'saturated_low_pct': saturated_low,
            'saturated_high_pct': saturated_high,
            'reward_mean': r_mean,
            'reward_std': r_std,
            'verdict': verdict,
        })

    # =====================================================================
    # 3. QQ-PLOTOK
    # =====================================================================
    print(f"\n--- 3. QQ-PLOTOK generálása ---")

    n_junctions = len(junction_ids)
    n_cols = 5
    n_rows = (n_junctions + n_cols - 1) // n_cols

    for metric_name, col in [('TotalWaitingTime', 'TotalWaitingTime'), ('TotalCO2', 'TotalCO2')]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'QQ-Plot: log({metric_name}) vs Normal — per Junction', fontsize=14)

        for idx, jid in enumerate(junction_ids):
            row, col_idx = idx // n_cols, idx % n_cols
            ax = axes[row, col_idx]

            jdf = full_df[full_df['junction'] == jid]
            jdf_valid = jdf[jdf['VehCount'] > 0]
            vals = jdf_valid[col].values
            vals_pos = vals[vals > 0]

            if len(vals_pos) < 50:
                ax.set_title(f'{jid}\n(insufficient)', fontsize=9)
                continue

            log_vals = np.log(vals_pos + 1e-5)

            # QQ-plot
            (osm_vals, fit_line), (slope, intercept, r) = stats.probplot(log_vals, dist='norm')
            ax.scatter(osm_vals, sorted(log_vals), s=1, alpha=0.3, color='steelblue')
            # Fit line
            x_line = np.array([osm_vals[0], osm_vals[-1]])
            ax.plot(x_line, slope * x_line + intercept, 'r-', lw=1.5)

            # R² és AD stat keresése
            fit_row = [r for r in fit_results if r['junction'] == jid and r['metric'] == metric_name]
            ad_info = f"AD={fit_row[0]['ad_statistic']:.1f}" if fit_row else ""
            verdict = fit_row[0]['verdict'] if fit_row else ""

            ax.set_title(f'{jid}\nR²={r**2:.4f} {ad_info} [{verdict}]', fontsize=8)
            ax.set_xlabel('Theoretical', fontsize=7)
            ax.set_ylabel('Observed', fontsize=7)
            ax.tick_params(labelsize=6)

        for idx in range(len(junction_ids), n_rows * n_cols):
            row, col_idx = idx // n_cols, idx % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        fname = f'fit_qq_{metric_name.lower()}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  {fname} mentve")

    # =====================================================================
    # 4. SIGMOID LEFEDETTSÉG ÁBRA
    # =====================================================================
    if coverage_results:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Sigmoid Coverage per Junction (Local params)\nGreen = useful [0.15-0.85], Red = saturated',
                     fontsize=14)

        for idx, jid in enumerate(junction_ids):
            row, col_idx = idx // n_cols, idx % n_cols
            ax = axes[row, col_idx]

            jdf = full_df[full_df['junction'] == jid]
            mask_pos = (jdf['TotalWaitingTime'].values > 0) & (jdf['TotalCO2'].values > 0)
            w_pos = jdf['TotalWaitingTime'].values[mask_pos]
            c_pos = jdf['TotalCO2'].values[mask_pos]

            if len(w_pos) < 10:
                ax.set_title(f'{jid}\n(insufficient)', fontsize=9)
                continue

            p = junction_params[jid]
            rewards = np.array([calc_reward(w, c, p['MU_WAIT'], p['STD_WAIT'], p['MU_CO2'], p['STD_CO2'])
                                for w, c in zip(w_pos, c_pos)])

            # Hisztogram színezéssel
            bins = np.linspace(0, 1, 41)
            n_vals, bin_edges, patches = ax.hist(rewards, bins=bins, density=True, alpha=0.8)
            for patch, left_edge in zip(patches, bin_edges[:-1]):
                center = left_edge + (bin_edges[1] - bin_edges[0]) / 2
                if 0.15 <= center <= 0.85:
                    patch.set_facecolor('#2ca02c')  # zöld — hasznos
                else:
                    patch.set_facecolor('#d62728')  # piros — telített

            ax.axvline(0.15, color='black', ls=':', lw=1, alpha=0.7)
            ax.axvline(0.85, color='black', ls=':', lw=1, alpha=0.7)

            cov_row = [c for c in coverage_results if c['junction'] == jid]
            useful = cov_row[0]['useful_pct'] if cov_row else 0
            verdict = cov_row[0]['verdict'] if cov_row else ''
            ax.set_title(f'{jid}\nHasznos: {useful:.0f}% [{verdict}]', fontsize=9)
            ax.set_xlim(0, 1)

        for idx in range(len(junction_ids), n_rows * n_cols):
            row, col_idx = idx // n_cols, idx % n_cols
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fit_sigmoid_coverage.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  fit_sigmoid_coverage.png mentve")

    # =====================================================================
    # 5. ÖSSZEFOGLALÓ
    # =====================================================================
    print(f"\n{'=' * 80}")
    print("  ILLESZKEDÉS ÖSSZEFOGLALÓ")
    print(f"{'=' * 80}")

    if fit_results:
        fit_df = pd.DataFrame(fit_results)
        fit_df.to_csv(os.path.join(output_dir, 'fit_quality_results.csv'), index=False)
        print(f"  fit_quality_results.csv mentve")

        # Anderson-Darling összefoglaló
        for metric in ['TotalWaitingTime', 'TotalCO2']:
            mdf = fit_df[fit_df['metric'] == metric]
            n_ok = (mdf['verdict'] == 'OK').sum()
            n_marg = (mdf['verdict'] == 'MARGINAL').sum()
            n_fail = (mdf['verdict'] == 'FAIL').sum()
            print(f"\n  {metric}:")
            print(f"    Log-normalitás OK:       {n_ok}/{len(mdf)}")
            print(f"    Log-normalitás MARGINAL: {n_marg}/{len(mdf)}")
            print(f"    Log-normalitás FAIL:     {n_fail}/{len(mdf)}")
            print(f"    Átlag ferdeség:  {mdf['skewness'].mean():+.3f}  (ideális: 0)")
            print(f"    Átlag csúcsosság: {mdf['kurtosis'].mean():+.3f}  (ideális: 0)")

    if coverage_results:
        cov_df = pd.DataFrame(coverage_results)
        n_jo = (cov_df['verdict'] == 'JO').sum()
        n_elf = (cov_df['verdict'] == 'ELFOGADHATO').sum()
        n_gyenge = (cov_df['verdict'] == 'GYENGE').sum()
        print(f"\n  Sigmoid lefedettség (lokális paraméterekkel):")
        print(f"    JÓ (>60% hasznos):        {n_jo}/{len(cov_df)}")
        print(f"    ELFOGADHATÓ (40-60%):      {n_elf}/{len(cov_df)}")
        print(f"    GYENGE (<40%):             {n_gyenge}/{len(cov_df)}")
        print(f"    Átlag hasznos%: {cov_df['useful_pct'].mean():.1f}%")
        print(f"    Min hasznos%:   {cov_df['useful_pct'].min():.1f}% ({cov_df.loc[cov_df['useful_pct'].idxmin(), 'junction']})")
        print(f"    Max hasznos%:   {cov_df['useful_pct'].max():.1f}% ({cov_df.loc[cov_df['useful_pct'].idxmax(), 'junction']})")

        if n_gyenge > 0:
            print(f"\n  FIGYELEM: {n_gyenge} junction(ök)nél gyenge a sigmoid lefedettség!")
            print(f"  Érintett junction-ök:")
            for _, row in cov_df[cov_df['verdict'] == 'GYENGE'].iterrows():
                print(f"    {row['junction']}: hasznos={row['useful_pct']:.1f}%, "
                      f"mean_R={row['reward_mean']:.3f}, std_R={row['reward_std']:.3f}")
            print(f"  → Ezeknél érdemes lehet kézi STD skálázás vagy másféle normalizáció.")

    # --- Végső konklúzió ---
    print(f"\n{'=' * 80}")
    print("  KONKLUZIO")
    print(f"{'=' * 80}")
    if fit_results and coverage_results:
        fit_df = pd.DataFrame(fit_results)
        cov_df = pd.DataFrame(coverage_results)

        total_ad_ok = ((fit_df['verdict'] == 'OK') | (fit_df['verdict'] == 'MARGINAL')).sum()
        total_ad = len(fit_df)
        avg_useful = cov_df['useful_pct'].mean()

        print(f"  Log-normalitás:    {total_ad_ok}/{total_ad} metrika-junction pár OK/MARGINAL "
              f"({total_ad_ok/total_ad*100:.0f}%)")
        print(f"  Sigmoid hasznos%:  átlag {avg_useful:.1f}%")

        if total_ad_ok / total_ad >= 0.7 and avg_useful >= 50:
            print(f"\n  ✓ A log-sigmoid normalizáció MEGFELELŐ a per-junction adatokra.")
            print(f"    A lokális MU/STD paraméterek használhatók.")
        elif total_ad_ok / total_ad >= 0.5 and avg_useful >= 40:
            print(f"\n  ~ A log-sigmoid normalizáció ELFOGADHATÓ, de van tér a javításra.")
            print(f"    Javaslat: STD paraméterek finomhangolása a gyengébb junction-öknél.")
        else:
            print(f"\n  ✗ A log-sigmoid normalizáció PROBLÉMÁS néhány junction-nél.")
            print(f"    Javaslat: alternatív normalizáció vizsgálata (pl. quantile, adaptive).")


def compare_normalization_methods(output_dir):
    """
    Különböző normalizációs módszerek összehasonlítása junction-önként.

    Tesztelt módszerek:
      1. log-sigmoid (jelenlegi)     : R = 1 - sigmoid((log(x) - mu) / std)
      2. linear-sigmoid              : R = 1 - sigmoid((x - mu) / std)
      3. sqrt-sigmoid                : R = 1 - sigmoid((sqrt(x) - mu) / std)
      4. quantile (empirikus CDF)    : R = 1 - F_empirikus(x)
      5. min-max                     : R = 1 - (x - p5) / (p95 - p5)
      6. box-cox + sigmoid           : R = 1 - sigmoid((boxcox(x) - mu) / std)

    Értékelési szempontok:
      - Reward range (p90-p10): minél nagyobb, annál informatívabb
      - Sigmoid hasznos zóna [0.15-0.85]: minél több adat esik ide
      - Reward std: minél nagyobb, annál jobban differenciál
      - Normalitás (Anderson-Darling): a transzformált adat illeszkedése
    """
    from scipy import stats as sp_stats
    from scipy.special import inv_boxcox
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("  NORMALIZACIOS MODSZEREK OSSZEHASONLITASA")
    print("=" * 80)

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

    # =====================================================================
    # Normalizációs módszerek definíciója
    # =====================================================================

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    # =================================================================
    # 1. LOG-SIGMOID (jelenlegi módszer)
    #    R = 1 - sigmoid((log(x) - mu) / std)
    # =================================================================
    def method_log_sigmoid(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        log_v = np.log(vals_pos + 1e-5)
        mu, std = np.mean(log_v), np.std(log_v)
        rewards = 1.0 - sigmoid((log_v - mu) / (std + 1e-9))
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 2. LINEAR-SIGMOID
    #    R = 1 - sigmoid((x - mu) / std)
    # =================================================================
    def method_linear_sigmoid(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        mu, std = np.mean(vals_pos), np.std(vals_pos)
        rewards = 1.0 - sigmoid((vals_pos - mu) / (std + 1e-9))
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 3. SQRT-SIGMOID
    #    R = 1 - sigmoid((sqrt(x) - mu) / std)
    # =================================================================
    def method_sqrt_sigmoid(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        sqrt_v = np.sqrt(vals_pos)
        mu, std = np.mean(sqrt_v), np.std(sqrt_v)
        rewards = 1.0 - sigmoid((sqrt_v - mu) / (std + 1e-9))
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 4. QUANTILE (empirikus CDF)
    #    R = 1 - F_empirikus(x)
    # =================================================================
    def method_quantile(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        sorted_vals = np.sort(vals_pos)
        ranks = np.searchsorted(sorted_vals, vals_pos, side='right')
        rewards = 1.0 - ranks / len(vals_pos)
        params = {'p5': np.percentile(vals_pos, 5), 'p95': np.percentile(vals_pos, 95)}
        return rewards, params, vals_pos

    # =================================================================
    # 5. MIN-MAX (p5-p95 clipped)
    #    R = clip(1 - (x - p5) / (p95 - p5), 0, 1)
    # =================================================================
    def method_minmax(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        p5 = np.percentile(vals_pos, 5)
        p95 = np.percentile(vals_pos, 95)
        if p95 - p5 < 1e-9:
            return None, None, None
        normalized = (vals_pos - p5) / (p95 - p5)
        rewards = np.clip(1.0 - normalized, 0, 1)
        params = {'p5': p5, 'p95': p95}
        return rewards, params, vals_pos

    # =================================================================
    # 6. BOX-COX + SIGMOID
    #    R = 1 - sigmoid((boxcox(x, lambda) - mu) / std)
    # =================================================================
    def method_boxcox_sigmoid(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        try:
            bc_vals, lmbda = sp_stats.boxcox(vals_pos)
            mu, std = np.mean(bc_vals), np.std(bc_vals)
            rewards = 1.0 - sigmoid((bc_vals - mu) / (std + 1e-9))
            params = {'lambda': lmbda, 'mu': mu, 'std': std}
            return rewards, params, vals_pos
        except Exception:
            return None, None, None

    # =================================================================
    # 7. TANH (Z-score)
    #    z = (x - mu) / std, R = (1 - tanh(z)) / 2  →  [0, 1]
    # =================================================================
    def method_tanh_zscore(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        mu, std = np.mean(vals_pos), np.std(vals_pos)
        z = (vals_pos - mu) / (std + 1e-9)
        rewards = (1.0 - np.tanh(z)) / 2.0
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 8. LOG-TANH (log transzformáció + tanh)
    #    z = (log(x) - mu) / std, R = (1 - tanh(z)) / 2
    # =================================================================
    def method_log_tanh(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        log_v = np.log(vals_pos + 1e-5)
        mu, std = np.mean(log_v), np.std(log_v)
        z = (log_v - mu) / (std + 1e-9)
        rewards = (1.0 - np.tanh(z)) / 2.0
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 9. ALGEBRAIC
    #    R = 1 - z / sqrt(1 + z^2),  z = (x - mu) / std,  → [0, 1]
    # =================================================================
    def method_algebraic(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        mu, std = np.mean(vals_pos), np.std(vals_pos)
        z = (vals_pos - mu) / (std + 1e-9)
        # algebraic sigmoid: z / sqrt(1 + z^2) → [-1, 1], shift to [0, 1]
        rewards = (1.0 - z / np.sqrt(1.0 + z**2)) / 2.0
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 10. LOG-ALGEBRAIC
    #    z = (log(x) - mu) / std, R = (1 - z/sqrt(1+z^2)) / 2
    # =================================================================
    def method_log_algebraic(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        log_v = np.log(vals_pos + 1e-5)
        mu, std = np.mean(log_v), np.std(log_v)
        z = (log_v - mu) / (std + 1e-9)
        rewards = (1.0 - z / np.sqrt(1.0 + z**2)) / 2.0
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    # =================================================================
    # 11. ROBUST IQR
    #    z = (x - median) / IQR,  R = 1 - sigmoid(z)
    # =================================================================
    def method_robust_iqr(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        median = np.median(vals_pos)
        q25 = np.percentile(vals_pos, 25)
        q75 = np.percentile(vals_pos, 75)
        iqr = q75 - q25
        if iqr < 1e-9:
            return None, None, None
        z = (vals_pos - median) / iqr
        rewards = 1.0 - sigmoid(z)
        params = {'median': median, 'q25': q25, 'q75': q75}
        return rewards, params, vals_pos

    # =================================================================
    # 12. LOG-ROBUST IQR
    #    z = (log(x) - median_log) / IQR_log,  R = 1 - sigmoid(z)
    # =================================================================
    def method_log_robust_iqr(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        log_v = np.log(vals_pos + 1e-5)
        median = np.median(log_v)
        q25 = np.percentile(log_v, 25)
        q75 = np.percentile(log_v, 75)
        iqr = q75 - q25
        if iqr < 1e-9:
            return None, None, None
        z = (log_v - median) / iqr
        rewards = 1.0 - sigmoid(z)
        params = {'median': median, 'q25': q25, 'q75': q75}
        return rewards, params, vals_pos

    # =================================================================
    # 13. RANK GAUSS (rank → inverse normal CDF → z-score)
    #    R = 1 - sigmoid(z)  ahol z = Phi^{-1}(rank/N)
    # =================================================================
    def method_rank_gauss(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        n = len(vals_pos)
        sorted_vals = np.sort(vals_pos)
        ranks = np.searchsorted(sorted_vals, vals_pos, side='right')
        # Ranks → (0, 1) tartomány (elkerülve a 0 és 1 széleket)
        u = (ranks - 0.5) / n
        u = np.clip(u, 1e-6, 1 - 1e-6)
        # Inverse normal CDF → z-score
        z = sp_stats.norm.ppf(u)
        rewards = 1.0 - sigmoid(z)
        params = {'n_samples': n}
        return rewards, params, vals_pos

    # =================================================================
    # 14. CLIPPED 95% MIN-MAX
    #    clip to [p2.5, p97.5], then min-max → [0, 1]
    # =================================================================
    def method_clipped95_minmax(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        p2_5 = np.percentile(vals_pos, 2.5)
        p97_5 = np.percentile(vals_pos, 97.5)
        if p97_5 - p2_5 < 1e-9:
            return None, None, None
        clipped = np.clip(vals_pos, p2_5, p97_5)
        normalized = (clipped - p2_5) / (p97_5 - p2_5)
        rewards = 1.0 - normalized
        params = {'p2.5': p2_5, 'p97.5': p97_5}
        return rewards, params, vals_pos

    # =================================================================
    # 15. HARMONIC (1/x alapú)
    #    R = x_min / x  (alacsony érték = jó → magas reward)
    # =================================================================
    def method_harmonic(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        p5 = np.percentile(vals_pos, 5)
        if p5 < 1e-9:
            p5 = np.min(vals_pos[vals_pos > 0])
        rewards = np.clip(p5 / (vals_pos + 1e-9), 0, 1)
        params = {'p5': p5}
        return rewards, params, vals_pos

    # =================================================================
    # 16. LOG-NORMAL CDF
    #    R = 1 - Phi((log(x) - mu) / std)  — analitikus lognormál CDF
    # =================================================================
    def method_lognormal_cdf(vals):
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None, None, None
        log_v = np.log(vals_pos + 1e-5)
        mu, std = np.mean(log_v), np.std(log_v)
        # Normális CDF a log értékekre
        rewards = 1.0 - sp_stats.norm.cdf(log_v, loc=mu, scale=std + 1e-9)
        params = {'mu': mu, 'std': std}
        return rewards, params, vals_pos

    methods = [
        # Jelenlegi
        ('log-sigmoid', method_log_sigmoid),
        # Sigmoid variánsok
        ('linear-sigmoid', method_linear_sigmoid),
        ('sqrt-sigmoid', method_sqrt_sigmoid),
        ('box-cox-sigmoid', method_boxcox_sigmoid),
        # Tanh variánsok
        ('tanh-zscore', method_tanh_zscore),
        ('log-tanh', method_log_tanh),
        # Algebraic variánsok
        ('algebraic', method_algebraic),
        ('log-algebraic', method_log_algebraic),
        # Robust (IQR-based)
        ('robust-iqr', method_robust_iqr),
        ('log-robust-iqr', method_log_robust_iqr),
        # Rank-based
        ('quantile', method_quantile),
        ('rank-gauss', method_rank_gauss),
        ('lognormal-cdf', method_lognormal_cdf),
        # Range-based
        ('min-max', method_minmax),
        ('clipped95-minmax', method_clipped95_minmax),
        ('harmonic', method_harmonic),
    ]

    # =====================================================================
    # Minden junction + metrika + módszer tesztelése
    # =====================================================================
    all_results = []

    for jid in junction_ids:
        jdf = full_df[full_df['junction'] == jid]
        jdf_valid = jdf[jdf['VehCount'] > 0]

        for metric_name in ['TotalWaitingTime', 'TotalCO2']:
            raw_vals = jdf_valid[metric_name].values

            for method_name, method_fn in methods:
                rewards, params, vals_pos = method_fn(raw_vals)
                if rewards is None:
                    continue

                # --- Metrikák ---
                r_range = float(np.percentile(rewards, 90) - np.percentile(rewards, 10))
                r_std = float(np.std(rewards))
                r_mean = float(np.mean(rewards))
                useful_pct = float(np.mean((rewards >= 0.15) & (rewards <= 0.85)) * 100)
                sat_low = float(np.mean(rewards < 0.15) * 100)
                sat_high = float(np.mean(rewards > 0.85) * 100)

                # Monotonitás: reward vs flow level
                # vals_pos a method_fn-ből jött — az jdf_valid-ból szűrt pozitív értékek
                # A flow_level infó a jdf_valid-ban van, de a vals_pos indexelése eltérhet
                # Ezért a full jdf_valid-on számolunk
                mono_corr = 0.0
                flow_vals = jdf_valid['flow_level'].values
                raw_metric = jdf_valid[metric_name].values
                pos_mask = raw_metric > 0
                if pos_mask.sum() == len(rewards):
                    flow_for_rewards = flow_vals[pos_mask]
                    unique_flows_m = sorted(set(flow_for_rewards))
                    if len(unique_flows_m) >= 3:
                        from scipy.stats import spearmanr as _spearmanr
                        flow_r_means = [np.mean(rewards[flow_for_rewards == fl]) for fl in unique_flows_m]
                        mono_corr, _ = _spearmanr(unique_flows_m, flow_r_means)

                all_results.append({
                    'junction': jid,
                    'metric': metric_name,
                    'method': method_name,
                    'reward_range': r_range,
                    'reward_std': r_std,
                    'reward_mean': r_mean,
                    'useful_pct': useful_pct,
                    'saturated_low_pct': sat_low,
                    'saturated_high_pct': sat_high,
                    'monotonicity_corr': mono_corr,
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'normalization_methods_comparison.csv'), index=False)
    print(f"  normalization_methods_comparison.csv mentve")

    # =====================================================================
    # Összefoglaló tábla — módszerenként átlagolt eredmények
    # =====================================================================
    print(f"\n{'=' * 100}")
    print(f"  MODSZEREK ATLAGOS TELJESITMENYE (mindket metrikara, minden junction-re)")
    print(f"{'=' * 100}")

    print(f"\n  SCORING: 25% range + 20% std + 25% useful% + 30% monotonitás")
    print(f"  Monotonitás = Spearman(flow_level, mean_reward): -1 = tökéletes csökkenés\n")

    print(f"{'Módszer':20} {'R_range':>8} {'R_std':>8} {'Use%':>6} "
          f"{'Mono_r':>8} {'R_mean':>8} {'Score':>8}")
    print("-" * 72)

    method_order = [m[0] for m in methods]
    method_scores = {}

    for method in method_order:
        mdf = results_df[results_df['method'] == method]
        if len(mdf) == 0:
            continue
        r_range = mdf['reward_range'].mean()
        r_std = mdf['reward_std'].mean()
        useful = mdf['useful_pct'].mean()
        r_mean = mdf['reward_mean'].mean()
        mono = mdf['monotonicity_corr'].mean() if 'monotonicity_corr' in mdf.columns else 0.0

        mono_norm = (-mono + 1) / 2  # [-1,+1] → [0,1]
        score = r_range * 0.25 + r_std * 0.20 + (useful / 100) * 0.25 + mono_norm * 0.30
        method_scores[method] = score

        print(f"{method:20} {r_range:8.4f} {r_std:8.4f} {useful:6.1f} "
              f"{mono:+8.3f} {r_mean:8.4f} {score:8.4f}")

    # Rangsor
    print(f"\n--- RANGSOR (25% range + 20% std + 25% useful% + 30% monotonitás) ---")
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (method, score) in enumerate(sorted_methods, 1):
        marker = " <<<< BEST" if rank == 1 else ""
        print(f"  {rank}. {method:20} score={score:.4f}{marker}")

    # =====================================================================
    # Per-junction legjobb módszer
    # =====================================================================
    print(f"\n{'=' * 100}")
    print(f"  PER-JUNCTION LEGJOBB MODSZER")
    print(f"{'=' * 100}")

    print(f"\n{'Junction':15} {'Wait_best':20} {'Wait_range':>10} {'CO2_best':20} {'CO2_range':>10}")
    print("-" * 80)

    best_methods_per_junction = {}

    for jid in junction_ids:
        best = {}
        for metric in ['TotalWaitingTime', 'TotalCO2']:
            mdf = results_df[(results_df['junction'] == jid) & (results_df['metric'] == metric)]
            if len(mdf) == 0:
                continue
            # Legjobb = scoring konzisztens a globálissal
            mdf = mdf.copy()
            mono_n = (-mdf['monotonicity_corr'] + 1) / 2
            mdf['score'] = mdf['reward_range'] * 0.25 + mdf['reward_std'] * 0.20 + (mdf['useful_pct'] / 100) * 0.25 + mono_n * 0.30
            best_row = mdf.loc[mdf['score'].idxmax()]
            best[metric] = {
                'method': best_row['method'],
                'range': best_row['reward_range'],
                'useful_pct': best_row['useful_pct'],
                'score': best_row['score'],
            }

        best_methods_per_junction[jid] = best

        w_best = best.get('TotalWaitingTime', {}).get('method', 'N/A')
        w_range = best.get('TotalWaitingTime', {}).get('range', 0)
        c_best = best.get('TotalCO2', {}).get('method', 'N/A')
        c_range = best.get('TotalCO2', {}).get('range', 0)
        print(f"{jid:15} {w_best:20} {w_range:10.4f} {c_best:20} {c_range:10.4f}")

    # Konszenzus: melyik módszer nyer a legtöbbször?
    wait_wins = {}
    co2_wins = {}
    for jid, best in best_methods_per_junction.items():
        if 'TotalWaitingTime' in best:
            m = best['TotalWaitingTime']['method']
            wait_wins[m] = wait_wins.get(m, 0) + 1
        if 'TotalCO2' in best:
            m = best['TotalCO2']['method']
            co2_wins[m] = co2_wins.get(m, 0) + 1

    print(f"\n--- KONSZENZUS (hány junction-nél nyer) ---")
    print(f"  TotalWaitingTime:")
    for m, cnt in sorted(wait_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"    {m:20} {cnt:3d}/{len(junction_ids)} junction")
    print(f"  TotalCO2:")
    for m, cnt in sorted(co2_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"    {m:20} {cnt:3d}/{len(junction_ids)} junction")

    # =====================================================================
    # Ábra 1: Összefoglaló bar chart — módszerenként (2 metrika)
    # =====================================================================
    n_methods = len([m for m in method_order if m in results_df['method'].values])
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    fig.suptitle('Normalization Methods Comparison — 16 Methods', fontsize=16)

    for metric_idx, metric in enumerate(['TotalWaitingTime', 'TotalCO2']):
        mdf = results_df[results_df['metric'] == metric]
        present_methods = [m for m in method_order if m in mdf['method'].values]

        # 1. Reward range bar chart (átlag + szórás)
        ax = axes[metric_idx, 0]
        x = np.arange(len(present_methods))
        means = [mdf[mdf['method'] == m]['reward_range'].mean() for m in present_methods]
        stds = [mdf[mdf['method'] == m]['reward_range'].std() for m in present_methods]
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color='steelblue', edgecolor='navy')
        # Legjobb zöldre
        best_idx = np.argmax(means)
        bars[best_idx].set_facecolor('#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels(present_methods, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Reward Range (p90-p10)')
        ax.set_title(f'{metric} — Reward Range')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Hasznos% bar chart
        ax = axes[metric_idx, 1]
        means = [mdf[mdf['method'] == m]['useful_pct'].mean() for m in present_methods]
        stds = [mdf[mdf['method'] == m]['useful_pct'].std() for m in present_methods]
        bars = ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color='darkorange', edgecolor='brown')
        best_idx = np.argmax(means)
        bars[best_idx].set_facecolor('#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels(present_methods, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Useful Zone % [0.15-0.85]')
        ax.set_title(f'{metric} — Sigmoid Coverage')
        ax.axhline(60, color='red', ls='--', alpha=0.5, label='Min target (60%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'normalization_methods_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  normalization_methods_comparison.png mentve")

    # =====================================================================
    # Ábra 2: Heatmap — összes módszer × összes junction
    # =====================================================================
    for metric in ['TotalWaitingTime', 'TotalCO2']:
        mdf = results_df[results_df['metric'] == metric]
        pivot = mdf.pivot_table(index='junction', columns='method', values='reward_range', aggfunc='mean')
        pivot = pivot.reindex(columns=[m for m in method_order if m in pivot.columns])

        fig, ax = plt.subplots(figsize=(max(18, len(pivot.columns) * 1.2), 10))
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f'{metric} — Reward Range per Junction × Method', fontsize=14)
        # Annotáció
        for i in range(len(pivot.index)):
            row_vals = pivot.values[i]
            best_idx = np.nanargmax(row_vals)
            for j in range(len(pivot.columns)):
                val = row_vals[j]
                if not np.isnan(val):
                    weight = 'bold' if j == best_idx else 'normal'
                    color = 'white' if val > 0.55 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.5, fontweight=weight, color=color)
        fig.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        fname = f'normalization_heatmap_{metric.lower()}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  {fname} mentve")

    # =====================================================================
    # Ábra 3: Score scatter — range vs useful% (trade-off megjelenítés)
    # =====================================================================
    fig, axes_sc = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Method Trade-off: Reward Range vs Useful Zone Coverage', fontsize=14)
    for metric_idx, metric in enumerate(['TotalWaitingTime', 'TotalCO2']):
        ax = axes_sc[metric_idx]
        mdf = results_df[results_df['metric'] == metric]
        present_methods = [m for m in method_order if m in mdf['method'].values]
        cmap = plt.cm.tab20(np.linspace(0, 1, len(present_methods)))
        for i, m in enumerate(present_methods):
            sub = mdf[mdf['method'] == m]
            ax.scatter(sub['useful_pct'].mean(), sub['reward_range'].mean(),
                       s=150, color=cmap[i], edgecolors='black', linewidths=0.8,
                       label=m, zorder=5)
            # Error bars
            ax.errorbar(sub['useful_pct'].mean(), sub['reward_range'].mean(),
                        xerr=sub['useful_pct'].std(), yerr=sub['reward_range'].std(),
                        color=cmap[i], alpha=0.4, capsize=3, zorder=4)
        ax.set_xlabel('Useful Zone % [0.15-0.85]')
        ax.set_ylabel('Reward Range (p90-p10)')
        ax.set_title(metric)
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        # Ideális sarok jelölése
        ax.annotate('IDEAL', xy=(100, 0.8), fontsize=10, color='green', alpha=0.5,
                     ha='center', va='center')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'normalization_tradeoff.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  normalization_tradeoff.png mentve")

    # =====================================================================
    # Per-junction reward eloszlás a top 5 módszerrel
    # =====================================================================
    top_n = min(5, len(sorted_methods))
    top_methods = [m for m, _ in sorted_methods[:top_n]]
    # Színek dinamikusan
    all_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    method_color_map = {m: all_colors[i % len(all_colors)] for i, m in enumerate(method_order)}

    n_cols = 5
    n_rows = (len(junction_ids) + n_cols - 1) // n_cols

    for metric_name in ['TotalWaitingTime', 'TotalCO2']:
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes_grid = axes_grid.reshape(1, -1)
        fig.suptitle(f'Top-{top_n} Methods: {metric_name} Reward Distribution per Junction', fontsize=14)

        for idx, jid in enumerate(junction_ids):
            row, col = idx // n_cols, idx % n_cols
            ax = axes_grid[row, col]

            jdf = full_df[full_df['junction'] == jid]
            jdf_valid = jdf[jdf['VehCount'] > 0]
            raw_vals = jdf_valid[metric_name].values

            plotted = False
            for method_name, method_fn in methods:
                if method_name not in top_methods:
                    continue
                rewards, _, _ = method_fn(raw_vals)
                if rewards is None:
                    continue
                ax.hist(rewards, bins=30, alpha=0.35, density=True,
                        label=method_name, color=method_color_map.get(method_name, 'gray'))
                plotted = True

            if plotted:
                ax.axvline(0.15, color='black', ls=':', lw=0.8, alpha=0.5)
                ax.axvline(0.85, color='black', ls=':', lw=0.8, alpha=0.5)
                ax.set_xlim(0, 1)
                ax.set_title(jid, fontsize=9)
                if idx == 0:
                    ax.legend(fontsize=5)

        for idx in range(len(junction_ids), n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes_grid[row, col].set_visible(False)

        plt.tight_layout()
        fname = f'normalization_top{top_n}_{metric_name.lower()}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  {fname} mentve")

    # Végső ajánlás
    best_method = sorted_methods[0][0]
    print(f"\n{'=' * 80}")
    print(f"  AJANLAS")
    print(f"{'=' * 80}")
    print(f"  Globálisan legjobb módszer: {best_method} (score={sorted_methods[0][1]:.4f})")
    print(f"  Második legjobb:            {sorted_methods[1][0]} (score={sorted_methods[1][1]:.4f})")
    if sorted_methods[0][1] - sorted_methods[1][1] < 0.01:
        print(f"  MEGJEGYZÉS: Az első két módszer nagyon közel van — mindkettő használható.")
    print(f"\n  A junction_reward_params.json-ban a '{best_method}' módszer paramétereit érdemes használni.")


def analyze_metric_selection(output_dir):
    """
    Melyik metrikák a legjobbak a reward-hoz?

    1. Per-junction PCA: melyik metrikák hordozzák a legtöbb információt
    2. Per-junction korreláció mátrix: redundancia vizsgálat
    3. Metrika-kombináció teszt: különböző reward-képletek összehasonlítása
       (TotalWait+CO2, AvgWait+CO2, Wait+Queue, Wait+Speed, stb.)

    Cél: megtalálni azt a 2-3 metrikát, ami a legjobb reward jelet adja.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n" + "=" * 80)
    print("  METRIKA SZELEKCIÓ — MELYIK METRIKÁK A LEGJOBBAK A REWARD-HOZ?")
    print("=" * 80)

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

    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime',
                   'TotalCO2', 'AvgCO2', 'VehCount',
                   'AvgSpeed', 'AvgOccupancy', 'QueueLength']

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def log_sigmoid_reward(vals):
        """Log-sigmoid reward egy metrikára (lokális params)."""
        vals_pos = vals[vals > 0]
        if len(vals_pos) < 10:
            return None
        log_v = np.log(vals_pos + 1e-5)
        mu, std = np.mean(log_v), np.std(log_v)
        return 1.0 - sigmoid((log_v - mu) / (std + 1e-9))

    # =====================================================================
    # 1. PER-JUNCTION PCA
    # =====================================================================
    print(f"\n--- 1. PER-JUNCTION PCA (PC1+PC2 variancia, top loadings) ---\n")

    epsilon = 1e-5
    pca_results = []

    print(f"{'Junction':15} {'PC1%':>8} {'PC2%':>8} {'PC1+2%':>8} {'PC1 top-3 loadings':>50}")
    print("-" * 95)

    for jid in junction_ids:
        jdf = full_df[(full_df['junction'] == jid) & (full_df['VehCount'] > 0)].copy()
        if len(jdf) < 50:
            continue

        available_cols = [c for c in metric_cols if c in jdf.columns]
        df_log = np.log(jdf[available_cols].clip(lower=epsilon) + epsilon)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_log)

        pca = PCA()
        pca.fit(data_scaled)
        ev = pca.explained_variance_ratio_ * 100

        # Top-3 loading a PC1-en
        loadings_pc1 = pca.components_[0]
        top3_idx = np.argsort(np.abs(loadings_pc1))[-3:][::-1]
        top3_str = ", ".join([f"{available_cols[i]}({loadings_pc1[i]:+.3f})" for i in top3_idx])

        print(f"{jid:15} {ev[0]:8.1f} {ev[1]:8.1f} {ev[0]+ev[1]:8.1f} {top3_str:>50}")

        pca_results.append({
            'junction': jid,
            'pc1_var': ev[0], 'pc2_var': ev[1],
            'pc1_top1': available_cols[top3_idx[0]],
            'pc1_top2': available_cols[top3_idx[1]],
            'pc1_top3': available_cols[top3_idx[2]],
        })

    # =====================================================================
    # 2. KORRELÁCIÓS MÁTRIX — GLOBÁLIS + PER-JUNCTION ÁTLAG
    # =====================================================================
    print(f"\n--- 2. KORRELÁCIÓS MÁTRIX (globális, log-transzformált) ---\n")

    df_all_valid = full_df[full_df['VehCount'] > 0].copy()
    available_cols = [c for c in metric_cols if c in df_all_valid.columns]
    df_log_all = np.log(df_all_valid[available_cols].clip(lower=epsilon) + epsilon)
    corr_global = df_log_all.corr()

    # A reward szempontjából fontos: melyek korrelálnak erősen egymással?
    # Ha két metrika korr > 0.9, redundánsak
    print(f"  Erős korrelációk (|r| > 0.85):")
    pairs_shown = set()
    for i, c1 in enumerate(available_cols):
        for j, c2 in enumerate(available_cols):
            if i >= j:
                continue
            r = corr_global.loc[c1, c2]
            if abs(r) > 0.85:
                pair = tuple(sorted([c1, c2]))
                if pair not in pairs_shown:
                    pairs_shown.add(pair)
                    print(f"    {c1:25} ↔ {c2:25} r={r:+.3f} {'(REDUNDÁNS)' if abs(r) > 0.95 else ''}")

    # Ábra: globális korreláció mátrix
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_global, dtype=bool), k=1)
    sns.heatmap(corr_global, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax,
                square=True, vmin=-1, vmax=1, mask=mask,
                annot_kws={'fontsize': 7})
    ax.set_title('Global Metric Correlation Matrix (log-transformed, per-junction data)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metric_correlation_global.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  metric_correlation_global.png mentve")

    # =====================================================================
    # 3. METRIKA-KOMBINÁCIÓ TESZT
    # =====================================================================
    print(f"\n--- 3. METRIKA-KOMBINÁCIÓ TESZT ---")
    print(f"  Minden kombináció log-sigmoid normalizációval, lokális MU/STD-vel")
    print(f"  Értékelés: reward range (p90-p10), std, hasznos zóna%\n")

    # Reward-hoz használható metrikák (minimize = alacsonyabb jobb)
    # AvgSpeed: maximize (magasabb jobb → reward = sigmoid, nem 1-sigmoid)
    minimize_metrics = ['TotalWaitingTime', 'AvgWaitingTime', 'TotalCO2', 'AvgCO2',
                        'TotalTravelTime', 'QueueLength', 'AvgOccupancy']
    maximize_metrics = ['AvgSpeed']

    # Metrika-kombinációk definíciója
    combinations = [
        # Jelenlegi
        ('TotalWait + TotalCO2 (jelenlegi)', ['TotalWaitingTime', 'TotalCO2'], [0.5, 0.5]),
        # Avg variánsok
        ('AvgWait + AvgCO2', ['AvgWaitingTime', 'AvgCO2'], [0.5, 0.5]),
        ('AvgWait + TotalCO2', ['AvgWaitingTime', 'TotalCO2'], [0.5, 0.5]),
        ('TotalWait + AvgCO2', ['TotalWaitingTime', 'AvgCO2'], [0.5, 0.5]),
        # Queue bevonása
        ('TotalWait + Queue', ['TotalWaitingTime', 'QueueLength'], [0.5, 0.5]),
        ('TotalWait + CO2 + Queue', ['TotalWaitingTime', 'TotalCO2', 'QueueLength'], [0.4, 0.3, 0.3]),
        # Speed bevonása
        ('TotalWait + Speed', ['TotalWaitingTime', 'AvgSpeed'], [0.5, 0.5]),
        ('TotalWait + CO2 + Speed', ['TotalWaitingTime', 'TotalCO2', 'AvgSpeed'], [0.4, 0.3, 0.3]),
        # Travel time
        ('TotalTT + TotalCO2', ['TotalTravelTime', 'TotalCO2'], [0.5, 0.5]),
        ('TotalWait + TotalTT', ['TotalWaitingTime', 'TotalTravelTime'], [0.5, 0.5]),
        # Occupancy
        ('TotalWait + Occupancy', ['TotalWaitingTime', 'AvgOccupancy'], [0.5, 0.5]),
        # Egyes metrikák önállóan
        ('TotalWait ONLY', ['TotalWaitingTime'], [1.0]),
        ('TotalCO2 ONLY', ['TotalCO2'], [1.0]),
        ('Queue ONLY', ['QueueLength'], [1.0]),
        ('AvgSpeed ONLY', ['AvgSpeed'], [1.0]),
        # Szélesebb kombinációk
        ('Wait + CO2 + Queue + Speed', ['TotalWaitingTime', 'TotalCO2', 'QueueLength', 'AvgSpeed'], [0.3, 0.25, 0.25, 0.2]),
    ]

    combo_results = []

    for combo_name, metric_list, weights in combinations:
        combo_scores = []

        for jid in junction_ids:
            jdf = full_df[(full_df['junction'] == jid) & (full_df['VehCount'] > 0)].copy()
            if len(jdf) < 50:
                continue

            # Check all metrics available
            if not all(m in jdf.columns for m in metric_list):
                continue

            # Per-metric reward számítás
            metric_rewards = []
            valid_mask = np.ones(len(jdf), dtype=bool)

            for m in metric_list:
                vals = jdf[m].values
                pos_mask = vals > 0
                valid_mask &= pos_mask

            if valid_mask.sum() < 50:
                continue

            combined_reward = np.zeros(valid_mask.sum())

            for m, w in zip(metric_list, weights):
                vals = jdf[m].values[valid_mask]
                log_v = np.log(vals + 1e-5)
                mu, std = np.mean(log_v), np.std(log_v)
                z = (log_v - mu) / (std + 1e-9)

                if m in maximize_metrics:
                    # Magasabb jobb → reward = sigmoid(z)
                    r = sigmoid(z)
                else:
                    # Alacsonyabb jobb → reward = 1 - sigmoid(z)
                    r = 1.0 - sigmoid(z)

                combined_reward += w * r

            # --- Metrikák ---
            r_range = float(np.percentile(combined_reward, 90) - np.percentile(combined_reward, 10))
            r_std = float(np.std(combined_reward))
            r_mean = float(np.mean(combined_reward))
            useful_pct = float(np.mean((combined_reward >= 0.15) & (combined_reward <= 0.85)) * 100)

            # --- Monotonitás: a reward csökken-e a forgalom növekedésével? ---
            # Ez a fizikai konzisztencia teszt: nagyobb forgalom → rosszabb → alacsonyabb reward
            flow_vals = jdf['flow_level'].values[valid_mask]
            unique_flows = sorted(set(flow_vals))
            if len(unique_flows) >= 3:
                flow_means = [np.mean(combined_reward[flow_vals == fl]) for fl in unique_flows]
                # Spearman korreláció: flow ↑ → reward ↓ → negatív korreláció = jó
                from scipy.stats import spearmanr
                mono_corr, _ = spearmanr(unique_flows, flow_means)
                # Monoton decrease check: hány egymást követő pár csökken?
                n_decreasing = sum(1 for i in range(len(flow_means)-1) if flow_means[i] > flow_means[i+1])
                mono_pct = n_decreasing / (len(flow_means) - 1) * 100
            else:
                mono_corr = 0.0
                mono_pct = 0.0

            combo_scores.append({
                'junction': jid,
                'reward_range': r_range,
                'reward_std': r_std,
                'reward_mean': r_mean,
                'useful_pct': useful_pct,
                'monotonicity_corr': mono_corr,   # Spearman: -1 = tökéletes csökkenés
                'monotonicity_pct': mono_pct,      # % csökkenő lépések
            })

        if combo_scores:
            cdf = pd.DataFrame(combo_scores)
            combo_results.append({
                'combination': combo_name,
                'metrics': '+'.join(metric_list),
                'weights': str(weights),
                'avg_range': cdf['reward_range'].mean(),
                'avg_std': cdf['reward_std'].mean(),
                'avg_useful_pct': cdf['useful_pct'].mean(),
                'min_useful_pct': cdf['useful_pct'].min(),
                'avg_mean': cdf['reward_mean'].mean(),
                'avg_mono_corr': cdf['monotonicity_corr'].mean(),
                'avg_mono_pct': cdf['monotonicity_pct'].mean(),
                'n_junctions': len(cdf),
            })

    # Eredmények kiírása
    if combo_results:
        combo_df = pd.DataFrame(combo_results)

        # =====================================================================
        # SCORING — az RL szempontból fontos szempontok:
        #
        #   1. DIFFERENCIÁLHATÓSÁG (25%): reward_range — az ágens képes-e
        #      megkülönböztetni a jó/rossz akciókat? Nagyobb range = több tér.
        #
        #   2. JEL ERŐSSÉG (20%): reward_std — mennyire szóródik a reward?
        #      Ha kicsi, minden "egyformának" tűnik az ágensnek.
        #
        #   3. SIGMOID COVERAGE (25%): useful_pct [0.15-0.85] — a saturált
        #      zónákban a gradiens ≈ 0, az ágens „vak". 100% = nincs vak zóna.
        #
        #   4. MONOTONITÁS (30%): mono_corr — a reward fizikailag konzisztens-e?
        #      Nagyobb forgalom → rosszabb → alacsonyabb reward. Ez a LEGFONTOSABB
        #      mert ha nem monoton, az ágens "jutalmat kap" a rossz helyzetért.
        #      Spearman korreláció: -1 = tökéletes, 0 = nincs összefüggés.
        #      Normalizáljuk [0,1]-re: score = (-mono_corr + 1) / 2
        # =====================================================================
        mono_score = (-combo_df['avg_mono_corr'] + 1) / 2  # [-1,+1] → [0,1]
        combo_df['score'] = (
            combo_df['avg_range'] * 0.25 +
            combo_df['avg_std'] * 0.20 +
            (combo_df['avg_useful_pct'] / 100) * 0.25 +
            mono_score * 0.30
        )
        combo_df = combo_df.sort_values('score', ascending=False).reset_index(drop=True)
        combo_df.to_csv(os.path.join(output_dir, 'metric_combinations_comparison.csv'), index=False)

        print(f"\n  SCORING: 25% range + 20% std + 25% useful% + 30% monotonitás")
        print(f"  Monotonitás = Spearman(flow_level, mean_reward): -1 = tökéletes csökkenés\n")

        print(f"{'Rank':>4} {'Kombináció':40} {'R_range':>8} {'R_std':>8} {'Use%':>6} "
              f"{'Mono_r':>8} {'Mono%':>7} {'Score':>7}")
        print("-" * 100)

        for idx, row in combo_df.iterrows():
            marker = " <<<" if idx == 0 else ""
            print(f"{idx+1:4d} {row['combination']:40} {row['avg_range']:8.4f} {row['avg_std']:8.4f} "
                  f"{row['avg_useful_pct']:6.1f} {row['avg_mono_corr']:+8.3f} "
                  f"{row['avg_mono_pct']:7.1f} {row['score']:7.4f}{marker}")

        # Ábra: kombináció összehasonlítás
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Metric Combination Comparison for Reward Function', fontsize=15)

        n_combos = len(combo_df)
        x = np.arange(n_combos)
        labels = [row['combination'] for _, row in combo_df.iterrows()]

        # Score bar chart
        ax = axes[0]
        colors_bar = ['#2ca02c' if i == 0 else '#4e79a7' for i in range(n_combos)]
        ax.barh(x, combo_df['score'].values, color=colors_bar, alpha=0.8, edgecolor='navy')
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Combined Score')
        ax.set_title('Overall Score (25%range + 20%std + 25%useful + 30%monotonicity)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Range bar chart
        ax = axes[1]
        ax.barh(x, combo_df['avg_range'].values, color='steelblue', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Avg Reward Range (p90-p10)')
        ax.set_title('Reward Differentiability')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Useful% bar chart
        ax = axes[2]
        ax.barh(x, combo_df['avg_useful_pct'].values, color='darkorange', alpha=0.8)
        ax.axvline(60, color='red', ls='--', lw=1, label='Min target')
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Avg Useful Zone %')
        ax.set_title('Sigmoid Coverage [0.15-0.85]')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'metric_combinations_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  metric_combinations_comparison.png mentve")

        # Trade-off scatter
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(combo_df['avg_useful_pct'], combo_df['avg_range'],
                            s=combo_df['score'] * 500, c=combo_df['score'],
                            cmap='RdYlGn', edgecolors='black', linewidths=0.8, zorder=5)
        for idx, row in combo_df.iterrows():
            ax.annotate(row['combination'], (row['avg_useful_pct'], row['avg_range']),
                       fontsize=6, ha='center', va='bottom',
                       xytext=(0, 8), textcoords='offset points')
        ax.set_xlabel('Useful Zone % [0.15-0.85]', fontsize=12)
        ax.set_ylabel('Avg Reward Range (p90-p10)', fontsize=12)
        ax.set_title('Metric Combination Trade-off\n(bubble size = overall score)', fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.colorbar(scatter, label='Score')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'metric_combinations_tradeoff.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  metric_combinations_tradeoff.png mentve")

        # Végső ajánlás
        best = combo_df.iloc[0]
        second = combo_df.iloc[1]
        current = combo_df[combo_df['combination'].str.contains('jelenlegi')]

        print(f"\n{'=' * 80}")
        print(f"  METRIKA SZELEKCIO AJANLAS")
        print(f"{'=' * 80}")
        print(f"  Legjobb kombináció:   {best['combination']}")
        print(f"    Range: {best['avg_range']:.4f}, Useful%: {best['avg_useful_pct']:.1f}%, Score: {best['score']:.4f}")
        print(f"  Második legjobb:      {second['combination']}")
        print(f"    Range: {second['avg_range']:.4f}, Useful%: {second['avg_useful_pct']:.1f}%, Score: {second['score']:.4f}")
        if len(current) > 0:
            cur = current.iloc[0]
            cur_rank = combo_df.index[combo_df['combination'] == cur['combination']].tolist()[0] + 1
            print(f"  Jelenlegi módszer:    {cur['combination']} (#{cur_rank})")
            print(f"    Range: {cur['avg_range']:.4f}, Useful%: {cur['avg_useful_pct']:.1f}%, Score: {cur['score']:.4f}")
            if cur_rank > 1:
                improvement = (best['score'] - cur['score']) / cur['score'] * 100
                print(f"  Lehetséges javulás:   {improvement:+.1f}% a legjobb kombóval")

    # =====================================================================
    # 4. PCA LOADING ÖSSZEFOGLALÓ — mely metrikák a legfontosabbak?
    # =====================================================================
    if pca_results:
        pca_df = pd.DataFrame(pca_results)
        print(f"\n--- 4. PCA PC1 LEGGYAKORIBB TOP LOADING METRIKÁK ---")
        from collections import Counter
        top1_counts = Counter(pca_df['pc1_top1'].values)
        print(f"  Mely metrika a PC1 #1 loading a legtöbb junction-nél?")
        for m, cnt in top1_counts.most_common():
            print(f"    {m:25} {cnt:3d}/{len(pca_df)} junction ({cnt/len(pca_df)*100:.0f}%)")

        # Top-3 összesítés
        all_tops = list(pca_df['pc1_top1'].values) + list(pca_df['pc1_top2'].values) + list(pca_df['pc1_top3'].values)
        top_all_counts = Counter(all_tops)
        print(f"\n  Mely metrika jelenik meg a PC1 top-3-ban?")
        for m, cnt in top_all_counts.most_common():
            print(f"    {m:25} {cnt:3d}/{len(pca_df)*3} hely ({cnt/(len(pca_df)*3)*100:.0f}%)")


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

    # Elemzés
    analyze_per_junction(OUTPUT_DIR)

    # Illeszkedés-ellenőrzés
    check_fit_quality(OUTPUT_DIR)

    # Normalizációs módszerek összehasonlítása
    compare_normalization_methods(OUTPUT_DIR)

    # Metrika szelekció — melyik metrikák a legjobbak a reward-hoz?
    analyze_metric_selection(OUTPUT_DIR)


if __name__ == "__main__":
    main()
