#!/usr/bin/env python3
"""
Metrika gyűjtés különböző forgalmi szinteken - PCA elemzéshez.

Több forgalmi szintet futtat (flow max: 300, 400, 500, 600, 700, 800),
és minden lépésben junction-önként gyűjti a következő metrikákat:
  - TotalTravelTime: összes travel time a bejövő sávokon
  - AvgTravelTime: átlag travel time / jármű
  - TotalWaitingTime: összes várakozási idő a bejövő sávokon
  - AvgWaitingTime: átlag várakozási idő / jármű
  - TotalCO2: összes CO2 kibocsátás
  - AvgCO2: átlag CO2 / jármű
  - VehCount: járművek száma a bejövő sávokon
  - AvgSpeed: átlagsebesség a bejövő sávokon
  - AvgOccupancy: átlagos detektor foglaltság
  - QueueLength: a junction-re várakozó járművek (halted)

Kimenet: CSV fájlok junction-önként és metrikánként, + episode_summary.csv
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
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
NET_FILE = os.path.join(DATA_DIR, "mega_catalogue_v2.net.xml")
LOGIC_FILE = os.path.join(DATA_DIR, "traffic_lights.json")
DETECTOR_FILE = os.path.join(DATA_DIR, "detectors.add.xml")

# --- Config ---
FLOW_MIN = 100
FLOW_MAX_LEVELS = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
DURATION = 3600       # 1 óra szimuláció
DELTA_TIME = 5        # lépésméret (sec)
WARMUP = 100          # warmup (sec)
EPISODES_PER_LEVEL = 3  # ismétlések forgalmi szintenként
MAX_REALISTIC_TT = 1000  # travel time szűrő


def run_simulation(flow_max, episode_idx, output_dir):
    """Egy szimulációs epizód futtatása és metrika gyűjtés."""

    # Lazy import - macOS traci compatibility
    import traci
    import sumolib

    route_file = os.path.join(SCRIPT_DIR, f"_metric_test_{flow_max}_{episode_idx}.rou.xml")

    # --- Forgalom generálás (lane-szintű, az env logikája alapján) ---
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

    # --- Junction → bejövő lane-ek és detektorok feltérképezése ---
    junction_lanes = {}   # {jid: [lane_id, ...]}
    junction_dets = {}    # {jid: [det_id, ...]}

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

    # --- Metrikák inicializálása ---
    # Per junction, per step: gyűjtünk egy dict-et
    metrics = {jid: [] for jid in junction_ids}

    # --- Warmup ---
    for _ in range(WARMUP):
        traci.simulationStep()

    # --- Fő ciklus (random jelzőlámpa akciók) ---
    step = 0
    total_steps = (DURATION - WARMUP) // DELTA_TIME

    # Jelzőlámpa fázis infó
    junction_phases = {}
    for jid in junction_ids:
        programs = traci.trafficlight.getAllProgramLogics(jid)
        if programs:
            num_phases = len(programs[0].phases)
            junction_phases[jid] = num_phases
        else:
            junction_phases[jid] = 1

    for step_i in range(total_steps):
        # Random jelzőlámpa akció
        for jid in junction_ids:
            phase = random.randint(0, junction_phases[jid] - 1)
            traci.trafficlight.setPhase(jid, phase)

        # Delta time lépések
        for _ in range(DELTA_TIME):
            traci.simulationStep()

        # Metrika gyűjtés junction-önként
        for jid in junction_ids:
            lanes = junction_lanes[jid]
            dets = junction_dets[jid]

            total_tt = 0.0          # Limitált (< MAX_REALISTIC_TT)
            total_tt_raw = 0.0      # Nem limitált (minden SUMO érték)
            total_waiting = 0.0
            total_co2 = 0.0
            total_veh = 0
            total_speed = 0.0
            total_halted = 0
            valid_tt_lanes = 0
            valid_tt_raw_lanes = 0
            valid_speed_lanes = 0

            for lane in lanes:
                # Travel time - limitált
                tt = traci.lane.getTraveltime(lane)
                if 0 < tt < MAX_REALISTIC_TT:
                    total_tt += tt
                    valid_tt_lanes += 1
                # Travel time - raw (minden pozitív érték, sentinel-ekkel együtt)
                if tt > 0:
                    total_tt_raw += tt
                    valid_tt_raw_lanes += 1

                # Waiting time
                total_waiting += traci.lane.getWaitingTime(lane)

                # CO2
                total_co2 += traci.lane.getCO2Emission(lane)

                # Vehicle count
                veh_count = traci.lane.getLastStepVehicleNumber(lane)
                total_veh += veh_count

                # Speed
                speed = traci.lane.getLastStepMeanSpeed(lane)
                if speed >= 0:
                    total_speed += speed
                    valid_speed_lanes += 1

                # Halted (queue)
                total_halted += traci.lane.getLastStepHaltingNumber(lane)

            # Detektor occupancy
            total_occ = 0.0
            for det in dets:
                total_occ += traci.inductionloop.getLastStepOccupancy(det)
            avg_occ = total_occ / len(dets) if dets else 0.0

            # Derived metrics
            avg_tt = total_tt / valid_tt_lanes if valid_tt_lanes > 0 else 0.0
            avg_tt_raw = total_tt_raw / valid_tt_raw_lanes if valid_tt_raw_lanes > 0 else 0.0
            avg_waiting = total_waiting / total_veh if total_veh > 0 else 0.0
            avg_co2 = total_co2 / total_veh if total_veh > 0 else 0.0
            avg_speed = total_speed / valid_speed_lanes if valid_speed_lanes > 0 else 0.0

            metrics[jid].append({
                'step': step_i,
                'TotalTravelTime': total_tt,
                'AvgTravelTime': avg_tt,
                'TotalTravelTime_Raw': total_tt_raw,
                'AvgTravelTime_Raw': avg_tt_raw,
                'TotalWaitingTime': total_waiting,
                'AvgWaitingTime': avg_waiting,
                'TotalCO2': total_co2,
                'AvgCO2': avg_co2,
                'VehCount': total_veh,
                'AvgSpeed': avg_speed,
                'AvgOccupancy': avg_occ,
                'QueueLength': total_halted,
            })

        if (step_i + 1) % 50 == 0:
            print(f"    Step {step_i+1}/{total_steps}")

    traci.close()

    # Cleanup route file
    if os.path.exists(route_file):
        os.remove(route_file)

    # --- CSV mentés ---
    for jid in junction_ids:
        df = pd.DataFrame(metrics[jid])
        csv_path = os.path.join(output_dir, f"{jid}_flow{flow_max}_ep{episode_idx}.csv")
        df.to_csv(csv_path, index=False)

    return metrics


def run_pca_analysis(output_dir):
    """PCA elemzés az összes összegyűjtött adaton."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n" + "=" * 60)
    print("PCA ELEMZES")
    print("=" * 60)

    # Összes CSV betöltése
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and '_flow' in f]

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv_file))
        # Flow level kinyerése a fájlnévből
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        df['junction'] = csv_file.split('_flow')[0]
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Osszes adatpont: {len(full_df)}")

    # Metrika oszlopok
    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime',
                   'TotalCO2', 'AvgCO2', 'VehCount',
                   'AvgSpeed', 'AvgOccupancy', 'QueueLength']

    # Szűrés: csak pozitív sorok (ahol van forgalom)
    mask = full_df['VehCount'] > 0
    df_valid = full_df[mask].copy()
    print(f"Ervenyes adatpont (VehCount > 0): {len(df_valid)}")

    # --- 1. Korrelációs mátrix ---
    corr = df_valid[metric_cols].corr()

    # --- 2. Log transzformáció + PCA ---
    epsilon = 1e-5
    df_log = np.log(df_valid[metric_cols].clip(lower=epsilon) + epsilon)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_log)

    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    explained_var = pca.explained_variance_ratio_ * 100
    components = pca.components_

    # --- 3. Közös adatok a plotokhoz ---
    sns.set(style='whitegrid')
    from matplotlib.patches import FancyArrowPatch

    flow_levels = df_valid['flow_level'].values
    unique_flows = sorted(df_valid['flow_level'].unique())
    cmap_flows = plt.cm.viridis(np.linspace(0, 1, len(unique_flows)))
    flow_color_map = {fl: cmap_flows[i] for i, fl in enumerate(unique_flows)}

    # Ritkítás scatter-hez
    max_points = 50000
    if len(pca_result) > max_points:
        idx = np.random.choice(len(pca_result), max_points, replace=False)
    else:
        idx = np.arange(len(pca_result))

    # --- Helper: biplot rajzoló ---
    def draw_biplot(ax, show_legend=True, fontsize_labels=7, fontsize_legend=7):
        for fl in unique_flows:
            fl_mask = flow_levels[idx] == fl
            ax.scatter(pca_result[idx[fl_mask], 0], pca_result[idx[fl_mask], 1],
                       alpha=0.12, s=6, color=flow_color_map[fl], label=f'{fl}/h')

        scaling = np.max(np.abs(pca_result[idx])) * 0.65

        # Kézi label offset-ek a sűrű területekhez (feat_name → (dx, dy) relatív eltolás)
        label_offsets = {
            'TotalTravelTime':     (-3.0, 1.0),
            'AvgTravelTime':       (3.0, 1.0),
            'TotalTravelTime_Raw': (-3.0, -0.5),
            'AvgTravelTime_Raw':   (3.5, -0.8),
            'TotalCO2':            (-2.0, 2.0),
            'VehCount':            (2.0, 2.0),
            'AvgOccupancy':        (0, -1.8),
            'QueueLength':         (2.0, -1.5),
            'TotalWaitingTime':    (2.0, -1.0),
            'AvgWaitingTime':      (2.0, 1.2),
            'AvgSpeed':            (-1.5, 0),
        }

        for i, feat in enumerate(metric_cols):
            x_vec = components[0, i] * scaling
            y_vec = components[1, i] * scaling
            arrow = FancyArrowPatch((0, 0), (x_vec, y_vec),
                                    arrowstyle='->', mutation_scale=12,
                                    color='#cc0000', lw=1.5, zorder=10)
            ax.add_patch(arrow)
            # Label pozíció: nyíl vége + manuális offset ha van
            dx, dy = label_offsets.get(feat, (0, 0))
            tx = x_vec * 1.15 + dx
            ty = y_vec * 1.15 + dy
            ax.annotate(feat, xy=(x_vec, y_vec), xytext=(tx, ty),
                        fontsize=fontsize_labels, weight='bold', color='black',
                        ha='center', va='center', zorder=11,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9, lw=0.5),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

        ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}%) — Congestion', fontsize=10)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}%) — Travel Time', fontsize=10)
        ax.axhline(0, color='black', lw=0.4, ls='--', alpha=0.5)
        ax.axvline(0, color='black', lw=0.4, ls='--', alpha=0.5)
        if show_legend:
            ax.legend(fontsize=fontsize_legend, loc='upper right', markerscale=2.5,
                      framealpha=0.95, title='Flow max', title_fontsize=fontsize_legend)

    # --- Helper: scree plot ---
    def draw_scree(ax):
        ax.bar(range(1, len(explained_var)+1), explained_var, alpha=0.7,
               align='center', color='steelblue', label='Individual')
        ax.step(range(1, len(explained_var)+1), np.cumsum(explained_var),
                where='mid', color='red', lw=2, label='Cumulative')
        ax.set_ylabel('Explained Variance (%)')
        ax.set_xlabel('Principal Components')
        ax.set_title(f'PCA Scree Plot\nPC1+PC2 = {explained_var[0]+explained_var[1]:.1f}%')
        ax.legend()
        ax.set_ylim(0, 105)

    # --- Helper: korrelációs mátrix ---
    def draw_corr(ax, fontsize_annot=6):
        sns.heatmap(corr, annot=False, cmap='RdYlBu_r', ax=ax,
                    square=True, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix (log-transformed)')
        # Kézi annotáció kontrasztos szövegszínnel
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                # Fehér szöveg ha a háttér sötét (erős pozitív/negatív korreláció)
                text_color = 'white' if abs(val) > 0.55 else 'black'
                ax.text(j + 0.5, i + 0.5, f'{val:.2f}',
                        ha='center', va='center', fontsize=fontsize_annot,
                        color=text_color, fontweight='bold')

    # --- Helper: boxplot ---
    def draw_boxplot(ax):
        box_data = []
        for fl in unique_flows:
            vals = df_valid[df_valid['flow_level'] == fl]['AvgWaitingTime'].values
            box_data.append(vals)
        ax.boxplot(box_data, tick_labels=[str(fl) for fl in unique_flows], showfliers=False)
        ax.set_xlabel('Flow Max (veh/h/lane)')
        ax.set_ylabel('AvgWaitingTime (s)')
        ax.set_title('Average Waiting Time by Traffic Level')

    # ===== 1. Összetett 2×2 kép =====
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    draw_scree(axes[0, 0])
    draw_corr(axes[0, 1])
    draw_biplot(axes[1, 0], fontsize_labels=6.5, fontsize_legend=6.5)
    axes[1, 0].set_title('PCA Biplot (color = traffic flow level)')
    draw_boxplot(axes[1, 1])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=200)
    plt.close()
    print(f"  Mentve: pca_analysis.png (osszetett)")

    # ===== 2. Külön képek nagy felbontásban =====
    # A) Scree plot
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_scree(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_01_scree.png'), dpi=300)
    plt.close()
    print(f"  Mentve: pca_01_scree.png")

    # B) Korrelációs mátrix
    fig, ax = plt.subplots(figsize=(14, 12))
    draw_corr(ax, fontsize_annot=9)
    ax.set_title('Metric Correlation Matrix (log-transformed)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_02_correlation.png'), dpi=300)
    plt.close()
    print(f"  Mentve: pca_02_correlation.png")

    # C) Biplot - nagy felbontás
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_biplot(ax, fontsize_labels=9, fontsize_legend=9)
    ax.set_title('PCA Biplot — Metric Loadings by Traffic Flow Level', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_03_biplot.png'), dpi=300)
    plt.close()
    print(f"  Mentve: pca_03_biplot.png")

    # D) Boxplot
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_boxplot(ax)
    ax.set_title('Average Waiting Time Distribution by Traffic Level', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pca_04_waitingtime_boxplot.png'), dpi=300)
    plt.close()
    print(f"  Mentve: pca_04_waitingtime_boxplot.png")

    # --- 4. Loadings tábla ---
    print("\n--- LOADINGS (PC1, PC2, PC3) ---")
    loadings_df = pd.DataFrame(
        components[:3].T,
        columns=['PC1', 'PC2', 'PC3'],
        index=metric_cols
    )
    print(loadings_df.round(4))

    print(f"\n--- EXPLAINED VARIANCE ---")
    for i, var in enumerate(explained_var[:5]):
        print(f"  PC{i+1}: {var:.2f}%  (cumulative: {sum(explained_var[:i+1]):.2f}%)")

    # --- 5. Normalizációs paraméterek minden metrikára ---
    print("\n--- NORMALIZACIOS PARAMETEREK (log-sigmoid) ---")
    for col in metric_cols:
        vals = df_valid[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            log_vals = np.log(vals + 1e-5)
            mu = np.mean(log_vals)
            std = np.std(log_vals)
            print(f"  {col:<20} MU={mu:10.4f}  STD={std:8.4f}  (raw: median={np.median(vals):.2f}, p5={np.percentile(vals,5):.2f}, p95={np.percentile(vals,95):.2f})")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "metric_pca_test_v2")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  METRIKA GYUJTES - PCA ELEMZESHEZ")
    print("=" * 60)
    print(f"  Flow szintek (max): {FLOW_MAX_LEVELS}")
    print(f"  Epizodok szintenkent: {EPISODES_PER_LEVEL}")
    print(f"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s")
    print("=" * 60)

    summary_rows = []

    for flow_max in FLOW_MAX_LEVELS:
        for ep in range(EPISODES_PER_LEVEL):
            print(f"\n--- Flow max: {flow_max}/h | Epizod {ep+1}/{EPISODES_PER_LEVEL} ---")
            metrics = run_simulation(flow_max, ep, output_dir)

            # Összefoglaló az epizódhoz
            for jid, data in metrics.items():
                if data:
                    df = pd.DataFrame(data)
                    summary_rows.append({
                        'flow_max': flow_max,
                        'episode': ep,
                        'junction': jid,
                        'mean_AvgTravelTime': df['AvgTravelTime'].mean(),
                        'mean_AvgWaitingTime': df['AvgWaitingTime'].mean(),
                        'mean_AvgCO2': df['AvgCO2'].mean(),
                        'mean_VehCount': df['VehCount'].mean(),
                        'mean_AvgSpeed': df['AvgSpeed'].mean(),
                        'mean_QueueLength': df['QueueLength'].mean(),
                    })

    # Episode summary mentés
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'episode_summary.csv'), index=False)
    print(f"\nEpisode summary mentve: {os.path.join(output_dir, 'episode_summary.csv')}")

    # PCA elemzés
    run_pca_analysis(output_dir)


if __name__ == "__main__":
    main()
