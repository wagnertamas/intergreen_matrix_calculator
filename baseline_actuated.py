#!/usr/bin/env python3
"""
Baseline értékelés — 4 stratégia összehasonlítása az RL reward függvénnyel.

Stratégiák:
  1. static       — SUMO beépített fix ciklus (.net.xml programja)
  2. actuated     — SUMO actuated vezérlés (detektor-alapú, fix ciklussorrend)
  3. random       — random fázisválasztás (same action space as RL)
  4. max_pressure — klasszikus heurisztika: legtöbb várakozó → zöld

A random és max_pressure PONTOSAN ugyanazt az environment-et használja,
mint az RL agent (SumoRLEnvironment), tehát a transition logic, intergreen
matrix, reward számítás mind azonos → fair összehasonlítás.

Használat:
    python baseline_actuated.py --episodes 10
    python baseline_actuated.py --episodes 20 --wandb-project sumo-rl-single-2
    python baseline_actuated.py --episodes 10 --strategies random,max_pressure
    python baseline_actuated.py --episodes 50 --flow-min 200 --flow-max 800

Kimenet:
    - eval_results/baseline_summary.csv         — epizódonkénti eredmények
    - eval_results/baseline_comparison.png       — box plot összehasonlítás
    - eval_results/baseline_per_junction.csv     — junction-szintű részletek
"""

import os
import sys
import argparse
import json
import random
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SUMO import — MINDIG traci (IPC), hogy ne ütközzön a futó libsumo traininggel
# ---------------------------------------------------------------------------
import traci

# ---------------------------------------------------------------------------
# Reward függvény — AZONOS a sumo_rl_environment.py-ben lévővel
# ---------------------------------------------------------------------------
MU_WAIT  = 4.584800
STD_WAIT = 1.824900
MU_CO2   = 10.870600
STD_CO2  = 0.962900

def compute_reward(avg_waiting: float, avg_co2_raw: float,
                   w_wait: float = 1.0, w_co2: float = 1.0) -> float:
    """Pontosan ugyanaz a reward, mint sumo_rl_environment.py step()-ben."""
    if avg_waiting == 0.0 and avg_co2_raw == 0.0:
        return 0.0
    z_wait = (np.log(avg_waiting + 1e-5) - MU_WAIT) / (STD_WAIT + 1e-9)
    z_co2  = (np.log(avg_co2_raw + 1e-5) - MU_CO2)  / (STD_CO2  + 1e-9)
    r_wait = 1.0 - 1 / (1 + np.exp(-z_wait))
    r_co2  = 1.0 - 1 / (1 + np.exp(-z_co2))
    w_sum = w_wait + w_co2
    return (w_wait * r_wait + w_co2 * r_co2) / w_sum


# ===========================================================================
# SUMO-native stratégiák (static, actuated) — saját SUMO loop
# ===========================================================================
def generate_random_traffic(net_file, route_file, duration, flow_range=(100, 900)):
    """Lane-szintű random forgalom, azonos a sumo_rl_environment.py-vel."""
    import sumolib
    net = sumolib.net.readNet(net_file)
    with open(route_file, 'w') as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" accel="0.8" decel="4.5" sigma="0.5" '
                'length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
        veh_id = 0
        for edge in net.getEdges():
            if edge.getFunction() == 'internal':
                continue
            for lane in edge.getLanes():
                flow = random.randint(flow_range[0], flow_range[1])
                period = 3600.0 / flow
                t = random.uniform(0, period)
                while t < duration:
                    outgoing = edge.getOutgoing()
                    if outgoing:
                        to_edge = random.choice(list(outgoing.keys()))
                        f.write(f'  <trip id="v{veh_id}" type="car" depart="{t:.2f}" '
                                f'from="{edge.getID()}" to="{to_edge.getID()}" />\n')
                    veh_id += 1
                    t += period + random.uniform(-period*0.1, period*0.1)
        f.write('</routes>\n')
    return veh_id


def run_sumo_native(net_file, detector_file, logic_file, route_file,
                    strategy, duration, delta_time=5,
                    flow_range=(100, 900), single_junction=None):
    """Static vagy Actuated futtatás — SUMO vezérli a lámpákat."""
    generate_random_traffic(net_file, route_file, duration, flow_range)

    traci.start(["sumo",
        "-n", net_file, "-r", route_file, "-a", detector_file,
        "--no-step-log", "true", "--waiting-time-memory", "10000",
        "--ignore-route-errors", "true", "--random", "true",
        "--no-warnings", "true", "--xml-validation", "never"])

    with open(logic_file, 'r') as f:
        logic_data = json.load(f)
    junction_ids = list(logic_data.keys())
    if single_junction and single_junction in junction_ids:
        junction_ids = [single_junction]

    import sumolib
    net = sumolib.net.readNet(net_file)
    junction_lanes = {}
    for jid in junction_ids:
        node = net.getNode(jid)
        lanes = set()
        for edge in node.getIncoming():
            if edge.getFunction() == 'internal':
                continue
            for lane in edge.getLanes():
                lanes.add(lane.getID())
        junction_lanes[jid] = sorted(lanes)

    # Actuated beállítás
    if strategy == "actuated":
        for jid in junction_ids:
            try:
                all_logics = traci.trafficlight.getAllProgramLogics(jid)
                current_logic = all_logics[0] if all_logics else None
                if not current_logic:
                    continue
                phases = []
                for phase in current_logic.phases:
                    state = phase.state
                    has_green = any(c in ('G', 'g') for c in state)
                    has_only_ry = all(c in ('r', 'y', 'u') for c in state)
                    if has_green and not has_only_ry:
                        phases.append(traci.trafficlight.Phase(
                            duration=phase.duration, state=state,
                            minDur=max(5.0, phase.duration * 0.5),
                            maxDur=max(phase.duration * 2.0, 60.0)))
                    else:
                        phases.append(traci.trafficlight.Phase(
                            duration=phase.duration, state=state,
                            minDur=phase.duration, maxDur=phase.duration))
                actuated_logic = traci.trafficlight.Logic(
                    programID="actuated_baseline", type=1,
                    currentPhaseIndex=0, phases=phases)
                traci.trafficlight.setProgramLogic(jid, actuated_logic)
                traci.trafficlight.setProgram(jid, "actuated_baseline")
            except Exception as e:
                print(f"[WARNING] Actuated setup failed for {jid}: {e}")

    # Szimulációs loop + metrika gyűjtés
    junction_metrics = {jid: {
        'acc_waiting': 0.0, 'acc_co2': 0.0,
        'steps_measured': 0, 'step_rewards': [],
    } for jid in junction_ids}
    sim_step = 0

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < duration:
        traci.simulationStep()
        sim_step += 1
        for jid in junction_ids:
            m = junction_metrics[jid]
            lanes = junction_lanes[jid]
            if not lanes:
                continue
            m['acc_waiting'] += sum(traci.lane.getWaitingTime(l) for l in lanes)
            m['acc_co2'] += sum(traci.lane.getCO2Emission(l) for l in lanes)
            m['steps_measured'] += 1

        if sim_step % delta_time == 0:
            for jid in junction_ids:
                m = junction_metrics[jid]
                if m['steps_measured'] > 0:
                    r = compute_reward(
                        m['acc_waiting'] / m['steps_measured'],
                        m['acc_co2'] / m['steps_measured'])
                    m['step_rewards'].append(r)
                    m['acc_waiting'] = 0.0
                    m['acc_co2'] = 0.0
                    m['steps_measured'] = 0
    traci.close()

    results = {}
    for jid in junction_ids:
        rr = junction_metrics[jid]['step_rewards']
        results[jid] = {
            'avg_reward': np.mean(rr) if rr else 0.0,
            'std_reward': np.std(rr) if rr else 0.0,
            'min_reward': np.min(rr) if rr else 0.0,
            'max_reward': np.max(rr) if rr else 0.0,
            'num_steps': len(rr),
        }
    return results


# ===========================================================================
# RL-kompatibilis stratégiák (random, max_pressure) — SumoRLEnvironment
# ===========================================================================
def _build_phase_lane_map(env):
    """
    Junction-önként felépíti a fázis → incoming lane set mapping-et.
    Ez kell a max-pressure-nak, hogy tudja melyik action melyik sávokat zöldíti.

    Returns: {jid: {action_idx: set_of_incoming_lane_ids}}
    """
    phase_lane_map = {}
    for jid, agent in env.agents.items():
        controlled_links = traci.trafficlight.getControlledLinks(jid)
        action_lanes = {}
        for action_idx, sumo_phase_idx in agent.logic_phases.items():
            if sumo_phase_idx not in agent.phase_registry:
                continue
            state = agent.phase_registry[sumo_phase_idx]['state']
            green_lanes = set()
            for link_idx, char in enumerate(state):
                if char in ('G', 'g') and link_idx < len(controlled_links):
                    for link in controlled_links[link_idx]:
                        if link:
                            green_lanes.add(link[0])  # incoming lane
            action_lanes[action_idx] = green_lanes
        phase_lane_map[jid] = action_lanes
    return phase_lane_map


def _max_pressure_action(jid, agent, phase_lane_map):
    """
    Max-pressure: válaszd azt a fázist, amelyiknek a bemenő sávjain
    a legtöbb várakozó jármű van.

    pressure(action) = Σ halting vehicles on green incoming lanes
    """
    action_lanes = phase_lane_map.get(jid, {})
    best_action = 0
    best_pressure = -1

    for action_idx in range(agent.num_phases):
        green_lanes = action_lanes.get(action_idx, set())
        if not green_lanes:
            continue
        pressure = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in green_lanes
        )
        if pressure > best_pressure:
            best_pressure = pressure
            best_action = action_idx

    return best_action


def run_env_strategy(net_file, detector_file, logic_file,
                     strategy, duration, delta_time=5,
                     flow_range=(100, 900), single_junction=None):
    """
    Random vagy Max-Pressure futtatás a SumoRLEnvironment-en keresztül.
    Pontosan ugyanaz az env, mint amit az RL agent lát.
    """
    # Force traci (ne libsumo-t használjon, mert az ütközhet a traininggel)
    os.environ['USE_LIBSUMO'] = '0'

    from sumo_rl_environment import SumoRLEnvironment

    env = SumoRLEnvironment(
        net_file=net_file,
        logic_json_file=logic_file,
        detector_file=detector_file,
        delta_time=delta_time,
        random_traffic=True,
        traffic_duration=duration,
        flow_range=flow_range,
        single_agent_id=single_junction,
    )

    obs, info = env.reset(options={'traffic_duration': duration, 'flow_range': flow_range})

    # Max-pressure: felépítjük a fázis → lane mapping-et
    phase_lane_map = None
    if strategy == "max_pressure":
        phase_lane_map = _build_phase_lane_map(env)

    agent_ids = list(env.agents.keys())
    all_rewards = {jid: [] for jid in agent_ids}
    done = False

    while not done:
        actions = {}
        for jid in agent_ids:
            agent = env.agents[jid]
            if not info[jid].get('ready', True):
                # Átmenet alatt: maradjon a jelenlegi fázisnál
                actions[jid] = agent.current_logic_idx
                continue

            if strategy == "random":
                actions[jid] = random.randint(0, agent.num_phases - 1)
            elif strategy == "max_pressure":
                actions[jid] = _max_pressure_action(jid, agent, phase_lane_map)

        obs, rewards, done, _, info = env.step(actions)

        for jid in agent_ids:
            if jid in rewards:
                all_rewards[jid].append(rewards[jid])

    env.close()

    results = {}
    for jid in agent_ids:
        rr = all_rewards[jid]
        results[jid] = {
            'avg_reward': np.mean(rr) if rr else 0.0,
            'std_reward': np.std(rr) if rr else 0.0,
            'min_reward': np.min(rr) if rr else 0.0,
            'max_reward': np.max(rr) if rr else 0.0,
            'num_steps': len(rr),
        }
    return results


# ===========================================================================
# Fő kiértékelés
# ===========================================================================
ALL_STRATEGIES = ["static", "actuated", "random", "max_pressure"]

def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: static, actuated, random, max-pressure")
    parser.add_argument("--net", default="data/mega_catalogue_v2.net.xml")
    parser.add_argument("--logic", default="data/traffic_lights.json")
    parser.add_argument("--detector", default="data/detectors.add.xml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--duration", type=int, default=None,
                        help="Fix duration (default: random [900,1800,2700,3600])")
    parser.add_argument("--flow-min", type=int, default=100)
    parser.add_argument("--flow-max", type=int, default=900)
    parser.add_argument("--single-junction", default=None)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--delta-time", type=int, default=5)
    parser.add_argument("--wandb-project", default=None,
                        help="WandB project (ha megadod, logol WandB-be)")
    parser.add_argument("--strategies", default=None,
                        help="Vesszővel elválasztott lista (default: mind a 4)")
    args = parser.parse_args()

    strategies = (args.strategies.split(',') if args.strategies
                  else ALL_STRATEGIES)
    for s in strategies:
        if s not in ALL_STRATEGIES:
            print(f"[ERROR] Unknown strategy: {s}. Valid: {ALL_STRATEGIES}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    route_file = "baseline_traffic.rou.xml"
    duration_options = [900, 1800, 2700, 3600]

    all_results = []
    per_junction_results = []

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        # WandB run indítása
        wb_run = None
        if args.wandb_project:
            try:
                import wandb
                wb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"baseline-{strategy}",
                    tags=["baseline", strategy],
                    config={
                        "strategy": strategy,
                        "episodes": args.episodes,
                        "flow_min": args.flow_min,
                        "flow_max": args.flow_max,
                        "delta_time": args.delta_time,
                        "duration": args.duration or "random",
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[WARNING] WandB init failed: {e}")
                wb_run = None

        global_step_counter = 0

        for ep in range(args.episodes):
            duration = args.duration if args.duration else random.choice(duration_options)
            flow_range = (args.flow_min, args.flow_max)

            print(f"\n  Episode {ep+1}/{args.episodes} | "
                  f"duration={duration}s | flow={flow_range}")

            t0 = time.time()
            try:
                if strategy in ("static", "actuated"):
                    results = run_sumo_native(
                        net_file=args.net,
                        detector_file=args.detector,
                        logic_file=args.logic,
                        route_file=route_file,
                        strategy=strategy,
                        duration=duration,
                        delta_time=args.delta_time,
                        flow_range=flow_range,
                        single_junction=args.single_junction,
                    )
                else:  # random, max_pressure
                    results = run_env_strategy(
                        net_file=args.net,
                        detector_file=args.detector,
                        logic_file=args.logic,
                        strategy=strategy,
                        duration=duration,
                        delta_time=args.delta_time,
                        flow_range=flow_range,
                        single_junction=args.single_junction,
                    )
            except Exception as e:
                print(f"  [ERROR] Episode failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            elapsed = time.time() - t0

            avg_rewards = [r['avg_reward'] for r in results.values()]
            global_avg = np.mean(avg_rewards) if avg_rewards else 0.0
            ep_steps = max(r['num_steps'] for r in results.values()) if results else 0
            global_step_counter += ep_steps

            all_results.append({
                'strategy': strategy,
                'episode': ep + 1,
                'duration': duration,
                'flow_min': flow_range[0],
                'flow_max': flow_range[1],
                'avg_reward': global_avg,
                'elapsed_s': elapsed,
            })
            for jid, r in results.items():
                per_junction_results.append({
                    'strategy': strategy, 'episode': ep + 1,
                    'junction': jid, **r,
                })

            if wb_run:
                log_data = {
                    "avg_reward": global_avg,
                    "episode": ep + 1,
                    "episode_length": ep_steps,
                    "duration": duration,
                    "global_step": global_step_counter,
                }
                for jid, r in results.items():
                    log_data[f"reward/{jid}"] = r['avg_reward']
                wandb.log(log_data)

            print(f"  -> avg_reward={global_avg:.4f} ({elapsed:.1f}s)")

        # WandB lezárás
        if wb_run:
            s_df = pd.DataFrame([r for r in all_results if r['strategy'] == strategy])
            if not s_df.empty:
                wandb.run.summary["final_avg_reward"] = s_df['avg_reward'].mean()
                wandb.run.summary["final_std_reward"] = s_df['avg_reward'].std()
                wandb.run.summary["total_episodes"] = len(s_df)
                wandb.run.summary["total_steps"] = global_step_counter
            wandb.finish()
            print(f"  [WandB] Run 'baseline-{strategy}' finished.")

    # --- Mentés ---
    df_summary = pd.DataFrame(all_results)
    df_junctions = pd.DataFrame(per_junction_results)

    summary_path = os.path.join(args.output_dir, "baseline_summary.csv")
    junction_path = os.path.join(args.output_dir, "baseline_per_junction.csv")
    df_summary.to_csv(summary_path, index=False)
    df_junctions.to_csv(junction_path, index=False)
    print(f"\n[OK] Saved: {summary_path}")
    print(f"[OK] Saved: {junction_path}")

    # --- Összefoglaló ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for s in strategies:
        s_df = df_summary[df_summary['strategy'] == s]
        if not s_df.empty:
            print(f"  {s.upper():>14s}:  "
                  f"avg_reward = {s_df['avg_reward'].mean():.4f} "
                  f"+/- {s_df['avg_reward'].std():.4f}  "
                  f"(min={s_df['avg_reward'].min():.4f}, "
                  f"max={s_df['avg_reward'].max():.4f})")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = {'static': '#95a5a6', 'actuated': '#3498db',
                  'random': '#e74c3c', 'max_pressure': '#2ecc71'}

        # 1. Box plot
        ax = axes[0]
        plot_strategies = [s for s in strategies if s in df_summary['strategy'].unique()]
        data = [df_summary[df_summary['strategy'] == s]['avg_reward'].values
                for s in plot_strategies]
        if data:
            bp = ax.boxplot(data, labels=[s.upper().replace('_', '\n')
                            for s in plot_strategies], patch_artist=True)
            for patch, s in zip(bp['boxes'], plot_strategies):
                patch.set_facecolor(colors.get(s, '#999'))
                patch.set_alpha(0.7)
        ax.set_ylabel('Average Reward (log-sigmoid)')
        ax.set_title('Reward Distribution per Strategy')
        ax.grid(axis='y', alpha=0.3)

        # 2. Junction-szintű összehasonlítás (max_pressure vs legjobb baseline)
        ax = axes[1]
        if not df_junctions.empty and len(plot_strategies) >= 2:
            pivot = df_junctions.groupby(['strategy', 'junction'])['avg_reward'].mean()
            if 'max_pressure' in plot_strategies and 'static' in plot_strategies:
                mp = pivot.get('max_pressure', pd.Series())
                st = pivot.get('static', pd.Series())
                if not mp.empty and not st.empty:
                    common = mp.index.intersection(st.index)
                    diff = (mp[common] - st[common]).sort_values()
                    c = ['#e74c3c' if v < 0 else '#2ecc71' for v in diff.values]
                    diff.plot(kind='barh', ax=ax, color=c)
                    ax.set_xlabel('Reward diff (max_pressure - static)')
                    ax.set_title('Max-Pressure vs Static per Junction')
                    ax.axvline(x=0, color='black', linewidth=0.8)
                    ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "baseline_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[OK] Plot: {plot_path}")
    except Exception as e:
        print(f"[WARNING] Plot failed: {e}")


if __name__ == "__main__":
    main()
