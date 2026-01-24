"""
Random Baseline Runner - Reward statisztikák gyűjtése random akciókkal.

Célja: Statisztikai baseline adatok gyűjtése a reward-okról,
       ágens nélkül, minimális gépterheléssel.

Használat:
    python misc/random_baseline_runner.py

    # Vagy paraméterekkel:
    python misc/random_baseline_runner.py --episodes 200 --steps 500 --output baseline_stats
"""

import os
import sys
import argparse
import random
import time
from datetime import datetime

# Projekt root meghatározása és hozzáadása a path-hoz
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Munkakönyvtár beállítása a projekt gyökerére (hogy a relatív fájlok működjenek)
os.chdir(PROJECT_ROOT)

from sumo_rl_environment import SumoRLEnvironment
from misc.reward_monitor import RewardMonitor


def run_random_baseline(
    episodes: int = 100,
    steps_per_episode: int = 500,
    output_dir: str = "baseline_reward_logs",
    net_file: str = "data/Biharia_Magnoliei.net.xml",
    logic_json: str = "data/traffic_lights.json",
    detector_file: str = "data/detectors.add.xml",
    traffic_period: float = 1.0,
    min_green_time: int = 5,
    skip_warmup: bool = False,
    statistic_output_file: str = None # [NEW]
):
    """
    Random akciókkal futtatja a szimulációt és gyűjti a reward statisztikákat.

    Args:
        episodes: Epizódok száma
        steps_per_episode: Lépések száma epizódonként
        output_dir: Kimeneti mappa a CSV fájloknak
        net_file: SUMO hálózat fájl
        logic_json: Jelzőlámpa logika JSON
        detector_file: Detektor fájl
        traffic_period: Forgalom sűrűség (kisebb = sűrűbb)
        min_green_time: Minimális zöld idő
        skip_warmup: Warmup kihagyása (gyorsabb, de kevésbé realisztikus)
    """

    print("="*70)
    print("  RANDOM BASELINE RUNNER - Reward Statisztikák Gyűjtése")
    print("="*70)
    print(f"  Epizódok: {episodes}")
    print(f"  Lépések/epizód: {steps_per_episode}")
    print(f"  Kimenet: {output_dir}/")
    print("="*70)

    # Kimenet mappa létrehozása időbélyeggel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}_{timestamp}"
    os.makedirs(output_path, exist_ok=True)

    # Environment létrehozása
    print("\n[1/3] Environment inicializálása...")
    env = SumoRLEnvironment(
        net_file=net_file,
        logic_json_file=logic_json,
        detector_file=detector_file,
        reward_weights={'time': 1.0, 'co2': 1.0},  # Nem számít, mert nyers értékeket mentünk
        min_green_time=min_green_time,
        delta_time=1,
        sumo_gui=False,
        random_traffic=True,
        traffic_period=traffic_period,

        traffic_duration=steps_per_episode + 100,
        statistic_output_file=statistic_output_file # [NEW]
    )

    junction_ids = env.junction_ids
    print(f"  Kereszteződések: {junction_ids}")

    # Monitor létrehozása - GLOBÁLIS az összes epizódhoz
    monitor = RewardMonitor(output_dir=output_path)

    # Epizódonkénti statisztikák
    episode_stats = []

    global_step = 0
    start_time = time.time()

    print(f"\n[2/3] Futtatás ({episodes} epizód)...\n")

    for episode in range(episodes):
        episode_start = time.time()

        # Reset
        obs, info = env.reset()

        episode_co2 = {jid: 0.0 for jid in junction_ids}
        episode_tt = {jid: 0.0 for jid in junction_ids}
        episode_actions = {jid: 0 for jid in junction_ids}

        for step in range(steps_per_episode):
            # Random akciók minden ágensnek
            actions = {}
            for jid in junction_ids:
                # Csak ha ready, akkor választunk új akciót
                if info.get(jid, {}).get('ready', True):
                    # FONTOS: num_phases = zöld fázisok száma (logic_phases), NEM az összes fázis!
                    num_phases = env.agents[jid].num_phases
                    actions[jid] = random.randint(0, num_phases - 1)
                    episode_actions[jid] += 1
                else:
                    # Ha nem ready, tartjuk az előző fázist (vagy 0)
                    actions[jid] = env.agents[jid].target_logic_idx

            # Step
            obs, rewards, terminated, truncated, infos = env.step(actions)

            # Logolás (globális step counter)
            monitor.log_step(global_step, infos)
            global_step += 1

            # Epizód statisztika
            for jid in junction_ids:
                if jid in infos:
                    episode_co2[jid] += infos[jid].get('metric_co2', 0)
                    episode_tt[jid] += infos[jid].get('metric_travel_time', 0)

            # Ha véget ért a szimuláció
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        episode_time = time.time() - episode_start

        # Epizód összegzés
        total_co2 = sum(episode_co2.values())
        total_tt = sum(episode_tt.values())
        total_actions = sum(episode_actions.values())

        episode_stats.append({
            'episode': episode,
            'total_co2': total_co2,
            'total_travel_time': total_tt,
            'total_actions': total_actions,
            'duration_sec': episode_time
        })

        # Progress kiírás
        elapsed = time.time() - start_time
        eta = (elapsed / (episode + 1)) * (episodes - episode - 1)
        print(f"  Ep {episode+1:3d}/{episodes} | "
              f"CO2: {total_co2:10.1f} | "
              f"TT: {total_tt:8.1f} | "
              f"Actions: {total_actions:4d} | "
              f"Time: {episode_time:5.1f}s | "
              f"ETA: {eta/60:5.1f}min")

    # Szimuláció lezárása
    try:
        env.close()
    except:
        pass

    print(f"\n[3/3] Eredmények mentése...")

    # Monitor mentés
    monitor.save_all()

    # Epizód statisztikák mentése
    import csv
    episode_stats_file = os.path.join(output_path, "episode_summary.csv")
    with open(episode_stats_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'total_co2', 'total_travel_time', 'total_actions', 'duration_sec'])
        writer.writeheader()
        writer.writerows(episode_stats)

    # Összefoglaló statisztikák
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("  ÖSSZEFOGLALÓ")
    print("="*70)
    print(f"  Összes lépés: {global_step}")
    print(f"  Futási idő: {total_time/60:.1f} perc")
    print(f"  Átlag lépés/sec: {global_step/total_time:.1f}")
    print("-"*70)

    # Átlagok számítása
    avg_co2 = sum(e['total_co2'] for e in episode_stats) / len(episode_stats)
    avg_tt = sum(e['total_travel_time'] for e in episode_stats) / len(episode_stats)
    print(f"  Átlag CO2/epizód: {avg_co2:.2f}")
    print(f"  Átlag Travel Time/epizód: {avg_tt:.2f}")

    monitor.print_summary()

    print(f"\n  Fájlok mentve: {output_path}/")
    print("    - episode_summary.csv (epizódonkénti összegzés)")
    print("    - <junction>_co2.csv (lépésenkénti CO2)")
    print("    - <junction>_travel_time.csv (lépésenkénti travel time)")
    print("="*70)

    return output_path, episode_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Baseline Runner - Reward statisztikák gyűjtése")

    parser.add_argument("--episodes", "-e", type=int, default=100,
                        help="Epizódok száma (default: 100)")
    parser.add_argument("--steps", "-s", type=int, default=500,
                        help="Lépések száma epizódonként (default: 500)")
    parser.add_argument("--output", "-o", type=str, default="baseline_reward_logs",
                        help="Kimeneti mappa neve (default: baseline_reward_logs)")
    parser.add_argument("--net", type=str, default="data/mega_catalogue_v2.net.xml",
                        help="SUMO hálózat fájl")
    parser.add_argument("--logic", type=str, default="data/traffic_lights.json",
                        help="Jelzőlámpa logika JSON")
    parser.add_argument("--detectors", type=str, default="data/detectors.add.xml",
                        help="Detektor fájl")
    parser.add_argument("--period", type=float, default=1.0,
                        help="Forgalom periódus (kisebb = sűrűbb)")
    parser.add_argument("--min-green", type=int, default=5,
                        help="Minimális zöld idő (default: 5)")
    parser.add_argument("--stats-output", type=str, default=None,
                        help="Kimeneti XML fájl a SUMO statisztikákhoz (pl. stats.xml)")

    args = parser.parse_args()

    run_random_baseline(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output,
        net_file=args.net,
        logic_json=args.logic,
        detector_file=args.detectors,
        traffic_period=args.period,
        min_green_time=args.min_green,
        statistic_output_file=args.stats_output
    )
