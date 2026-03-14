#!/usr/bin/env python3
"""
SUMO GUI Preview - Edge-alapú forgalom generálás vizualizáció.

Elindítja a SUMO-GUI-t az új edge-alapú forgalom generálással,
és véletlenszerű jelzőlámpa váltásokkal futtatja a szimulációt.

Használat:
    python gui_preview.py
    python gui_preview.py --duration 1800
    python gui_preview.py --flow-min 200 --flow-max 600
"""

import os
import sys
import random
import time
import argparse

# Paths - adjust if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

NET_FILE = os.path.join(DATA_DIR, "mega_catalogue_v2.net.xml")
LOGIC_FILE = os.path.join(DATA_DIR, "traffic_lights.json")
DETECTOR_FILE = os.path.join(DATA_DIR, "detectors.add.xml")
ROUTE_FILE = os.path.join(SCRIPT_DIR, "gui_preview_traffic.rou.xml")


def main():
    parser = argparse.ArgumentParser(description="SUMO GUI Preview")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Szimulacios ido masodpercben (default: 3600)")
    parser.add_argument("--flow-min", type=int, default=100,
                        help="Min forgalom edge-enkent jarmu/ora (default: 100)")
    parser.add_argument("--flow-max", type=int, default=900,
                        help="Max forgalom edge-enkent jarmu/ora (default: 900)")
    parser.add_argument("--delta-time", type=int, default=5,
                        help="Lepes meret masodpercben (default: 5)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Szimulacio sebesseg (delay ms, 0=max speed)")
    args = parser.parse_args()

    from sumo_rl_environment import SumoRLEnvironment

    print("=" * 60)
    print("  SUMO GUI Preview - Edge-alapu forgalom")
    print("=" * 60)
    print(f"  Forgalom: {args.flow_min}-{args.flow_max} jarmu/ora/edge")
    print(f"  Idotartam: {args.duration}s")
    print(f"  Delta time: {args.delta_time}s")
    print("=" * 60)

    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_FILE,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        sumo_gui=True,
        random_traffic=True,
        traffic_duration=args.duration,
        delta_time=args.delta_time,
    )

    obs, info = env.reset(options={
        'flow_range': (args.flow_min, args.flow_max),
        'warmup_seconds': 100,
    })

    print(f"\n  Agensek: {len(env.agents)}")
    for jid, agent in env.agents.items():
        print(f"    {jid}: {agent.num_phases} fazis, "
              f"{len(agent.detectors)} detektor, "
              f"{len(agent.incoming_lanes)} bejovo sav")

    print(f"\n  Szimulacio indul... (Ctrl+C a leallitashoz)\n")

    step = 0
    done = False

    try:
        while not done:
            # Random akciok minden agenshez
            actions = {}
            for jid, agent in env.agents.items():
                if agent.is_ready_for_action():
                    actions[jid] = random.randint(0, agent.num_phases - 1)
                else:
                    actions[jid] = agent.current_logic_idx

            obs, rewards, done, truncated, infos = env.step(actions)
            step += 1

            # Kiiras minden 100. lepesben
            if step % 100 == 0:
                avg_reward = sum(rewards.values()) / len(rewards)
                ready_count = sum(1 for jid in infos if infos[jid].get("ready"))
                print(f"  Step {step:5d} | "
                      f"Avg reward: {avg_reward:+.4f} | "
                      f"Ready: {ready_count}/{len(env.agents)}")

    except KeyboardInterrupt:
        print("\n  Leallitva.")
    finally:
        env.close()
        print("  SUMO bezarva.")

        # Cleanup route file
        if os.path.exists(ROUTE_FILE):
            os.remove(ROUTE_FILE)
            print(f"  Route file torolve: {ROUTE_FILE}")


if __name__ == "__main__":
    main()
