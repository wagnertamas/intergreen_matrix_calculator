import time
import random
import os
import numpy as np
from sumo_rl_environment import SumoRLEnvironment

# --- KONFIGURÁCIÓ ---
NET_FILE = "mega_catalogue_v2.net.xml"
LOGIC_JSON = "traffic_lights.json"
DETECTOR_FILE = "detectors.add.xml"
ROUTE_FILE = "random_traffic.rou.xml"


def main():
    print("--- SUMO RL: Resource Optimized Execution ---")

    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_JSON,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        reward_weights={'time': 1.0, 'co2': 0.05},
        min_green_time=5,
        delta_time=1,
        measure_during_transition=False,
        sumo_gui=True,
        random_traffic=True,
        traffic_period=0.1,
        traffic_duration=3600
    )

    print("Resetting environment...")
    observations, infos = env.reset()
    print("Szimuláció elindult!")

    ai_calls = 0
    total_opportunities = 0

    try:
        total_steps = 1000
        for step in range(total_steps):

            # --- INTELLIGENS DÖNTÉSHOZATAL ---
            actions = {}

            for jid, agent in env.agents.items():
                total_opportunities += 1

                # Itt használjuk az INFOS-t a státusz ellenőrzésére
                is_ready = infos[jid]['ready']

                if is_ready:
                    ai_calls += 1
                    available_phases = list(agent.logic_phases.keys())
                    action = random.choice(available_phases)
                    actions[jid] = action
                else:
                    pass

                    # --- LÉPÉS ---
            next_obs, rewards, terminated, truncated, next_infos = env.step(actions)

            infos = next_infos
            observations = next_obs

            # --- DEBUG KIÍRATÁS (JAVÍTVA) ---
            if step % 10 == 0:
                first_jid = list(env.agents.keys())[0]
                obs = observations[first_jid]

                current_phase = int(obs['phase'][0])
                reward = rewards[first_jid]

                # A státuszt most már az 'infos' (is_ready) alapján írjuk ki,
                # mert az obs-ban nincs benne.
                is_ready_now = infos[first_jid]['ready']
                status_str = "READY" if is_ready_now else "BUSY"

                print(f"Step {step:03d} | Agent {first_jid}: Ph={current_phase} [{status_str}] | "
                      f"Reward={reward:.2f} | AI Calls: {ai_calls}/{total_opportunities}")

            if terminated["__all__"]:
                break

    except KeyboardInterrupt:
        print("\nTeszt leállítva.")

    finally:
        print("Bezárás...")
        env.close()


if __name__ == "__main__":
    if not os.path.exists(NET_FILE):
        print("HIBA: Hiányzó fájlok.")
    else:
        main()