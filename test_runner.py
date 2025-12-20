import time
import random
import os
import numpy as np
from sumo_rl_environment import SumoRLEnvironment

# --- KONFIGURÁCIÓ ---
NET_FILE = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/mega_catalogue_v2.net.xml"
LOGIC_JSON = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/traffic_lights.json"
DETECTOR_FILE = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/detectors.add.xml"
ROUTE_FILE = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/random_traffic.rou.xml"

def main():
    print("--- SUMO RL Environment Teszt (Heavy Traffic) ---")

    # MÓDOSÍTÁS: A traffic_period értékét 0.1-re vettem.
    # Ez azt jelenti, hogy másodpercenként 10 autó próbál elindulni a hálózaton.
    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_JSON,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        reward_weights={'time': 1.0, 'co2': 0.05},
        min_green_time=5,
        delta_time=1,
        measure_during_transition=False,
        sumo_gui=False,
        random_traffic=True,
        
        # --- ITT A LÉNYEG ---
        traffic_period=0.1,   # 0.1 = 10 autó/mp (Nagyon sűrű!) | 0.2 = 5 autó/mp (Sűrű)
        traffic_duration=3600 # 1 órányi forgalom generálása
        # --------------------
    )

    print("Resetting environment (Forgalom generálása)...")
    observations, infos = env.reset()
    print("Szimuláció elindult! (Várj kicsit, amíg a járművek beérnek a központba)")

    try:
        total_steps = 1000
        for step in range(total_steps):
            
            # Véletlenszerű cselekvések
            actions = {}
            for jid, agent in env.agents.items():
                available_phases = list(agent.logic_phases.keys())
                action = random.choice(available_phases)
                actions[jid] = action

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            # Debug kiíratás minden 10. lépésben, hogy ne spammelje tele a konzolt
            if step % 10 == 0:
                first_jid = list(env.agents.keys())[0]
                
                # Mostantól az obs egy DICT!
                obs = next_obs[first_jid] 
                current_phase = int(obs['phase'][0]) # Így érjük el a fázist
                occ_vector = obs['occupancy']        # Ez egy tömb
                flow_vector = obs['flow']            # Ez is egy tömb
                
                reward = rewards[first_jid]
                is_ready = infos[first_jid]['ready']
                status_str = "READY" if is_ready else "BUSY"
                
                print(f"Step {step:03d} | Agent {first_jid}: Ph={current_phase} [{status_str}] | "
                      f"AvgOcc={np.mean(occ_vector):.2f} | Reward={reward:.2f}")

            if terminated["__all__"]:
                break

    except KeyboardInterrupt:
        print("\nTeszt leállítva.")
    
    finally:
        print("Bezárás...")
        env.close()

if __name__ == "__main__":
    if not os.path.exists(NET_FILE) or not os.path.exists(LOGIC_JSON):
        print(f"HIBA: Hiányzó fájlok ({NET_FILE}, {LOGIC_JSON}).")
    else:
        main()