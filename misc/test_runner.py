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


def select_junction(env):
    """Interaktív csomópont választó."""
    junction_ids = list(env.agents.keys())

    print("\n" + "="*50)
    print("Elérhető csomópontok:")
    print("="*50)

    for i, jid in enumerate(junction_ids):
        agent = env.agents[jid]
        num_phases = agent.num_phases
        num_detectors = len(agent.detectors)
        print(f"  [{i:2d}] {jid} - {num_phases} fázis, {num_detectors} detektor")

    print("="*50)

    while True:
        try:
            choice = input(f"\nVálassz csomópontot [0-{len(junction_ids)-1}] (vagy 'q' kilépés): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice)
            if 0 <= idx < len(junction_ids):
                return junction_ids[idx]
            print("Érvénytelen index!")
        except ValueError:
            print("Adj meg egy számot!")


def select_phase(agent):
    """Interaktív fázis választó."""
    print("\n" + "-"*40)
    print("Elérhető fázisok:")
    print("-"*40)

    for logic_idx, sumo_idx in agent.logic_phases.items():
        state_str = agent.phase_registry.get(sumo_idx, {}).get('state', '???')
        print(f"  [{logic_idx}] {state_str}")

    print("-"*40)

    while True:
        try:
            choice = input(f"Válassz célfázist [0-{agent.num_phases-1}] (vagy 'q' kilépés): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice)
            if idx in agent.logic_phases:
                return idx
            print("Érvénytelen fázis!")
        except ValueError:
            print("Adj meg egy számot!")


def print_agent_state(step, jid, agent, obs, reward, info, target_phase=None):
    """Részletes állapot kiírás egy ágensről."""
    phase = int(obs['phase'][0])
    occupancy = obs['occupancy']
    flow = obs['flow']
    is_ready = info['ready']

    # Logikai fázis string (zöld fázis)
    logic_sumo_idx = agent.logic_phases.get(phase, 0)
    logic_state_str = agent.phase_registry.get(logic_sumo_idx, {}).get('state', '???')

    # Aktuális SUMO állapot (ami ténylegesen ki van adva)
    actual_state = agent.current_sumo_state
    actual_idx = agent.current_sumo_phase_idx

    status = "\033[92mREADY\033[0m" if is_ready else "\033[93mBUSY \033[0m"

    print(f"\n┌─ Step {step:03d} ─ {jid} ─────────────────────────────")
    print(f"│ Logikai fázis: {phase} ({logic_state_str}) [{status}]")

    # Ha átmenetben van, színezzük pirosra az aktuális állapotot
    if agent.is_transitioning:
        print(f"│ \033[91mAktuális SUMO: idx={actual_idx} -> {actual_state}\033[0m")
    else:
        print(f"│ Aktuális SUMO: idx={actual_idx} -> {actual_state}")

    # Célfázis kiírása
    if target_phase is not None:
        target_sumo_idx = agent.logic_phases.get(target_phase, 0)
        target_state_str = agent.phase_registry.get(target_sumo_idx, {}).get('state', '???')
        if phase == target_phase and not agent.is_transitioning:
            print(f"│ \033[92mCélfázis: {target_phase} ({target_state_str}) ✓ ELÉRVE\033[0m")
        else:
            print(f"│ \033[96mCélfázis: {target_phase} ({target_state_str})\033[0m")

    print(f"│ Reward: {reward:+.3f}")
    print(f"│ Transitioning: {agent.is_transitioning} (cursor: {agent.transition_cursor}/{len(agent.transition_queue)})")
    print(f"│ Min Green Timer: {agent.min_green_timer}")

    if len(occupancy) > 0:
        print(f"│ Occupancy (avg): {np.mean(occupancy)*100:.1f}%")
        print(f"│ Flow (sum): {np.sum(flow):.1f} veh")

    print(f"└{'─'*50}")


def main():
    print("\n" + "="*60)
    print("   SUMO RL Environment - Manuális Tesztelő")
    print("="*60)

    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_JSON,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        reward_weights={'waiting': 1.0, 'co2': 0.05},
        min_green_time=5,
        delta_time=1,
        measure_during_transition=False,
        sumo_gui=False,  # GUI bekapcsolva a vizuális ellenőrzéshez
        random_traffic=True,
        traffic_period=0.5,
        traffic_duration=3600
    )

    print("\nResetting environment...")
    observations, infos = env.reset()
    print(f"Szimuláció elindult! ({len(env.agents)} ágens)")

    # Csomópont választás
    selected_jid = select_junction(env)
    if selected_jid is None:
        print("Kilépés...")
        env.close()
        return

    print(f"\n>>> Kiválasztva: {selected_jid}")

    agent = env.agents[selected_jid]
    step = 0
    target_phase = None

    try:
        while True:
            # Ha nincs célfázis VAGY elértük a célt -> kérjünk újat
            current_phase = int(observations[selected_jid]['phase'][0])
            reached_target = (target_phase is not None and
                              current_phase == target_phase and
                              not agent.is_transitioning)

            if target_phase is None or reached_target:
                if reached_target:
                    print(f"\n\033[92m>>> Célfázis ({target_phase}) elérve! <<<\033[0m")

                target_phase = select_phase(agent)
                if target_phase is None:
                    print("Kilépés...")
                    break

                print(f"\n>>> Új célfázis: {target_phase}")
                print(">>> Lépkedés a célig...\n")
                time.sleep(0.5)

            # --- AKCIÓ BEÁLLÍTÁSA ---
            actions = {}

            # A kiválasztott ágensnek a célfázist adjuk
            if infos[selected_jid]['ready']:
                actions[selected_jid] = target_phase

            # Többi ágensnek random akció (hogy a szimuláció fusson)
            for jid, ag in env.agents.items():
                if jid != selected_jid and infos[jid]['ready']:
                    available = list(ag.logic_phases.keys())
                    actions[jid] = random.choice(available)

            # --- LÉPÉS ---
            next_obs, rewards, terminated, truncated, next_infos = env.step(actions)

            # --- KIÍRÁS ---
            obs = next_obs[selected_jid]
            reward = rewards[selected_jid]
            info = next_infos[selected_jid]

            print_agent_state(step, selected_jid, agent, obs, reward, info, target_phase)

            infos = next_infos
            observations = next_obs
            step += 1

            if terminated["__all__"]:
                print("\n[!] Szimuláció vége (nincs több jármű)")
                break

            # Kis szünet, hogy olvasható legyen
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n[!] Teszt leállítva felhasználó által.")

    finally:
        print(f"\nÖsszesen {step} lépés történt.")
        print("Bezárás...")
        env.close()


if __name__ == "__main__":
    if not os.path.exists(NET_FILE):
        print(f"HIBA: Hiányzó fájl: {NET_FILE}")
    elif not os.path.exists(LOGIC_JSON):
        print(f"HIBA: Hiányzó fájl: {LOGIC_JSON}")
    elif not os.path.exists(DETECTOR_FILE):
        print(f"HIBA: Hiányzó fájl: {DETECTOR_FILE}")
    else:
        main()
