"""
Automatizált tesztek a SUMO RL környezethez.

Futtatás:
    python test_automated.py

Tesztelési esetek:
    1. Környezet inicializálás és reset
    2. Fázisváltás minden irányba (0->1, 0->2, 0->3, 1->0, stb.)
    3. Átmeneti állapotok ellenőrzése (sárga, piros-sárga)
    4. Min green timer működése
    5. READY/BUSY állapotok
    6. Observation space struktúra
    7. Reward számítás
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple

# --- KONFIGURÁCIÓ ---
NET_FILE = "mega_catalogue_v2.net.xml"
LOGIC_JSON = "traffic_lights.json"
DETECTOR_FILE = "detectors.add.xml"
ROUTE_FILE = "random_traffic.rou.xml"

# Színes kimenet
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


class TestResult:
    """Teszt eredmény tárolása."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = []

    def success(self, msg: str = "OK"):
        self.passed = True
        self.message = msg

    def fail(self, msg: str, details: List[str] = None):
        self.passed = False
        self.message = msg
        self.details = details or []


class SumoRLTester:
    """Automatizált tesztelő osztály."""

    def __init__(self):
        self.env = None
        self.results: List[TestResult] = []

    def run_all_tests(self):
        """Összes teszt futtatása."""
        print("\n" + "="*60)
        print(f"   {CYAN}SUMO RL Environment - Automatizált Tesztek{RESET}")
        print("="*60 + "\n")

        # Fájlok ellenőrzése
        if not self._check_files():
            return

        # Tesztek futtatása
        self._test_01_environment_init()
        self._test_02_reset()
        self._test_03_observation_structure()
        self._test_04_action_space()
        self._test_05_same_phase_no_transition()
        self._test_06_phase_transition_all_directions()
        self._test_07_transition_states_yellow()
        self._test_08_min_green_timer()
        self._test_09_ready_busy_states()
        self._test_10_reward_calculation()
        self._test_11_multiple_agents()
        self._test_12_parallel_independent_transitions()
        self._test_13_agents_different_states_same_time()
        self._test_14_no_state_interference()
        self._test_15_all_agents_all_transitions()

        # Cleanup
        if self.env:
            self.env.close()

        # Összesítés
        self._print_summary()

    def _check_files(self) -> bool:
        """Szükséges fájlok ellenőrzése."""
        files = [NET_FILE, LOGIC_JSON, DETECTOR_FILE]
        missing = [f for f in files if not os.path.exists(f)]

        if missing:
            print(f"{RED}HIBA: Hiányzó fájlok:{RESET}")
            for f in missing:
                print(f"  - {f}")
            return False
        return True

    def _add_result(self, result: TestResult):
        """Eredmény hozzáadása és kiírása."""
        self.results.append(result)
        status = f"{GREEN}✓ PASS{RESET}" if result.passed else f"{RED}✗ FAIL{RESET}"
        print(f"  [{status}] {result.name}: {result.message}")
        for detail in result.details:
            print(f"           {detail}")

    # =========================================================================
    # TESZTEK
    # =========================================================================

    def _test_01_environment_init(self):
        """Teszt: Környezet inicializálása."""
        result = TestResult("Környezet inicializálás")

        try:
            from sumo_rl_environment import SumoRLEnvironment

            self.env = SumoRLEnvironment(
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
                traffic_period=1.0,
                traffic_duration=300
            )
            result.success(f"{len(self.env.agents)} ágens betöltve")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_02_reset(self):
        """Teszt: Environment reset."""
        result = TestResult("Environment reset")

        try:
            observations, infos = self.env.reset()

            if not isinstance(observations, dict):
                result.fail("observations nem dict")
            elif not isinstance(infos, dict):
                result.fail("infos nem dict")
            elif len(observations) != len(self.env.agents):
                result.fail(f"observations méret: {len(observations)} != {len(self.env.agents)}")
            else:
                result.success(f"{len(observations)} ágens inicializálva")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_03_observation_structure(self):
        """Teszt: Observation space struktúra."""
        result = TestResult("Observation struktúra")

        try:
            observations, _ = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            obs = observations[jid]

            errors = []

            if 'phase' not in obs:
                errors.append("'phase' kulcs hiányzik")
            elif obs['phase'].shape != (1,):
                errors.append(f"phase shape: {obs['phase'].shape}, várt: (1,)")

            if 'occupancy' not in obs:
                errors.append("'occupancy' kulcs hiányzik")
            elif not isinstance(obs['occupancy'], np.ndarray):
                errors.append("occupancy nem numpy array")

            if 'flow' not in obs:
                errors.append("'flow' kulcs hiányzik")
            elif not isinstance(obs['flow'], np.ndarray):
                errors.append("flow nem numpy array")

            if errors:
                result.fail("Struktúra hibák", errors)
            else:
                agent = self.env.agents[jid]
                result.success(f"phase, occupancy[{len(obs['occupancy'])}], flow[{len(obs['flow'])}]")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_04_action_space(self):
        """Teszt: Action space és logic_phases."""
        result = TestResult("Action space")

        try:
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            errors = []

            if not hasattr(agent, 'logic_phases'):
                errors.append("logic_phases attribútum hiányzik")
            elif len(agent.logic_phases) == 0:
                errors.append("logic_phases üres")

            if not hasattr(agent, 'num_phases'):
                errors.append("num_phases attribútum hiányzik")
            elif agent.num_phases != len(agent.logic_phases):
                errors.append(f"num_phases ({agent.num_phases}) != len(logic_phases) ({len(agent.logic_phases)})")

            if errors:
                result.fail("Action space hibák", errors)
            else:
                phases = list(agent.logic_phases.keys())
                result.success(f"{agent.num_phases} fázis: {phases}")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_05_same_phase_no_transition(self):
        """Teszt: Azonos fázis választása nem indít átmenetet."""
        result = TestResult("Azonos fázis = nincs átmenet")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            current_phase = int(observations[jid]['phase'][0])

            # Ugyanazt a fázist választjuk
            actions = {jid: current_phase}
            next_obs, _, _, _, next_infos = self.env.step(actions)

            if agent.is_transitioning:
                result.fail("Átmenet indult azonos fázisra")
            else:
                result.success("Nincs átmenet azonos fázisra")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_06_phase_transition_all_directions(self):
        """Teszt: Fázisváltás minden irányba."""
        result = TestResult("Fázisváltás minden irányba")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            num_phases = agent.num_phases
            transitions_tested = []
            errors = []

            # Teszteljük az első néhány átmenetet
            test_pairs = []
            for i in range(min(num_phases, 3)):
                for j in range(min(num_phases, 3)):
                    if i != j:
                        test_pairs.append((i, j))

            for from_phase, to_phase in test_pairs[:4]:  # Max 4 tesztelése
                # Reset és várakozás READY állapotra
                observations, infos = self.env.reset()

                # Várjuk meg, hogy READY legyen
                for _ in range(20):
                    if infos[jid]['ready']:
                        break
                    next_obs, _, _, _, next_infos = self.env.step({})
                    observations, infos = next_obs, next_infos

                if not infos[jid]['ready']:
                    errors.append(f"{from_phase}->{to_phase}: Nem lett READY")
                    continue

                # Először menjünk from_phase-re
                current = int(observations[jid]['phase'][0])
                if current != from_phase:
                    actions = {jid: from_phase}
                    for _ in range(30):
                        next_obs, _, _, _, next_infos = self.env.step(actions)
                        observations, infos = next_obs, next_infos
                        if int(observations[jid]['phase'][0]) == from_phase and infos[jid]['ready']:
                            break

                # Most váltsunk to_phase-re
                actions = {jid: to_phase}
                transition_started = False

                for step in range(30):
                    next_obs, _, _, _, next_infos = self.env.step(actions)

                    if agent.is_transitioning:
                        transition_started = True

                    observations, infos = next_obs, next_infos

                    if int(observations[jid]['phase'][0]) == to_phase and not agent.is_transitioning:
                        transitions_tested.append(f"{from_phase}->{to_phase}")
                        break
                else:
                    errors.append(f"{from_phase}->{to_phase}: Nem ért célba 30 lépés alatt")

            if errors:
                result.fail(f"Hibás átmenetek", errors)
            else:
                result.success(f"Tesztelve: {transitions_tested}")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_07_transition_states_yellow(self):
        """Teszt: Átmeneti állapotok tartalmaznak sárgát."""
        result = TestResult("Átmeneti állapotok (sárga)")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            # Várjuk meg READY-t
            for _ in range(20):
                if infos[jid]['ready']:
                    break
                _, _, _, _, infos = self.env.step({})

            # Váltsunk másik fázisra
            current = int(observations[jid]['phase'][0])
            target = (current + 1) % agent.num_phases

            actions = {jid: target}
            observed_states = []
            yellow_found = False
            red_yellow_found = False

            for _ in range(30):
                next_obs, _, _, _, next_infos = self.env.step(actions)

                if agent.is_transitioning:
                    state = agent.current_sumo_state
                    observed_states.append(state)
                    if 'y' in state:
                        yellow_found = True
                    if 'u' in state:
                        red_yellow_found = True

                observations, infos = next_obs, next_infos

                if int(observations[jid]['phase'][0]) == target and not agent.is_transitioning:
                    break

            if not observed_states:
                result.fail("Nem volt átmeneti állapot")
            elif not yellow_found:
                result.fail(f"Nincs sárga ('y') az átmenetben: {observed_states}")
            else:
                unique_states = list(dict.fromkeys(observed_states))  # Sorrend megtartása
                result.success(f"Átmenetek: {unique_states}")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_08_min_green_timer(self):
        """Teszt: Min green timer működése."""
        result = TestResult("Min green timer")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            # Várjunk, amíg egy átmenet befejeződik
            target = (int(observations[jid]['phase'][0]) + 1) % agent.num_phases

            # READY állapot megvárása
            for _ in range(20):
                if infos[jid]['ready']:
                    break
                _, _, _, _, infos = self.env.step({})

            # Átmenet indítása
            for _ in range(30):
                next_obs, _, _, _, next_infos = self.env.step({jid: target})
                observations, infos = next_obs, next_infos

                if int(observations[jid]['phase'][0]) == target and not agent.is_transitioning:
                    break

            # Most ellenőrizzük a min_green_timer-t
            initial_timer = agent.min_green_timer

            if initial_timer <= 0:
                result.fail(f"min_green_timer nem inicializálódott: {initial_timer}")
            else:
                # Lépkedjünk és figyeljük a csökkenést
                timer_values = [initial_timer]
                for _ in range(initial_timer + 2):
                    self.env.step({jid: target})  # Maradjunk ugyanabban a fázisban
                    timer_values.append(agent.min_green_timer)

                # Ellenőrzés: monoton csökkenő 0-ig
                decreasing = all(timer_values[i] >= timer_values[i+1] for i in range(len(timer_values)-1))
                reaches_zero = 0 in timer_values

                if decreasing and reaches_zero:
                    result.success(f"Timer: {timer_values[:6]}... -> 0")
                else:
                    result.fail(f"Timer nem csökken helyesen: {timer_values}")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_09_ready_busy_states(self):
        """Teszt: READY/BUSY állapotok."""
        result = TestResult("READY/BUSY állapotok")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]
            agent = self.env.agents[jid]

            ready_states = []
            busy_states = []

            # Váltás és megfigyelés
            target = (int(observations[jid]['phase'][0]) + 1) % agent.num_phases

            for step in range(40):
                is_ready = infos[jid]['ready']

                if is_ready:
                    ready_states.append(step)
                else:
                    busy_states.append(step)

                actions = {jid: target} if is_ready else {}
                next_obs, _, _, _, next_infos = self.env.step(actions)
                observations, infos = next_obs, next_infos

            if not ready_states:
                result.fail("Soha nem volt READY")
            elif not busy_states:
                result.fail("Soha nem volt BUSY (átmenet alatt kéne)")
            else:
                result.success(f"READY: {len(ready_states)}x, BUSY: {len(busy_states)}x")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_10_reward_calculation(self):
        """Teszt: Reward számítás."""
        result = TestResult("Reward számítás")

        try:
            observations, infos = self.env.reset()
            jid = list(self.env.agents.keys())[0]

            rewards = []
            for _ in range(20):
                _, step_rewards, _, _, _ = self.env.step({})
                rewards.append(step_rewards[jid])

            if all(r == 0 for r in rewards):
                # Ez lehet OK, ha nincs forgalom
                result.success(f"Reward értékek: mind 0 (nincs forgalom?)")
            elif any(np.isnan(r) or np.isinf(r) for r in rewards):
                result.fail(f"NaN vagy Inf reward: {rewards}")
            else:
                result.success(f"Reward tartomány: [{min(rewards):.2f}, {max(rewards):.2f}]")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_11_multiple_agents(self):
        """Teszt: Több ágens párhuzamos működése."""
        result = TestResult("Több ágens párhuzamos")

        try:
            observations, infos = self.env.reset()

            if len(self.env.agents) < 2:
                result.success("Csak 1 ágens, teszt kihagyva")
                self._add_result(result)
                return

            # Minden ágensnek random akciót adunk
            for step in range(20):
                actions = {}
                for jid, agent in self.env.agents.items():
                    if infos[jid]['ready']:
                        actions[jid] = (int(observations[jid]['phase'][0]) + 1) % agent.num_phases

                next_obs, rewards, terminated, truncated, next_infos = self.env.step(actions)

                # Ellenőrzés: minden ágensnek van observation és reward
                if set(next_obs.keys()) != set(self.env.agents.keys()):
                    result.fail(f"Hiányzó observation kulcsok: step {step}")
                    self._add_result(result)
                    return

                if set(rewards.keys()) != set(self.env.agents.keys()):
                    result.fail(f"Hiányzó reward kulcsok: step {step}")
                    self._add_result(result)
                    return

                observations, infos = next_obs, next_infos

            result.success(f"{len(self.env.agents)} ágens szinkronban")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_12_parallel_independent_transitions(self):
        """Teszt: Ágensek egymástól függetlenül váltanak fázist."""
        result = TestResult("Független párhuzamos átmenetek")

        try:
            observations, infos = self.env.reset()

            if len(self.env.agents) < 3:
                result.success("Kevés ágens, teszt kihagyva")
                self._add_result(result)
                return

            # Válasszunk 3 ágenst
            agent_ids = list(self.env.agents.keys())[:3]

            # Várjuk meg, hogy READY legyenek
            for _ in range(20):
                all_ready = all(infos[jid]['ready'] for jid in agent_ids)
                if all_ready:
                    break
                _, _, _, _, infos = self.env.step({})

            # Mindegyiknek különböző célfázist adunk
            targets = {}
            for i, jid in enumerate(agent_ids):
                agent = self.env.agents[jid]
                current = int(observations[jid]['phase'][0])
                # Mindegyik másik fázisba megy
                targets[jid] = (current + i + 1) % agent.num_phases

            # Futtatás és követés
            transition_counts = {jid: 0 for jid in agent_ids}
            phase_reached = {jid: False for jid in agent_ids}

            for step in range(50):
                actions = {}
                for jid in agent_ids:
                    if infos[jid]['ready']:
                        actions[jid] = targets[jid]

                next_obs, _, _, _, next_infos = self.env.step(actions)

                # Átmenetek számlálása
                for jid in agent_ids:
                    agent = self.env.agents[jid]
                    if agent.is_transitioning:
                        transition_counts[jid] += 1

                    current = int(next_obs[jid]['phase'][0])
                    if current == targets[jid] and not agent.is_transitioning:
                        phase_reached[jid] = True

                observations, infos = next_obs, next_infos

                # Ha mindenki elérte a célt, kész
                if all(phase_reached.values()):
                    break

            errors = []
            for jid in agent_ids:
                if not phase_reached[jid]:
                    errors.append(f"{jid}: nem érte el a {targets[jid]} fázist")
                if transition_counts[jid] == 0:
                    errors.append(f"{jid}: nem volt átmenete")

            if errors:
                result.fail("Párhuzamos átmenet hibák", errors)
            else:
                result.success(f"3 ágens függetlenül váltott: {transition_counts}")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_13_agents_different_states_same_time(self):
        """Teszt: Ágensek különböző állapotokban lehetnek egyszerre."""
        result = TestResult("Különböző állapotok egyidőben")

        try:
            observations, infos = self.env.reset()

            if len(self.env.agents) < 2:
                result.success("Kevés ágens, teszt kihagyva")
                self._add_result(result)
                return

            agent_ids = list(self.env.agents.keys())[:4]

            # Első ágensnek váltást indítunk, másiknak nem
            # Várjuk meg READY-t
            for _ in range(20):
                if all(infos[jid]['ready'] for jid in agent_ids):
                    break
                _, _, _, _, infos = self.env.step({})

            # Csak az első ágensnek adunk akciót
            first_jid = agent_ids[0]
            first_agent = self.env.agents[first_jid]
            current = int(observations[first_jid]['phase'][0])
            target = (current + 1) % first_agent.num_phases

            found_different_states = False

            for step in range(30):
                # Csak az elsőnek adunk váltási parancsot
                actions = {first_jid: target}

                next_obs, _, _, _, next_infos = self.env.step(actions)

                # Ellenőrizzük: az első BUSY (átmenetben), a többi READY?
                first_transitioning = self.env.agents[first_jid].is_transitioning
                others_ready = [next_infos[jid]['ready'] for jid in agent_ids[1:]]

                if first_transitioning and any(others_ready):
                    found_different_states = True
                    break

                observations, infos = next_obs, next_infos

            if found_different_states:
                result.success("Ágensek különböző READY/BUSY állapotban")
            else:
                result.fail("Nem sikerült különböző állapotokat elérni")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_14_no_state_interference(self):
        """Teszt: Egy ágens váltása nem befolyásolja a másik belső állapotát."""
        result = TestResult("Nincs állapot-interferencia")

        try:
            observations, infos = self.env.reset()

            if len(self.env.agents) < 2:
                result.success("Kevés ágens, teszt kihagyva")
                self._add_result(result)
                return

            agent_ids = list(self.env.agents.keys())[:2]
            jid_a, jid_b = agent_ids[0], agent_ids[1]
            agent_a = self.env.agents[jid_a]
            agent_b = self.env.agents[jid_b]

            # Várjuk meg READY-t
            for _ in range(20):
                if infos[jid_a]['ready'] and infos[jid_b]['ready']:
                    break
                _, _, _, _, infos = self.env.step({})

            # Jegyezzük meg B állapotát
            b_phase_before = int(observations[jid_b]['phase'][0])
            b_sumo_state_before = agent_b.current_sumo_state

            # A-nak váltást indítunk
            current_a = int(observations[jid_a]['phase'][0])
            target_a = (current_a + 1) % agent_a.num_phases

            # Lépkedünk amíg A átmenetben van
            errors = []
            for step in range(20):
                actions = {jid_a: target_a}  # Csak A-nak adunk akciót!
                next_obs, _, _, _, next_infos = self.env.step(actions)

                # B változatlan marad?
                b_phase_now = int(next_obs[jid_b]['phase'][0])

                # B nem kapott akciót, tehát a fázisa nem változhat
                # (kivéve ha véletlenül pont ő is READY lett és random akciót kapott - de mi nem adtunk neki)
                if b_phase_now != b_phase_before and not agent_b.is_transitioning:
                    # B READY és nem váltott - OK
                    pass

                observations, infos = next_obs, next_infos

                if not agent_a.is_transitioning:
                    break

            # Végső ellenőrzés: B nem indított átmenetet
            if agent_b.is_transitioning:
                errors.append("B ágens átmenetbe került anélkül, hogy akciót kapott volna")

            if errors:
                result.fail("Állapot-interferencia", errors)
            else:
                result.success("Ágensek állapota független maradt")

        except Exception as e:
            result.fail(f"Kivétel: {e}")

        self._add_result(result)

    def _test_15_all_agents_all_transitions(self):
        """Teszt: MINDEN ágens MINDEN fázisváltása tesztelve."""
        result = TestResult("Összes ágens összes átmenete")

        try:
            total_agents = len(self.env.agents)
            total_transitions_tested = 0
            total_transitions_passed = 0
            failed_transitions = []

            print(f"\n  {YELLOW}[...] Összes ágens tesztelése ({total_agents} db)...{RESET}")

            # Egyszer generálunk forgalmat, utána kikapcsoljuk
            observations, infos = self.env.reset()  # Ez generál forgalmat
            self.env.random_traffic = False  # Többet ne generáljon

            for agent_idx, (jid, agent) in enumerate(self.env.agents.items()):
                num_phases = agent.num_phases

                # Összes lehetséges átmenet ehhez az ágenshez
                transitions_for_agent = []
                for from_phase in range(num_phases):
                    for to_phase in range(num_phases):
                        if from_phase != to_phase:
                            transitions_for_agent.append((from_phase, to_phase))

                agent_passed = 0
                agent_failed = []

                for from_phase, to_phase in transitions_for_agent:
                    total_transitions_tested += 1

                    # Reset környezet (forgalom generálás nélkül)
                    observations, infos = self.env.reset()

                    # 1. Várjuk meg READY-t
                    for _ in range(30):
                        if infos[jid]['ready']:
                            break
                        _, _, _, _, infos = self.env.step({})

                    if not infos[jid]['ready']:
                        agent_failed.append(f"{from_phase}->{to_phase}: nem lett READY")
                        continue

                    # 2. Menjünk from_phase-re (ha nem ott vagyunk)
                    current = int(observations[jid]['phase'][0])
                    if current != from_phase:
                        for _ in range(50):
                            actions = {jid: from_phase} if infos[jid]['ready'] else {}
                            next_obs, _, _, _, next_infos = self.env.step(actions)
                            observations, infos = next_obs, next_infos

                            current = int(observations[jid]['phase'][0])
                            if current == from_phase and infos[jid]['ready']:
                                break
                        else:
                            agent_failed.append(f"{from_phase}->{to_phase}: nem tudott {from_phase}-ra menni")
                            continue

                    # 3. Váltsunk to_phase-re
                    transition_started = False
                    yellow_seen = False
                    reached_target = False

                    for step in range(60):
                        actions = {jid: to_phase} if infos[jid]['ready'] else {}
                        next_obs, _, _, _, next_infos = self.env.step(actions)

                        if agent.is_transitioning:
                            transition_started = True
                            if 'y' in agent.current_sumo_state:
                                yellow_seen = True

                        observations, infos = next_obs, next_infos

                        current = int(observations[jid]['phase'][0])
                        if current == to_phase and not agent.is_transitioning:
                            reached_target = True
                            break

                    # Értékelés
                    if not transition_started:
                        agent_failed.append(f"{from_phase}->{to_phase}: átmenet nem indult")
                    elif not reached_target:
                        agent_failed.append(f"{from_phase}->{to_phase}: nem érte el a célt")
                    elif not yellow_seen:
                        agent_failed.append(f"{from_phase}->{to_phase}: nem volt sárga")
                    else:
                        agent_passed += 1
                        total_transitions_passed += 1

                # Ágens összesítés
                status = "✓" if len(agent_failed) == 0 else "✗"
                print(f"       [{agent_idx+1:2d}/{total_agents}] {jid}: {agent_passed}/{len(transitions_for_agent)} átmenet OK {status}")

                if agent_failed:
                    failed_transitions.extend([f"{jid}: {f}" for f in agent_failed[:3]])  # Max 3 hiba/ágens
                    if len(agent_failed) > 3:
                        failed_transitions.append(f"{jid}: ... és még {len(agent_failed)-3} hiba")

            # Végeredmény
            if total_transitions_passed == total_transitions_tested:
                result.success(f"{total_transitions_passed}/{total_transitions_tested} átmenet OK ({total_agents} ágens)")
            else:
                fail_count = total_transitions_tested - total_transitions_passed
                result.fail(
                    f"{total_transitions_passed}/{total_transitions_tested} OK, {fail_count} FAIL",
                    failed_transitions[:10]  # Max 10 hiba kiírása
                )

        except Exception as e:
            import traceback
            result.fail(f"Kivétel: {e}")
            traceback.print_exc()

        self._add_result(result)

    # =========================================================================
    # ÖSSZESÍTÉS
    # =========================================================================

    def _print_summary(self):
        """Eredmények összesítése."""
        print("\n" + "="*60)
        print(f"   {CYAN}ÖSSZESÍTÉS{RESET}")
        print("="*60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"\n  Összes teszt: {total}")
        print(f"  {GREEN}Sikeres: {passed}{RESET}")
        print(f"  {RED}Sikertelen: {failed}{RESET}")

        if failed > 0:
            print(f"\n  {RED}Sikertelen tesztek:{RESET}")
            for r in self.results:
                if not r.passed:
                    print(f"    - {r.name}: {r.message}")

        print("\n" + "="*60)

        if failed == 0:
            print(f"  {GREEN}✓ MINDEN TESZT SIKERES!{RESET}")
        else:
            print(f"  {RED}✗ {failed} TESZT SIKERTELEN{RESET}")

        print("="*60 + "\n")


def main():
    tester = SumoRLTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
