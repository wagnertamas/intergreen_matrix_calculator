import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import sys
import subprocess
import time
import random

# libsumo használata (gyorsabb, nincs socket overhead)
# A felhasználó kérésére CSAK libsumo-t használunk.
try:
    import libsumo as traci
    USE_LIBSUMO = True
except ImportError:
    raise ImportError("Hiba: A 'libsumo' modul nem található! Kérlek állítsd be helyesen a PYTHONPATH-t vagy a SUMO_HOME-ot.")


class SumoRLEnvironment(gym.Env):
    """
    Aszinkron Multi-Agent RL Környezet (Dict Observation Space).
    JSON Struktúra: Index-alapú lookup a 'phases' listából.
    libsumo-t használ a traci helyett (gyorsabb, nincs retry).
    """
    def __init__(self,
                 net_file,
                 logic_json_file,
                 detector_file,
                 route_file="random_traffic.rou.xml",
                 reward_weights={'time': 1.0, 'co2': 1.0},
                 min_green_time=5,
                 delta_time=1,
                 measure_during_transition=False,
                 sumo_gui=False,
                 random_traffic=True,
                 traffic_period=1.0,

                 traffic_duration=3600,
                 statistic_output_file=None): # [NEW]

        self.net_file = net_file
        self.logic_json_file = logic_json_file
        self.detector_file = detector_file
        self.route_file = route_file
        self.statistic_output_file = statistic_output_file # [NEW]

        self.reward_weights = reward_weights
        self.min_green_time = min_green_time
        self.delta_time = int(delta_time)
        self.measure_during_transition = measure_during_transition
        self.sumo_gui = sumo_gui
        self.random_traffic = random_traffic
        self.traffic_period = traffic_period
        self.traffic_duration = traffic_duration

        with open(self.logic_json_file, 'r') as f:
            self.logic_data = json.load(f)

        self.junction_ids = list(self.logic_data.keys())

        self.agents = {
            jid: TrafficAgent(jid, self.logic_data[jid], min_green_time, measure_during_transition)
            for jid in self.junction_ids
        }
        
        # [VEC ENV COMPATIBILITY]
        # VecEnv (Subproc/Dummy) requires 'observation_space' and 'action_space' to be defined in __init__.
        # Often we need SUMO running to know the exact spaces (number of detectors).
        # We will ESTIMATE them by parsing the detector file manually here.
        self._estimate_spaces_from_xml()

        self.is_running = False

    def _estimate_spaces_from_xml(self):
        """
        [FIX] Instead of guessing from XML (which causes mismatch errors),
        we perform a 'Dry Run' simulation to measure exact spaces.
        This ensures SubprocVecEnv receives the CORRECT observation_space.
        """
        print("[Env Init] Measuring Observation Spaces via short boot...")
        
        # 1. Setup Boot Command
        # Use dummy route file if original doesn't exist yet (in parallel it might not)
        temp_route = self.route_file
        if not os.path.exists(temp_route):
             # Create empty route file just for boot
             with open(temp_route, 'w') as f:
                 f.write('<routes></routes>')
        
        sumo_cmd = [
            "-n", self.net_file,
            "-r", temp_route,
            "-a", self.detector_file,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--random", "true" # Avoid seed conflicts
        ]
        
        try:
             # 2. Start Simulation
             traci.start(sumo_cmd)
             
             # 3. Use normal map logic
             self._map_network_elements()
             
             # 4. Spaces are now set in self.agents
             # Build Global Spaces
             self.observation_space = spaces.Dict({jid: agent.observation_space for jid in self.agents.keys()})
             self.action_space = spaces.Dict({jid: agent.action_space for jid in self.agents.keys()})
             
             # 5. Close
             traci.close()
             print(f"[Env Init] Spaces measured: {self.observation_space}")
             
        except Exception as e:
             print(f"[Critical] Failed to measure spaces: {e}")
             # Raise error because training WILL fail if mismatch occurs
             raise e

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ha fut, leállítjuk
        if self.is_running:
            try:
                traci.close()
            except:
                pass
            self.is_running = False

        if self.random_traffic:
            # Dynamic Traffic Generation for Robustness
            # Allow override via options (e.g. for testing/debugging)
            if options and 'traffic_period' in options:
                dynamic_period = options['traffic_period']
            else:
                # Randomize traffic desnity (period) to expose agent to sparse and dense traffic.
                # Range: 0.001 (Max) to 0.005 (Min).
                dynamic_period = round(random.uniform(0.001, 0.005), 4)
            self.generate_random_traffic(period=dynamic_period)

        # SUMO parancs összeállítása (libsumo nem támogatja a GUI-t!)
        if self.sumo_gui:
            print("FIGYELEM: libsumo nem támogatja a sumo-gui-t, sumo lesz használva.")

        sumo_cmd = [
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.detector_file,
            # "--threads", "4", # REMOVED: Unsafe/Warning in SUMO 1.25
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--ignore-route-errors", "true",
            "--random", "true",
            "--no-warnings", "true",
            "--xml-validation", "never"
        ]

        # [NEW] Add statistics export options if a file path is provided
        if self.statistic_output_file:
            sumo_cmd.extend([
                "--statistic-output", self.statistic_output_file,
                "--duration-log.statistics", "true"
            ])

        # libsumo: load() használata (nincs label, nincs socket)
        traci.load(sumo_cmd)

        self.is_running = True

        self._map_network_elements()

        # Warm-up Phase
        # Allow override via options (e.g. for testing)
        warmup_seconds = random.randint(180, 600)
        if options and 'warmup_seconds' in options:
            warmup_seconds = options['warmup_seconds']
            
        # warmup_seconds = 0 # Uncomment to disable for debugging
        print(f"[WARMUP] Warming up for {warmup_seconds} seconds ({warmup_seconds} steps)...")
        
        for _ in range(warmup_seconds):
            # Step agents randomly
            for jid, agent in self.agents.items():
                agent.collect_measurements() # IMPORTANT: Update internal timers (min_green, transition)
                agent.update_logic()
                
                if agent.is_ready_for_action():
                     # Pick Random Phase
                     # self.action_space is defined as Discrete(num_phases) in setup_spaces.
                     # So efficient random choice:
                     rnd_action = random.randint(0, agent.num_phases - 1)
                     agent.set_target_phase(rnd_action)
            
            traci.simulationStep()
            
        print(f"[WARMUP] Completed.")

        # Ágensek inicializálása (Reset metrics AFTER warmup)
        for agent in self.agents.values():
            # Keep logic state but reset metrics
            # agent.current_logic_idx = ... # Don't reset phase state, stick to where we are
            agent.reset_step_metrics()
            # We don't want to reset 'is_transitioning' etc because we might be mid-transition!
            # Just reset the reward accumulators.

        observations = {jid: agent.get_observation() for jid, agent in self.agents.items()}
        infos = {jid: {"ready": agent.is_ready_for_action()} for jid in self.junction_ids}

        return observations, infos

    def generate_random_traffic(self, period=None):
        p = period if period is not None else self.traffic_period
        # Use simple string formatting to avoid truncation of small periods like 0.0005
        print(f"Forgalom generálása folyamatban (Period={p})...")
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            try:
                import sumolib
                tools = os.path.join(os.path.dirname(sumolib.__file__), '..', '..', 'tools')
                tools = os.path.abspath(tools)
            except:
                print("HIBA: Nincs SUMO_HOME.")
                return

        random_trips_script = os.path.join(tools, "randomTrips.py")
        cmd = [
            sys.executable, random_trips_script,
            "-n", self.net_file,
            "-o", self.route_file,
            "-e", str(self.traffic_duration),
            "-p", str(p),
            "--fringe-factor", "10",
            "--validate",
            "--min-distance", "50",
            "--duarouter-routing-algorithm", "CH",
            "--duarouter-routing-threads", "6",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"Kész: {self.route_file}")
        except subprocess.CalledProcessError as e:
            print(f"HIBA a randomTrips futtatásakor:\n{e.stderr.decode()}")

    def _map_network_elements(self):
        all_loops = traci.inductionloop.getIDList()

        for jid, agent in self.agents.items():
            agent.reset_step_metrics()
            agent.reset_logic()  # [FIX] Reset internal logic state (e.g. queue, timer) to sync with fresh SUMO

            temp_detectors = []
            agent.incoming_lanes = []

            controlled_links = traci.trafficlight.getControlledLinks(jid)
            incoming_lanes_set = set()
            for link_group in controlled_links:
                for link in link_group:
                    incoming_lanes_set.add(link[0])
            agent.incoming_lanes = list(incoming_lanes_set)

            for loop in all_loops:
                parts = loop.split("_", 1)
                if len(parts) > 1 and parts[1] in incoming_lanes_set:
                    temp_detectors.append(loop)

            agent.detectors = sorted(temp_detectors)
            agent.setup_spaces()
            agent.reset_step_metrics()

    def step(self, actions):
        if not self.is_running:
            raise RuntimeError("Szimuláció nem fut.")

        for jid, action in actions.items():
            self.agents[jid].set_target_phase(int(action))

        for agent in self.agents.values():
            agent.reset_step_metrics()

        for _ in range(self.delta_time):
            for agent in self.agents.values():
                agent.update_logic()
            traci.simulationStep()
            for agent in self.agents.values():
                agent.collect_measurements()

        observations = {}
        # Multi-Agent Rewards/Dones will be stored in info
        ma_rewards = {}
        ma_terminated = {}
        ma_truncated = {}
        
        infos = {}
        
        # Determine global done (if simulation ends)
        sim_terminated = not self.is_running
        sim_truncated = False

        w_time = self.reward_weights.get('time', 1.0)
        w_co2 = self.reward_weights.get('co2', 0.1)

        for jid, agent in self.agents.items():
            observations[jid] = agent.get_observation()
            
            avg_travel_time = agent.get_avg_travel_time_metric()
            total_co2 = agent.get_total_co2_metric()
            
            
            # [REWARD TRANSFORMATION] Log-Sigmoid (Log-Distance)
            # Based on PCA analysis constants provided by user:
            # MU_TIME = 11.211313, STD_TIME = 1.129861
            # MU_CO2 = 0.483506, STD_CO2 = 0.121195
            
            # NOTE: avg_travel_time and total_co2 are "Total per step" (Sum of vehicles), NOT normalized.
            # The PCA analysis was performed on these "Total" metrics?
            # From previous context, user reverted to Total metrics for reward. 
            # Assuming the constants match the scale of 'avg_travel_time' (which is actually Total Travel Time per step).
            
            MU_TIME  = 11.211313
            STD_TIME = 1.129861
            MU_CO2   = 0.483506
            STD_CO2  = 0.121195

            # 1. Z-score in Log Space
            # Add small epsilon to avoid log(0)
            z_time = (np.log(avg_travel_time + 1e-5) - MU_TIME) / (STD_TIME + 1e-9)
            z_co2  = (np.log(total_co2 + 1e-5) - MU_CO2)  / (STD_CO2  + 1e-9)
            
            # 2. Sigmoid (0 -> 1)
            # This maps the "badness" (accumulated cost) to a probability-like score.
            # Higher Z (more congestion) -> Score close to 1.
            # Lower Z (free flow) -> Score close to 0.
            score_time = 1 / (1 + np.exp(-z_time))
            score_co2  = 1 / (1 + np.exp(-z_co2))
            
            # 3. Reward (Maximize Quality)
            # We want to MAXIMIZE reward, so we invert the score (1 - Badness).
            # Best possible = 1.0 (Zero congestion)
            # Worst possible = 0.0 (Infinite congestion)
            r_time = 1.0 - score_time
            r_co2  = 1.0 - score_co2
            
            reward = w_time * r_time + w_co2 * r_co2
            ma_rewards[jid] = reward
            
            # Store raw metrics in info for logging
            infos[jid] = {
                "ready": agent.is_ready_for_action(),
                "metric_travel_time": avg_travel_time,
                "metric_co2": total_co2,
                "metric_veh_count": agent.accumulated_veh_count / max(1, agent.steps_measured) # Average vehicles seen per step
            }
            
            ma_terminated[jid] = False
            ma_truncated[jid] = False

        # [PARALLEL COMPATIBILITY]
        # VecEnv expects scalar Reward and scalar Done.
        global_reward = sum(ma_rewards.values()) # Just for VecEnv interface compliance
        
        # Check global termination
        if traci.simulation.getMinExpectedNumber() <= 0:
             sim_terminated = True
        if traci.simulation.getTime() >= self.traffic_duration:
             sim_truncated = True
             
        # Pack MA data into info for the Trainer to unpack
        combined_info = {
            "ma_rewards": ma_rewards,
            "ma_terminated": ma_terminated,
            "ma_truncated": ma_truncated,
            "agent_infos": infos
        }
        
        return observations, global_reward, sim_terminated, sim_truncated, combined_info

    def close(self):
        if self.is_running:
            try:
                traci.close()
            except:
                pass
            self.is_running = False


class TrafficAgent:
    """
    TrafficAgent JSON index-lookup logikával.
    Minden adatot (duration, state) a 'phases' listából olvas ki az index alapján.
    """
    def __init__(self, jid, logic_data, min_green_time, measure_during_transition):
        self.jid = jid
        self.min_green_const = min_green_time
        self.measure_during_transition = measure_during_transition

        # 1. 'phases' lista feldolgozása egy gyorskereső dictionary-be (Registry)
        # { index: {'state': "GGGrrr", 'duration': 5.0}, ... }
        self.phase_registry = {}
        if 'phases' in logic_data:
            for p in logic_data['phases']:
                idx = p['index']
                self.phase_registry[idx] = {
                    'state': p['state'],
                    'duration': float(p['duration'])
                }
        else:
            raise ValueError(f"JSON Error: '{jid}' csomópontnál hiányzik a 'phases' lista!")

        # 2. RL Action Mapping: logic_idx -> SUMO phase index (int)
        # Pl: "0": 0, "1": 4, ...
        self.logic_phases = {int(k): v for k, v in logic_data['logic_phases'].items()}

        # 3. Transitions: key -> list of SUMO phase indices (int)
        # Pl: "0->1": [1, 2, 3]
        self.transitions = logic_data['transitions']

        self.num_phases = len(self.logic_phases)

        self.reset_logic()

        # Aktuális SUMO állapot követése (debug célra)
        self.current_sumo_state = "??????"
        self.current_sumo_phase_idx = 0

        self.detectors = []
        self.incoming_lanes = []
        self.det_accumulated_flow = {}
        self.det_accumulated_occ = {}

        self.reset_step_metrics()
        self.next_logic_idx_cache = 0
        self.action_space = None
        self.observation_space = None

    def reset_logic(self):
        """Reset internal logic state to initial values."""
        self.current_logic_idx = 0
        self.target_logic_idx = 0
        self.is_transitioning = False
        self.transition_queue = []
        self.transition_cursor = 0
        self.transition_step_timer = 0
        self.min_green_timer = 0
        self.current_sumo_state = "??????"
        self.current_sumo_phase_idx = 0

    def setup_spaces(self):
        num_detectors = len(self.detectors)
        self.action_space = spaces.Discrete(self.num_phases)
        self.observation_space = spaces.Dict({
            "phase": spaces.Box(low=0, high=self.num_phases-1, shape=(1,), dtype=np.int32),
            "occupancy": spaces.Box(low=0, high=1, shape=(num_detectors,), dtype=np.float32),
            "flow": spaces.Box(low=0, high=np.inf, shape=(num_detectors,), dtype=np.float32)
        })

    def reset_step_metrics(self):
        self.steps_measured = 0
        self.det_accumulated_flow = {d: 0 for d in self.detectors}
        self.det_accumulated_occ = {d: 0.0 for d in self.detectors}
        self.accumulated_travel_time = 0.0
        self.accumulated_co2 = 0.0
        self.accumulated_veh_count = 0 # [NEW] Track total vehicles for normalization

    def is_ready_for_action(self):
        return (not self.is_transitioning) and (self.min_green_timer <= 0)

    def set_target_phase(self, target_idx):
        if self.is_ready_for_action():
            # Validáció: csak érvényes logic index fogadható el
            if target_idx in self.logic_phases:
                self.target_logic_idx = target_idx
            else:
                print(f"[WARN] {self.jid}: Érvénytelen target_idx={target_idx}, num_phases={self.num_phases}")

    def update_logic(self):
        """
        Frissített logika: Az indexek alapján kikeresi a state stringet és a durationt.
        """
        # 1. Átmenet (Transition)
        if self.is_transitioning:
            if self.transition_step_timer > 0:
                self.transition_step_timer -= 1
            else:
                # Következő fázis a sorban (ami egy Index)
                if self.transition_cursor < len(self.transition_queue):
                    sumo_phase_idx = self.transition_queue[self.transition_cursor]

                    # ADATOK KIKERESÉSE A REGISTRY-BŐL
                    if sumo_phase_idx in self.phase_registry:
                        phase_data = self.phase_registry[sumo_phase_idx]
                        state_str = phase_data['state']
                        duration = phase_data['duration']

                        # Lámpa állítása
                        traci.trafficlight.setRedYellowGreenState(self.jid, state_str)

                        # Aktuális állapot mentése (debug)
                        self.current_sumo_state = state_str
                        self.current_sumo_phase_idx = sumo_phase_idx

                        # Időzítő beállítása (-1, mert ez a step már 1 mp)
                        self.transition_step_timer = max(0, int(duration) - 1)
                    else:
                        print(f"Error: Ismeretlen fázis index {sumo_phase_idx} a {self.jid}-nél!")

                    self.transition_cursor += 1
                else:
                    # Átmenet vége -> Cél Zöld fázis
                    self.is_transitioning = False
                    self.current_logic_idx = self.next_logic_idx_cache

                    # Cél fázis indexe - ELLENŐRZÉS
                    if self.current_logic_idx not in self.logic_phases:
                        print(f"[ERROR] {self.jid}: current_logic_idx={self.current_logic_idx} nem található a logic_phases-ben!")
                        print(f"        logic_phases keys: {list(self.logic_phases.keys())}")
                        print(f"        next_logic_idx_cache: {self.next_logic_idx_cache}")
                        # Fallback: visszaállítás 0-ra
                        self.current_logic_idx = 0
                    target_sumo_idx = self.logic_phases[self.current_logic_idx]

                    # Kikeresés
                    if target_sumo_idx in self.phase_registry:
                        target_state = self.phase_registry[target_sumo_idx]['state']
                        traci.trafficlight.setRedYellowGreenState(self.jid, target_state)

                        # Aktuális állapot mentése (debug)
                        self.current_sumo_state = target_state
                        self.current_sumo_phase_idx = target_sumo_idx

                    self.min_green_timer = self.min_green_const

        # 2. Zöld állapot (Green Hold)
        else:
            # Mindig biztosítjuk, hogy a helyes lámpakép legyen kint
            if self.current_logic_idx not in self.logic_phases:
                print(f"[ERROR-GREEN] {self.jid}: current_logic_idx={self.current_logic_idx} nem található!")
                print(f"              logic_phases keys: {list(self.logic_phases.keys())}")
                self.current_logic_idx = 0
            target_sumo_idx = self.logic_phases[self.current_logic_idx]
            target_state = self.phase_registry[target_sumo_idx]['state']

            # Aktuális állapot mentése (debug)
            self.current_sumo_state = target_state
            self.current_sumo_phase_idx = target_sumo_idx

            if self.min_green_timer > 0:
                self.min_green_timer -= 1
                traci.trafficlight.setRedYellowGreenState(self.jid, target_state)
            else:
                if self.target_logic_idx != self.current_logic_idx:
                    self._start_transition(self.target_logic_idx)
                else:
                    traci.trafficlight.setRedYellowGreenState(self.jid, target_state)

    def _start_transition(self, next_idx):
        key = f"{self.current_logic_idx}->{next_idx}"
        if key in self.transitions:
            self.transition_queue = self.transitions[key]  # Ez egy lista indexekből: [1, 2, 3]
        else:
            self.transition_queue = []

        self.is_transitioning = True
        self.transition_cursor = 0
        self.transition_step_timer = 0
        self.next_logic_idx_cache = next_idx

    def collect_measurements(self):
        # [FIX] Always measure! Transition time counts for Total Travel Time.
        # if self.is_transitioning and not self.measure_during_transition:
        #    return

        self.steps_measured += 1



        self.steps_measured += 1

        for det_id in self.detectors:
            self.det_accumulated_flow[det_id] += traci.inductionloop.getLastStepVehicleNumber(det_id)
            self.det_accumulated_occ[det_id] += traci.inductionloop.getLastStepOccupancy(det_id)

        step_tt_sum = 0.0
        step_co2_sum = 0.0
        step_veh_count = 0
        
        for lane in self.incoming_lanes:
            # [REVERTED] User requested Estimated Travel Time for Transfer Learning context.
            # Using getTraveltime which returns (Lane Length / Mean Speed).
            # This is "Instantaneous Travel Time" for a hypothetical vehicle entering the lane.
            tt = traci.lane.getTraveltime(lane)
            veh_on_lane = traci.lane.getLastStepVehicleNumber(lane)
            
            # Weighted average by vehicle count? Or just sum for the lane?
            # User wants: "Travel Time... normalized".
            # If we sum travel times of all lanes, we get a huge number dependent on # of lanes.
            # We should probably average it?
            
            # Let's accumulate both sums and normalize in get_avg_metric.
            step_tt_sum += tt
            step_co2_sum += traci.lane.getCO2Emission(lane)
            step_veh_count += veh_on_lane
        
        self.accumulated_travel_time += step_tt_sum
        self.accumulated_co2 += step_co2_sum
        self.accumulated_veh_count += step_veh_count # Track total vehicles for normalization

    def get_observation(self):
        num_dets = len(self.detectors)
        occupancy_vec = np.zeros(num_dets, dtype=np.float32)
        flow_vec = np.zeros(num_dets, dtype=np.float32)

        if self.steps_measured > 0:
            for i, det_id in enumerate(self.detectors):
                occupancy_vec[i] = self.det_accumulated_occ[det_id] / self.steps_measured
                flow_vec[i] = self.det_accumulated_flow[det_id] / self.steps_measured

        occupancy_vec = occupancy_vec / 100.0

        return {
            "phase": np.array([self.current_logic_idx], dtype=np.int32),
            "occupancy": occupancy_vec,
            "flow": flow_vec
        }

    def get_avg_travel_time_metric(self):
        # [REVERTED AGAIN] User requested NO normalization by vehicle count.
        # Just return the accumulated Estimated Travel Time per step.
        if self.steps_measured == 0:
            return 0.0
        return self.accumulated_travel_time / self.steps_measured

    def get_total_co2_metric(self):
        if not self.incoming_lanes:
            return 0.0
        
        # [REVERTED AGAIN] User requested NO normalization by vehicle count.
        # Total CO2 per step.
        if self.steps_measured == 0:
            return 0.0
            
        return (self.accumulated_co2 / 1000.0) / self.steps_measured
