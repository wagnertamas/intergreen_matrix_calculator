import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
import numpy as np
import json
import os
import sys
import subprocess
import time

class SumoRLEnvironment(gym.Env):
    """
    Aszinkron Multi-Agent RL Környezet (Dict Observation Space).
    JSON Struktúra: Index-alapú lookup a 'phases' listából.
    """
    def __init__(self, 
                 net_file, 
                 logic_json_file, 
                 detector_file, 
                 route_file="random_traffic.rou.xml", 
                 reward_weights={'time': 1.0, 'co2': 0.1},
                 min_green_time=5,
                 delta_time=1,
                 measure_during_transition=False,
                 sumo_gui=False,
                 random_traffic=True,
                 traffic_period=1.0,
                 traffic_duration=3600):
        
        self.net_file = net_file
        self.logic_json_file = logic_json_file
        self.detector_file = detector_file
        self.route_file = route_file
        
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
        
        self.sumo_binary = "sumo-gui" if self.sumo_gui else "sumo"
        self.connection_label = f"sim_rl_{int(time.time())}"
        self.is_running = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.is_running:
            try: traci.close()
            except: pass
            self.is_running = False
        
        if self.random_traffic:
            self.generate_random_traffic()
        
        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.detector_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--ignore-route-errors", "true",
            "--random", "true"
        ]
        
        try:
            traci.start(sumo_cmd, label=self.connection_label)
        except traci.FatalTraCIError:
            try: traci.close()
            except: pass
            time.sleep(1)
            traci.start(sumo_cmd, label=self.connection_label)
            
        self.is_running = True
        
        self._map_network_elements()
        
        observations = {jid: agent.get_observation() for jid, agent in self.agents.items()}
        infos = {jid: {"ready": True} for jid in self.junction_ids}
        
        return observations, infos

    def generate_random_traffic(self):
        print("Forgalom generálása folyamatban...")
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
            "-p", str(self.traffic_period),
            "--fringe-factor", "10",
            "--validate",
            "--min-distance", "50"
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
        rewards = {}
        terminated = {jid: False for jid in self.junction_ids}
        truncated = {jid: False for jid in self.junction_ids}
        infos = {}

        w_time = self.reward_weights.get('time', 1.0)
        w_co2 = self.reward_weights.get('co2', 0.1)

        for jid, agent in self.agents.items():
            observations[jid] = agent.get_observation()
            infos[jid] = {"ready": agent.is_ready_for_action()}
            
            avg_travel_time = agent.get_avg_travel_time_metric()
            total_co2 = agent.get_total_co2_metric()
            
            reward = -1 * (w_time * avg_travel_time + w_co2 * total_co2)
            rewards[jid] = reward

        terminated["__all__"] = (traci.simulation.getMinExpectedNumber() <= 0)
        
        return observations, rewards, terminated, truncated, infos

    def close(self):
        traci.close()
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
        
        self.current_logic_idx = 0 
        self.target_logic_idx = 0
        
        self.is_transitioning = False
        self.transition_queue = [] # Ez most indexek listája lesz: [1, 2, 3]
        self.transition_cursor = 0
        self.transition_step_timer = 0 
        
        self.min_green_timer = 0
        
        self.detectors = [] 
        self.incoming_lanes = []
        self.det_accumulated_flow = {}
        self.det_accumulated_occ = {}
        
        self.reset_step_metrics()
        self.next_logic_idx_cache = 0
        self.action_space = None
        self.observation_space = None

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

    def is_ready_for_action(self):
        return (not self.is_transitioning) and (self.min_green_timer <= 0)

    def set_target_phase(self, target_idx):
        if self.is_ready_for_action():
            self.target_logic_idx = target_idx

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
                        
                        # Időzítő beállítása (-1, mert ez a step már 1 mp)
                        self.transition_step_timer = max(0, int(duration) - 1)
                    else:
                        print(f"Error: Ismeretlen fázis index {sumo_phase_idx} a {self.jid}-nél!")
                    
                    self.transition_cursor += 1
                else:
                    # Átmenet vége -> Cél Zöld fázis
                    self.is_transitioning = False
                    self.current_logic_idx = self.next_logic_idx_cache
                    
                    # Cél fázis indexe
                    target_sumo_idx = self.logic_phases[self.current_logic_idx]
                    
                    # Kikeresés
                    if target_sumo_idx in self.phase_registry:
                        target_state = self.phase_registry[target_sumo_idx]['state']
                        traci.trafficlight.setRedYellowGreenState(self.jid, target_state)
                    
                    self.min_green_timer = self.min_green_const

        # 2. Zöld állapot (Green Hold)
        else:
            # Mindig biztosítjuk, hogy a helyes lámpakép legyen kint
            target_sumo_idx = self.logic_phases[self.current_logic_idx]
            target_state = self.phase_registry[target_sumo_idx]['state']
            
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
            self.transition_queue = self.transitions[key] # Ez egy lista indexekből: [1, 2, 3]
        else:
            self.transition_queue = []
        
        self.is_transitioning = True
        self.transition_cursor = 0
        self.transition_step_timer = 0
        self.next_logic_idx_cache = next_idx

    def collect_measurements(self):
        if self.is_transitioning and not self.measure_during_transition:
            return

        self.steps_measured += 1
        
        for det_id in self.detectors:
            self.det_accumulated_flow[det_id] += traci.inductionloop.getLastStepVehicleNumber(det_id)
            self.det_accumulated_occ[det_id] += traci.inductionloop.getLastStepOccupancy(det_id)
            
        step_tt_sum = 0.0
        step_co2_sum = 0.0
        valid = 0
        for lane in self.incoming_lanes:
            step_tt_sum += traci.lane.getTraveltime(lane)
            step_co2_sum += traci.lane.getCO2Emission(lane)
            valid += 1
        if valid > 0:
            self.accumulated_travel_time += (step_tt_sum / valid)
        self.accumulated_co2 += step_co2_sum

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
        if self.steps_measured == 0: return 0.0
        return self.accumulated_travel_time / self.steps_measured

    def get_total_co2_metric(self):
        if not self.incoming_lanes: return 0.0
        return (self.accumulated_co2 / 1000.0) / len(self.incoming_lanes)