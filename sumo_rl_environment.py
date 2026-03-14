import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
import sys
import subprocess
import random

# Global traci placeholder
traci = None

class SumoRLEnvironment(gym.Env):
    """
    Egyszerűsített, Single-Instance Multi-Agent RL Környezet.
    """
    def __init__(self,
                 net_file,
                 logic_json_file,
                 detector_file,
                 route_file="random_traffic.rou.xml",
                 reward_weights={'waiting': 1.0, 'co2': 1.0},
                 min_green_time=5,
                 delta_time=5,
                 measure_during_transition=False,
                 sumo_gui=False,
                 random_traffic=True,
                 traffic_period=1.0,
                 traffic_duration=3600,
                 flow_range=(100, 900),
                 statistic_output_file=None,
                 single_agent_id=None,
                 run_id=None):

        self.net_file = net_file
        self.logic_json_file = logic_json_file
        self.detector_file = detector_file

        # Unique route file handling
        self.run_id = run_id
        if self.run_id:
            base, ext = os.path.splitext(route_file)
            self.route_file = f"{base}_{self.run_id}{ext}"
        else:
            self.route_file = route_file

        self.statistic_output_file = statistic_output_file

        self.reward_weights = reward_weights
        self.min_green_time = min_green_time
        self.delta_time = int(delta_time)
        self.measure_during_transition = measure_during_transition
        self.sumo_gui = sumo_gui
        self.random_traffic = random_traffic
        self.traffic_period = traffic_period
        self.traffic_duration = traffic_duration
        self.flow_range = tuple(flow_range)
        self.single_agent_id = single_agent_id

        # --- Import Logic for GUI vs Headless ---
        global traci

        # Döntési logika:
        #   GUI mód         → mindig traci (libsumo nem támogat GUI-t)
        #   USE_LIBSUMO=1   → libsumo (env var-ral kényszeríthető)
        #   USE_LIBSUMO=0   → traci  (env var-ral tiltható, pl. parallel 2+ futásnál)
        #   default (nincs env var) → libsumo próba, fallback traci
        #
        # Párhuzamos futásnál a libsumo singleton → csak 1 process használhatja.
        # A start.sh az első futásnak USE_LIBSUMO=1-et ad, a többinek USE_LIBSUMO=0-t.
        env_libsumo = os.environ.get('USE_LIBSUMO', None)

        if self.sumo_gui:
            # GUI → kizárólag traci
            print(f"[INFO] Using 'traci' (GUI mode).")
            import traci as t
            traci = t
        elif env_libsumo == '0':
            # Explicit tiltás (parallel 2+ futás)
            print(f"[INFO] Using 'traci' (USE_LIBSUMO=0, parallel slave).")
            import traci as t
            traci = t
        else:
            # Default: libsumo próba (USE_LIBSUMO=1 vagy nincs beállítva)
            if traci is None or traci.__name__ != 'libsumo':
                try:
                    import libsumo as t
                    traci = t
                    print(f"[INFO] Using 'libsumo' (fast headless mode, OS={sys.platform}).")
                except ImportError:
                    print(f"[INFO] 'libsumo' not available, falling back to 'traci'.")
                    import traci as t
                    traci = t

        with open(self.logic_json_file, 'r') as f:
            self.logic_data = json.load(f)

        self.junction_ids = list(self.logic_data.keys())

        # Ágensek létrehozása
        if self.single_agent_id:
             if self.single_agent_id in self.junction_ids:
                 self.junction_ids = [self.single_agent_id]
             else:
                 print(f"[WARNING] Single agent ID {self.single_agent_id} not found in logic file.")
                 print(f"WARNING: Single agent ID {self.single_agent_id} not found in logic file. Using all agents.")
        
        self.agents = {
            jid: TrafficAgent(
                jid, 
                self.logic_data[jid], 
                min_green_time=self.min_green_time,
                measure_during_transition=self.measure_during_transition
            )
            for jid in self.junction_ids
        }

        # --- Action Logging Setup (Only in GUI mode) ---
        if self.sumo_gui:
            self.log_dir = "temp_action_logs"
            if os.path.exists(self.log_dir):
                import shutil
                shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Init files with headers
            for jid in self.agents.keys():
                with open(os.path.join(self.log_dir, f"{jid}.csv"), "w") as f:
                    f.write("Step,Action\n")
            print(f"[INFO] Action logging enabled in {self.log_dir}")
        
        self.observation_space = None
        self.action_space = None
        self.is_running = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 0. Szimuláció hossz rotáció — minden epizódban más időtartam
        #    Reális forgalmi időszakok: 15 perc → 2 óra
        #    Rövid epizódok: gyorsabb feedback, de zajosabb metrikák
        #    Hosszú epizódok: stabilabb átlagok, lassabb tanulás
        DURATION_OPTIONS = [900, 1800, 2700, 3600, 5400, 7200]  # 15min, 30min, 45min, 1h, 1.5h, 2h
        if options and 'traffic_duration' in options:
            self.traffic_duration = int(options['traffic_duration'])
        else:
            self.traffic_duration = random.choice(DURATION_OPTIONS)
        print(f"[INFO] Episode duration: {self.traffic_duration}s ({self.traffic_duration/60:.0f} min)")

        # 1. Random forgalom generálás
        if self.random_traffic:
            # flow_range: constructor default, felülírható options-ből
            flow_range = self.flow_range
            if options and 'flow_range' in options:
                flow_range = options['flow_range']

            if self.single_agent_id:
                # Focused traffic: period-alapú, ha explicit kérés van
                if options and 'traffic_period' in options:
                    dynamic_period = options['traffic_period']
                else:
                    dynamic_period = round(random.uniform(4.0, 8.0), 4)
                self.generate_focused_traffic(period=dynamic_period)
            else:
                self.generate_random_traffic(flow_range=flow_range)

        # 2. SUMO parancs összeállítása
        sumo_bin = "sumo-gui" if self.sumo_gui else "sumo"
        sumo_args = [
            "-n", self.net_file,
            "-r", self.route_file,
            "-a", self.detector_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--ignore-route-errors", "true",
            "--random", "true",
            "--no-warnings", "true",
            "--xml-validation", "never"
        ]

        if self.statistic_output_file:
            sumo_args.extend([
                "--statistic-output", self.statistic_output_file,
                "--duration-log.statistics", "true"
            ])

        # 3. SUMO indítása
        if self.is_running:
            try:
                traci.load(sumo_args)
            except Exception as e:
                # Ha a load nem sikerül (pl. connection closed), próbáljuk újraindítani
                print(f"Hiba a traci.load hívásakor: {e}. Újraindítás...")
                try:
                    traci.close()
                except:
                    pass
                traci.start([sumo_bin] + sumo_args)
        else:
            try:
                traci.close()
            except Exception:
                pass
            traci.start([sumo_bin] + sumo_args)
            self.is_running = True

        # Hálózat elemeinek feltérképezése és FÁZIS HOSSZ JAVÍTÁSA
        self._map_network_elements()
        
        # 4. Terek beállítása
        if self.observation_space is None:
             self.observation_space = spaces.Dict({
                 jid: self.agents[jid].observation_space for jid in self.agents.keys()
             })
             self.action_space = spaces.Dict({
                 jid: self.agents[jid].action_space for jid in self.agents.keys()
             })

        # 5. Warm-up
        warmup_seconds = random.randint(50, 300)
        if options and 'warmup_seconds' in options:
            warmup_seconds = options['warmup_seconds']

        for _ in range(warmup_seconds):
            for agent in self.agents.values():
                agent.collect_measurements() 
                agent.update_logic()
                if agent.is_ready_for_action():
                     agent.set_target_phase(random.randint(0, agent.num_phases - 1))
            traci.simulationStep()

        for agent in self.agents.values():
            agent.reset_step_metrics()

        observations = {jid: agent.get_observation() for jid, agent in self.agents.items()}
        infos = {jid: {"ready": agent.is_ready_for_action()} for jid in self.junction_ids}

        return observations, infos

    def generate_random_traffic(self, period=None, flow_range=(100, 900)):
        """
        Lane-szintű forgalom generálás: minden bejövő lane-re külön-külön
        generál járműveket a flow_range tartományban (jármű/óra).

        Sumolib-bal meghatározza az összes bejövő lane-t a vezérelt
        junction-ökhöz, és mindegyikre véletlenszerű forgalmat ír.
        A célpont az adott lane-ről elérhető kimenő edge-ek közül
        véletlenszerűen választott, departLane-nel rögzítve a sávot.
        """
        import sumolib

        try:
            net = sumolib.net.readNet(self.net_file)
        except Exception as e:
            print(f"[ERROR] Nem sikerult beolvasni a halozatot: {e}")
            return

        # 1. Összegyűjtjük az összes bejövő lane-t és a célpontjaikat
        #    lane_routes: {lane_id: (edge_id, lane_index, [to_edge_id, ...])}
        lane_routes = {}

        for jid in self.junction_ids:
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

        # 2. Járművek generálása lane-enként
        all_trips = []
        min_flow, max_flow = flow_range
        total_vehicles = 0

        for lane_id, (edge_id, lane_idx, to_edges) in lane_routes.items():
            flow_per_hour = random.randint(min_flow, max_flow)
            num_vehicles = int(flow_per_hour * (self.traffic_duration / 3600.0))

            if num_vehicles <= 0:
                continue

            total_vehicles += num_vehicles

            avg_gap = self.traffic_duration / num_vehicles
            for i in range(num_vehicles):
                depart = i * avg_gap + random.uniform(0, avg_gap * 0.5)
                depart = min(depart, self.traffic_duration - 1.0)
                to_edge = random.choice(to_edges)
                all_trips.append((depart, edge_id, lane_idx, to_edge))

        # 3. Rendezés indulási idő szerint és kiírás
        all_trips.sort(key=lambda x: x[0])

        with open(self.route_file, 'w') as f:
            f.write('<routes>\n')
            f.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" '
                    'length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
            for idx, (depart, from_e, lane_idx, to_e) in enumerate(all_trips):
                f.write(f'    <trip id="veh_{idx}" type="car" depart="{depart:.2f}" '
                        f'from="{from_e}" to="{to_e}" departLane="{lane_idx}" />\n')
            f.write('</routes>\n')

        print(f"[INFO] Lane-szintu forgalom generalas kesz | "
              f"{len(lane_routes)} lane | {total_vehicles} jarmu | "
              f"Tartomany: {min_flow}-{max_flow}/h lane-enkent")

    def generate_focused_traffic(self, period):
        print(f"\n[DEBUG] 🎯 Célzott Forgalom Generálása ({self.single_agent_id}) | Period: {period}")
        
        if 'SUMO_HOME' in os.environ:
             tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
             sys.path.append(tools)
        else:
             try:
                 import sumolib
                 tools = os.path.join(os.path.dirname(sumolib.__file__), '..', '..', 'tools')
                 sys.path.append(tools)
             except:
                 print("Sumolib not found cannot generate focused traffic.")
                 return

        try:
             import sumolib
             net = sumolib.net.readNet(self.net_file)
        except Exception as e:
             print(f"Failed to load net with sumolib: {e}")
             return

        node = net.getNode(self.single_agent_id)
        if not node:
             print(f"Node {self.single_agent_id} not found in network.")
             return

        valid_routes = []
        incoming_edges = node.getIncoming()
        for inc_edge in incoming_edges:
             outgoing_connections = inc_edge.getOutgoing()
             for conn in outgoing_connections:
                 # Filter connections that actually go through our node logic
                 # Usually connections from incoming edges of a node go through that node.
                 # Check if the connection 'via' or 'to' makes sense.
                 if hasattr(conn, 'getTo'):
                     out_edge = conn.getTo()
                 else:
                     # It's already an edge
                     out_edge = conn
                 
                 valid_routes.append((inc_edge.getID(), out_edge.getID()))

        if not valid_routes:
             print("No valid routes found for focused traffic.")
             self.generate_random_traffic(period)
             return

        print(f"Found {len(valid_routes)} valid OD pairs for {self.single_agent_id}")

        with open(self.route_file, 'w') as f:
             f.write('<routes>\n')
             f.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
             
             # Direct usage of period as requested by user for high density
             if period <= 1e-6: period = 0.0001
             print(f"Generating focused traffic: Period {period:.4f}s")
             
             vehicle_count = int(self.traffic_duration / period)
             trips = []
             for i in range(vehicle_count):
                 depart = i * period + random.uniform(0, period * 0.1)
                 # Avoid generating beyond duration
                 if depart > self.traffic_duration: break
                 
                 route = random.choice(valid_routes)
                 trips.append((depart, route))
             
             trips.sort(key=lambda x: x[0])
             
             for depart, route in trips:
                 f.write(f'    <trip id="veh_single_{depart:.2f}" type="car" depart="{depart:.2f}" from="{route[0]}" to="{route[1]}" />\n')
             
             f.write('</routes>\n')

    def _map_network_elements(self):
        all_loops = traci.inductionloop.getIDList()
        
        for jid, agent in self.agents.items():
            # --- JAVÍTÁS KEZDETE: Fázis string hossz ellenőrzése és javítása ---
            try:
                # Lekérjük a SUMO-tól az aktuális fázist, hogy lássuk a helyes hosszt
                current_state_sumo = traci.trafficlight.getRedYellowGreenState(jid)
                correct_len = len(current_state_sumo)
                
                # Végigmegyünk az ágens tárolt fázisain
                for p_idx, p_data in agent.phase_registry.items():
                    current_len = len(p_data['state'])
                    if current_len != correct_len:
                        print(f"WARNING: Phase size mismatch for {jid} (Phase {p_idx}). "
                              f"JSON: {current_len}, SUMO: {correct_len}. Auto-fixing...")
                        
                        if current_len > correct_len:
                            # Ha túl hosszú a JSON string, levágjuk a végét
                            p_data['state'] = p_data['state'][:correct_len]
                        else:
                            # Ha túl rövid, kiegészítjük 'r' (piros) karakterekkel
                            p_data['state'] = p_data['state'].ljust(correct_len, 'r')
            except Exception as e:
                print(f"Error checking TLS state for {jid}: {e}")
            # --- JAVÍTÁS VÉGE ---

            agent.reset_step_metrics()
            agent.reset_logic()

            controlled_links = traci.trafficlight.getControlledLinks(jid)
            incoming_lanes_set = set()
            for link_group in controlled_links:
                for link in link_group:
                    if link: incoming_lanes_set.add(link[0])
            agent.incoming_lanes = list(incoming_lanes_set)

            temp_detectors = []
            for loop in all_loops:
                lane_id = traci.inductionloop.getLaneID(loop)
                if lane_id in incoming_lanes_set:
                    temp_detectors.append(loop)

            agent.detectors = sorted(temp_detectors)
            agent.setup_spaces()
            agent.reset_step_metrics()

    def step(self, actions):
        if not self.is_running:
            raise RuntimeError("Szimuláció nem fut.")

        for jid, action in actions.items():
            self.agents[jid].set_target_phase(int(action))
            
            # --- Log Action (Only in GUI mode) ---
            if self.sumo_gui:
                try:
                    step_num = 0
                    if traci:
                        try:
                            step_num = traci.simulation.getTime()
                        except:
                            pass
                            
                    with open(os.path.join(self.log_dir, f"{jid}.csv"), "a") as f:
                        f.write(f"{step_num},{int(action)}\n")
                except Exception as e:
                    print(f"[WARNING] Failed to log action for {jid}: {e}")

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
        dones = {}
        infos = {}
        
        sim_terminated = (traci.simulation.getMinExpectedNumber() <= 0)
        sim_truncated = (traci.simulation.getTime() >= self.traffic_duration)
        global_done = sim_terminated or sim_truncated

        w_wait = self.reward_weights.get('waiting', 1.0)
        w_co2  = self.reward_weights.get('co2', 1.0)

        # Log-Sigmoid normalizálási paraméterek
        # Forrás: PCA elemzés 527,747 mintából, 10 flow szint (300-1200 veh/h/lane),
        #         3 epizód × 21 junction, RL-kompatibilis akkumulált átlagok
        #         (delta_time substepeken gyűjtve, steps_measured-del osztva)
        #         Lásd: metric_pca_test_v2/
        #
        # Metrika: TotalWaitingTime = Σ lane.getWaitingTime / steps [sec/step]
        #   PCA PC1 loading: -0.3797 (domináns torlódási dimenzió, 48.5%)
        #   medián: 120.20 sec, p5: 3.20 sec, p95: 1340.40 sec
        MU_WAIT = 4.584800   # E[log(TotalWaitingTime)]
        STD_WAIT = 1.824900  # Std[log(TotalWaitingTime)]
        #
        # Metrika: TotalCO2 = Σ lane.getCO2Emission / steps [mg/s átlag]
        #   PCA PC1 loading: -0.3409 (korrelál torlódással, önálló környezeti dimenzió)
        #   medián: 70,129 mg/s, p5: 8,383 mg/s, p95: 173,545 mg/s
        MU_CO2 = 10.870600   # E[log(TotalCO2)]
        STD_CO2 = 0.962900   # Std[log(TotalCO2)]

        for jid, agent in self.agents.items():
            observations[jid] = agent.get_observation()

            # WaitingTime: Σ lane waiting time / steps → átlag sec/step (= pillanatérték skála)
            avg_waiting = agent.get_avg_waiting_time_metric()
            # CO2: nyers mg/s skálán (NEM /1000!), hogy a PCA paraméterek konzisztensek legyenek
            avg_co2_raw = (agent.accumulated_co2 / agent.steps_measured) if agent.steps_measured > 0 else 0.0

            if avg_waiting == 0.0 and avg_co2_raw == 0.0:
                # Üres kereszteződés: sem várakozás, sem CO2 → semleges reward (0.0)
                rewards[jid] = 0.0
            else:
                z_wait = (np.log(avg_waiting + 1e-5) - MU_WAIT) / (STD_WAIT + 1e-9)
                z_co2  = (np.log(avg_co2_raw + 1e-5) - MU_CO2)  / (STD_CO2  + 1e-9)

                score_wait = 1 / (1 + np.exp(-z_wait))
                score_co2  = 1 / (1 + np.exp(-z_co2))

                r_wait = 1.0 - score_wait  # alacsony waiting → magas reward
                r_co2  = 1.0 - score_co2   # alacsony CO2 → magas reward

                # Súlyozott átlag → reward mindig [0, 1] tartományban
                w_sum = w_wait + w_co2
                rewards[jid] = (w_wait * r_wait + w_co2 * r_co2) / w_sum

            infos[jid] = {
                "ready": agent.is_ready_for_action(),
                "metric_waiting_time": avg_waiting,
                "metric_co2_raw": avg_co2_raw,
            }
            dones[jid] = global_done

        return observations, rewards, global_done, False, infos

    def close(self):
        if self.is_running:
            try:
                traci.close()
            except:
                pass
            self.is_running = False


class TrafficAgent:
    def __init__(self, jid, logic_data, min_green_time, measure_during_transition):
        self.jid = jid
        self.min_green_const = min_green_time
        
        self.phase_registry = {}
        if 'phases' in logic_data:
            for p in logic_data['phases']:
                self.phase_registry[p['index']] = {'state': p['state'], 'duration': float(p['duration'])}
        
        self.logic_phases = {int(k): v for k, v in logic_data['logic_phases'].items()}
        self.transitions = logic_data['transitions']
        self.num_phases = len(self.logic_phases)

        self.detectors = []
        self.incoming_lanes = []
        self.det_accumulated_flow = {}
        self.det_accumulated_occ = {}
        self.reset_logic()
        self.reset_step_metrics()
        
        self.action_space = None
        self.observation_space = None

    def reset_logic(self):
        self.current_logic_idx = 0
        self.target_logic_idx = 0
        self.is_transitioning = False
        self.transition_queue = []
        self.transition_cursor = 0
        self.transition_step_timer = 0
        self.min_green_timer = 0
        self.current_sumo_state = "??????"
        self.phase_timer = 0  # Hány lépése tart már az aktuális fázis

    def setup_spaces(self):
        num_detectors = len(self.detectors)
        # Ha valamiért 0 detektor lenne, kezeljük le
        if num_detectors == 0:
            num_detectors = 1 # Dummy dimenzió, hogy ne crasheljen a Box space
            
        self.action_space = spaces.Discrete(self.num_phases)
        self.observation_space = spaces.Dict({
            "phase": spaces.Box(low=0, high=self.num_phases-1, shape=(1,), dtype=np.int32),
            "phase_timer": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "occupancy": spaces.Box(low=0, high=1, shape=(num_detectors,), dtype=np.float32),
            "flow": spaces.Box(low=0, high=np.inf, shape=(num_detectors,), dtype=np.float32)
        })

    def reset_step_metrics(self):
        self.steps_measured = 0
        self.det_accumulated_flow = {d: 0 for d in self.detectors}
        self.det_accumulated_occ = {d: 0.0 for d in self.detectors}
        self.accumulated_travel_time = 0.0
        self.accumulated_waiting_time = 0.0
        self.accumulated_co2 = 0.0
        self.accumulated_veh_count = 0

    def is_ready_for_action(self):
        return (not self.is_transitioning) and (self.min_green_timer <= 0)

    def set_target_phase(self, target_idx):
        if self.is_ready_for_action():
            if target_idx in self.logic_phases:
                self.target_logic_idx = target_idx

    def update_logic(self):
        if self.is_transitioning:
            if self.transition_step_timer > 0:
                self.transition_step_timer -= 1
            else:
                if self.transition_cursor < len(self.transition_queue):
                    sumo_idx = self.transition_queue[self.transition_cursor]
                    if sumo_idx in self.phase_registry:
                        p = self.phase_registry[sumo_idx]
                        self.current_sumo_state = p['state']
                        try:
                            traci.trafficlight.setRedYellowGreenState(self.jid, p['state'])
                        except Exception as e:
                            print(f"Error setting TLS state for {self.jid}: {e}")
                        self.transition_step_timer = max(0, int(p['duration']) - 1)
                    self.transition_cursor += 1
                else:
                    self.is_transitioning = False
                    self.current_logic_idx = self.next_logic_idx_cache
                    self._apply_current_phase()
                    self.min_green_timer = self.min_green_const
                    self.phase_timer = 0  # Fázisváltáskor reset
        else:
            if self.min_green_timer > 0:
                self.min_green_timer -= 1
                self._apply_current_phase()
                self.phase_timer += 1
            else:
                if self.target_logic_idx != self.current_logic_idx:
                    self._start_transition(self.target_logic_idx)
                else:
                    self._apply_current_phase()
                    self.phase_timer += 1

    def _apply_current_phase(self):
        if self.current_logic_idx in self.logic_phases:
            sumo_idx = self.logic_phases[self.current_logic_idx]
            if sumo_idx in self.phase_registry:
                state = self.phase_registry[sumo_idx]['state']
                self.current_sumo_state = state
                try:
                    traci.trafficlight.setRedYellowGreenState(self.jid, state)
                except Exception as e:
                    print(f"Error setting TLS state for {self.jid}: {e}")

    def _start_transition(self, next_idx):
        key = f"{self.current_logic_idx}->{next_idx}"
        self.transition_queue = self.transitions.get(key, [])
        self.is_transitioning = True
        self.transition_cursor = 0
        self.transition_step_timer = 0
        self.next_logic_idx_cache = next_idx

    def collect_measurements(self):
        self.steps_measured += 1
        for det_id in self.detectors:
            self.det_accumulated_flow[det_id] += traci.inductionloop.getLastStepVehicleNumber(det_id)
            self.det_accumulated_occ[det_id] += traci.inductionloop.getLastStepOccupancy(det_id)
            
        step_tt = 0.0
        step_waiting = 0.0
        step_co2 = 0.0
        step_veh_count = 0
        valid_tt_lanes = 0
        MAX_REALISTIC_TT = 1000  # Egy ~90m savon max ~1000s realis (extrem dugo)
        for lane in self.incoming_lanes:
            tt = traci.lane.getTraveltime(lane)
            if 0 < tt < MAX_REALISTIC_TT:  # Csak valodi, mert ertekeket akkumulalunk
                step_tt += tt
                valid_tt_lanes += 1
            step_waiting += traci.lane.getWaitingTime(lane)  # sec, pillanatnyi összes várakozás
            step_co2 += traci.lane.getCO2Emission(lane)      # mg/s, pillanatnyi kibocsátás
            step_veh_count += traci.lane.getLastStepVehicleNumber(lane)
        # Ha csak sentinel értékek jöttek, ne adjunk hozzá semmit
        if valid_tt_lanes > 0:
            self.accumulated_travel_time += step_tt
        self.accumulated_waiting_time += step_waiting
        self.accumulated_co2 += step_co2
        self.accumulated_veh_count += step_veh_count

    def get_observation(self):
        num_dets = len(self.detectors)
        if num_dets == 0:
             # Fallback ha nincs detektor
             return {
                "phase": np.array([self.current_logic_idx], dtype=np.int32),
                "phase_timer": np.array([0.0], dtype=np.float32),
                "occupancy": np.zeros(1, dtype=np.float32),
                "flow": np.zeros(1, dtype=np.float32)
             }
             
        occ = np.zeros(num_dets, dtype=np.float32)
        flow = np.zeros(num_dets, dtype=np.float32)
        if self.steps_measured > 0:
            for i, det in enumerate(self.detectors):
                occ[i] = (self.det_accumulated_occ[det] / self.steps_measured) / 100.0
                flow[i] = (self.det_accumulated_flow[det] / self.steps_measured)
        
        # phase_timer normalizálva [0, 1] tartományba
        # 120 lépés (600 mp delta_time=5 esetén) a max referencia
        normalized_timer = min(self.phase_timer / 120.0, 1.0)
        
        return {
            "phase": np.array([self.current_logic_idx], dtype=np.int32),
            "phase_timer": np.array([normalized_timer], dtype=np.float32),
            "occupancy": occ,
            "flow": flow
        }

    def get_avg_travel_time_metric(self):
        return (self.accumulated_travel_time / self.steps_measured) if self.steps_measured > 0 else 0.0

    def get_avg_waiting_time_metric(self):
        """Átlagos várakozási idő a delta_time alatt [sec].
        traci.lane.getWaitingTime() → pillanatnyi összes várakozás (sec) a lane-en.
        Akkumulálva steps_measured lépésen, elosztva → átlag sec/step."""
        return (self.accumulated_waiting_time / self.steps_measured) if self.steps_measured > 0 else 0.0

    def get_total_co2_metric(self):
        """Átlagos CO2 kibocsátási ráta a delta_time alatt [g/s].
        traci.lane.getCO2Emission() → mg/s pillanatnyi.
        Akkumulálva, /1000 → g/s, /steps → átlag."""
        return ((self.accumulated_co2 / 1000.0) / self.steps_measured) if self.steps_measured > 0 else 0.0