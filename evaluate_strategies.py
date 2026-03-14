import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Adjust path to import sumo_rl_environment if needed
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from sumo_rl_environment import SumoRLEnvironment
    from rl_trainer import IndependentDQNTrainer # Might need this for ONNX loading, or we can use the env directly if it supports it, actually rl_trainer is better for loading the model. No, rl_trainer has the IndependentDQNTrainer which manages the env. Let's see.
except ImportError:
    print("Error: Could not import sumo_rl_environment. Ensure it is in the Python path.")
    sys.exit(1)

class EvaluationRunner:
    def __init__(self, net_file, detector_file, logic_file, fixed_tls_file, output_dir="eval_results"):
        self.net_file = net_file
        self.detector_file = detector_file
        self.logic_file = logic_file
        self.fixed_tls_file = fixed_tls_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        
        # Determine paths
        self.base_dir = os.path.dirname(self.net_file)
        self.scenarios = {
            "Low": {"period": 18.0, "duration": 3600},
            "Peak": {"period": 4.5, "duration": 3600},
            "Fluctuating": {"duration": 3600}
        }
        
        # Tools path
        if 'SUMO_HOME' in os.environ:
             self.tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
        else:
             import sumolib
             self.tools_path = os.path.join(os.path.dirname(sumolib.__file__), '..', '..', 'tools')

    def generate_scenarios(self, target_junction):
        print("Starting Scenario Generation...")
        import sumolib
        net = sumolib.net.readNet(self.net_file)
        node = net.getNode(target_junction)
        valid_routes = []
        for inc_edge in node.getIncoming():
             for conn in inc_edge.getOutgoing():
                 out_edge = conn.getTo() if hasattr(conn, 'getTo') else conn
                 valid_routes.append((inc_edge.getID(), out_edge.getID()))
                 
        if not valid_routes:
             print("No valid routes for focused traffic!")
             return

        for sc_name, sc_data in self.scenarios.items():
            route_file = os.path.join(self.base_dir, f"eval_{sc_name}_traffic.rou.xml")
            sc_data['route_file'] = route_file
            
            with open(route_file, 'w') as f:
                f.write('<routes>\n')
                f.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>\n')
                
                trips = []
                if sc_name in ["Low", "Peak"]:
                    period = sc_data["period"]
                    veh_count = int(sc_data["duration"] / period)
                    for i in range(veh_count):
                        depart = i * period + random.uniform(0, period * 0.1)
                        if depart <= sc_data["duration"]:
                            route = random.choice(valid_routes)
                            trips.append((depart, route))
                elif sc_name == "Fluctuating":
                    # 0-20m (1200s): Low (18s)
                    # 20-40m (2400s): Peak (4.5s)
                    # 40-60m (3600s): Low (18s)
                    for phase, (start_t, end_t, period) in enumerate([(0, 1200, 18.0), (1200, 2400, 4.5), (2400, 3600, 18.0)]):
                        veh_count = int((end_t - start_t) / period)
                        for i in range(veh_count):
                            depart = start_t + i * period + random.uniform(0, period * 0.1)
                            route = random.choice(valid_routes)
                            trips.append((depart, route))
                            
                trips.sort(key=lambda x: x[0])
                for depart, route in trips:
                    f.write(f'    <trip id="veh_{sc_name}_{depart:.2f}" type="car" depart="{depart:.2f}" from="{route[0]}" to="{route[1]}" />\n')
                
                f.write('</routes>\n')
            print(f"Generated {sc_name} scenario (Total trips: {len(trips)}).")
        
        
    def run_strategy(self, scenario_name, strategy_name, target_junction, model_path=None):
        print(f"\n--- Running strategy '{strategy_name}' on scenario '{scenario_name}' ---")
        sc_data = self.scenarios[scenario_name]
        route_file = sc_data['route_file']
        
        # Prepare additional files (for Fixed Baseline)
        add_files = self.detector_file
        if strategy_name == "Fixed_Baseline" and self.fixed_tls_file and os.path.exists(self.fixed_tls_file):
             add_files += "," + self.fixed_tls_file

        # Init environment
        env = SumoRLEnvironment(
            net_file=self.net_file,
            logic_json_file=self.logic_file,
            detector_file=add_files,
            route_file=route_file,
            random_traffic=False,
            traffic_duration=sc_data['duration'],
            single_agent_id=target_junction,
        )
        
        obs, info = env.reset()
        
        # RL Model setup
        rl_agent = None
        if strategy_name.startswith("RL"):
            if not model_path or not os.path.exists(model_path):
                 print(f"Warning: Model path {model_path} not found. Skipping.")
                 env.close()
                 return
            
            # Simple ONNX loading logic via stable baselines or direct
            try:
                from sb3_contrib import QRDQN
                # We need net_arch. For simplicity, assume [64,64] or use the trainer logic.
                # Since we are decoupling, let's instantiate a default agent and load weights via numpy
                # This duplicates rl_trainer logic but avoids tangled dependencies
                rl_agent = QRDQN("MultiInputPolicy", env, verbose=0, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
                
                # Manual ONNX weight loading
                import onnx
                from onnx import numpy_helper
                import torch
                model_proto = onnx.load(model_path, load_external_data=False)
                onnx_weights = [numpy_helper.to_array(init) for init in model_proto.graph.initializer]
                
                target_sd = rl_agent.policy.state_dict()
                target_keys = [k for k in target_sd.keys() if "weight" in k or "bias" in k]
                used_idx = set()
                
                for k in target_keys:
                    target_param = target_sd[k]
                    t_shape = tuple(target_param.shape)
                    for i, w in enumerate(onnx_weights):
                        if i in used_idx: continue
                        w_shape = tuple(w.shape)
                        if w_shape == t_shape:
                             with torch.no_grad(): target_param.copy_(torch.from_numpy(w))
                             used_idx.add(i)
                             break
                        elif len(w_shape)==2 and len(t_shape)==2 and w_shape[::-1] == t_shape:
                             with torch.no_grad(): target_param.copy_(torch.from_numpy(w.T))
                             used_idx.add(i)
                             break
                print(f"ONNX Model {model_path} loaded directly.")
            except Exception as e:
                print(f"Failed to load RL model: {e}")
                env.close()
                return

        # TraCI Setup for baselines
        import traci
        if strategy_name == "Actuated":
             try:
                 traci.trafficlight.setProgram(target_junction, "actuated")
                 print(f"Switched TLS {target_junction} to 'actuated' program.")
             except Exception as e:
                 print(f"Warning: Failed to switch to 'actuated' program via TraCI ({e}). "
                       "Ensure an actuated program is defined in your network or add.xml.")
        elif strategy_name == "Fixed_Baseline":
             try:
                 # If the user defines a custom baseline program in add.xml, switch to it to avoid conflicts.
                 # E.g. programID="fixed_baseline", but we'll try "fixed_baseline" and fallback to "0".
                 programs = traci.trafficlight.getProgram(target_junction)
                 traci.trafficlight.setProgram(target_junction, "fixed_baseline")
                 print(f"Switched TLS {target_junction} to custom 'fixed_baseline' program.")
             except Exception as e:
                 # If "fixed_baseline" doesn't exist, it means they might have just overwritten "0".
                 pass

        # Metrics lists
        step_waiting_times = []
        step_queue_lengths = []
        step_co2 = []
        phase_alignments = []

        step = 0
        done = False
        
        while not done:
             # Decide action
             action_dict = {}
             if strategy_name.startswith("RL"):
                  if info[target_junction].get('ready', True):
                      action, _ = rl_agent.predict(obs[target_junction], deterministic=True)
                      action_dict[target_junction] = int(action)
                      
                      # Alignment logic: Was a transition queued?
                      # We can peek at the env agent state
                      agent_state = env.agents[target_junction]
                      if agent_state.is_transitioning:
                           phase_alignments.append(0) # Not aligned (delay needed)
                      else:
                           phase_alignments.append(1) # Aligned (instant switch/stay)
             else:
                  # For Fixed/Actuated, we want SUMO to control the lights, NOT the env!
                  # So we bypass the env's setTargetPhase and let TraCI do its thing.
                  # But env.step() forces current_sumo_state. 
                  pass 
             
             # Step simulation
             # If Fixed/Actuated, we need to bypass RL environment's phase forcing
             if strategy_name in ["Fixed_Baseline", "Actuated"]:
                  # Step manually in traci to preserve SUMO's native logic
                  traci.simulationStep()
                  done = (traci.simulation.getMinExpectedNumber() <= 0) or (traci.simulation.getTime() >= sc_data['duration'])
             else:
                  obs, rewards, done, _, info = env.step(action_dict)
             
             # Collect Metrics
             # Average Waiting Time and Max Queue
             lanes = env.agents[target_junction].incoming_lanes
             if lanes:
                 wait_t = sum(traci.lane.getWaitingTime(l) for l in lanes) / len(lanes)
                 queue_l = max(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                 co2_e = sum(traci.lane.getCO2Emission(l) for l in lanes)
                 
                 step_waiting_times.append(wait_t)
                 step_queue_lengths.append(queue_l)
                 step_co2.append(co2_e)
                 
                 if scenario_name == "Fluctuating" and (strategy_name == "Actuated" or strategy_name.startswith("RL_Finetuned_")):
                      if not hasattr(self, 'queue_history'):
                           self.queue_history = {}
                      if strategy_name not in self.queue_history:
                           self.queue_history[strategy_name] = []
                      self.queue_history[strategy_name].append(queue_l)
             
             step += 1

        env.close()
        
        # Aggregate results
        mean_wait = np.mean(step_waiting_times) if step_waiting_times else 0.0
        max_queue = np.max(step_queue_lengths) if step_queue_lengths else 0.0
        total_co2 = np.sum(step_co2) if step_co2 else 0.0
        align_idx = np.mean(phase_alignments) if phase_alignments else 1.0

        self.results.append({
             "Scenario": scenario_name,
             "Strategy": strategy_name,
             "Mean_Waiting_Time": mean_wait,
             "Max_Queue_Length": max_queue,
             "Total_CO2_Emissions": total_co2,
             "Phase_Alignment_Index": align_idx
        })
        print(f"Done. Wait: {mean_wait:.2f}s, Queue: {max_queue:.1f}, CO2: {total_co2:.1f}")

    def evaluate_all(self, target_junction, models):
        self.generate_scenarios(target_junction)
        scenarios = ["Low", "Peak", "Fluctuating"]
        
        for scenario in scenarios:
            self.run_strategy(scenario, "Fixed_Baseline", target_junction)
            self.run_strategy(scenario, "Actuated", target_junction)
            self.run_strategy(scenario, "RL_Pretrained", target_junction, model_path=models.get('0k'))
            
            for steps in ['10k', '25k', '50k', '75k', '100k']:
                if models.get(steps):
                    self.run_strategy(scenario, f"RL_Finetuned_{steps}", target_junction, model_path=models.get(steps))
                    
        self.save_results()
        self.generate_plots()

    def save_results(self):
        df = pd.DataFrame(self.results)
        output_path = os.path.join(self.output_dir, "results_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    def generate_plots(self):
        print("Generating plots...")
        df = pd.DataFrame(self.results)
        if df.empty: return

        # 1. Performance Trend (Mean Waiting Time vs Fine-tuning Steps)
        plt.figure(figsize=(10, 6))
        for scenario in df['Scenario'].unique():
            sc_df = df[df['Scenario'] == scenario]
            
            # Extract RL data points
            rl_steps = []
            rl_waits = []
            strategies_to_plot = ['RL_Pretrained', 'RL_Finetuned_10k', 'RL_Finetuned_25k', 'RL_Finetuned_50k', 'RL_Finetuned_75k', 'RL_Finetuned_100k']
            labels_to_plot = ['0k', '10k', '25k', '50k', '75k', '100k']
            for s, label in zip(strategies_to_plot, labels_to_plot):
                row = sc_df[sc_df['Strategy'] == s]
                if not row.empty:
                    rl_steps.append(label)
                    rl_waits.append(row['Mean_Waiting_Time'].values[0])
            
            if rl_steps:
                plt.plot(rl_steps, rl_waits, marker='o', label=f'RL ({scenario})')
            
            # Add baselines as horizontal lines
            fixed_row = sc_df[sc_df['Strategy'] == 'Fixed_Baseline']
            if not fixed_row.empty:
                plt.axhline(y=fixed_row['Mean_Waiting_Time'].values[0], linestyle='--', label=f'Fixed ({scenario})', alpha=0.5)
                
            actuated_row = sc_df[sc_df['Strategy'] == 'Actuated']
            if not actuated_row.empty:
                plt.axhline(y=actuated_row['Mean_Waiting_Time'].values[0], linestyle=':', label=f'Actuated ({scenario})', alpha=0.5)

        plt.title("Performance Trend: Mean Waiting Time vs Fine-Tuning")
        plt.xlabel("Fine-tuning Steps")
        plt.ylabel("Mean Waiting Time (s)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_trend.png"))
        plt.close()

        # 2. Demand Sensitivity (Grouped Bar: CO2)
        plt.figure(figsize=(12, 6))
        scenarios = df['Scenario'].unique()
        strategies = df['Strategy'].unique()
        
        x = np.arange(len(scenarios))
        width = 0.15
        
        for i, strat in enumerate(strategies):
            co2_vals = []
            for sc in scenarios:
                row = df[(df['Scenario'] == sc) & (df['Strategy'] == strat)]
                co2_vals.append(row['Total_CO2_Emissions'].values[0] if not row.empty else 0)
            
            offset = (i - len(strategies)/2) * width + width/2
            plt.bar(x + offset, co2_vals, width, label=strat)

        plt.xlabel('Traffic Demand Scenario')
        plt.ylabel('Total CO2 Emissions (kg)')
        plt.title('Demand Sensitivity: CO2 Emissions across Scenarios')
        plt.xticks(x, scenarios)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demand_sensitivity.png"))
        plt.close()

        # 3. Queue Line Plot (Queue Length over time for Fluctuating)
        if hasattr(self, 'queue_history'):
            plt.figure(figsize=(12, 6))
            for strat, q_hist in self.queue_history.items():
                # Smooth the data for better visualization (moving average over 60 steps)
                window = 60
                smoothed = np.convolve(q_hist, np.ones(window)/window, mode='valid')
                plt.plot(smoothed, label=strat)
                
            plt.title("Queue Length Over Time (Fluctuating Scenario)")
            plt.xlabel("Simulation Steps (Smoothed)")
            plt.ylabel("Max Queue Length (Vehicles)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "queue_heatmap.png")) # Using line plot as it's clearer
            plt.close()
            
        print("Plots generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Evaluation Framework for Transfer Learning")
    parser.add_argument("--net", required=True, help="Path to network file (.net.xml)")
    parser.add_argument("--logic", required=True, help="Path to logic file (.json)")
    parser.add_argument("--detector", required=True, help="Path to detector file (.xml)")
    parser.add_argument("--fixed-tls", required=True, help="Path to external TLS definition for Fixed baseline (.add.xml)")
    parser.add_argument("--junction", required=True, help="Target Junction ID")
    parser.add_argument("--model-0k", help="Path to pre-trained ONNX model (0k steps)")
    parser.add_argument("--model-10k", help="Path to fine-tuned ONNX model (10k steps)")
    parser.add_argument("--model-25k", help="Path to fine-tuned ONNX model (25k steps)")
    parser.add_argument("--model-50k", help="Path to fine-tuned ONNX model (50k steps)")
    parser.add_argument("--model-75k", help="Path to fine-tuned ONNX model (75k steps)")
    parser.add_argument("--model-100k", help="Path to fine-tuned ONNX model (100k steps)")
    
    args = parser.parse_args()
    
    models = {
        '0k': args.model_0k,
        '10k': args.model_10k,
        '25k': args.model_25k,
        '50k': args.model_50k,
        '75k': args.model_75k,
        '100k': args.model_100k
    }
    
    runner = EvaluationRunner(
        net_file=args.net,
        detector_file=args.detector,
        logic_file=args.logic,
        fixed_tls_file=args.fixed_tls
    )
    runner.evaluate_all(args.junction, models)
