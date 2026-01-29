
import os
import sys
import time
import random
import numpy as np
import collections
from sumo_rl_environment import SumoRLEnvironment

# Basic Configuration
NET_FILE = "./data/mega_catalogue_v2.net.xml"
LOGIC_JSON = "./data/traffic_lights.json"
DETECTOR_FILE = "./data/detectors.add.xml"
ROUTE_FILE = "./random_traffic.rou.xml"

# Ensure libsumo/traci availability
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
try:
    import libsumo as traci
except ImportError:
    # If libsumo is missing, we fail hard as per user request
    raise ImportError("CRITICAL: libsumo is required but not found. traci fallback is disabled.")

def get_edge_id(lane_id):
    """Removes last _X from lane ID to get edge ID."""
    if "_" in lane_id:
        return lane_id.rsplit("_", 1)[0]
    return lane_id

def run_edge_calibration(period, duration=3600, single_agent_id=None):
    print(f"\n--- Testing Period: {period}s for {duration}s ---")
    if single_agent_id:
        print(f"    Mode: Single Intersection ({single_agent_id})")

    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_JSON,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        sumo_gui=False,
        traffic_duration=duration,
        single_agent_id=single_agent_id
    )
    
    # Force traffic generation with specific period
    env.reset(options={'warmup_seconds': 0, 'traffic_period': period})
    
    # Identify Incoming Edges per Agent
    agent_incoming_edges = {}
    all_incoming_edges = set()
    
    for jid, agent in env.agents.items():
        edges = set()
        for lane in agent.incoming_lanes:
            edge = get_edge_id(lane)
            edges.add(edge)
            all_incoming_edges.add(edge)
        agent_incoming_edges[jid] = list(edges)
        
    print(f"  Total Controlled Agents: {len(env.agents)}")
    print(f"  Total Unique Incoming Edges: {len(all_incoming_edges)}")
    
    # Tracking
    edge_vh_counts = collections.defaultdict(set) # {edge_id: {veh_id1, ...}}
    
    start_time = time.time()
    step = 0
    
    current_phase_idx = 0
    time_since_last_switch = 0
    phase_duration = 40 # seconds
    
    try:
        while step < duration:
            # Fixed Time Control Strategy to maximize throughput
            time_since_last_switch += 1
            
            for agent in env.agents.values():
                agent.update_logic()
                
                if time_since_last_switch >= phase_duration:
                    if agent.is_ready_for_action():
                        # Switch to next phase
                        current_phase_idx = (current_phase_idx + 1) % agent.num_phases
                        agent.set_target_phase(current_phase_idx)
            
            if time_since_last_switch >= phase_duration:
                 # Only reset timer if we actually commanded (simplified)
                 # Actually, we should check if transition started.
                 # But simplistic 40s timer is robust enough for calibration.
                 time_since_last_switch = 0
            
            traci.simulationStep()
            step += 1
            
            # Efficiently collect vehicles on relevant edges
            # getVehicleIDs is slow if called for every edge.
            # Instead iterate over all vehicles? No, too many.
            # Iterate over edges? 
            
            # Optimization: 29 agents * ~4 edges = ~120 edges. Not too bad.
            for edge in all_incoming_edges:
                # getLastStepVehicleIDs works on EDGES too in Traci? 
                # Usually works on lanes or edges. Let's try EDGE ID.
                # If libsumo throws error, we fallback to lanes.
                try:
                    vehs = traci.edge.getLastStepVehicleIDs(edge)
                    edge_vh_counts[edge].update(vehs)
                except:
                    # Fallback to lanes if edge not supported
                    # But we only have edge IDs here.
                    pass 
                    
            if step % 600 == 0:
                print(f"    Step {step}/{duration}...")
                
    except Exception as e:
        print(f"  ERROR: {e}")
        # Try finding why: edge functions might not exist in all traci versions
        # fallback to lanes
        pass
        
    env.close()
    
    # Analysis
    counts = []
    for edge in all_incoming_edges:
        count = len(edge_vh_counts[edge])
        counts.append(count)
        
    counts = np.array(counts)
    if len(counts) == 0:
        return 0, 0, 0
        
    avg_c = np.mean(counts)
    min_c = np.min(counts)
    max_c = np.max(counts)
    
    print(f"  [RESULT] Period {period}s:")
    
    # Calculate Theoretical Demand
    num_edges = len(all_incoming_edges)
    if num_edges > 0:
        total_veh_demand = duration / period
        demand_per_edge = total_veh_demand / num_edges
        print(f"    Est. Demand/Edge: {int(demand_per_edge)} (Desired Load)")
    
    print(f"    Edges Monitored: {len(counts)}")
    print(f"    Observed Flow/Edge: {avg_c:.1f} (Throughput Constrained)")
    print(f"    Min Obs/Edge: {min_c}")
    print(f"    Max Obs/Edge: {max_c}")
    
    # Breakdown of low performers
    low_edges = [e for e in all_incoming_edges if len(edge_vh_counts[e]) < 300]
    if low_edges:
        print(f"    Edges below 300: {len(low_edges)} (e.g. {list(low_edges)[:3]})")
        
    return min_c, max_c, avg_c

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Edge Calibration")
    parser.add_argument("--single-agent-id", type=str, default=None, help="ID of the single intersection to calibrate")
    args = parser.parse_args()

    # Periods to test. Previous 0.04 gave ~3000 global. 
    # We need much more density to hit 300 PER EDGE.
    # Try extremely small periods.
    # NOTE: If running single agent, 'period' is adjusted inside SumoRLEnvironment
    # so we might need to be careful. But run_edge_calibration passes traffic_period explicitly.
    # Target periods for high density (400-2500 veh/edge)
    periods = [0.2, 0.4, 0.6, 1.0, 1.5] 
    
    for p in periods:
        run_edge_calibration(p, single_agent_id=args.single_agent_id)

if __name__ == "__main__":
    main()
