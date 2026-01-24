
import os
import sys
import time
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
    sys.path.append(tools)
    
try:
    import libsumo as traci
except ImportError:
    import traci

def get_edge_id(lane_id):
    """Removes last _X from lane ID to get edge ID."""
    if "_" in lane_id:
        return lane_id.rsplit("_", 1)[0]
    return lane_id

def run_edge_calibration(period, duration=3600):
    print(f"\n--- Testing Period: {period}s for {duration}s ---")
    
    env = SumoRLEnvironment(
        net_file=NET_FILE,
        logic_json_file=LOGIC_JSON,
        detector_file=DETECTOR_FILE,
        route_file=ROUTE_FILE,
        sumo_gui=False,
        traffic_duration=duration
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
    
    try:
        while step < duration:
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
    print(f"    Edges Monitored: {len(counts)}")
    print(f"    Avg Vehicles/Edge: {avg_c:.1f}")
    print(f"    Min Vehicles/Edge: {min_c}")
    print(f"    Max Vehicles/Edge: {max_c}")
    
    # Breakdown of low performers
    low_edges = [e for e in all_incoming_edges if len(edge_vh_counts[e]) < 300]
    if low_edges:
        print(f"    Edges below 300: {len(low_edges)} (e.g. {list(low_edges)[:3]})")
        
    return min_c, max_c, avg_c

def main():
    # Periods to test. Previous 0.04 gave ~3000 global. 
    # We need much more density to hit 300 PER EDGE.
    # Try extremely small periods.
    periods = [0.002, 0.001, 0.0005, 0.0002, 0.0001] 
    
    for p in periods:
        run_edge_calibration(p)

if __name__ == "__main__":
    main()
