
import os
import sys
import collections

# Ensure sumolib is available
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

try:
    import sumolib
except ImportError:
    sys.exit("CRITICAL: sumolib not found. Please set SUMO_HOME.")

NET_FILE = "./data/mega_catalogue_v2.net.xml"

def analyze():
    print(f"Loading network: {NET_FILE}")
    net = sumolib.net.readNet(NET_FILE)
    
    # Store groups: signature -> list of junction IDs
    # signature: tuple of (num_incoming_edges, tuple(sorted_lane_counts))
    topology_groups = collections.defaultdict(list)
    
    nodes = net.getNodes()
    tl_nodes = [n for n in nodes if n.getType() == "traffic_light"]
    
    print(f"Found {len(tl_nodes)} traffic light controlled intersections.")
    
    for node in tl_nodes:
        incoming_edges = node.getIncoming()
        # Filter out internal edges (starting with :) if any, though getIncoming usually returns normal edges
        # Also sumolib might return connections? No, getIncoming returns Edge objects.
        
        valid_incoming = []
        for edge in incoming_edges:
            # We want only "real" edges, not internal ones inside the junction
            if edge.getFunction() == "normal": # or empty string?
                valid_incoming.append(edge)
        
        # If valid_incoming is empty, maybe function is not set, let's just take all that don't start with :
        if not valid_incoming:
             valid_incoming = [e for e in incoming_edges if not e.getID().startswith(":")]
             
        lane_counts = []
        for edge in valid_incoming:
            lane_counts.append(edge.getLaneNumber())
            
        lane_counts.sort()
        
        signature = (len(valid_incoming), tuple(lane_counts))
        topology_groups[signature].append(node.getID())
        
    print("\n--- Topology Groups ---")
    for sig, ids in sorted(topology_groups.items()):
        num_edges, lanes = sig
        print(f"Config: {num_edges} incoming edges, Lanes: {lanes} -> {len(ids)} intersections")
        # Print a few examples
        examples = ids[:3]
        print(f"    Examples: {examples}")

if __name__ == "__main__":
    analyze()
