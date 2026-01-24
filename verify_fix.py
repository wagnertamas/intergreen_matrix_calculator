import os
import sys
import json
import random

# Ensure libsumo is in path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

try:
    import libsumo as traci
except ImportError:
    print("libsumo not found, trying traci")
    import traci

NET_FILE = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/mega_catalogue_v2.net.xml"
ROUTE_FILE = "/Users/wagnertamas/Documents/Munka/cikkek/AI_agens_konyvtar/finish/random_traffic.rou.xml"
JSON_FILE = "data/traffic_lights.json"
JID = "R1C4_C"

# Load patched JSON
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

if JID not in data:
    print(f"Error: {JID} not found in JSON.")
    exit(1)

phases = data[JID]['phases']
print(f"Loaded {len(phases)} phases for {JID} from JSON.")

sumo_cmd = [
    "sumo",
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--no-step-log", "true",
    "--no-warnings", "true"
]

print("Starting SUMO...")
traci.start(sumo_cmd)
try:
    print(f"Testing {JID} state setting...")
    for i, p in enumerate(phases):
        state = p['state']
        try:
            print(f"  [{i}] Setting state: '{state}' (Len: {len(state)})")
            traci.trafficlight.setRedYellowGreenState(JID, state)
            
            # Verify it persisted (optional, logic might change it instantly if program runs)
            # current = traci.trafficlight.getRedYellowGreenState(JID)
            # print(f"      Current:     '{current}'")
        except Exception as e:
            print(f"  [FAIL] Failed to set state '{state}': {e}")
            raise e
            
    print("[PASS] All phases set successfully.")
    
except Exception as e:
    print(f"[CRITICAL FAIL] {e}")
    
finally:
    traci.close()
