import os
import sys

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
JID = "R1C4_C"

sumo_cmd = [
    "sumo",
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--no-step-log", "true",
    "--no-warnings", "true"
]

print("Starting SUMO...")
traci.start(sumo_cmd)

print(f"Inspecting TLS: {JID}")
try:
    initial_state = traci.trafficlight.getRedYellowGreenState(JID)
    print(f"Current State: '{initial_state}'")
    print(f"Length: {len(initial_state)}")
    
    controlled_links = traci.trafficlight.getControlledLinks(JID)
    print(f"Controlled Links Groups: {len(controlled_links)}")
    count = 0 
    for i, group in enumerate(controlled_links):
        print(f"  Index {i}: {len(group)} links -> {group}")
        count += 1
    print(f"Total Logic Indices: {count}")
    
except Exception as e:
    print(f"Error accessing TLS: {e}")

traci.close()
