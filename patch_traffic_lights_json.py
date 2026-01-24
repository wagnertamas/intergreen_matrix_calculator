import json
import shutil
import os

JSON_FILE = "data/traffic_lights.json"
BACKUP_FILE = "data/traffic_lights.json.bak"
JID = "R1C4_C"

# 1. Create backup
if not os.path.exists(BACKUP_FILE):
    shutil.copy(JSON_FILE, BACKUP_FILE)
    print(f"Backup created: {BACKUP_FILE}")

# 2. Load JSON
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

if JID not in data:
    print(f"Error: {JID} not found in JSON.")
    exit(1)

phases = data[JID]['phases']
print(f"Found {len(phases)} phases for {JID}.")

patched_count = 0
for p in phases:
    state = p['state']
    # Check if length is 13 (missing indices 3-4)
    if len(state) == 13:
        # Insert 'gg' at index 3
        # Indices 0-2 (3 chars) + 'gg' + Indices 3-12 (10 chars)
        new_state = state[:3] + "gg" + state[3:]
        
        if len(new_state) != 15:
            print(f"Error patching state: {state} -> {new_state} (Len: {len(new_state)})")
        else:
            p['state'] = new_state
            patched_count += 1
    elif len(state) == 15:
        print("State already has length 15, skipping.")
    else:
        print(f"Warning: Unexpected state length {len(state)}: {state}")

print(f"Patched {patched_count} phases.")

# 3. Save JSON
with open(JSON_FILE, 'w') as f:
    json.dump(data, f, indent=4)
    
print("Successfully saved patched JSON.")
