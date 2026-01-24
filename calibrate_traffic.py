
import os
import sys
import subprocess
import re

# Config
NET_FILE = "./data/mega_catalogue_v2.net.xml"
TEMP_ROUTE_FILE = "./temp_calibration.rou.xml"

# Find randomTrips
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
else:
    # Attempt to locate via sumolib
    try:
        import sumolib
        tools = os.path.join(os.path.dirname(sumolib.__file__), '..', '..', 'tools')
        tools = os.path.abspath(tools)
    except:
        print("ERROR: SUMO_HOME not set and sumolib not found.")
        sys.exit(1)

random_trips_script = os.path.join(tools, "randomTrips.py")

def count_trips(file_path):
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '<trip' in line:
                    count += 1
    except FileNotFoundError:
        return 0
    return count

def test_params(period, fringe_factor, min_dist):
    print(f"\nTesting: Period={period}, Fringe={fringe_factor}, MinDist={min_dist}")
    
    cmd = [
        sys.executable, random_trips_script,
        "-n", NET_FILE,
        "-o", TEMP_ROUTE_FILE,
        "-e", "3600",
        "-p", str(period),
        "--fringe-factor", str(fringe_factor),
        "--min-distance", str(min_dist),
        "--validate"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        count = count_trips(TEMP_ROUTE_FILE)
        print(f"  => TRIP COUNT: {count}")
        return count
    except subprocess.CalledProcessError as e:
        print(f"  => ERROR: {e.stderr.decode()}")
        return 0

def main():
    print("=== Traffic Calibration ===")
    
    # Baseline (Current)
    # count = test_params(period=0.2, fringe_factor=10, min_dist=50)
    
    # Experiments
    # We want roughly 300 to 2500 trips.
    # If 0.2 gave ~637, maybe 0.05 gives ~2500?
    
    configs = [
        {'p': 1.0, 'f': 10, 'd': 50},  # Originalish
        {'p': 0.2, 'f': 10, 'd': 50},  # My recent 'dense'
        {'p': 0.1, 'f': 10, 'd': 50},  # Even denser
        
        # Relaxing constraints
        {'p': 0.2, 'f': 5, 'd': 50},   # Less fringe focus
        {'p': 0.2, 'f': 10, 'd': 0},   # Any distance
        
        # Trying to hit 2500
        {'p': 0.05, 'f': 10, 'd': 50},
        {'p': 0.04, 'f': 10, 'd': 50},
    ]
    
    best_config = None
    target_min = 300
    target_max = 2500
    closest_dist = float('inf')
    
    for c in configs:
        cnt = test_params(c['p'], c['f'], c['d'])
        if target_min <= cnt <= target_max:
             print(f"  [SUCCESS] Match found!")
             
        dist = min(abs(cnt - target_min), abs(cnt - target_max))
        # Logic to pick 'best' default? 
        # User wants min 300, max 2500.
        # Maybe we want a range of periods that map to this.
        
    # Clean up
    if os.path.exists(TEMP_ROUTE_FILE):
        os.remove(TEMP_ROUTE_FILE)

if __name__ == "__main__":
    main()
