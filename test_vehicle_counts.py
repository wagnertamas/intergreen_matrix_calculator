import os
import sys
import numpy as np
import traci

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sumo_rl_environment import SumoRLEnvironment

def test_metrics():
    print("="*50)
    print("TEST: Verifying Vehicle Count (TTS) Metrics")
    print("="*50)

    stats_file = "test_stats_metrics.xml"
    if os.path.exists(stats_file):
        os.remove(stats_file)

    # Switch to project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    print(f"[DEBUG] CWD set to: {root_dir}")

    # Use relative paths (SUMO tools often behave better with them or file URLs)
    net_file = "data/mega_catalogue_v2.net.xml"
    logic_file = "data/traffic_lights.json"
    detector_file = "data/detectors.add.xml"

    if not os.path.exists(net_file):
        print(f"[ERROR] Net file not found at: {os.path.abspath(net_file)}")
        return

    # Initialize Environment with statistics output
    env = SumoRLEnvironment(
        net_file=net_file,
        logic_json_file=logic_file,
        detector_file=detector_file,
        reward_weights={'time': 1.0, 'co2': 1.0},
        min_green_time=5,
        delta_time=1,
        sumo_gui=False,
        random_traffic=True,
        traffic_period=0.002, # Dense traffic to ensure counts
        traffic_duration=200,
        statistic_output_file=stats_file
    )

    print("[INFO] Environment initialized.")
    obs, info = env.reset()
    print("[INFO] Reset complete. Running simulation steps...")

    # Run for 50 steps
    for i in range(50):
        # Dummy actions (keep phase 0)
        actions = {jid: 0 for jid in env.junction_ids}
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 10 == 0:
            print(f"  Step {i}:")
            for jid, agent in env.agents.items():
                print(f"    Agent {jid}: accumulated_tt={agent.accumulated_travel_time:.2f}, accumulated_co2={agent.accumulated_co2:.2f}")

    print("\n[INFO] Simulation loop finished.")

    # Check metrics
    passed = True
    for jid, agent in env.agents.items():
        avg_tt = agent.get_avg_travel_time_metric()
        total_tt = agent.accumulated_travel_time
        steps = agent.steps_measured
        
        print(f"\n[CHECK] Agent {jid}:")
        print(f"  - Total Accumulated Estimate (Sum of Speed-based TT): {total_tt:.2f}")
        print(f"  - Vehicles Counted (for normalization): {agent.accumulated_veh_count}")
        print(f"  - Steps Measured: {steps}")
        print(f"  - Normalized Avg Travel Time (Latency per Vehicle): {avg_tt:.4f} s")

        if total_tt <= 0:
            print("  [FAIL] Total TT should be positive with dense traffic.")
            passed = False
        else:
            print("  [PASS] Total TT is positive.")

        # Heuristic check: With period 0.002, we expect roughly 500 vehicles/sec generated globally?
        # No, 1/0.002 = 500 veh/sec is HUGE. The probability is period=0.002 means 1 vehicle every 2 seconds?
        # randomTrips.py period is "seconds per vehicle". 
        # If period=0.002, that is 500 vehicles PER SECOND. That is wildly high.
        # Wait, my previous calibration said "0.001 to 0.005" provided "300-2500 per edge".
        # If period is small, density is HIGH.
        # So we definitely should see vehicles.

    env.close()

    if os.path.exists(stats_file):
        print(f"\n[PASS] Statistics file '{stats_file}' was created.")
        with open(stats_file, 'r') as f:
            content = f.read()
            if "vehicleTripStatistics" in content:
                print("  [PASS] File contains 'vehicleTripStatistics'.")
            else:
                print("  [FAIL] File missing 'vehicleTripStatistics'.")
                passed = False
    else:
        print(f"\n[FAIL] Statistics file '{stats_file}' not found.")
        passed = False

    if passed:
        print("\nTEST PASSED: Metrics are accumulating and output is generated.")
    else:
        print("\nTEST FAILED: Issues detected.")

if __name__ == "__main__":
    test_metrics()
