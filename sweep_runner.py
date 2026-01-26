import os
import sys
import glob
from rl_trainer import IndependentDQNTrainer

def find_files():
    # Helper to find files automatically like in main.py
    # Check current dir AND data/ dir
    net_files = glob.glob("*.net.xml") + glob.glob("data/*.net.xml")
    
    if not net_files:
        raise FileNotFoundError("No .net.xml found in root or data/!")
        
    net_file = os.path.abspath(net_files[0])
    base_name_full = os.path.basename(net_file)
    base_name = base_name_full.replace('.net.xml', '')
    dir_name = os.path.dirname(net_file)
    
    # Try generic names first (traffic_lights.json), then specific names
    possible_logics = [
        os.path.join(dir_name, "traffic_lights.json"),
        os.path.join(dir_name, f"{base_name}.json"),
        os.path.join("data", "traffic_lights.json"),
        "traffic_lights.json"
    ]
    
    logic_file = None
    for f in possible_logics:
        if os.path.exists(f):
            logic_file = os.path.abspath(f)
            break
            
    possible_detectors = [
        os.path.join(dir_name, "detectors.add.xml"),
        os.path.join(dir_name, f"{base_name}.add.xml"),
        os.path.join("data", "detectors.add.xml"),
        "detectors.add.xml"
    ]
    
    detector_file = None
    for f in possible_detectors:
         if os.path.exists(f):
             detector_file = os.path.abspath(f)
             break

    if not logic_file or not detector_file:
         raise FileNotFoundError("Could not find logic or detector files!")
    
    return net_file, logic_file, detector_file

def main():
    print("Starting Sweep Agent...")
    
    try:
        net_file, logic_file, detector_file = find_files()
        
        # Initialize Trainer
        # JAVÍTÁS: 'project_name' HELYETT 'wandb_project'
        trainer = IndependentDQNTrainer(
            net_file=net_file,
            logic_file=logic_file,
            detector_file=detector_file,
            total_timesteps=100000, 
            wandb_project="sumo-rl-sweep"  # <--- ITT VOLT A HIBA
        )
        
        # Run
        trainer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()