import yaml
import argparse
import os
import sys

# Ensure SUMO_HOME is set (Crucial for Colab/Linux)
if 'SUMO_HOME' not in os.environ:
    # Try typical Linux location or warn
    # For Colab, user usually installs to /usr/share/sumo
    DEFAULT_SUMO = "/usr/share/sumo"
    if os.path.exists(DEFAULT_SUMO):
        os.environ['SUMO_HOME'] = DEFAULT_SUMO
    else:
        print("WARNING: SUMO_HOME is not set. Simulation might fail.")

# Add local directory to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rl_trainer import IndependentDQNTrainer
except ImportError:
    print("Error: Could not import rl_trainer. Make sure you are in the correct directory.")
    sys.exit(1)

def run_headless(config_path):
    print(f"Loading configuration from {config_path}...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Configuration loaded successfully.")
    
    # Set WandB API Key if provided in config or env
    if config.get('wandb_api_key'):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key'] 
    
    # Path Adjustment for Colab / Remote
    # Users might upload files to a different folder structure.
    # We should optionally allow overriding the absolute paths from the config.
    # Simple heuristic: Check if file exists, if not, check local dir.
    
    def check_path(p):
        if not p: return p
        if os.path.exists(p): return p
        
        # Try finding in current directory by basename
        basename = os.path.basename(p)
        if os.path.exists(basename):
            print(f"Path adjustment: {p} -> {basename}")
            return os.path.abspath(basename)
        
        # Try 'data' folder
        data_path = os.path.join("data", basename)
        if os.path.exists(data_path):
            print(f"Path adjustment: {p} -> {data_path}")
            return os.path.abspath(data_path)
            
        print(f"Warning: File not found: {p}")
        return p

    net_file = check_path(config['net_file'])
    logic_file = check_path(config['logic_file'])
    detector_file = check_path(config['detector_file'])
    
    print("-" * 40)
    print(f"Starting Headless Training: {config.get('project_name', 'sumo-rl')}")
    print(f"Steps: {config['total_timesteps']}")
    print(f"Envs: {config['n_envs']}")
    print("-" * 40)
    
    trainer = IndependentDQNTrainer(
        net_file=net_file,
        logic_file=logic_file,
        detector_file=detector_file,
        wandb_project=config.get('project_name'),
        total_timesteps=config['total_timesteps'],
        hyperparams=config['hyperparams'],
        n_envs=config['n_envs'],
        **config['env_kwargs']
    )
    
    trainer.run()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO RL Training Headless")
    parser.add_argument("--config", type=str, required=True, help="Path to training_config.yaml")
    args = parser.parse_args()
    
    run_headless(args.config)
