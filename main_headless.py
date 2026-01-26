import argparse
import json
import os
import sys
import wandb
from rl_trainer import IndependentDQNTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training_config.json")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Config not found.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    hp = config["hyperparams"]
    files = config["files"]

    if hp.get("wandb_api_key"):
        os.environ["WANDB_API_KEY"] = hp["wandb_api_key"]
    
    # Colab-on ez interaktív login-t dob, ha nincs key
    try:
        wandb.login()
    except:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(base_dir, files["net"])
    logic_file = os.path.join(base_dir, files["logic"])
    detector_file = os.path.join(base_dir, files["detector"])
    
    trainer = IndependentDQNTrainer(
        net_file=net_file,
        logic_file=logic_file,
        detector_file=detector_file,
        total_timesteps=int(hp["total_timesteps"]),
        wandb_project=hp["wandb_project"],
        hyperparams=hp,
        reward_weights={'time': float(hp.get("w_time", 1.0)), 'co2': float(hp.get("w_co2", 1.0))},
        log_queue=None
    )

    try:
        trainer.run()
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()