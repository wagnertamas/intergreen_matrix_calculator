"""
WandB Sweep runner — a sweep agent ezt hívja minden egyes futásnál.

A hiperparamétereket a wandb.config-ból veszi (sweep_config.yaml definiálja),
a fájl útvonalakat a training_config.yaml-ból (vagy .json-ból).
"""
import os
import sys
import yaml
import json
import wandb
from rl_trainer import IndependentDQNTrainer


def load_config():
    """YAML vagy JSON training config betöltése."""
    for candidate in ["training_config.yaml", "training_config.json",
                       "data/training_config.yaml", "data/training_config.json"]:
        if os.path.exists(candidate):
            with open(candidate, 'r') as f:
                if candidate.endswith('.yaml') or candidate.endswith('.yml'):
                    return yaml.safe_load(f), candidate
                else:
                    return json.load(f), candidate
    return None, None


def find_file(candidates):
    """Az első létező fájlt adja vissza a jelöltek közül."""
    for f in candidates:
        if os.path.exists(f):
            return os.path.abspath(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WandB Sweep Runner")
    parser.add_argument("--gui", action="store_true", help="SUMO GUI engedélyezése")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps (felülírja a configot)")
    parser.add_argument("--project", type=str, default=None,
                        help="WandB projekt név (felülírja a configot)")
    args, _ = parser.parse_known_args()

    # --- Config betöltése ---
    config, config_path = load_config()

    # Fájlok megkeresése
    if config:
        # YAML formátum
        if "files" in config:
            files = config["files"]
            net_file = find_file([files.get("net", ""), "data/mega_catalogue_v2.net.xml"])
            logic_file = find_file([files.get("logic", ""), "data/traffic_lights.json"])
            detector_file = find_file([files.get("detector", ""), "data/detectors.add.xml"])
        else:
            net_file = find_file([config.get("net_file", ""), "data/mega_catalogue_v2.net.xml"])
            logic_file = find_file([config.get("logic_file", ""), "data/traffic_lights.json"])
            detector_file = find_file([config.get("detector_file", ""), "data/detectors.add.xml"])

        hp = config.get("hyperparams", {})
        env_kw = config.get("env_kwargs", {})
        default_timesteps = int(config.get("total_timesteps", hp.get("total_timesteps", 50000)))
        default_project = config.get("project_name", hp.get("wandb_project", "sumo-rl-sweep"))
    else:
        # Fallback: automatikus keresés
        import glob
        net_files = glob.glob("*.net.xml") + glob.glob("data/*.net.xml")
        net_file = os.path.abspath(net_files[0]) if net_files else None
        logic_file = find_file(["data/traffic_lights.json", "traffic_lights.json"])
        detector_file = find_file(["data/detectors.add.xml", "detectors.add.xml"])
        hp = {}
        env_kw = {}
        default_timesteps = 50000
        default_project = "sumo-rl-sweep"

    if not net_file or not logic_file or not detector_file:
        print(f"[ERROR] Hiányzó fájlok! net={net_file}, logic={logic_file}, det={detector_file}")
        sys.exit(1)

    # Prioritás: CLI arg > env var > config default
    total_timesteps = args.timesteps or int(os.environ.get("SWEEP_TIMESTEPS", 0)) or default_timesteps
    project = args.project or os.environ.get("SWEEP_PROJECT", "") or default_project

    # Reward weights
    rw = env_kw.get("reward_weights", {})
    reward_weights = {
        'waiting': float(rw.get('waiting', hp.get('w_waiting', 1.0))),
        'co2': float(rw.get('co2', hp.get('w_co2', 1.0)))
    }

    print(f"[SWEEP] Config: {config_path or 'auto-detect'}")
    print(f"[SWEEP] Net: {net_file}")
    print(f"[SWEEP] Timesteps: {total_timesteps} | Project: {project}")
    print(f"[SWEEP] Reward weights: {reward_weights}")
    print(f"[SWEEP] Base hyperparams: {hp}")

    # --- Trainer ---
    trainer = IndependentDQNTrainer(
        net_file=net_file,
        logic_file=logic_file,
        detector_file=detector_file,
        total_timesteps=total_timesteps,
        wandb_project=project,
        hyperparams=hp,
        reward_weights=reward_weights,
        sumo_gui=args.gui,
    )

    try:
        trainer.run()
    except Exception as e:
        print(f"[SWEEP ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
