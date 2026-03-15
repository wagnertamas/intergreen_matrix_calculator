"""
Headless (nem-GUI) tanítás indító script.
Támogatja a YAML és JSON konfigurációs fájlokat.
"""
import argparse
import json
import os
import sys
import yaml
import wandb
from rl_trainer import IndependentDQNTrainer


def load_config(path):
    """YAML vagy JSON config betöltése."""
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="SUMO RL Headless Trainer")
    parser.add_argument("--config", type=str, default="training_config.yaml",
                        help="Training config (YAML or JSON)")
    parser.add_argument("--single-agent", type=str, default=None,
                        help="Csak egy konkrét ágens tanítása (pl. R2C2_C)")
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI engedélyezése")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Pre-trained modell betöltése (.zip vagy .onnx)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps (felülírja a configot)")
    parser.add_argument("--project", type=str, default=None,
                        help="WandB projekt név")
    # GUI-ból átadott hiperparaméterek (felülírják a config default-ot)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--layer-size", type=int, default=None)
    parser.add_argument("--w-waiting", type=float, default=None)
    parser.add_argument("--w-co2", type=float, default=None)
    args = parser.parse_args()

    # Config keresés: explicit path → yaml → json
    config_path = args.config
    if not os.path.exists(config_path):
        # Próbáljuk a másik formátumot
        alternatives = ["training_config.yaml", "training_config.json",
                        "data/training_config.yaml", "data/training_config.json"]
        config_path = None
        for alt in alternatives:
            if os.path.exists(alt):
                config_path = alt
                break
        if not config_path:
            print("[ERROR] Nem találom a konfigurációs fájlt!")
            sys.exit(1)

    print(f"[INFO] Config: {config_path}")
    config = load_config(config_path)

    hp = config.get("hyperparams", {})
    env_kw = config.get("env_kwargs", {})

    # CLI args felülírják a config default-ot (GUI-ból jönnek)
    cli_hp_map = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma,
        "exploration_fraction": args.exploration_fraction,
        "num_layers": args.num_layers,
        "layer_size": args.layer_size,
    }
    for key, val in cli_hp_map.items():
        if val is not None:
            hp[key] = val

    # WandB login
    api_key = hp.get("wandb_api_key") or config.get("wandb_api_key", "")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    try:
        wandb.login()
    except Exception:
        pass

    # Fájl útvonalak (YAML és JSON kompatibilis)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if "files" in config:
        files = config["files"]
        net_file = files["net"]
        logic_file = files["logic"]
        detector_file = files["detector"]
    else:
        net_file = config.get("net_file", "data/mega_catalogue_v2.net.xml")
        logic_file = config.get("logic_file", "data/traffic_lights.json")
        detector_file = config.get("detector_file", "data/detectors.add.xml")

    # Relatív → abszolút útvonal
    for name, path in [("net", net_file), ("logic", logic_file), ("det", detector_file)]:
        if not os.path.isabs(path):
            full = os.path.join(base_dir, path)
            if name == "net": net_file = full
            elif name == "logic": logic_file = full
            else: detector_file = full

    # Paraméterek
    total_timesteps = args.timesteps or int(config.get("total_timesteps", hp.get("total_timesteps", 100000)))
    project = args.project or config.get("project_name", hp.get("wandb_project", "sumo-rl-single"))
    load_model = args.load_model or hp.get("load_model_path")

    # Reward weights (CLI args felülírják)
    rw = env_kw.get("reward_weights", {})
    reward_weights = {
        'waiting': args.w_waiting if args.w_waiting is not None else float(rw.get('waiting', hp.get('w_waiting', 1.0))),
        'co2': args.w_co2 if args.w_co2 is not None else float(rw.get('co2', hp.get('w_co2', 1.0)))
    }

    print(f"[INFO] Net: {net_file}")
    print(f"[INFO] Timesteps: {total_timesteps} | Project: {project}")
    print(f"[INFO] Single agent: {args.single_agent or 'ALL'}")
    print(f"[INFO] Load model: {load_model or 'None (fresh)'}")

    trainer = IndependentDQNTrainer(
        net_file=net_file,
        logic_file=logic_file,
        detector_file=detector_file,
        total_timesteps=total_timesteps,
        wandb_project=project,
        hyperparams=hp,
        reward_weights=reward_weights,
        log_queue=None,
        single_agent_id=args.single_agent,
        sumo_gui=args.gui,
        load_model_path=load_model,
    )

    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n[INFO] Megszakítva (Ctrl+C).")
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
