"""
Headless (nem-GUI) tanítás indító script.
Támogatja a YAML és JSON konfigurációs fájlokat.

Használat:
  # Egy junction tanítás QRDQN-nel (default)
  python main_headless.py --junction R1C1_C --timesteps 50000

  # PPO algoritmussal, halt_ratio reward-dal
  python main_headless.py --junction R2C2_C --algorithm ppo --reward-mode halt_ratio

  # Legjobb kombó (TotalWaitingTime + triplet TP+Std+Halt)
  python main_headless.py --junction R1C1_C --reward-mode wait_haltratio

  # Összes junction szekvenciálisan
  for jid in R1C1_C R1C2_C R1C3_C; do
    python main_headless.py --junction $jid --timesteps 50000
  done
"""
import argparse
import json
import os
import sys
import yaml
import wandb
from rl_trainer import IndependentDQNTrainer, SUPPORTED_ALGORITHMS


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
                        help="[DEPRECATED] Használd a --junction-t helyette")
    parser.add_argument("--junction", type=str, default=None,
                        help="Junction ID a per-junction tanításhoz (pl. R2C2_C)")
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI engedélyezése")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Pre-trained modell betöltése (.zip vagy .onnx)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps (felülírja a configot)")
    parser.add_argument("--project", type=str, default=None,
                        help="WandB projekt név")
    # Algoritmus és reward mód
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=list(SUPPORTED_ALGORITHMS.keys()),
                        help=f"RL algoritmus ({', '.join(SUPPORTED_ALGORITHMS.keys())}). Default: qrdqn")
    parser.add_argument("--reward-mode", type=str, default=None,
                        choices=["speed_throughput", "speed_throughput_freq", "halt_ratio",
                                 "co2_speedstd", "wait_triplet_tpstdhalt", "wait_haltratio"],
                        help="Reward számítási mód. Default: speed_throughput\n"
                             "  wait_triplet_tpstdhalt = TotalWaitingTime + (Throughput+SpeedStd+HaltRatio)/3  [AJÁNLOTT]")
    parser.add_argument("--junction-params", type=str, default=None,
                        help="junction_reward_params.json útvonal (default: auto-keresés)")
    # Hiperparaméterek
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--layer-size", type=int, default=None)
    parser.add_argument("--w-waiting", type=float, default=None)
    parser.add_argument("--w-co2", type=float, default=None)
    # Fix forgalom
    parser.add_argument("--flow-target", type=int, default=None,
                        help="Fix forgalom target (veh/h/lane)")
    parser.add_argument("--flow-spread", type=int, default=0,
                        help="Fix forgalom spread (±)")
    args = parser.parse_args()

    # --junction és --single-agent kompatibilitás
    junction_id = args.junction or args.single_agent

    # Config keresés
    config_path = args.config
    if not os.path.exists(config_path):
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

    # CLI hiperparaméterek felülírják a configot
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

    # Fájl útvonalak
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

    # Algoritmus és reward mód (CLI > config > default)
    algorithm = args.algorithm or config.get("algorithm", "qrdqn")
    reward_mode = args.reward_mode or config.get("reward_mode", "speed_throughput")
    junction_params = args.junction_params  # None → auto-keresés az env-ben

    # Reward weights (backward compat)
    rw = env_kw.get("reward_weights", {})
    reward_weights = {
        'waiting': args.w_waiting if args.w_waiting is not None else float(rw.get('waiting', hp.get('w_waiting', 1.0))),
        'co2': args.w_co2 if args.w_co2 is not None else float(rw.get('co2', hp.get('w_co2', 1.0)))
    }

    # Fix forgalom
    fixed_flow = None
    if args.flow_target is not None:
        fixed_flow = {'target': args.flow_target, 'spread': args.flow_spread}

    print(f"[INFO] Net: {net_file}")
    print(f"[INFO] Timesteps: {total_timesteps} | Project: {project}")
    print(f"[INFO] Junction: {junction_id or 'ALL'}")
    print(f"[INFO] Algorithm: {algorithm} | Reward: {reward_mode}")
    print(f"[INFO] Load model: {load_model or 'None (fresh)'}")
    if fixed_flow:
        print(f"[INFO] Fixed traffic: target={fixed_flow['target']}, spread=±{fixed_flow['spread']}")
    else:
        print(f"[INFO] Traffic: random target+spread (epizódonként)")

    trainer = IndependentDQNTrainer(
        net_file=net_file,
        logic_file=logic_file,
        detector_file=detector_file,
        total_timesteps=total_timesteps,
        wandb_project=project,
        hyperparams=hp,
        reward_weights=reward_weights,
        log_queue=None,
        single_agent_id=junction_id,
        sumo_gui=args.gui,
        load_model_path=load_model,
        fixed_flow=fixed_flow,
        algorithm=algorithm,
        reward_mode=reward_mode,
        junction_params_path=junction_params,
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
