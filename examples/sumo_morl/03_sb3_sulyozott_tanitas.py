"""
03 – MORL "külső hurok" lineáris skalarizálással (Stable-Baselines3 DQN).

Minden súlyvektorra külön ágenst tanítunk a ``w · r`` skalár jutalommal,
majd kiértékeljük a vektor-hozamokat, és felrajzoljuk a Pareto-frontot.

Telepítés:  pip install -e ".[rl]"
Futtatás:   python examples/sumo_morl/03_sb3_sulyozott_tanitas.py --timesteps 20000

A tanítás hosszabb ideig tart; a kis timesteps csak kipróbálásra jó.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np
from stable_baselines3 import DQN

from sumo_morl import (LinearScalarization, TJunctionMOEnv, equally_spaced_weights, evaluate_policy,
                       hypervolume, pareto_mask, plot_pareto, sb3_policy)

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=20000)
parser.add_argument("--divisions", type=int, default=2, help="súlyrács finomsága")
args = parser.parse_args()

OBJ = ("queue", "fairness", "switches")
weights = equally_spaced_weights(len(OBJ), args.divisions)
print(f"{len(weights)} súlyvektor:\n{weights}")

eval_env = TJunctionMOEnv(objectives=OBJ, episode_length=1800)
points = []
for w in weights:
    train_env = LinearScalarization(TJunctionMOEnv(objectives=OBJ, episode_length=1800, seed=0), w)
    model = DQN("MlpPolicy", train_env, learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
                exploration_fraction=0.3, target_update_interval=500, verbose=0, seed=0)
    model.learn(total_timesteps=args.timesteps)
    train_env.close()
    res = evaluate_policy(eval_env, sb3_policy(model), seeds=[100, 101])
    points.append(res["mean"])
    print(f"w={np.round(w, 2)}  hozam={np.round(res['mean'], 1)}  "
          f"késés={res['ground_truth']['mean_delay_per_vehicle']:.1f} s")
eval_env.close()

points = np.array(points)
ref = points.min(axis=0) - 1.0
print(f"\nPareto-optimális súlyok: {weights[pareto_mask(points)].round(2).tolist()}")
print(f"Hipertérfogat (ref={ref.round(1)}): {hypervolume(points, ref):.1f}")
fig = plot_pareto(points, OBJ, labels=[str(np.round(w, 2)) for w in weights], title="DQN, lineáris skalarizálás")
fig.savefig("dqn_pareto.png", dpi=120)
print("Ábra mentve: dqn_pareto.png")
