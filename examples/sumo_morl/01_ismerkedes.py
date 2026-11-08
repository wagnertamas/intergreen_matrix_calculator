"""
01 – Ismerkedés a környezettel.

Futtatás (a csomag gyökérkönyvtárából):
    python examples/sumo_morl/01_ismerkedes.py
    python examples/sumo_morl/01_ismerkedes.py --gui     # SUMO grafikus felülettel
"""
import os
import sys

# Telepítés nélküli futtatáshoz: a csomag gyökérkönyvtára a modulkeresési útvonalra kerül.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse

import numpy as np

from sumo_morl import TJunctionMOEnv, list_objectives

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
args = parser.parse_args()

env = TJunctionMOEnv(objectives=("queue", "fairness", "switches"), episode_length=600, gui=args.gui)

print("=== A csomópont és a jelzésterv ===")
print(env.scenario.describe())
print("\n=== Elérhető célfüggvények ===")
print(list_objectives())
print("\nAkciók (fázisok):")
for i, name in enumerate(env.phase_names):
    print(f"  {i}: {name}")
print(f"\nMegfigyelés: {env.observation_space.shape[0]} elem, pl. {env.observation_names()[:5]} ...")
print(f"Jutalomvektor: {env.objective_names}")

obs, info = env.reset(seed=42)
print(f"\nForgalmi igény ebben az epizódban: {info['demand']}")

total = np.zeros(env.reward_dim)
done = False
while not done:
    action = env.action_space.sample()            # itt jönne az ágens döntése
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
    if info["time"] % 120 < info["elapsed"]:
        print(f"t={info['time']:4d}s  fázis={info['phase']}  jutalom={np.round(reward, 2)}  "
              f"becsült járművek={ {k: round(v) for k, v in info['detectors'].approach_vehicles.items()} }")

print(f"\nEpizód összjutalma: {dict(zip(env.objective_names, np.round(total, 2).tolist()))}")
print(f"Valós mutatók (csak kiértékeléshez!): {info['ground_truth']}")
env.close()
