"""
02 – Klasszikus (nem tanuló) vezérlők összehasonlítása azonos forgalmon.

Ezek adják az alapvonalat (baseline), amit egy jó MORL ágensnek meg kell vernie.

    python examples/sumo_morl/02_baseline_osszehasonlitas.py
"""
import os
import sys

# Telepítés nélküli futtatáshoz: a csomag gyökérkönyvtára a modulkeresési útvonalra kerül.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from sumo_morl import (ActuatedPolicy, FixedTimePolicy, LongestQueuePolicy, RandomPolicy, TJunctionMOEnv,
                       evaluate_policy, plot_pareto)

env = TJunctionMOEnv(objectives=("queue", "fairness", "switches"), episode_length=1800)
seeds = [0, 1, 2]  # azonos magok = azonos forgalom minden vezérlőnek

policies = {
    "véletlen": RandomPolicy(env, seed=0),
    "fix 30/10/15": FixedTimePolicy(env, [30, 10, 15]),
    "fix 20/5/10": FixedTimePolicy(env, [20, 5, 10]),
    "járműkövető": ActuatedPolicy(env, max_green=45, gap=3),
    "leghosszabb sor": LongestQueuePolicy(env),
}

names, means = [], []
print(f"{'vezérlő':<16}" + "".join(f"{n:>12}" for n in env.objective_names) + f"{'átl. késés [s]':>16}")
for name, pol in policies.items():
    res = evaluate_policy(env, pol, seeds=seeds)
    names.append(name)
    means.append(res["mean"])
    print(f"{name:<16}" + "".join(f"{v:12.1f}" for v in res["mean"])
          + f"{res['ground_truth']['mean_delay_per_vehicle']:16.1f}")
env.close()

fig = plot_pareto(np.array(means), env.objective_names, labels=names, title="Baseline vezérlők")
fig.savefig("baseline_pareto.png", dpi=120)
print("Ábra mentve: baseline_pareto.png")
