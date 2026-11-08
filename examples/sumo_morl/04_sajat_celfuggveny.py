"""
04 – Saját célfüggvény (jutalomkomponens) definiálása.

A célfüggvény csak hurokdetektor-adatokból (``snap``) és a vezérlő által ismert
adatokból (``ctx``) számolhat. Konvenció: nagyobb = jobb.

    python examples/sumo_morl/04_sajat_celfuggveny.py
"""
import os
import sys

# Telepítés nélküli futtatáshoz: a csomag gyökérkönyvtára a modulkeresési útvonalra kerül.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from sumo_morl import FixedTimePolicy, TJunctionMOEnv, register_objective, run_episode


@register_objective("side_road_queue", scale=0.02,
                    doc="-(a déli mellékág becsült járműszáma, másodpercekre összegezve)")
def side_road_queue(snap, ctx):
    return -snap.approach_vehicle_seconds["S"]


@register_objective("main_speed", scale=0.01,
                    doc="A főirány előjelző hurkain mért átlagsebesség [m/s] * lépéshossz")
def main_speed(snap, ctx):
    mask = snap.select(kind="advance") & ~snap.select(approach="S")
    return float(snap.speed[mask].mean()) * snap.duration


env = TJunctionMOEnv(objectives=("side_road_queue", "main_speed"), episode_length=900)
for greens in ([40, 10, 8], [20, 5, 25]):
    res = run_episode(env, FixedTimePolicy(env, greens), seed=0)
    print(f"zöldidők={greens}: {dict(zip(env.objective_names, np.round(res.vector_return, 2).tolist()))}")
env.close()
