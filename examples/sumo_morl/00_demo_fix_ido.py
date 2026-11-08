"""
00 – DEMÓ: klasszikus, fix idejű jelzőlámpa-program.

Ez a legegyszerűbb vezérlés: a fázisok mindig ugyanabban a sorrendben,
mindig ugyanannyi ideig kapnak zöldet, a forgalomtól függetlenül. A
fázisváltásokat (sárga, piros, piros-sárga) a környezet a közbenső idők
alapján automatikusan beilleszti.

A demóval megnézhető, hogyan működik a szimuláció, mit látnak a
hurokdetektorok, és milyen jutalmakat ad a környezet – még mielőtt bármilyen
tanuló ágenst írnánk.

Futtatás (a csomag gyökérkönyvtárából):
    python examples/sumo_morl/00_demo_fix_ido.py --gui                # SUMO grafikus felülettel
    python examples/sumo_morl/00_demo_fix_ido.py                      # csak konzol
    python examples/sumo_morl/00_demo_fix_ido.py --green 40 10 20     # más zöldidők
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np

from sumo_morl import Demand, TJunctionMOEnv
from sumo_morl.visualize import plot_signal_timeline

parser = argparse.ArgumentParser(description="Fix idejű jelzőlámpa-program demó")
parser.add_argument("--gui", action="store_true", help="sumo-gui megjelenítés")
parser.add_argument("--green", type=int, nargs=3, default=[30, 10, 15], metavar=("Z0", "Z1", "Z2"),
                    help="zöldidők fázisonként [s] (alapértelmezés: 30 10 15)")
parser.add_argument("--duration", type=int, default=900, help="szimulált idő [s]")
parser.add_argument("--main", type=float, default=550, help="főirány forgalma [jármű/óra/irány]")
parser.add_argument("--side", type=float, default=250, help="mellékág forgalma [jármű/óra]")
args = parser.parse_args()

# decision_interval=1 és min_green=1: a környezet másodpercenként kérdez, így a
# zöldidők pontosan betarthatók (a fix program maga dönt a váltásról).
env = TJunctionMOEnv(
    objectives=("queue", "fairness", "switches"),
    demand=Demand.constant(main=args.main, side=args.side),
    episode_length=args.duration,
    decision_interval=1,
    min_green=1,
    gui=args.gui,
    record_signals=True,
)
plan = env.plan
green = args.green

cycle = sum(green) + sum(plan.transition_duration(p, (p + 1) % plan.n_phases) for p in range(plan.n_phases))
print("=" * 72)
print(" FIX IDEJŰ PROGRAM")
print("=" * 72)
for p, name in enumerate(env.phase_names):
    nxt = (p + 1) % plan.n_phases
    print(f"  Fázis {p}: {green[p]:>3} s zöld   {plan.green_state(p)}   {name}")
    print(f"           átmenet {p}->{nxt}: {plan.transition_duration(p, nxt)} s  "
          + " -> ".join(f"{s}({d}s)" for s, d in plan.transition(p, nxt)))
print(f"  Ciklusidő: {cycle} s")
print(f"  Forgalom:  {env.demand_spec}")
print("-" * 72)
print(f"{'idő':>6}  {'esemény':<34} {'becsült jármű (W/E/S)':>22}")

obs, info = env.reset(seed=0)
total = np.zeros(env.reward_dim)
done = False
while not done:
    p = info["phase"]
    action = (p + 1) % plan.n_phases if info["time_in_phase"] >= green[p] else p
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
    if info["phase_changed"]:
        veh = info["detectors"].approach_vehicles
        print(f"{info['time']:>5}s  váltás -> fázis {info['phase']} ({plan.green_state(info['phase'])})"
              f"{veh['W']:>12.0f} /{veh['E']:>3.0f} /{veh['S']:>3.0f}")

gt = info["ground_truth"]
print("-" * 72)
print("Epizód összjutalma (ezt látná egy ágens):")
for n, v in zip(env.objective_names, total):
    print(f"  {n:<10} {v:10.2f}")
print("Valós mutatók (csak kiértékeléshez):")
print(f"  átlagos késés járművenként: {gt['mean_delay_per_vehicle']:.1f} s")
print(f"  átlagos utazási idő:        {gt['mean_travel_time']:.1f} s")
print(f"  elindult / célba ért:       {gt['departed']} / {gt['arrived']}")

ax = plot_signal_timeline(env.signal_log, plan, t_from=1, t_to=min(2 * cycle + 1, args.duration))
ax.figure.savefig("fix_ido_jelzesterv.png", dpi=120)
print("Jelzésterv-ábra (első két ciklus) mentve: fix_ido_jelzesterv.png")
env.close()
