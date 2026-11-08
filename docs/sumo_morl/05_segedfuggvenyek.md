# MORL segédfüggvények és összehasonlító vezérlők

Minden alább felsorolt függvény a `sumo_morl` csomagból importálható.

## Epizód futtatása, kiértékelés

```python
from sumo_morl import run_episode, evaluate_policy

res = run_episode(env, policy, seed=0, record=True)
res.vector_return        # np.ndarray – az epizód vektor-hozama
res.ground_truth         # valós mutatók (kiértékeléshez)
res.history              # lépésenkénti napló (record=True esetén)

stats = evaluate_policy(env, policy, seeds=[0, 1, 2])
stats["mean"], stats["std"], stats["returns"], stats["ground_truth"]
```

A `policy` bármilyen függvény, amely `policy(obs, info) -> akció`. Egy
Stable-Baselines3 modellből az `sb3_policy(model)` készít ilyet.

Az `evaluate_policy` azonos `seeds` listával minden vezérlőt **azonos
forgalmon** futtat – összehasonlításnál mindig így tegyél.

## Skalarizálás és súlyok

```python
from sumo_morl import LinearScalarization, WeightConditioned, equally_spaced_weights, random_weights

env_s = LinearScalarization(env, weights=[0.5, 0.3, 0.2])     # skalár jutalom
env_w = WeightConditioned(env)                                # véletlen súly epizódonként, a megfigyelésben

equally_spaced_weights(3, 4)    # egyenletes rács a súly-szimplexen (15 súlyvektor)
random_weights(3, 10)           # 10 véletlen súlyvektor (Dirichlet)
```

## Pareto-front és hipertérfogat

```python
from sumo_morl import pareto_mask, pareto_front, hypervolume, plot_pareto

points = np.array([...])          # pl. több vezérlő átlagos vektor-hozama (sorok)
pareto_mask(points)               # mely pontok nem domináltak (maximalizálás!)
pareto_front(points)
hypervolume(points, ref=points.min(axis=0) - 1)   # nagyobb = jobb front
plot_pareto(points, env.objective_names, labels=[...]).savefig("pareto.png")
```

* Egy `a` pont **dominálja** `b`-t, ha minden célban legalább olyan jó és
  legalább egyben jobb.
* A **hipertérfogat** a front és egy (mindennél rosszabb) referenciapont közötti
  térfogat – egyetlen számmal jellemzi, mennyire jó és mennyire szerteágazó a
  front. Különböző módszereket **azonos referenciaponttal** hasonlíts össze.

## Ábrák

```python
from sumo_morl.visualize import plot_scenario, plot_phases, plot_intergreen, plot_episode

res = run_episode(env, policy, seed=0, record=True)
plot_episode(res.history, env.phase_names).savefig("epizod.png")   # fázisok, sorok, jutalmak

# klasszikus jelzésterv-ábra (sávos diagram) – ehhez: TJunctionMOEnv(record_signals=True)
from sumo_morl.visualize import plot_signal_timeline
plot_signal_timeline(env.signal_log, env.plan, t_from=0, t_to=150).figure.savefig("jelzesterv.png")
```

## Összehasonlító vezérlők (baseline-ok)

Mindegyik csak detektoradatot használ, és `policy(obs, info)` alakú.

| Osztály | Működés |
|---|---|
| `RandomPolicy(env)` | véletlen fázis |
| `FixedTimePolicy(env, green_times=[30, 10, 15])` | fix idejű, ciklikus program |
| `ActuatedPolicy(env, max_green=45, gap=3)` | járműkövető: amíg az előjelző hurkon `gap` s-on belül jön jármű, hosszabbít; utána a következő igényelt fázisra vált |
| `LongestQueuePolicy(env, min_hold=10)` | mohó: a legnagyobb becsült igényű fázist választja |

Egy jó MORL ágensnek legalább a fix idejű vezérlést kell vernie, és érdemes a
járműkövetővel is összevetni. Egy tipikus eredmény (1800 s, 3 véletlen forgalom):

| vezérlő | queue | fairness | switches | átl. késés [s] |
|---|---|---|---|---|
| véletlen | −463 | −735 | −148 | 59 |
| fix 30/10/15 | −175 | −226 | −78 | 17 |
| járműkövető | −136 | −157 | −121 | 9 |
| leghosszabb sor | −134 | −156 | −112 | 9 |

Látszik a kompromisszum: a fix idejű program kevesebbet vált, de több a
várakozás.

## Párhuzamos környezetek

Minden `TJunctionMOEnv` saját SUMO-folyamatot indít, így több is futhat
egyszerre, pl. Stable-Baselines3 `SubprocVecEnv`-vel:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from sumo_morl import TJunctionMOEnv, LinearScalarization

def make(i):
    return lambda: LinearScalarization(TJunctionMOEnv(seed=i), [0.5, 0.3, 0.2])

vec_env = SubprocVecEnv([make(i) for i in range(4)])
```

## Sebesség

Egy 3600 s-os epizód egy átlagos gépen nagyjából 6 másodperc alatt fut le (ground truth nélkül
gyorsabb: `ground_truth=False`). Tanításhoz érdemes `ground_truth=False`-t
használni, kiértékeléshez `True`-t.
