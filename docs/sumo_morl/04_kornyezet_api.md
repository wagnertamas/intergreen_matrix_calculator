# A környezet API-ja

```python
from sumo_morl import TJunctionMOEnv, Demand

env = TJunctionMOEnv(
    objectives=("queue", "fairness", "switches"),  # jutalomkomponensek
    demand=None,              # None = epizódonként véletlen igény
    episode_length=3600,      # szimulált másodperc
    decision_interval=5,      # "maradok" akció: ennyi másodperccel hosszabb zöld
    min_green=5,              # váltás után ennyi ideig biztosan zöld
    max_green=None,           # pl. 90: ennél hosszabb zöld után kényszerített váltás
    warmup=0,                 # ennyi s fix idejű bemelegítés az epizód elején
    gui=False,                # sumo-gui megjelenítés
    seed=None,
    ground_truth=True,        # info["ground_truth"] kiértékeléshez
)
```

A környezet a [Gymnasium](https://gymnasium.farama.org/) API-t és az
MO-Gymnasium konvenciót követi (vektor jutalom + `env.reward_space`).
`gym.make("sumo_morl/TJunction-v0", ...)` is működik.

## Akció

`env.action_space = Discrete(3)`: a **következő fázis** sorszáma.

* Ha az akció az aktuális fázis → a zöld `decision_interval` (5) másodperccel
  meghosszabbodik.
* Ha más fázis → lefut a szabályos átmenet (4–6 s, lásd
  [jelzésterv](01_csomopont_es_jelzesterv.md)), majd az új fázis `min_green` (5)
  másodpercig biztosan zöld. Csak ezután jön a következő döntés.

Emiatt a lépések hossza **változó** (5 s vagy 9–11 s). Az `info["elapsed"]`
megadja az adott lépés hosszát. A jutalmak a lépés másodperceire összegzettek,
így ez nem torzítja az összjutalmat. (Ha diszkontálsz, érdemes lehet a
diszkontot a lépés hosszához igazítani: `gamma ** info["elapsed"]` – ez egy jó
haladó feladat.)

Bármelyik akció mindig érvényes; szabálytalan jelzéskép nem állhat elő.

## Lépés és epizód

```python
obs, info = env.reset(seed=0)                      # új epizód, új forgalom
obs, reward, terminated, truncated, info = env.step(action)
```

* `reward`: `np.ndarray`, alakja `(len(objectives),)`.
* `terminated` mindig `False`; `truncated` akkor `True`, ha letelt az
  `episode_length`.
* Azonos `seed` → azonos forgalom (azonos járművek, azonos időpontokban), így a
  vezérlők igazságosan összehasonlíthatók.
* `env.close()` leállítja a SUMO-t. Több környezet futhat párhuzamosan.

## Az `info` szótár

| Kulcs | Tartalom |
|---|---|
| `time` | eltelt szimulációs idő az epizód kezdete óta [s] |
| `elapsed` | az utolsó lépés hossza [s] |
| `phase`, `phase_name` | az aktuális fázis |
| `time_in_phase` | az aktuális zöld eddigi hossza [s] |
| `phase_changed`, `forced_switch` | volt-e váltás / kényszerített volt-e (`max_green`) |
| `switches` | eddigi váltások száma az epizódban |
| `objectives` | a jutalomvektor névvel: `{'queue': -1.2, ...}` |
| `detectors` | a nyers hurokadatok (`DetectorSnapshot`) |
| `demand` | az epizód forgalmi igénye (`Demand`) |
| `ground_truth` | valós SUMO mutatók – **csak kiértékeléshez** |

## Forgalmi igény

```python
from sumo_morl import Demand

Demand.constant(main=600, side=250)                 # állandó igény
Demand({("W", "E"): 500, ("W", "S"): 100, ("E", "W"): 450,
        ("E", "S"): 150, ("S", "W"): 120, ("S", "E"): 80})
Demand.constant(500, 200).scaled(1.3)               # 30%-kal több forgalom
Demand.random(rng, profile=(0.5, 1.0, 1.5, 1.0))    # csúcsórás hullám

env = TJunctionMOEnv(demand=Demand.constant(600, 250))
env = TJunctionMOEnv(demand=lambda rng: Demand.random(rng, main_range=(400, 800)))
env.reset(options={"demand": Demand.constant(300, 100)})   # egy epizódra felülírva
```

Mozgások: `("W","E")`, `("W","S")`, `("E","W")`, `("E","S")`, `("S","W")`,
`("S","E")`; egység: jármű/óra. A járművek véletlenszerűen (Poisson-folyamat
szerint) érkeznek.

Alapértelmezés (véletlen igény): főirány 200–700 jármű/óra irányonként,
mellékág 80–400 jármű/óra, kanyarodási arányok 10–45%.

## Megjelenítés

`TJunctionMOEnv(gui=True)` – elindul a `sumo-gui`, a szimuláció lassítva fut.
Tanításhoz ne használd, mert lassú.
