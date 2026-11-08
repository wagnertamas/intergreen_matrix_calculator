# Megfigyelés: hurokdetektorok (csak mérhető adatok)

Egy valódi csomópontban a vezérlő **nem látja** a járműveket: nem tudja, hány
autó áll a sorban, ki mióta vár, vagy hová tart. Amit tud, azt az úttestbe
épített **indukcióshurok-detektorok** mérik. Hogy az ágens a valóságba is
átvihető legyen, a `sumo_morl` megfigyelése és jutalma **kizárólag** ilyen
adatokból készül.

## A detektorok elhelyezése

Minden bejövő sávon (4 sáv) három hurok van, összesen 12:

```
 ág eleje                                                    stopvonal
 |--[entry]--------------------------------[advance]---------[stopline]|
    10 m az ág elejétől                     40 m-re a          3 m-re a
                                            stopvonaltól       stopvonaltól
```

| Típus | Mire jó |
|---|---|
| `entry` (belépő) | a csomópont felé érkező járművek számlálása |
| `advance` (előjelző) | közeledő járművek érzékelése; ha tartósan foglalt, a sor már idáig ér |
| `stopline` (stopvonali) | a stopvonalon áthaladó járművek számlálása, a sor eleje |

A detektorok azonosítója `<sáv>_<típus>`, pl. `E_C_1_stopline`.

## Mit mér egy hurok?

| Mennyiség | Jelentés | Valóságban |
|---|---|---|
| `count` | a lépés alatt áthaladt járművek száma | impulzusszámlálás |
| `occupancy` | az idő hányadában volt jármű a hurok felett (0..1) | foglaltság |
| `speed` | átlagsebesség a hurkon (m/s), jármű nélkül a megengedett sebesség | dupla hurokkal mérhető |
| `gap` | a legutóbbi érzékelés óta eltelt idő (s) | időrés, a járműkövető vezérlők alapja |

### Becsült járműszám ágankként (bemenet–kimenet módszer)

Az `entry` és a `stopline` hurkok számlálóiból folyamatosan számoljuk, hány jármű
van a két hurok között: `N(t) = N(t−1) + belépők − kilépők`. Ez a klasszikus,
gyakorlatban is használt sorhossz-becslés. Tesztjeink szerint a becslés
legfeljebb 1–3 járművel tér el a SUMO-beli valós értéktől.

## A megfigyelésvektor

A `obs` egy 55 elemű, `[0, 1]` tartományba normalizált `float32` vektor:

| Elemek | Darab | Normalizálás |
|---|---|---|
| minden detektorra: `count`, `occupancy`, `speed`, `gap` | 12 × 4 = 48 | forgalom / 1800 jármű/óra; foglaltság; sebesség / sebességkorlát; időrés / 60 s |
| ágankénti becsült járműszám (W, E, S) | 3 | / az ág tárolókapacitása (~7 m / jármű) |
| aktuális fázis (one-hot) | 3 | 0 vagy 1 |
| az aktuális zöld eddigi hossza | 1 | / 120 s |

Az elemek neveit az `env.observation_names()` adja vissza, pl.
`'E_C_1_stopline.occupancy'`, `'approach_S.vehicles'`, `'phase_2'`.

Ha saját megfigyelést szeretnél (pl. csak a stopvonali hurkokat), írj egy
`gymnasium.ObservationWrapper`-t, és használd az `info["detectors"]` nyers adatait:

```python
import gymnasium as gym
import numpy as np

class OnlyStopline(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.idx = [i for i, n in enumerate(env.unwrapped.observation_names())
                    if "stopline" in n or n.startswith("phase_")]
        self.observation_space = gym.spaces.Box(0, 1, (len(self.idx),), np.float32)

    def observation(self, obs):
        return obs[self.idx]
```

## A nyers detektoradatok: `info["detectors"]`

Egy `DetectorSnapshot` objektum az utolsó lépés aggregált adataival:

```python
snap = info["detectors"]
snap.duration                         # a lépés hossza [s]
snap.count, snap.occupancy, snap.speed, snap.gap   # numpy tömbök, detektoronként
snap.approach_vehicles                # {'W': 12.0, 'E': 4.0, 'S': 7.0} – becslés a lépés végén
snap.approach_vehicle_seconds         # ugyanez a lépés másodperceire összegezve
snap.select(kind="stopline")          # bool maszk a szűréshez
snap.select(lane_id="E_C_1")
snap.by_id()["E_C_1_stopline"]        # {'count': 2.0, 'occupancy': 0.31, ...}
```

## Amit szándékosan NEM kap meg az ágens

* jármű-azonosítók, pozíciók, úticélok,
* pontos sorhossz és várakozási idő,
* emisszió, utazási idő.

Ezeket a környezet `ground_truth=True` (alapértelmezett) beállítással az
`info["ground_truth"]` alatt kiírja, de **csak kiértékeléshez** (pl. a
dolgozatban annak bemutatására, hogy a tanult vezérlő tényleg csökkenti-e a
késést). A megfigyelésbe vagy a jutalomba beépíteni „csalás”, mert a
valóságban nem elérhető.

| `info["ground_truth"]` kulcs | Jelentés |
|---|---|
| `halting_vehicle_seconds` | álló járművek száma másodpercenként összegezve (a bejövő ágakon) |
| `mean_delay_per_vehicle` | ugyanez elosztva az elindult járművek számával [s] |
| `mean_travel_time` | a célba ért járművek átlagos utazási ideje [s] |
| `co2_g` | CO2-kibocsátás a bejövő ágakon [g] |
| `departed`, `arrived` | elindult / célba ért járművek |
| `waiting_to_enter` | a hálózatba be sem tudó (visszatorlódott) járművek |
