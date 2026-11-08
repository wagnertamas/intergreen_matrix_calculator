# Jutalomvektor: célfüggvények

A többcélú tanulásban a jutalom nem egy szám, hanem egy **vektor**: minden
elem egy-egy célt mér. A célok általában ellentmondanak egymásnak (pl. kevés
várakozás vs. kevés fázisváltás), ezért nincs egyetlen „legjobb” vezérlés,
hanem **Pareto-optimális** vezérlések halmaza van.

A célokat a környezet létrehozásakor választod ki, a jutalomvektor ebben a
sorrendben tartalmazza őket:

```python
env = TJunctionMOEnv(objectives=("queue", "fairness", "switches"))   # alapértelmezés
env.objective_names    # ('queue', 'fairness', 'switches')
```

## Beépített célfüggvények

Konvenció: **nagyobb = jobb** (a költségek negatív előjelűek). Minden érték a
lépés másodperceire összegzett mennyiség, így az epizód összjutalma a teljes
időre vett integrál, függetlenül attól, hogy a lépések milyen hosszúak. A
`skála` csak arra szolgál, hogy a tipikus értékek 1 körüliek legyenek.

| Név | Képlet (lépésenként) | Skála | Mit jelent |
|---|---|---|---|
| `queue` | −Σ_ágak Σ_másodpercek becsült járműszám | 0,01 | a teljes késleltetés közelítése (kevés jármű a csomópont előtt) |
| `fairness` | −max_ág Σ_másodpercek becsült járműszám | 0,02 | méltányosság: a legrosszabb helyzetű ág se várjon sokat |
| `switches` | −1, ha volt fázisváltás | 1 | stabil vezérlés, kevés elveszett idő az átmenetekben |
| `throughput` | a stopvonali hurkokon áthaladt járművek | 0,1 | áteresztőképesség (főleg túltelített helyzetben számít) |
| `stopline_occupancy` | −Σ stopvonali foglalt másodpercek | 0,1 | a stopvonalnál álló járművek közelítése |

Mindegyik csak hurokdetektor-adatokból (és a vezérlő saját állapotából) számol.
A listát a programból is lekérheted:

```python
from sumo_morl import list_objectives
print(list_objectives())
```

## Saját célfüggvény

Egy célfüggvény egy függvény `(snap, ctx) -> float`, ahol

* `snap` a lépés hurokadatai (`DetectorSnapshot`, lásd
  [Megfigyelés](02_megfigyeles.md)),
* `ctx` egy szótár: `phase_changed`, `phase`, `duration`, `time`, `forced`.

```python
from sumo_morl import TJunctionMOEnv, register_objective

@register_objective("side_road_queue", scale=0.02,
                    doc="A déli mellékág becsült járműszáma (min.)")
def side_road_queue(snap, ctx):
    return -snap.approach_vehicle_seconds["S"]

env = TJunctionMOEnv(objectives=("queue", "side_road_queue"))
```

Fontos: a célfüggvényben **ne** kérdezd le a SUMO-t közvetlenül (TraCI), mert
akkor valóságban nem mérhető adatot használnál.

## Skalarizálás

A legtöbb RL algoritmus skalár jutalmat vár. A két legegyszerűbb módszer:

* **Lineáris skalarizálás** (`LinearScalarization` wrapper): `r = w · r_vec`,
  rögzített súlyvektorral. Különböző súlyokkal külön ágenseket tanítva
  feltérképezhető a Pareto-front (konvex része).
* **Súly-kondicionált ágens** (`WeightConditioned` wrapper): epizódonként
  véletlen súlyt sorsol, és azt a megfigyelés végére fűzi. Egyetlen háló tanulja
  meg az összes kompromisszumot; kiértékeléskor a kívánt súlyt kell megadni.

Mindkettő az eredeti vektort az `info["vector_reward"]` alatt is visszaadja.

Fejlettebb MORL módszerek (pl. Pareto Q-learning, Envelope Q-learning,
nemlineáris hasznossági függvények) a vektor-jutalommal közvetlenül
megvalósíthatók.
