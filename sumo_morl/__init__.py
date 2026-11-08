"""
sumo_morl – többcélú megerősítéses tanulás (MORL) jelzőlámpa-vezérléshez SUMO-ban.

Ez az API-referencia a könyvtár összes nyilvános osztályát és függvényét leírja.
A használat bevezetője a *Kézikönyv* (``kezikonyv.pdf``), a belső felépítés a
*Programfelépítés* (``programfelepites.pdf``).

Gyors kezdés::

    from sumo_morl import TJunctionMOEnv

    env = TJunctionMOEnv(objectives=("queue", "fairness", "switches"))
    obs, info = env.reset(seed=0)
    obs, reward_vec, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

Modulok:

* `sumo_morl.env` – a Gymnasium környezet (`TJunctionMOEnv`)
* `sumo_morl.objectives` – célfüggvények, saját cél regisztrálása
* `sumo_morl.detectors` – hurokdetektorok és az aggregált mérések (`DetectorSnapshot`)
* `sumo_morl.traffic` – forgalmi igény (`Demand`)
* `sumo_morl.morl` – skalarizálás, súlyok, Pareto-front, hipertérfogat, kiértékelés
* `sumo_morl.policies` – összehasonlító (klasszikus) vezérlők
* `sumo_morl.visualize` – ábrák
* `sumo_morl.scenario` – a T-csomópont forgatókönyv
* `sumo_morl.intergreen` – jelzőcsoportok, fázisok, átmenetek a közbenső időkből
* `sumo_morl.network` – a hálózat generálása
* `sumo_morl.sumo_utils` – SUMO-binárisok megkeresése
"""
from .detectors import DetectorInfo, DetectorSnapshot
from .env import DEFAULT_OBJECTIVES, TJunctionMOEnv
from .intergreen import SignalPlan, build_signal_plan
from .morl import (
    EpisodeResult,
    LinearScalarization,
    WeightConditioned,
    equally_spaced_weights,
    evaluate_policy,
    hypervolume,
    pareto_front,
    pareto_mask,
    plot_pareto,
    random_weights,
    run_episode,
    sb3_policy,
)
from .objectives import OBJECTIVES, list_objectives, register_objective
from .policies import ActuatedPolicy, FixedTimePolicy, LongestQueuePolicy, Policy, RandomPolicy
from .scenario import TJunctionScenario
from .traffic import Demand

__version__ = "0.1.0"


def make_env(**kwargs):
    """Rövidítés: ``TJunctionMOEnv(**kwargs)``."""
    return TJunctionMOEnv(**kwargs)


try:  # regisztráció a gymnasium-ba: gym.make("sumo_morl/TJunction-v0")
    import gymnasium as _gym

    _gym.register(id="sumo_morl/TJunction-v0", entry_point="sumo_morl.env:TJunctionMOEnv",
                  disable_env_checker=True)  # a vektor-jutalom miatt
except Exception:  # pragma: no cover
    pass

__all__ = [
    "TJunctionMOEnv", "TJunctionScenario", "Demand", "SignalPlan", "build_signal_plan",
    "DetectorInfo", "DetectorSnapshot", "DEFAULT_OBJECTIVES", "OBJECTIVES", "list_objectives",
    "register_objective", "LinearScalarization", "WeightConditioned", "equally_spaced_weights",
    "random_weights", "pareto_mask", "pareto_front", "hypervolume", "plot_pareto", "run_episode",
    "evaluate_policy", "sb3_policy", "EpisodeResult", "Policy", "RandomPolicy", "FixedTimePolicy",
    "ActuatedPolicy", "LongestQueuePolicy", "make_env",
]
