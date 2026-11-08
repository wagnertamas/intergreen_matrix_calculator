"""
MORL segédfüggvények: skalarizálás, súlyvektorok, Pareto-front, hipertérfogat,
epizód futtatás és kiértékelés, ábrázolás.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np


# =========================================================================== wrapper
class LinearScalarization(gym.Wrapper):
    """A vektor-jutalmat súlyozott összeggé alakítja: ``r = w · r_vec``.

    Így bármely egycélú RL könyvtár (pl. Stable-Baselines3) használható. Az
    eredeti vektor az ``info["vector_reward"]`` alatt marad elérhető.

    Args:
        env: a többcélú környezet.
        weights: súlyvektor (célfüggvényenként egy szám, tipikusan összegük 1).
    """

    def __init__(self, env: gym.Env, weights: Sequence[float]):
        super().__init__(env)
        self.weights = np.asarray(weights, dtype=np.float32)
        if self.weights.shape != (env.unwrapped.reward_dim,):
            raise ValueError(f"{env.unwrapped.reward_dim} súly kell, kaptam: {self.weights.shape}")

    def set_weights(self, weights: Sequence[float]) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info = dict(info)
        info["vector_reward"] = r
        return obs, float(np.dot(self.weights, r)), term, trunc, info


class WeightConditioned(gym.ObservationWrapper):
    """A súlyvektort hozzáfűzi a megfigyeléshez (súly-kondicionált ágensekhez).

    Minden ``reset``-kor új súlyt sorsol (vagy fixet használ), és a jutalmat
    is ezzel skalarizálja. Egyetlen hálóval így a teljes Pareto-front tanulható.
    """

    def __init__(self, env: gym.Env, weights: Optional[Sequence[float]] = None, seed: Optional[int] = None):
        super().__init__(env)
        self.k = env.unwrapped.reward_dim
        self.fixed = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.weights = self.fixed if self.fixed is not None else random_weights(self.k, 1, self.rng)[0]
        low = np.concatenate([env.observation_space.low, np.zeros(self.k)]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, np.ones(self.k)]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kw):
        if self.fixed is None:
            self.weights = random_weights(self.k, 1, self.rng)[0]
        return super().reset(**kw)

    def observation(self, obs):
        return np.concatenate([obs, self.weights]).astype(np.float32)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info = dict(info)
        info["vector_reward"] = r
        return self.observation(obs), float(np.dot(self.weights, r)), term, trunc, info


# =========================================================================== súlyok
def equally_spaced_weights(n_objectives: int, n_divisions: int) -> np.ndarray:
    """Egyenletes rács a súly-szimplexen (összeg = 1).

    Pl. ``equally_spaced_weights(2, 4)`` ->
    ``[[0,1], [.25,.75], [.5,.5], [.75,.25], [1,0]]``.
    """
    pts = []
    for c in itertools.product(range(n_divisions + 1), repeat=n_objectives):
        if sum(c) == n_divisions:
            pts.append(np.array(c, dtype=float) / n_divisions)
    return np.array(sorted(pts, key=lambda p: tuple(p)))


def random_weights(n_objectives: int, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Egyenletes eloszlású véletlen súlyvektorok a szimplexen (Dirichlet(1))."""
    rng = rng or np.random.default_rng()
    return rng.dirichlet(np.ones(n_objectives), size=n).astype(np.float32)


# =========================================================================== Pareto
def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Igaz azokra a pontokra, amelyek nem dominált (Pareto-optimális) pontok.

    Konvenció: minden célt **maximalizálunk** (a környezet jutalmai ilyenek).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def pareto_front(points: np.ndarray) -> np.ndarray:
    """A nem dominált pontok (lásd :func:`pareto_mask`)."""
    pts = np.asarray(points, dtype=float)
    return pts[pareto_mask(pts)]


def hypervolume(points: np.ndarray, ref: Sequence[float]) -> float:
    """Hipertérfogat (maximalizálás, ``ref`` = minden pontnál rosszabb referencia).

    Pontos, rekurzív szeletelős algoritmus; kis pontszámra (< néhány száz) és
    2–4 célra bőven elég gyors.
    """
    ref = np.asarray(ref, dtype=float)
    pts = np.asarray(points, dtype=float)
    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pareto_front(pts)
    return float(_hv(pts - ref))


def _hv(p: np.ndarray) -> float:
    if p.shape[1] == 1:
        return float(p.max())
    p = p[np.argsort(-p[:, -1])]
    vol, prev = 0.0, 0.0
    # szeletek az utolsó koordináta mentén, alulról felfelé
    levels = np.unique(p[:, -1])
    for lvl in levels:
        sub = p[p[:, -1] >= lvl][:, :-1]
        vol += _hv(pareto_front(sub)) * (lvl - prev)
        prev = lvl
    return vol


# =========================================================================== futtatás
@dataclass
class EpisodeResult:
    vector_return: np.ndarray
    objective_names: Sequence[str]
    steps: int
    ground_truth: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        d = {n: float(v) for n, v in zip(self.objective_names, self.vector_return)}
        d.update({f"gt_{k}": v for k, v in self.ground_truth.items()})
        return d


def run_episode(env: gym.Env, policy: Callable, seed: Optional[int] = None, record: bool = False,
                gamma: float = 1.0) -> EpisodeResult:
    """Egy teljes epizód lefuttatása egy vezérlővel.

    Args:
        env: a (nem skalarizált) többcélú környezet.
        policy: ``policy(obs, info) -> akció`` függvény (vagy :class:`~sumo_morl.policies.Policy`).
            Stable-Baselines3 modellhez lásd :func:`sb3_policy`.
        seed: a reset magja (azonos mag = azonos forgalom).
        record: lépésenkénti napló (``history``) gyűjtése.
        gamma: diszkont (1.0 = egyszerű összeg).
    """
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    unwrapped = env.unwrapped
    ret = np.zeros(unwrapped.reward_dim)
    disc, steps, hist = 1.0, 0, []
    done = False
    while not done:
        a = policy(obs, info)
        obs, r, term, trunc, info = env.step(a)
        r_vec = np.asarray(info.get("vector_reward", r), dtype=float)
        ret += disc * r_vec
        disc *= gamma
        steps += 1
        if record:
            hist.append({"time": info["time"], "action": int(a), "phase": info["phase"],
                         **{f"r_{k}": v for k, v in info["objectives"].items()},
                         **{f"veh_{k}": v for k, v in info["detectors"].approach_vehicles.items()}})
        done = term or trunc
    return EpisodeResult(ret, unwrapped.objective_names, steps, dict(info.get("ground_truth", {})), hist)


def evaluate_policy(env: gym.Env, policy: Callable, n_episodes: int = 3,
                    seeds: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Több epizódos kiértékelés. Visszaadja a vektor-hozamok átlagát és szórását.

    Azonos ``seeds`` listával a különböző vezérlők azonos forgalmon hasonlíthatók össze.
    """
    seeds = list(seeds) if seeds is not None else list(range(n_episodes))
    results = [run_episode(env, policy, seed=s) for s in seeds]
    R = np.array([r.vector_return for r in results])
    gt_keys = results[0].ground_truth.keys()
    gt = {k: float(np.nanmean([r.ground_truth[k] for r in results])) for k in gt_keys}
    return {"mean": R.mean(axis=0), "std": R.std(axis=0), "returns": R, "ground_truth": gt,
            "objective_names": list(env.unwrapped.objective_names)}


def sb3_policy(model, weights: Optional[Sequence[float]] = None, deterministic: bool = True) -> Callable:
    """Stable-Baselines3 modellből ``policy(obs, info)`` függvényt készít.

    Ha a modellt :class:`WeightConditioned` környezeten tanítottuk, add meg a
    ``weights`` súlyt is, ezt fűzzük a megfigyeléshez.
    """
    w = None if weights is None else np.asarray(weights, dtype=np.float32)

    def _pol(obs, info):
        o = obs if w is None else np.concatenate([obs, w]).astype(np.float32)
        a, _ = model.predict(o, deterministic=deterministic)
        return int(a)

    return _pol


# =========================================================================== ábra
def plot_pareto(points: np.ndarray, objective_names: Sequence[str], labels: Optional[Sequence[str]] = None,
                ax=None, highlight_front: bool = True, title: str = "Pareto-front"):
    """2D Pareto-ábra (több cél esetén minden célpárra külön ábra).

    Returns:
        A matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=float)
    k = pts.shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    if ax is None:
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
        axes = axes[0]
    else:
        axes = [ax]
        fig = ax.figure
    mask = pareto_mask(pts) if highlight_front else np.zeros(len(pts), bool)
    for a, (i, j) in zip(axes, pairs):
        a.scatter(pts[~mask, i], pts[~mask, j], c="0.6", label="dominált")
        a.scatter(pts[mask, i], pts[mask, j], c="C3", label="Pareto-optimális")
        if labels is not None:
            for p, lab in zip(pts, labels):
                a.annotate(str(lab), (p[i], p[j]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel(objective_names[i])
        a.set_ylabel(objective_names[j])
        a.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
