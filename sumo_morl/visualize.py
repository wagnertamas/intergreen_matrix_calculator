"""
Ábrák a forgatókönyvről és az epizódokról (matplotlib).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence

import numpy as np

STATE_COLORS = {"G": "#2ca02c", "g": "#2ca02c", "y": "#ffbf00", "u": "#ff7f0e", "r": "#d62728"}


def _shapes(net_file: str):
    root = ET.parse(net_file).getroot()
    lanes, internal = {}, {}
    for e in root.findall("edge"):
        for l in e.findall("lane"):
            pts = np.array([list(map(float, p.split(","))) for p in l.get("shape").split()])
            (internal if e.get("function") == "internal" else lanes)[l.get("id")] = pts
    via = {}
    for c in root.findall("connection"):
        if c.get("linkIndex") is not None and c.get("via"):
            via[int(c.get("linkIndex"))] = c.get("via")
    # összefűzött belső útvonalak (pl. :C_1_0 + :C_6_0)
    paths = {}
    for idx, v in via.items():
        pts = internal[v]
        changed = True
        while changed:
            changed = False
            for lid, p2 in internal.items():
                if lid != v and np.linalg.norm(p2[0] - pts[-1]) < 0.3 and lid not in [x for x in via.values()]:
                    pts = np.vstack([pts, p2[1:]])
                    changed = True
                    break
        paths[idx] = pts
    return lanes, paths


def _draw_lanes(ax, lanes, color="0.8"):
    from matplotlib.patches import Polygon
    from shapely.geometry import LineString

    for pts in lanes.values():
        poly = LineString(pts).buffer(1.55, cap_style=2)
        ax.add_patch(Polygon(np.array(poly.exterior.coords), closed=True, fc=color, ec="0.6", lw=0.3,
                             zorder=1))


def plot_scenario(scenario=None, show_detectors: bool = True, figsize=(12, 5.5)):
    """A csomópont rajza két nézetben (teljes + nagyított): sávok, mozgások,
    hurokdetektorok."""
    import matplotlib.pyplot as plt
    from .scenario import TJunctionScenario

    scenario = scenario or TJunctionScenario()
    lanes, paths = _shapes(scenario.net_file)
    cx, cy = np.mean([p.mean(axis=0) for p in paths.values()], axis=0)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]})
    colors = {"entry": "C0", "advance": "C1", "stopline": "C3"}
    kind_hu = {"entry": "belépő (entry)", "advance": "előjelző (advance)", "stopline": "stopvonali (stopline)"}
    for ax, zoom in ((ax0, None), (ax1, 55.0)):
        _draw_lanes(ax, lanes)
        for pts in paths.values():
            ax.plot(pts[:, 0], pts[:, 1], color="0.35", lw=1, zorder=3)
            ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color="0.35"))
        if show_detectors:
            seen = set()
            for d in scenario.detectors:
                pts = lanes[d.lane_id]
                seg = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
                x, y = np.interp(d.pos, seg, pts[:, 0]), np.interp(d.pos, seg, pts[:, 1])
                ax.plot(x, y, "s", color=colors[d.kind], ms=5 if zoom is None else 8, zorder=5,
                        label=kind_hu[d.kind] if d.kind not in seen else None)
                seen.add(d.kind)
                if zoom is not None and abs(x - cx) < zoom and abs(y - cy) < zoom:
                    ax.annotate(d.id, (x, y), fontsize=7, xytext=(5, 5), textcoords="offset points")
        ax.set_aspect("equal")
        if zoom is not None:
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cy - zoom, cy + zoom * 0.5)
            ax.set_title("Nagyítás a csomópontra")
        else:
            ax.set_title("T-csomópont (teljes hálózat)")
            for txt, lid, end in (("Nyugat (W)", "W_C_0", 0), ("Kelet (E)", "E_C_0", 0), ("Dél (S)", "S_C_0", 0)):
                ax.annotate(txt, lanes[lid][end], fontsize=10, fontweight="bold", xytext=(0, 8),
                            textcoords="offset points", ha="center")
            ax.legend(loc="lower left", title="hurokdetektor", fontsize=8)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    fig.tight_layout()
    return fig


def plot_phases(scenario=None, figsize=None):
    """Fázisonként egy ábra: melyik mozgás kap zöldet."""
    import matplotlib.pyplot as plt
    from .scenario import TJunctionScenario

    scenario = scenario or TJunctionScenario()
    plan = scenario.plan
    lanes, paths = _shapes(scenario.net_file)
    n = plan.n_phases
    fig, axes = plt.subplots(1, n, figsize=figsize or (4 * n, 4))
    cx, cy = np.mean([p.mean(axis=0) for p in paths.values()], axis=0)
    for p, ax in enumerate(np.atleast_1d(axes)):
        state = plan.green_state(p)
        for pts in lanes.values():
            ax.plot(pts[:, 0], pts[:, 1], color="0.85", lw=7, solid_capstyle="butt")
        for idx, pts in paths.items():
            green = state[idx] in "Gg"
            col = STATE_COLORS["G"] if green else "0.8"
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.5 if green else 1, zorder=3 if green else 2)
            ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=col))
        ax.set_xlim(cx - 30, cx + 30)
        ax.set_ylim(cy - 30, cy + 20)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        name = plan.phase_names[p] if p < len(plan.phase_names) else ""
        ax.set_title(f"Fázis {p}\n{name}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_intergreen(plan, ax=None):
    """A jelzőcsoportok közötti közbenső idő mátrix hőtérképe."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    m = plan.intergreen
    ax.imshow(np.where(np.isnan(m), -1, m), cmap="Oranges", vmin=0)
    names = [g.name for g in plan.groups]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, "-" if np.isnan(m[i, j]) else int(m[i, j]), ha="center", va="center")
    ax.set_xlabel("belépő (zöldet kap)")
    ax.set_ylabel("kiürítő (zöldje véget ér)")
    ax.set_title("Közbenső idők [s]")
    return ax


def plot_episode(history: Sequence[Dict], phase_names: Optional[List[str]] = None):
    """Egy rögzített epizód (``run_episode(..., record=True).history``) ábrája:
    fázisok az időben, becsült járműszám ágankként és a jutalomkomponensek."""
    import matplotlib.pyplot as plt

    t = np.array([h["time"] for h in history])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].step(t, [h["phase"] for h in history], where="pre")
    axes[0].set_ylabel("fázis")
    if phase_names:
        axes[0].set_yticks(range(len(phase_names)))
    for k in [k for k in history[0] if k.startswith("veh_")]:
        axes[1].plot(t, [h[k] for h in history], label=k[4:])
    axes[1].set_ylabel("becsült jármű/ág")
    axes[1].legend()
    for k in [k for k in history[0] if k.startswith("r_")]:
        axes[2].plot(t, [h[k] for h in history], label=k[2:])
    axes[2].set_ylabel("jutalom")
    axes[2].set_xlabel("idő [s]")
    axes[2].legend()
    fig.tight_layout()
    return fig


def plot_signal_timeline(signal_log, plan, t_from: int = 0, t_to: Optional[int] = None, ax=None):
    """Klasszikus jelzésterv-ábra (sávos diagram): jelzőcsoportonként a jelzés
    színe az időben. ``signal_log`` = ``env.signal_log`` (``record_signals=True``)."""
    import matplotlib.pyplot as plt

    log = [(t, s) for t, s in signal_log if t >= t_from and (t_to is None or t <= t_to)]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 0.6 * len(plan.groups) + 1.2))
    for gi, g in enumerate(plan.groups):
        li = g.links[0]
        for t, s in log:
            ax.barh(gi, 1, left=t - 1, color=STATE_COLORS.get(s[li], "0.5"), edgecolor="none", height=0.6)
    ax.set_yticks(range(len(plan.groups)), [g.name for g in plan.groups])
    ax.invert_yaxis()
    ax.set_xlabel("idő [s]")
    ax.set_title("Jelzésterv (zöld / sárga / piros-sárga / piros)")
    ax.set_xlim(log[0][0] - 1 if log else 0, log[-1][0] if log else 1)
    ax.figure.tight_layout()
    return ax
