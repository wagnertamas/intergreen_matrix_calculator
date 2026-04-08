#!/usr/bin/env python3
"""
Sweep Analysis — Egyedi futások rangsor vizualizáció.

Minden futás egyedileg jelenik meg. A szórást az utolsó 100 lépés
reward értékeiből számítja. A legjobb futás felül jelenik meg.

Kimenet: grid_results/sweep_analysis/
    01_top20_sweep_configs.png
    top_sweep_configs.csv

Használat:
    python misc/sweep_analysis.py
    python misc/sweep_analysis.py --top 30
    python misc/sweep_analysis.py --last-steps 50   # utolsó N lépés (default: 100)
    python misc/sweep_analysis.py --min-steps 50    # min. lépésszám a futáshoz
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

# --- Projekt root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# --- Stílus ---
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

ALGO_COLORS = {
    'qrdqn': '#e74c3c',
    'dqn':   '#3498db',
    'ppo':   '#2ecc71',
    'a2c':   '#f39c12',
}

REWARD_LABELS = {
    'speed_throughput': 'Speed+TP',
    'halt_ratio':       'HaltRatio',
    'co2_speedstd':     'CO2+Std',
}

WANDB_DIR = os.path.join(PROJECT_ROOT, "wandb")


# ================================================================
# ADAT BETÖLTÉS
# ================================================================

def extract_history_from_wandb_file(wandb_file: str) -> list:
    """
    Kinyeri az avg_reward értékeket lépésenként a .wandb bináris fájlból.
    Returns: list of float (reward per step)
    """
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal import datastore as ds_mod
    except ImportError:
        return []

    rewards = []
    try:
        ds = ds_mod.DataStore()
        ds.open_for_scan(wandb_file)
        while True:
            data = ds.scan_data()
            if data is None:
                break
            try:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.HasField('history'):
                    for item in record.history.item:
                        # nested_key is a RepeatedScalarContainer (list), not a string
                        key = item.nested_key[0] if len(item.nested_key) == 1 else item.key
                        if key == "avg_reward":
                            try:
                                rewards.append(float(json.loads(item.value_json)))
                            except Exception:
                                pass
            except Exception:
                pass
    except Exception:
        pass

    return rewards


def load_runs(wandb_dir: str, last_steps: int = 100, min_steps: int = 10) -> pd.DataFrame:
    """
    Beolvassa az összes WandB futást. Minden futáshoz kiszámítja az utolsó
    `last_steps` lépés átlagát és szórását.
    """
    records = []

    run_dirs = [e for e in os.scandir(wandb_dir) if e.name.startswith("run-")]
    total = len(run_dirs)
    print(f"  {total} run mappa megtalálva, feldolgozás...")

    for idx, entry in enumerate(run_dirs):
        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{total}")

        # --- Config kinyerése ---
        config_path = os.path.join(entry.path, "files", "config.yaml")
        cfg = {}
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path) as f:
                    raw = yaml.safe_load(f)
                for k, v in raw.items():
                    if isinstance(v, dict) and 'value' in v:
                        cfg[k] = v['value']
                    else:
                        cfg[k] = v
            except Exception:
                pass

        algorithm   = str(cfg.get("algorithm",   "unknown")).lower()
        reward_mode = str(cfg.get("reward_mode", "unknown"))
        num_layers  = cfg.get("num_layers",  None)
        layer_size  = cfg.get("layer_size",  None)
        total_ts    = cfg.get("total_timesteps", None)

        if algorithm == "unknown":
            continue

        # --- Run ID kinyerése (pl. "pjlthg18" a "run-20260405_194132-pjlthg18"-ból) ---
        parts = entry.name.split("-")
        run_id = parts[-1] if len(parts) >= 3 else entry.name

        # --- Lépésenkénti reward kinyerése ---
        wandb_bin = os.path.join(entry.path, f"run-{run_id}.wandb")
        rewards = []
        if os.path.exists(wandb_bin):
            rewards = extract_history_from_wandb_file(wandb_bin)

        if len(rewards) < min_steps:
            # Ha nincs elég lépés, próbálj summary-ból visszaesni
            summary_path = os.path.join(entry.path, "files", "wandb-summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    val = summary.get("avg_reward",
                          summary.get("episode/avg_reward",
                          summary.get("reward_smooth", None)))
                    if val is not None and isinstance(val, (int, float)):
                        rewards = [float(val)]
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue

        # --- Utolsó N lépés statisztikái ---
        tail = rewards[-last_steps:]
        mean_r = float(np.mean(tail))
        std_r  = float(np.std(tail))
        n_steps = len(rewards)

        records.append({
            "run_id":          run_id,
            "run_dir":         entry.name,
            "algorithm":       algorithm,
            "reward_mode":     reward_mode,
            "num_layers":      int(num_layers)  if num_layers  is not None else None,
            "layer_size":      int(layer_size)  if layer_size  is not None else None,
            "total_timesteps": int(total_ts)    if total_ts    is not None else None,
            "mean_reward":     mean_r,
            "std_reward":      std_r,
            "n_steps":         n_steps,
            "last_steps_used": len(tail),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
    return df


# ================================================================
# VIZUALIZÁCIÓ
# ================================================================

def make_label(row) -> str:
    algo    = row["algorithm"].upper()
    rmode   = REWARD_LABELS.get(row["reward_mode"], row["reward_mode"])
    nl      = row["num_layers"]
    ls      = row["layer_size"]
    arch    = f"{int(nl)}×{int(ls)}" if nl is not None and ls is not None else "?"
    run_id  = row["run_id"]
    return f"{run_id}  |  {algo}  |  {rmode}  |  {arch}"


def plot_top_n(df: pd.DataFrame, top_n: int, last_steps: int, out_dir: str) -> str:
    """
    Vízszintes oszlopdiagram — legjobb futás FELÜL.
    Szórás = az utolsó `last_steps` lépés szórása.
    """
    data = df.head(top_n).copy()
    # Matplotlib-ban az utolsó sor jelenik meg felül → megfordítjuk
    data = data.iloc[::-1].reset_index(drop=True)

    labels = [make_label(r) for _, r in data.iterrows()]
    means  = data["mean_reward"].values
    stds   = data["std_reward"].values
    n_steps = data["n_steps"].values
    colors = [ALGO_COLORS.get(a, "#888888") for a in data["algorithm"]]

    row_height = 0.42
    fig_height = max(7, top_n * row_height + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    ax.barh(
        range(len(data)),
        means,
        xerr=stds,
        color=colors,
        alpha=0.85,
        height=0.70,
        capsize=3,
        error_kw=dict(elinewidth=1.1, ecolor="#333333", capthick=1.1),
    )

    # Annotáció: lépésszám jobbra
    for i, (m, s, n) in enumerate(zip(means, stds, n_steps)):
        x_pos = m + s + 0.003
        ax.text(x_pos, i, f"n={n}", va="center", ha="left",
                fontsize=8, color="#555555")

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=8.5)

    ax.set_xlabel(f"Átlag Reward (utolsó {last_steps} lépés)")
    ax.set_title(f"Top {top_n} Sweep Futás  (átlag ± szórás, utolsó {last_steps} lépés)", pad=12)

    valid_means = means[np.isfinite(means)]
    valid_stds  = stds[np.isfinite(stds)]
    if len(valid_means) > 0:
        xmin = max(0.0, valid_means.min() - (valid_stds.max() if len(valid_stds) else 0) - 0.05)
        xmax = valid_means.max() + (valid_stds.max() if len(valid_stds) else 0) + 0.10
        ax.set_xlim(xmin, xmax)

    # Jelmagyarázat
    present_algos = data["algorithm"].unique()
    legend_patches = [
        matplotlib.patches.Patch(color=ALGO_COLORS.get(a, "#888888"), label=a.upper())
        for a in ALGO_COLORS
        if a in present_algos
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", framealpha=0.85,
                  title="Algoritmus")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "01_top20_sweep_configs.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Sweep Analysis — Egyedi futások")
    parser.add_argument("--top",        type=int, default=20,  help="Hány top futás (default: 20)")
    parser.add_argument("--last-steps", type=int, default=100, help="Utolsó N lépés a szóráshoz (default: 100)")
    parser.add_argument("--min-steps",  type=int, default=10,  help="Min. lépésszám (default: 10)")
    parser.add_argument("--out",        type=str, default="grid_results/sweep_analysis",
                        help="Kimeneti almappa (default: grid_results/sweep_analysis)")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] WandB futások beolvasása: {WANDB_DIR}")
    print(f"      Szórás az utolsó {args.last_steps} lépésből, min. {args.min_steps} lépés")
    df = load_runs(WANDB_DIR, last_steps=args.last_steps, min_steps=args.min_steps)
    print(f"      {len(df)} érvényes futás találva")

    if df.empty:
        print("[ERROR] Nincs adat! Ellenőrizd a wandb mappát.")
        sys.exit(1)

    # CSV mentés (teljes lista)
    csv_path = os.path.join(out_dir, "top_sweep_configs.csv")
    df.to_csv(csv_path, index=False)
    print(f"[2/3] CSV mentve: {csv_path}")

    top_n = min(args.top, len(df))
    print(f"[3/3] Diagram generálása (top {top_n})...")
    png_path = plot_top_n(df, top_n=top_n, last_steps=args.last_steps, out_dir=out_dir)
    print(f"      PNG: {png_path}")

    # Top 5 konzolra
    print(f"\n  TOP 5 FUTÁS (utolsó {args.last_steps} lépés alapján):")
    hdr = f"  {'Rang':<5} {'Run ID':<12} {'Algo':<8} {'Reward mód':<16} {'Arch':<8} {'Átlag':>7} {'Szórás':>7} {'Lépés':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in df.head(5).iterrows():
        nl = r['num_layers']
        ls = r['layer_size']
        arch = f"{int(nl)}×{int(ls)}" if nl is not None and ls is not None else "?"
        print(f"  {int(r['rank']):<5} {r['run_id']:<12} {r['algorithm'].upper():<8} "
              f"{r['reward_mode']:<16} {arch:<8} {r['mean_reward']:>7.4f} "
              f"{r['std_reward']:>7.4f} {int(r['n_steps']):>6}")


if __name__ == "__main__":
    main()
