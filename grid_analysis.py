#!/usr/bin/env python3
"""
Grid Search comprehensive analysis — WandB data.

4 algorithms × 3 reward modes × 15 network sizes × 3 repeats = 540 runs

Output plots (grid_results/):
  01-15: Summary heatmaps, boxplots, bar charts, variance analysis
  curves/: Smooth learning curves with confidence bands (all groupings)
"""

import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Global plot style ---
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 17,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': False,
})

CACHE_FILE = "grid_results/wandb_cache.json"

# --- Constants ---
ALGO_ORDER = ['qrdqn', 'dqn', 'ppo', 'a2c']
REWARD_ORDER = ['speed_throughput', 'halt_ratio', 'co2_speedstd']
REWARD_LABELS = {'speed_throughput': 'Speed+Throughput', 'halt_ratio': 'HaltRatio', 'co2_speedstd': 'CO2+SpeedStd'}
REWARD_SHORT_ORDER = [REWARD_LABELS[r] for r in REWARD_ORDER]

ALGO_COLORS = {'qrdqn': '#e74c3c', 'dqn': '#3498db', 'ppo': '#2ecc71', 'a2c': '#f39c12'}
REWARD_COLORS = {'Speed+Throughput': '#e74c3c', 'HaltRatio': '#3498db', 'CO2+SpeedStd': '#2ecc71'}
LAYER_COLORS = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}
NEURON_COLORS = {8: '#e74c3c', 16: '#e67e22', 32: '#2ecc71', 64: '#3498db', 128: '#9b59b6'}


# ================================================================
# DATA LOADING
# ================================================================
def fetch_wandb_data(project="sumo-rl-stat", entity=None):
    """Fetch all runs from WandB with caching."""

    if os.path.exists(CACHE_FILE):
        print(f"Loading cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            cached = json.load(f)
        df = pd.DataFrame(cached['records'])
        histories = {k: pd.DataFrame(v) for k, v in cached.get('histories', {}).items()}
        print(f"  {len(df)} runs, {len(histories)} histories from cache")
        return df, histories

    import wandb
    api = wandb.Api()
    if not entity:
        entity = api.default_entity

    project_path = f"{entity}/{project}"
    print(f"Fetching WandB data: {project_path} ...")
    runs = list(api.runs(project_path))
    print(f"  {len(runs)} runs found")

    records = []
    run_objects = {}

    for i, run in enumerate(runs):
        name = run.name or ""
        match = re.match(r'^(\w+?)_(speed_throughput|halt_ratio|co2_speedstd)_(\d+)x(\d+)_(\d+)_(.+)$', name)
        if not match:
            continue

        algo, reward, layers, neurons, rep, junction = match.groups()
        layers, neurons, rep = int(layers), int(neurons), int(rep)
        summary = run.summary._json_dict
        avg_reward = summary.get('avg_reward', None)

        records.append({
            'name': name, 'algorithm': algo, 'reward_mode': reward,
            'layers': layers, 'neurons': neurons, 'rep': rep,
            'junction': junction, 'final_avg_reward': avg_reward,
            'state': run.state, 'net_label': f"{layers}x{neurons}",
            'net_params': layers * neurons,
        })
        run_objects[name] = run
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(runs)} summaries processed...")

    df = pd.DataFrame(records)
    print(f"Total: {len(df)} runs loaded")

    # Parallel history download (ALL runs)
    histories = {}
    if len(df) > 0:
        history_names = set(df['name'])
        print(f"Downloading history for {len(history_names)} runs (parallel)...")

        def fetch_history(name):
            try:
                hist = run_objects[name].history(keys=['avg_reward', 'global_step'], samples=500)
                if not hist.empty:
                    return name, hist
            except Exception:
                pass
            return name, None

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(fetch_history, n): n for n in history_names if n in run_objects}
            done = 0
            for future in as_completed(futures):
                name, hist = future.result()
                if hist is not None:
                    histories[name] = hist
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(history_names)} histories done...")

        print(f"  {len(histories)} histories loaded")

    # Save cache
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'records': records,
            'histories': {k: v.to_dict(orient='list') for k, v in histories.items()},
        }, f)
    print(f"Cache saved: {CACHE_FILE}")
    return df, histories


# ================================================================
# SMOOTH CURVE HELPERS
# ================================================================
def smooth_curve(y, sigma=5):
    """Gaussian smoothing for a 1D array."""
    return gaussian_filter1d(y.astype(float), sigma=sigma)


def plot_smooth_band(ax, curves_list, color, label, sigma=5, alpha_fill=0.15):
    """
    Plot mean ± std band from a list of 1D arrays.
    Applies Gaussian smoothing for a publication-quality look.
    Uses NaN-padding to max length (not truncation to min).
    """
    if not curves_list:
        return
    max_len = max(len(c) for c in curves_list)
    # Pad shorter curves with NaN instead of truncating all to min
    arr = np.full((len(curves_list), max_len), np.nan)
    for i, c in enumerate(curves_list):
        arr[i, :len(c)] = c

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)

    mean_s = smooth_curve(mean, sigma=sigma)
    std_s = smooth_curve(std, sigma=sigma)

    x = np.arange(max_len)
    ax.plot(x, mean_s, color=color, label=label, linewidth=2)
    ax.fill_between(x, mean_s - std_s, mean_s + std_s, color=color, alpha=alpha_fill)


def get_curves_for_filter(df, histories, filter_dict):
    """Get avg_reward curves for runs matching filter criteria."""
    mask = pd.Series(True, index=df.index)
    for col, val in filter_dict.items():
        mask &= (df[col] == val)
    curves = []
    for _, row in df[mask].iterrows():
        if row['name'] in histories:
            h = histories[row['name']]
            if 'avg_reward' in h.columns:
                curves.append(h['avg_reward'].values)
    return curves


# ================================================================
# SUMMARY PLOTS (01-15) — English labels
# ================================================================
def make_summary_plots(df, histories, output_dir="grid_results"):
    """Generate all summary plots with English labels."""
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df['reward_short'] = df['reward_mode'].map(REWARD_LABELS)

    neuron_order = sorted(df['neurons'].unique())
    layer_order = sorted(df['layers'].unique())
    valid_df = df.dropna(subset=['final_avg_reward']).copy()

    # ---- Global color scale for all heatmaps ----
    global_vmin = valid_df['final_avg_reward'].min()
    global_vmax = valid_df['final_avg_reward'].max()

    # ---- 01: Algo × Reward heatmap ----
    print("01 — Algorithm × Reward heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (agg, title) in enumerate([('mean', 'Mean final reward'), ('max', 'Best final reward')]):
        pivot = df.pivot_table(values='final_avg_reward', index='algorithm',
                               columns='reward_short', aggfunc=agg
                               ).reindex(index=ALGO_ORDER, columns=REWARD_SHORT_ORDER)
        im = axes[ax_idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=global_vmin, vmax=global_vmax)
        axes[ax_idx].set_xticks(range(len(REWARD_SHORT_ORDER)))
        axes[ax_idx].set_xticklabels(REWARD_SHORT_ORDER, rotation=30, ha='right', fontsize=13)
        axes[ax_idx].set_yticks(range(len(ALGO_ORDER)))
        axes[ax_idx].set_yticklabels([a.upper() for a in ALGO_ORDER], fontsize=13)
        axes[ax_idx].grid(False)
        for i in range(len(ALGO_ORDER)):
            for j in range(len(REWARD_SHORT_ORDER)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    axes[ax_idx].text(j, i, f'{v:.3f}', ha='center', va='center', fontweight='bold', fontsize=14)
        axes[ax_idx].set_title(title, fontweight='bold', fontsize=15)
        plt.colorbar(im, ax=axes[ax_idx], shrink=0.8)
    fig.suptitle('Algorithm × Reward Comparison', fontsize=17, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/01_algo_reward_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 02: Network heatmap per algo × reward (4×3 grid) ----
    print("02 — Network size heatmaps (algo × reward)...")
    fig, axes = plt.subplots(4, 3, figsize=(22, 22))
    for i, algo in enumerate(ALGO_ORDER):
        for j, reward in enumerate(REWARD_ORDER):
            ax = axes[i][j]
            sub = df[(df['algorithm'] == algo) & (df['reward_mode'] == reward)]
            pivot = sub.pivot_table(
                values='final_avg_reward', index='layers', columns='neurons', aggfunc='mean'
            ).reindex(index=layer_order, columns=neuron_order)
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=global_vmin, vmax=global_vmax)
            ax.grid(False)
            ax.set_xticks(range(len(neuron_order))); ax.set_xticklabels(neuron_order, fontsize=11)
            ax.set_yticks(range(len(layer_order))); ax.set_yticklabels(layer_order, fontsize=11)
            if i == 3: ax.set_xlabel('Neurons', fontsize=12)
            if j == 0: ax.set_ylabel('Layers', fontsize=12)
            for ri in range(len(layer_order)):
                for ci in range(len(neuron_order)):
                    v = pivot.values[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f'{v:.3f}', ha='center', va='center', fontweight='bold', fontsize=11)
            ax.set_title(f'{algo.upper()} / {REWARD_LABELS[reward]}', fontweight='bold', fontsize=13)
            plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle('Network Size Effect — Algorithm × Reward (mean final reward)\nUnified color scale across all subplots',
                 fontsize=17, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/02_network_heatmap_algo_reward.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 03: Algo boxplots per reward ----
    print("03 — Algorithm boxplots...")
    median_props = dict(color='black', linewidth=2.5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        sub = df[df['reward_mode'] == reward]
        data = [sub[sub['algorithm'] == a]['final_avg_reward'].dropna() for a in ALGO_ORDER]
        bp = ax.boxplot(data, labels=[a.upper() for a in ALGO_ORDER], patch_artist=True, widths=0.6, medianprops=median_props)
        for patch, algo in zip(bp['boxes'], ALGO_ORDER):
            patch.set_facecolor(ALGO_COLORS[algo]); patch.set_alpha(0.7)
        for i, algo in enumerate(ALGO_ORDER):
            vals = sub[sub['algorithm'] == algo]['final_avg_reward'].dropna()
            ax.scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.3, s=15, color=ALGO_COLORS[algo], zorder=5)
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=13)
        ax.set_ylabel('Final avg_reward' if idx == 0 else ''); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Algorithm Comparison by Reward Mode', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/03_algo_boxplot_per_reward.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 04: Reward boxplots per algo ----
    print("04 — Reward boxplots...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx]
        sub = df[df['algorithm'] == algo]
        data = [sub[sub['reward_mode'] == r]['final_avg_reward'].dropna() for r in REWARD_ORDER]
        bp = ax.boxplot(data, labels=REWARD_SHORT_ORDER, patch_artist=True, widths=0.6, medianprops=median_props)
        for patch, r_short in zip(bp['boxes'], REWARD_SHORT_ORDER):
            patch.set_facecolor(REWARD_COLORS[r_short]); patch.set_alpha(0.7)
        for i, r in enumerate(REWARD_ORDER):
            vals = sub[sub['reward_mode'] == r]['final_avg_reward'].dropna()
            ax.scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.3, s=15, color=REWARD_COLORS[REWARD_LABELS[r]], zorder=5)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=13)
        ax.set_ylabel('Final avg_reward' if idx == 0 else ''); ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Reward Mode Comparison by Algorithm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/04_reward_boxplot_per_algo.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 05: Neuron × Layer effect — grouped bar chart ----
    print("05 — Neuron × Layer effect...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    bar_width = 0.25
    x_positions = np.arange(len(neuron_order))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        sub = df[df['algorithm'] == algo]
        for li, nl in enumerate(layer_order):
            means, stds = [], []
            for ns in neuron_order:
                vals = sub[(sub['layers'] == nl) & (sub['neurons'] == ns)]['final_avg_reward'].dropna()
                means.append(vals.mean()); stds.append(vals.std())
            offset = (li - 1) * bar_width
            bars = ax.bar(x_positions + offset, means, bar_width, yerr=stds,
                         color=LAYER_COLORS[nl], alpha=0.8, capsize=3,
                         label=f'{nl} layer{"s" if nl > 1 else ""}', edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Number of neurons', fontsize=12); ax.set_ylabel('Final avg_reward', fontsize=12)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=14)
        ax.set_xticks(x_positions); ax.set_xticklabels(neuron_order, fontsize=11)
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Effect of Network Width by Depth (mean ± std)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/05_neurons_effect.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 07: Detailed heatmap — one file per reward mode ----
    print("07 — Detailed heatmaps (separate files)...")
    net_labels = [f"{nl}x{ns}" for nl in layer_order for ns in neuron_order]
    for idx, reward in enumerate(REWARD_ORDER):
        fig, ax = plt.subplots(figsize=(22, 6))
        sub = df[df['reward_mode'] == reward]
        matrix = np.full((len(ALGO_ORDER), len(net_labels)), np.nan)
        for i, algo in enumerate(ALGO_ORDER):
            for j, nl_label in enumerate(net_labels):
                vals = sub[(sub['algorithm'] == algo) & (sub['net_label'] == nl_label)]['final_avg_reward']
                if len(vals) > 0: matrix[i, j] = vals.mean()
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=global_vmin, vmax=global_vmax)
        ax.grid(False)
        ax.set_xticks(range(len(net_labels))); ax.set_xticklabels(net_labels, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(ALGO_ORDER))); ax.set_yticklabels([a.upper() for a in ALGO_ORDER], fontsize=14)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_title(f'Network × Algorithm — {REWARD_LABELS[reward]}\n(mean final reward, 3 repeats)', fontweight='bold', fontsize=16)
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(f'{output_dir}/07{chr(97+idx)}_heatmap_{reward}.png', bbox_inches='tight')
        plt.close(fig)

    # ---- 08: Top 20 configs ----
    print("08 — Top 20 configs...")
    grouped = df.groupby(['algorithm', 'reward_mode', 'net_label']).agg(
        mean_reward=('final_avg_reward', 'mean'), std_reward=('final_avg_reward', 'std'),
        count=('final_avg_reward', 'count'),
    ).reset_index()
    grouped['config_label'] = grouped.apply(
        lambda r: f"{r['algorithm'].upper()} | {REWARD_LABELS.get(r['reward_mode'], r['reward_mode'])} | {r['net_label']}", axis=1)
    top20 = grouped.nlargest(20, 'mean_reward')

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = [ALGO_COLORS.get(row['algorithm'], '#999') for _, row in top20.iterrows()]
    ax.barh(range(len(top20)), top20['mean_reward'], xerr=top20['std_reward'],
            color=colors, alpha=0.8, capsize=3, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20['config_label'], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Mean final reward (mean ± std, 3 repeats)')
    ax.set_title('Top 20 Configurations', fontweight='bold', fontsize=14); ax.grid(axis='x', alpha=0.3)
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(row['mean_reward'] + row['std_reward'] + 0.005, i, f"{row['mean_reward']:.3f}", va='center', fontsize=8)
    ax.legend(handles=[Patch(facecolor=ALGO_COLORS[a], label=a.upper()) for a in ALGO_ORDER], loc='lower right')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/08_top20_configs.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 09: Variance analysis (ANOVA-style) ----
    print("09 — Variance analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    factors = {'Reward mode': 'reward_mode', 'Algorithm': 'algorithm', 'Network (L×N)': 'net_label',
               'Layers': 'layers', 'Neurons': 'neurons'}
    grand_mean = valid_df['final_avg_reward'].mean()
    sst = ((valid_df['final_avg_reward'] - grand_mean) ** 2).sum()
    eta_sq = {}
    for label, col in factors.items():
        ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in valid_df.groupby(col)['final_avg_reward'])
        eta_sq[label] = ssb / sst if sst > 0 else 0
    sorted_f = sorted(eta_sq.items(), key=lambda x: x[1], reverse=True)
    labels_f, values_f = zip(*sorted_f)
    bars = axes[0].barh(range(len(labels_f)), values_f, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(labels_f))); axes[0].set_yticklabels(labels_f, fontsize=13); axes[0].invert_yaxis()
    axes[0].set_xlabel('η² (explained variance ratio)', fontsize=12)
    axes[0].set_title('Factor Importance (η² — ANOVA)\nHigher = factor explains more variance in reward', fontweight='bold', fontsize=13)
    for i, v in enumerate(values_f): axes[0].text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].set_xlim(0, max(values_f) * 1.2)

    for algo in ALGO_ORDER:
        means = [valid_df[(valid_df['algorithm'] == algo) & (valid_df['reward_mode'] == r)]['final_avg_reward'].mean() for r in REWARD_ORDER]
        axes[1].plot(REWARD_SHORT_ORDER, means, '-o', color=ALGO_COLORS[algo], label=algo.upper(), linewidth=2.5, markersize=10)
    axes[1].set_xlabel('Reward mode', fontsize=12); axes[1].set_ylabel('Mean final reward', fontsize=12)
    axes[1].set_title('Interaction: Algorithm × Reward\nHow each algorithm responds to different rewards', fontweight='bold', fontsize=13)
    axes[1].legend(fontsize=11); axes[1].grid(alpha=0.3)
    fig.suptitle('Variance Analysis — Which factors matter most?', fontsize=17, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/09_variance_analysis.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 11: Reproducibility — Performance vs Variance across repeats ----
    print("11 — Reproducibility analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        sub = grouped[grouped['algorithm'] == algo]
        for reward in REWARD_ORDER:
            r_sub = sub[sub['reward_mode'] == reward]
            ax.scatter(r_sub['mean_reward'], r_sub['std_reward'],
                      color=REWARD_COLORS[REWARD_LABELS[reward]], alpha=0.7, s=60, label=REWARD_LABELS[reward], edgecolors='white', linewidth=0.5)
        ax.set_xlabel('Mean reward (across 3 repeats)', fontsize=11)
        ax.set_ylabel('Std deviation (across 3 repeats)', fontsize=11)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=14)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.suptitle('Reproducibility: Mean Reward vs Variance Across Repeats\n'
                 'Each dot = one network config. Bottom-right = best (high reward, low variance)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/11_reproducibility.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 12: Within-reward ranking — which algo ranks best per reward mode ----
    print("12 — Within-reward algorithm ranking...")
    # Normalize within each reward mode: 0% = worst config, 100% = best config
    valid_df['norm_reward'] = 0.0
    for reward in REWARD_ORDER:
        mask = valid_df['reward_mode'] == reward
        vals = valid_df.loc[mask, 'final_avg_reward']
        rmin, rmax = vals.min(), vals.max()
        if rmax > rmin:
            valid_df.loc[mask, 'norm_reward'] = (vals - rmin) / (rmax - rmin) * 100

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        sub = valid_df[valid_df['reward_mode'] == reward]
        data = [sub[sub['algorithm'] == a]['norm_reward'].dropna() for a in ALGO_ORDER]
        bp = ax.boxplot(data, labels=[a.upper() for a in ALGO_ORDER], patch_artist=True, widths=0.6, medianprops=median_props)
        for patch, algo in zip(bp['boxes'], ALGO_ORDER):
            patch.set_facecolor(ALGO_COLORS[algo]); patch.set_alpha(0.7)
        for i, algo in enumerate(ALGO_ORDER):
            vals = sub[sub['algorithm'] == algo]['norm_reward'].dropna()
            ax.scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.3, s=15, color=ALGO_COLORS[algo], zorder=5)
            # Mean label
            ax.text(i + 1, vals.mean() + 2, f'{vals.mean():.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=14)
        ax.set_ylabel('Rank within reward mode (%)' if idx == 0 else ''); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-5, 105)
    fig.suptitle('Within-Reward Ranking: Which algorithm performs best?\n'
                 '(0% = worst config in this reward mode, 100% = best)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/12_within_reward_ranking.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 13: Network size ranking within each algo (not cross-reward) ----
    print("13 — Network size ranking per algo...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        pivot = valid_df[valid_df['algorithm'] == algo].pivot_table(
            values='norm_reward', index='layers', columns='neurons', aggfunc='mean'
        ).reindex(index=layer_order, columns=neuron_order)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.grid(False)
        ax.set_xticks(range(len(neuron_order))); ax.set_xticklabels(neuron_order, fontsize=12)
        ax.set_yticks(range(len(layer_order))); ax.set_yticklabels(layer_order, fontsize=12)
        ax.set_xlabel('Neurons', fontsize=12); ax.set_ylabel('Layers', fontsize=12)
        for i in range(len(layer_order)):
            for j in range(len(neuron_order)):
                v = pivot.values[i, j]
                if not np.isnan(v): ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontweight='bold', fontsize=12)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.8, label='%')
    fig.suptitle('Network Size Ranking (within-reward normalized %)\nHigher = better relative to other configs in same reward mode',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/13_network_ranking_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 14: Normalized variance ----
    print("14 — Normalized variance...")
    fig, ax = plt.subplots(figsize=(10, 6))
    gm_n = valid_df['norm_reward'].mean()
    sst_n = ((valid_df['norm_reward'] - gm_n) ** 2).sum()
    eta_n = {}
    for label, col in factors.items():
        ssb = sum(len(g) * (g.mean() - gm_n) ** 2 for _, g in valid_df.groupby(col)['norm_reward'])
        eta_n[label] = ssb / sst_n if sst_n > 0 else 0
    sf_n = sorted(eta_n.items(), key=lambda x: x[1], reverse=True)
    ln, vn = zip(*sf_n)
    ax.barh(range(len(ln)), vn, color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(len(ln))); ax.set_yticklabels(ln, fontsize=13); ax.invert_yaxis()
    ax.set_xlabel('η² (explained variance ratio)', fontsize=12)
    ax.set_title('Factor Importance — Normalized Reward (within-reward)\nHigher η² = factor explains more of the performance difference',
                 fontweight='bold', fontsize=14)
    for i, v in enumerate(vn): ax.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3); ax.set_xlim(0, max(vn) * 1.2)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/14_normalized_variance.png', bbox_inches='tight')
    plt.close(fig)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    algo_means = valid_df.groupby('algorithm')['final_avg_reward'].mean().sort_values(ascending=False)
    print("\nAlgorithm ranking (mean reward):")
    for algo, val in algo_means.items(): print(f"  {algo.upper():8s}  {val:.4f}")
    reward_means = valid_df.groupby('reward_mode')['final_avg_reward'].mean().sort_values(ascending=False)
    print("\nReward mode ranking:")
    for r, val in reward_means.items(): print(f"  {REWARD_LABELS.get(r, r):20s}  {val:.4f}")
    net_means = valid_df.groupby('net_label')['final_avg_reward'].mean().sort_values(ascending=False)
    print("\nNetwork size ranking (top 5):")
    for nl, val in net_means.head(5).items(): print(f"  {nl:8s}  {val:.4f}")
    print(f"\nTop 5 configurations (mean of 3 repeats):")
    for i, (_, row) in enumerate(top20.head(5).iterrows()):
        print(f"  {i+1}. {row['config_label']:40s}  {row['mean_reward']:.4f} +/- {row['std_reward']:.4f}")
    print("=" * 60)

    return valid_df, grouped


# ================================================================
# LaTeX TABLE EXPORT
# ================================================================
def export_latex_tables(df, grouped, output_dir="grid_results"):
    """Export key results as LaTeX tables."""
    os.makedirs(output_dir, exist_ok=True)
    valid_df = df.dropna(subset=['final_avg_reward']).copy()
    valid_df['reward_short'] = valid_df['reward_mode'].map(REWARD_LABELS)
    tables = {}

    # ---- T1: Algorithm × Reward (mean ± std) ----
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean final reward by algorithm and reward mode ($\bar{r} \pm \sigma$, $n=45$ per cell).}")
    lines.append(r"\label{tab:algo_reward}")
    lines.append(r"\begin{tabular}{l" + "c" * len(REWARD_ORDER) + "}")
    lines.append(r"\toprule")
    header = "Algorithm & " + " & ".join([REWARD_LABELS[r] for r in REWARD_ORDER]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for algo in ALGO_ORDER:
        cells = [algo.upper()]
        for reward in REWARD_ORDER:
            vals = valid_df[(valid_df['algorithm'] == algo) & (valid_df['reward_mode'] == reward)]['final_avg_reward']
            m, s = vals.mean(), vals.std()
            cells.append(f"${m:.3f} \\pm {s:.3f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # Overall per reward
    cells = [r"\textit{Overall}"]
    for reward in REWARD_ORDER:
        vals = valid_df[valid_df['reward_mode'] == reward]['final_avg_reward']
        cells.append(f"${vals.mean():.3f} \\pm {vals.std():.3f}$")
    lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tables['T1_algo_reward.tex'] = "\n".join(lines)

    # ---- T2: Top 20 configurations ----
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top 20 configurations ranked by mean final reward (3 repeats).}")
    lines.append(r"\label{tab:top20}")
    lines.append(r"\begin{tabular}{clllcc}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Algorithm & Reward Mode & Network & $\bar{r}$ & $\sigma$ \\")
    lines.append(r"\midrule")
    top20 = grouped.nlargest(20, 'mean_reward')
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        algo = row['algorithm'].upper()
        reward = REWARD_LABELS.get(row['reward_mode'], row['reward_mode'])
        net = row['net_label']
        m = row['mean_reward']
        s = row['std_reward']
        lines.append(f"{rank} & {algo} & {reward} & {net} & ${m:.4f}$ & ${s:.4f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tables['T2_top20.tex'] = "\n".join(lines)

    # ---- T3: Factor importance (η²) ----
    factors = {'Reward mode': 'reward_mode', 'Algorithm': 'algorithm', 'Network (L×N)': 'net_label',
               'Layers': 'layers', 'Neurons': 'neurons'}
    grand_mean = valid_df['final_avg_reward'].mean()
    sst = ((valid_df['final_avg_reward'] - grand_mean) ** 2).sum()
    eta_sq = {}
    for label, col in factors.items():
        ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in valid_df.groupby(col)['final_avg_reward'])
        eta_sq[label] = ssb / sst if sst > 0 else 0

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{ANOVA-style factor importance ($\eta^2$) for final reward variance.}")
    lines.append(r"\label{tab:eta_squared}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Factor & $\eta^2$ & Explained \% \\")
    lines.append(r"\midrule")
    for label, val in sorted(eta_sq.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"{label} & ${val:.4f}$ & ${val*100:.1f}\\%$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tables['T3_factor_importance.tex'] = "\n".join(lines)

    # ---- T4: Network size effect (layers × neurons, pooled across algo+reward) ----
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean final reward by network architecture (all algorithms and reward modes pooled).}")
    lines.append(r"\label{tab:network_size}")
    neuron_order = sorted(valid_df['neurons'].unique())
    layer_order = sorted(valid_df['layers'].unique())
    lines.append(r"\begin{tabular}{l" + "c" * len(neuron_order) + "}")
    lines.append(r"\toprule")
    lines.append("Layers & " + " & ".join([f"{n} neurons" for n in neuron_order]) + r" \\")
    lines.append(r"\midrule")
    for nl in layer_order:
        cells = [f"{nl} layer{'s' if nl > 1 else ' '}"]
        for ns in neuron_order:
            vals = valid_df[(valid_df['layers'] == nl) & (valid_df['neurons'] == ns)]['final_avg_reward']
            cells.append(f"${vals.mean():.3f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tables['T4_network_size.tex'] = "\n".join(lines)

    # ---- T5: Best config per algorithm ----
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Best configuration per algorithm (highest mean final reward, 3 repeats).}")
    lines.append(r"\label{tab:best_per_algo}")
    lines.append(r"\begin{tabular}{llllcc}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Reward Mode & Network & Type & $\bar{r}$ & $\sigma$ \\")
    lines.append(r"\midrule")
    for algo in ALGO_ORDER:
        sub = grouped[grouped['algorithm'] == algo].nlargest(1, 'mean_reward').iloc[0]
        reward = REWARD_LABELS.get(sub['reward_mode'], sub['reward_mode'])
        algo_type = 'On-policy' if algo in ('ppo', 'a2c') else 'Off-policy'
        lines.append(f"{algo.upper()} & {reward} & {sub['net_label']} & {algo_type} & ${sub['mean_reward']:.4f}$ & ${sub['std_reward']:.4f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tables['T5_best_per_algo.tex'] = "\n".join(lines)

    # ---- T6: Algorithm × Network (detailed, per reward) ----
    for reward in REWARD_ORDER:
        lines = []
        r_label = REWARD_LABELS[reward]
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{Mean final reward — {r_label} reward mode.}}")
        lines.append(f"\\label{{tab:detail_{reward}}}")
        net_labels = [f"{nl}×{ns}" for nl in layer_order for ns in neuron_order]
        lines.append(r"\resizebox{\textwidth}{!}{")
        lines.append(r"\begin{tabular}{l" + "c" * len(net_labels) + "}")
        lines.append(r"\toprule")
        lines.append("Algorithm & " + " & ".join(net_labels) + r" \\")
        lines.append(r"\midrule")
        sub = valid_df[valid_df['reward_mode'] == reward]
        for algo in ALGO_ORDER:
            cells = [algo.upper()]
            for nl in layer_order:
                for ns in neuron_order:
                    vals = sub[(sub['algorithm'] == algo) & (sub['layers'] == nl) & (sub['neurons'] == ns)]['final_avg_reward']
                    if len(vals) > 0:
                        cells.append(f"${vals.mean():.3f}$")
                    else:
                        cells.append("--")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")
        tables[f'T6_detail_{reward}.tex'] = "\n".join(lines)

    # Save all tables
    for filename, content in tables.items():
        path = f"{output_dir}/{filename}"
        with open(path, 'w') as f:
            f.write(content)
    print(f"\nLaTeX tables exported: {len(tables)} files to {output_dir}/")
    for f in sorted(tables.keys()):
        print(f"  {f}")


# ================================================================
# SMOOTH LEARNING CURVES (all groupings)
# ================================================================
def make_curve_plots(df, histories, output_dir="grid_results/curves"):
    """Generate smooth learning curves with confidence bands for every grouping."""
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df['reward_short'] = df['reward_mode'].map(REWARD_LABELS)
    neuron_order = sorted(df['neurons'].unique())
    layer_order = sorted(df['layers'].unique())

    sigma = 8  # Gaussian smoothing sigma

    # ---- C01: By Algorithm (all data pooled) ----
    print("C01 — Curves by algorithm...")
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in ALGO_ORDER:
        curves = get_curves_for_filter(df, histories, {'algorithm': algo})
        plot_smooth_band(ax, curves, ALGO_COLORS[algo], algo.upper(), sigma=sigma)
    ax.set_xlabel('Training step (sample index)'); ax.set_ylabel('avg_reward')
    ax.set_title('Learning Curves by Algorithm (all rewards & networks pooled)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C01_by_algorithm.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C02: By Reward Mode (all data pooled) ----
    print("C02 — Curves by reward mode...")
    fig, ax = plt.subplots(figsize=(12, 6))
    for reward in REWARD_ORDER:
        curves = get_curves_for_filter(df, histories, {'reward_mode': reward})
        plot_smooth_band(ax, curves, REWARD_COLORS[REWARD_LABELS[reward]], REWARD_LABELS[reward], sigma=sigma)
    ax.set_xlabel('Training step (sample index)'); ax.set_ylabel('avg_reward')
    ax.set_title('Learning Curves by Reward Mode (all algorithms & networks pooled)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C02_by_reward.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C03: By Algorithm × Reward (4 subplots, one per algo) ----
    print("C03 — Curves: algo subplots, reward lines...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        for reward in REWARD_ORDER:
            curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'reward_mode': reward})
            plot_smooth_band(ax, curves, REWARD_COLORS[REWARD_LABELS[reward]], REWARD_LABELS[reward], sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Algorithm × Reward Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C03_algo_x_reward.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C04: By Reward × Algorithm (3 subplots, one per reward) ----
    print("C04 — Curves: reward subplots, algo lines...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        for algo in ALGO_ORDER:
            curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'reward_mode': reward})
            plot_smooth_band(ax, curves, ALGO_COLORS[algo], algo.upper(), sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Reward Mode × Algorithm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C04_reward_x_algo.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C05: By Number of Layers (pooled) ----
    print("C05 — Curves by layers...")
    fig, ax = plt.subplots(figsize=(12, 6))
    for nl in layer_order:
        curves = get_curves_for_filter(df, histories, {'layers': nl})
        plot_smooth_band(ax, curves, LAYER_COLORS[nl], f'{nl} layer{"s" if nl > 1 else ""}', sigma=sigma)
    ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
    ax.set_title('Learning Curves by Number of Layers', fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C05_by_layers.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C06: By Number of Neurons (pooled) ----
    print("C06 — Curves by neurons...")
    fig, ax = plt.subplots(figsize=(12, 6))
    for ns in neuron_order:
        curves = get_curves_for_filter(df, histories, {'neurons': ns})
        plot_smooth_band(ax, curves, NEURON_COLORS[ns], f'{ns} neurons', sigma=sigma)
    ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
    ax.set_title('Learning Curves by Number of Neurons', fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C06_by_neurons.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C07: By Layers, per Algorithm (4 subplots) ----
    print("C07 — Curves: algo subplots, layer lines...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        for nl in layer_order:
            curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'layers': nl})
            plot_smooth_band(ax, curves, LAYER_COLORS[nl], f'{nl} layer{"s" if nl > 1 else ""}', sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Algorithm × Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C07_algo_x_layers.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C08: By Neurons, per Algorithm (4 subplots) ----
    print("C08 — Curves: algo subplots, neuron lines...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        for ns in neuron_order:
            curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'neurons': ns})
            plot_smooth_band(ax, curves, NEURON_COLORS[ns], f'{ns} neurons', sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Algorithm × Neurons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C08_algo_x_neurons.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C09: By Network Size (LxN), per Algorithm ----
    print("C09 — Curves: algo subplots, network size lines (top 5)...")
    # Only top 5 net sizes per algo to avoid clutter
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    net_colors_all = plt.cm.tab10(np.linspace(0, 1, 15))
    net_labels_all = [f"{nl}x{ns}" for nl in layer_order for ns in neuron_order]
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        # Find top 5 net sizes for this algo by mean final reward
        sub = df[df['algorithm'] == algo].groupby('net_label')['final_avg_reward'].mean().nlargest(5)
        for rank, (nl_label, _) in enumerate(sub.items()):
            nl, ns = map(int, nl_label.split('x'))
            curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'layers': nl, 'neurons': ns})
            plot_smooth_band(ax, curves, net_colors_all[rank], nl_label, sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{algo.upper()} — Top 5 networks', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Best Network Sizes per Algorithm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C09_algo_x_topnets.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C10: By Neurons, per Reward ----
    print("C10 — Curves: reward subplots, neuron lines...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        for ns in neuron_order:
            curves = get_curves_for_filter(df, histories, {'reward_mode': reward, 'neurons': ns})
            plot_smooth_band(ax, curves, NEURON_COLORS[ns], f'{ns} neurons', sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Reward Mode × Neurons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C10_reward_x_neurons.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C11: By Layers, per Reward ----
    print("C11 — Curves: reward subplots, layer lines...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        for nl in layer_order:
            curves = get_curves_for_filter(df, histories, {'reward_mode': reward, 'layers': nl})
            plot_smooth_band(ax, curves, LAYER_COLORS[nl], f'{nl} layer{"s" if nl > 1 else ""}', sigma=sigma)
        ax.set_xlabel('Training step'); ax.set_ylabel('avg_reward')
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=12); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Learning Curves: Reward Mode × Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C11_reward_x_layers.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C12: Full grid — Algo × Reward × Layers (12 subplots) ----
    print("C12 — Full grid: algo × reward × layers...")
    fig, axes = plt.subplots(4, 3, figsize=(24, 24))
    for i, algo in enumerate(ALGO_ORDER):
        for j, reward in enumerate(REWARD_ORDER):
            ax = axes[i][j]
            for nl in layer_order:
                curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'reward_mode': reward, 'layers': nl})
                plot_smooth_band(ax, curves, LAYER_COLORS[nl], f'{nl}L', sigma=sigma)
            ax.set_xlabel('Step'); ax.set_ylabel('avg_reward')
            ax.set_title(f'{algo.upper()} / {REWARD_LABELS[reward]}', fontweight='bold', fontsize=10)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Full Grid: Algorithm × Reward × Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C12_full_grid_layers.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C13: Full grid — Algo × Reward × Neurons (12 subplots) ----
    print("C13 — Full grid: algo × reward × neurons...")
    fig, axes = plt.subplots(4, 3, figsize=(24, 24))
    for i, algo in enumerate(ALGO_ORDER):
        for j, reward in enumerate(REWARD_ORDER):
            ax = axes[i][j]
            for ns in neuron_order:
                curves = get_curves_for_filter(df, histories, {'algorithm': algo, 'reward_mode': reward, 'neurons': ns})
                plot_smooth_band(ax, curves, NEURON_COLORS[ns], f'{ns}N', sigma=sigma)
            ax.set_xlabel('Step'); ax.set_ylabel('avg_reward')
            ax.set_title(f'{algo.upper()} / {REWARD_LABELS[reward]}', fontweight='bold', fontsize=10)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Full Grid: Algorithm × Reward × Neurons', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C13_full_grid_neurons.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C14: Top 10 individual runs ----
    print("C14 — Top 10 individual runs...")
    fig, ax = plt.subplots(figsize=(14, 7))
    top10 = df.nlargest(10, 'final_avg_reward')
    cmap = plt.cm.tab10
    for rank, (_, row) in enumerate(top10.iterrows()):
        if row['name'] in histories:
            h = histories[row['name']]
            if 'avg_reward' in h.columns:
                y = smooth_curve(h['avg_reward'].values, sigma=sigma)
                label = f"{row['algorithm'].upper()} | {REWARD_LABELS[row['reward_mode']]} | {row['net_label']}"
                ax.plot(y, color=cmap(rank / 10), linewidth=2, alpha=0.8, label=label)
    ax.set_xlabel('Training step (sample index)'); ax.set_ylabel('avg_reward')
    ax.set_title('Top 10 Individual Runs — Learning Curves', fontweight='bold', fontsize=14)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C14_top10_individual.png', bbox_inches='tight')
    plt.close(fig)

    # ---- C15: Worst 10 individual runs ----
    print("C15 — Worst 10 individual runs...")
    fig, ax = plt.subplots(figsize=(14, 7))
    worst10 = df.nsmallest(10, 'final_avg_reward')
    for rank, (_, row) in enumerate(worst10.iterrows()):
        if row['name'] in histories:
            h = histories[row['name']]
            if 'avg_reward' in h.columns:
                y = smooth_curve(h['avg_reward'].values, sigma=sigma)
                label = f"{row['algorithm'].upper()} | {REWARD_LABELS[row['reward_mode']]} | {row['net_label']}"
                ax.plot(y, color=cmap(rank / 10), linewidth=2, alpha=0.8, label=label)
    ax.set_xlabel('Training step (sample index)'); ax.set_ylabel('avg_reward')
    ax.set_title('Worst 10 Individual Runs — Learning Curves', fontweight='bold', fontsize=14)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/C15_worst10_individual.png', bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll curve plots saved to: {output_dir}/")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    df, histories = fetch_wandb_data()
    valid_df, grouped = make_summary_plots(df, histories)
    export_latex_tables(df, grouped)
    make_curve_plots(df, histories)
