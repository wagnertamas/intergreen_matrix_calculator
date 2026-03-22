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
    'axes.grid': True,
    'grid.alpha': 0.3,
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

    # ---- 01: Algo × Reward heatmap ----
    print("01 — Algorithm × Reward heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, (agg, title) in enumerate([('mean', 'Mean final reward'), ('max', 'Best final reward')]):
        pivot = df.pivot_table(values='final_avg_reward', index='algorithm',
                               columns='reward_short', aggfunc=agg
                               ).reindex(index=ALGO_ORDER, columns=REWARD_SHORT_ORDER)
        im = axes[ax_idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        axes[ax_idx].set_xticks(range(len(REWARD_SHORT_ORDER)))
        axes[ax_idx].set_xticklabels(REWARD_SHORT_ORDER, rotation=30, ha='right', fontsize=13)
        axes[ax_idx].set_yticks(range(len(ALGO_ORDER)))
        axes[ax_idx].set_yticklabels([a.upper() for a in ALGO_ORDER], fontsize=13)
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

    # ---- 02: Network heatmap per algo ----
    print("02 — Network size heatmaps...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        pivot = df[df['algorithm'] == algo].pivot_table(
            values='final_avg_reward', index='layers', columns='neurons', aggfunc='mean'
        ).reindex(index=layer_order, columns=neuron_order)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(neuron_order))); ax.set_xticklabels(neuron_order)
        ax.set_yticks(range(len(layer_order))); ax.set_yticklabels(layer_order)
        ax.set_xlabel('Number of neurons'); ax.set_ylabel('Number of layers')
        for i in range(len(layer_order)):
            for j in range(len(neuron_order)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Network Size Effect by Algorithm (mean final reward)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/02_network_heatmap_per_algo.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 03: Algo boxplots per reward ----
    print("03 — Algorithm boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for idx, reward in enumerate(REWARD_ORDER):
        ax = axes[idx]
        sub = df[df['reward_mode'] == reward]
        data = [sub[sub['algorithm'] == a]['final_avg_reward'].dropna() for a in ALGO_ORDER]
        bp = ax.boxplot(data, labels=[a.upper() for a in ALGO_ORDER], patch_artist=True, widths=0.6)
        for patch, algo in zip(bp['boxes'], ALGO_ORDER):
            patch.set_facecolor(ALGO_COLORS[algo]); patch.set_alpha(0.7)
        for i, algo in enumerate(ALGO_ORDER):
            vals = sub[sub['algorithm'] == algo]['final_avg_reward'].dropna()
            ax.scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.3, s=15, color=ALGO_COLORS[algo], zorder=5)
        ax.set_title(f'{REWARD_LABELS[reward]}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Final avg_reward' if idx == 0 else ''); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Algorithm Comparison by Reward Mode', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/03_algo_boxplot_per_reward.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 04: Reward boxplots per algo ----
    print("04 — Reward boxplots...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx]
        sub = df[df['algorithm'] == algo]
        data = [sub[sub['reward_mode'] == r]['final_avg_reward'].dropna() for r in REWARD_ORDER]
        bp = ax.boxplot(data, labels=REWARD_SHORT_ORDER, patch_artist=True, widths=0.6)
        for patch, r_short in zip(bp['boxes'], REWARD_SHORT_ORDER):
            patch.set_facecolor(REWARD_COLORS[r_short]); patch.set_alpha(0.7)
        for i, r in enumerate(REWARD_ORDER):
            vals = sub[sub['reward_mode'] == r]['final_avg_reward'].dropna()
            ax.scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.3, s=15, color=REWARD_COLORS[REWARD_LABELS[r]], zorder=5)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12)
        ax.set_ylabel('Final avg_reward' if idx == 0 else ''); ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Reward Mode Comparison by Algorithm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/04_reward_boxplot_per_algo.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 05: Neuron effect line plots ----
    print("05 — Neuron effect...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layer_styles = {1: '-o', 2: '--s', 3: ':^'}
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        sub = df[df['algorithm'] == algo]
        for nl in layer_order:
            means, stds = [], []
            for ns in neuron_order:
                vals = sub[(sub['layers'] == nl) & (sub['neurons'] == ns)]['final_avg_reward'].dropna()
                means.append(vals.mean()); stds.append(vals.std())
            ax.errorbar(neuron_order, means, yerr=stds, fmt=layer_styles[nl], color=LAYER_COLORS[nl],
                       label=f'{nl} layer{"s" if nl > 1 else ""}', capsize=4, markersize=6, linewidth=1.5)
        ax.set_xlabel('Number of neurons'); ax.set_ylabel('Final avg_reward')
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12)
        ax.set_xscale('log', base=2); ax.set_xticks(neuron_order); ax.set_xticklabels(neuron_order)
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
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
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(net_labels))); ax.set_xticklabels(net_labels, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(range(len(ALGO_ORDER))); ax.set_yticklabels([a.upper() for a in ALGO_ORDER], fontsize=14)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_title(f'Network × Algorithm — {REWARD_LABELS[reward]}', fontweight='bold', fontsize=16)
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

    # ---- 09: Variance analysis ----
    print("09 — Variance analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
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
    axes[0].barh(range(len(labels_f)), values_f, color='#3498db', alpha=0.8)
    axes[0].set_yticks(range(len(labels_f))); axes[0].set_yticklabels(labels_f); axes[0].invert_yaxis()
    axes[0].set_xlabel('η² (explained variance ratio)'); axes[0].set_title('Factor Importance', fontweight='bold')
    for i, v in enumerate(values_f): axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    axes[0].grid(axis='x', alpha=0.3)

    for algo in ALGO_ORDER:
        means = [valid_df[(valid_df['algorithm'] == algo) & (valid_df['reward_mode'] == r)]['final_avg_reward'].mean() for r in REWARD_ORDER]
        axes[1].plot(REWARD_SHORT_ORDER, means, '-o', color=ALGO_COLORS[algo], label=algo.upper(), linewidth=2, markersize=8)
    axes[1].set_xlabel('Reward mode'); axes[1].set_ylabel('Mean final reward')
    axes[1].set_title('Interaction: Algorithm × Reward', fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle('Variance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/09_variance_analysis.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 11: Stability ----
    print("11 — Stability analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        sub = grouped[grouped['algorithm'] == algo]
        for reward in REWARD_ORDER:
            r_sub = sub[sub['reward_mode'] == reward]
            ax.scatter(r_sub['mean_reward'], r_sub['std_reward'],
                      color=REWARD_COLORS[REWARD_LABELS[reward]], alpha=0.6, s=40, label=REWARD_LABELS[reward])
        ax.set_xlabel('Mean reward'); ax.set_ylabel('Std (3 repeats)')
        ax.set_title(f'{algo.upper()} — Performance vs Stability', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Stability: Does Higher Reward Mean More Stable?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/11_stability.png', bbox_inches='tight')
    plt.close(fig)

    # ---- 12-15: Normalized comparison ----
    print("12-15 — Normalized comparison...")
    valid_df['norm_reward'] = 0.0
    for reward in REWARD_ORDER:
        mask = valid_df['reward_mode'] == reward
        vals = valid_df.loc[mask, 'final_avg_reward']
        rmin, rmax = vals.min(), vals.max()
        if rmax > rmin:
            valid_df.loc[mask, 'norm_reward'] = (vals - rmin) / (rmax - rmin) * 100

    # 12: Normalized heatmap + boxplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pivot_n = valid_df.pivot_table(values='norm_reward', index='algorithm', columns='reward_short',
                                    aggfunc='mean').reindex(index=ALGO_ORDER, columns=REWARD_SHORT_ORDER)
    im = axes[0].imshow(pivot_n.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0].set_xticks(range(len(REWARD_SHORT_ORDER))); axes[0].set_xticklabels(REWARD_SHORT_ORDER, rotation=45, ha='right')
    axes[0].set_yticks(range(len(ALGO_ORDER))); axes[0].set_yticklabels([a.upper() for a in ALGO_ORDER])
    for i in range(len(ALGO_ORDER)):
        for j in range(len(REWARD_SHORT_ORDER)):
            v = pivot_n.values[i, j]
            if not np.isnan(v): axes[0].text(j, i, f'{v:.1f}%', ha='center', va='center', fontweight='bold', fontsize=11)
    axes[0].set_title('Mean performance (normalized %)', fontweight='bold')
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='%')
    data_n = [valid_df[valid_df['algorithm'] == a]['norm_reward'].dropna() for a in ALGO_ORDER]
    bp = axes[1].boxplot(data_n, labels=[a.upper() for a in ALGO_ORDER], patch_artist=True, widths=0.6)
    for patch, algo in zip(bp['boxes'], ALGO_ORDER): patch.set_facecolor(ALGO_COLORS[algo]); patch.set_alpha(0.7)
    for i, algo in enumerate(ALGO_ORDER):
        vals = valid_df[valid_df['algorithm'] == algo]['norm_reward'].dropna()
        axes[1].scatter(np.random.normal(i + 1, 0.08, len(vals)), vals, alpha=0.2, s=12, color=ALGO_COLORS[algo], zorder=5)
    axes[1].set_ylabel('Normalized performance (%)'); axes[1].set_title('Algorithms (pooled across rewards)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    fig.suptitle('Percentile-Normalized Comparison (cross-reward)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/12_normalized_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # 13: Normalized network heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for idx, algo in enumerate(ALGO_ORDER):
        ax = axes[idx // 2][idx % 2]
        pivot = valid_df[valid_df['algorithm'] == algo].pivot_table(
            values='norm_reward', index='layers', columns='neurons', aggfunc='mean'
        ).reindex(index=layer_order, columns=neuron_order)
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(neuron_order))); ax.set_xticklabels(neuron_order)
        ax.set_yticks(range(len(layer_order))); ax.set_yticklabels(layer_order)
        ax.set_xlabel('Neurons'); ax.set_ylabel('Layers')
        for i in range(len(layer_order)):
            for j in range(len(neuron_order)):
                v = pivot.values[i, j]
                if not np.isnan(v): ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
        ax.set_title(f'{algo.upper()}', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8, label='%')
    fig.suptitle('Network Size Effect — Normalized % (cross-reward)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/13_normalized_network_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # 14: Normalized top 20
    grouped_n = valid_df.groupby(['algorithm', 'reward_mode', 'net_label']).agg(
        mean_norm=('norm_reward', 'mean'), std_norm=('norm_reward', 'std')).reset_index()
    grouped_n['config_label'] = grouped_n.apply(
        lambda r: f"{r['algorithm'].upper()} | {REWARD_LABELS.get(r['reward_mode'], r['reward_mode'])} | {r['net_label']}", axis=1)
    top20n = grouped_n.nlargest(20, 'mean_norm')
    fig, ax = plt.subplots(figsize=(14, 8))
    colors_n = [ALGO_COLORS.get(row['algorithm'], '#999') for _, row in top20n.iterrows()]
    ax.barh(range(len(top20n)), top20n['mean_norm'], xerr=top20n['std_norm'],
            color=colors_n, alpha=0.8, capsize=3, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top20n))); ax.set_yticklabels(top20n['config_label'], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Normalized performance (%, mean ± std)')
    ax.set_title('Top 20 Configurations — Normalized Ranking', fontweight='bold', fontsize=14); ax.grid(axis='x', alpha=0.3)
    for i, (_, row) in enumerate(top20n.iterrows()):
        ax.text(row['mean_norm'] + row['std_norm'] + 1, i, f"{row['mean_norm']:.1f}%", va='center', fontsize=8)
    ax.legend(handles=[Patch(facecolor=ALGO_COLORS[a], label=a.upper()) for a in ALGO_ORDER], loc='lower right')
    plt.tight_layout()
    fig.savefig(f'{output_dir}/14_normalized_top20.png', bbox_inches='tight')
    plt.close(fig)

    # 15: Normalized variance
    fig, ax = plt.subplots(figsize=(8, 5))
    gm_n = valid_df['norm_reward'].mean()
    sst_n = ((valid_df['norm_reward'] - gm_n) ** 2).sum()
    eta_n = {}
    for label, col in factors.items():
        ssb = sum(len(g) * (g.mean() - gm_n) ** 2 for _, g in valid_df.groupby(col)['norm_reward'])
        eta_n[label] = ssb / sst_n if sst_n > 0 else 0
    sf_n = sorted(eta_n.items(), key=lambda x: x[1], reverse=True)
    ln, vn = zip(*sf_n)
    ax.barh(range(len(ln)), vn, color='#e74c3c', alpha=0.8)
    ax.set_yticks(range(len(ln))); ax.set_yticklabels(ln); ax.invert_yaxis()
    ax.set_xlabel('η² (explained variance ratio)'); ax.set_title('Factor Importance — Normalized Reward', fontweight='bold')
    for i, v in enumerate(vn): ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/15_normalized_variance.png', bbox_inches='tight')
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
    make_summary_plots(df, histories)
    make_curve_plots(df, histories)
