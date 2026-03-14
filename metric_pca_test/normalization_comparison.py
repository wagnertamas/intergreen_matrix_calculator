#!/usr/bin/env python3
"""
Normalizációs módszerek összehasonlítása - Python megfelelője a MATLAB anal_normalization.m-nek.

10 normalizációs módszer × 6 metrika (reward jelöltek) = 60 hisztogram mátrix
+ Traffic dependency scatterek
+ Cross-korreláció
+ Minden módszerre: statisztikai összefoglaló (variancia, eloszlás tulajdonságok)

Adatforrás: metric_pca_test/ CSV-k (6 flow level × 3 epizód × 21 junction)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import erfinv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adatforrás: ha van v2 mappa a szülő könyvtárban, azt használjuk
_parent = os.path.dirname(SCRIPT_DIR)
_v2_dir = os.path.join(_parent, "metric_pca_test_v2")
if os.path.isdir(_v2_dir):
    DATA_SOURCE_DIR = _v2_dir
    OUTPUT_DIR = _v2_dir
else:
    DATA_SOURCE_DIR = SCRIPT_DIR
    OUTPUT_DIR = SCRIPT_DIR

# ==============================================================================
# 1. NORMALIZÁCIÓS FÜGGVÉNYEK (MATLAB-ból portolva)
# ==============================================================================

def norm_raw(x):
    """1. Raw - Nincs transzformáció"""
    return x.copy()

def norm_tanh_zscore(x):
    """2. Tanh (Z-score): tanh((x - mean) / std) → [-1, 1]"""
    mu, sigma = np.mean(x), np.std(x) + 1e-9
    return np.tanh((x - mu) / sigma)

def norm_algebraic(x):
    """3. Algebraic: x / (x + median) → [0, 1]"""
    med = np.median(x) + 1e-9
    return x / (x + med)

def norm_log(x):
    """4. Log-Normal: ln(x + eps)"""
    return np.log(x + 1e-5)

def norm_log_sigmoid(x):
    """5. Log-Sigmoid: sigmoid(z) ahol z = (log(x) - mu_log) / std_log → [0, 1]"""
    log_x = np.log(x + 1e-5)
    mu = np.mean(log_x)
    sigma = np.std(log_x) + 1e-9
    z = (log_x - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))

def norm_robust_iqr(x):
    """6. Robust (IQR): tanh((x - median) / IQR) → [-1, 1]"""
    med = np.median(x)
    iqr = sp_stats.iqr(x) + 1e-5
    return np.tanh((x - med) / iqr)

def norm_rank_gauss(x):
    """7. Rank Gauss: erfinv transzformáció → ~[-3, 3]"""
    ranks = sp_stats.rankdata(x)
    uniform = (ranks - 0.5) / len(x)
    # Clip hogy ne legyen +-inf
    uniform = np.clip(uniform, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * erfinv(2 * uniform - 1)

def norm_boxcox(x):
    """8. Box-Cox (lambda=0.5): (x^0.5 - 1)/0.5, majd Z-score"""
    bc = (np.sqrt(x + 1e-5) - 1) / 0.5
    mu, sigma = np.mean(bc), np.std(bc) + 1e-9
    return (bc - mu) / sigma

def norm_clipped95(x):
    """9. Clipped 95%: MinMax normalizálás 95. percentilisnél vágva → [0, 1]"""
    lim = np.percentile(x, 95)
    clipped = np.minimum(x, lim)
    mn, mx = np.min(clipped), np.max(clipped)
    if mx > mn:
        return (clipped - mn) / (mx - mn)
    return clipped * 0

def norm_harmonic(x):
    """10. Harmonic (Reciprocal): K / x  ahol K = 5. percentilis → [0, ~1.5]"""
    K = np.percentile(x, 5) + 1e-5
    result = K / (x + 1e-5)
    return np.clip(result, 0, 1.5)

# Összes módszer (online = real-time RL-ben használható előre kiszámolt paraméterekkel)
METHODS = [
    ('Raw Data\nNo Transform',     norm_raw,         '#333333',  True),
    ('Tanh (Z-score)\n[-1, 1]',    norm_tanh_zscore, None,       True),
    ('Algebraic\n[0, 1]',          norm_algebraic,   None,       True),
    ('Log-Normal\nln(x)',          norm_log,         None,       True),
    ('Log-Sigmoid\nReward [0,1]',  norm_log_sigmoid, None,       True),
    ('Robust (IQR)\n[-1, 1]',     norm_robust_iqr,  None,       True),
    ('Rank Gauss\n~[-3, 3]',      norm_rank_gauss,  None,       False),  # ❌ Offline only!
    ('Box-Cox\nZ-score',          norm_boxcox,      None,       True),
    ('Clipped 95%\nMinMax [0,1]', norm_clipped95,   'black',    True),
    ('Harmonic\nK/x [0,1]',      norm_harmonic,    '#7E2F8E',  True),
]

# Csak online-kompatibilis módszerek (RL reward-hoz)
METHODS_ONLINE = [m for m in METHODS if m[3]]

# ==============================================================================
# 2. ADATBETÖLTÉS
# ==============================================================================

def load_all_data():
    """Betölti az összes CSV-t a metric_pca_test mappából."""
    csv_files = [f for f in os.listdir(DATA_SOURCE_DIR)
                 if f.endswith('.csv') and f != 'episode_summary.csv'
                 and '_flow' in f]

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATA_SOURCE_DIR, csv_file))
        flow_level = int(csv_file.split('_flow')[1].split('_')[0])
        df['flow_level'] = flow_level
        df['junction'] = csv_file.split('_flow')[0]
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    # Csak pozitív forgalommal rendelkező sorok
    full_df = full_df[full_df['VehCount'] > 0].copy()
    print(f"Összesen {len(full_df)} érvényes adatpont betöltve ({len(csv_files)} CSV)")
    return full_df


# ==============================================================================
# 3. ÁBR. 1: MEGA DISTRIBUTION MATRIX (6 metrika × 10 módszer)
# ==============================================================================

def plot_distribution_matrix(df):
    """MATLAB-szerű 6×10 mátrix: sorok = metrikák, oszlopok = normalizációs módszerek."""

    # Reward-jelölt metrikák (6 db)
    metric_info = [
        ('TotalTravelTime',       'TT Limited',      '#0072BD'),
        ('AvgTravelTime',         'AvgTT Limited',   '#D95319'),
        ('TotalTravelTime_Raw',   'TT Raw (no filter)', '#0072BD'),
        ('AvgTravelTime_Raw',     'AvgTT Raw',       '#D95319'),
        ('TotalWaitingTime',      'TOTAL WaitingTime','#77AC30'),
        ('AvgWaitingTime',        'AVG WaitingTime',  '#A2142F'),
        ('TotalCO2',              'TOTAL CO2',        '#EDB120'),
        ('AvgCO2',                'AVG CO2',          '#7E2F8E'),
    ]

    n_rows = len(metric_info)
    n_cols = len(METHODS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 24))
    fig.suptitle('Normalization Strategies Comparison\n(8 Reward-Candidate Metrics × 10 Methods, incl. Raw vs Limited TravelTime)',
                 fontsize=16, fontweight='bold', y=0.98)

    for row_i, (col_name, row_label, row_color) in enumerate(metric_info):
        data = df[col_name].values
        # Szűrés: csak pozitív értékek (log-hoz kell)
        data_pos = data[data > 0]
        if len(data_pos) < 100:
            print(f"  [WARN] {col_name}: kevés pozitív adat ({len(data_pos)})")
            data_pos = np.abs(data) + 1e-5

        for col_i, (method_title, method_fn, method_color, _online) in enumerate(METHODS):
            ax = axes[row_i, col_i]

            try:
                transformed = method_fn(data_pos)
                # NaN/Inf szűrés
                valid = transformed[np.isfinite(transformed)]
                if len(valid) < 10:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                else:
                    c = method_color if method_color else row_color
                    ax.hist(valid, bins=50, density=True, color=c, edgecolor='none', alpha=0.85)
            except Exception as e:
                ax.text(0.5, 0.5, f'Err', ha='center', va='center', transform=ax.transAxes, fontsize=7)

            # Oszlop címek (első sor felett)
            if row_i == 0:
                ax.set_title(method_title, fontsize=8, fontweight='bold')

            # Sor címkék (első oszlop)
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

            # Utolsó sornál x label
            if row_i == n_rows - 1:
                ax.set_xlabel('Value', fontsize=6, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, '01_Normalization_Matrix.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mentve: {path}")


# ==============================================================================
# 4. ÁBR. 2: NORMALIZÁCIÓ STATISZTIKÁK (skewness, kurtosis, range, stb.)
# ==============================================================================

def compute_norm_stats(df):
    """Minden metrika × módszer kombinációra statisztikákat számol."""

    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime', 'TotalCO2', 'AvgCO2']

    rows = []
    for col_name in metric_cols:
        data = df[col_name].values
        data_pos = data[data > 0]
        if len(data_pos) < 100:
            data_pos = np.abs(data) + 1e-5

        for method_title, method_fn, _, _online in METHODS:
            try:
                transformed = method_fn(data_pos)
                valid = transformed[np.isfinite(transformed)]
                if len(valid) < 10:
                    continue

                method_short = method_title.split('\n')[0]
                rows.append({
                    'Metric': col_name,
                    'Method': method_short,
                    'Mean': np.mean(valid),
                    'Std': np.std(valid),
                    'Skewness': sp_stats.skew(valid),
                    'Kurtosis': sp_stats.kurtosis(valid),
                    'Min': np.min(valid),
                    'Max': np.max(valid),
                    'Range': np.max(valid) - np.min(valid),
                    'IQR': sp_stats.iqr(valid),
                    'P5': np.percentile(valid, 5),
                    'P95': np.percentile(valid, 95),
                })
            except:
                continue

    stats_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'normalization_stats.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"  Statisztikák mentve: {csv_path}")
    return stats_df


def plot_skewness_kurtosis(stats_df):
    """Skewness és Kurtosis összehasonlítás per módszer -
    tengely-töréssel (axis break) hogy a nagy kiugró értékek ne nyomják el a kicsiket."""

    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime', 'TotalCO2', 'AvgCO2']

    # 2 sor × 4 oszlop (8 metrika), minden cellában 2 ax (broken axis)
    fig = plt.figure(figsize=(30, 16))
    fig.suptitle('Normalization Quality: |Skewness| + |Excess Kurtosis| per Method\n'
                 '(Lower = More Gaussian → Better for RL Reward)',
                 fontsize=15, fontweight='bold', y=0.98)

    # Outer grid: 2×4 metrikák
    outer_gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.45,
                                 left=0.06, right=0.97, top=0.92, bottom=0.05)

    for idx, metric in enumerate(metric_cols):
        row_i, col_i = idx // 4, idx % 4

        sub = stats_df[stats_df['Metric'] == metric].copy()
        sub['Quality'] = np.abs(sub['Skewness']) + np.abs(sub['Kurtosis'])
        sub = sub.sort_values('Quality')

        qualities = sub['Quality'].values
        methods = sub['Method'].values

        colors_bar = ['#2ecc71' if q < 1.0 else '#f39c12' if q < 3.0 else '#e74c3c'
                       for q in qualities]

        # Eldöntjük kell-e tengely törés
        # Ha a max > 10 és van érték < 5 is → törés
        q_max = np.max(qualities)
        q_good_max = 10.0  # A "normál" tartomány felső határa
        needs_break = q_max > q_good_max and np.min(qualities) < q_good_max

        if needs_break:
            # Két tengely: bal (0..break_at) és jobb (break_from..max)
            break_at = min(8.0, q_good_max)
            break_from = q_max * 0.6  # A nagy értékek kezdete

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer_gs[row_i, col_i],
                width_ratios=[3, 1], wspace=0.02)

            ax_left = fig.add_subplot(inner_gs[0])
            ax_right = fig.add_subplot(inner_gs[1])

            # Bal oldal: 0 .. break_at
            y_pos = np.arange(len(methods))
            ax_left.barh(y_pos, qualities, color=colors_bar, edgecolor='none')
            ax_left.set_xlim(0, break_at)
            ax_left.set_yticks(y_pos)
            ax_left.set_yticklabels(methods, fontsize=9)
            ax_left.set_title(metric, fontweight='bold', fontsize=12)
            ax_left.axvline(1.0, color='green', ls='--', alpha=0.6, lw=1.5)
            ax_left.axvline(3.0, color='orange', ls='--', alpha=0.6, lw=1.5)
            ax_left.grid(True, alpha=0.3, axis='x')
            ax_left.set_xlabel('|Skew| + |Kurt|', fontsize=9)

            # Értékek kiírása a sávokra
            for i, (q, m) in enumerate(zip(qualities, methods)):
                if q <= break_at:
                    ax_left.text(q + break_at * 0.02, i, f'{q:.2f}',
                                va='center', fontsize=8, fontweight='bold')

            # Jobb oldal: break_from .. max+padding
            ax_right.barh(y_pos, qualities, color=colors_bar, edgecolor='none')
            ax_right.set_xlim(break_from, q_max * 1.15)
            ax_right.set_yticks(y_pos)
            ax_right.set_yticklabels([])
            ax_right.grid(True, alpha=0.3, axis='x')

            # Értékek kiírása a nagy sávokra
            for i, q in enumerate(qualities):
                if q > break_at:
                    ax_right.text(q + q_max * 0.02, i, f'{q:.1f}',
                                 va='center', fontsize=8, fontweight='bold', color='white')

            # Törés jelek (ferde vonalak)
            d = 0.015
            kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, lw=1.5)
            ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            kwargs.update(transform=ax_right.transAxes)
            ax_right.plot((-d, +d), (-d, +d), **kwargs)
            ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)

            # Spines
            ax_left.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)

            if idx == 0:
                ax_left.plot([], [], color='green', ls='--', label='Good (<1)')
                ax_left.plot([], [], color='orange', ls='--', label='OK (<3)')
                ax_left.legend(fontsize=8, loc='lower right')

        else:
            # Nincs törés - egyszerű barh
            ax = fig.add_subplot(outer_gs[row_i, col_i])
            y_pos = np.arange(len(methods))
            ax.barh(y_pos, qualities, color=colors_bar, edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(methods, fontsize=9)
            ax.set_xlabel('|Skewness| + |Excess Kurtosis|', fontsize=9)
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.axvline(1.0, color='green', ls='--', alpha=0.6, lw=1.5)
            ax.axvline(3.0, color='orange', ls='--', alpha=0.6, lw=1.5)
            ax.grid(True, alpha=0.3, axis='x')

            # Értékek kiírása
            for i, q in enumerate(qualities):
                ax.text(q + q_max * 0.02, i, f'{q:.2f}',
                        va='center', fontsize=8, fontweight='bold')

            if idx == 0:
                ax.plot([], [], color='green', ls='--', label='Good (<1)')
                ax.plot([], [], color='orange', ls='--', label='OK (<3)')
                ax.legend(fontsize=8, loc='lower right')

    path = os.path.join(OUTPUT_DIR, '02_Normalization_Quality.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mentve: {path}")


# ==============================================================================
# 5. ÁBR. 3: TRAFFIC DEPENDENCY + CROSS-CORRELATIONS
# ==============================================================================

def plot_traffic_dependency(df):
    """VehCount vs metrikák scatter + cross-korreláció."""

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle('Traffic Dependency & Cross-Correlations', fontsize=14, fontweight='bold')

    veh = df['VehCount'].values
    max_pts = 30000
    if len(veh) > max_pts:
        idx = np.random.choice(len(veh), max_pts, replace=False)
    else:
        idx = np.arange(len(veh))

    # Sor 1: VehCount vs metrikák
    metrics_dep = [
        ('TotalTravelTime', 'Total TravelTime'),
        ('TotalWaitingTime', 'Total WaitingTime'),
        ('TotalCO2', 'Total CO2'),
        ('QueueLength', 'Queue Length'),
    ]
    for i, (col, title) in enumerate(metrics_dep):
        ax = axes[0, i]
        ax.scatter(veh[idx], df[col].values[idx], s=5, alpha=0.1, c='black')
        ax.set_xlabel('VehCount')
        ax.set_ylabel(title)
        ax.set_title(f'VehCount vs {title}')
        ax.grid(True, alpha=0.3)

    # Sor 2: Cross-korreláció (log scale)
    cross_pairs = [
        ('TotalTravelTime', 'TotalWaitingTime'),
        ('AvgTravelTime', 'AvgWaitingTime'),
        ('TotalWaitingTime', 'TotalCO2'),
        ('AvgWaitingTime', 'AvgSpeed'),
    ]
    for i, (col_x, col_y) in enumerate(cross_pairs):
        ax = axes[1, i]
        x_vals = df[col_x].values[idx]
        y_vals = df[col_y].values[idx]
        # Csak pozitívak (log scale-hez)
        mask = (x_vals > 0) & (y_vals > 0)
        ax.scatter(x_vals[mask], y_vals[mask], s=5, alpha=0.1, c='navy')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f'{col_x} vs {col_y}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, '03_Traffic_Dependency.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mentve: {path}")


# ==============================================================================
# 6. ÁBR. 4: REWARD FUNCTION ÖSSZEHASONLÍTÁS
#    A legjobb módszerekkel reward értékek szimulálása
# ==============================================================================

def plot_reward_comparison(df):
    """A két legjobb normalizálás (Log-Sigmoid vs Harmonic) összehasonlítása
    reward-ként, a kulcs metrikákra, flow level szerinti bontásban."""

    reward_metrics = ['AvgWaitingTime', 'AvgTravelTime', 'TotalWaitingTime', 'TotalCO2']
    flow_levels = sorted(df['flow_level'].unique())

    fig, axes = plt.subplots(len(reward_metrics), 3, figsize=(22, 16))
    fig.suptitle('Reward Function Comparison per Traffic Level\n'
                 '(Log-Sigmoid vs Robust-IQR vs Harmonic)',
                 fontsize=14, fontweight='bold')

    reward_methods = [
        ('Log-Sigmoid', norm_log_sigmoid, '#2ecc71'),
        ('Robust IQR', norm_robust_iqr, '#3498db'),
        ('Harmonic', norm_harmonic, '#9b59b6'),
    ]

    for row_i, metric in enumerate(reward_metrics):
        for col_i, (method_name, method_fn, color) in enumerate(reward_methods):
            ax = axes[row_i, col_i]

            box_data = []
            for fl in flow_levels:
                data = df[df['flow_level'] == fl][metric].values
                data_pos = data[data > 0]
                if len(data_pos) > 0:
                    try:
                        transformed = method_fn(data_pos)
                        # Reward: 1 - transformed (alacsony cost = magas reward)
                        if method_name == 'Robust IQR':
                            reward = -transformed  # tanh: negatív = jó
                        elif method_name == 'Harmonic':
                            reward = transformed  # K/x: már reward
                        else:  # Log-Sigmoid
                            reward = 1.0 - transformed  # 1-sigmoid: reward

                        valid = reward[np.isfinite(reward)]
                        box_data.append(valid)
                    except:
                        box_data.append(np.array([]))
                else:
                    box_data.append(np.array([]))

            bp = ax.boxplot(box_data, labels=[str(fl) for fl in flow_levels],
                           showfliers=False, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            ax.set_xlabel('Flow Max (veh/h/lane)')
            if col_i == 0:
                ax.set_ylabel(f'Reward ({metric})', fontweight='bold')
            ax.set_title(f'{method_name}: {metric}', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, '04_Reward_Comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mentve: {path}")


# ==============================================================================
# 7. ÁBR. 5: ÖSSZESÍTŐ RANKING TÁBLA (szöveges + heatmap)
# ==============================================================================

def plot_ranking_heatmap(stats_df):
    """Heatmap: mely módszer a legjobb melyik metrikára
    (alacsony skewness+kurtosis = jobb)."""

    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime', 'TotalCO2', 'AvgCO2']

    # Quality score kiszámolása
    pivot_data = {}
    for metric in metric_cols:
        sub = stats_df[stats_df['Metric'] == metric].copy()
        sub['Quality'] = np.abs(sub['Skewness']) + np.abs(sub['Kurtosis'])
        for _, row in sub.iterrows():
            method = row['Method']
            if method not in pivot_data:
                pivot_data[method] = {}
            pivot_data[method][metric] = row['Quality']

    pivot_df = pd.DataFrame(pivot_data).T
    # Rendezés átlagos quality szerint
    pivot_df['Mean'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Mean')

    fig, ax = plt.subplots(figsize=(16, 9))

    # Heatmap (Mean oszlop nélkül)
    display_df = pivot_df.drop('Mean', axis=1)

    # 3 színű colormap: Zöld (<1) → Piros (1-10) → Lila (>10)
    from matplotlib.colors import LinearSegmentedColormap
    max_val = max(display_df.max().max(), 15)  # minimum 15 a skálán
    vmax_sqrt = np.sqrt(max_val)

    bp = lambda v: min(np.sqrt(v) / vmax_sqrt, 1.0)
    custom_colors = [
        (0.0,        '#1a9641'),   # sötétzöld - kiváló (0)
        (bp(0.5),    '#2db84d'),   # zöld (0.5)
        (bp(1.0),    '#4dce4d'),   # világoszöld - HATÁR (1.0)
        (bp(1.5),    '#f0c93a'),   # sárga - átmenet (1.5)
        (bp(3.0),    '#fc8d59'),   # narancs (3)
        (bp(5.0),    '#e84530'),   # piros (5)
        (bp(10.0),   '#d73027'),   # sötétpiros - HATÁR (10)
        (bp(20.0),   '#9e0142'),   # magenta (20)
        (bp(50.0),   '#762a83'),   # sötétlila (50)
        (1.0,        '#40004b'),   # mélylila - katasztrofális (max)
    ]
    cmap_custom = LinearSegmentedColormap.from_list('quality', custom_colors)

    display_sqrt = np.sqrt(display_df)

    sns.heatmap(display_sqrt, annot=False, cmap=cmap_custom, ax=ax,
                linewidths=0.8, vmin=0, vmax=vmax_sqrt,
                cbar_kws={'label': '|Skewness| + |Kurtosis|'})

    # Kézzel annotálunk EREDETI értékekkel
    for i in range(len(display_df)):
        for j in range(len(display_df.columns)):
            val = display_df.iloc[i, j]
            if val < 10:
                txt = f'{val:.2f}'
            elif val < 100:
                txt = f'{val:.1f}'
            else:
                txt = f'{val:.0f}'
            # Szövegszín: zöld háttér (val<1) → fehér, sárga/narancs (1-5) → fekete,
            # piros/lila (>5) → fehér
            if val < 1.0:
                text_color = 'white'
            elif val < 5.0:
                text_color = 'black'
            else:
                text_color = 'white'
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)

    # Colorbar tick-ek eredeti értékekre
    cbar = ax.collections[0].colorbar
    tick_vals = [0, 0.5, 1, 2, 5, 10, 25, 50, 100, 200]
    tick_vals = [t for t in tick_vals if t <= max_val]
    cbar.set_ticks([np.sqrt(t) for t in tick_vals])
    cbar.set_ticklabels([str(t) for t in tick_vals])
    cbar.set_label('|Skewness| + |Kurtosis|  (lower = better)')

    ax.set_title('Normalization Quality Ranking (Online-Compatible Methods)\n'
                 'Green = Good (<1)  |  Red = Bad (1-10)  |  Purple = Very Bad (>10)\n'
                 '(Rank Gauss excluded: requires full dataset, not usable in real-time RL)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalization Method')
    ax.set_xlabel('Metric')

    # Jelöljük meg az offline-only módszereket a y-tengelyen
    ytick_labels = ax.get_yticklabels()
    for label in ytick_labels:
        if 'Rank Gauss' in label.get_text():
            label.set_text(label.get_text() + '  [OFFLINE ONLY]')
            label.set_color('#999999')
            label.set_fontstyle('italic')
    ax.set_yticklabels(ytick_labels)

    # Rangsor kiírás
    print("\n" + "=" * 70)
    print("NORMALIZÁCIÓS MÓDSZEREK RANGSOR (átlagos quality)")
    print("=" * 70)
    for i, (method, row) in enumerate(pivot_df.iterrows()):
        print(f"  {i+1:2d}. {method:<20s}  avg_quality = {row['Mean']:.3f}")
    print("=" * 70)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '05_Ranking_Heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Mentve: {path}")

    return pivot_df


# ==============================================================================
# 8. ÁBR. 6: NORMALIZÁCIÓS PARAMÉTEREK a kiválasztott módszerekhez
# ==============================================================================

def compute_final_params(df):
    """A Log-Sigmoid módszer paraméterei minden metrikához (MU, STD)."""

    metric_cols = ['TotalTravelTime', 'AvgTravelTime',
                   'TotalTravelTime_Raw', 'AvgTravelTime_Raw',
                   'TotalWaitingTime', 'AvgWaitingTime', 'TotalCO2', 'AvgCO2',
                   'VehCount', 'AvgSpeed', 'AvgOccupancy', 'QueueLength']

    print("\n" + "=" * 70)
    print("LOG-SIGMOID NORMALIZÁCIÓS PARAMÉTEREK")
    print("  Formula: z = (log(x+eps) - MU) / STD")
    print("           sigmoid = 1 / (1 + exp(-z))")
    print("           reward  = 1 - sigmoid  (cost→reward inverz)")
    print("=" * 70)

    for col in metric_cols:
        vals = df[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            log_vals = np.log(vals + 1e-5)
            mu = np.mean(log_vals)
            std = np.std(log_vals)
            print(f"  {col:<20s}  MU = {mu:10.6f}  STD = {std:8.6f}  "
                  f"(median={np.median(vals):.2f}, p5={np.percentile(vals,5):.2f}, "
                  f"p95={np.percentile(vals,95):.2f})")

    print("\n--- ROBUST IQR PARAMÉTEREK ---")
    print("  Formula: tanh((x - median) / IQR)")
    print("           reward = -tanh(...)  (negatív tanh = jó)")
    for col in metric_cols:
        vals = df[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            med = np.median(vals)
            iqr = sp_stats.iqr(vals)
            print(f"  {col:<20s}  MEDIAN = {med:10.4f}  IQR = {iqr:8.4f}")

    print("\n--- HARMONIC PARAMÉTEREK ---")
    print("  Formula: K / (x + eps)")
    print("           K = percentile(data, 5)")
    for col in metric_cols:
        vals = df[col].values
        vals = vals[vals > 0]
        if len(vals) > 0:
            K = np.percentile(vals, 5)
            print(f"  {col:<20s}  K = {K:10.4f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("  NORMALIZÁCIÓS MÓDSZEREK ÖSSZEHASONLÍTÁSA")
    print("  (MATLAB anal_normalization.m Python portja)")
    print("=" * 70)

    # 1. Adat betöltés
    print("\n[1/7] Adatok betöltése...")
    df = load_all_data()

    # 2. Distribution Matrix (6×10)
    print("\n[2/7] Eloszlás mátrix generálás (6 metrika × 10 módszer)...")
    plot_distribution_matrix(df)

    # 3. Statisztikák
    print("\n[3/7] Normalizációs statisztikák számítása...")
    stats_df = compute_norm_stats(df)

    # 4. Quality (Skewness + Kurtosis)
    print("\n[4/7] Quality barplot generálás...")
    plot_skewness_kurtosis(stats_df)

    # 5. Traffic Dependency
    print("\n[5/7] Traffic dependency + cross-korreláció...")
    plot_traffic_dependency(df)

    # 6. Reward összehasonlítás
    print("\n[6/7] Reward function összehasonlítás flow level szerint...")
    plot_reward_comparison(df)

    # 7. Ranking Heatmap
    print("\n[7/7] Ranking heatmap + paraméterek...")
    ranking = plot_ranking_heatmap(stats_df)

    # Final: Paraméterek
    compute_final_params(df)

    print("\n" + "=" * 70)
    print("  KÉSZ! Generált fájlok:")
    print("  01_Normalization_Matrix.png   - 6×10 hisztogram mátrix")
    print("  02_Normalization_Quality.png  - Skewness+Kurtosis összehasonlítás")
    print("  03_Traffic_Dependency.png     - Forgalom-függőség scatterek")
    print("  04_Reward_Comparison.png      - Reward per flow level")
    print("  05_Ranking_Heatmap.png        - Módszer rangsor heatmap")
    print("  normalization_stats.csv       - Részletes statisztikák")
    print("=" * 70)


if __name__ == '__main__':
    main()
