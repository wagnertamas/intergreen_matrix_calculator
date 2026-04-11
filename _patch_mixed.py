#!/usr/bin/env python3
"""Patch: add mixed mode to metric_collection_per_junction.py"""
import sys

with open('metric_collection_per_junction.py', 'rb') as f:
    raw = f.read()

changes = []

# 1. Constants
old1 = b"ACTUATED_EPISODES = 8   # ism\xc3\xa9tl\xc3\xa9sek forgalmi szintenk\xc3\xa9nt (actuated kontroll)\r\n"
new1 = (
    b"ACTUATED_EPISODES = 8   # ism\xc3\xa9tl\xc3\xa9sek forgalmi szintenk\xc3\xa9nt (actuated kontroll)\r\n"
    b"MIXED_EPISODES    = 4   # ism\xc3\xa9tl\xc3\xa9sek forgalmi szintenk\xc3\xa9nt (mixed: 80% actuated + 20% random)\r\n"
    b"MIXED_EPSILON     = 0.20  # random akci\xc3\xb3 val\xc3\xb3sz\xc3\xadn\xc5\xb1s\xc3\xa9ge mixed m\xc3\xb3dban (\xcf\xb5-greedy)\r\n"
)
if old1 in raw:
    raw = raw.replace(old1, new1, 1); changes.append("1:OK")
elif b"MIXED_EPISODES" in raw:
    changes.append("1:SKIP")
else:
    changes.append("1:FAIL"); sys.exit("FAIL 1")

# 2. TLController creation
old2 = (
    b"    # Vez\xc3\xa9rl\xc5\x91k l\xc3\xa9trehoz\xc3\xa1sa (csak random m\xc3\xb3dhoz kell)\r\n"
    b"    tl_controllers = {}\r\n"
    b"    if control_mode == \"random\":\r\n"
)
new2 = (
    b"    # Vez\xc3\xa9rl\xc5\x91k l\xc3\xa9trehoz\xc3\xa1sa (random \xc3\xa9s mixed m\xc3\xb3dhoz)\r\n"
    b"    tl_controllers = {}\r\n"
    b"    if control_mode == \"random\" or control_mode.startswith(\"mixed\"):\r\n"
)
if old2 in raw:
    raw = raw.replace(old2, new2, 1); changes.append("2:OK")
elif b'control_mode.startswith("mixed")' in raw:
    changes.append("2:SKIP")
else:
    changes.append("2:FAIL"); sys.exit("FAIL 2")

# 3. Action loop
old3 = (
    b"        if control_mode == \"random\":\r\n"
    b"            for jid in junction_ids:\r\n"
    b"                ctrl = tl_controllers.get(jid)\r\n"
    b"                if ctrl:\r\n"
    b"                    # Csak akkor adunk \xc3\xbaj akci\xc3\xb3t, ha a kontroller k\xc3\xa9szen \xc3\xa1ll\r\n"
    b"                    # (nem transitioning \xc3\xa9s letelt a min green time)\r\n"
    b"                    if ctrl.is_ready():\r\n"
    b"                        ctrl.set_target(random.randint(0, ctrl.num_phases - 1))\r\n"
    b"                    ctrl.update()\r\n"
    b"                    current_phase[jid] = ctrl.current_logic_idx\r\n"
    b"                else:\r\n"
    b"                    current_phase[jid] = 0\r\n"
    b"        else:\r\n"
    b"            # Actuated: a SUMO maga kezeli, csak kiolvassuk az aktu\xc3\xa1lis f\xc3\xa1zist\r\n"
    b"            for jid in junction_ids:\r\n"
    b"                current_phase[jid] = traci.trafficlight.getPhase(jid)\r\n"
)
new3 = (
    b"        if control_mode == \"random\":\r\n"
    b"            for jid in junction_ids:\r\n"
    b"                ctrl = tl_controllers.get(jid)\r\n"
    b"                if ctrl:\r\n"
    b"                    # Csak akkor adunk \xc3\xbaj akci\xc3\xb3t, ha a kontroller k\xc3\xa9szen \xc3\xa1ll\r\n"
    b"                    # (nem transitioning \xc3\xa9s letelt a min green time)\r\n"
    b"                    if ctrl.is_ready():\r\n"
    b"                        ctrl.set_target(random.randint(0, ctrl.num_phases - 1))\r\n"
    b"                    ctrl.update()\r\n"
    b"                    current_phase[jid] = ctrl.current_logic_idx\r\n"
    b"                else:\r\n"
    b"                    current_phase[jid] = 0\r\n"
    b"        elif control_mode.startswith(\"mixed\"):\r\n"
    b"            # Mixed (\xcf\xb5-greedy): MIXED_EPSILON val\xc3\xb3sz\xc3\xadn\xc5\xb1s\xc3\xa9ggel random akci\xc3\xb3,\r\n"
    b"            # egy\xc3\xa9bk\xc3\xa9nt megtartja az aktu\xc3\xa1lis f\xc3\xa1zist (actuated-szer\xc5\xb1 viselked\xc3\xa9s).\r\n"
    b"            try:\r\n"
    b"                _eps_rand = int(control_mode.split('_')[1]) / 100.0\r\n"
    b"            except Exception:\r\n"
    b"                _eps_rand = MIXED_EPSILON\r\n"
    b"            for jid in junction_ids:\r\n"
    b"                ctrl = tl_controllers.get(jid)\r\n"
    b"                if ctrl:\r\n"
    b"                    if ctrl.is_ready() and random.random() < _eps_rand:\r\n"
    b"                        ctrl.set_target(random.randint(0, ctrl.num_phases - 1))\r\n"
    b"                    # else: ne tegy\xc3\xbcnk semmit \xe2\x80\x94 a f\xc3\xa1zis marad\r\n"
    b"                    ctrl.update()\r\n"
    b"                    current_phase[jid] = ctrl.current_logic_idx\r\n"
    b"                else:\r\n"
    b"                    current_phase[jid] = 0\r\n"
    b"        else:\r\n"
    b"            # Actuated: a SUMO maga kezeli, csak kiolvassuk az aktu\xc3\xa1lis f\xc3\xa1zist\r\n"
    b"            for jid in junction_ids:\r\n"
    b"                current_phase[jid] = traci.trafficlight.getPhase(jid)\r\n"
)
if old3 in raw:
    raw = raw.replace(old3, new3, 1); changes.append("3:OK")
elif b'elif control_mode.startswith("mixed"):' in raw:
    changes.append("3:SKIP")
else:
    changes.append("3:FAIL"); sys.exit("FAIL 3")

# 4a. step curve mode detection
old4a = (
    b"        mode = 'actuated' if '_actuated.csv' in csv_file else 'random'\r\n"
    b"        df['junction']     = jid\r\n"
    b"        df['flow_level']   = flow_level\r\n"
    b"        df['episode']      = ep\r\n"
    b"        df['control_mode'] = df['control_mode'] if 'control_mode' in df.columns else mode\r\n"
    b"        df['run_id']       = f\"{jid}_flow{flow_level}_ep{ep}_{mode}\"\r\n"
)
new4a = (
    b"        if '_actuated.csv' in csv_file:\r\n"
    b"            mode = 'actuated'\r\n"
    b"        elif '_mixed' in csv_file:\r\n"
    b"            _m = csv_file.split('_ep')[1].split('_', 1)\r\n"
    b"            mode = 'mixed_' + _m[1].replace('.csv', '') if len(_m) > 1 else 'mixed'\r\n"
    b"        else:\r\n"
    b"            mode = 'random'\r\n"
    b"        df['junction']     = jid\r\n"
    b"        df['flow_level']   = flow_level\r\n"
    b"        df['episode']      = ep\r\n"
    b"        df['control_mode'] = df['control_mode'] if 'control_mode' in df.columns else mode\r\n"
    b"        df['run_id']       = f\"{jid}_flow{flow_level}_ep{ep}_{df['control_mode'].iloc[0]}\"\r\n"
)
if old4a in raw:
    raw = raw.replace(old4a, new4a, 1); changes.append("4a:OK")
elif b"elif '_mixed' in csv_file:\r\n            _m = csv_file.split('_ep')[1]" in raw:
    changes.append("4a:SKIP")
else:
    changes.append("4a:FAIL"); sys.exit("FAIL 4a")

# 4b. combo loader mode detection
old4b = (
    b"        if 'control_mode' not in df.columns:\r\n"
    b"            df['control_mode'] = 'actuated' if '_actuated.csv' in csv_file else 'random'\r\n"
    b"        # Egyedi run azonos\xc3\xadt\xc3\xb3\r\n"
    b"        df['run_id'] = f\"{jid}_flow{flow_level}_ep{ep}_{df['control_mode'].iloc[0]}\"\r\n"
)
new4b = (
    b"        if 'control_mode' not in df.columns:\r\n"
    b"            if '_actuated.csv' in csv_file:\r\n"
    b"                df['control_mode'] = 'actuated'\r\n"
    b"            elif '_mixed' in csv_file:\r\n"
    b"                _cm = csv_file.split('_ep')[1].split('_', 1)\r\n"
    b"                df['control_mode'] = 'mixed_' + _cm[1].replace('.csv', '') if len(_cm) > 1 else 'mixed'\r\n"
    b"            else:\r\n"
    b"                df['control_mode'] = 'random'\r\n"
    b"        # Egyedi run azonos\xc3\xadt\xc3\xb3\r\n"
    b"        df['run_id'] = f\"{jid}_flow{flow_level}_ep{ep}_{df['control_mode'].iloc[0]}\"\r\n"
)
if old4b in raw:
    raw = raw.replace(old4b, new4b, 1); changes.append("4b:OK")
elif b"                df['control_mode'] = 'mixed_'" in raw:
    changes.append("4b:SKIP")
else:
    changes.append("4b:FAIL"); sys.exit("FAIL 4b")

# 5. reward_step_curve: replace single-plot with 3-panel plot by control_mode
old5 = (
    b"    cmap        = plt.cm.get_cmap('RdYlBu_r')\r\n"
    b"    EMA_ALPHA   = 0.05\r\n"
    b"\r\n"
    b"    fig, ax = plt.subplots(figsize=(16, 6))\r\n"
    b"    all_smoothed = []\r\n"
    b"\r\n"
    b"    for run_id in run_ids:\r\n"
    b"        rd = full[full['run_id'] == run_id].sort_values('step')\r\n"
    b"        if len(rd) < 5:\r\n"
    b"            continue\r\n"
    b"        steps  = rd['step'].values\r\n"
    b"        reward = rd['reward_plain_step'].values\r\n"
    b"        color  = cmap(flow_norm[rd['flow_level'].iloc[0]])\r\n"
    b"\r\n"
    b"        ax.plot(steps, reward, color=color, alpha=0.18, linewidth=0.7)\r\n"
    b"\r\n"
    b"        ema = np.zeros_like(reward, dtype=float)\r\n"
    b"        ema[0] = reward[0]\r\n"
    b"        for i in range(1, len(reward)):\r\n"
    b"            ema[i] = EMA_ALPHA * reward[i] + (1 - EMA_ALPHA) * ema[i - 1]\r\n"
    b"        ax.plot(steps, ema, color=color, alpha=0.55, linewidth=1.2)\r\n"
    b"        all_smoothed.append((steps, ema))\r\n"
    b"\r\n"
    b"    if all_smoothed:\r\n"
    b"        max_step     = max(s[-1] for s, _ in all_smoothed)\r\n"
    b"        common_steps = np.arange(0, int(max_step) + 1)\r\n"
    b"        interp_rows  = [np.interp(common_steps, s, e) for s, e in all_smoothed if len(s) > 1]\r\n"
    b"        if interp_rows:\r\n"
    b"            gm = np.mean(interp_rows, axis=0)\r\n"
    b"            ema_g = np.zeros_like(gm)\r\n"
    b"            ema_g[0] = gm[0]\r\n"
    b"            for i in range(1, len(gm)):\r\n"
    b"                ema_g[i] = EMA_ALPHA * gm[i] + (1 - EMA_ALPHA) * ema_g[i - 1]\r\n"
    b"            ax.plot(common_steps, ema_g, color='black', linewidth=2.5,\r\n"
    b"                    label=f'\xc3\x81tlag (n={len(interp_rows)} run)', zorder=10)\r\n"
    b"\r\n"
    b"    sm = plt.cm.ScalarMappable(cmap=cmap,\r\n"
    b"                               norm=plt.Normalize(vmin=min(flow_levels), vmax=max(flow_levels)))\r\n"
    b"    sm.set_array([])\r\n"
    b"    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)\r\n"
    b"    cbar.set_label('Flow level (veh/h)', fontsize=10)\r\n"
    b"\r\n"
    b"    title_jid = junction_filter if junction_filter else '\xc3\xb6sszes junction'\r\n"
    b"    ax.set_xlabel('Step', fontsize=12)\r\n"
    b"    ax.set_ylabel('Normalized reward  [0 \xe2\x80\x93 1]', fontsize=12)\r\n"
    b"    ax.set_title(\r\n"
    b"        f'Step-szint\xc5\xb1 reward \xe2\x80\x94 Speed + Throughput, log-tanh  |  {title_jid}  |  {param_label} params\\n'\r\n"
    b"        f'\xce\xbc_s={mu_s:.3f}  \xcf\x83_s={std_s:.3f}  |  \xce\xbc_tp={mu_t:.3f}  \xcf\x83_tp={std_t:.3f}  |  '\r\n"
    b"        f'{n_runs} run  |  EMA \xce\xb1={EMA_ALPHA}',\r\n"
    b"        fontsize=11)\r\n"
    b"    ax.set_ylim(-0.02, 1.02)\r\n"
    b"    ax.legend(fontsize=10, loc='upper left')\r\n"
    b"    ax.grid(True, alpha=0.25)\r\n"
    b"\r\n"
    b"    plt.tight_layout()\r\n"
    b"    out_path = os.path.join(output_dir, 'reward_step_curve.png')\r\n"
    b"    fig.savefig(out_path, dpi=200, bbox_inches='tight')\r\n"
    b"    plt.close()\r\n"
    b"    print(f\"  reward_step_curve.png mentve  ({n_runs} run, {title_jid}, {param_label})\")\r\n"
)
new5 = (
    b"    cmap        = plt.cm.get_cmap('RdYlBu_r')\r\n"
    b"    EMA_ALPHA   = 0.05\r\n"
    b"\r\n"
    b"    # Vez\xc3\xa9rl\xc3\xa9si m\xc3\xb3dok csoportos\xc3\xadt\xc3\xa1sa\r\n"
    b"    def _mode_group(cm):\r\n"
    b"        if cm == 'actuated': return 'actuated'\r\n"
    b"        if str(cm).startswith('mixed'): return 'mixed'\r\n"
    b"        return 'random'\r\n"
    b"    full['mode_group'] = full['control_mode'].apply(_mode_group)\r\n"
    b"    mode_order  = ['random', 'mixed', 'actuated']\r\n"
    b"    mode_colors = {'random': '#e74c3c', 'mixed': '#f39c12', 'actuated': '#2980b9'}\r\n"
    b"    mode_titles = {\r\n"
    b"        'random':   'Random kontroll  (\xcf\xb5=1.0)',\r\n"
    b"        'mixed':    f'Mixed kontroll  (\xcf\xb5={MIXED_EPSILON})',\r\n"
    b"        'actuated': 'Actuated kontroll  (\xcf\xb5=0.0)',\r\n"
    b"    }\r\n"
    b"    present_modes = [m for m in mode_order if m in full['mode_group'].values]\r\n"
    b"    n_panels = max(len(present_modes), 1)\r\n"
    b"\r\n"
    b"    title_jid = junction_filter if junction_filter else '\xc3\xb6sszes junction'\r\n"
    b"    fig, axes = plt.subplots(1, n_panels, figsize=(16 * n_panels // 3 + 8, 6),\r\n"
    b"                             sharey=True)\r\n"
    b"    if n_panels == 1:\r\n"
    b"        axes = [axes]\r\n"
    b"\r\n"
    b"    for ax, mg in zip(axes, present_modes):\r\n"
    b"        sub = full[full['mode_group'] == mg]\r\n"
    b"        sub_run_ids = sorted(sub['run_id'].unique())\r\n"
    b"        all_smoothed = []\r\n"
    b"        base_color = mode_colors.get(mg, '#555555')\r\n"
    b"\r\n"
    b"        for run_id in sub_run_ids:\r\n"
    b"            rd = sub[sub['run_id'] == run_id].sort_values('step')\r\n"
    b"            if len(rd) < 5:\r\n"
    b"                continue\r\n"
    b"            steps  = rd['step'].values\r\n"
    b"            reward = rd['reward_plain_step'].values\r\n"
    b"            color  = cmap(flow_norm.get(rd['flow_level'].iloc[0], 0.5))\r\n"
    b"\r\n"
    b"            ax.plot(steps, reward, color=color, alpha=0.15, linewidth=0.6)\r\n"
    b"\r\n"
    b"            ema = np.zeros_like(reward, dtype=float)\r\n"
    b"            ema[0] = reward[0]\r\n"
    b"            for i in range(1, len(reward)):\r\n"
    b"                ema[i] = EMA_ALPHA * reward[i] + (1 - EMA_ALPHA) * ema[i - 1]\r\n"
    b"            ax.plot(steps, ema, color=color, alpha=0.5, linewidth=1.0)\r\n"
    b"            all_smoothed.append((steps, ema))\r\n"
    b"\r\n"
    b"        if all_smoothed:\r\n"
    b"            max_step     = max(s[-1] for s, _ in all_smoothed)\r\n"
    b"            common_steps = np.arange(0, int(max_step) + 1)\r\n"
    b"            interp_rows  = [np.interp(common_steps, s, e)\r\n"
    b"                            for s, e in all_smoothed if len(s) > 1]\r\n"
    b"            if interp_rows:\r\n"
    b"                gm = np.mean(interp_rows, axis=0)\r\n"
    b"                ema_g = np.zeros_like(gm)\r\n"
    b"                ema_g[0] = gm[0]\r\n"
    b"                for i in range(1, len(gm)):\r\n"
    b"                    ema_g[i] = EMA_ALPHA * gm[i] + (1 - EMA_ALPHA) * ema_g[i - 1]\r\n"
    b"                ax.plot(common_steps, ema_g, color=base_color, linewidth=2.8,\r\n"
    b"                        label=f'\xc3\x81tlag (n={len(interp_rows)} run)', zorder=10)\r\n"
    b"\r\n"
    b"        sm = plt.cm.ScalarMappable(cmap=cmap,\r\n"
    b"                                   norm=plt.Normalize(vmin=min(flow_levels), vmax=max(flow_levels)))\r\n"
    b"        sm.set_array([])\r\n"
    b"        cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)\r\n"
    b"        cbar.set_label('Flow level (veh/h)', fontsize=9)\r\n"
    b"\r\n"
    b"        ax.set_xlabel('Step', fontsize=11)\r\n"
    b"        ax.set_ylabel('Normalized reward  [0 \xe2\x80\x93 1]', fontsize=11)\r\n"
    b"        ax.set_title(mode_titles.get(mg, mg), fontsize=13, color=base_color, fontweight='bold')\r\n"
    b"        ax.set_ylim(-0.02, 1.02)\r\n"
    b"        ax.legend(fontsize=9, loc='upper left')\r\n"
    b"        ax.grid(True, alpha=0.25)\r\n"
    b"\r\n"
    b"    fig.suptitle(\r\n"
    b"        f'Step-szint\xc5\xb1 reward \xe2\x80\x94 Speed + Throughput, log-tanh  |  {title_jid}  |  {param_label} params\\n'\r\n"
    b"        f'\xce\xbc_s={mu_s:.3f}  \xcf\x83_s={std_s:.3f}  |  \xce\xbc_tp={mu_t:.3f}  \xcf\x83_tp={std_t:.3f}  |  '\r\n"
    b"        f'{n_runs} run  |  EMA \xce\xb1={EMA_ALPHA}',\r\n"
    b"        fontsize=11)\r\n"
    b"\r\n"
    b"    plt.tight_layout()\r\n"
    b"    out_path = os.path.join(output_dir, 'reward_step_curve.png')\r\n"
    b"    fig.savefig(out_path, dpi=200, bbox_inches='tight')\r\n"
    b"    plt.close()\r\n"
    b"    print(f\"  reward_step_curve.png mentve  ({n_runs} run, {title_jid}, {param_label})\")\r\n"
)
if old5 in raw:
    raw = raw.replace(old5, new5, 1); changes.append("5:OK")
elif b"mode_group" in raw:
    changes.append("5:SKIP")
else:
    changes.append("5:FAIL"); sys.exit("FAIL 5")

# 6. Main loop: add mixed episodes
old6 = (
    b"        print(f\"  Epizodok szintenkent: {EPISODES_PER_LEVEL} random + {ACTUATED_EPISODES} actuated\")\r\n"
    b"        print(f\"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s\")\r\n"
    b"        print(f\"  Kimenet: {OUTPUT_DIR}\")\r\n"
    b"        print(\"=\" * 80)\r\n"
    b"\r\n"
    b"        total_random = len(FLOW_MAX_LEVELS) * EPISODES_PER_LEVEL\r\n"
    b"        total_actuated = len(FLOW_MAX_LEVELS) * ACTUATED_EPISODES\r\n"
    b"        total_sims = total_random + total_actuated\r\n"
    b"        sim_count = 0\r\n"
    b"\r\n"
    b"        # --- Random epiz\xc3\xb3dok ---\r\n"
    b"        print(f\"\\n  [1/2] RANDOM kontroll: {total_random} epizod\")\r\n"
    b"        for flow_max in FLOW_MAX_LEVELS:\r\n"
    b"            for ep in range(EPISODES_PER_LEVEL):\r\n"
    b"                sim_count += 1\r\n"
    b"                print(f\"\\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | \"\r\n"
    b"                      f\"Epizod {ep+1}/{EPISODES_PER_LEVEL} | RANDOM ---\")\r\n"
    b"                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=\"random\", use_gui=use_gui)\r\n"
    b"\r\n"
    b"        # --- Actuated epiz\xc3\xb3dok ---\r\n"
    b"        print(f\"\\n  [2/2] ACTUATED kontroll: {total_actuated} epizod\")\r\n"
    b"        for flow_max in FLOW_MAX_LEVELS:\r\n"
    b"            for ep in range(ACTUATED_EPISODES):\r\n"
    b"                sim_count += 1\r\n"
    b"                print(f\"\\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | \"\r\n"
    b"                      f\"Epizod {ep+1}/{ACTUATED_EPISODES} | ACTUATED ---\")\r\n"
    b"                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=\"actuated\", use_gui=use_gui)\r\n"
    b"\r\n"
    b"        print(f\"\\n  Szimulacio kesz! ({total_sims} epizod: {total_random} random + {total_actuated} actuated)\")\r\n"
)
new6 = (
    b"        print(f\"  Epizodok szintenkent: {EPISODES_PER_LEVEL} random + {MIXED_EPISODES} mixed + {ACTUATED_EPISODES} actuated\")\r\n"
    b"        print(f\"  Mixed epsilon: {MIXED_EPSILON}  (random akci\xc3\xb3 val\xc3\xb3sz\xc3\xadn\xc5\xb1s\xc3\xa9ge)\")\r\n"
    b"        print(f\"  Idotartam: {DURATION}s, Warmup: {WARMUP}s, Delta: {DELTA_TIME}s\")\r\n"
    b"        print(f\"  Kimenet: {OUTPUT_DIR}\")\r\n"
    b"        print(\"=\" * 80)\r\n"
    b"\r\n"
    b"        total_random   = len(FLOW_MAX_LEVELS) * EPISODES_PER_LEVEL\r\n"
    b"        total_mixed    = len(FLOW_MAX_LEVELS) * MIXED_EPISODES\r\n"
    b"        total_actuated = len(FLOW_MAX_LEVELS) * ACTUATED_EPISODES\r\n"
    b"        total_sims     = total_random + total_mixed + total_actuated\r\n"
    b"        sim_count      = 0\r\n"
    b"\r\n"
    b"        # --- Random epiz\xc3\xb3dok ---\r\n"
    b"        print(f\"\\n  [1/3] RANDOM kontroll: {total_random} epizod\")\r\n"
    b"        for flow_max in FLOW_MAX_LEVELS:\r\n"
    b"            for ep in range(EPISODES_PER_LEVEL):\r\n"
    b"                sim_count += 1\r\n"
    b"                print(f\"\\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | \"\r\n"
    b"                      f\"Epizod {ep+1}/{EPISODES_PER_LEVEL} | RANDOM ---\")\r\n"
    b"                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=\"random\", use_gui=use_gui)\r\n"
    b"\r\n"
    b"        # --- Mixed epiz\xc3\xb3dok (\xcf\xb5-greedy) ---\r\n"
    b"        _mixed_mode = f\"mixed_{int(MIXED_EPSILON * 100)}\"\r\n"
    b"        print(f\"\\n  [2/3] MIXED kontroll ({_mixed_mode}): {total_mixed} epizod\")\r\n"
    b"        for flow_max in FLOW_MAX_LEVELS:\r\n"
    b"            for ep in range(MIXED_EPISODES):\r\n"
    b"                sim_count += 1\r\n"
    b"                print(f\"\\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | \"\r\n"
    b"                      f\"Epizod {ep+1}/{MIXED_EPISODES} | MIXED (\xcf\xb5={MIXED_EPSILON}) ---\")\r\n"
    b"                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=_mixed_mode, use_gui=use_gui)\r\n"
    b"\r\n"
    b"        # --- Actuated epiz\xc3\xb3dok ---\r\n"
    b"        print(f\"\\n  [3/3] ACTUATED kontroll: {total_actuated} epizod\")\r\n"
    b"        for flow_max in FLOW_MAX_LEVELS:\r\n"
    b"            for ep in range(ACTUATED_EPISODES):\r\n"
    b"                sim_count += 1\r\n"
    b"                print(f\"\\n--- [{sim_count}/{total_sims}] Flow max: {flow_max}/h | \"\r\n"
    b"                      f\"Epizod {ep+1}/{ACTUATED_EPISODES} | ACTUATED ---\")\r\n"
    b"                run_simulation(flow_max, ep, OUTPUT_DIR, control_mode=\"actuated\", use_gui=use_gui)\r\n"
    b"\r\n"
    b"        print(f\"\\n  Szimulacio kesz! ({total_sims} epizod: \"\r\n"
    b"              f\"{total_random} random + {total_mixed} mixed + {total_actuated} actuated)\")\r\n"
)
if old6 in raw:
    raw = raw.replace(old6, new6, 1); changes.append("6:OK")
elif b"[2/3] MIXED" in raw:
    changes.append("6:SKIP")
else:
    changes.append("6:FAIL"); sys.exit("FAIL 6")

print("All changes:", changes)
with open('metric_collection_per_junction.py', 'wb') as f:
    f.write(raw)
print("DONE")
