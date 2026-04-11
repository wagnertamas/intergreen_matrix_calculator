#!/usr/bin/env python3
"""Patch: extend junction_reward_params.json with wait/speedstd/halt params for wait_triplet_tpstdhalt mode"""
import sys

with open('metric_collection_per_junction.py', 'rb') as f:
    raw = f.read()

changes = []

# ── 1. Global params block: add MU_WAIT / MU_SPEEDSTD / MU_HALT after MU_THROUGHPUT ──────────
old1 = (
    b"    GLOBAL_PARAMS['MU_THROUGHPUT'] = round(glob_mu_t, 6)\r\n"
    b"    GLOBAL_PARAMS['STD_THROUGHPUT'] = round(glob_std_t, 6)\r\n"
    b"\r\n"
    b"    print(f\"\\n  Globalis parameterek (log-space, median/IQR):\")\r\n"
    b"    print(f\"    MU_SPEED:      {glob_mu_s:.6f}\")\r\n"
    b"    print(f\"    STD_SPEED:     {glob_std_s:.6f}\")\r\n"
    b"    print(f\"    MU_THROUGHPUT: {glob_mu_t:.6f}\")\r\n"
    b"    print(f\"    STD_THROUGHPUT:{glob_std_t:.6f}\")\r\n"
)
new1 = (
    b"    GLOBAL_PARAMS['MU_THROUGHPUT'] = round(glob_mu_t, 6)\r\n"
    b"    GLOBAL_PARAMS['STD_THROUGHPUT'] = round(glob_std_t, 6)\r\n"
    b"\r\n"
    b"    # --- Extra glob\xc3\xa1lis param\xc3\xa9terek: wait_triplet_tpstdhalt m\xc3\xb3dhoz ---\r\n"
    b"    _g_wait = valid_global['TotalWaitingTime'].values if 'TotalWaitingTime' in valid_global.columns else np.array([])\r\n"
    b"    _g_std  = valid_global['SpeedStd'].values         if 'SpeedStd'         in valid_global.columns else np.array([])\r\n"
    b"    _g_halt = valid_global['HaltRatio'].values        if 'HaltRatio'        in valid_global.columns else np.array([])\r\n"
    b"    glob_mu_w,  glob_std_w  = robust_log_params(_g_wait)  or (6.189495, 2.666679)\r\n"
    b"    glob_mu_ss, glob_std_ss = robust_log_params(_g_std)   or (0.767463, 0.671022)\r\n"
    b"    glob_mu_h,  glob_std_h  = robust_log_params(_g_halt)  or (-0.445295, 0.510940)\r\n"
    b"    if glob_mu_w  is None: glob_mu_w,  glob_std_w  = 6.189495, 2.666679\r\n"
    b"    if glob_mu_ss is None: glob_mu_ss, glob_std_ss = 0.767463, 0.671022\r\n"
    b"    if glob_mu_h  is None: glob_mu_h,  glob_std_h  = -0.445295, 0.510940\r\n"
    b"\r\n"
    b"    print(f\"\\n  Globalis parameterek (log-space, median/IQR):\")\r\n"
    b"    print(f\"    MU_SPEED:      {glob_mu_s:.6f}\")\r\n"
    b"    print(f\"    STD_SPEED:     {glob_std_s:.6f}\")\r\n"
    b"    print(f\"    MU_THROUGHPUT: {glob_mu_t:.6f}\")\r\n"
    b"    print(f\"    STD_THROUGHPUT:{glob_std_t:.6f}\")\r\n"
)
if old1 in raw:
    raw = raw.replace(old1, new1, 1); changes.append("glob_extra:OK")
elif b"glob_mu_w" in raw:
    changes.append("glob_extra:SKIP")
else:
    sys.exit("FAIL glob_extra")

# ── 2. Per-junction dict: add WAIT / SPEEDSTD / HALT keys ────────────────────────────────────
old2 = (
    b"        junction_params[jid] = {\r\n"
    b"            'MU_SPEED': round(mu_s, 6),\r\n"
    b"            'STD_SPEED': round(std_s, 6),\r\n"
    b"            'MU_THROUGHPUT': round(mu_t, 6),\r\n"
    b"            'STD_THROUGHPUT': round(std_t, 6),\r\n"
    b"        }\r\n"
)
new2 = (
    b"        # Extra param\xc3\xa9terek a wait_triplet_tpstdhalt m\xc3\xb3dhoz\r\n"
    b"        _jw   = jdf_valid['TotalWaitingTime'].values if 'TotalWaitingTime' in jdf_valid.columns else np.array([])\r\n"
    b"        _jss  = jdf_valid['SpeedStd'].values         if 'SpeedStd'         in jdf_valid.columns else np.array([])\r\n"
    b"        _jh   = jdf_valid['HaltRatio'].values        if 'HaltRatio'        in jdf_valid.columns else np.array([])\r\n"
    b"        mu_w,  std_w  = robust_log_params(_jw)  if len(_jw) > 5  else (None, None)\r\n"
    b"        mu_ss, std_ss = robust_log_params(_jss) if len(_jss) > 5 else (None, None)\r\n"
    b"        mu_h,  std_h  = robust_log_params(_jh)  if len(_jh) > 5  else (None, None)\r\n"
    b"        if mu_w  is None: mu_w,  std_w  = glob_mu_w,  glob_std_w\r\n"
    b"        if mu_ss is None: mu_ss, std_ss = glob_mu_ss, glob_std_ss\r\n"
    b"        if mu_h  is None: mu_h,  std_h  = glob_mu_h,  glob_std_h\r\n"
    b"\r\n"
    b"        junction_params[jid] = {\r\n"
    b"            'MU_SPEED':       round(mu_s, 6),\r\n"
    b"            'STD_SPEED':      round(std_s, 6),\r\n"
    b"            'MU_THROUGHPUT':  round(mu_t, 6),\r\n"
    b"            'STD_THROUGHPUT': round(std_t, 6),\r\n"
    b"            'MU_WAIT':        round(mu_w, 6),\r\n"
    b"            'STD_WAIT':       round(std_w, 6),\r\n"
    b"            'MU_SPEEDSTD':    round(mu_ss, 6),\r\n"
    b"            'STD_SPEEDSTD':   round(std_ss, 6),\r\n"
    b"            'MU_HALT':        round(mu_h, 6),\r\n"
    b"            'STD_HALT':       round(std_h, 6),\r\n"
    b"        }\r\n"
)
if old2 in raw:
    raw = raw.replace(old2, new2, 1); changes.append("per_jid:OK")
elif b"MU_WAIT" in raw:
    changes.append("per_jid:SKIP")
else:
    sys.exit("FAIL per_jid")

# ── 3. JSON global block: add extra keys ──────────────────────────────────────────────────────
old3 = (
    b"    output_json = {\r\n"
    b"        'reward_function': 'AvgSpeed + Throughput, log-tanh normalization',\r\n"
    b"        'global': {\r\n"
    b"            'MU_SPEED': round(glob_mu_s, 6),\r\n"
    b"            'STD_SPEED': round(glob_std_s, 6),\r\n"
    b"            'MU_THROUGHPUT': round(glob_mu_t, 6),\r\n"
    b"            'STD_THROUGHPUT': round(glob_std_t, 6),\r\n"
    b"        },\r\n"
    b"        'per_junction': junction_params,\r\n"
    b"    }\r\n"
)
new3 = (
    b"    output_json = {\r\n"
    b"        'reward_function': 'multi-mode log-tanh normalization',\r\n"
    b"        'global': {\r\n"
    b"            'MU_SPEED':       round(glob_mu_s, 6),\r\n"
    b"            'STD_SPEED':      round(glob_std_s, 6),\r\n"
    b"            'MU_THROUGHPUT':  round(glob_mu_t, 6),\r\n"
    b"            'STD_THROUGHPUT': round(glob_std_t, 6),\r\n"
    b"            'MU_WAIT':        round(glob_mu_w, 6),\r\n"
    b"            'STD_WAIT':       round(glob_std_w, 6),\r\n"
    b"            'MU_SPEEDSTD':    round(glob_mu_ss, 6),\r\n"
    b"            'STD_SPEEDSTD':   round(glob_std_ss, 6),\r\n"
    b"            'MU_HALT':        round(glob_mu_h, 6),\r\n"
    b"            'STD_HALT':       round(glob_std_h, 6),\r\n"
    b"        },\r\n"
    b"        'per_junction': junction_params,\r\n"
    b"    }\r\n"
)
if old3 in raw:
    raw = raw.replace(old3, new3, 1); changes.append("json_global:OK")
elif b"MU_WAIT.*round.*glob_mu_w" in raw or b"'MU_WAIT':        round" in raw:
    changes.append("json_global:SKIP")
else:
    sys.exit("FAIL json_global")

print("Changes:", changes)
with open('metric_collection_per_junction.py', 'wb') as f:
    f.write(raw)
print("DONE")
