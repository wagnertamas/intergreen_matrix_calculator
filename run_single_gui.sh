#!/bin/bash
# =============================================================================
# Egyszeri futtatás GUI-val (Vizuális ellenőrzéshez)
# =============================================================================
# Használat: ./run_single_gui.sh
# Módosíthatod a fájlban az ALGO, REWARD vagy JUNCTION változókat.
# =============================================================================

set -euo pipefail

# Kényszerítjük a traci-t (process-level isolation), mert a GUI mód csak így megy
export USE_LIBSUMO=0

# --- Alapértelmezések (itt átírhatod, ha mást akarsz tesztelni) ---
JUNCTION="R1C1_C"
ALGO="qrdqn"                 # Lehet: ppo, qrdqn, dqn, a2c
REWARD="wait_haltratio"    # Lehet: speed_throughput, halt_ratio, co2_speedstd, wait_triplet_tpstdhalt, wait_haltratio
TIMESTEPS=50000              # Rövidebb idő a teszthez

echo "============================================================"
echo "[INFO] Szimuláció indítása GUI módban..."
echo "[INFO] Kereszteződés: ${JUNCTION}"
echo "[INFO] Algoritmus:    ${ALGO}"
echo "[INFO] Reward mód:    ${REWARD}"
echo "============================================================"

# Futtatás
python main_headless.py \
    --junction "${JUNCTION}" \
    --algorithm "${ALGO}" \
    --reward-mode "${REWARD}" \
    --timesteps "${TIMESTEPS}" \
    --num-layers 2 \
    --layer-size 64 \
    --gui

echo "[INFO] Futás befejeződött."
