#!/bin/bash
# =============================================================================
# Grid Search: Algorithm × Reward × Network Architecture
# =============================================================================
# Minden kombináció 3× fut le az R1C1_C junction-re.
#
# Használat:
#   ./run_grid_search.sh              # Teljes grid (4×3×15×3 = 540 run)
#   ./run_grid_search.sh 50000        # Custom timesteps
#   ./run_grid_search.sh 100000 2     # Custom timesteps + ismétlések
# =============================================================================

set -e

# --- Konfiguráció ---
JUNCTION="R1C1_C"
PROJECT="sumo-rl-stat"
TIMESTEPS="${1:-100000}"
REPEATS="${2:-3}"

# Algoritmusok
ALGORITHMS=(qrdqn dqn ppo a2c)

# Reward módok
REWARDS=(speed_throughput halt_ratio co2_speedstd)

# Neurális háló méretek: layers × neurons
LAYERS=(1 2 3)
NEURONS=(8 16 32 64 128)

# --- Összesítés ---
TOTAL_COMBOS=$(( ${#ALGORITHMS[@]} * ${#REWARDS[@]} * ${#LAYERS[@]} * ${#NEURONS[@]} * REPEATS ))
echo "============================================================"
echo "  GRID SEARCH — ${JUNCTION}"
echo "============================================================"
echo "  Algoritmusok:    ${ALGORITHMS[*]}"
echo "  Reward módok:    ${REWARDS[*]}"
echo "  Layers:          ${LAYERS[*]}"
echo "  Neurons:         ${NEURONS[*]}"
echo "  Ismétlések:      ${REPEATS}"
echo "  Timesteps:       ${TIMESTEPS}"
echo "  Projekt:         ${PROJECT}"
echo "  Összes futás:    ${TOTAL_COMBOS}"
echo "============================================================"
echo ""

RUN_IDX=0

for ALGO in "${ALGORITHMS[@]}"; do
    for REWARD in "${REWARDS[@]}"; do
        for NL in "${LAYERS[@]}"; do
            for NS in "${NEURONS[@]}"; do
                for REP in $(seq 1 "$REPEATS"); do
                    RUN_IDX=$((RUN_IDX + 1))

                    # Run név: algo_reward_NLxNS_rep_junction
                    # Pl: qrdqn_speed_throughput_2x64_1_R1C1_C
                    RUN_NAME="${ALGO}_${REWARD}_${NL}x${NS}_${REP}_${JUNCTION}"

                    echo ""
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    echo "  [${RUN_IDX}/${TOTAL_COMBOS}] ${RUN_NAME}"
                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                    WANDB_RUN_NAME="${RUN_NAME}" \
                    python main_headless.py \
                        --junction "${JUNCTION}" \
                        --algorithm "${ALGO}" \
                        --reward-mode "${REWARD}" \
                        --num-layers "${NL}" \
                        --layer-size "${NS}" \
                        --timesteps "${TIMESTEPS}" \
                        --project "${PROJECT}"

                    echo "  ✓ ${RUN_NAME} kész"

                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "  GRID SEARCH KÉSZ — ${RUN_IDX} futás befejezve"
echo "============================================================"
