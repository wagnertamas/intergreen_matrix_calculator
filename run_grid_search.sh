#!/bin/bash
# =============================================================================
# Grid Search: Algorithm × Reward × Network Architecture  (PARALLEL)
# =============================================================================
# Minden kombináció 3× fut le az R1C1_C junction-re, párhuzamosan.
# Befejezett futásokat kihagyja (grid_done/ marker fájlok alapján).
#
# Használat:
#   ./run_grid_search.sh                  # Teljes grid, 4 párhuzamos job
#   ./run_grid_search.sh -j 6            # 6 párhuzamos job
#   ./run_grid_search.sh -t 50000        # Custom timesteps
#   ./run_grid_search.sh -r 2            # 2 ismétlés
#   ./run_grid_search.sh -j 6 -t 50000  # Kombinálva
#
# Folytatás: egyszerűen futtasd újra — a kész futásokat kihagyja.
# Leállítás: Ctrl+C — a futó jobok befejeződnek, újraindításkor folytatódik.
# =============================================================================

set -euo pipefail

# Kényszerítjük a traci-t (process-level isolation), mert a libsumo C++ párhuzamosan segfault-ot okoz macOS-en!
export USE_LIBSUMO=0

# --- Alapértelmezések ---
JUNCTION="R1C1_C"
PROJECT="sumo-rl-stat-2"
TIMESTEPS=100000
REPEATS=1
PARALLEL_JOBS=9  # 12 magból 6 párhuzamos SUMO szimuláció

# Algoritmusok
ALGORITHMS=(qrdqn ppo)
#ALGORITHMS=(qrdqn dqn ppo a2c)

# Reward módok
REWARDS=(speed_throughput)
#REWARDS=(speed_throughput halt_ratio co2_speedstd)

# Neurális háló méretek
LAYERS=(1 2 3)
NEURONS=(32 64 128)

# --- CLI argumentumok ---
while getopts "j:t:r:" opt; do
    case $opt in
        j) PARALLEL_JOBS="$OPTARG" ;;
        t) TIMESTEPS="$OPTARG" ;;
        r) REPEATS="$OPTARG" ;;
        *) echo "Használat: $0 [-j jobs] [-t timesteps] [-r repeats]"; exit 1 ;;
    esac
done

# --- Marker könyvtár ---
DONE_DIR="grid_done"
LOG_DIR="grid_logs"
mkdir -p "$DONE_DIR" "$LOG_DIR"

# --- Job lista generálás ---
ALL_JOBS=()
SKIP_COUNT=0
for ALGO in "${ALGORITHMS[@]}"; do
    for REWARD in "${REWARDS[@]}"; do
        for NL in "${LAYERS[@]}"; do
            for NS in "${NEURONS[@]}"; do
                for REP in $(seq 1 "$REPEATS"); do
                    RUN_NAME="${ALGO}_${REWARD}_${NL}x${NS}_${REP}_${JUNCTION}"
                    if [[ -f "${DONE_DIR}/${RUN_NAME}.done" ]]; then
                        SKIP_COUNT=$((SKIP_COUNT + 1))
                    else
                        ALL_JOBS+=("${ALGO}|${REWARD}|${NL}|${NS}|${REP}|${RUN_NAME}")
                    fi
                done
            done
        done
    done
done

TOTAL_GRID=$(( ${#ALGORITHMS[@]} * ${#REWARDS[@]} * ${#LAYERS[@]} * ${#NEURONS[@]} * REPEATS ))
REMAINING=${#ALL_JOBS[@]}

echo "============================================================"
echo "  GRID SEARCH — ${JUNCTION}  (PARALLEL)"
echo "============================================================"
echo "  Algoritmusok:    ${ALGORITHMS[*]}"
echo "  Reward módok:    ${REWARDS[*]}"
echo "  Layers:          ${LAYERS[*]}"
echo "  Neurons:         ${NEURONS[*]}"
echo "  Ismétlések:      ${REPEATS}"
echo "  Timesteps:       ${TIMESTEPS}"
echo "  Projekt:         ${PROJECT}"
echo "  Párhuzamos:      ${PARALLEL_JOBS}"
echo "  ---"
echo "  Összes:          ${TOTAL_GRID}"
echo "  Kész (kihagyva): ${SKIP_COUNT}"
echo "  Hátralevő:       ${REMAINING}"
echo "============================================================"
echo ""

# --- WandB árva runok ellenőrzése ---
echo "WandB szinkron ellenőrzés..."
python wandb_cleanup.py --project "${PROJECT}" --done-dir "${DONE_DIR}" || true
echo ""

if [[ $REMAINING -eq 0 ]]; then
    echo "Minden futás kész! Nincs teendő."
    exit 0
fi

# --- Párhuzamos futtatás ---
RUNNING_PIDS=()
COMPLETED=0
FAILED=0

# Ctrl+C kezelés: ÖSSZES gyerekfolyamat (python is) leállítása
cleanup() {
    echo ""
    echo "Leállítás kérve — összes gyerekfolyamat leállítása..."
    # Teljes processz-csoport leállítása (shell + python gyerekek)
    kill -- -$$ 2>/dev/null || true
    # Biztonsági háló: ha maradt volna main_headless.py
    pkill -f "main_headless.py" 2>/dev/null || true
    echo "Leállítva. ${COMPLETED}/${REMAINING} kész ebben a menetben."
    echo "Futtasd újra a folytatáshoz."
    exit 1
}
trap cleanup SIGINT SIGTERM

# Egy job futtatása
run_job() {
    local ALGO="$1" REWARD="$2" NL="$3" NS="$4" REP="$5" RUN_NAME="$6"
    local LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

    echo "[START] ${RUN_NAME}"

    if WANDB_RUN_NAME="${RUN_NAME}" \
       python main_headless.py \
           --junction "${JUNCTION}" \
           --algorithm "${ALGO}" \
           --reward-mode "${REWARD}" \
           --num-layers "${NL}" \
           --layer-size "${NS}" \
           --timesteps "${TIMESTEPS}" \
           --project "${PROJECT}" \
           > "${LOG_FILE}" 2>&1; then
        # Sikeres — marker fájl
        touch "${DONE_DIR}/${RUN_NAME}.done"
        echo "[DONE]  ${RUN_NAME}"
        return 0
    else
        echo "[FAIL]  ${RUN_NAME} — lásd: ${LOG_FILE}"
        return 1
    fi
}

# Semaphore: max PARALLEL_JOBS egyidejű futás
JOB_IDX=0
for JOB in "${ALL_JOBS[@]}"; do
    # Várj ha elértük a limitet
    while [[ ${#RUNNING_PIDS[@]} -ge $PARALLEL_JOBS ]]; do
        # Várj bármelyik gyerekfolyamat befejezésére
        NEW_PIDS=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            else
                wait "$pid" 2>/dev/null && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))
            fi
        done
        if [[ ${#NEW_PIDS[@]} -gt 0 ]]; then
            RUNNING_PIDS=("${NEW_PIDS[@]}")
        else
            RUNNING_PIDS=()
        fi
        if [[ ${#RUNNING_PIDS[@]} -ge $PARALLEL_JOBS ]]; then
            sleep 2
        fi
    done

    # Job indítás háttérben
    IFS='|' read -r ALGO REWARD NL NS REP RUN_NAME <<< "$JOB"
    JOB_IDX=$((JOB_IDX + 1))
    echo ""
    echo "━━━ [${JOB_IDX}/${REMAINING}] Indítás (${#RUNNING_PIDS[@]}+1/${PARALLEL_JOBS} slot) ━━━"

    run_job "$ALGO" "$REWARD" "$NL" "$NS" "$REP" "$RUN_NAME" &
    RUNNING_PIDS+=($!)
done

# Várakozás az összes háttérfolyamatra
echo ""
echo "Várakozás az utolsó jobok befejezésére..."
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null && COMPLETED=$((COMPLETED + 1)) || FAILED=$((FAILED + 1))
done

echo ""
echo "============================================================"
echo "  GRID SEARCH KÉSZ"
echo "  Sikeres:  ${COMPLETED}"
echo "  Hibás:    ${FAILED}"
echo "  Korábban: ${SKIP_COUNT}"
echo "============================================================"
