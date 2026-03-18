#!/bin/bash
# =============================================================================
# SUMO RL Training — Unified Launcher
# =============================================================================
# Használat: ./start.sh
#
# Interaktív menü:
#   1. WandB projekt
#   2. Mód (Headless / GUI / Sweep / Transfer Learning)
#   3. Platform (Lokális / Docker)
#   4. Párhuzamos futások száma
#   5. Timesteps
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Cleanup: Ctrl+C megöli az ÖSSZES gyermek processt ---
CHILD_PIDS=()

cleanup() {
    echo ""
    echo -e "\033[1;33m[!] Leállítás... összes futás megszakítása.\033[0m"
    # Először TERM jelzés az összes gyerek process-csoportnak (beleértve a sweep_runner.py-t is)
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  Leállítás: PID $pid (+ gyerek processek)"
            # Process group kill: megöli a wandb agent-et ÉS az általa indított sweep_runner.py-t is
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # Várunk max 3 mp-et, utána SIGKILL
    sleep 3
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  \033[0;31mKényszerített leállítás: PID $pid\033[0m"
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    # Háttérben maradt SUMO processzek leállítása
    pkill -f "sumo.*mega_catalogue" 2>/dev/null || true
    echo -e "\033[0;32m[✓] Minden leállítva.\033[0m"
    exit 0
}

trap cleanup SIGINT SIGTERM

# --- Színek ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

header() {
    clear
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}SUMO RL Traffic Light Control — Training Launcher${NC}  ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# --- .env ellenőrzés ---
check_wandb_key() {
    if [ -f .env ]; then
        source .env
    fi
    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${YELLOW}[!] WANDB_API_KEY nincs beállítva.${NC}"
        read -p "    WandB API key (vagy üres a skip-hez): " key
        if [ -n "$key" ]; then
            export WANDB_API_KEY="$key"
            echo "WANDB_API_KEY=$key" > .env
            echo -e "${GREEN}    Mentve: .env${NC}"
        fi
    else
        echo -e "${GREEN}[✓] WANDB_API_KEY beállítva${NC}"
    fi
}

# --- 1. Projekt választás ---
select_project() {
    echo ""
    echo -e "${BOLD}1. WandB projekt név?${NC}"
    echo "  1) sumo-rl-single       (alap multi-agent tanítás)"
    echo "  2) sumo-rl-sweep        (hiperparaméter sweep)"
    echo "  3) sumo-rl-finetune     (transfer learning finomhangolás)"
    echo "  4) Egyéni..."
    read -p "  Választás [1]: " proj_choice
    case "${proj_choice:-1}" in
        1) PROJECT="sumo-rl-single" ;;
        2) PROJECT="sumo-rl-sweep" ;;
        3) PROJECT="sumo-rl-finetune" ;;
        4) read -p "  Projekt név: " PROJECT ;;
        *) PROJECT="$proj_choice" ;;  # Ha nem szám, egyéni névként kezeljük
    esac
    echo -e "  → ${GREEN}$PROJECT${NC}"
}

# --- 2. Mód választás ---
select_mode() {
    echo ""
    echo -e "${BOLD}2. Futtatási mód?${NC}"
    echo "  1) Headless tanítás        (single run, nincs GUI)"
    echo "  2) GUI tanítás             (SUMO-gui ablakkal — csak lokálisan)"
    echo "  3) WandB Sweep             (hiperparaméter keresés)"
    echo "  4) Transfer Learning       (pre-trained modell finomhangolása)"
    read -p "  Választás [1]: " mode_choice
    MODE="${mode_choice:-1}"
}

# --- 3. Platform választás ---
select_platform() {
    echo ""
    echo -e "${BOLD}3. Hol fusson?${NC}"
    echo "  1) Lokálisan (python)"
    echo "  2) Docker konténerben"
    read -p "  Választás [1]: " platform_choice
    PLATFORM="${platform_choice:-1}"

    if [ "$PLATFORM" == "2" ] && [ "$MODE" == "2" ]; then
        echo -e "${RED}  [!] GUI mód nem támogatott Dockerben. Visszaállítás: Headless${NC}"
        MODE=1
    fi
}

# --- 4. Párhuzamosság ---
select_parallelism() {
    echo ""
    # Sweep módban és headless módban is értelmesek a párhuzamos futások
    if [ "$MODE" == "3" ] || [ "$MODE" == "1" ]; then
        echo -e "${BOLD}4. Hány párhuzamos futás?${NC}"
        echo -e "  ${DIM}(Sweep: mindegyik más hiperparamétert próbál"
        echo -e "   Headless: ugyanazok a paraméterek, több random seed)${NC}"
        read -p "  Darabszám [1]: " parallel_count
        PARALLEL="${parallel_count:-1}"
    else
        PARALLEL=1
    fi
    if [ "$PARALLEL" -gt 1 ]; then echo -e "  → ${GREEN}$PARALLEL párhuzamos futás${NC}"; fi
}

# --- 5. Timesteps ---
select_timesteps() {
    echo ""
    echo -e "${BOLD}5. Total timesteps?${NC}"
    echo "  1)  50,000   (gyors teszt, ~30 perc)"
    echo "  2) 100,000   (standard, ~1 óra)"
    echo "  3) 200,000   (hosszú, ~2-3 óra)"
    echo "  4) 500,000   (nagyon hosszú, ~6-8 óra)"
    echo "  5) Egyéni..."
    read -p "  Választás [2]: " ts_choice
    case "${ts_choice:-2}" in
        1) TIMESTEPS=50000 ;;
        2) TIMESTEPS=100000 ;;
        3) TIMESTEPS=200000 ;;
        4) TIMESTEPS=500000 ;;
        5) read -p "  Timesteps: " TIMESTEPS ;;
        *) TIMESTEPS=100000 ;;
    esac
    echo -e "  → ${GREEN}$TIMESTEPS steps${NC}"
}

# --- Transfer learning model ---
select_model() {
    MODEL_PATH=""
    if [ "$MODE" == "4" ]; then
        echo ""
        echo -e "${BOLD}Pre-trained modell útvonala (.zip vagy .onnx):${NC}"
        read -p "  Path: " MODEL_PATH
        if [ ! -f "$MODEL_PATH" ]; then
            echo -e "${RED}  [!] Fájl nem található: $MODEL_PATH${NC}"
            exit 1
        fi
        echo -e "  → ${GREEN}$MODEL_PATH${NC}"
    fi
}

# ===========================================================================
# FUTTATÁS
# ===========================================================================

run_local() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local gui_flag=""
    [ "$MODE" == "2" ] && gui_flag="--gui"

    case "$MODE" in
        1|2)
            # Headless / GUI — egy vagy több párhuzamos futás
            if [ "$PARALLEL" -le 1 ]; then
                echo -e "${GREEN}[▶] Tanítás indítása (libsumo)...${NC}"
                USE_LIBSUMO=1 python main_headless.py \
                    --config training_config.yaml \
                    --timesteps "$TIMESTEPS" \
                    --project "$PROJECT" \
                    $gui_flag
            else
                echo -e "${GREEN}[▶] $PARALLEL párhuzamos tanítás indítása...${NC}"
                echo -e "  ${DIM}#1 = libsumo (gyors), #2+ = traci (parallel safe)${NC}"
                CHILD_PIDS=()
                for i in $(seq 1 "$PARALLEL"); do
                    if [ "$i" -eq 1 ]; then
                        echo -e "${GREEN}  [▶] Run #$i indítása (libsumo)...${NC}"
                        USE_LIBSUMO=1 python main_headless.py \
                            --config training_config.yaml \
                            --timesteps "$TIMESTEPS" \
                            --project "$PROJECT" \
                            $gui_flag &
                    else
                        echo -e "${GREEN}  [▶] Run #$i indítása (traci)...${NC}"
                        USE_LIBSUMO=0 python main_headless.py \
                            --config training_config.yaml \
                            --timesteps "$TIMESTEPS" \
                            --project "$PROJECT" \
                            $gui_flag &
                    fi
                    CHILD_PIDS+=($!)
                    sleep 2  # Staggered start — ne egyszerre próbáljon mindegyik SUMO-t indítani
                done
                echo ""
                echo -e "${GREEN}  $PARALLEL futás elindítva. PID-ek: ${CHILD_PIDS[*]}${NC}"
                echo -e "  ${DIM}Várakozás az összes befejezésére... (Ctrl+C a leállításhoz)${NC}"
                wait "${CHILD_PIDS[@]}"
            fi
            ;;
        3)
            echo -e "${GREEN}[▶] WandB Sweep indítása...${NC}"

            # Exportáljuk a projekt és timesteps-et env var-ként,
            # mert a wandb agent nem engedi custom CLI args-ot átadni
            export SWEEP_PROJECT="$PROJECT"
            export SWEEP_TIMESTEPS="$TIMESTEPS"

            echo ""
            echo -e "${YELLOW}  1) Új sweep létrehozása${NC}"
            echo -e "${YELLOW}  2) Meglévő sweep-hez csatlakozás${NC}"
            read -p "  Választás [1]: " sweep_mode

            if [ "${sweep_mode:-1}" == "1" ]; then
                echo -e "  Sweep config: data/sweep_config.yaml"
                SWEEP_OUTPUT=$(wandb sweep data/sweep_config.yaml --project "$PROJECT" 2>&1)
                echo "$SWEEP_OUTPUT"
                # Sweep ID kinyerése — különböző wandb verziók más formátumban adják vissza
                SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oE '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9]+' | tail -1 || true)
                if [ -z "$SWEEP_ID" ]; then
                    echo -e "${YELLOW}  Nem sikerült kinyerni a sweep ID-t automatikusan.${NC}"
                    read -p "  Sweep ID (entity/project/id): " SWEEP_ID
                fi
            else
                read -p "  Sweep ID (entity/project/id): " SWEEP_ID
            fi

            echo -e "  → Sweep: ${GREEN}$SWEEP_ID${NC}"
            echo ""

            CHILD_PIDS=()
            echo -e "  ${DIM}Sweep: minden agent traci-t használ (parallel safe, libsumo crash-el sweep-ben)${NC}"
            echo -e "  ${DIM}Ctrl+C a leállításhoz${NC}"
            for i in $(seq 1 "$PARALLEL"); do
                echo -e "${GREEN}  [▶] Sweep Agent #$i (traci) indítása...${NC}"
                USE_LIBSUMO=0 wandb agent "$SWEEP_ID" &
                CHILD_PIDS+=($!)
                sleep 1
            done
            echo -e "${GREEN}  $PARALLEL sweep agent fut. PID-ek: ${CHILD_PIDS[*]}${NC}"
            wait "${CHILD_PIDS[@]}"
            ;;
        4)
            echo -e "${GREEN}[▶] Transfer Learning indítása...${NC}"
            python main_headless.py \
                --config training_config.yaml \
                --timesteps "$TIMESTEPS" \
                --project "$PROJECT" \
                --load-model "$MODEL_PATH" \
                $gui_flag
            ;;
    esac
}

run_docker() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # GPU/CPU választás
    echo -e "${BOLD}Docker: GPU vagy CPU?${NC}"
    echo "  1) CPU  (alapértelmezett, mindig működik)"
    echo "  2) GPU  (NVIDIA CUDA — nvidia-docker szükséges)"
    read -p "  Választás [1]: " docker_hw
    if [ "${docker_hw:-1}" == "2" ]; then
        DOCKER_SERVICE="sumo-trainer-gpu"
        DOCKER_TAG="gpu"
        echo -e "  → ${GREEN}GPU mód (CUDA)${NC}"
    else
        DOCKER_SERVICE="sumo-trainer"
        DOCKER_TAG="cpu"
        echo -e "  → ${GREEN}CPU mód${NC}"
    fi

    # Docker image ellenőrzése / build
    if ! docker image inspect "sumo-rl-agent:${DOCKER_TAG}" &>/dev/null; then
        echo -e "${YELLOW}[!] Docker image (${DOCKER_TAG}) nem létezik, build indítása...${NC}"
        docker-compose build "$DOCKER_SERVICE"
    else
        echo -e "${BOLD}Docker image újraépítése?${NC}"
        read -p "  Rebuild? [n]: " rebuild
        if [ "${rebuild:-n}" == "y" ]; then
            docker-compose build "$DOCKER_SERVICE"
        fi
    fi

    case "$MODE" in
        1)
            if [ "$PARALLEL" -le 1 ]; then
                echo -e "${GREEN}[▶] Docker headless tanítás ($DOCKER_TAG)...${NC}"
                docker-compose run --rm "$DOCKER_SERVICE" \
                    python main_headless.py \
                        --config training_config.yaml \
                        --timesteps "$TIMESTEPS" \
                        --project "$PROJECT"
            else
                echo -e "${GREEN}[▶] Docker $PARALLEL párhuzamos tanítás ($DOCKER_TAG)...${NC}"
                for i in $(seq 1 "$PARALLEL"); do
                    container_name="train-agent-$i"
                    echo -e "${GREEN}  [▶] $container_name indítása...${NC}"
                    docker-compose run -d --name "$container_name" "$DOCKER_SERVICE" \
                        python main_headless.py \
                            --config training_config.yaml \
                            --timesteps "$TIMESTEPS" \
                            --project "$PROJECT"
                done
                echo ""
                echo -e "${GREEN}  $PARALLEL agent fut. Monitorozás:${NC}"
                echo -e "  docker logs -f train-agent-1"
            fi
            ;;
        3)
            echo -e "${GREEN}[▶] Docker Sweep ($PARALLEL agent, $DOCKER_TAG)...${NC}"
            export SWEEP_PROJECT="$PROJECT"
            export SWEEP_TIMESTEPS="$TIMESTEPS"
            echo ""
            echo -e "${YELLOW}  1) Új sweep létrehozása (lokálisan, aztán docker agent-ek)${NC}"
            echo -e "${YELLOW}  2) Meglévő sweep-hez csatlakozás${NC}"
            read -p "  Választás [1]: " sweep_mode

            if [ "${sweep_mode:-1}" == "1" ]; then
                SWEEP_OUTPUT=$(wandb sweep data/sweep_config.yaml --project "$PROJECT" 2>&1)
                echo "$SWEEP_OUTPUT"
                SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oE '[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[a-zA-Z0-9]+' | tail -1 || true)
                if [ -z "$SWEEP_ID" ]; then
                    read -p "  Sweep ID (entity/project/id): " SWEEP_ID
                fi
            else
                read -p "  Sweep ID (entity/project/id): " SWEEP_ID
            fi

            echo -e "  → Sweep: ${GREEN}$SWEEP_ID${NC}"

            for i in $(seq 1 "$PARALLEL"); do
                container_name="sweep-agent-$i"
                echo -e "${GREEN}  [▶] $container_name indítása...${NC}"
                docker-compose run -d --name "$container_name" "$DOCKER_SERVICE" \
                    wandb agent "$SWEEP_ID"
            done
            echo ""
            echo -e "${GREEN}  $PARALLEL sweep agent fut. Monitorozás:${NC}"
            echo -e "  docker logs -f sweep-agent-1"
            echo -e "  docker-compose ps"
            ;;
        4)
            echo -e "${GREEN}[▶] Docker Transfer Learning ($DOCKER_TAG)...${NC}"
            docker-compose run --rm \
                -v "$(cd "$(dirname "$MODEL_PATH")" && pwd):/app/pretrained" \
                "$DOCKER_SERVICE" \
                python main_headless.py \
                    --config training_config.yaml \
                    --timesteps "$TIMESTEPS" \
                    --project "$PROJECT" \
                    --load-model "/app/pretrained/$(basename "$MODEL_PATH")"
            ;;
    esac
}

# ===========================================================================
# MAIN
# ===========================================================================

header
check_wandb_key
select_project
select_mode
select_platform
select_parallelism
select_timesteps
select_model

# Összefoglaló
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BOLD}Összefoglaló:${NC}"
echo -e "    Projekt:    ${GREEN}$PROJECT${NC}"
mode_names=("" "Headless" "GUI" "Sweep" "Transfer Learning")
echo -e "    Mód:        ${GREEN}${mode_names[$MODE]}${NC}"
platform_names=("" "Lokális" "Docker")
echo -e "    Platform:   ${GREEN}${platform_names[$PLATFORM]}${NC}"
echo -e "    Timesteps:  ${GREEN}$TIMESTEPS${NC}"
if [ "$PARALLEL" -gt 1 ]; then echo -e "    Párhuzamos: ${GREEN}$PARALLEL futás${NC}"; fi
if [ -n "$MODEL_PATH" ]; then echo -e "    Modell:     ${GREEN}$MODEL_PATH${NC}"; fi
echo -e ""
echo -e "  ${DIM}Szimuláció: minden epizódban random hossz (15min–2h)${NC}"
echo -e "  ${DIM}Forgalom:   target+spread (epizódonként random nehézség és aszimmetria)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "  Indítás? [Y/n]: " confirm
if [ "${confirm:-Y}" == "n" ]; then
    echo "Megszakítva."
    exit 0
fi

if [ "$PLATFORM" == "1" ]; then
    run_local
else
    run_docker
fi

echo ""
echo -e "${GREEN}[✓] Kész!${NC}"
