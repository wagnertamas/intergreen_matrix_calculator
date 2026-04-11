@echo off
REM =============================================================================
REM Egyszeri futtatas GUI-val Windows rendszerre (Vizualis ellenorzeshez)
REM =============================================================================
REM Hasznalat: Dupla kattintas a run_single_gui.bat fajlra, vagy futtatas CMD-bol.
REM Modosithatod a fajlban az ALGO, REWARD vagy JUNCTION valtozokat.
REM =============================================================================

REM Kenyszeritjuk a traci-t (process-level isolation), mert a GUI mod csak igy megy
set USE_LIBSUMO=0

REM --- Alapertelemezesek (itt atirhatod, ha mast akarsz tesztelni) ---
set JUNCTION=R1C1_C
set ALGO=qrdqn
set REWARD=wait_haltratio
rem  Lehet: speed_throughput, halt_ratio, co2_speedstd, wait_triplet_tpstdhalt, wait_haltratio
set TIMESTEPS=50000

echo ============================================================
echo [INFO] Szimulacio inditasa GUI modban...
echo [INFO] Keresztezodes: %JUNCTION%
echo [INFO] Algoritmus:    %ALGO%
echo [INFO] Reward mod:    %REWARD%
echo ============================================================

REM Futtatas
REM Megj: Ha a Windows gepen is virtualis kornyezetet hasznalsz (venv),
REM erdemes elotte aktivalni (pl. "call .venv\Scripts\activate"). 
REM Ha conda-t hasznalsz, akkor elotte inditsd el onnan a scriptet.

python main_headless.py ^
    --junction "%JUNCTION%" ^
    --algorithm "%ALGO%" ^
    --reward-mode "%REWARD%" ^
    --timesteps "%TIMESTEPS%" ^
    --num-layers 2 ^
    --layer-size 64 ^
    --gui

echo [INFO] Futas befejezodott.
pause
