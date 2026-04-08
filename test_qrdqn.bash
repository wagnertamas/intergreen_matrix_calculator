#!/bin/bash
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate base
source .venv/bin/activate
echo "Running QRDQN check..."
python main_headless.py --algorithm qrdqn --reward-mode speed_throughput --junction R1C1_C --timesteps 500
