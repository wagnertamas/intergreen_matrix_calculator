#!/bin/bash
# SUMO telepítése
echo "SUMO telepítése..."
sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc -y

# Python csomagok
echo "Python libek telepítése..."
pip install gymnasium stable-baselines3 wandb shimmy libsumo

# Futtatás
echo "Tanítás indítása..."
python main_headless.py --config training_config.json
