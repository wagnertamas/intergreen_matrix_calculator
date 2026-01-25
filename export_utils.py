import os
import json
import zipfile
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

def export_to_colab_package(settings_dict, required_files):
    """
    Összeállít egy ZIP csomagot a Colab/Headless futtatáshoz.
    
    Args:
        settings_dict (dict): A GUI-ból kinyert beállítások.
        required_files (list): A szükséges fájlok útvonalai (net, rou, det, json).
    """
    # 1. Célfájl kiválasztása
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_name = f"sumo_rl_colab_{timestamp}.zip"
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".zip",
        filetypes=[("ZIP files", "*.zip")],
        initialfile=default_name,
        title="Mentés Colab Csomagként"
    )
    
    if not save_path:
        return

    # 2. Config fájl létrehozása (training_config.json)
    config_data = {
        "hyperparams": settings_dict,
        "files": {
            # A zip gyökerében lesznek a fájlok, így csak a fájlnevet mentjük
            "net": os.path.basename(required_files['net']),
            "logic": os.path.basename(required_files['logic']),
            "detector": os.path.basename(required_files['detector']),
            "route": "random_traffic.rou.xml" # Ez generálódik, de a hivatkozás kell
        }
    }

    try:
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 3. Python kódok hozzáadása
            py_files = [
                "sumo_rl_environment.py", 
                "rl_trainer.py", 
                "main_headless.py", # Ezt mindjárt megírjuk
            ]
            
            for py in py_files:
                if os.path.exists(py):
                    zipf.write(py, arcname=py)
                else:
                    print(f"Figyelem: {py} nem található, kihagyva.")

            # 4. SUMO/Config fájlok hozzáadása
            # A hálózati fájlok (net, json, xml)
            for file_key, file_path in required_files.items():
                if os.path.exists(file_path):
                    zipf.write(file_path, arcname=os.path.basename(file_path))
            
            # 5. Config JSON hozzáadása
            zipf.writestr("training_config.json", json.dumps(config_data, indent=4))
            
            # 6. Colab telepítő script hozzáadása (opcionális, de hasznos)
            install_script = """#!/bin/bash
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
"""
            zipf.writestr("install_and_run.sh", install_script)

        messagebox.showinfo("Siker", f"Csomag sikeresen exportálva:\n{save_path}\n\nTöltsd fel ezt a fájlt Colab-ra!")
        
    except Exception as e:
        messagebox.showerror("Hiba", f"Hiba az exportálás során:\n{e}")