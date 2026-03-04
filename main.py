# FIX: Import numpy BEFORE tkinter to prevent macOS crash (binary incompatibility)
import numpy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os

# Importáljuk a meglévő és új modulokat
from sumo_parser import SumoInternalParser
from gui import JunctionApp  # A meglévő mátrix kalkulátor
from detector_editor import DetectorEditor  # Ezt hozzuk létre lejjebb
from transfer_learning import TransferLearningDialog # [NEW] Transfer Learning

class MainLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("SUMO RL Traffic Control Suite")
        self.root.geometry("600x400")
        
        # Stílus
        style = ttk.Style()
        style.theme_use('clam')
        
        # Fejléc
        header = tk.Label(root, text="Traffic Control RL Workflow", font=("Helvetica", 16, "bold"))
        header.pack(pady=20)

        # Fájl kiválasztás állapota
        self.network_file = None
        self.parser = None

        # Fájl választó keret
        file_frame = tk.LabelFrame(root, text="1. Hálózat Betöltése", padx=10, pady=10)
        file_frame.pack(fill="x", padx=20, pady=5)
        
        self.lbl_file = tk.Label(file_frame, text="Nincs fájl kiválasztva", fg="red")
        self.lbl_file.pack(side="left", padx=5)
        
        btn_load = ttk.Button(file_frame, text="Tallózás (.net.xml)", command=self.load_network)
        btn_load.pack(side="right")

        # Eszközök keret
        tools_frame = tk.LabelFrame(root, text="2. Eszközök", padx=10, pady=10)
        tools_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Gomb: Intergreen Matrix Calculator
        self.btn_matrix = ttk.Button(tools_frame, text="Intergreen Mátrix & Fázisok", command=self.open_matrix_tool, state="disabled")
        self.btn_matrix.pack(fill="x", pady=5)

        # Gomb: Detektor Elhelyező (ÚJ)
        self.btn_detectors = ttk.Button(tools_frame, text="Detektorok Elhelyezése (RL Input)", command=self.open_detector_tool, state="disabled")
        self.btn_detectors.pack(fill="x", pady=5)

        # Gomb: Transfer Learning (ÚJ)
        self.btn_transfer = ttk.Button(tools_frame, text="Transfer Learning (Fine-Tuning)", command=self.open_transfer_tool, state="disabled")
        self.btn_transfer.pack(fill="x", pady=5)
        
        # Gomb: Tanítás (Későbbi fejlesztés)
        self.btn_train = ttk.Button(tools_frame, text="RL Ágens Tanítás Indítása", command=self.open_training_tool, state="disabled")
        self.btn_train.pack(fill="x", pady=5)

    def load_network(self):
        file_path = filedialog.askopenfilename(
            title="Select SUMO Network File",
            filetypes=[("SUMO Network Files", "*.xml"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.parser = SumoInternalParser(file_path)
                self.network_file = file_path
                self.lbl_file.config(text=os.path.basename(file_path), fg="green")
                
                # Gombok aktiválása
                self.btn_matrix.config(state="normal")
                self.btn_detectors.config(state="normal")
                self.btn_train.config(state="normal")
                self.btn_transfer.config(state="normal")
            except Exception as e:
                messagebox.showerror("Hiba", f"Nem sikerült betölteni a hálózatot:\n{e}")

    def open_matrix_tool(self):
        if not self.parser: return
        # Új ablakban nyitjuk meg a meglévő GUI-t
        new_window = tk.Toplevel(self.root)
        app = JunctionApp(new_window, self.parser)

    def open_training_tool(self):
        """Opens the RL Training Dialog."""
        if not self.network_file: return
        
        # Feltételezzük, hogy a .json és a detektor fájl ugyanott van, mint a hálózat,
        # vagy a korábbi logikából következtetjük ki.
        # A gui.py szerint: settings_file = parser.file_path + ".settings.json"
        
        # Itt egyszerűsítünk: keressük a 'traffic_lights.json' és 'detectors.add.xml' fájlokat
        # az aktuális munkakönyvtárban, vagy a hálózat mellett.
        
        base_dir = os.path.dirname(self.network_file)
        logic_file = os.path.join(base_dir, "traffic_lights.json")
        detector_file = os.path.join(base_dir, "detectors.add.xml")
        
        if not os.path.exists(logic_file):
            # Fallback 1: Current Dir
            logic_file = "traffic_lights.json"
            if not os.path.exists(logic_file):
                # Fallback 2: data/ Dir
                logic_file = os.path.join("data", "traffic_lights.json")
            
        if not os.path.exists(detector_file):
            # Fallback 1: Current Dir
            detector_file = "detectors.add.xml"
            if not os.path.exists(detector_file):
                # Fallback 2: data/ Dir
                detector_file = os.path.join("data", "detectors.add.xml")
            
        if not os.path.exists(logic_file) or not os.path.exists(detector_file):
            messagebox.showerror("Hiba", f"Nem találhatók a konfigurációs fájlok:\n{logic_file}\n{detector_file}")
            return

        from rl_trainer import TrainingDialog
        TrainingDialog(self.root, self.network_file, logic_file, detector_file)

    def open_transfer_tool(self):
        """Opens the Transfer Learning Dialog."""
        if not self.network_file: return
        
        # Reuse logic to find config files
        base_dir = os.path.dirname(self.network_file)
        # Note: logic_file and detector_file paths logic is duplicated here, 
        # could be refactored into a helper if we wanted to be cleaner.
        logic_file = os.path.join(base_dir, "traffic_lights.json")
        detector_file = os.path.join(base_dir, "detectors.add.xml")
        
        # Fallback logic same as above
        if not os.path.exists(logic_file):
            logic_file = "traffic_lights.json"
            if not os.path.exists(logic_file):
                 logic_file = os.path.join("data", "traffic_lights.json")
                 
        if not os.path.exists(detector_file):
            detector_file = "detectors.add.xml"
            if not os.path.exists(detector_file):
                detector_file = os.path.join("data", "detectors.add.xml")

        if not os.path.exists(logic_file) or not os.path.exists(detector_file):
             messagebox.showerror("Hiba", f"Nem találhatók a konfigurációs fájlok (Exportálj először!):\n{logic_file}\n{detector_file}")
             return

        # [NEW] Validate Logic File IDs against Network
        try:
            import json
            with open(logic_file, 'r') as f:
                logic_data = json.load(f)
            
            logic_ids = set(logic_data.keys())
            network_ids = set(j['id'] for j in self.parser.junctions)
            
            # Check if there is ANY overlap
            common = logic_ids.intersection(network_ids)
            if not common:
                 messagebox.showerror("Konfigurációs Hiba", 
                                      f"A 'traffic_lights.json' fájlban lévő csomópontok ({list(logic_ids)[:3]}...) "
                                      f"NEM találhatóak a betöltött hálózatban!\n\n"
                                      "Valószínűleg egy korábbi hálózathoz tartozik.\n"
                                      "Kérlek használd az 'Intergreen Mátrix' eszközt az új fájlok generálásához!")
                 return
                 
        except Exception as e:
            print(f"Validation error: {e}")
            # Non-blocking warning
            pass

        TransferLearningDialog(self.root, self.network_file, logic_file, detector_file)

    def open_detector_tool(self):
        if not self.parser: return
        new_window = tk.Toplevel(self.root)
        app = DetectorEditor(new_window, self.parser)

def main():
    root = tk.Tk()
    app = MainLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()