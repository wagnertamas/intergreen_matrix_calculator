import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os

# Importáljuk a meglévő és új modulokat
from sumo_parser import SumoInternalParser
from gui import JunctionApp  # A meglévő mátrix kalkulátor
from detector_editor import DetectorEditor  # Ezt hozzuk létre lejjebb

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
        
        # Gomb: Tanítás (Későbbi fejlesztés)
        self.btn_train = ttk.Button(tools_frame, text="RL Ágens Tanítás Indítása", state="disabled")
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
            except Exception as e:
                messagebox.showerror("Hiba", f"Nem sikerült betölteni a hálózatot:\n{e}")

    def open_matrix_tool(self):
        if not self.parser: return
        # Új ablakban nyitjuk meg a meglévő GUI-t
        new_window = tk.Toplevel(self.root)
        app = JunctionApp(new_window, self.parser)
        # Figyelem: A JunctionApp-ban lévő root.mainloop() hívást ki kell venni vagy kezelni, 
        # mert a főablak már futtat egy mainloop-ot.
        # A gui.py-t kicsit módosítani kell (lásd lejjebb).

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