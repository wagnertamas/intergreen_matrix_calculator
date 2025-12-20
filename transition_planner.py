import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

class TransitionPlannerDialog:
    """
    A dialog window for planning phase transitions between two traffic light phases.
    
    This class provides a visual interface to:
    1. Select a starting phase (Phase A) and an ending phase (Phase B).
    2. Visualize the transition timeline including clearing and entering traffic.
    3. Calculate and display the required intergreen times.
    """
    def __init__(self, parent, all_ids, phases, matrix):
        """
        Initialize the transition planner dialog.
        
        Args:
            parent (tk.Widget): The parent widget (usually the main window).
            all_ids (list): List of all lane IDs.
            phases (list): List of available phases (cliques).
            matrix (list): The intergreen matrix.
        """
        self.top = tk.Toplevel(parent)
        self.top.title("Fázisátmenet Tervező")
        self.top.geometry("1100x800")

        self.all_ids = all_ids
        self.phases = phases
        self.matrix = matrix

        ctrl_frame = tk.Frame(self.top, bg="#eee", pady=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(ctrl_frame, text="Honnan (A Fázis):", bg="#eee").pack(side=tk.LEFT, padx=5)
        self.cb_from = ttk.Combobox(ctrl_frame,
                                    values=[f"Fázis {i + 1}" for i in range(len(phases))],
                                    state="readonly")
        self.cb_from.pack(side=tk.LEFT, padx=5)
        self.cb_from.current(0)

        tk.Label(ctrl_frame, text="Hova (B Fázis):", bg="#eee").pack(side=tk.LEFT, padx=5)
        self.cb_to = ttk.Combobox(ctrl_frame, values=[f"Fázis {i + 1}" for i in range(len(phases))],
                                  state="readonly")
        self.cb_to.pack(side=tk.LEFT, padx=5)
        if len(phases) > 1:
            self.cb_to.current(1)
        else:
            self.cb_to.current(0)

        tk.Button(ctrl_frame, text="Terv Generálása", command=self.generate_plan, bg="#ddd").pack(
            side=tk.LEFT, padx=15)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.generate_plan()

    def generate_plan(self):
        """
        Generates and visualizes the transition plan based on selected phases.
        
        CORRECTION:
        - The intergreen time (K) starts at the end of the Green A phase (Start of Yellow).
        - t=0 is the start of Yellow.
        - Green B starts exactly at t=K.
        """
        idx_a = self.cb_from.current()
        idx_b = self.cb_to.current()
        if idx_a == -1 or idx_b == -1: return

        phase_a_indices = set(self.phases[idx_a])
        phase_b_indices = set(self.phases[idx_b])
        n = len(self.all_ids)

        AMBER_TIME = 3.0
        RED_AMBER_TIME = 2.0

        # Állapotok meghatározása
        lane_states = {}
        clearing_indices = []
        entering_indices = []

        for i in range(n):
            in_a = i in phase_a_indices
            in_b = i in phase_b_indices
            if in_a and in_b:
                lane_states[i] = 'stay_green'
            elif in_a and not in_b:
                lane_states[i] = 'clearing'
                clearing_indices.append(i)
            elif not in_a and in_b:
                lane_states[i] = 'entering'
                entering_indices.append(i)
            else:
                lane_states[i] = 'stay_red'

        # Időzítés számítása
        green_start_times = {}
        dependencies = []

        for j in entering_indices:
            max_intergreen = 0.0
            crit_i = None
            for i in clearing_indices:
                # Mátrix olvasás [stop][start] vagy [start][stop] 
                # (Ellenőrizd, hogy a gui.py-ban melyiket használod végül, itt feltételezzük a helyes irányt)
                k_val = 0
                
                # A biztonság kedvéért itt olvassuk ki mindkét irányt, és a nagyobbat vesszük,
                # vagy használjuk a gui.py-ban beállított logikát.
                # Itt most feltételezzük, hogy a mátrix[i][j] a helyes clearing->entering érték.
                if self.matrix[i][j] is not None:
                    k_val = self.matrix[i][j][0]

                if k_val > max_intergreen:
                    max_intergreen = k_val
                    crit_i = i

            # JAVÍTÁS: Nem adjuk hozzá az AMBER_TIME-ot!
            # A K érték (max_intergreen) már tartalmazza a sárgát is definíció szerint
            # (End Green A -> Start Green B).
            # Tehát a Zöld B pontosan K másodpercnél kezdődik (ahol 0 a sárga kezdete).
            green_start_times[j] = max(max_intergreen, 0.0)
            
            if crit_i is not None:
                dependencies.append((crit_i, j, max_intergreen))

        # Rajzolás
        self.ax.clear()

        # Időtengely határok (dinamikus)
        max_time = 10.0
        if green_start_times:
            max_time = max(max(green_start_times.values()) + 5, 10.0)

        for i in range(n):
            state = lane_states[i]
            y = i

            # Színek
            C_RED = '#d9534f'
            C_YEL = '#f0ad4e'
            C_GRE = '#5cb85c'
            C_ORA = '#e67e22'

            if state == 'stay_green':
                # Végig zöld
                self.ax.barh(y, max_time + 5, left=-5, height=0.6, color=C_GRE, alpha=0.9)
                self.ax.text(-0.5, y, f"{self.all_ids[i]}", va='center', ha='right', fontsize=9, fontweight='bold')

            elif state == 'stay_red':
                # Végig piros
                self.ax.barh(y, max_time + 5, left=-5, height=0.6, color=C_RED, alpha=0.4)
                self.ax.text(-0.5, y, f"{self.all_ids[i]}", va='center', ha='right', fontsize=9, fontweight='bold')

            elif state == 'clearing':
                # Zöld -> Sárga -> Piros
                # Sárga kezdődik 0-nál, tart AMBER_TIME-ig
                self.ax.barh(y, 5, left=-5, height=0.6, color=C_GRE, alpha=0.9) # Előző zöld
                self.ax.barh(y, AMBER_TIME, left=0, height=0.6, color=C_YEL, edgecolor='#d68e00') # Sárga
                self.ax.barh(y, max_time - AMBER_TIME, left=AMBER_TIME, height=0.6, color=C_RED, alpha=0.4) # Piros
                
                self.ax.text(-0.5, y, f"{self.all_ids[i]} (Ki)", va='center', ha='right', fontsize=9, fontweight='bold')

            elif state == 'entering':
                # Piros -> Piros-Sárga -> Zöld
                g_start = green_start_times[i]
                ra_start = g_start - RED_AMBER_TIME

                # Piros a kezdetektől a Piros-Sárga kezdetéig
                self.ax.barh(y, ra_start + 5, left=-5, height=0.6, color=C_RED, alpha=0.4)
                
                # Piros-Sárga
                self.ax.barh(y, RED_AMBER_TIME, left=ra_start, height=0.6, color=C_ORA, edgecolor='#c05c00')
                
                # Zöld
                self.ax.barh(y, max_time - g_start, left=g_start, height=0.6, color=C_GRE, alpha=0.9)

                # Címke
                self.ax.text(g_start + 0.5, y, f"Start: {g_start:.1f}s", va='center', ha='left',
                             fontsize=8, fontweight='bold', color='black')
                self.ax.text(-0.5, y, f"{self.all_ids[i]} (Be)", va='center', ha='right',
                             fontsize=9, fontweight='bold')

        # Nyilak és K értékek kirajzolása
        for dep in dependencies:
            i, j, k_val = dep
            
            # JAVÍTÁS: A nyíl 0-tól (Sárga kezdete) K-ig tart (Zöld kezdete)
            self.ax.annotate("", 
                             xy=(k_val, j),       # Nyíl vége (Entering Zöld kezdete)
                             xytext=(0, i),       # Nyíl kezdete (Clearing Sárga kezdete)
                             arrowprops=dict(arrowstyle="->", color='black', lw=2.0, ls='-'))
            
            mid_y = (i + j) / 2
            # Szöveg pozíciója a nyíl felénél
            self.ax.text(k_val / 2, mid_y, f"K={k_val:.0f}", color='black', fontsize=9,
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='black', alpha=1.0))

        self.ax.set_yticks(range(n))
        self.ax.set_yticklabels([f"{i + 1}." for i in range(n)], fontsize=9, fontweight='bold')
        self.ax.set_xlabel("Idő [s] (0 = 'A' Fázis vége / Sárga indul)", fontsize=10)

        self.ax.set_xlim(-6, max_time) # Kicsit igazítottam a nézeten
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self.ax.axvline(0, color='black', lw=1.5, linestyle='-') # Jelöljük a 0 pontot erősebben

        patches = [
            mpatches.Patch(color='#5cb85c', label='Zöld'),
            mpatches.Patch(color='#f0ad4e', label=f'Sárga ({AMBER_TIME}s)'),
            mpatches.Patch(color='#e67e22', label=f'Piros-Sárga ({RED_AMBER_TIME}s)'),
            mpatches.Patch(color='#d9534f', label='Piros'),
        ]
        self.ax.legend(handles=patches, loc='upper right', fontsize=9)
        self.canvas.draw()

