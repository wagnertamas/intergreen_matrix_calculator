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
        
        Calculates:
        - Which lanes are staying green, clearing, entering, or staying red.
        - The start time for entering lanes based on the maximum intergreen time required by clearing lanes.
        - Draws a Gantt-chart-like visualization of the signal states.
        """
        idx_a = self.cb_from.current()
        idx_b = self.cb_to.current()
        if idx_a == -1 or idx_b == -1: return

        phase_a_indices = set(self.phases[idx_a])
        phase_b_indices = set(self.phases[idx_b])
        n = len(self.all_ids)

        AMBER_TIME = 3.0
        RED_AMBER_TIME = 2.0

        # Állapotok
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

        # Időzítés
        green_start_times = {}
        dependencies = []

        for j in entering_indices:
            max_intergreen = 0.0
            crit_i = None
            for i in clearing_indices:
                k_val = 0
                if self.matrix[i][j] is not None:
                    k_val = self.matrix[i][j][0]

                if k_val > max_intergreen:
                    max_intergreen = k_val
                    crit_i = i

            green_start_times[j] = AMBER_TIME + max_intergreen
            if crit_i is not None:
                dependencies.append((crit_i, j, max_intergreen))

        # Rajzolás
        self.ax.clear()

        for i in range(n):
            state = lane_states[i]
            y = i

            # Színek: Bootstrap/Traffic szabványosabb
            C_RED = '#d9534f'
            C_YEL = '#f0ad4e'
            C_GRE = '#5cb85c'
            C_ORA = '#e67e22'  # Piros-Sárga

            if state == 'stay_green':
                self.ax.barh(y, 25, left=-5, height=0.6, color=C_GRE, alpha=0.9)
                self.ax.text(-0.5, y, f"{self.all_ids[i]}", va='center', ha='right', fontsize=9,
                             fontweight='bold')

            elif state == 'stay_red':
                self.ax.barh(y, 25, left=-5, height=0.6, color=C_RED, alpha=0.4)
                self.ax.text(-0.5, y, f"{self.all_ids[i]}", va='center', ha='right', fontsize=9,
                             fontweight='bold')

            elif state == 'clearing':
                self.ax.barh(y, 5, left=-5, height=0.6, color=C_GRE, alpha=0.9)
                self.ax.barh(y, AMBER_TIME, left=0, height=0.6, color=C_YEL, edgecolor='#d68e00')
                self.ax.barh(y, 22 - AMBER_TIME, left=AMBER_TIME, height=0.6, color=C_RED,
                             alpha=0.4)
                self.ax.text(-0.5, y, f"{self.all_ids[i]} (Ki)", va='center', ha='right',
                             fontsize=9, fontweight='bold')

            elif state == 'entering':
                g_start = green_start_times[i]
                ra_start = g_start - RED_AMBER_TIME

                self.ax.barh(y, ra_start + 5, left=-5, height=0.6, color=C_RED, alpha=0.4)
                self.ax.barh(y, RED_AMBER_TIME, left=ra_start, height=0.6, color=C_ORA,
                             edgecolor='#c05c00')
                self.ax.barh(y, 20 - g_start, left=g_start, height=0.6, color=C_GRE, alpha=0.9)

                self.ax.text(g_start + 0.5, y, f"Start: {g_start:.1f}s", va='center', ha='left',
                             fontsize=8, fontweight='bold', color='black')
                self.ax.text(-0.5, y, f"{self.all_ids[i]} (Be)", va='center', ha='right',
                             fontsize=9, fontweight='bold')

        # Nyilak javítása
        for dep in dependencies:
            i, j, k_val = dep
            # A nyíl a sárga végétől a zöld elejéig tart
            self.ax.annotate("", xy=(AMBER_TIME + k_val, j), xytext=(AMBER_TIME, i),
                             arrowprops=dict(arrowstyle="->", color='black', lw=2.0,
                                             ls='-'))  # Vastagabb vonal
            mid_y = (i + j) / 2
            self.ax.text(AMBER_TIME + k_val / 2, mid_y, f"K={k_val:.0f}", color='black', fontsize=9,
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='black', alpha=1.0))

        self.ax.set_yticks(range(n))
        self.ax.set_yticklabels([f"{i + 1}." for i in range(n)], fontsize=9, fontweight='bold')
        self.ax.set_xlabel("Idő [s] (0 = Váltás kezdete / Sárga indul)", fontsize=10)

        # BAL OLDALI MARGÓ MEGNÖVELÉSE, hogy a szöveg kiférjen
        self.ax.set_xlim(-12, 20)

        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self.ax.axvline(0, color='black', lw=1.5)

        patches = [
            mpatches.Patch(color='#5cb85c', label='Zöld'),
            mpatches.Patch(color='#f0ad4e', label=f'Sárga ({AMBER_TIME}s)'),
            mpatches.Patch(color='#e67e22', label=f'Piros-Sárga ({RED_AMBER_TIME}s)'),
            mpatches.Patch(color='#d9534f', label='Piros'),
        ]
        self.ax.legend(handles=patches, loc='upper right', fontsize=9)
        self.canvas.draw()
