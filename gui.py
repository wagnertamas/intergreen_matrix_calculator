import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.lines as mlines
from matplotlib.widgets import Cursor, CheckButtons
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import csv
import json
import os
import networkx as nx
import xml.etree.ElementTree as ET

from geometry_utils import calculate_junction_data
from transition_planner import TransitionPlannerDialog

class JunctionApp:
    """
    The main GUI application for the Intergreen Matrix Calculator.
    
    This class handles:
    - Displaying the junction geometry and conflict points.
    - Managing user interactions (zooming, panning, selecting conflicts).
    - Calculating and displaying the intergreen matrix and conflict matrix.
    - Exporting results to CSV, PDF, and SUMO formats.
    """
    def __init__(self, root, parser):
        """
        Initialize the application.
        
        Args:
            root (tk.Tk): The root Tkinter window.
            parser (SumoInternalParser): The parser instance with loaded network data.
        """
        self.root = root
        self.root.title("SUMO Intersection Analyzer")
        self.root.geometry("1600x950")
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.parser = parser
        self.junctions = parser.junctions
        self.current_idx = 0

        self.settings_file = self.parser.file_path + ".settings.json"
        self.saved_settings = {}
        self.load_settings()

        main_container = tk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_container)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(top_frame, text="<<", command=self.prev_junction, width=3).pack(side=tk.LEFT,
                                                                                   padx=2)
        ttk.Button(top_frame, text=">>", command=self.next_junction, width=3).pack(side=tk.LEFT,
                                                                                   padx=2)

        self.combo_values = [f"{i + 1}. {j['id']}" for i, j in enumerate(self.junctions)]
        self.combo_var = tk.StringVar()
        self.combo = ttk.Combobox(top_frame, textvariable=self.combo_var, values=self.combo_values,
                                  state="readonly", width=20)
        self.combo.pack(side=tk.LEFT, padx=10)
        self.combo.bind("<<ComboboxSelected>>", self.jump_to_junction)
        if self.combo_values: self.combo.current(0)

        ttk.Button(top_frame, text="Zoom +", command=self.zoom_in, width=6).pack(side=tk.LEFT,
                                                                                 padx=2)
        ttk.Button(top_frame, text="Zoom -", command=self.zoom_out, width=6).pack(side=tk.LEFT,
                                                                                  padx=2)
        ttk.Button(top_frame, text="Reset", command=self.reset_view, width=5).pack(side=tk.LEFT,
                                                                                   padx=2)
        ttk.Button(top_frame, text="Címkék Ki/Be", command=self.toggle_annotations).pack(
            side=tk.LEFT, padx=10)


        # ÚJ GOMB
        ttk.Button(top_frame, text="Fázisátmenet Terv", command=self.open_transition_planner).pack(
            side=tk.RIGHT, padx=10)

        ttk.Button(top_frame, text="Mentés", command=self.save_settings).pack(side=tk.RIGHT,
                                                                              padx=10)
        ttk.Button(top_frame, text="PDF Export", command=self.export_pdf_dialog).pack(side=tk.RIGHT,
                                                                                      padx=2)
        ttk.Button(top_frame, text="CSV Export", command=self.export_csv).pack(side=tk.RIGHT,
                                                                               padx=2)
        ttk.Button(top_frame, text="SUMO Export (Összes)", command=self.export_all_to_sumo).pack(
            side=tk.RIGHT, padx=10)

        content_frame = tk.Frame(main_container)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_pane = tk.Frame(content_frame)
        self.left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('#f0f0f0')

        self.ax = self.fig.add_axes([0.02, 0.05, 0.63, 0.90])

        self.ax_check = self.fig.add_axes([0.68, 0.70, 0.28, 0.25])
        self.ax_check.set_axis_off()

        self.ax_combined = self.fig.add_axes([0.68, 0.40, 0.28, 0.25])
        self.ax_combined.set_axis_off()

        self.ax_inhibit = self.fig.add_axes([0.68, 0.05, 0.28, 0.25])
        self.ax_inhibit.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_pane)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.right_pane = tk.Frame(content_frame, width=400, bg="#e1e1e1")
        self.right_pane.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)

        self.right_pane.bind('<Enter>', self._bound_to_mousewheel)
        self.right_pane.bind('<Leave>', self._unbound_to_mousewheel)

        tk.Label(self.right_pane, text="Lehetséges Fázisok", font=("Arial", 11, "bold"),
                 bg="#e1e1e1").pack(pady=5)

        self.scroll_canvas = tk.Canvas(self.right_pane, bg="#e1e1e1", bd=0, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.right_pane, orient="vertical",
                                      command=self.scroll_canvas.yview)
        self.phase_frame_inner = tk.Frame(self.scroll_canvas, bg="#e1e1e1")

        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.create_window((0, 0), window=self.phase_frame_inner, anchor="nw")

        self.phase_frame_inner.bind("<Configure>", lambda e: self.scroll_canvas.configure(
            scrollregion=self.scroll_canvas.bbox("all")))

        self.cursor = Cursor(self.ax, useblit=True, color='gray', linewidth=1, linestyle='--')
        self.interactive_items = []
        self.lines_map = {}
        self.check_buttons = None
        self.all_annotations = []
        self.annotations_visible = True
        self.highlighted_ids = None

        self.pan_start = None
        self.pan_ax_lims = None

        self.combined_table = None
        self.inhibit_table = None

        self.matrix_mask = None
        self.conflict_matrix = None

        self.ids_for_table = []
        self.current_limits = None
        self.current_data_cache = None
        self.phases_data = []  # Tároljuk a generált fázisokat

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.plot_current()

    # --- ÚJ FUNKCIÓ: ABLAK NYITÁSA ---
    def open_transition_planner(self):
        """Opens the transition planner dialog."""
        if not self.current_data_cache or not self.phases_data:
            messagebox.showinfo("Info", "Nincsenek elérhető fázisok vagy adat.")
            return

        # Színeket már nem kell átadni, mert fix traffic light színeket használunk
        TransitionPlannerDialog(self.root, self.ids_for_table, self.phases_data,
                                self.current_data_cache['matrix'])

    # --- EGÉRGÖRGŐ KEZELÉS ---

    def _bound_to_mousewheel(self, event):
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.scroll_canvas.unbind_all("<MouseWheel>")
        self.scroll_canvas.unbind_all("<Button-4>")
        self.scroll_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.scroll_canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.scroll_canvas.yview_scroll(-1, "units")

    # --- PERZISZTENCIA ---

    def load_settings(self):
        """Loads saved settings (matrix masks, conflicts) from a JSON file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.saved_settings = json.load(f)
            except Exception as e:
                print(f"Hiba a beállítások betöltésekor: {e}")
                self.saved_settings = {}

    def save_settings(self):
        """Saves current settings to a JSON file."""
        if self.current_data_cache and self.matrix_mask is not None and self.conflict_matrix is not None:
            jid = self.current_data_cache['jdata']['id']
            self.saved_settings[jid] = {
                'matrix_mask': self.matrix_mask.tolist(),
                'conflict_matrix': self.conflict_matrix.tolist()
            }
            try:
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(self.saved_settings, f, indent=4)
                messagebox.showinfo("Mentés", "Beállítások (Mátrixok) elmentve.")
            except Exception as e:
                messagebox.showerror("Hiba", str(e))

    # --- NAVIGÁCIÓ & VIEW ---

    def next_junction(self):
        if not self.junctions: return
        self.current_idx = (self.current_idx + 1) % len(self.junctions)
        self.combo.current(self.current_idx)
        self.reset_matrices()
        self.plot_current()

    def prev_junction(self):
        if not self.junctions: return
        self.current_idx = (self.current_idx - 1) % len(self.junctions)
        self.combo.current(self.current_idx)
        self.reset_matrices()
        self.plot_current()

    def jump_to_junction(self, event):
        idx = self.combo.current()
        if idx != -1:
            self.current_idx = idx
            self.reset_matrices()
            self.plot_current()

    def reset_matrices(self):
        self.matrix_mask = None
        self.conflict_matrix = None
        self.current_limits = None

    def zoom_in(self):
        self._apply_zoom(0.8)

    def zoom_out(self):
        self._apply_zoom(1.25)

    def reset_view(self):
        self.current_limits = None; self.plot_current()

    def _apply_zoom(self, factor):
        xlim = self.ax.get_xlim();
        ylim = self.ax.get_ylim()
        xc = (xlim[0] + xlim[1]) / 2;
        yc = (ylim[0] + ylim[1]) / 2
        nw = (xlim[1] - xlim[0]) * factor;
        nh = (ylim[1] - ylim[0]) * factor
        self.ax.set_xlim(xc - nw / 2, xc + nw / 2);
        self.ax.set_ylim(yc - nh / 2, yc + nh / 2)
        self.current_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw_idle()

    def toggle_annotations(self):
        self.annotations_visible = not self.annotations_visible
        for artist in self.all_annotations:
            artist.set_visible(self.annotations_visible)
        self.canvas.draw_idle()

    def toggle_visibility(self, label):
        if label in self.lines_map and self.lines_map[label]:
            first_artist = self.lines_map[label][0]
            new_state = not first_artist.get_visible()
            for artist in self.lines_map[label]:
                if isinstance(artist, plt.Annotation) or isinstance(artist, plt.Text):
                    if self.annotations_visible:
                        artist.set_visible(new_state)
                    else:
                        artist.set_visible(False)
                else:
                    artist.set_visible(new_state)
            self.canvas.draw_idle()

    # --- EXPORT ---

    def export_csv(self):
        """Exports the current intergreen matrix to a CSV file."""
        if not self.current_data_cache: return
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV file", "*.csv")])
        if not filename: return
        lines = self.current_data_cache['lines'];
        matrix = self.current_data_cache['matrix']
        ids = [l['id'] for l in lines];
        n = len(ids)
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["From/To"] + ids)
                for i in range(n):
                    row = [ids[i]]
                    for j in range(n):
                        val = ""
                        if i != j and self.matrix_mask[i, j] and matrix[i][j]:
                            val = f"{matrix[i][j][0]:.2f}"
                        row.append(val)
                    writer.writerow(row)
            messagebox.showinfo("Info", "Mentve.")
        except Exception as e:
            messagebox.showerror("Hiba", str(e))

    def export_pdf_dialog(self):
        ans = messagebox.askyesno("PDF Export",
                                  "Minden csomópontot exportáljak? (Nem = csak a jelenlegi)")
        if ans:
            indices = range(len(self.junctions))
        else:
            indices = [self.current_idx]
        self.export_pdf_process(indices)

    def export_pdf_process(self, selected_indices):
        """Generates a PDF report for the selected junctions."""
        filename = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF file", "*.pdf")])
        if not filename: return
        try:
            with PdfPages(filename) as pdf:
                total = len(selected_indices)
                for k, idx in enumerate(selected_indices):
                    jdata = self.junctions[idx]
                    data = calculate_junction_data(jdata)
                    jid = jdata['id']

                    n = len(data['lines'])
                    current_mask = np.ones((n, n), dtype=bool)
                    current_conflict = None

                    if jid in self.saved_settings:
                        s = self.saved_settings[jid]
                        if 'matrix_mask' in s:
                            m = np.array(s['matrix_mask'])
                            if m.shape == (n, n): current_mask = m
                        if 'conflict_matrix' in s:
                            c = np.array(s['conflict_matrix'])
                            if c.shape == (n, n): current_conflict = c

                    if current_conflict is None:
                        current_conflict = np.zeros((n, n), dtype=bool)
                        for i in range(n):
                            for j in range(n):
                                if i != j:
                                    if data['matrix'][i][j] is not None:
                                        current_conflict[i, j] = True

                    fig_pdf = Figure(figsize=(8.27, 11.69))
                    # --- JAVÍTOTT FEJLÉC ---
                    header_txt = f"Oldal {k + 1} / {total} | Készült a BME Traffic Lab programjával | http://traffic.bme.hu"
                    fig_pdf.text(0.5, 0.98, header_txt, ha='center', fontsize=9, color='black')
                    fig_pdf.suptitle(f"Csomópont: {jid}", fontsize=14, fontweight='bold', y=0.95)
                    # --- HALVÁNY, ÁTLÓS VÍZJEL ---
                    fig_pdf.text(0.5, 0.5, "BME Traffic Lab", ha='center', va='center', rotation=45,
                                 fontsize=80, color='gray', alpha=0.1)

                    ax_plot = fig_pdf.add_axes([0.05, 0.45, 0.60, 0.45])

                    if 'shape' in data['jdata'] and data['jdata']['shape']:
                        ax_plot.add_patch(MplPolygon(data['jdata']['shape'], facecolor='#e6e6e6',
                                                     edgecolor='#999999', alpha=0.5, zorder=-20))

                    legend_handles = []
                    for i, line in enumerate(data['lines']):
                        ax_plot.plot(line['x_smooth'], line['y_smooth'], color=line['color'], lw=2)
                        xs, ys = line['x_smooth'], line['y_smooth']
                        if len(xs) >= 2:
                            idx_start = -2
                            if len(xs) > 10: idx_start = -10
                            ax_plot.arrow(xs[idx_start], ys[idx_start],
                                          xs[-1] - xs[idx_start],
                                          ys[-1] - ys[idx_start],
                                          head_width=0.8, color=line['color'],
                                          length_includes_head=True, zorder=1)
                        legend_handles.append(
                            mlines.Line2D([], [], color=line['color'], linewidth=2,
                                          label=f"{i + 1}. {line['id']}"))

                    all_x, all_y = data['bounds']
                    if all_x:
                        mnx, mxx = min(all_x), max(all_x);
                        mny, mxy = min(all_y), max(all_y)
                        mgx = (mxx - mnx) * 0.05;
                        mgy = (mxy - mny) * 0.05
                        ax_plot.set_xlim(mnx - mgx, mxx + mgx)
                        ax_plot.set_ylim(mny - mgy, mxy + mgy)
                    ax_plot.set_aspect('equal')
                    ax_plot.grid(True, linestyle=':', alpha=0.5)

                    ax_plot.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
                                   loc='upper left', borderaxespad=0., fontsize=7)

                    ax_tbl = fig_pdf.add_axes([0.1, 0.05, 0.8, 0.35]);
                    ax_tbl.axis('off')
                    ids = [l['id'] for l in data['lines']]

                    cell_text = []
                    cell_colors = []
                    for r in range(n):
                        row_txt = []
                        row_col = []
                        for c in range(n):
                            if r == c:
                                txt = "";
                                col = "#444444"
                            else:
                                is_active = current_mask[r, c]
                                if not is_active:
                                    txt = "";
                                    col = "#d9d9d9"
                                else:
                                    if data['matrix'][r][c]:
                                        txt = f"{data['matrix'][r][c][0]:.0f}"
                                        col = "#e6ffe6"
                                    else:
                                        txt = "";
                                        col = "white"
                            row_txt.append(txt)
                            row_col.append(col)
                        cell_text.append(row_txt)
                        cell_colors.append(row_col)

                    tbl = ax_tbl.table(cellText=cell_text, cellColours=cell_colors, rowLabels=ids,
                                       colLabels=ids, loc='center')
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(6)

                    pdf.savefig(fig_pdf)
                    plt.close(fig_pdf)
                    print(f"Export: {k + 1}/{total}")
            messagebox.showinfo("Kész", "PDF generálva.")
        except Exception as e:
            messagebox.showerror("Hiba", str(e))

    # --- MÁTRIXOK LOGIKÁJA ---

    def update_combined_table(self):
        self.ax_combined.clear();
        self.ax_combined.set_axis_off()
        self.combined_table = None
        if not self.ids_for_table: return
        n = len(self.ids_for_table)
        matrix = self.current_data_cache['matrix']

        cell_text = []
        cell_colors = []

        for i in range(n):
            row_txt = []
            row_col = []
            for j in range(n):
                bg_color = "white"
                txt = ""

                if i == j:
                    bg_color = "#444444"
                else:
                    is_active = self.matrix_mask[i, j]

                    if not is_active:
                        bg_color = "#d9d9d9"
                        txt = ""
                    else:
                        # ITT A JAVÍTÁS:
                        # EREDETI: if matrix[i][j]:
                        # ÚJ (Exporttal egyező): if matrix[j][i]:
                        if matrix[j][i]:
                            # EREDETI: txt = f"{matrix[i][j][0]:.0f}"
                            # ÚJ:
                            txt = f"{matrix[j][i][0]:.0f}"
                            bg_color = "#e6ffe6"
                        else:
                            txt = ""

                row_txt.append(txt)
                row_col.append(bg_color)
            cell_text.append(row_txt)
            cell_colors.append(row_col)

        headers = [str(i + 1) for i in range(n)]

        self.ax_combined.text(0.5, 1.12, "Számítás Kijelölés & Eredmény [s]", ha='center',
                              transform=self.ax_combined.transAxes, fontsize=9, fontweight='bold')
        self.ax_combined.text(-0.08, 0.5, "Behaladó (Honnan)", va='center', ha='center',
                              rotation=90, transform=self.ax_combined.transAxes, fontsize=8)
        self.ax_combined.text(0.5, 1.05, "Kihaladó (Hova)", ha='center',
                              transform=self.ax_combined.transAxes, fontsize=8)

        self.combined_table = self.ax_combined.table(
            cellText=cell_text, cellColours=cell_colors,
            rowLabels=headers, colLabels=headers,
            loc='center', cellLoc='center', bbox=[0, 0, 1, 1]
        )
        self.combined_table.auto_set_font_size(False)
        self.combined_table.set_fontsize(7)

    def update_inhibit_table(self):
        self.ax_inhibit.clear();
        self.ax_inhibit.set_axis_off()
        self.inhibit_table = None
        if not self.ids_for_table: return
        n = len(self.ids_for_table)

        cell_text = []
        cell_colors = []

        for i in range(n):
            row_txt = []
            row_col = []
            for j in range(n):
                bg = "white"
                txt = ""
                if i == j:
                    bg = "#444444"
                elif self.conflict_matrix[i, j]:
                    bg = "#ffcccc"
                    txt = "X"
                else:
                    bg = "#ccffcc"
                row_txt.append(txt)
                row_col.append(bg)
            cell_text.append(row_txt)
            cell_colors.append(row_col)

        headers = [str(i + 1) for i in range(n)]

        self.ax_inhibit.text(0.5, 1.12, "Tiltás Mátrix (Konfliktusok)", ha='center',
                             transform=self.ax_inhibit.transAxes, fontsize=9, fontweight='bold')
        self.ax_inhibit.text(-0.08, 0.5, "Behaladó (Honnan)", va='center', ha='center', rotation=90,
                             transform=self.ax_inhibit.transAxes, fontsize=8)
        self.ax_inhibit.text(0.5, 1.05, "Kihaladó (Hova)", ha='center',
                             transform=self.ax_inhibit.transAxes, fontsize=8)

        self.inhibit_table = self.ax_inhibit.table(
            cellText=cell_text, cellColours=cell_colors,
            rowLabels=headers, colLabels=headers,
            loc='center', cellLoc='center', bbox=[0, 0, 1, 1]
        )
        self.inhibit_table.auto_set_font_size(False)
        self.inhibit_table.set_fontsize(7)

    def refresh_phases(self):
        """
        Recalculates and displays possible signal phases (cliques) based on the conflict matrix.
        Uses graph theory (finding cliques in the compatibility graph).
        """
        for widget in self.phase_frame_inner.winfo_children():
            widget.destroy()

        n = len(self.ids_for_table)
        if n == 0: return

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if not self.conflict_matrix[i, j]:
                    G.add_edge(i, j)

        cliques = list(nx.find_cliques(G))
        cliques.sort(key=len, reverse=True)

        self.phases_data = cliques

        def on_enter_phase(event, ids_in_clique):
            self.update_highlights(set(ids_in_clique))

        def on_leave_phase(event):
            self.update_highlights(None)

        COLUMNS = 2
        for idx, clique in enumerate(cliques):
            if not clique: continue
            clique = sorted(clique)

            current_ids = [self.ids_for_table[x] for x in clique]

            fr = tk.LabelFrame(self.phase_frame_inner, text=f"Fázis {idx + 1}",
                               font=("Arial", 9, "bold"), bg="white", padx=5, pady=5)

            row = idx // COLUMNS
            col = idx % COLUMNS
            fr.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            fr.bind("<Enter>", lambda e, c=current_ids: on_enter_phase(e, c))
            fr.bind("<Leave>", on_leave_phase)

            for x in clique:
                line_color_rgba = self.current_data_cache['lines'][x]['color']
                line_color_hex = mcolors.to_hex(line_color_rgba)
                lbl_text = f"{x + 1}. {self.ids_for_table[x]}"

                lbl = tk.Label(fr, text=lbl_text, justify=tk.LEFT, bg="white",
                               fg=line_color_hex, font=("Consolas", 9, "bold"))
                lbl.pack(anchor="w")

                lbl.bind("<Enter>", lambda e, c=current_ids: on_enter_phase(e, c))
                lbl.bind("<Leave>", on_leave_phase)

    # --- PLOTOLÁS ---

    def plot_current(self):
        """
        Draws the current junction state on the Matplotlib canvas.
        Includes lanes, conflict points, and annotations.
        """
        self.ax.clear()
        self.ax_check.clear();
        self.ax_check.set_axis_off()
        self.interactive_items = []
        self.lines_map = {}
        self.all_annotations = []
        self.ids_for_table = []
        self.highlighted_ids = None
        self.pan_start = None  # Reset pan state

        if not self.junctions:
            self.ax.text(0.5, 0.5, "Nincs adat.", ha='center')
            self.canvas.draw()
            return

        data = calculate_junction_data(self.junctions[self.current_idx])
        self.current_data_cache = data
        self.ax.set_title(f"Csomópont: {data['jdata']['id']}", fontsize=11, fontweight='bold')

        current_id = data['jdata']['id']
        n = len(data['lines'])

        if current_id in self.saved_settings and 'matrix_mask' in self.saved_settings[current_id]:
            loaded = np.array(self.saved_settings[current_id]['matrix_mask'])
            if loaded.shape == (n, n):
                self.matrix_mask = loaded
            else:
                self.matrix_mask = np.ones((n, n), dtype=bool)
        else:
            if self.matrix_mask is None or self.matrix_mask.shape != (n, n):
                self.matrix_mask = np.ones((n, n), dtype=bool)

        if current_id in self.saved_settings and 'conflict_matrix' in self.saved_settings[
            current_id]:
            loaded = np.array(self.saved_settings[current_id]['conflict_matrix'])
            if loaded.shape == (n, n):
                self.conflict_matrix = loaded
            else:
                self.conflict_matrix = None

        if self.conflict_matrix is None or self.conflict_matrix.shape != (n, n):
            self.conflict_matrix = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if data['matrix'][i][j] is not None:
                            self.conflict_matrix[i, j] = True
                        else:
                            self.conflict_matrix[i, j] = False

        if 'shape' in data['jdata'] and data['jdata']['shape']:
            self.ax.add_patch(
                MplPolygon(data['jdata']['shape'], facecolor='#e6e6e6', edgecolor='#999999',
                           alpha=0.5, zorder=-20))

        colors = [l['color'] for l in data['lines']]

        labels_for_checkbox = []
        all_x, all_y = data['bounds']

        for i, line_data in enumerate(data['lines']):
            display_id = line_data['id']
            checkbox_label = f"{i + 1}. {display_id}"
            labels_for_checkbox.append(checkbox_label)
            self.ids_for_table.append(display_id)

            artists = []
            l, = self.ax.plot(line_data['x_smooth'], line_data['y_smooth'], '-', lw=2,
                              color=line_data['color'], alpha=0.7, zorder=1)
            artists.append(l)

            xs, ys = line_data['x_smooth'], line_data['y_smooth']
            if len(xs) >= 2:
                idx_start = -2
                if len(xs) > 10: idx_start = -10

                arr = self.ax.arrow(xs[idx_start], ys[idx_start],
                                    xs[-1] - xs[idx_start],
                                    ys[-1] - ys[idx_start],
                                    head_width=0.8, color=line_data['color'],
                                    length_includes_head=True, zorder=1)
                artists.append(arr)

            if line_data['radius_data']:
                min_r, px, py, nx, ny = line_data['radius_data']
                lx, ly = px + nx * 3.0, py + ny * 3.0
                ann = self.ax.annotate(f"R={min_r:.1f}m", xy=(px, py), xytext=(lx, ly),
                                       arrowprops=dict(arrowstyle="->", color=line_data['color'],
                                                       linestyle='--'),
                                       fontsize=8, fontweight='bold', ha='center', zorder=2,
                                       bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                                 ec=line_data['color'], alpha=0.9))
                artists.append(ann)
                self.all_annotations.append(ann)

            self.lines_map[checkbox_label] = artists

        for inter in data['intersections']:
            val, mx, my = inter['val'], inter['pos'][0], inter['pos'][1]
            i = inter['idx_i']

            lbl1 = self.ax.annotate(f"K={val}s", xy=(mx, my), xytext=(-20, 20),
                                    textcoords='offset points',
                                    arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.5),
                                    color='white', fontweight='bold', fontsize=9, ha='right',
                                    va='bottom', zorder=15,
                                    bbox=dict(boxstyle="round,pad=0.15", fc=colors[i], ec='none',
                                              alpha=0.85))
            self.all_annotations.append(lbl1)
            key1 = labels_for_checkbox[i]
            if key1 in self.lines_map: self.lines_map[key1].append(lbl1)

        if not self.annotations_visible:
            for artist in self.all_annotations: artist.set_visible(False)

        if labels_for_checkbox:
            self.ax_check.set_facecolor('white')
            self.check_buttons = CheckButtons(self.ax_check, labels_for_checkbox,
                                              [True] * len(labels_for_checkbox))
            for i, label in enumerate(self.check_buttons.labels):
                label.set_fontsize(7)
                label.set_color(colors[i])
            self.check_buttons.on_clicked(self.toggle_visibility)

        self.update_combined_table()
        self.update_inhibit_table()
        self.refresh_phases()

        if self.current_limits:
            self.ax.set_xlim(self.current_limits[0]);
            self.ax.set_ylim(self.current_limits[1])
        elif all_x:
            mnx, mxx = min(all_x), max(all_x);
            mny, mxy = min(all_y), max(all_y)
            mgx = (mxx - mnx) * 0.05;
            mgy = (mxy - mny) * 0.05
            self.ax.set_xlim(mnx - mgx, mxx + mgx);
            self.ax.set_ylim(mny - mgy, mxy + mgy)

        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.ax.set_aspect('equal')
        self.fig.canvas.draw()

    # --- INTERAKCIÓ ---

    def on_click(self, event):
        if event.inaxes == self.ax:
            self.pan_start = (event.x, event.y)
            self.pan_ax_lims = (self.ax.get_xlim(), self.ax.get_ylim())
            return

        if not self.ids_for_table: return
        n = len(self.ids_for_table)

        if event.inaxes == self.ax_combined and self.combined_table:
            for (row, col), cell in self.combined_table.get_celld().items():
                if cell.contains(event)[0]:
                    r, c = row - 1, col
                    if 0 <= r < n and 0 <= c < n and r != c:
                        self.matrix_mask[r, c] = not self.matrix_mask[r, c]
                        self.update_combined_table()
                        self.fig.canvas.draw_idle()
                    return

        if event.inaxes == self.ax_inhibit and self.inhibit_table:
            for (row, col), cell in self.inhibit_table.get_celld().items():
                if cell.contains(event)[0]:
                    r, c = row - 1, col
                    if 0 <= r < n and 0 <= c < n and r != c:
                        val = not self.conflict_matrix[r, c]
                        self.conflict_matrix[r, c] = val
                        self.conflict_matrix[c, r] = val
                        self.update_inhibit_table()
                        self.update_combined_table()
                        self.refresh_phases()
                        self.fig.canvas.draw_idle()
                    return

    def on_release(self, event):
        if self.pan_start:
            self.pan_start = None
            self.pan_ax_lims = None
            self.current_limits = (self.ax.get_xlim(), self.ax.get_ylim())

    def on_hover(self, event):
        if self.pan_start and event.x is not None and event.y is not None:
            dx_pix = event.x - self.pan_start[0]
            dy_pix = event.y - self.pan_start[1]

            start_xlim, start_ylim = self.pan_ax_lims
            width_data = start_xlim[1] - start_xlim[0]
            height_data = start_ylim[1] - start_ylim[0]

            bbox = self.ax.bbox
            scale_x = width_data / bbox.width
            scale_y = height_data / bbox.height

            dx_data = dx_pix * scale_x
            dy_data = dy_pix * scale_y

            self.ax.set_xlim(start_xlim[0] - dx_data, start_xlim[1] - dx_data)
            self.ax.set_ylim(start_ylim[0] - dy_data, start_ylim[1] - dy_data)

            self.canvas.draw_idle()
            return

        hovered_ids = None
        target_tables = []
        if self.combined_table: target_tables.append((self.ax_combined, self.combined_table))
        if self.inhibit_table: target_tables.append((self.ax_inhibit, self.inhibit_table))

        found = False
        for ax_t, table_t in target_tables:
            if event.inaxes == ax_t:
                for (r, c), cell in table_t.get_celld().items():
                    if cell.contains(event)[0]:
                        idx1, idx2 = r - 1, c
                        if 0 <= idx1 < len(self.ids_for_table) and 0 <= idx2 < len(
                                self.ids_for_table):
                            hovered_ids = {self.ids_for_table[idx1], self.ids_for_table[idx2]}
                            found = True
                        break
            if found: break

        self.update_highlights(hovered_ids)

    def update_highlights(self, active_ids):
        if active_ids == self.highlighted_ids: return
        self.highlighted_ids = active_ids

        for label, artists in self.lines_map.items():
            clean_id = label.split(". ")[1]
            is_active = (active_ids is None) or (clean_id in active_ids)

            for artist in artists:
                alpha = 0.8 if (is_active or active_ids is None) else 0.1
                lw = 3 if is_active and active_ids is not None else 2

                if isinstance(artist, plt.Line2D):
                    artist.set_alpha(alpha);
                    artist.set_linewidth(lw)
                    artist.set_zorder(10 if is_active and active_ids is not None else 1)

                elif isinstance(artist, mpatches.Patch):
                    artist.set_alpha(alpha)
                    artist.set_zorder(10 if is_active and active_ids is not None else 1)

        self.fig.canvas.draw_idle()

    # --- 5. ÚJ: FULL EXPORT (JAVÍTOTT MÁTRIX + RÉSZLETES JSON LÁNC) ---

    # --- 5. ÚJ: FULL EXPORT (ÖSSZES KOMBINÁCIÓ + JAVÍTOTT MÁTRIX) ---

    # --- 5. ÚJ: FULL EXPORT (VISSZAÁLLÍTOTT MÁTRIX OLVASÁS + TELJES LÁNC) ---

    # --- 5. ÚJ: FULL EXPORT (VÉGLEGES: GRAFIKONNAL EGYEZŐ LOGIKA) ---

    def export_all_to_sumo(self):
        """
        Kiexportálja a közlekedési lámpákat SUMO (.add.xml) és JSON (.json) formátumban.
        
        MŰKÖDÉS:
        1. Mátrix olvasása a helyes [stop][start] (Honnan->Hova) irányban.
        2. Idővonal-vágásos (Timeline Slicing) generálás:
           - Nem fix blokkokat rak egymás után, hanem kiszámolja a "kritikus időpontokat"
             (Sárga vége, Piros-Sárga kezdete), és ezek mentén vágja fel az időt.
           - Ez pontosan ugyanazt az eredményt adja, mint a 'Fázisátmenet Tervező' vizuális sávjai.
        """
        filename_xml = filedialog.asksaveasfilename(defaultextension=".add.xml",
                                                    filetypes=[("SUMO Additional", "*.add.xml"),
                                                               ("XML", "*.xml")],
                                                    initialfile="traffic_lights.add.xml")
        if not filename_xml:
            return

        filename_json = filename_xml.rsplit('.', 2)[0] + ".json"

        root_xml = ET.Element("additional")
        json_export_data = {}

        # --- IDŐZÍTÉSI KONSTANSOK ---
        MIN_GREEN_TIME = 5.0
        YELLOW_TIME = 3.0
        RED_YELLOW_TIME = 2.0

        progress_win = tk.Toplevel(self.root)
        progress_win.title("Exportálás...")
        lbl = tk.Label(progress_win, text="Feldolgozás...", padx=20, pady=20)
        lbl.pack()
        progress_win.update()

        try:
            total_junc = len(self.junctions)
            for idx, jdata in enumerate(self.junctions):
                jid = jdata['id']
                lbl.config(text=f"Feldolgozás: {jid} ({idx + 1}/{total_junc})")
                progress_win.update()

                # 1. Adatok
                data = calculate_junction_data(jdata)
                n = len(data['lines'])
                if n == 0: continue

                c_matrix = np.zeros((n, n), dtype=bool)
                saved_conflicts = None
                if jid in self.saved_settings and 'conflict_matrix' in self.saved_settings[jid]:
                    saved_conflicts = np.array(self.saved_settings[jid]['conflict_matrix'])

                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        if saved_conflicts is not None and saved_conflicts.shape == (n, n):
                            if saved_conflicts[i, j]: c_matrix[i, j] = True
                        else:
                            if data['matrix'][i][j] is not None:
                                c_matrix[i, j] = True

                # 2. Klikkek (Logikai Zöldek)
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for i in range(n):
                    for j in range(i + 1, n):
                        if not c_matrix[i, j]:
                            G.add_edge(i, j)

                cliques = list(nx.find_cliques(G))
                cliques.sort(key=len, reverse=True)

                if not cliques: continue

                # --- EXPORT STRUKTÚRA ---
                tlLogic = ET.SubElement(root_xml, "tlLogic", id=jid, type="static",
                                        programID="0", offset="0")

                junction_json = {
                    "programID": "0",
                    "phases": [],
                    "logic_phases": {},
                    "transitions": {}
                }

                sumo_phase_counter = 0
                num_logic_phases = len(cliques)

                # ==========================================
                # LÉPÉS 1: CSAK A ZÖLD FÁZISOK (Hogy fix indexük legyen az elején)
                # ==========================================
                for i in range(num_logic_phases):
                    current_phase_idxs = set(cliques[i])

                    state_g = ['r'] * n
                    for lane_idx in range(n):
                        if lane_idx in current_phase_idxs:
                            state_g[lane_idx] = 'G'
                    state_g_str = "".join(state_g)

                    ET.SubElement(tlLogic, "phase", duration=str(MIN_GREEN_TIME),
                                  state=state_g_str)

                    p_obj = {
                        "index": sumo_phase_counter,
                        "duration": MIN_GREEN_TIME,
                        "state": state_g_str,
                        "type": "green",
                        "logic_idx": i
                    }
                    junction_json["phases"].append(p_obj)
                    junction_json["logic_phases"][str(i)] = sumo_phase_counter

                    sumo_phase_counter += 1

                # ==========================================
                # LÉPÉS 2: AZ ÖSSZES ÁTMENET GENERÁLÁSA
                # ==========================================
                for i in range(num_logic_phases):       # HONNAN
                    for j in range(num_logic_phases):   # HOVA

                        if i == j: continue

                        current_phase_idxs = set(cliques[i])
                        next_phase_idxs = set(cliques[j])

                        transition_steps = []
                        
                        stopping = []
                        starting = []
                        staying = []

                        for lane_idx in range(n):
                            is_curr = lane_idx in current_phase_idxs
                            is_next = lane_idx in next_phase_idxs

                            if is_curr and not is_next:
                                stopping.append(lane_idx)
                            elif not is_curr and is_next:
                                starting.append(lane_idx)
                            elif is_curr and is_next:
                                staying.append(lane_idx)

                        # --- MÁTRIX MAX KERESÉS (HELYES IRÁNY) ---
                        K_needed = 0.0
                        for stop_idx in stopping:
                            for start_idx in starting:
                                # [stop][start] = Clearing -> Entering
                                # Ez adja a nagyobb, helyes biztonsági időket
                                val = data['matrix'][stop_idx][start_idx]
                                if val is not None:
                                    if val[0] > K_needed: K_needed = val[0]

                        # Ha nincs mit csinálni
                        if not stopping and not starting:
                            key = f"{i}->{j}"
                            junction_json["transitions"][key] = []
                            continue

                        # --- IDŐZÍTÉS ÉS VÁGÓPONTOK (A GRAFIKON LOGIKÁJA) ---
                        # A teljes átmenet hossza a mátrix érték (K), de minimum a sárga hossza.
                        transition_total_time = max(K_needed, YELLOW_TIME)
                        
                        # Időpontok az idővonalon (t=0 a Sárga kezdete)
                        t_yellow_end = YELLOW_TIME
                        t_redyellow_start = max(0.0, transition_total_time - RED_YELLOW_TIME)
                        t_end = transition_total_time

                        # Vágópontok összegyűjtése és rendezése
                        cut_points = sorted(
                            list(set([0.0, t_yellow_end, t_redyellow_start, t_end])))
                        # Csak a [0, t_end] tartomány érdekes
                        cut_points = [t for t in cut_points if t >= 0 and t <= t_end]

                        prev_t = 0.0
                        last_phase_obj = None

                        # Szakaszok generálása a vágópontok között
                        for t_idx, t in enumerate(cut_points):
                            if t <= prev_t: continue
                            duration = t - prev_t
                            if duration < 0.1: continue # Túl rövid szakaszok szűrése

                            mid_t = (prev_t + t) / 2.0
                            current_state_chars = ['r'] * n

                            for idx_lane in range(n):
                                if idx_lane in staying:
                                    # Aki marad, az végig Zöld
                                    current_state_chars[idx_lane] = 'G'
                                elif idx_lane in stopping:
                                    # LEÁLLÓ: Sárga (0-tól t_yellow_end-ig), utána Piros
                                    if mid_t < t_yellow_end:
                                        current_state_chars[idx_lane] = 'y'
                                    else:
                                        current_state_chars[idx_lane] = 'r'
                                elif idx_lane in starting:
                                    # INDULÓ: Piros, majd a végén Piros-Sárga
                                    if mid_t < t_redyellow_start:
                                        current_state_chars[idx_lane] = 'r'
                                    else:
                                        current_state_chars[idx_lane] = 'u'
                                else:
                                    current_state_chars[idx_lane] = 'r'

                            state_str = "".join(current_state_chars)

                            ET.SubElement(tlLogic, "phase", duration=f"{duration:.1f}",
                                          state=state_str)

                            # Típus meghatározása a karakterek alapján
                            p_type = "transition"
                            has_y = 'y' in state_str
                            has_u = 'u' in state_str # 'u' = red-yellow
                            
                            if has_y and has_u:
                                p_type = "overlap_yellow_redyellow" # Átlapolás
                            elif has_y:
                                p_type = "yellow"
                            elif has_u:
                                p_type = "red_yellow"
                            elif not has_y and not has_u:
                                p_type = "all_red" # Tiszta vörös

                            t_obj = {
                                "index": sumo_phase_counter,
                                "duration": float(f"{duration:.1f}"),
                                "state": state_str,
                                "type": p_type,
                                "from_logic": i,
                                "to_logic": j
                            }
                            junction_json["phases"].append(t_obj)

                            if last_phase_obj:
                                last_phase_obj["next_index"] = sumo_phase_counter
                            last_phase_obj = t_obj

                            transition_steps.append(sumo_phase_counter)
                            sumo_phase_counter += 1
                            prev_t = t

                        key = f"{i}->{j}"
                        junction_json["transitions"][key] = transition_steps

                json_export_data[jid] = junction_json

            # --- MENTÉS ---
            tree = ET.ElementTree(root_xml)
            self.indent(root_xml)
            tree.write(filename_xml, encoding="utf-8", xml_declaration=True)

            with open(filename_json, 'w', encoding='utf-8') as f:
                json.dump(json_export_data, f, indent=4)

            messagebox.showinfo("Siker",
                                f"Export Kész!\nJSON és SUMO átmenetek pontosan követik a grafikus tervet.")

        except Exception as e:
            messagebox.showerror("Hiba", str(e))
            import traceback
            traceback.print_exc()
        finally:
            progress_win.destroy()
    
    def indent(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
