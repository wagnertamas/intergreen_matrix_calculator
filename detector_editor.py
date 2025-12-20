import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import xml.etree.ElementTree as ET
import os

class DetectorEditor:
    def __init__(self, root, parser):
        self.root = root
        self.root.title("Detektor Konfigurátor (RL)")
        self.root.geometry("1400x800")
        
        self.parser = parser
        self.junctions = parser.junctions
        self.current_idx = 0
        
        # --- AUTOMATA FÁJLNÉV ---
        # A network fájl neve alapján: 'nev.net.xml' -> 'nev.detectors.add.xml'
        net_path = self.parser.file_path
        dir_name = os.path.dirname(net_path)
        base_name = os.path.basename(net_path)
        
        if base_name.endswith(".net.xml"):
            out_name = base_name.replace(".net.xml", ".detectors.add.xml")
        elif base_name.endswith(".xml"):
            out_name = base_name.replace(".xml", ".detectors.add.xml")
        else:
            out_name = base_name + ".detectors.add.xml"
            
        self.auto_filename = os.path.join(dir_name, out_name)
        
        # Adatstruktúra a beállítások tárolására
        self.detector_config = {} 
        
        # Jelenleg kijelölt sáv ID-ja
        self.selected_lane_id = None
        
        # Segédváltozó az eseménykezeléshez
        self.just_picked = False 
        
        # UI változók
        self.var_lane_id = tk.StringVar(value="-")
        self.var_enabled = tk.BooleanVar(value=True)
        self.var_dist = tk.DoubleVar(value=20.0)
        self.default_dist_global = tk.DoubleVar(value=20.0)

        # --- AUTOMATIKUS BETÖLTÉS ---
        self.load_existing_detectors()

        self._init_ui()
        self.plot_current()

    def _init_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- BAL OLDAL ---
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        nav_frame = tk.Frame(left_panel, bd=1, relief=tk.RAISED)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="<< Előző", command=self.prev_junction).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Következő >>", command=self.next_junction).pack(side=tk.LEFT, padx=2)
        self.lbl_junc = ttk.Label(nav_frame, text="Junction: ...", font=("Arial", 10, "bold"))
        self.lbl_junc.pack(side=tk.LEFT, padx=10)
        
        # Információ a fájlról
        lbl_file = tk.Label(nav_frame, text=f"Fájl: {os.path.basename(self.auto_filename)}", fg="#555")
        lbl_file.pack(side=tk.RIGHT, padx=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click_canvas)

        # --- JOBB OLDAL ---
        right_panel = tk.Frame(main_frame, width=300, bg="#f0f0f0", bd=1, relief=tk.SUNKEN)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)
        right_panel.pack_propagate(False) 
        
        tk.Label(right_panel, text="Detektor Beállítások", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)
        
        gb_frame = tk.LabelFrame(right_panel, text="Globális Alapértelmezés", bg="#f0f0f0", padx=5, pady=5)
        gb_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(gb_frame, text="Távolság (m):", bg="#f0f0f0").pack(side=tk.LEFT)
        entry_def = ttk.Entry(gb_frame, textvariable=self.default_dist_global, width=5)
        entry_def.pack(side=tk.LEFT, padx=5)
        entry_def.bind('<Return>', lambda e: self.plot_current())
        ttk.Button(gb_frame, text="Mind Frissít", command=self.reset_current_junction_defaults).pack(side=tk.RIGHT)

        self.editor_frame = tk.LabelFrame(right_panel, text="Kijelölt Sáv Szerkesztése", bg="#f0f0f0", padx=5, pady=5)
        self.editor_frame.pack(fill=tk.X, padx=10, pady=20)
        
        tk.Label(self.editor_frame, text="Sáv ID:", bg="#f0f0f0").grid(row=0, column=0, sticky="w", pady=5)
        tk.Label(self.editor_frame, textvariable=self.var_lane_id, font=("Consolas", 10, "bold"), bg="#f0f0f0", fg="blue").grid(row=0, column=1, sticky="w")
        
        self.chk_enabled = tk.Checkbutton(self.editor_frame, text="Detektor elhelyezése", variable=self.var_enabled, bg="#f0f0f0", command=self.update_lane_settings)
        self.chk_enabled.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        tk.Label(self.editor_frame, text="Távolság a stopvonaltól (m):", bg="#f0f0f0").grid(row=2, column=0, columnspan=2, sticky="w")
        self.ent_dist = ttk.Entry(self.editor_frame, textvariable=self.var_dist)
        self.ent_dist.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.ent_dist.bind('<Return>', lambda e: self.update_lane_settings())
        self.ent_dist.bind('<FocusOut>', lambda e: self.update_lane_settings())
        
        self.toggle_editor_state("disabled")

        btn_frame = tk.Frame(right_panel, bg="#f0f0f0")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)
        
        # MENTÉS GOMB (Automata)
        ttk.Button(btn_frame, text="MENTÉS", command=self.save_detectors_auto).pack(fill=tk.X, pady=5)

    def load_existing_detectors(self):
        """Megpróbálja betölteni az auto_filename fájlt és populálni a configot."""
        if not os.path.exists(self.auto_filename):
            print("Nincs meglévő detektor fájl, tiszta lappal indulunk.")
            return

        try:
            tree = ET.parse(self.auto_filename)
            root = tree.getroot()
            
            # 1. Beolvassuk az összes létező detektort: {lane_id: pos_from_start}
            loaded_data = {}
            for det in root.findall("inductionLoop"):
                lid = det.get("lane")
                pos = det.get("pos")
                if lid and pos:
                    loaded_data[lid] = float(pos)
            
            print(f"Betöltve {len(loaded_data)} detektor a fájlból.")

            # 2. Végigmegyünk az ÖSSZES csomóponton és sávon, hogy beállítsuk a configot
            # Ez azért kell, hogy ami NINCS a fájlban, az Disabled legyen.
            for jdata in self.junctions:
                jid = jdata['id']
                
                # Sávok keresése (XML lookup)
                j_node = self.parser.root.find(f".//junction[@id='{jid}']")
                if j_node is None: continue
                inc_lanes_str = j_node.get("incLanes")
                if not inc_lanes_str: continue
                
                lids = inc_lanes_str.split(" ")
                
                for lid in lids:
                    if lid in loaded_data:
                        # Ha megvan a fájlban: Enabled + Távolság visszaszámolása
                        pos_start = loaded_data[lid]
                        lane_len = self.get_lane_length(lid)
                        dist_end = max(0, lane_len - pos_start)
                        
                        self.set_lane_config(jid, lid, enabled=True, dist=dist_end)
                    else:
                        # Ha nincs a fájlban: Disabled
                        # (Különben a get_lane_config defaultja True lenne)
                        self.set_lane_config(jid, lid, enabled=False, dist=self.default_dist_global.get())
                        
        except Exception as e:
            print(f"Hiba a betöltéskor: {e}")
            messagebox.showwarning("Betöltési Hiba", f"Nem sikerült beolvasni a meglévő fájlt:\n{e}")

    def get_lane_length(self, lid):
        """Segédfüggvény: sáv hossza a geometriából."""
        if lid in self.parser.normal_lanes:
            points = self.parser.normal_lanes[lid]
            length = 0
            for i in range(len(points)-1):
                length += math.dist(points[i], points[i+1])
            return length
        return 100.0 # Fallback

    def set_lane_config(self, jid, lid, enabled, dist):
        if jid not in self.detector_config:
            self.detector_config[jid] = {}
        self.detector_config[jid][lid] = {'enabled': enabled, 'dist': dist}

    def toggle_editor_state(self, state):
        for child in self.editor_frame.winfo_children():
            try:
                child.configure(state=state)
            except: pass

    def prev_junction(self):
        self.current_idx = (self.current_idx - 1) % len(self.junctions)
        self.selected_lane_id = None
        self.toggle_editor_state("disabled")
        self.var_lane_id.set("-")
        self.plot_current()

    def next_junction(self):
        self.current_idx = (self.current_idx + 1) % len(self.junctions)
        self.selected_lane_id = None
        self.toggle_editor_state("disabled")
        self.var_lane_id.set("-")
        self.plot_current()

    def get_lane_config(self, jid, lid):
        if jid not in self.detector_config:
            self.detector_config[jid] = {}
        if lid not in self.detector_config[jid]:
            # Ha még nincs bejegyzés (és nem volt a fájlban sem), akkor Default True
            self.detector_config[jid][lid] = {
                'enabled': True,
                'dist': self.default_dist_global.get()
            }
        return self.detector_config[jid][lid]

    def update_lane_settings(self):
        if self.selected_lane_id:
            jid = self.junctions[self.current_idx]['id']
            cfg = self.detector_config[jid][self.selected_lane_id]
            cfg['enabled'] = self.var_enabled.get()
            try:
                cfg['dist'] = float(self.var_dist.get())
            except ValueError: pass 
            self.plot_current()

    def reset_current_junction_defaults(self):
        jid = self.junctions[self.current_idx]['id']
        # Töröljük a configot -> visszaáll a get_lane_config defaultjára (ami True + Default Dist)
        if jid in self.detector_config:
            del self.detector_config[jid]
        self.plot_current()

    def calculate_point_on_polyline(self, shape, distance_from_end):
        if not shape or len(shape) < 2: return None, None
        total_len = 0
        segments = []
        for i in range(len(shape)-1, 0, -1):
            p1 = shape[i]; p0 = shape[i-1]
            seg_len = math.dist(p0, p1)
            segments.append((p0, p1, seg_len))
            total_len += seg_len
            
        current_dist = 0
        for p0, p1, seg_len in segments:
            if current_dist + seg_len >= distance_from_end:
                remaining = distance_from_end - current_dist
                ratio = remaining / seg_len
                dx = p0[0] - p1[0]; dy = p0[1] - p1[1]
                x = p1[0] + dx * ratio; y = p1[1] + dy * ratio
                
                perp_dx = -dy; perp_dy = dx
                mag = math.sqrt(perp_dx**2 + perp_dy**2)
                if mag > 0: perp_dx /= mag; perp_dy /= mag
                return (x, y), (perp_dx, perp_dy)
            current_dist += seg_len
        return shape[0], (0, 1)

    # --- ESEMÉNYKEZELÉS (JAVÍTOTT) ---

    def on_pick(self, event):
        line = event.artist
        lid = line.get_label()
        
        if lid and not lid.startswith("_"):
            self.selected_lane_id = lid
            self.just_picked = True
            
            jid = self.junctions[self.current_idx]['id']
            cfg = self.get_lane_config(jid, lid)
            
            self.var_lane_id.set(lid)
            self.var_enabled.set(cfg['enabled'])
            self.var_dist.set(cfg['dist'])
            
            self.toggle_editor_state("normal")
            self.plot_current()

    def on_click_canvas(self, event):
        if event.inaxes != self.ax: return
        if self.just_picked:
            self.just_picked = False
            return
        if self.selected_lane_id is not None:
            self.selected_lane_id = None
            self.toggle_editor_state("disabled")
            self.var_lane_id.set("-")
            self.plot_current()

    # --- PLOT ---

    def plot_current(self):
        self.ax.clear()
        jdata = self.junctions[self.current_idx]
        jid = jdata['id']
        self.lbl_junc.config(text=f"Junction: {jid}")
        self.ax.set_title(f"Szerkesztés: {jid}")

        if 'shape' in jdata and jdata['shape']:
             poly = plt.Polygon(jdata['shape'], facecolor='#e6e6e6', edgecolor='black', alpha=0.5, zorder=-10, picker=False)
             self.ax.add_patch(poly)

        # Sávok lekérése XML-ből
        incoming_lane_ids = []
        j_node = self.parser.root.find(f".//junction[@id='{jid}']")
        if j_node is not None:
            inc_lanes_str = j_node.get("incLanes")
            if inc_lanes_str:
                incoming_lane_ids = inc_lanes_str.split(" ")

        jx, jy = jdata['x'], jdata['y']

        for lid in incoming_lane_ids:
            if lid in self.parser.normal_lanes:
                points = self.parser.normal_lanes[lid]
                rel_points = [(p[0] - jx, p[1] - jy) for p in points]
                
                cfg = self.get_lane_config(jid, lid)
                is_selected = (lid == self.selected_lane_id)
                
                color = 'blue'
                width = 2
                alpha = 0.6
                zorder = 1
                
                if is_selected:
                    color = 'cyan'
                    width = 4
                    alpha = 1.0
                    zorder = 10
                elif not cfg['enabled']:
                    color = 'gray'
                    alpha = 0.3
                
                self.ax.plot(*zip(*rel_points), color=color, lw=width, alpha=alpha, label=lid, picker=5, zorder=zorder)
                
                if cfg['enabled']:
                    pos, vec = self.calculate_point_on_polyline(rel_points, cfg['dist'])
                    if pos:
                        dx, dy = vec
                        det_color = 'orange' if not is_selected else 'red'
                        self.ax.plot([pos[0]-dx, pos[0]+dx], [pos[1]-dy, pos[1]+dy], color=det_color, lw=3, zorder=zorder+1)
                        self.ax.plot(pos[0], pos[1], 'o', color='red', markersize=3, zorder=zorder+2)

        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle=':', alpha=0.3)
        self.canvas.draw()

    # --- MENTÉS (AUTOMATA) ---

    def save_detectors_auto(self):
        """Mentés az automatikusan generált fájlnévre."""
        root = ET.Element("additional")
        all_junction_ids = [j['id'] for j in self.junctions]
        
        count = 0
        for jid in all_junction_ids:
            j_node = self.parser.root.find(f".//junction[@id='{jid}']")
            if j_node is None: continue
            
            inc_lanes_str = j_node.get("incLanes")
            if not inc_lanes_str: continue
            lids = inc_lanes_str.split(" ")
            
            for lid in lids:
                # Itt fontos: ellenőrizzük, hogy van-e explicit config.
                # Ha nincs, akkor a get_lane_config defaultot gyárt (enabled=True).
                # Ha betöltöttünk fájlt, akkor a 'set_lane_config' már beállította False-ra, ami nincs benne.
                cfg = self.get_lane_config(jid, lid)
                
                if cfg['enabled']:
                    if lid in self.parser.normal_lanes:
                        points = self.parser.normal_lanes[lid]
                        length = 0
                        for i in range(len(points)-1):
                            length += math.dist(points[i], points[i+1])
                            
                        final_pos = max(0.0, length - cfg['dist'])
                        
                        ET.SubElement(root, "inductionLoop",
                                      id=f"e1_{lid}",
                                      lane=lid,
                                      pos=f"{final_pos:.2f}",
                                      freq="60",
                                      file="detectors.out.xml")
                        count += 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        try:
            tree.write(self.auto_filename, encoding="utf-8", xml_declaration=True)
            messagebox.showinfo("Siker", f"Mentve: {os.path.basename(self.auto_filename)}\n({count} db detektor)")
        except Exception as e:
            messagebox.showerror("Hiba", f"Nem sikerült menteni:\n{e}")