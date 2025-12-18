import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from matplotlib.widgets import Cursor, CheckButtons
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, MultiPoint
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import sys
import gc
import csv
import itertools


# --- 1. PARSER ---

class SumoInternalParser:
    def __init__(self, file_path):
        print(f"Fájl feldolgozása: {file_path}")
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.internal_lanes = {}
        self.normal_lanes = {}
        self._load_data()
        self.junctions = self._group_by_junctions()
        print(f"Betöltve: {len(self.junctions)} csomópont.")

    def _load_data(self):
        for edge in self.root.findall("edge"):
            func = edge.get("function")
            target_dict = self.internal_lanes if func == "internal" else self.normal_lanes
            for lane in edge.findall("lane"):
                lid = lane.get("id")
                shape = lane.get("shape")
                if lid and shape:
                    points = []
                    for pair in shape.split(" "):
                        if "," in pair:
                            x, y = map(float, pair.split(","))
                            points.append((x, y))
                    target_dict[lid] = points

    def _group_by_junctions(self):
        junctions_data = []
        for junc in self.root.findall("junction"):
            if junc.get("type") in ["traffic_light", "priority", "right_before_left", "zipper"]:
                jx = float(junc.get("x"));
                jy = float(junc.get("y"));
                jid = junc.get("id")
                shape_str = junc.get("shape")
                rel_shape = []
                if shape_str:
                    for pair in shape_str.split(" "):
                        if "," in pair:
                            sx, sy = map(float, pair.split(","))
                            rel_shape.append((sx - jx, sy - jy))
                inc_lanes_str = junc.get("incLanes")
                incoming_shapes = []
                if inc_lanes_str:
                    for lane_id in inc_lanes_str.split(" "):
                        if lane_id in self.normal_lanes:
                            points = self.normal_lanes[lane_id]
                            rel_points = [(p[0] - jx, p[1] - jy) for p in points]
                            incoming_shapes.append(rel_points)
                prefix = f":{jid}_"
                segments = []
                for lid, points in self.internal_lanes.items():
                    if lid.startswith(prefix):
                        rel_points = [(p[0] - jx, p[1] - jy) for p in points]
                        segments.append({'id': lid, 'points': rel_points})
                if segments:
                    junctions_data.append(
                        {'id': jid, 'x': jx, 'y': jy, 'segments': segments, 'shape': rel_shape,
                         'incoming_lanes': incoming_shapes})
        return junctions_data


# --- 2. GEOMETRIA ---

def remove_close_points(coords, min_dist=0.8):
    if len(coords) < 2: return coords
    clean = [coords[0]]
    for i in range(1, len(coords)):
        if np.linalg.norm(np.array(coords[i]) - np.array(clean[-1])) > min_dist:
            clean.append(coords[i])
    if np.linalg.norm(np.array(coords[-1]) - np.array(clean[-1])) > 0.001:
        if np.linalg.norm(np.array(coords[-1]) - np.array(clean[-1])) < min_dist:
            clean[-1] = coords[-1]
        else:
            clean.append(coords[-1])
    return clean


def merge_segments_average(segments):
    merged = [s.copy() for s in segments]
    something_changed = True
    TOLERANCE = 0.5
    while something_changed:
        something_changed = False
        i = 0
        while i < len(merged):
            j = 0
            merged_in_this_step = False
            while j < len(merged):
                if i == j: j += 1; continue
                seg_a = merged[i];
                seg_b = merged[j]
                end_a = np.array(seg_a['points'][-1]);
                start_b = np.array(seg_b['points'][0])
                if np.linalg.norm(end_a - start_b) < TOLERANCE:
                    mid_point = (end_a + start_b) / 2.0
                    new_points = seg_a['points'][:-1] + [tuple(mid_point)] + seg_b['points'][1:]
                    new_id = f"{seg_a['id']} + {seg_b['id']}"
                    merged[i]['points'] = new_points;
                    merged[i]['id'] = new_id
                    del merged[j]
                    something_changed = True;
                    merged_in_this_step = True;
                    break
                j += 1
            if not merged_in_this_step: i += 1
    return merged


def clip_polyline_by_distance(coords, keep_length=25.0):
    if not coords or len(coords) < 2: return coords
    total_dist = 0.0
    for i in range(len(coords) - 1, 0, -1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i - 1])
        dist = np.linalg.norm(p1 - p2)
        if total_dist + dist >= keep_length:
            remaining = keep_length - total_dist
            ratio = remaining / dist
            new_start_x = p1[0] + (p2[0] - p1[0]) * ratio
            new_start_y = p1[1] + (p2[1] - p1[1]) * ratio
            return [(new_start_x, new_start_y)] + coords[i:]
        total_dist += dist
    return coords


def calculate_radius(tck, u_val):
    dx, dy = splev(u_val, tck, der=1)
    ddx, ddy = splev(u_val, tck, der=2)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx ** 2 + dy ** 2, 1.5)
    if num < 1e-6: return float('inf'), (0, 0), (0, 0)
    r = 1.0 / (num / den)
    l = np.sqrt(dx ** 2 + dy ** 2)
    s = np.sign(dx * ddy - dy * ddx)
    return r, (-dy * s / l, dx * s / l)


def get_geometric_radius(p1, p2, p3):
    x1, y1 = p1;
    x2, y2 = p2;
    x3, y3 = p3
    a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 1e-9: return float('inf')
    area = np.sqrt(area_sq)
    R = (a * b * c) / (4 * area)
    return R


def get_velocity(radius, mode='clearing'):
    if radius <= 6.0:
        v = 5.0
    elif 6.0 < radius < 25.0:
        v = math.sqrt(4 * radius)
    else:
        v = 10.0
    if mode == 'entering' and radius >= 25.0: v = 13.89
    return v


def calculate_intergreen_time(dist_clearing, dist_entering, r_clearing, r_entering):
    A = 3.0;
    L_VEHICLE = 6.0
    v_clearing = get_velocity(r_clearing, 'clearing')
    v_entering = get_velocity(r_entering, 'entering')
    U = (dist_clearing + L_VEHICLE) / v_clearing
    B = dist_entering / v_entering
    K = A + U - B
    return max(0, math.ceil(K))


def get_short_name(raw_id):
    if " + " in raw_id:
        parts = raw_id.split(" + ")
        try:
            s_id = parts[0].split("_")[-2:];
            e_id = parts[-1].split("_")[-2:]
            return f"{s_id[0]}_{s_id[1]}->{e_id[0]}_{e_id[1]}"
        except:
            return raw_id
    else:
        try:
            s_id = raw_id.split("_")[-2:];
            return f"{s_id[0]}_{s_id[1]}"
        except:
            return raw_id


def calculate_junction_data(jdata):
    merged_paths = merge_segments_average(jdata['segments'])
    colors = plt.cm.tab20(np.linspace(0, 1, len(merged_paths)))
    lines_data = [];
    all_x, all_y = [], []
    if 'shape' in jdata and jdata['shape']:
        all_x.extend([p[0] for p in jdata['shape']]);
        all_y.extend([p[1] for p in jdata['shape']])
    for idx, path_data in enumerate(merged_paths):
        coords = remove_close_points(path_data['points'], min_dist=0.8)
        coords_np = np.array(coords)
        if len(coords_np) > 0: all_x.extend(coords_np[:, 0]); all_y.extend(coords_np[:, 1])
        color = colors[idx]
        short_name = get_short_name(path_data['id'])
        res = {'id': short_name, 'color': color, 'x_smooth': [], 'y_smooth': [],
               'radius_data': None, 'line_obj': None, 'min_r': float('inf'), 'start_vec': None,
               'end_vec': None}
        is_spline = False
        if len(coords_np) > 2:
            try:
                deg = 3 if len(coords_np) > 3 else 2
                tck, u = splprep([coords_np[:, 0], coords_np[:, 1]], s=0.014, k=deg)
                u_new = np.linspace(0, 1, 400)
                x_s, y_s = splev(u_new, tck)
                res['x_smooth'] = x_s;
                res['y_smooth'] = y_s
                res['line_obj'] = LineString(list(zip(x_s, y_s)))
                is_spline = True
                points_smooth = np.column_stack((x_s, y_s))
                step = 5;
                min_r = float('inf');
                best_idx = -1
                for k in range(step, len(points_smooth) - step):
                    p1 = points_smooth[k - step];
                    p2 = points_smooth[k];
                    p3 = points_smooth[k + step]
                    r_geo = get_geometric_radius(p1, p2, p3)
                    if 2.0 < r_geo < 150.0:
                        if r_geo < min_r: min_r = r_geo; best_idx = k
                res['min_r'] = min_r
                if best_idx != -1:
                    px, py = points_smooth[best_idx]
                    p1 = points_smooth[best_idx - step];
                    p3 = points_smooth[best_idx + step]
                    dx = p3[0] - p1[0];
                    dy = p3[1] - p1[1];
                    l = np.sqrt(dx * dx + dy * dy)
                    if l > 0:
                        v1x, v1y = p2[0] - p1[0], p2[1] - p1[1];
                        v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
                        cross = v1x * v2y - v1y * v2x
                        sign = -1 if cross > 0 else 1
                        nx = -dy * sign / l;
                        ny = dx * sign / l
                        res['radius_data'] = (min_r, px, py, nx, ny)
                # Vektorok az ikonhoz
                res['start_vec'] = (points_smooth[0], points_smooth[10])
                res['end_vec'] = (points_smooth[-10], points_smooth[-1])
            except:
                pass
        if not is_spline:
            res['x_smooth'] = coords_np[:, 0];
            res['y_smooth'] = coords_np[:, 1]
            res['line_obj'] = LineString(coords)
            if len(coords) >= 2:
                res['start_vec'] = (coords[0], coords[1])
                res['end_vec'] = (coords[-2], coords[-1])
        lines_data.append(res)
    matrix_res = [];
    intersections = [];
    n = len(lines_data)
    for i in range(n):
        row = [];
        l1 = lines_data[i]['line_obj'];
        r1 = lines_data[i]['min_r']
        for j in range(n):
            if i == j:
                row.append(None)
            else:
                l2 = lines_data[j]['line_obj'];
                r2 = lines_data[j]['min_r'];
                val = None
                if l1.intersects(l2):
                    intersect = l1.intersection(l2)
                    found_pts = []
                    if isinstance(intersect, Point):
                        found_pts.append(intersect)
                    elif isinstance(intersect, MultiPoint):
                        found_pts.extend(intersect.geoms)
                    valid_pts = []
                    for p in found_pts:
                        if l1.project(p) > 0.2: valid_pts.append(p)
                    if valid_pts:
                        valid_pts.sort(key=lambda p: l1.project(p))
                        unique_pts = [valid_pts[0]]
                        for k in range(1, len(valid_pts)):
                            if valid_pts[k].distance(unique_pts[-1]) > 1.0: unique_pts.append(
                                valid_pts[k])
                        dist_c = l1.project(unique_pts[0]);
                        dist_e = l2.project(unique_pts[0])
                        K_val = calculate_intergreen_time(dist_c, dist_e, r1, r2)
                        val = (K_val, unique_pts[0].x, unique_pts[0].y)
                        for pt in unique_pts:
                            d_c = l1.project(pt);
                            d_e = l2.project(pt)
                            k_pt = calculate_intergreen_time(d_c, d_e, r1, r2)
                            intersections.append(
                                {'pos': (pt.x, pt.y), 'val': k_pt, 'idx_i': i, 'idx_j': j})
                row.append(val)
        matrix_res.append(row)
    return {'lines': lines_data, 'matrix': matrix_res, 'intersections': intersections,
            'bounds': (all_x, all_y), 'jdata': jdata}


# --- 3. PHASE DESIGNER ---

class PhaseDesignDialog:
    def __init__(self, parent, data):
        self.top = tk.Toplevel(parent)
        self.top.title("Fázistervező - Tiltások és Fázisok")
        self.top.geometry("1400x900")

        self.data = data
        self.n = len(data['lines'])

        # Kezdeti konfliktus mátrix (Auto: Ha van metszés = True (Tiltott))
        self.conflict_matrix = np.zeros((self.n, self.n), dtype=bool)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    if data['matrix'][i][j] is not None:
                        # Ha van közbenső idő, akkor geometriailag metszik egymást -> Konfliktus
                        self.conflict_matrix[i, j] = True
                        self.conflict_matrix[j, i] = True  # Szimmetria (bár a K idő nem az)

        # UI Layout
        # Bal oldal: Tiltásmátrix
        # Jobb oldal: Fázisok

        left_frame = tk.Frame(self.top);
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right_frame = tk.Frame(self.top);
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Bal oldal: Mátrix ---
        tk.Label(left_frame, text="Tiltásmátrix (Kattintással módosítható)",
                 font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(left_frame, text="Piros = Tiltott (Ütközik) | Zöld = Engedélyezett").pack(pady=2)

        self.fig_matrix = Figure(figsize=(8, 8))
        self.ax_matrix = self.fig_matrix.add_subplot(111)
        self.canvas_matrix = FigureCanvasTkAgg(self.fig_matrix, master=left_frame)
        self.canvas_matrix.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_matrix.mpl_connect("button_press_event", self.on_matrix_click)

        self.draw_conflict_matrix()

        # Gomb a generáláshoz
        tk.Button(left_frame, text="FÁZISOK GENERÁLÁSA >>", command=self.generate_phases,
                  font=("Arial", 14, "bold"), bg="#dddddd", height=2).pack(fill=tk.X, pady=20)

        # --- Jobb oldal: Eredmények ---
        tk.Label(right_frame, text="Lehetséges Fázisok (Maximal Cliques)",
                 font=("Arial", 12, "bold")).pack(pady=5)

        self.scroll_canvas = tk.Canvas(right_frame)
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.result_frame = tk.Frame(self.scroll_canvas)

        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.create_window((0, 0), window=self.result_frame, anchor="nw")
        self.result_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(
            scrollregion=self.scroll_canvas.bbox("all")))

    def draw_conflict_matrix(self):
        self.ax_matrix.clear()
        self.ax_matrix.axis('off')

        cell_colors = [['white' for _ in range(self.n)] for _ in range(self.n)]
        cell_text = [['' for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    cell_colors[i][j] = "black"
                elif self.conflict_matrix[i, j]:
                    cell_colors[i][j] = "#ffcccc"  # Piros (Tiltott)
                    cell_text[i][j] = "X"
                else:
                    cell_colors[i][j] = "#ccffcc"  # Zöld (Mehet)

        headers = [str(i + 1) for i in range(self.n)]
        table = self.ax_matrix.table(
            cellText=cell_text, cellColours=cell_colors,
            rowLabels=headers, colLabels=headers,
            loc='center', cellLoc='center', bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        self.canvas_matrix.draw()
        self.table_obj = table  # Hogy a klikknél elérjük

    def on_matrix_click(self, event):
        if event.inaxes == self.ax_matrix:
            for (row, col), cell in self.table_obj.get_celld().items():
                if cell.contains(event)[0]:
                    r = row - 1;
                    c = col  # Fejléc miatt
                    if 0 <= r < self.n and 0 <= c < self.n and r != c:
                        # Toggle state
                        new_state = not self.conflict_matrix[r, c]
                        self.conflict_matrix[r, c] = new_state
                        self.conflict_matrix[c, r] = new_state  # Szimmetrikus legyen
                        self.draw_conflict_matrix()
                    return

    def get_arrow_type(self, idx):
        """Meghatározza a nyíl típusát (egyenes, balra, jobbra) a geometria alapján."""
        line = self.data['lines'][idx]
        if not line['start_vec'] or not line['end_vec']: return 'straight', 0

        p1_start = line['start_vec'][0];
        p2_start = line['start_vec'][1]
        p1_end = line['end_vec'][0];
        p2_end = line['end_vec'][1]

        # Vektorok szöge
        angle_start = math.atan2(p2_start[1] - p1_start[1], p2_start[0] - p1_start[0])
        angle_end = math.atan2(p2_end[1] - p1_end[1], p2_end[0] - p1_end[0])

        diff = math.degrees(angle_end - angle_start)
        # Normalizálás -180..180
        while diff > 180: diff -= 360
        while diff < -180: diff += 360

        # Beérkezési szög (honnan jön a csomópontba)
        # Egyszerűsítés: 4 fő irány (É, K, D, NY)
        # A belépési szög alapján forgatjuk az ikont
        entry_angle = math.degrees(angle_start)

        arrow_type = 'straight'
        if diff > 45:
            arrow_type = 'left'
        elif diff < -45:
            arrow_type = 'right'

        return arrow_type, entry_angle

    def draw_phase_icon(self, ax, indices):
        """Kirajzol egy kis fázisdiagramot nyilakkal."""
        ax.set_xlim(-1.5, 1.5);
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        ax.set_aspect('equal')

        # Alapkereszt (halványan)
        ax.plot([-1, 1], [0, 0], color='#eeeeee', lw=1)
        ax.plot([0, 0], [-1, 1], color='#eeeeee', lw=1)

        for idx in indices:
            atype, angle = self.get_arrow_type(idx)
            color = self.data['lines'][idx]['color']

            # Koordináta rendszer forgatása a belépési szög szerint
            # Alap: Délről jön (270 fok), Északra megy
            # Belépés: angle. Ha angle=90 (Keletről jön?), akkor elforgatjuk.
            # Egyszerűsítés: a nyíl mindig lentről indul a rajzon, és a kanyarodás mutatja az irányt?
            # VAGY: Tényleges irányokat rajzolunk. A képed alapján (image_84d274) a nyilak a valós irányokat mutatják.

            # Vektor hossza
            L = 1.0
            rad = math.radians(angle)

            # Start pont (a kör szélén)
            # A jármű a középpont felé tart
            sx = -math.cos(rad) * 0.8
            sy = -math.sin(rad) * 0.8

            # End pont
            if atype == 'straight':
                ex = math.cos(rad) * 0.8
                ey = math.sin(rad) * 0.8
                style = "Simple, tail_width=1.5, head_width=6, head_length=6"
                conn = "arc3,rad=0"
            elif atype == 'left':
                # Balra kanyar (90 fok plusz)
                end_rad = rad + math.pi / 2
                ex = math.cos(end_rad) * 0.8
                ey = math.sin(end_rad) * 0.8
                style = "Simple, tail_width=1.5, head_width=6, head_length=6"
                conn = "arc3,rad=-0.3"  # Görbület
            else:  # right
                end_rad = rad - math.pi / 2
                ex = math.cos(end_rad) * 0.8
                ey = math.sin(end_rad) * 0.8
                style = "Simple, tail_width=1.5, head_width=6, head_length=6"
                conn = "arc3,rad=0.3"

            arrow = FancyArrowPatch((sx, sy), (ex, ey), connectionstyle=conn,
                                    arrowstyle=style, color=color, alpha=0.9)
            ax.add_patch(arrow)

            # Szám a nyíl tövéhez
            ax.text(sx * 1.2, sy * 1.2, str(idx + 1), fontsize=8, ha='center', va='center',
                    color=color, fontweight='bold')

    def generate_phases(self):
        # 1. Gráf építése (Complement Graph)
        # Ahol NINCS konfliktus, ott van él.
        # Klikk a komplementerben = Független halmaz az eredetiben

        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not self.conflict_matrix[i, j]:  # Ha NINCS konfliktus
                    G.add_edge(i, j)

        # 2. Maximális Klikkek keresése
        cliques = list(nx.find_cliques(G))

        # Rendezzük méret szerint (legnagyobbak elöl)
        cliques.sort(key=len, reverse=True)

        # 3. Megjelenítés a jobb oldalon
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        for i, clique in enumerate(cliques):
            # Csak azokat mutassuk, ahol van legalább 1 elem (bár klikk def. szerint van)
            if not clique: continue

            # Keret
            f = tk.LabelFrame(self.result_frame, text=f"Fázis {i + 1} (Méret: {len(clique)})",
                              font=("Arial", 10, "bold"), pady=5, padx=5)
            f.pack(fill=tk.X, padx=5, pady=5)

            # Kis Matplotlib Figure a fázisnak
            fig = Figure(figsize=(3, 3), dpi=100)
            ax = fig.add_subplot(111)

            clique_sorted = sorted(clique)
            self.draw_phase_icon(ax, clique_sorted)

            canvas = FigureCanvasTkAgg(fig, master=f)
            canvas.get_tk_widget().pack(side=tk.LEFT)

            # Szöveges lista
            list_text = "\n".join(
                [f"{idx + 1}. {self.data['lines'][idx]['id']}" for idx in clique_sorted])
            tk.Label(f, text=list_text, justify=tk.LEFT, font=("Consolas", 9)).pack(side=tk.LEFT,
                                                                                    padx=10)


# --- 4. GUI (Main) ---

class JunctionApp:
    def __init__(self, root, parser):
        self.root = root
        self.root.title("SUMO Intersection Analyzer - High Res")
        self.root.geometry("1600x1000")
        self.style = ttk.Style();
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        self.parser = parser;
        self.junctions = parser.junctions;
        self.current_idx = 0

        top_frame = ttk.Frame(root);
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Button(top_frame, text="<< Előző", command=self.prev_junction).pack(side=tk.LEFT,
                                                                                padx=5)
        ttk.Button(top_frame, text="Következő >>", command=self.next_junction).pack(side=tk.LEFT,
                                                                                    padx=5)
        self.combo_values = [f"{i + 1}. {j['id']}" for i, j in enumerate(self.junctions)]
        self.combo_var = tk.StringVar()
        self.combo = ttk.Combobox(top_frame, textvariable=self.combo_var, values=self.combo_values,
                                  state="readonly", width=25)
        self.combo.pack(side=tk.LEFT, padx=20)
        self.combo.bind("<<ComboboxSelected>>", self.jump_to_junction)
        if self.combo_values: self.combo.current(0)
        ttk.Label(top_frame, text="Zoom:").pack(side=tk.LEFT, padx=(15, 2))
        ttk.Button(top_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(top_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(top_frame, text="Rst", command=self.reset_view, width=4).pack(side=tk.LEFT,
                                                                                 padx=1)

        # ÚJ GOMB: FÁZISTERVEZŐ
        ttk.Button(top_frame, text="Fázistervező & Tiltások",
                   command=self.open_phase_designer).pack(side=tk.RIGHT, padx=5)

        ttk.Button(top_frame, text="PDF Riport", command=self.export_pdf_dialog).pack(side=tk.RIGHT,
                                                                                      padx=5)
        ttk.Button(top_frame, text="Mátrix Export", command=self.export_csv).pack(side=tk.RIGHT,
                                                                                  padx=5)
        ttk.Button(top_frame, text="Címkék Ki/Be", command=self.toggle_annotations).pack(
            side=tk.RIGHT, padx=10)

        self.fig = Figure(figsize=(12, 8));
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax = self.fig.add_axes([0.05, 0.05, 0.65, 0.90])
        self.ax_check = self.fig.add_axes([0.72, 0.55, 0.26, 0.40]);
        self.ax_check.set_axis_off()
        self.ax_control = self.fig.add_axes([0.72, 0.32, 0.26, 0.20]);
        self.ax_control.set_axis_off()
        self.ax_table = self.fig.add_axes([0.72, 0.02, 0.26, 0.28]);
        self.ax_table.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.cursor = Cursor(self.ax, useblit=True, color='gray', linewidth=1, linestyle='--')

        self.interactive_items = [];
        self.lines_map = {};
        self.check_buttons = None
        self.all_annotations = [];
        self.annotations_visible = True;
        self.highlighted_ids = None
        self.result_table = None;
        self.control_table = None;
        self.matrix_mask = None
        self.ids_for_table = [];
        self.current_limits = None;
        self.current_data_cache = None

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.plot_current()

    # --- ÚJ METÓDUS ---
    def open_phase_designer(self):
        if not self.current_data_cache: return
        PhaseDesignDialog(self.root, self.current_data_cache)

    # ... (A TÖBBI METÓDUS VÁLTOZATLAN) ...
    def next_junction(self):
        if not self.junctions: return
        self.current_idx = (self.current_idx + 1) % len(self.junctions)
        self.combo.current(self.current_idx);
        self.matrix_mask = None;
        self.current_limits = None;
        self.plot_current()

    def prev_junction(self):
        if not self.junctions: return
        self.current_idx = (self.current_idx - 1) % len(self.junctions)
        self.combo.current(self.current_idx);
        self.matrix_mask = None;
        self.current_limits = None;
        self.plot_current()

    def jump_to_junction(self, event):
        idx = self.combo.current()
        if idx != -1: self.current_idx = idx; self.matrix_mask = None; self.current_limits = None; self.plot_current()

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
        for artist in self.all_annotations: artist.set_visible(self.annotations_visible)
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

    def export_csv(self):
        if not self.current_data_cache: return
        filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                                filetypes=[("CSV file", "*.csv")])
        if not filename: return
        lines = self.current_data_cache['lines'];
        matrix = self.current_data_cache['matrix']
        ids = [l['id'] for l in lines];
        n = len(ids)
        mask = self.matrix_mask if self.matrix_mask is not None else np.ones((n, n), dtype=bool)
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["From/To"] + ids)
                for i in range(n):
                    row = [ids[i]]
                    for j in range(n):
                        val = ""
                        if i != j and mask[i, j] and matrix[i][j]: val = f"{matrix[i][j][0]:.2f}"
                        row.append(val)
                    writer.writerow(row)
            messagebox.showinfo("Info", "Mentve.")
        except Exception as e:
            messagebox.showerror("Hiba", str(e))

    def export_pdf_dialog(self):
        dialog = ExportSelectionDialog(self.root, self.junctions)
        self.root.wait_window(dialog.top)
        if dialog.result is not None: self.export_pdf_process(dialog.result)

    def export_pdf_process(self, selected_indices):
        filename = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                filetypes=[("PDF file", "*.pdf")])
        if not filename: return
        try:
            from matplotlib.figure import Figure
            with PdfPages(filename) as pdf:
                total = len(selected_indices)
                for k, idx in enumerate(selected_indices):
                    jdata = self.junctions[idx];
                    data = calculate_junction_data(jdata)
                    fig_pdf = Figure(figsize=(8.27, 11.69))
                    txt = f"Oldal {k + 1} / {total} | Készítette: BME Traffic Lab | http://traffic.bme.hu"
                    fig_pdf.text(0.5, 0.97, txt, ha='center', fontsize=10, color='gray')
                    fig_pdf.suptitle(f"Csomópont: {jdata['id']}", fontsize=14, fontweight='bold',
                                     y=0.95)
                    ax_plot = fig_pdf.add_axes([0.05, 0.45, 0.75, 0.45])
                    self._draw_pdf_plot(ax_plot, data)
                    legend_handles = [];
                    legend_labels = []
                    for line in data['lines']:
                        l, = ax_plot.plot([], [], color=line['color'], lw=2, label=line['id'])
                        legend_handles.append(l);
                        legend_labels.append(line['id'])
                    ax_plot.legend(handles=legend_handles, labels=legend_labels,
                                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5,
                                   borderaxespad=0.)
                    ax_tbl = fig_pdf.add_axes([0.1, 0.05, 0.8, 0.35]);
                    ax_tbl.axis('off')
                    self._draw_pdf_table(ax_tbl, data)
                    pdf.savefig(fig_pdf)
                    fig_pdf.clf();
                    del fig_pdf;
                    gc.collect()
                    print(f"Export: {k + 1}/{total}")
                    self.root.update()
            messagebox.showinfo("Kész", "PDF generálva.")
        except Exception as e:
            messagebox.showerror("Hiba", str(e)); import traceback; traceback.print_exc()

    def _draw_pdf_plot(self, ax, data):
        if 'shape' in data['jdata'] and data['jdata']['shape']:
            ax.add_patch(
                MplPolygon(data['jdata']['shape'], facecolor='#e6e6e6', edgecolor='#999999',
                           alpha=0.5, zorder=-20))
        if 'incoming_lanes' in data['jdata']:
            for lane_shape in data['jdata']['incoming_lanes']:
                if not lane_shape: continue
                clipped = clip_polyline_by_distance(lane_shape, keep_length=25.0)
                ls = LineString(clipped);
                poly_geo = ls.buffer(1.6, cap_style=2)
                if not poly_geo.is_empty:
                    xx, yy = poly_geo.exterior.xy
                    ax.add_patch(
                        MplPolygon(list(zip(xx, yy)), facecolor='#d9d9d9', edgecolor='#b3b3b3',
                                   alpha=0.6, zorder=-15))
        for line in data['lines']:
            ax.plot(line['x_smooth'], line['y_smooth'], '-', lw=2, color=line['color'], alpha=0.8)
            if len(line['x_smooth']) > 10:
                ax.arrow(line['x_smooth'][-10], line['y_smooth'][-10],
                         line['x_smooth'][-1] - line['x_smooth'][-10],
                         line['y_smooth'][-1] - line['y_smooth'][-10],
                         head_width=0.8, color=line['color'], length_includes_head=True)
            if line['radius_data']:
                min_r, px, py, nx, ny = line['radius_data'];
                lx, ly = px + nx * 3, py + ny * 3
                ax.annotate(f"R={min_r:.1f}m", xy=(px, py), xytext=(lx, ly),
                            arrowprops=dict(arrowstyle="->", color=line['color'], linestyle='--'),
                            fontsize=5, ha='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=line['color'],
                                      alpha=0.9))
        for inter in data['intersections']:
            val, mx, my = inter['val'], inter['pos'][0], inter['pos'][1]
            color = data['lines'][inter['idx_i']]['color']
            ax.plot(mx, my, 'ko', mfc='none', mew=1.0, ms=5)
            ax.annotate(f"K={val}s", xy=(mx, my), xytext=(-15, 15), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.5), color='black',
                        fontsize=5, ha='right',
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', alpha=0.7))
        ax.plot(0, 0, 'k+', markersize=15, label='KÖZÉPPONT')
        all_x, all_y = data['bounds']
        if all_x:
            mnx, mxx = min(all_x), max(all_x);
            mny, mxy = min(all_y), max(all_y);
            mgx = (mxx - mnx) * 0.05;
            mgy = (mxy - mny) * 0.05
            ax.set_xlim(mnx - mgx, mxx + mgx);
            ax.set_ylim(mny - mgy, mxy + mgy)
        ax.grid(True, linestyle=':', alpha=0.5);
        ax.set_aspect('equal')

    def _draw_pdf_table(self, ax, data):
        ids = [l['id'] for l in data['lines']];
        n = len(ids);
        matrix = data['matrix'];
        cell_text = []
        for i in range(n):
            row = []
            for j in range(n):
                if matrix[i][j]:
                    row.append(f"{matrix[i][j][0]}")
                elif i == j:
                    row.append("-")
                else:
                    row.append("")
            cell_text.append(row)
        table = ax.table(cellText=cell_text, rowLabels=ids, colLabels=ids, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False);
        table.set_fontsize(5);
        table.scale(1, 1.2)

    def update_control_table(self):
        self.ax_control.clear();
        self.ax_control.set_axis_off()
        n = len(self.ids_for_table)
        if n == 0: return
        if self.matrix_mask is None or self.matrix_mask.shape != (n, n): self.matrix_mask = np.ones(
            (n, n), dtype=bool)
        cell_text = [['' for _ in range(n)] for _ in range(n)]
        cell_colors = [['white' for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    cell_colors[i][j] = "black"
                else:
                    cell_colors[i][j] = "#90ee90" if self.matrix_mask[i, j] else "#ffcccb"
        headers = [str(i + 1) for i in range(n)]
        self.ax_control.text(0.5, 1.05, "Számítás Kijelölése", ha='center',
                             transform=self.ax_control.transAxes, fontsize=9, fontweight='bold')
        self.control_table = self.ax_control.table(cellText=cell_text, cellColours=cell_colors,
                                                   rowLabels=headers, colLabels=headers,
                                                   loc='center', cellLoc='center',
                                                   bbox=[0, 0, 1, 1])
        self.control_table.auto_set_font_size(False);
        self.control_table.set_fontsize(5)

    def update_matrix_table(self):
        self.ax_table.clear();
        self.ax_table.set_axis_off();
        self.result_table = None
        if not self.ids_for_table: return
        n = len(self.ids_for_table)
        if not self.current_data_cache: return
        matrix = self.current_data_cache['matrix']
        cell_text = [];
        cell_colors = []
        for i in range(n):
            row_data = [];
            row_colors = []
            for j in range(n):
                val = "";
                color = "white"
                if i == j:
                    color = "black"
                else:
                    if self.matrix_mask[i, j] and matrix[i][j]:
                        val = f"{matrix[i][j][0]}"
                    elif not self.matrix_mask[i, j]:
                        color = "#f2f2f2"
                row_data.append(val);
                row_colors.append(color)
            cell_text.append(row_data);
            cell_colors.append(row_colors)
        headers = [str(i + 1) for i in range(n)]
        self.ax_table.text(0.5, 1.05, "Eredmény [s]", ha='center',
                           transform=self.ax_table.transAxes, fontsize=9, fontweight='bold')
        self.result_table = self.ax_table.table(cellText=cell_text, cellColours=cell_colors,
                                                rowLabels=headers, colLabels=headers, loc='center',
                                                cellLoc='center', bbox=[0, 0, 1, 1])
        self.result_table.auto_set_font_size(False);
        self.result_table.set_fontsize(5);
        self.result_table.scale(1, 1.1)

    def plot_current(self):
        self.ax.clear();
        self.ax_check.clear();
        self.ax_check.set_axis_off()
        self.interactive_items = [];
        self.lines_map = {};
        self.current_line_objects = []
        self.all_annotations = [];
        self.ids_for_table = [];
        self.highlighted_ids = None
        if not self.junctions: self.ax.text(0.5, 0.5, "Nincs adat.",
                                            ha='center'); self.canvas.draw(); return
        data = calculate_junction_data(self.junctions[self.current_idx])
        self.current_data_cache = data
        self.ax.set_title(f"Csomópont: {data['jdata']['id']}", fontsize=11, fontweight='bold')
        if 'shape' in data['jdata'] and data['jdata']['shape']:
            self.ax.add_patch(
                MplPolygon(data['jdata']['shape'], facecolor='#e6e6e6', edgecolor='#999999',
                           alpha=0.5, zorder=-20))
        if 'incoming_lanes' in data['jdata']:
            for lane_shape in data['jdata']['incoming_lanes']:
                if not lane_shape: continue
                clipped = clip_polyline_by_distance(lane_shape, keep_length=25.0)
                ls = LineString(clipped);
                poly_geo = ls.buffer(1.6, cap_style=2)
                if not poly_geo.is_empty:
                    if poly_geo.geom_type == 'MultiPolygon':
                        for p in poly_geo.geoms: xx, yy = p.exterior.xy; self.ax.add_patch(
                            MplPolygon(list(zip(xx, yy)), facecolor='#d9d9d9', edgecolor='#b3b3b3',
                                       alpha=0.6, zorder=-15))
                    else:
                        xx, yy = poly_geo.exterior.xy; self.ax.add_patch(
                            MplPolygon(list(zip(xx, yy)), facecolor='#d9d9d9', edgecolor='#b3b3b3',
                                       alpha=0.6, zorder=-15))
        colors = plt.cm.tab20(np.linspace(0, 1, len(data['lines'])))
        labels_for_checkbox = []
        all_x, all_y = data['bounds']
        for i, line_data in enumerate(data['lines']):
            display_id = line_data['id'];
            checkbox_label = f"{i + 1}. {display_id}"
            labels_for_checkbox.append(checkbox_label);
            self.ids_for_table.append(display_id)
            current_artists = []
            l, = self.ax.plot(line_data['x_smooth'], line_data['y_smooth'], '-', lw=2,
                              color=line_data['color'], alpha=0.7, zorder=1)
            current_artists.append(l)
            if len(line_data['x_smooth']) > 10:
                arr = self.ax.arrow(line_data['x_smooth'][-10], line_data['y_smooth'][-10],
                                    line_data['x_smooth'][-1] - line_data['x_smooth'][-10],
                                    line_data['y_smooth'][-1] - line_data['y_smooth'][-10],
                                    head_width=0.8, color=line_data['color'],
                                    length_includes_head=True, zorder=1)
                current_artists.append(arr)
            if line_data['radius_data']:
                min_r, px, py, nx, ny = line_data['radius_data'];
                lx, ly = px + nx * 3.0, py + ny * 3.0
                ann = self.ax.annotate(f"R={min_r:.1f}m", xy=(px, py), xytext=(lx, ly),
                                       arrowprops=dict(arrowstyle="->", color=line_data['color'],
                                                       linestyle='--'), fontsize=6, ha='center',
                                       zorder=2, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                                           ec=line_data['color'], alpha=0.9))
                current_artists.append(ann);
                self.all_annotations.append(ann)
            self.lines_map[checkbox_label] = current_artists;
            self.current_line_objects.append(line_data['line_obj'])
        for inter in data['intersections']:
            val, mx, my = inter['val'], inter['pos'][0], inter['pos'][1]
            i, j = inter['idx_i'], inter['idx_j']
            p_art, = self.ax.plot(mx, my, 'ko', mfc='none', mew=1.5, ms=8, picker=True)
            lbl1 = self.ax.annotate(f"K={val}s", xy=(mx, my), xytext=(-20, 20),
                                    textcoords='offset points',
                                    arrowprops=dict(arrowstyle="-", color=colors[i], lw=0.5),
                                    color='white', fontweight='bold', fontsize=5, ha='right',
                                    va='bottom', zorder=15,
                                    bbox=dict(boxstyle="round,pad=0.15", fc=colors[i], ec='none',
                                              alpha=0.85))
            self.all_annotations.append(lbl1)
            self.interactive_items.append(
                {'point': p_art, 'labels': [lbl1], 'colors': [colors[i]], 'pos': (mx, my)})
            key1 = labels_for_checkbox[i];
            key2 = labels_for_checkbox[j]
            if key1 in self.lines_map: self.lines_map[key1].extend([p_art, lbl1])
            if key2 in self.lines_map: self.lines_map[key2].extend([p_art])
        if labels_for_checkbox:
            self.ax_check.set_facecolor('white');
            self.check_buttons = CheckButtons(self.ax_check, labels_for_checkbox,
                                              [True] * len(labels_for_checkbox))
            for i, label in enumerate(self.check_buttons.labels):
                label.set_fontsize(8);
                label.set_color(colors[i]);
                label.set_fontweight('bold')
            if hasattr(self.check_buttons, 'rectangles'):
                for i, rect in enumerate(self.check_buttons.rectangles): rect.set_facecolor(
                    colors[i]); rect.set_alpha(0.5)
            self.check_buttons.on_clicked(self.toggle_visibility)
        self.update_control_table();
        self.update_matrix_table()
        if not self.annotations_visible:
            for artist in self.all_annotations: artist.set_visible(False)
        self.ax.plot(0, 0, 'k+', markersize=20, markeredgewidth=2, label='KÖZÉPPONT', zorder=5)
        if self.current_limits:
            self.ax.set_xlim(self.current_limits[0]);
            self.ax.set_ylim(self.current_limits[1])
        elif all_x:
            mnx, mxx = min(all_x), max(all_x);
            mny, mxy = min(all_y), max(all_y);
            mgx = (mxx - mnx) * 0.05;
            mgy = (mxy - mny) * 0.05
            self.ax.set_xlim(mnx - mgx, mxx + mgx);
            self.ax.set_ylim(mny - mgy, mxy + mgy)
        self.ax.grid(True, linestyle=':', alpha=0.5);
        self.ax.set_aspect('equal');
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax_control and self.control_table:
            for (row, col), cell in self.control_table.get_celld().items():
                if cell.contains(event)[0]:
                    data_r = row - 1;
                    data_c = col;
                    n = len(self.ids_for_table)
                    if 0 <= data_r < n and 0 <= data_c < n and data_r != data_c:
                        self.matrix_mask[data_r, data_c] = not self.matrix_mask[data_r, data_c]
                        self.update_control_table();
                        self.update_matrix_table();
                        self.fig.canvas.draw_idle();
                        return

    def on_hover(self, event):
        if (event.inaxes == self.ax_table and self.result_table) or (
                event.inaxes == self.ax_control and self.control_table):
            target_table = self.result_table if event.inaxes == self.ax_table else self.control_table
            found = None
            for (r, c), cell in target_table.get_celld().items():
                if cell.contains(event)[0]: found = (r, c); break
            if found:
                dr, dc = found[0] - 1, found[1]
                if 0 <= dr < len(self.ids_for_table) and 0 <= dc < len(self.ids_for_table):
                    self.update_highlights({self.ids_for_table[dr], self.ids_for_table[dc]});
                    return
        if self.highlighted_ids is not None: self.update_highlights(None)
        if event.inaxes == self.ax:
            mx, my = event.xdata, event.ydata
            if mx is None: return
            closest = None;
            min_dist = float('inf')
            for item in self.interactive_items:
                if not item['point'].get_visible(): continue
                px, py = item['pos'];
                dist = math.hypot(px - mx, py - my)
                if dist < 1.5 and dist < min_dist: min_dist = dist; closest = item
            needs_redraw = False
            for item in self.interactive_items:
                is_high = (item == closest);
                pt = item['point'];
                curr = pt.get_markerfacecolor()
                if (is_high and curr != 'red') or (not is_high and curr == 'red'):
                    needs_redraw = True
                    if is_high:
                        pt.set_markerfacecolor('red');
                        pt.set_markersize(12);
                        pt.set_zorder(100)
                        for i, lbl in enumerate(item['labels']):
                            lbl.set_zorder(101);
                            p = lbl.get_bbox_patch()
                            p.set_facecolor(item['colors'][i]);
                            p.set_linewidth(2);
                            p.set_alpha(1.0);
                            lbl.set_color('white')
                    else:
                        pt.set_markerfacecolor('none');
                        pt.set_markersize(8);
                        pt.set_zorder(10)
                        for i, lbl in enumerate(item['labels']):
                            lbl.set_zorder(15);
                            p = lbl.get_bbox_patch()
                            p.set_facecolor(item['colors'][i]);
                            p.set_linewidth(0);
                            p.set_alpha(0.85);
                            lbl.set_color('white')
            if needs_redraw: self.fig.canvas.draw_idle()

    def update_highlights(self, active_ids):
        if active_ids == self.highlighted_ids: return
        self.highlighted_ids = active_ids
        for label, artists in self.lines_map.items():
            clean_id = label.split(". ")[1];
            is_active = (active_ids is None) or (clean_id in active_ids)
            for artist in artists:
                if isinstance(artist, plt.Annotation) or isinstance(artist, plt.Text):
                    continue
                elif isinstance(artist, MplPolygon):
                    if active_ids is None:
                        artist.set_alpha(0.4); artist.set_linewidth(1)
                    else:
                        if is_active:
                            artist.set_alpha(0.8); artist.set_linewidth(2)
                        else:
                            artist.set_alpha(0.05)
                else:
                    if hasattr(artist, 'set_alpha'): artist.set_alpha(
                        0.8 if (is_active or active_ids is None) else 0.05)
                    if hasattr(artist, 'set_linewidth'): artist.set_linewidth(2 if is_active else 1)
                artist.set_zorder(10 if is_active and active_ids is not None else (
                    1 if isinstance(artist, plt.Line2D) else -5))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk();
    root.withdraw()
    print("Válassz fájlt...")
    fp = filedialog.askopenfilename(filetypes=[("SUMO Network", "*.net.xml"), ("XML", "*.xml")])
    if fp:
        root.deiconify()
        try:
            parser = SumoInternalParser(fp)
            app = JunctionApp(root, parser)
            root.mainloop()
        except Exception as e:
            print(f"Hiba: {e}");
            import traceback;

            traceback.print_exc();
            input("Enter...")