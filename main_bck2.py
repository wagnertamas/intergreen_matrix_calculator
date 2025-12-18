import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
import json
import os
import networkx as nx


# --- 1. PARSER (Változatlan) ---

class SumoInternalParser:
    def __init__(self, file_path):
        print(f"Fájl feldolgozása: {file_path}")
        self.file_path = file_path
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
                jx = float(junc.get("x"))
                jy = float(junc.get("y"))
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


# --- 2. GEOMETRIA (Változatlan) ---

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
        p1 = np.array(coords[i]);
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

    base_colors = plt.cm.tab20(np.linspace(0, 1, len(merged_paths)))
    darker_colors = []
    for c in base_colors:
        darker_colors.append((c[0] * 0.7, c[1] * 0.7, c[2] * 0.7, 1.0))
    colors = darker_colors

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
                if len(points_smooth) > 20:
                    res['start_vec'] = (points_smooth[0], points_smooth[10])
                    res['end_vec'] = (points_smooth[-10], points_smooth[-1])
            except:
                pass
        if not is_spline:
            res['x_smooth'] = coords_np[:, 0];
            res['y_smooth'] = coords_np[:, 1]
            res['line_obj'] = LineString(coords)
            if len(coords) >= 2:
                res['start_vec'] = (coords[0], coords[1]);
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
                sx1, sy1 = lines_data[i]['x_smooth'][0], lines_data[i]['y_smooth'][0]
                sx2, sy2 = lines_data[j]['x_smooth'][0], lines_data[j]['y_smooth'][0]
                dist_start = math.hypot(sx1 - sx2, sy1 - sy2)

                if dist_start < 5.0:
                    row.append(None)
                    continue

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


# --- 3. ÚJ: FÁZISÁTMENET TERVEZŐ ABLAK (Javított Layout) ---

class TransitionPlannerDialog:
    def __init__(self, parent, all_ids, phases, matrix):
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
            mpatches.Patch(color='#f0ad4e', label='Sárga ({AMBER_TIME}s)'),
            mpatches.Patch(color='#e67e22', label=f'Piros-Sárga ({RED_AMBER_TIME}s)'),
            mpatches.Patch(color='#d9534f', label='Piros'),
        ]
        self.ax.legend(handles=patches, loc='upper right', fontsize=9)
        self.canvas.draw()


# --- 4. GUI (Főalkalmazás) ---

class JunctionApp:
    def __init__(self, root, parser):
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
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.saved_settings = json.load(f)
            except Exception as e:
                print(f"Hiba a beállítások betöltésekor: {e}")
                self.saved_settings = {}

    def save_settings(self):
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
                        if matrix[i][j]:
                            txt = f"{matrix[i][j][0]:.0f}"
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


        # --- 5. ÚJ: DINAMIKUS OVERLAP EXPORT (ÁTLAPOLÁSSAL) ---

    def export_all_to_sumo(self):
        """
        Kiexportálja a közlekedési lámpákat SUMO (.add.xml) és JSON (.json) formátumban.
        LOGIKA: Két független idővonalat (leállás és indulás) fésül össze.
        Ez lehetővé teszi, hogy a Piros-Sárga (u) már elkezdődjön, miközben a másik még Sárga (y).
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
        YELLOW_TIME = 3.0  # Mikor vált 'y'-ról 'r'-re a leálló
        RED_YELLOW_TIME = 2.0  # Mennyi idővel a zöld előtt vált 'r'-ről 'u'-ra az induló

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

                # 1. Adatok és Mátrix
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

                # 2. Klikkek
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

                # 3. Ciklus generálása
                for i in range(num_logic_phases):
                    current_phase_idxs = set(cliques[i])
                    next_phase_idxs = set(cliques[(i + 1) % num_logic_phases])

                    # ==========================================
                    # A) ZÖLD FÁZIS (Stabil állapot)
                    # ==========================================
                    state_g = ['r'] * n
                    for lane_idx in range(n):
                        if lane_idx in current_phase_idxs:
                            state_g[lane_idx] = 'G'
                    state_g_str = "".join(state_g)

                    ET.SubElement(tlLogic, "phase", duration=str(MIN_GREEN_TIME),
                                  state=state_g_str)

                    junction_json["phases"].append({
                        "index": sumo_phase_counter,
                        "duration": MIN_GREEN_TIME,
                        "state": state_g_str,
                        "type": "green",
                        "logic_idx": i
                    })
                    junction_json["logic_phases"][str(i)] = sumo_phase_counter
                    sumo_phase_counter += 1

                    # ==========================================
                    # B) ÁTMENET (Dinamikus Overlap)
                    # ==========================================
                    transition_steps = []

                    # Csoportosítás
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

                    # 1. Kiszámoljuk a szükséges átmeneti időt (K)
                    # Ez a mátrix maximuma a leállók és indulók között
                    K_needed = 0.0
                    for stop_idx in stopping:
                        for start_idx in starting:
                            val = data['matrix'][stop_idx][start_idx]
                            if val is not None:
                                if val[0] > K_needed: K_needed = val[0]

                    # Ha nincs konfliktus (K=0) és nincs megálló sáv, akkor nincs átmenet
                    if not stopping and not starting:
                        # Ez ritka, de technikailag lehetséges (pl. ugyanaz a fázis)
                        key = f"{i}->{(i + 1) % num_logic_phases}"
                        junction_json["transitions"][key] = []
                        continue

                    # BIZTONSÁGI KORREKCIÓK:
                    # 1. A sárgát mindenképp le kell játszani (YELLOW_TIME).
                    #    Tehát az átmenet hossza legalább YELLOW_TIME.
                    transition_total_time = max(K_needed, YELLOW_TIME)

                    # ESEMÉNYEK IDŐPONTJAI (timeline):
                    # T=0: Átmenet kezdete (Zöld vége)
                    # T_yellow_end: Mikor vált a sárga pirosra (fix 3.0s)
                    t_yellow_end = YELLOW_TIME

                    # T_redyellow_start: Mikor vált a piros piros-sárgára.
                    # A végcél (transition_total_time) előtt RED_YELLOW_TIME-mal.
                    t_redyellow_start = max(0.0, transition_total_time - RED_YELLOW_TIME)

                    # T_end: Az egész vége
                    t_end = transition_total_time

                    # Összegyűjtjük a vágási pontokat és sorba rendezzük
                    # Pl. K=4 esetén: {0, 3, 2, 4} -> [0, 2, 3, 4]
                    cut_points = sorted(
                        list(set([0.0, t_yellow_end, t_redyellow_start, t_end])))

                    # Csak a 0 és t_end közötti pontok kellenek
                    cut_points = [t for t in cut_points if t >= 0 and t <= t_end]

                    # Generáljuk a szakaszokat a vágási pontok között
                    prev_t = 0.0
                    for t in cut_points:
                        if t <= prev_t: continue  # Duplikációk vagy 0 elkerülése

                        duration = t - prev_t
                        if duration < 0.1: continue  # Túl rövid szakaszok szűrése

                        # Állapot meghatározása a szakasz közepén
                        mid_t = (prev_t + t) / 2.0

                        current_state_chars = ['r'] * n

                        for idx_lane in range(n):
                            if idx_lane in staying:
                                # Aki marad, az végig Zöld
                                current_state_chars[idx_lane] = 'G'

                            elif idx_lane in stopping:
                                # Leálló logika: Sárga amíg el nem éri a limitet, utána Piros
                                if mid_t < t_yellow_end:
                                    current_state_chars[idx_lane] = 'y'
                                else:
                                    current_state_chars[idx_lane] = 'r'

                            elif idx_lane in starting:
                                # Induló logika: Piros amíg el nem éri a felkészülést, utána Piros-Sárga (u)
                                if mid_t < t_redyellow_start:
                                    current_state_chars[idx_lane] = 'r'
                                else:
                                    current_state_chars[idx_lane] = 'u'

                            else:
                                # Aki se nem jön, se nem megy, az marad piros
                                current_state_chars[idx_lane] = 'r'

                        state_str = "".join(current_state_chars)

                        # XML mentés
                        ET.SubElement(tlLogic, "phase", duration=f"{duration:.1f}",
                                      state=state_str)

                        # JSON mentés (Típust becslünk a tartalom alapján)
                        p_type = "transition"
                        if 'y' in state_str and 'u' in state_str:
                            p_type = "overlap_yellow_redyellow"
                        elif 'y' in state_str:
                            p_type = "yellow"
                        elif 'u' in state_str:
                            p_type = "red_yellow"
                        elif 'r' in state_str and 'G' not in state_str:
                            p_type = "all_red"  # (kivéve ha staying van)

                        junction_json["phases"].append({
                            "index": sumo_phase_counter,
                            "duration": float(f"{duration:.1f}"),
                            "state": state_str,
                            "type": p_type,
                            "from_logic": i,
                            "to_logic": (i + 1) % num_logic_phases
                        })
                        transition_steps.append(sumo_phase_counter)
                        sumo_phase_counter += 1

                        prev_t = t

                    # Átmenet rögzítése a JSON-ban
                    key = f"{i}->{(i + 1) % num_logic_phases}"
                    junction_json["transitions"][key] = transition_steps

                json_export_data[jid] = junction_json

            # --- MENTÉS ---
            tree = ET.ElementTree(root_xml)
            self.indent(root_xml)
            tree.write(filename_xml, encoding="utf-8", xml_declaration=True)

            with open(filename_json, 'w', encoding='utf-8') as f:
                json.dump(json_export_data, f, indent=4)

            messagebox.showinfo("Siker", f"Dinamikus Overlap Export Kész!\nXML: {filename_xml}")

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

if __name__ == "__main__":
    root = tk.Tk()
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
            print(f"Hiba: {e}")
            import traceback

            traceback.print_exc()
            input("Enter...")