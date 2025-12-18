import numpy as np
import math
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, MultiPoint
import matplotlib.pyplot as plt

def remove_close_points(coords, min_dist=0.8):
    """
    Removes consecutive points that are closer than min_dist.
    
    Args:
        coords (list): List of (x, y) tuples.
        min_dist (float): Minimum distance threshold.
        
    Returns:
        list: Filtered list of coordinates.
    """
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
    """
    Merges connected lane segments into single continuous paths.
    Useful for handling SUMO's internal lane segmentation.
    
    Args:
        segments (list): List of segment dictionaries.
        
    Returns:
        list: List of merged segment dictionaries.
    """
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
    """
    Clips a polyline to keep only the last 'keep_length' meters.
    
    Args:
        coords (list): List of (x, y) tuples.
        keep_length (float): Length to keep from the end.
        
    Returns:
        list: Clipped coordinates.
    """
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
    """
    Calculates the curvature radius at a specific point on a spline.
    """
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
    """
    Calculates the radius of the circumscribed circle of three points.
    """
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
    """
    Estimates vehicle velocity based on turning radius.
    
    Args:
        radius (float): Turning radius in meters.
        mode (str): 'clearing' or 'entering'.
        
    Returns:
        float: Velocity in m/s.
    """
    if radius <= 6.0:
        v = 5.0
    elif 6.0 < radius < 25.0:
        v = math.sqrt(4 * radius)
    else:
        v = 10.0
    if mode == 'entering' and radius >= 25.0: v = 13.89
    return v


def calculate_intergreen_time(dist_clearing, dist_entering, r_clearing, r_entering):
    """
    Calculates the required intergreen time (K value) between two conflicting streams.
    Based on the German RiLSA or similar guidelines.
    
    Args:
        dist_clearing (float): Distance from conflict point to clearing line.
        dist_entering (float): Distance from stop line to conflict point.
        r_clearing (float): Radius of clearing vehicle path.
        r_entering (float): Radius of entering vehicle path.
        
    Returns:
        int: Calculated intergreen time in seconds (rounded up).
    """
    A = 3.0;
    L_VEHICLE = 6.0
    v_clearing = get_velocity(r_clearing, 'clearing')
    v_entering = get_velocity(r_entering, 'entering')
    U = (dist_clearing + L_VEHICLE) / v_clearing
    B = dist_entering / v_entering
    K = A + U - B
    return max(0, math.ceil(K))


def get_short_name(raw_id):
    """
    Generates a shorter, more readable name for a lane ID.
    """
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
    """
    Main function to process junction data and calculate conflict matrices.
    
    1. Merges segments.
    2. Smooths paths using splines.
    3. Calculates radii and velocities.
    4. Finds intersection points between all pairs of paths.
    5. Calculates intergreen times for all conflicts.
    
    Args:
        jdata (dict): Raw junction data from parser.
        
    Returns:
        dict: Processed data including lines, matrix, intersections, and bounds.
    """
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
