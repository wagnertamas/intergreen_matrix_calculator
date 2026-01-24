import xml.etree.ElementTree as ET

class SumoInternalParser:
    def __init__(self, file_path):
        print(f"Fájl feldolgozása: {file_path}")
        self.file_path = file_path
        self.tree = ET.parse(file_path)
        self.root = self.tree.getroot()
        self.internal_lanes = {}
        self.normal_lanes = {}
        self.lane_to_link_index = {} # [NEW] Map internal lane ID -> Link Index
        self._load_data()
        self.junctions = self._group_by_junctions()
        print(f"Betöltve: {len(self.junctions)} csomópont.")

    def _load_data(self):
        # 1. Edges & Lanes
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
        
        # 2. [NEW] Connections (Map 'via' lane to 'linkIndex')
        for conn in self.root.findall("connection"):
            via = conn.get("via")
            link_idx = conn.get("linkIndex")
            if via and link_idx is not None:
                try:
                    self.lane_to_link_index[via] = int(link_idx)
                except ValueError:
                    pass

    def _group_by_junctions(self):
        junctions_data = []
        for junc in self.root.findall("junction"):
            if junc.get("type") in ['traffic_light', 'traffic_light_right_on_red']:
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
                        # [NEW] Attach Logic Index
                        l_idx = self.lane_to_link_index.get(lid, -1)
                        # Fallback for old parsing logic if not found (though unlikely for valid TLS)
                        segments.append({'id': lid, 'points': rel_points, 'logic_idx': l_idx})
                if segments:
                    junctions_data.append(
                        {'id': jid, 'x': jx, 'y': jy, 'segments': segments, 'shape': rel_shape,
                         'incoming_lanes': incoming_shapes})
        return junctions_data
