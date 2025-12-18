import xml.etree.ElementTree as ET

class SumoInternalParser:
    """
    Parses a SUMO network XML file (.net.xml) to extract junction and lane information.
    
    This class is responsible for reading the XML structure of a SUMO network,
    identifying internal and normal lanes, and grouping them by junctions.
    It extracts geometry (shapes) which are essential for the intergreen time calculations.
    """
    def __init__(self, file_path):
        """
        Initialize the parser with the given file path.

        Args:
            file_path (str): The absolute or relative path to the .net.xml file.
        """
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
        """
        Iterates through all 'edge' elements in the XML to load lane shapes.
        Separates 'internal' lanes (inside junctions) from normal lanes.
        """
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
        """
        Groups the loaded lane data by junctions.
        
        Filters for specific junction types (traffic_light, priority, etc.).
        Calculates relative coordinates for visualization (relative to junction center).

        Returns:
            list: A list of dictionaries, where each dictionary represents a junction
                  and contains its ID, position, shape, and associated lane segments.
        """
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
