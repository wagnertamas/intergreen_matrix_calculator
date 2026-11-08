"""
Indukcióshurok-detektorok (E1) elhelyezése és olvasása.

**Csak valóságban is mérhető adatokat használunk.** Egy valódi hurokdetektor
azt érzékeli, hogy van-e jármű a hurok felett. Ebből a forgalomtechnikai
vezérlők a következőket számolják, és mi is pontosan ezeket adjuk át:

* ``count``      – a hurkon áthaladt járművek száma (felfutó élek),
* ``occupancy``  – foglaltság: az idő hányadában volt jármű a hurok felett [0..1],
* ``speed``      – átlagsebesség a hurkon [m/s] (dupla hurokkal mérhető;
  ha nem volt jármű, a megengedett sebességet adjuk vissza),
* ``gap``        – a legutóbbi érzékelés óta eltelt idő [s] (időrés).

Nem használunk SUMO "belső igazságot" (pl. jármű-azonosítók, várakozási idő,
pontos sorhossz). A jármű-azonosítókat a SUMO-ból csak arra használjuk, hogy a
hurok felfutó éleit megszámoljuk – ez egyenértékű egy valódi hurok
impulzusszámlálásával.

Minden bejövő sávon három hurok van::

    ág eleje                                                 stopvonal
    |--[entry]-----------------------------[advance]--------[stopline]|
       10 m-re az ág elejétől               40 m-re          3 m-re a stopvonaltól

Az ``entry`` és ``stopline`` hurkok számlálóiból ágankénti "bemenet–kimenet"
becslést készítünk a két hurok közötti járműszámra (ez a klasszikus,
valóságban is használt sorhossz-becslési módszer).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DETECTOR_KINDS = ("entry", "advance", "stopline")
FEATURES = ("count", "occupancy", "speed", "gap")


@dataclass
class DetectorInfo:
    id: str
    lane_id: str
    edge_id: str
    approach: str      # 'W', 'E', 'S'
    kind: str          # 'entry', 'advance', 'stopline'
    pos: float         # pozíció a sáv elejétől [m]
    speed_limit: float


def detector_layout(net_file: str, incoming_edges: Dict[str, str],
                    stopline_dist: float = 3.0, advance_dist: float = 40.0,
                    entry_dist: float = 10.0) -> List[DetectorInfo]:
    """Kiszámolja a hurkok helyét a bejövő sávokon."""
    root = ET.parse(net_file).getroot()
    edge_to_app = {e: a for a, e in incoming_edges.items()}
    dets: List[DetectorInfo] = []
    for edge in root.findall("edge"):
        eid = edge.get("id")
        if eid not in edge_to_app:
            continue
        for lane in edge.findall("lane"):
            lid, length, v = lane.get("id"), float(lane.get("length")), float(lane.get("speed"))
            for kind, pos in (("entry", entry_dist),
                              ("advance", length - advance_dist),
                              ("stopline", length - stopline_dist)):
                dets.append(DetectorInfo(f"{lid}_{kind}", lid, eid, edge_to_app[eid], kind, pos, v))
    dets.sort(key=lambda d: ("WES".index(d.approach) if d.approach in "WES" else 9,
                             d.lane_id, DETECTOR_KINDS.index(d.kind)))
    return dets


def write_detector_file(dets: List[DetectorInfo], out_file: str) -> str:
    """SUMO additional fájl (``.add.xml``) írása a hurkokkal."""
    root = ET.Element("additional")
    for d in dets:
        ET.SubElement(root, "inductionLoop", id=d.id, lane=d.lane_id, pos=f"{d.pos:.2f}",
                      period="3600", file="NUL")
    ET.indent(root)
    ET.ElementTree(root).write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file


@dataclass(eq=False)
class DetectorSnapshot:
    """Egy vezérlési lépés (több szimulációs másodperc) aggregált hurokadatai.

    Minden tömb a ``detectors`` lista sorrendjét követi.
    """

    duration: float                       # a lépés hossza [s]
    detectors: List[DetectorInfo]
    count: np.ndarray                     # áthaladt járművek [db]
    occupancy: np.ndarray                 # átlagos foglaltság [0..1]
    speed: np.ndarray                     # átlagsebesség [m/s]
    gap: np.ndarray                       # utolsó érzékelés óta [s] (lépés végén)
    approach_vehicles: Dict[str, float]   # bemenet-kimenet becslés ágankként [db] (lépés végén)
    approach_vehicle_seconds: Dict[str, float]  # a becslés időintegrálja a lépés alatt [db*s]
    stopline_occupied_seconds: np.ndarray = field(default=None)  # foglalt másodpercek [s]

    def __eq__(self, other) -> bool:
        if not isinstance(other, DetectorSnapshot):
            return NotImplemented
        arrs = ("count", "occupancy", "speed", "gap", "stopline_occupied_seconds")
        return (self.duration == other.duration and self.approach_vehicles == other.approach_vehicles
                and all(np.array_equal(getattr(self, a), getattr(other, a)) for a in arrs))

    def by_id(self) -> Dict[str, Dict[str, float]]:
        """``{detektor_id: {count, occupancy, speed, gap}}`` szótár."""
        return {d.id: {"count": float(self.count[i]), "occupancy": float(self.occupancy[i]),
                       "speed": float(self.speed[i]), "gap": float(self.gap[i])}
                for i, d in enumerate(self.detectors)}

    def select(self, kind: Optional[str] = None, lane_id: Optional[str] = None,
               approach: Optional[str] = None) -> np.ndarray:
        """Indexmaszk a detektorok szűréséhez (típus / sáv / ág szerint)."""
        return np.array([(kind is None or d.kind == kind) and
                         (lane_id is None or d.lane_id == lane_id) and
                         (approach is None or d.approach == approach)
                         for d in self.detectors], dtype=bool)


class DetectorReader:
    """A hurkok másodpercenkénti olvasása TraCI-n keresztül és aggregálása."""

    def __init__(self, dets: List[DetectorInfo], approaches: List[str]):
        self.dets = dets
        self.approaches = list(approaches)
        self._idx_entry = {a: [i for i, d in enumerate(dets) if d.approach == a and d.kind == "entry"]
                           for a in self.approaches}
        self._idx_stop = {a: [i for i, d in enumerate(dets) if d.approach == a and d.kind == "stopline"]
                          for a in self.approaches}
        self.reset()

    def reset(self) -> None:
        n = len(self.dets)
        self._prev_ids: List[set] = [set() for _ in range(n)]
        self._gap = np.zeros(n)
        self.approach_vehicles = {a: 0.0 for a in self.approaches}
        self._begin_step()

    def _begin_step(self) -> None:
        n = len(self.dets)
        self._count = np.zeros(n)
        self._occ_sum = np.zeros(n)
        self._speed_sum = np.zeros(n)
        self._speed_n = np.zeros(n)
        self._seconds = 0
        self._veh_seconds = {a: 0.0 for a in self.approaches}

    def subscribe(self, conn) -> None:
        """TraCI előfizetés a hurkokra (gyorsabb, mint az egyenkénti lekérdezés)."""
        import traci.constants as tc

        self._vars = (tc.LAST_STEP_VEHICLE_ID_LIST, tc.LAST_STEP_OCCUPANCY,
                      tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION)
        for d in self.dets:
            conn.inductionloop.subscribe(d.id, self._vars)

    def sample(self, conn) -> None:
        """Egy szimulációs lépés (1 s) utáni leolvasás. ``conn`` egy TraCI kapcsolat."""
        res = conn.inductionloop.getAllSubscriptionResults()
        v_ids, v_occ, v_spd, v_gap = self._vars
        new = np.zeros(len(self.dets))
        for i, d in enumerate(self.dets):
            r = res[d.id]
            ids = set(r[v_ids])
            new[i] = len(ids - self._prev_ids[i])  # felfutó élek = új járművek a hurkon
            self._prev_ids[i] = ids
            self._occ_sum[i] += r[v_occ] / 100.0
            sp = r[v_spd]
            if sp >= 0:
                self._speed_sum[i] += sp
                self._speed_n[i] += 1
            self._gap[i] = r[v_gap]
        self._count += new
        # bemenet-kimenet becslés ágankként: + entry hurok, - stopvonali hurok
        for a in self.approaches:
            v = (self.approach_vehicles[a] + new[self._idx_entry[a]].sum()
                 - new[self._idx_stop[a]].sum())
            self.approach_vehicles[a] = max(0.0, float(v))
            self._veh_seconds[a] += self.approach_vehicles[a]
        self._seconds += 1

    def snapshot(self) -> DetectorSnapshot:
        """Lezárja az aktuális lépést és visszaadja az aggregált adatokat."""
        secs = max(1, self._seconds)
        speed = np.array([self._speed_sum[i] / self._speed_n[i] if self._speed_n[i] > 0 else d.speed_limit
                          for i, d in enumerate(self.dets)])
        snap = DetectorSnapshot(
            duration=float(self._seconds), detectors=self.dets,
            count=self._count.copy(), occupancy=self._occ_sum / secs, speed=speed,
            gap=self._gap.copy(), approach_vehicles=dict(self.approach_vehicles),
            approach_vehicle_seconds=dict(self._veh_seconds),
            stopline_occupied_seconds=self._occ_sum.copy(),
        )
        self._begin_step()
        return snap
