"""
Jelzésterv (fázisok + közbenső idők) automatikus előállítása.

A közbenső időket (intergreen) **a repó meglévő közbensőidő-számító kódja**
(:mod:`geometry_utils` + :mod:`sumo_parser`) számolja ki a csomópont
geometriájából (ütközési pontok, kanyarodási sugarak, sebességek). Ez a modul
erre építve:

1. jelzőcsoportokat képez (egy bejövő sáv = egy jelzőcsoport, így egy sávon
   belül soha nem kap egyik irány zöldet a másik nélkül),
2. a mozgás-szintű közbenső idő mátrixból jelzőcsoport-szintű mátrixot képez
   (maximum a csoportok mozgásai között),
3. a konfliktus-mentes csoportokból (gráf klikkek) fázisokat képez,
4. bármely két fázis között legenerálja az átmenetet (sárga, piros,
   piros-sárga), úgy hogy minden közbenső idő teljesüljön.

A hallgatóknak mindezzel nem kell foglalkozniuk: a környezet egy kész
:class:`SignalPlan` objektumot használ, az ágens csak a következő fázis
sorszámát választja ki.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# A repó gyökerében lévő közbensőidő-számító modulok importálása.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:  # pragma: no cover - csak forrásból futtatva kell
    sys.path.insert(0, _REPO_ROOT)

DIR_NAMES = {"s": "egyenes", "l": "balra", "r": "jobbra", "t": "visszafordulás",
             "L": "balra (enyhén)", "R": "jobbra (enyhén)"}


@dataclass
class Link:
    """Egy SUMO "link": egy bejövő sávból egy kimenő sávba vezető mozgás."""

    index: int          # a jelzőlámpa állapot-sztringjében elfoglalt pozíció
    from_edge: str
    from_lane: int
    to_edge: str
    direction: str      # 's', 'l', 'r', ...

    @property
    def lane_id(self) -> str:
        return f"{self.from_edge}_{self.from_lane}"


@dataclass
class SignalGroup:
    """Jelzőcsoport: egy bejövő sáv összes mozgása, amelyek együtt kapnak jelzést."""

    name: str
    lane_id: str
    links: List[int]
    description: str = ""


@dataclass
class SignalPlan:
    """A csomópont teljes jelzésterve.

    Attributes:
        groups: jelzőcsoportok listája.
        phases: fázisok; mindegyik a benne zöldet kapó csoportok indexeinek listája.
        intergreen: ``[kiürítő][belépő]`` közbenső idő mátrix a csoportok között
            [s]; ``nan``, ha a két csoport nem konfliktusos.
        link_intergreen: a mozgás (link) szintű mátrix, ahogy a közbensőidő-számító
            kód kiadta.
        n_links: a SUMO állapot-sztring hossza.
        yellow: sárga idő [s].
        red_yellow: piros-sárga idő [s].
    """

    tls_id: str
    links: List[Link]
    groups: List[SignalGroup]
    phases: List[List[int]]
    intergreen: np.ndarray
    link_intergreen: np.ndarray
    yellow: int = 3
    red_yellow: int = 2
    phase_names: List[str] = field(default_factory=list)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def conflicts(self, gi: int, gj: int) -> bool:
        """Igaz, ha a két jelzőcsoport konfliktusos (nem kaphat egyszerre zöldet)."""
        return not np.isnan(self.intergreen[gi, gj]) or not np.isnan(self.intergreen[gj, gi])

    # ------------------------------------------------------------------ állapotok
    def _state(self, chars: Dict[int, str]) -> str:
        s = ["r"] * self.n_links
        for gi, ch in chars.items():
            for li in self.groups[gi].links:
                s[li] = ch
        return "".join(s)

    def green_state(self, phase: int) -> str:
        """A ``phase`` fázis SUMO állapot-sztringje (pl. ``'GrrrGG'``)."""
        return self._state({g: "G" for g in self.phases[phase]})

    def phase_lanes(self, phase: int) -> List[str]:
        """A fázisban zöldet kapó bejövő sávok azonosítói."""
        return [self.groups[g].lane_id for g in self.phases[phase]]

    def group_start_offsets(self, from_phase: int, to_phase: int) -> Dict[int, int]:
        """Az induló csoportok zöldkezdete a régi zöld vége után [s]."""
        cur, nxt = set(self.phases[from_phase]), set(self.phases[to_phase])
        stopping, starting = cur - nxt, nxt - cur
        out = {}
        for gj in starting:
            need = self.red_yellow
            for gi in stopping:
                k = self.intergreen[gi, gj]
                if not np.isnan(k):
                    need = max(need, int(k))
            out[gj] = need
        return out

    def transition(self, from_phase: int, to_phase: int) -> List[Tuple[str, int]]:
        """Átmenet két fázis között: ``[(állapot, időtartam_s), ...]``.

        ``t = 0`` a régi zöld vége. A megszűnő csoportok ``yellow`` ideig sárgát,
        utána pirosat kapnak; az induló csoportok annyi idő után kapnak zöldet,
        amennyit a közbenső idő mátrix előír (legalább ``red_yellow``), előtte
        ``red_yellow`` ideig piros-sárgát. A megmaradó csoportok végig zöldek.
        Az átmenet akkor ér véget, amikor az utolsó induló csoport is zöldet kap.
        """
        if from_phase == to_phase:
            return []
        cur, nxt = set(self.phases[from_phase]), set(self.phases[to_phase])
        stopping, staying = cur - nxt, cur & nxt
        starts = self.group_start_offsets(from_phase, to_phase)
        total = max([self.yellow] + list(starts.values()))
        cuts = {0, total, min(self.yellow, total)}
        for t in starts.values():
            cuts.update({t - self.red_yellow, t})
        cuts = sorted(c for c in cuts if 0 <= c <= total)
        steps: List[Tuple[str, int]] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            chars: Dict[int, str] = {g: "G" for g in staying}
            for g in stopping:
                chars[g] = "y" if a < self.yellow else "r"
            for g, t in starts.items():
                if a >= t:
                    chars[g] = "G"
                elif a >= t - self.red_yellow:
                    chars[g] = "u"
                else:
                    chars[g] = "r"
            st = self._state(chars)
            if steps and steps[-1][0] == st:
                steps[-1] = (st, steps[-1][1] + (b - a))
            else:
                steps.append((st, b - a))
        # Az utolsó szakasz már maga az új fázis zöldje -> nem része az átmenetnek.
        if steps and steps[-1][0] == self.green_state(to_phase):
            steps.pop()
        return steps

    def transition_duration(self, from_phase: int, to_phase: int) -> int:
        return int(sum(d for _, d in self.transition(from_phase, to_phase)))

    # ------------------------------------------------------------------ leírás
    def describe(self) -> str:
        """Ember által olvasható összefoglaló a jelzéstervről."""
        lines = [f"Jelzőlámpa: {self.tls_id}", "", "Jelzőcsoportok:"]
        for i, g in enumerate(self.groups):
            lines.append(f"  [{i}] {g.name:<12} sáv={g.lane_id:<6} linkek={g.links}  {g.description}")
        lines.append("")
        lines.append("Közbenső idők [s] (sor = kiürítő, oszlop = belépő, '-' = nincs konfliktus):")
        hdr = "      " + "".join(f"{i:>5}" for i in range(len(self.groups)))
        lines.append(hdr)
        for i in range(len(self.groups)):
            row = "".join(
                f"{'-' if np.isnan(v) else int(v):>5}" for v in self.intergreen[i]
            )
            lines.append(f"  [{i}]{row}")
        lines.append("")
        lines.append("Fázisok:")
        for p, groups in enumerate(self.phases):
            name = self.phase_names[p] if p < len(self.phase_names) else ""
            lines.append(f"  Fázis {p}: {self.green_state(p)}  csoportok={groups}  {name}")
        lines.append("")
        lines.append("Átmenetek hossza [s]:")
        for a in range(self.n_phases):
            for b in range(self.n_phases):
                if a != b:
                    steps = " -> ".join(f"{s}({d}s)" for s, d in self.transition(a, b))
                    lines.append(f"  {a} -> {b}: {self.transition_duration(a, b):>2} s   {steps}")
        return "\n".join(lines)


# ====================================================================== építés
def read_links(net_file: str, tls_id: str) -> List[Link]:
    """Kiolvassa a jelzőlámpához tartozó linkeket a ``.net.xml``-ből."""
    root = ET.parse(net_file).getroot()
    links: Dict[int, Link] = {}
    for c in root.findall("connection"):
        if c.get("tl") == tls_id and c.get("linkIndex") is not None:
            idx = int(c.get("linkIndex"))
            links[idx] = Link(idx, c.get("from"), int(c.get("fromLane")), c.get("to"), c.get("dir"))
    return [links[i] for i in sorted(links)]


def compute_link_intergreen(net_file: str, tls_id: str) -> np.ndarray:
    """A meglévő közbensőidő-számító kóddal kiszámolja a link-szintű mátrixot.

    Returns:
        ``n_links x n_links`` mátrix, ``[kiürítő][belépő]`` közbenső idő [s],
        ``nan`` ahol nincs ütközési pont.
    """
    from geometry_utils import calculate_junction_data  # repó gyökér
    from sumo_parser import SumoInternalParser          # repó gyökér

    with contextlib.redirect_stdout(io.StringIO()):
        parser = SumoInternalParser(net_file)
    jdata = next(j for j in parser.junctions if j["id"] == tls_id)
    data = calculate_junction_data(jdata)
    n_links = len(read_links(net_file, tls_id))
    mat = np.full((n_links, n_links), np.nan)
    for i, li in enumerate(data["lines"]):
        for j, lj in enumerate(data["lines"]):
            v = data["matrix"][i][j]
            if v is None:
                continue
            a, b = li["logic_idx"], lj["logic_idx"]
            if 0 <= a < n_links and 0 <= b < n_links:
                mat[a, b] = np.nanmax([mat[a, b], float(v[0])])
    # Ha csak az egyik irányban talált ütközési pontot, a másik irányt is
    # konfliktusnak tekintjük (a legnagyobb ismert értékkel).
    for a in range(n_links):
        for b in range(n_links):
            if np.isnan(mat[a, b]) and not np.isnan(mat[b, a]):
                mat[a, b] = mat[b, a]
    return mat


def build_signal_groups(links: Sequence[Link], names: Optional[Dict[str, str]] = None) -> List[SignalGroup]:
    """Jelzőcsoportok képzése: egy bejövő sáv = egy csoport."""
    by_lane: Dict[str, List[Link]] = {}
    for l in links:
        by_lane.setdefault(l.lane_id, []).append(l)
    groups = []
    for lane_id in sorted(by_lane):
        ls = by_lane[lane_id]
        desc = ", ".join(f"{DIR_NAMES.get(l.direction, l.direction)} ({l.to_edge})" for l in ls)
        name = (names or {}).get(lane_id, lane_id)
        groups.append(SignalGroup(name, lane_id, [l.index for l in ls], desc))
    return groups


def group_intergreen(link_mat: np.ndarray, groups: Sequence[SignalGroup]) -> np.ndarray:
    """Csoport-szintű közbenső idő mátrix (maximum a csoportok linkjei között)."""
    n = len(groups)
    out = np.full((n, n), np.nan)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i == j:
                continue
            vals = [link_mat[a, b] for a in gi.links for b in gj.links if not np.isnan(link_mat[a, b])]
            if vals:
                out[i, j] = max(vals)
    return out


def phases_from_conflicts(intergreen: np.ndarray) -> List[List[int]]:
    """Maximális konfliktusmentes csoporthalmazok (klikkek) = fázisok."""
    import networkx as nx

    n = intergreen.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(intergreen[i, j]) and np.isnan(intergreen[j, i]):
                g.add_edge(i, j)
    cliques = [sorted(c) for c in nx.find_cliques(g)]
    cliques.sort(key=lambda c: (-len(c), c))
    return cliques


def build_signal_plan(
    net_file: str,
    tls_id: str,
    phases: Optional[List[List[int]]] = None,
    yellow: int = 3,
    red_yellow: int = 2,
    group_names: Optional[Dict[str, str]] = None,
    phase_names: Optional[List[str]] = None,
) -> SignalPlan:
    """Teljes jelzésterv előállítása egy ``.net.xml`` fájlból.

    Args:
        net_file: hálózat fájl.
        tls_id: a jelzőlámpás csomópont azonosítója.
        phases: opcionális kézi fázislista (csoport-indexek); ha ``None``, a
            konfliktusmentes klikkek lesznek a fázisok.
        yellow, red_yellow: sárga és piros-sárga idő [s].
    """
    links = read_links(net_file, tls_id)
    link_mat = compute_link_intergreen(net_file, tls_id)
    groups = build_signal_groups(links, group_names)
    ig = group_intergreen(link_mat, groups)
    if phases is None:
        phases = phases_from_conflicts(ig)
    plan = SignalPlan(tls_id, links, groups, phases, ig, link_mat, yellow, red_yellow,
                      list(phase_names or []))
    for p in phases:
        for a in p:
            for b in p:
                if a != b and plan.conflicts(a, b):
                    raise ValueError(f"A(z) {p} fázisban konfliktusos csoportok vannak: {a}, {b}")
    return plan
