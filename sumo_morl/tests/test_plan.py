"""A jelzésterv (fázisok, közbenső idők, átmenetek) tesztjei."""
import numpy as np
import pytest

from sumo_morl import TJunctionScenario


@pytest.fixture(scope="module")
def plan():
    return TJunctionScenario().plan


def test_groups_and_phases(plan):
    assert [g.lane_id for g in plan.groups] == ["E_C_0", "E_C_1", "S_C_0", "W_C_0"]
    assert plan.n_phases == 3
    # az összes link valamelyik fázisban zöldet kap
    served = {li for p in range(plan.n_phases) for li, ch in enumerate(plan.green_state(p)) if ch == "G"}
    assert served == set(range(plan.n_links))


def test_no_conflicting_greens(plan):
    for p in range(plan.n_phases):
        for a in plan.phases[p]:
            for b in plan.phases[p]:
                assert a == b or not plan.conflicts(a, b)


def test_intergreen_from_geometry(plan):
    # a balra kanyarodó (E_C_1) és a szemből egyenesen haladó (W_C_0) konfliktusos
    assert plan.conflicts(1, 3)
    # a keleti egyenes és a nyugati egyenes nem konfliktusos
    assert not plan.conflicts(0, 3)
    assert np.nanmin(plan.intergreen) >= 1


def _green_intervals(plan, a, b):
    """Átmenet szimulálása másodpercenként; visszaadja a jelzésképek listáját."""
    seq = [plan.green_state(a)] * 3
    for st, d in plan.transition(a, b):
        seq += [st] * int(d)
    seq += [plan.green_state(b)] * 3
    return seq


@pytest.mark.parametrize("a,b", [(a, b) for a in range(3) for b in range(3) if a != b])
def test_transition_respects_intergreen(plan, a, b):
    seq = _green_intervals(plan, a, b)
    for gi, g_from in enumerate(plan.groups):
        for gj, g_to in enumerate(plan.groups):
            k = plan.intergreen[gi, gj]
            if np.isnan(k):
                continue
            li, lj = g_from.links[0], g_to.links[0]
            end_i = max((t for t, s in enumerate(seq) if s[li] == "G" and t < len(seq) - 3), default=None)
            start_j = [t for t, s in enumerate(seq) if s[lj] == "G" and (end_i is None or t > end_i)]
            if end_i is None or not start_j:
                continue
            assert start_j[0] - end_i - 1 >= k, (a, b, gi, gj)


@pytest.mark.parametrize("a,b", [(a, b) for a in range(3) for b in range(3) if a != b])
def test_transition_signal_sequence(plan, a, b):
    """Megszűnő zöld után sárga, induló zöld előtt piros-sárga következik."""
    seq = _green_intervals(plan, a, b)
    for li in range(plan.n_links):
        chars = "".join(s[li] for s in seq)
        assert "Gr" not in chars and "Gu" not in chars  # zöld után mindig sárga
        assert "rG" not in chars and "yG" not in chars  # zöld előtt mindig piros-sárga
