"""A környezet és a segédfüggvények tesztjei (rövid SUMO futásokkal)."""
import numpy as np
import pytest

from sumo_morl import (ActuatedPolicy, Demand, FixedTimePolicy, LinearScalarization, LongestQueuePolicy,
                       RandomPolicy, TJunctionMOEnv, WeightConditioned, equally_spaced_weights,
                       evaluate_policy, hypervolume, pareto_mask, run_episode)


@pytest.fixture(scope="module")
def env():
    e = TJunctionMOEnv(episode_length=300, seed=0)
    yield e
    e.close()


def test_spaces(env):
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert obs.shape == (len(env.observation_names()),)
    obs, r, term, trunc, info = env.step(1)
    assert r.shape == env.reward_space.shape == (3,)
    assert env.observation_space.contains(obs)
    assert info["phase"] == 1 and info["phase_changed"]
    assert info["elapsed"] == env.plan.transition_duration(0, 1) + env.min_green


def test_gym_checker():
    from gymnasium.utils.env_checker import check_env

    e = TJunctionMOEnv(episode_length=60, seed=0, ground_truth=False)  # a NaN-ok miatt
    try:
        check_env(e, skip_render_check=True)
    finally:
        e.close()


def test_determinism(env):
    pol = FixedTimePolicy(env, [20, 10, 10])
    r1 = run_episode(env, pol, seed=5).vector_return
    r2 = run_episode(env, pol, seed=5).vector_return
    np.testing.assert_allclose(r1, r2)


def test_policies_run(env):
    for pol in (RandomPolicy(env, 0), ActuatedPolicy(env), LongestQueuePolicy(env)):
        res = run_episode(env, pol, seed=2, record=True)
        assert res.steps > 0 and len(res.history) == res.steps
        assert res.ground_truth["departed"] > 0


def test_detector_estimate_close_to_truth():
    e = TJunctionMOEnv(episode_length=300, seed=0, demand=Demand.constant(600, 300))
    obs, info = e.reset(seed=0)
    pol = FixedTimePolicy(e, [20, 10, 10])
    while True:
        obs, r, te, tr, info = e.step(pol(obs, info))
        c = e._conn
        for a, edge in {"W": "W_C", "E": "E_C", "S": "S_C"}.items():
            true = sum(1 for v in c.edge.getLastStepVehicleIDs(edge) if c.vehicle.getLanePosition(v) > 10)
            assert abs(info["detectors"].approach_vehicles[a] - true) <= 3
        if tr:
            break
    e.close()


def test_scalarization(env):
    w = LinearScalarization(env, [1.0, 0.0, 0.0])
    w.reset(seed=0)
    _, r, _, _, info = w.step(0)
    assert isinstance(r, float) and r == pytest.approx(float(info["vector_reward"][0]))
    wc = WeightConditioned(env, seed=0)
    obs, _ = wc.reset(seed=0)
    assert obs.shape[0] == env.observation_space.shape[0] + 3


def test_evaluate(env):
    res = evaluate_policy(env, FixedTimePolicy(env, [20, 10, 10]), seeds=[0, 1])
    assert res["returns"].shape == (2, 3)


def test_pareto_and_hv():
    pts = np.array([[1, 3], [2, 2], [3, 1], [1, 1], [2, 1.5]])
    assert pareto_mask(pts).tolist() == [True, True, True, False, False]
    assert hypervolume(pts, [0, 0]) == pytest.approx(6.0)
    assert hypervolume(np.array([[1, 1, 1]]), [0, 0, 0]) == pytest.approx(1.0)
    pts3 = np.array([[1, 2, 1], [2, 1, 1]])
    assert hypervolume(pts3, [0, 0, 0]) == pytest.approx(3.0)
    w = equally_spaced_weights(3, 2)
    assert np.allclose(w.sum(axis=1), 1) and len(w) == 6


def test_signal_log():
    e = TJunctionMOEnv(episode_length=60, seed=0, record_signals=True)
    e.reset(seed=0)
    e.step(2)
    assert len(e.signal_log) == e.plan.transition_duration(0, 2) + e.min_green
    assert e.signal_log[-1][1] == e.plan.green_state(2)
    e.close()
