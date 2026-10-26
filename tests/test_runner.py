"""Прогон: запись шага, детерминизм, сохранение/загрузка, парность сидов."""
from __future__ import annotations

import numpy as np
import pytest

from bandit.configs import get_config
from bandit.envs.base import action_kind
from bandit.runner import (
    RECORD_FIELDS,
    RunConfig,
    RunRecords,
    make_seed_pairs,
    run_config,
    run_episode,
)


@pytest.fixture(scope="module")
def full_cycle_config():
    return get_config("pyramid-ucb1-full_cycle")


def test_run_episode_records_all_fields(full_cycle_config):
    arrays = run_episode(full_cycle_config, env_seed=5, agent_seed=6, n_steps=60)
    assert set(arrays) == set(RECORD_FIELDS)
    for name, values in arrays.items():
        assert values.shape == (60,), name
    assert np.all(np.diff(arrays["points"]) >= 0)
    assert np.all(np.diff(arrays["epoch_after"]) >= 0)
    assert np.all(arrays["n_arms_after"] >= arrays["n_arms_before"])


def test_points_are_cumulative_gains(full_cycle_config):
    arrays = run_episode(full_cycle_config, env_seed=5, agent_seed=6, n_steps=60)
    assert np.allclose(arrays["points"], np.cumsum(arrays["points_gained"]))


def test_meta_steps_have_zero_reward_and_arm_steps_have_values(full_cycle_config):
    arrays = run_episode(full_cycle_config, env_seed=5, agent_seed=6, n_steps=400)
    meta = arrays["kind"] != 0
    assert meta.any() and (~meta).any()
    assert np.all(arrays["reward"][meta] == 0.0)
    assert set(np.unique(arrays["reward"][~meta])) <= {0.0, 1.0}
    assert np.all(arrays["arm_regret"][meta] != arrays["arm_regret"][meta])  # nan
    for action, kind in zip(arrays["actions"], arrays["kind"]):
        assert action_kind(int(action)) == int(kind)


def test_run_episode_is_deterministic(full_cycle_config):
    first = run_episode(full_cycle_config, env_seed=11, agent_seed=12, n_steps=80)
    second = run_episode(full_cycle_config, env_seed=11, agent_seed=12, n_steps=80)
    for name in RECORD_FIELDS:
        assert np.array_equal(first[name], second[name], equal_nan=True), name


def test_seed_pairs_are_reproducible_and_paired():
    first = make_seed_pairs(42, 4)
    second = make_seed_pairs(42, 4)
    assert first == second
    assert all(0 <= pair.env_seed < 2**31 - 1 for pair in first)
    assert len({(pair.env_seed, pair.agent_seed) for pair in first}) == 4
    with pytest.raises(ValueError):
        make_seed_pairs(42, 0)


def test_run_config_is_reproducible_and_matches_serial(full_cycle_config):
    serial = run_config(full_cycle_config, n_runs=2, n_steps=50, progress=False)
    again = run_config(full_cycle_config, n_runs=2, n_steps=50, progress=False)
    assert np.array_equal(serial.points, again.points)
    assert serial.seeds == again.seeds
    assert serial.n_runs == 2 and serial.n_steps == 50 and serial.name == full_cycle_config.name


def test_parallel_cli_matches_serial_cli(tmp_path):
    """``--jobs 2`` (ProcessPoolExecutor) даёт те же числа, что ``--jobs 1``.

    Проверяется через настоящий CLI: там ``__main__`` — модуль ``bandit.cli``
    с guard'ом, поэтому spawn на Windows безопасен (в отличие от запуска
    пула из-под pytest).
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    base = [
        sys.executable,
        "-m",
        "bandit.cli",
        "run",
        "--config",
        "pyramid-ucb1-full_cycle",
        "--runs",
        "2",
        "--steps",
        "40",
        "--quiet",
        "--no-plots",
    ]
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    subprocess.run(base + ["--out", str(serial_dir)], cwd=root, check=True, env=env, timeout=300)
    subprocess.run(
        base + ["--out", str(parallel_dir), "--jobs", "2"],
        cwd=root,
        check=True,
        env=env,
        timeout=300,
    )

    serial = RunRecords.load(serial_dir, "pyramid-ucb1-full_cycle")
    parallel = RunRecords.load(parallel_dir, "pyramid-ucb1-full_cycle")
    assert serial.seeds == parallel.seeds
    for name in RECORD_FIELDS:
        assert np.array_equal(serial.arrays[name], parallel.arrays[name], equal_nan=True), name


def test_run_config_can_override_horizon(full_cycle_config):
    records = run_config(full_cycle_config, n_runs=1, n_steps=37, progress=False)
    assert records.n_steps == 37
    assert records.config["n_steps"] == 37


def test_run_records_save_load_roundtrip(tmp_path, full_cycle_config):
    records = run_config(full_cycle_config, n_runs=2, n_steps=40, progress=False)
    path = records.save(tmp_path)
    assert path.exists()

    loaded = RunRecords.load(tmp_path, records.name)
    assert loaded.name == records.name
    assert loaded.seeds == records.seeds
    for name in RECORD_FIELDS:
        assert np.array_equal(loaded.arrays[name], records.arrays[name], equal_nan=True), name

    subset = loaded.run(1)
    assert subset.n_runs == 1
    assert subset.points[0].tolist() == records.points[1].tolist()


def test_run_config_validation():
    base = {"name": "x", "env": {"kind": "layered"}, "agent": {"kind": "random"}}
    with pytest.raises(ValueError):
        RunConfig(n_steps=0, **base)
    with pytest.raises(ValueError):
        RunConfig(env={"params": {}}, agent={"kind": "random"}, name="x")
    with pytest.raises(ValueError):
        RunConfig(env={"kind": "layered"}, agent={"params": {}}, name="x")

    config = RunConfig(
        env={"kind": "layered", "params": {"n_arms_init": 3}},
        agent={"kind": "ucb1", "params": {"c": 1.0}},
        policy={"kind": "full_cycle"},
        n_steps=10,
        name="roundtrip",
    )
    assert RunConfig.from_dict(config.to_dict()) == config


def test_stationary_run_has_no_nan_regret():
    config = get_config("stationary-ucb1")
    records = run_config(config, n_runs=2, n_steps=60, progress=False)
    assert np.isfinite(records.arm_regret).all()
    assert np.all(records.kind == 0)
    assert np.all(records.epoch_after == 0)
