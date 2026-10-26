"""H6: информационная перегрузка — среда, конфиги и вердикт отчёта.

Проверяется три вещи:

* параметры перегрузки не меняют награду и очки вытягивания, а влияют только
  на учёт информации (иначе H6 смешалась бы с H1);
* при значениях по умолчанию поведение прежнее: ровно одна полезная единица и
  ни одного лишнего обращения к RNG (golden-тесты остаются в силе);
* отчёт выставляет вердикт H6 по данным, а не по имени набора конфигов.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bandit import report as rp
from bandit import stats as st
from bandit.cli import _oracle_finals
from bandit.configs import (
    INFO_REGIMES,
    OVERLOAD_AGENT_POLICIES,
    OVERLOAD_BASE_REGIME,
    REGISTRY,
    SUITES,
)
from bandit.envs import LayeredPyramidBandit, MetaAction, meta_action_id
from bandit.report import split_config_name
from bandit.runner import RECORD_DTYPES, RECORD_FIELDS, RunRecords, SeedPair


# ---------------------------------------------------------------------------
# Среда: поток информации отделён от награды
# ---------------------------------------------------------------------------
def test_default_pull_yields_one_useful_unit():
    env = LayeredPyramidBandit(seed=1)
    info = env.step(0)
    assert info.useful_gained == 1.0
    assert info.info_gained == 1.0
    assert env.info_units == 1
    assert env.data[0] == 1


def test_meta_actions_carry_no_information():
    env = LayeredPyramidBandit(seed=1)
    info = env.step(meta_action_id(MetaAction.INFO_TO_MULTIPLIER))
    assert info.useful_gained == 0.0
    assert info.info_gained == 0.0


def test_noise_is_counted_but_does_not_feed_economy():
    env = LayeredPyramidBandit(
        seed=2, info_per_pull=4, useful_info_prob=0.0, junk_feeds_multiplier=0.0
    )
    info = env.step(0)
    assert info.info_gained == 4.0
    assert info.useful_gained == 0.0
    assert env.info_units == 0
    assert env.data.get(0, 0) == 0
    # Шум не трогает очки: множитель по-прежнему 1, награда — базовый 0/1.
    assert info.points_gained == pytest.approx(info.reward)


def test_junk_feeds_multiplier_counts_volume():
    env = LayeredPyramidBandit(
        seed=2, info_per_pull=3, useful_info_prob=0.0, junk_feeds_multiplier=1.0
    )
    env.step(0)
    assert env.info_units == 3
    assert env.data.get(0, 0) == 0


def test_default_regime_keeps_rng_order():
    """При параметрах по умолчанию лишних обращений к RNG нет."""
    implicit = LayeredPyramidBandit(seed=7)
    explicit = LayeredPyramidBandit(
        seed=7, info_per_pull=1, useful_info_prob=1.0, junk_feeds_multiplier=1.0
    )
    assert implicit.means.tolist() == explicit.means.tolist()
    for _ in range(50):
        first, second = implicit.step(0), explicit.step(0)
        assert first.reward == second.reward
        assert first.points_gained == second.points_gained


def test_noise_does_not_shift_reward_stream():
    """Шум не должен менять очки политики, которая данные не использует."""
    noisy = LayeredPyramidBandit(
        seed=11, info_per_pull=4, useful_info_prob=0.5, junk_feeds_multiplier=0.0
    )
    clean = LayeredPyramidBandit(seed=11)
    for _ in range(20):
        assert noisy.step(0).reward == clean.step(0).reward
    assert noisy.points == pytest.approx(clean.points)


def test_overload_parameters_are_validated():
    with pytest.raises(ValueError):
        LayeredPyramidBandit(info_per_pull=0)
    with pytest.raises(ValueError):
        LayeredPyramidBandit(useful_info_prob=1.5)
    with pytest.raises(ValueError):
        LayeredPyramidBandit(junk_feeds_multiplier=-1.0)


# ---------------------------------------------------------------------------
# Конфиги: 6 режимов × (5 пар + оракул)
# ---------------------------------------------------------------------------
def test_overload_suite_names_and_env_params():
    assert len(INFO_REGIMES) == 6
    names = SUITES["pyramid-overload"]
    assert len(names) == 36 == len(INFO_REGIMES) * (len(OVERLOAD_AGENT_POLICIES) + 1)
    for name in names:
        parsed = split_config_name(name)
        assert parsed is not None, name
        env, agent, policy = parsed
        regime = env.split("pyramid-overload-", 1)[1]
        assert regime in INFO_REGIMES
        config = REGISTRY[name]
        for key, value in INFO_REGIMES[regime].items():
            assert config.env["params"][key] == value, (name, key)
        if agent == "oracle":
            assert config.expose_oracle and policy == "full_cycle"
        else:
            assert not config.expose_oracle
    assert OVERLOAD_BASE_REGIME in INFO_REGIMES


# ---------------------------------------------------------------------------
# Отчёт: вердикт H6 по данным
# ---------------------------------------------------------------------------
N_RUNS = 6
METRICS = [
    "final_points",
    "steps_to_epoch_1",
    "steps_to_epoch_2",
    "final_epoch",
    "useful_info_share",
    "junk_info_share",
]


def _write_runs(path, values):
    sizes = {len(table[metric]) for table in values.values() for metric in METRICS}
    assert len(sizes) == 1
    n_runs = sizes.pop()
    columns = list(st.CSV_INDEX_COLUMNS) + METRICS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, table in values.items():
            for run in range(n_runs):
                writer.writerow(
                    {
                        "config": name,
                        "run": run,
                        "env_seed": 10 + run,
                        "agent_seed": 20 + run,
                        **{metric: table[metric][run] for metric in METRICS},
                    }
                )
    return path


def _regime_values(regime, gain, share, reach=1.0):
    """Две строки режима: ``none`` (без эпох) и ``full_cycle`` с выигрышем."""
    base = [1000.0 + 10.0 * run for run in range(N_RUNS)]
    full = [value + gain + 5.0 * run for run, value in enumerate(base)]
    reached = reach > 0
    nan = [float("nan")] * N_RUNS
    return {
        f"{rp.OVERLOAD_ENV_PREFIX}{regime}-ucb1-none": {
            "final_points": base,
            "steps_to_epoch_1": nan,
            "steps_to_epoch_2": nan,
            "final_epoch": [0.0] * N_RUNS,
            "useful_info_share": [share] * N_RUNS,
            "junk_info_share": [1.0 - share] * N_RUNS,
        },
        f"{rp.OVERLOAD_ENV_PREFIX}{regime}-ucb1-full_cycle": {
            "final_points": full,
            "steps_to_epoch_1": (
                [120.0 + run for run in range(N_RUNS)] if reached else nan
            ),
            "steps_to_epoch_2": (
                [300.0 + run for run in range(N_RUNS)] if reached else nan
            ),
            "final_epoch": [2.0] * N_RUNS if reached else [0.0] * N_RUNS,
            "useful_info_share": [share] * N_RUNS,
            "junk_info_share": [1.0 - share] * N_RUNS,
        },
    }


def test_h6_supported_when_gain_survives_overload(tmp_path):
    values = {}
    values.update(_regime_values("base", gain=1000.0, share=1.0))
    values.update(_regime_values("noisy", gain=900.0, share=0.5))
    values.update(_regime_values("starved", gain=800.0, share=0.25))
    _write_runs(tmp_path / "runs.csv", values)

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h6 = next(item for item in report.payload["hypotheses"] if item["code"] == "H6")
    assert h6["status"] == rp.SUPPORTED
    assert report.payload["reference"] == f"{rp.OVERLOAD_ENV_PREFIX}base-ucb1-none"
    pairs = {pair["regime"]: pair for pair in h6["evidence"]["pairs"]}
    assert set(pairs) == {"base", "noisy", "starved"}
    assert pairs["starved"]["useful_info_share"] == pytest.approx(0.25)
    assert h6["evidence"]["quality_measured"] is True
    assert h6["evidence"]["median_retention"] == pytest.approx(0.8518, abs=1e-3)
    assert h6["evidence"]["min_retention"] == pytest.approx(0.8025, abs=1e-3)
    assert "сигнала 25%" in h6["detail"]


def test_h6_partial_when_one_regime_halves_gain(tmp_path):
    values = {}
    values.update(_regime_values("base", gain=1000.0, share=1.0))
    values.update(_regime_values("noisy", gain=400.0, share=0.5))
    _write_runs(tmp_path / "runs.csv", values)

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h6 = next(item for item in report.payload["hypotheses"] if item["code"] == "H6")
    assert h6["status"] == rp.PARTIAL
    assert h6["evidence"]["min_retention"] < rp.OVERLOAD_RETENTION_MIN


def test_h6_refuted_when_overload_breaks_advantage(tmp_path):
    values = {}
    values.update(_regime_values("base", gain=1000.0, share=1.0))
    values.update(_regime_values("flood", gain=-400.0, share=1.0))
    _write_runs(tmp_path / "runs.csv", values)

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h6 = next(item for item in report.payload["hypotheses"] if item["code"] == "H6")
    assert h6["status"] == rp.REFUTED
    flood = next(pair for pair in h6["evidence"]["pairs"] if pair["regime"] == "flood")
    assert flood["diff"] < 0 and flood["significant"]


def test_h6_refuted_when_epochs_stop_opening(tmp_path):
    values = {}
    values.update(_regime_values("base", gain=1000.0, share=1.0))
    values.update(_regime_values("starved", gain=100.0, share=0.25, reach=0.0))
    _write_runs(tmp_path / "runs.csv", values)

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h6 = next(item for item in report.payload["hypotheses"] if item["code"] == "H6")
    assert h6["status"] == rp.REFUTED
    assert h6["evidence"]["collapsed"] == ["starved"]
    starved = next(pair for pair in h6["evidence"]["pairs"] if pair["regime"] == "starved")
    assert starved["reach_epoch_1"] == 0.0


def test_h6_partial_without_control_regime(tmp_path):
    _write_runs(
        tmp_path / "runs.csv", _regime_values("noisy", gain=500.0, share=0.5)
    )
    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h6 = next(item for item in report.payload["hypotheses"] if item["code"] == "H6")
    assert h6["status"] == rp.PARTIAL
    assert h6["evidence"]["quality_measured"] is False
    assert "качество не измерено" in h6["detail"]


# ---------------------------------------------------------------------------
# Оракул ищется по спецификации среды, а не по имени
# ---------------------------------------------------------------------------
def _records(name, env_params, policy_kind, points, seeds=((1, 2),)):
    arrays = {
        field: np.zeros((len(seeds), 1), dtype=RECORD_DTYPES[field])
        for field in RECORD_FIELDS
    }
    arrays["points"][:, -1] = points
    config = {
        "name": name,
        "env": {"kind": "layered", "params": dict(env_params)},
        "agent": {"kind": "random"},
        "policy": {"kind": policy_kind},
        "n_steps": 1,
    }
    return RunRecords(
        config=config,
        seeds=[SeedPair(int(a), int(b)) for a, b in seeds],
        arrays=arrays,
    )


def test_oracle_lookup_matches_env_params_and_seeds():
    base_env = {"info_per_pull": 1, "useful_info_prob": 1.0}
    noisy_env = {"info_per_pull": 4, "useful_info_prob": 0.5}

    target = _records("pyramid-overload-base-ucb1-full_cycle", base_env, "full_cycle", [100.0])
    oracle_base = _records(
        "pyramid-overload-base-oracle-full_cycle", base_env, "oracle_full_cycle", [1000.0]
    )
    oracle_noisy = _records(
        "pyramid-overload-noisy-oracle-full_cycle", noisy_env, "oracle_full_cycle", [5000.0]
    )
    by_name = {record.name: record for record in (target, oracle_base, oracle_noisy)}
    assert _oracle_finals(by_name, target).tolist() == [1000.0]

    noisy_target = _records(
        "pyramid-overload-noisy-ucb1-full_cycle", noisy_env, "full_cycle", [200.0]
    )
    by_name[noisy_target.name] = noisy_target
    assert _oracle_finals(by_name, noisy_target).tolist() == [5000.0]

    # Другие сиды — оракул не подходит: КПД считать нельзя.
    foreign = _records("foreign", base_env, "full_cycle", [10.0], seeds=((9, 9),))
    by_name[foreign.name] = foreign
    assert _oracle_finals(by_name, foreign) is None
