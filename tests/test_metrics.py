"""Метрики на синтетических записях: без прогонов и без RNG."""
from __future__ import annotations

import numpy as np
import pytest

from bandit import metrics as m
from bandit.configs import get_config
from bandit.runner import run_config
from helpers import make_records


def test_final_and_cumulative_points():
    rec = make_records(points=[[1.0, 3.0, 6.0]])
    assert m.final_points(rec).tolist() == [6.0]
    assert m.cumulative_points(rec).shape == (1, 3)


def test_optimal_arm_indicator_and_rate():
    rec = make_records(
        actions=[[0, 1, 2, 10_000]],
        kind=[[0, 0, 0, 1]],
        best_arm=[[1, 1, 1, 0]],
    )
    indicator = m.optimal_arm_indicator(rec)
    assert np.isnan(indicator[0, 3])  # мета-шаг не считается
    assert indicator[0, 1] == 1.0
    assert indicator[0, 0] == 0.0
    assert m.optimal_arm_rate(rec).tolist() == pytest.approx([1.0 / 3.0])


def test_arm_regret_metrics_ignore_meta_steps():
    rec = make_records(
        kind=[[0, 0, 1, 0]],
        arm_regret=[[0.5, 0.25, np.nan, 1.0]],
        points=[[1.0, 2.0, 2.0, 6.0]],
    )
    cumulative = m.cumulative_arm_regret(rec)
    assert cumulative[0].tolist() == pytest.approx([0.5, 0.75, 0.75, 1.75])
    assert m.total_arm_regret(rec).tolist() == pytest.approx([1.75])
    assert m.mean_arm_regret(rec).tolist() == pytest.approx([1.75 / 3])
    assert m.arm_pull_share(rec).tolist() == pytest.approx([0.75])
    assert m.points_per_pull(rec).tolist() == pytest.approx([2.0])


def test_info_shares_describe_signal_to_noise_ratio():
    rec = make_records(useful_gained=[[1, 0, 0]], info_gained=[[1, 2, 1]])
    assert m.useful_info_share(rec).tolist() == pytest.approx([0.25])
    assert m.junk_info_share(rec).tolist() == pytest.approx([0.75])

    silent = make_records(points=[[1.0, 2.0]])
    assert np.isnan(m.useful_info_share(silent)[0])
    assert np.isnan(m.junk_info_share(silent)[0])


def test_normalized_auc_of_linear_growth():
    rec = make_records(points=[np.arange(1.0, 11.0)])
    # trapezoid по точкам 1..10 = 49.5; прямоугольник T * final = 10 * 10.
    assert m.normalized_auc(rec).tolist() == pytest.approx([0.495])


def test_steps_to_fraction_and_epoch():
    rec = make_records(
        points=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        epoch_after=[[0, 0, 1, 1, 2, 2, 2, 2, 2, 2]],
    )
    final = m.final_points(rec)
    assert m.steps_to_fraction(rec, final, 0.5).tolist() == [5.0]
    assert m.steps_to_fraction(rec, final, 0.9).tolist() == [9.0]
    assert m.steps_to_epoch(rec, 1).tolist() == [3.0]
    assert m.steps_to_epoch(rec, 2).tolist() == [5.0]
    assert np.isnan(m.steps_to_epoch(rec, 5)[0])


def test_points_after_epoch_counts_only_post_crossing_points():
    rec = make_records(
        points=[[1.0, 3.0, 7.0, 11.0, 19.0]],
        epoch_after=[[0, 0, 1, 1, 2]],
    )
    # Эпоха 1 наступает на шаге 3 (индекс 2): база — 3.0 очка, итог 19.0.
    assert m.points_after_epoch(rec, 1).tolist() == pytest.approx([16.0])
    assert m.points_after_epoch(rec, 2).tolist() == pytest.approx([8.0])
    assert np.isnan(m.points_after_epoch(rec, 3)[0])
    assert "points_after_epoch_1" in m.summarize(rec)


def test_efficiency_and_point_regret():
    rec = make_records(points=[[1.0, 5.0, 9.0]])
    oracle = [12.0]
    assert m.efficiency(rec, oracle).tolist() == pytest.approx([0.75])
    regret = m.point_regret(rec, [[12.0, 12.0, 12.0]])
    assert regret[0].tolist() == pytest.approx([11.0, 7.0, 3.0])
    with pytest.raises(ValueError):
        m.efficiency(rec, [1.0, 2.0])


def test_summarize_adds_efficiency_only_with_oracle():
    rec = make_records(
        kind=[[0, 0, 1]],
        actions=[[0, 1, 10_000]],
        best_arm=[[1, 1, 0]],
        points=[[1.0, 2.0, 5.0]],
        epoch_after=[[0, 1, 1]],
    )
    plain = m.summarize(rec)
    assert "efficiency" not in plain
    assert set(plain) >= {"final_points", "cumulative_arm_regret", "final_epoch"}
    with_oracle = m.summarize(rec, oracle_final=[10.0])
    assert with_oracle["efficiency"].tolist() == pytest.approx([0.5])
    assert set(with_oracle) == set(m.METRIC_DESCRIPTIONS)


def test_branch_metrics_cover_multiplier_budget():
    """Бюджет ветвей метацикла сходится к приросту множителя на реальном прогоне."""
    records = run_config(
        get_config("pyramid-ucb1-full_cycle"),
        n_runs=1,
        n_steps=200,
        progress=False,
    )
    values = m.summarize(records)
    assert set(values) == set(m.METRIC_DESCRIPTIONS) - {"efficiency"}

    counts = {name: float(values[name][0]) for name in m.META_BRANCH_KINDS}
    assert counts["conversions"] > 0
    assert counts["arm_buys"] > 0
    assert sum(counts.values()) == float(
        ((records.kind[0] != 0) & (records.effective[0] == 1)).sum()
    )

    budget = sum(float(values[name][0]) for name in m.MULTIPLIER_METRICS)
    final_multiplier = float(values["final_multiplier"][0])
    assert final_multiplier > 1.0
    assert budget == pytest.approx(final_multiplier - 1.0, abs=1e-9)
    assert np.allclose(m.multiplier_budget_residual(records), 0.0, atol=1e-9)

    total_share = float(values["arm_pull_share"][0]) + float(values["meta_step_share"][0])
    assert total_share == pytest.approx(1.0)

    empty = m.aggregate([np.nan, np.nan])
    assert empty["n"] == 0 and np.isnan(empty["mean"])

    single = m.aggregate([2.0])
    assert single["mean"] == 2.0 and np.isnan(single["sem"])

    stats = m.aggregate([1.0, 2.0, 3.0, np.nan])
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["sem"] == pytest.approx(1.0 / np.sqrt(3))
    assert stats["ci95_lo"] < stats["mean"] < stats["ci95_hi"]
