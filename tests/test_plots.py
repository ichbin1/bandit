"""Графики: nan-устойчивое скользящее среднее и сохранение файлов (Agg)."""
from __future__ import annotations

import numpy as np
import pytest

from bandit import plots


def test_rolling_ignores_nan_and_uses_trailing_window():
    values = np.array([[1.0, np.nan, 3.0, 5.0]])
    smoothed = plots.rolling(values, 2)
    assert smoothed.shape == values.shape
    assert np.isfinite(smoothed).all()
    assert smoothed[0, 0] == pytest.approx(1.0)
    assert smoothed[0, 1] == pytest.approx(1.0)  # окно [1, nan] → только 1
    assert smoothed[0, 2] == pytest.approx(3.0)  # окно [nan, 3] → только 3
    assert smoothed[0, 3] == pytest.approx(4.0)  # окно [3, 5]


def test_rolling_window_larger_than_series():
    values = np.array([[1.0, 3.0]])
    smoothed = plots.rolling(values, 50)
    assert smoothed.shape == values.shape
    assert smoothed[0].tolist() == pytest.approx([1.0, 2.0])


def test_rolling_window_one_is_identity():
    values = np.array([[1.0, 2.0, 3.0]])
    assert np.array_equal(plots.rolling(values, 1), values)


def test_panes_render_without_errors(short_records):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2)
    plots.pane_cumulative_points(axes[0, 0], {"ucb1": short_records})
    plots.pane_arm_regret(axes[0, 1], {"ucb1": short_records})
    plots.pane_optimal_arm_rate(axes[1, 0], {"ucb1": short_records})
    plots.pane_state(axes[1, 1], {"ucb1": short_records}, field="n_arms_after")
    plt.close(fig)


def test_save_dashboard_and_bars(tmp_path, short_records):
    records = {"ucb1 + full_cycle": short_records}
    dashboard = plots.save_dashboard(records, tmp_path / "sub" / "dashboard.png", title="тест")
    assert dashboard.exists() and dashboard.stat().st_size > 0

    bars = plots.save_final_points_bar(records, tmp_path / "bars.png")
    assert bars.exists() and bars.stat().st_size > 0
