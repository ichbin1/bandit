"""Графики прогонов.

Все функции получают ``records_by_name`` — отображение «подпись → RunRecords»
и рисуют среднее по прогонам с 95% коридором (±1.96 SEM по прогонам).

Бэкенд жёстко ``Agg``: скрипты не должны открывать окна и обязаны работать
без дисплея (CI, OneDrive-папка, ssh).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (после matplotlib.use)
import numpy as np  # noqa: E402

from . import metrics as m  # noqa: E402
from .runner import RunRecords  # noqa: E402

RecordsMap = Dict[str, RunRecords]
_Z95 = 1.96


def _nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Среднее без предупреждений про пустые срезы."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _nanstd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Выборочное СКО без предупреждений про пустые срезы."""
    values = np.asarray(values, dtype=float)
    mean = np.expand_dims(_nanmean(values, axis=axis), axis)
    finite = np.isfinite(values)
    squared = np.where(finite, (values - mean) ** 2, 0.0)
    count = finite.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where(count > 1, squared.sum(axis=axis) / np.maximum(count - 1, 1), np.nan)
    return np.sqrt(variance)


def _band(ax, x: np.ndarray, values: np.ndarray, label: str, color=None, alpha: float = 0.18):
    """Нарисовать среднее и 95% коридор по оси прогонов."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    mean = _nanmean(values, axis=0)
    ax.plot(x, mean, label=label, color=color, linewidth=1.6)
    if values.shape[0] > 1:
        sem = _nanstd(values, axis=0) / np.sqrt(values.shape[0])
        ax.fill_between(x, mean - _Z95 * sem, mean + _Z95 * sem, color=color, alpha=alpha, linewidth=0)


def _colors(n: int) -> Iterable:
    cmap = plt.get_cmap("tab10")
    for i in range(n):
        yield cmap(i % 10)


def _steps(records: RunRecords) -> np.ndarray:
    return np.arange(1, records.n_steps + 1)


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее по *завершающемуся* окну (nan-устойчивое).

    Окно ``window`` смотрит назад (никакого «заглядывания в будущее»), поэтому
    график на шаге ``t`` показывает только прошлое. Окно автоматически
    укорачивается в начале ряда и при ``window > n_steps``.
    """
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    if values.ndim == 1:
        values = values[None, :]
    window = min(int(window), values.shape[1])

    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    totals = np.cumsum(filled, axis=1)
    counts = np.cumsum(valid.astype(float), axis=1)

    zeros = np.zeros((values.shape[0], 1), dtype=float)
    totals = np.concatenate([zeros, totals], axis=1)  # длина T + 1, C[0] = 0
    counts = np.concatenate([zeros, counts], axis=1)

    right_total, right_counts = totals[:, 1:], counts[:, 1:]
    left_total = np.concatenate([np.zeros((values.shape[0], window - 1)), totals[:, : totals.shape[1] - window]], axis=1)
    left_counts = np.concatenate([np.zeros((values.shape[0], window - 1)), counts[:, : counts.shape[1] - window]], axis=1)

    total = right_total - left_total
    count = right_counts - left_counts
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def pane_cumulative_points(ax, records_by_name: RecordsMap, title: str = "Накопленные очки"):
    for (name, records), color in zip(records_by_name.items(), _colors(len(records_by_name))):
        _band(ax, _steps(records), m.cumulative_points(records), name, color=color)
    ax.set_title(title)
    ax.set_xlabel("шаг")
    ax.set_ylabel("очки")
    ax.grid(alpha=0.3)


def pane_arm_regret(ax, records_by_name: RecordsMap, title: str = "Сожаление по рукам"):
    for (name, records), color in zip(records_by_name.items(), _colors(len(records_by_name))):
        _band(ax, _steps(records), m.cumulative_arm_regret(records), name, color=color)
    ax.set_title(title)
    ax.set_xlabel("шаг")
    ax.set_ylabel("недобор (единицы награды)")
    ax.grid(alpha=0.3)


def pane_optimal_arm_rate(
    ax,
    records_by_name: RecordsMap,
    window: int = 50,
    title: str = "Доля выбора лучшей руки (скольз. среднее)",
):
    for (name, records), color in zip(records_by_name.items(), _colors(len(records_by_name))):
        rate = m.optimal_arm_indicator(records)
        _band(ax, _steps(records), rolling(rate, window), name, color=color)
    ax.set_title(title)
    ax.set_xlabel("шаг")
    ax.set_ylabel("доля")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def pane_state(
    ax,
    records_by_name: RecordsMap,
    field: str = "epoch_after",
    title: str = "Состояние среды",
    ylabel: Optional[str] = None,
):
    for (name, records), color in zip(records_by_name.items(), _colors(len(records_by_name))):
        _band(ax, _steps(records), getattr(records, field), name, color=color)
    ax.set_title(title)
    ax.set_xlabel("шаг")
    ax.set_ylabel(ylabel or field)
    ax.grid(alpha=0.3)


def pane_final_points_bar(
    ax,
    records_by_name: RecordsMap,
    title: str = "Итоговые очки (среднее ± SEM)",
):
    names = list(records_by_name)
    finals = [m.final_points(records_by_name[n]) for n in names]
    means = np.array([float(_nanmean(f)) for f in finals])
    sems = np.array([
        float(_nanstd(f) / np.sqrt(len(f))) if len(f) > 1 else 0.0 for f in finals
    ])
    colors = list(_colors(len(names)))
    positions = np.arange(len(names))
    ax.bar(positions, means, yerr=1.96 * sems, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("очки")
    ax.grid(alpha=0.3, axis="y")


def save_dashboard(
    records_by_name: RecordsMap,
    path,
    title: str = "Метацикл обучения",
    window: int = 50,
    figsize: Tuple[float, float] = (13.0, 9.0),
) -> Path:
    """Сохранить дашборд 2×2 с основными метриками и вернуть путь к файлу."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    pane_cumulative_points(axes[0, 0], records_by_name)
    pane_arm_regret(axes[0, 1], records_by_name)
    pane_optimal_arm_rate(axes[1, 0], records_by_name, window=window)
    pane_state(
        axes[1, 1],
        records_by_name,
        field="epoch_after",
        title="Эпохи среды (сколько раз среда менялась)",
        ylabel="эпоха",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_final_points_bar(records_by_name: RecordsMap, path, title: str = "Итоговые очки") -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    pane_final_points_bar(ax, records_by_name, title=title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
