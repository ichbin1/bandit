"""Визуализация результатов эксперимента."""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from experiment import ExperimentResult


def plot_cumulative_regret(
    results: Mapping[str, ExperimentResult],
    ax=None,
    show_std: bool = True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        cum = res.cumulative_regret()
        mean, std = cum.mean(axis=0), cum.std(axis=0)
        t = np.arange(1, res.n_steps + 1)
        ax.plot(t, mean, label=name)
        if show_std:
            ax.fill_between(t, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Накопленное псевдо-сожаление")
    ax.set_title("Накопленное псевдо-сожаление")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_optimal_action_rate(results: Mapping[str, ExperimentResult], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        rate = res.optimal_action_rate()
        ax.plot(np.arange(1, res.n_steps + 1), rate, label=name)
    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Доля выборов оптимальной руки")
    ax.set_title("Частота выбора оптимальной руки")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax