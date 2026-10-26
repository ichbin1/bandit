"""UCB1 — детерминированная верхняя доверительная граница.

Формула из версии 0.1 сохранена без изменений (вместе с tie-break через
``rng.choice``), чтобы golden-тесты продолжали совпадать.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent


class UCB1(Agent):
    """``value + sqrt(c * ln(t) / n)``; неисследованные руки берутся первыми."""

    def __init__(self, n_actions: int, c: float = 2.0, seed: Optional[int] = None) -> None:
        if c < 0:
            raise ValueError("c должно быть неотрицательным")
        self.c = float(c)
        super().__init__(n_actions, seed)

    def reset(self) -> None:
        self.values = np.zeros(self.n_actions)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.t = 0

    def _grow(self, extra: int) -> None:
        self.values = np.concatenate([self.values, np.zeros(extra)])
        self.counts = np.concatenate([self.counts, np.zeros(extra, dtype=int)])

    def select_action(self) -> int:
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(self.rng.choice(untried))
        ucb = self.values + np.sqrt(self.c * np.log(self.t) / self.counts)
        return self._argmax_random(ucb)

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.counts[action] += 1
        self.values[action] += (float(reward) - self.values[action]) / self.counts[action]

    def estimate(self, action: int) -> Optional[float]:
        return float(self.values[action])
