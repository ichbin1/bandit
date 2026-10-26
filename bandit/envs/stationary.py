"""Стационарные бандиты: Bernoulli и Gaussian.

Нужны как «песочница» для агентов: там, где сожаление определено корректно,
видно, что метрики и статистика работают, и только потом те же агенты
переносятся в ``LayeredPyramidBandit``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import BanditEnvironment


class BernoulliBandit(BanditEnvironment):
    """Каждая рука возвращает 1 с вероятностью ``means[i]``, иначе 0."""

    def __init__(
        self,
        n_arms: int = 10,
        means: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_arms, seed)
        if means is None:
            arr = self.rng.uniform(0.0, 1.0, self._n_arms_fixed)
        else:
            arr = np.asarray(means, dtype=float).reshape(-1)
            if arr.size != self._n_arms_fixed:
                raise ValueError(
                    f"means должно содержать {self._n_arms_fixed} значений, получено {arr.size}"
                )
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError("Вероятности должны лежать в [0, 1]")
        self.means = arr
        self.reset()

    def _draw(self, action: int) -> float:
        return float(self.rng.random() < self.means[action])


class GaussianBandit(BanditEnvironment):
    """Рука ``i`` возвращает ``N(means[i], std)``."""

    def __init__(
        self,
        n_arms: int = 10,
        means: Optional[Sequence[float]] = None,
        std: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_arms, seed)
        if means is None:
            arr = self.rng.normal(0.0, 1.0, self._n_arms_fixed)
        else:
            arr = np.asarray(means, dtype=float).reshape(-1)
            if arr.size != self._n_arms_fixed:
                raise ValueError(
                    f"means должно содержать {self._n_arms_fixed} значений, получено {arr.size}"
                )
        if std <= 0:
            raise ValueError("std должно быть положительным")
        self.std = float(std)
        self.means = arr
        self.reset()

    def _draw(self, action: int) -> float:
        return float(self.rng.normal(self.means[action], self.std))
