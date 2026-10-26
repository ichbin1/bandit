"""Thompson sampling: Bernoulli (для 0/1-сигнала) и Gaussian.

Bernoulli-версия ожидает награду в ``[0, 1]``: в ``LayeredPyramidBandit``
сигнал — это базовый успех руки, поэтому ограничение выполняется.
Для непрерывных наград используйте ``ThompsonSamplingGaussian``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent


class ThompsonSamplingBernoulli(Agent):
    """Бета-апостериор на каждую руку."""

    def reset(self) -> None:
        self.alpha = np.ones(self.n_actions)
        self.beta = np.ones(self.n_actions)

    def _grow(self, extra: int) -> None:
        self.alpha = np.concatenate([self.alpha, np.ones(extra)])
        self.beta = np.concatenate([self.beta, np.ones(extra)])

    def select_action(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, action: int, reward: float) -> None:
        r = float(reward)
        if not (-1e-9 <= r <= 1.0 + 1e-9):
            raise ValueError(
                "ThompsonSamplingBernoulli ожидает награду в [0, 1]; "
                f"получено {r}. Для непрерывных наград используйте "
                "ThompsonSamplingGaussian."
            )
        self.alpha[action] += r
        self.beta[action] += 1.0 - r

    def estimate(self, action: int) -> Optional[float]:
        return float(self.alpha[action] / (self.alpha[action] + self.beta[action]))


class ThompsonSamplingGaussian(Agent):
    """TS для гауссовых наград с известной дисперсией (сопряжённый априор)."""

    def __init__(
        self,
        n_actions: int,
        sigma: float = 1.0,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if sigma <= 0 or prior_std <= 0:
            raise ValueError("sigma и prior_std должны быть положительными")
        self.sigma2 = float(sigma) ** 2
        self.prior_mean = float(prior_mean)
        self.prior_precision = 1.0 / (float(prior_std) ** 2)
        super().__init__(n_actions, seed)

    def reset(self) -> None:
        self.sum_rewards = np.zeros(self.n_actions, dtype=float)
        self.counts = np.zeros(self.n_actions, dtype=int)

    def _grow(self, extra: int) -> None:
        self.sum_rewards = np.concatenate([self.sum_rewards, np.zeros(extra)])
        self.counts = np.concatenate([self.counts, np.zeros(extra, dtype=int)])

    def select_action(self) -> int:
        precisions = self.prior_precision + self.counts / self.sigma2
        post_vars = 1.0 / precisions
        post_means = post_vars * (
            self.prior_precision * self.prior_mean + self.sum_rewards / self.sigma2
        )
        samples = self.rng.normal(post_means, np.sqrt(post_vars))
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        self.sum_rewards[action] += float(reward)

    def estimate(self, action: int) -> Optional[float]:
        if self.counts[action] == 0:
            return self.prior_mean
        return float(self.sum_rewards[action] / self.counts[action])
