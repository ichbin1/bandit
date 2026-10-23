"""Алгоритмы (агенты) для задачи о многоруком бандите."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Agent(ABC):
    def __init__(self, n_actions_init: int, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions_init
        self.reset()

    def set_n_actions(self, n_actions: int) -> None:
        if n_actions > self.n_actions:
            self._grow(n_actions - self.n_actions)
            self.n_actions = n_actions

    # FIX #3: служебный метод в базовом классе
    def _argmax_random(self, values: np.ndarray) -> int:
        best = values.max()
        candidates = np.flatnonzero(values == best)
        return int(self.rng.choice(candidates))

    @abstractmethod
    def _grow(self, k: int) -> None: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def select_action(self) -> int: ...
    @abstractmethod
    def update(self, action: int, reward: float) -> None: ...


class RandomAgent(Agent):
    """Базовый уровень (случайный выбор)."""

    def reset(self) -> None:
        pass

    # FIX #5: реализация абстрактного метода и корректное поле
    def _grow(self, k: int) -> None:
        pass

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, action: int, reward: float) -> None:
        pass


class EpsilonGreedy(Agent):
    def __init__(self, n_actions, epsilon=0.1, initial_value=0.0, seed=None):
        self.epsilon = epsilon
        self.initial_value = initial_value
        super().__init__(n_actions, seed)

    def reset(self):
        self.values = np.full(self.n_actions, self.initial_value)
        self.counts = np.zeros(self.n_actions, dtype=int)

    def _grow(self, k):
        self.values = np.concatenate([self.values, np.full(k, self.initial_value)])
        self.counts = np.concatenate([self.counts, np.zeros(k, dtype=int)])

    def select_action(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._argmax_random(self.values)

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


class UCB1(Agent):
    def __init__(self, n_actions, c=2.0, seed=None):
        self.c = c
        super().__init__(n_actions, seed)

    def reset(self):
        self.values = np.zeros(self.n_actions)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.t = 0

    def _grow(self, k):
        self.values = np.concatenate([self.values, np.zeros(k)])
        self.counts = np.concatenate([self.counts, np.zeros(k, dtype=int)])

    def select_action(self):
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(self.rng.choice(untried))
        ucb = self.values + np.sqrt(self.c * np.log(self.t) / self.counts)
        return self._argmax_random(ucb)

    def update(self, action, reward):
        self.t += 1
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]


class ThompsonSamplingBernoulli(Agent):
    def reset(self):
        self.alpha = np.ones(self.n_actions)
        self.beta = np.ones(self.n_actions)

    def _grow(self, k):
        self.alpha = np.concatenate([self.alpha, np.ones(k)])
        self.beta = np.concatenate([self.beta, np.ones(k)])

    def select_action(self):
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward


class ThompsonSamplingGaussian(Agent):
    """TS для гауссовых наград с известной дисперсией (сопряжённый нормальный априор)."""

    def __init__(
        self,
        n_actions: int,
        sigma: float = 1.0,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.sigma2 = float(sigma) ** 2
        self.prior_mean = float(prior_mean)
        self.prior_precision = 1.0 / (float(prior_std) ** 2)
        super().__init__(n_actions, seed)

    # FIX #4: используем self.n_actions и добавляем _grow
    def reset(self) -> None:
        self.sum_rewards = np.zeros(self.n_actions, dtype=float)
        self.counts = np.zeros(self.n_actions, dtype=int)

    def _grow(self, k: int) -> None:
        self.sum_rewards = np.concatenate([self.sum_rewards, np.zeros(k)])
        self.counts = np.concatenate([self.counts, np.zeros(k, dtype=int)])

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