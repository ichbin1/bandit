"""Алгоритмы выбора действий.

Три группы:

* базовые линии из постановки задачи — ``RandomAgent`` (нижняя граница),
  ``GreedyAgent`` (чистая эксплуатация), ``EpsilonGreedy`` (ε-жадный);
* правила обучения, явно названные в документе, — ``RescorlaWagnerAgent``
  (ΔV = α(R − V)) и ``UCB1``;
* байесовские оценки — Thompson sampling.

Все агенты выбирают только руки; мета-действия — слой ``bandit.meta``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """Равномерно случайный выбор: нижняя граница по награде и регрету."""

    def reset(self) -> None:
        pass

    def _grow(self, extra: int) -> None:
        pass

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, action: int, reward: float) -> None:
        pass


class GreedyAgent(Agent):
    """Чистая эксплуатация: всегда argmax текущих оценок, без исследования.

    Нужен как «вырожденная» базовая линия: показывает, что даёт метацикл,
    когда агент вообще не исследует (он «застревает» на первой руке).
    """

    def __init__(
        self,
        n_actions: int,
        initial_value: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.initial_value = float(initial_value)
        super().__init__(n_actions, seed)

    def reset(self) -> None:
        self.values = np.full(self.n_actions, self.initial_value)
        self.counts = np.zeros(self.n_actions, dtype=int)

    def _grow(self, extra: int) -> None:
        self.values = np.concatenate([self.values, np.full(extra, self.initial_value)])
        self.counts = np.concatenate([self.counts, np.zeros(extra, dtype=int)])

    def select_action(self) -> int:
        return self._argmax_random(self.values)

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (float(reward) - self.values[action]) / n

    def estimate(self, action: int) -> Optional[float]:
        return float(self.values[action])


class EpsilonGreedy(Agent):
    """ε-жадный агент с опциональным затуханием ε.

    ``epsilon_decay < 1`` даёт «затухающий» вариант: исследование падает,
    и видно, помогает ли это в нестационарной среде (см. ``PROTOCOL.md``).
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        epsilon_min: Optional[float] = None,
        epsilon_decay: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon должен быть в [0, 1]")
        if not 0.0 < float(epsilon_decay) <= 1.0:
            raise ValueError("epsilon_decay должен быть в (0, 1]")
        self.epsilon_0 = float(epsilon)
        self.epsilon_min = float(epsilon) if epsilon_min is None else float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.initial_value = float(initial_value)
        self.epsilon = self.epsilon_0
        super().__init__(n_actions, seed)

    def reset(self) -> None:
        self.values = np.full(self.n_actions, self.initial_value)
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.epsilon = self.epsilon_0

    def _grow(self, extra: int) -> None:
        self.values = np.concatenate([self.values, np.full(extra, self.initial_value)])
        self.counts = np.concatenate([self.counts, np.zeros(extra, dtype=int)])

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._argmax_random(self.values)

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (float(reward) - self.values[action]) / n
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def estimate(self, action: int) -> Optional[float]:
        return float(self.values[action])


class RescorlaWagnerAgent(Agent):
    """Правило Рескорлы–Вагнера: ``ΔV = α · (R − V)``.

    Постоянная скорость обучения ``α`` означает «забывание» старого опыта,
    поэтому агент лучше ε-жадного отслеживает изменения среды (это второй
    механизм адаптации в гипотезе, наряду с метациклом).

    Выбор действия — ε-жадный по ``V``.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        initial_value: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha должен быть в (0, 1]")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon должен быть в [0, 1]")
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.initial_value = float(initial_value)
        super().__init__(n_actions, seed)

    def reset(self) -> None:
        self.values = np.full(self.n_actions, self.initial_value)

    def _grow(self, extra: int) -> None:
        self.values = np.concatenate([self.values, np.full(extra, self.initial_value)])

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self._argmax_random(self.values)

    def update(self, action: int, reward: float) -> None:
        self.values[action] += self.alpha * (float(reward) - self.values[action])

    def estimate(self, action: int) -> Optional[float]:
        return float(self.values[action])
