"""Базовый интерфейс агента.

Агент отвечает только за выбор руки. Мета-действия — это отдельный слой
(``bandit.meta``), который решает, когда «инвестировать» вместо «добывать».
Такое разделение позволяет честно факторизовать эксперимент:

* агент × мета-политика — 2D-матрица, а не один смешанный цикл;
* верхние границы (оракул) считаются той же машинерией.

Раньше агент получал ``env.n_actions``, включавшее мета-действия, и сам решал,
когда их применять, — из-за этого нельзя было отделить вклад агента от
вклада мета-расписания.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Agent(ABC):
    """Интерфейс агента: выбор руки и обновление по награде."""

    def __init__(self, n_actions: int, seed: Optional[int] = None) -> None:
        if int(n_actions) <= 0:
            raise ValueError("n_actions должно быть положительным")
        self.rng = np.random.default_rng(seed)
        self.n_actions = int(n_actions)
        self.reset()

    # ------------------------------------------------------------------
    # Обязательное к реализации
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        """Сбросить накопленную статистику (RNG не пересоздаётся)."""

    @abstractmethod
    def select_action(self) -> int:
        """Вернуть индекс руки в ``[0, n_actions)``."""

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Учесть наблюдаемую награду за действие."""

    @abstractmethod
    def _grow(self, extra: int) -> None:
        """Расширить внутренние массивы на ``extra`` действий."""

    # ------------------------------------------------------------------
    # Рост пространства действий
    # ------------------------------------------------------------------
    def grow_to(self, n_actions: int) -> None:
        """Учесть, что рук стало больше (значения старых сохраняются)."""
        n_actions = int(n_actions)
        if n_actions < 0:
            raise ValueError("n_actions не может быть отрицательным")
        if n_actions > self.n_actions:
            self._grow(n_actions - self.n_actions)
            self.n_actions = n_actions

    def set_n_actions(self, n_actions: int) -> None:
        """Совместимость с версией 0.1 (рост разрешён, сжатие — нет)."""
        n_actions = int(n_actions)
        if n_actions < self.n_actions:
            raise ValueError(
                "set_n_actions поддерживает только рост: "
                f"{self.n_actions} → {n_actions}. Для сброса используйте reset()."
            )
        self.grow_to(n_actions)

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def _argmax_random(self, values: np.ndarray) -> int:
        """Случайный выбор среди максимумов (честный tie-break).

        ``rng.choice`` вызывается даже при единственном кандидате: это
        сохраняет поток RNG версии 0.1 и позволяет golden-тестам фиксировать
        точные значения очков.
        """
        best = np.max(np.asarray(values))
        candidates = np.flatnonzero(np.asarray(values) == best)
        return int(self.rng.choice(candidates))

    def estimate(self, action: int) -> Optional[float]:
        """Текущая оценка ценности действия (для диагностики), если есть."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"{type(self).__name__}(n_actions={self.n_actions})"
