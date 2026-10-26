"""``PyramidBandit`` — простая экономика множителей (версия 0.1).

Оставлена как «малая» среда: два мета-действия (info → множитель, покупка
руки), без данных разной ценности и без эпох. Полезна, чтобы отделить эффект
метацикла от эффекта многоуровневой экономики.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .base import ActionSpace, MetaAction, StepEnvironment, meta_action_kind


class PyramidBandit(StepEnvironment):
    """K рук + 2 мета-действия; руки усиливаются множителем за данные."""

    META: Tuple[MetaAction, ...] = (MetaAction.INFO_TO_MULTIPLIER, MetaAction.BUY_ARM)

    def __init__(
        self,
        n_arms_init: int = 5,
        info_per_multiplier: int = 5,
        mult_per_arm_base: int = 3,
        arm_cost_increase: int = 1,
        mult_bonus: float = 0.5,
        arm_bonus: float = 0.5,
        max_arms: int = 50,
        arm_mean_sampler: Optional[Callable] = None,
        recombination_sigma: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        self.arm_mean_sampler = arm_mean_sampler or (lambda rng: rng.uniform(0, 1))
        self.n_arms_init = int(n_arms_init)
        self.info_per_multiplier = int(info_per_multiplier)
        self.mult_per_arm_base = int(mult_per_arm_base)
        self.arm_cost_increase = int(arm_cost_increase)
        self.mult_bonus = float(mult_bonus)
        self.arm_bonus = float(arm_bonus)
        self.max_arms = int(max_arms)
        self.recombination_sigma = float(recombination_sigma)
        self.reset()

    def reset(self) -> None:
        self.means = np.array([float(self.arm_mean_sampler(self.rng)) for _ in range(self.n_arms_init)])
        self.n_arms = int(self.n_arms_init)
        self.info_units = 0
        self.n_multipliers = 0
        self.arms_bought = 0
        self.total_arm_bonus = 0.0
        self.points = 0.0
        self.t = 0

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(self.n_arms, self.max_arms, self.META)

    @property
    def current_arm_cost(self) -> int:
        return self.mult_per_arm_base + self.arms_bought * self.arm_cost_increase

    @property
    def current_multiplier(self) -> float:
        diminishing_bonus = self.mult_bonus * np.log1p(self.n_multipliers)
        return 1.0 + diminishing_bonus + self.total_arm_bonus

    @property
    def can_convert_info(self) -> bool:
        return self.info_units >= self.info_per_multiplier

    @property
    def can_buy_arm(self) -> bool:
        return self.n_multipliers >= self.current_arm_cost and self.n_arms < self.max_arms

    def _apply_action(self, action: int) -> Tuple[float, float]:
        kind = meta_action_kind(action)
        if kind is None:
            base = float(self.rng.random() < self.means[action])
            gained = base * self.current_multiplier
            self.points += gained
            self.info_units += 1
            return base, gained
        if kind is MetaAction.INFO_TO_MULTIPLIER:
            if self.info_units >= self.info_per_multiplier:
                self.info_units -= self.info_per_multiplier
                self.n_multipliers += 1
            return 0.0, 0.0
        if kind is MetaAction.BUY_ARM:
            cost = self.current_arm_cost
            if self.n_multipliers >= cost and self.n_arms < self.max_arms:
                self.n_multipliers -= cost
                self.total_arm_bonus += self.arm_bonus + 0.1 * self.arms_bought
                self.arms_bought += 1
                top_indices = np.argsort(self.means)[-2:]
                p1 = float(self.means[top_indices[0]])
                p2 = float(self.means[top_indices[1]])
                synergy_mean = 1.0 - (1.0 - p1) * (1.0 - p2)
                new_mean = float(
                    np.clip(synergy_mean + self.rng.normal(0, self.recombination_sigma), 0.0, 1.0)
                )
                self.means = np.append(self.means, new_mean)
                self.n_arms += 1
            return 0.0, 0.0
        raise ValueError(f"PyramidBandit не поддерживает мета-действие {kind.name}")

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"PyramidBandit(n_arms={self.n_arms}, t={self.t}, points={self.points:.1f})"
