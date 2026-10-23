"""Окружения (многорукие бандиты)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


class BanditEnvironment(ABC):
    """Базовый стационарный бандит: у каждой руки фиксированное распределение."""

    def __init__(self, n_arms: int, seed: Optional[int] = None) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(seed)
        self.means: np.ndarray

    @property
    def optimal_mean(self) -> float:
        return float(self.means.max())

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    @abstractmethod
    def pull(self, action: int) -> float:
        raise NotImplementedError


class BernoulliBandit(BanditEnvironment):
    def __init__(self, n_arms, means=None, seed=None):
        super().__init__(n_arms, seed)
        if means is None:
            means = self.rng.uniform(0.0, 1.0, self.n_arms)
        means_arr = np.asarray(means, dtype=float)
        if means_arr.shape != (self.n_arms,):
            raise ValueError("means must have length n_arms")
        if np.any(means_arr < 0) or np.any(means_arr > 1):
            raise ValueError("Bernoulli means must be in [0, 1]")
        self.means = means_arr

    def pull(self, action: int) -> float:
        return float(self.rng.random() < self.means[action])


class GaussianBandit(BanditEnvironment):
    def __init__(self, n_arms, means=None, std=1.0, seed=None):
        super().__init__(n_arms, seed)
        if means is None:
            means = self.rng.normal(0.0, 1.0, self.n_arms)
        means_arr = np.asarray(means, dtype=float)
        if means_arr.shape != (self.n_arms,):
            raise ValueError("means must have length n_arms")
        self.means = means_arr
        self.std = float(std)

    def pull(self, action: int) -> float:
        return float(self.rng.normal(self.means[action], self.std))


class PyramidBandit:
    """Оригинальная простая среда (см. предыдущую версию). Оставлена для совместимости."""
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
    ):
        self.rng = np.random.default_rng(seed)
        self.arm_mean_sampler = arm_mean_sampler or (lambda rng: rng.uniform(0, 1))
        self.n_arms_init = n_arms_init
        self.info_per_multiplier = info_per_multiplier
        self.mult_per_arm_base = mult_per_arm_base
        self.arm_cost_increase = arm_cost_increase
        self.mult_bonus = mult_bonus
        self.arm_bonus = arm_bonus
        self.max_arms = max_arms
        self.recombination_sigma = recombination_sigma
        self.reset()

    def reset(self) -> None:
        self.means = np.array([self.arm_mean_sampler(self.rng) for _ in range(self.n_arms_init)])
        self.n_arms = self.n_arms_init
        self.info_units = 0
        self.n_multipliers = 0
        self.arms_bought = 0
        self.total_arm_bonus = 0.0
        self.total_points = 0.0
        self.t = 0

    @property
    def n_actions(self) -> int:
        return self.n_arms + 2

    @property
    def optimal_mean(self) -> float:
        return float(self.means.max())

    @property
    def current_arm_cost(self) -> int:
        return self.mult_per_arm_base + self.arms_bought * self.arm_cost_increase

    @property
    def current_multiplier(self) -> float:
        diminishing_bonus = self.mult_bonus * np.log1p(self.n_multipliers)
        return 1.0 + diminishing_bonus + self.total_arm_bonus

    def pull(self, action: int) -> float:
        self.t += 1
        if action < self.n_arms:
            base = float(self.rng.random() < self.means[action])
            self.total_points += base * self.current_multiplier
            self.info_units += 1
            return base
        if action == self.n_arms:
            if self.info_units >= self.info_per_multiplier:
                self.info_units -= self.info_per_multiplier
                self.n_multipliers += 1
            return 0.0
        if action == self.n_arms + 1:
            cost = self.current_arm_cost
            if self.n_multipliers >= cost and self.n_arms < self.max_arms:
                self.n_multipliers -= cost
                self.total_arm_bonus += (self.arm_bonus + 0.1 * self.arms_bought)
                self.arms_bought += 1
                top_indices = np.argsort(self.means)[-2:]
                p1 = self.means[top_indices[0]]
                p2 = self.means[top_indices[1]]
                synergy_mean = 1.0 - (1.0 - p1) * (1.0 - p2)
                new_mean = synergy_mean + self.rng.normal(0, self.recombination_sigma)
                new_mean = np.clip(new_mean, 0.0, 1.0)
                self.means = np.append(self.means, new_mean)
                self.n_arms += 1
            return 0.0
        raise ValueError(f"Invalid action {action}")


class LayeredPyramidBandit:
    """PyramidBandit + многоуровневая экономика данных и метацикл.

    Ключевые нововведения (по сравнению с PyramidBandit):
    ------------------------------------------------------------------
    1. **Данные разной ценности (tier).**
       У каждой руки есть `tier`. Пулл даёт `+1` данных этого tier'а.
       Чем выше tier — тем реже данные и тем они ценнее для технологий.

    2. **Технологии = комбинация данных.**
       `tech_costs[epoch]` = рецепт (сколько каких tier'ов нужно),
       чтобы перейти на следующий epoch. Нужна смесь — нельзя
       накопить только один тип.

    3. **Эпохи (epoch) и лимиты.**
       Прогресс возможен только последовательно: tech_0 → tech_1 → ...
       Это и есть "лимитирующие факторы" — advancement заблокирован,
       пока не собран нужный tech.

    4. **Открытие новых данных.**
       При смене эпохи разблокируется набор новых рук с более высоким
       tier и (из-за tech_arm_mean_boost * tier) более высоким mean.

    5. **Отложенная выгода.**
       `craft_gate` и `craft_amplifier` возвращают 0.0 немедленной
       награды. Только накопленный `tech_multiplier_total` и
       `amplifier_multiplier_total` дают выигрыш на будущих пуллах.

    Пространство действий: `[0, n_arms)` руки,
        `n_arms + 0` — info → multiplier (как в PyramidBandit)
        `n_arms + 1` — купить руку (как в PyramidBandit)
        `n_arms + 2` — craft gate / advance epoch (НОВОЕ)
        `n_arms + 3` — craft amplifier (НОВОЕ)
    ------------------------------------------------------------------
    """

    # Дефолтные рецепты технологий: tech_costs[i] — стоимость перехода из epoch i в i+1
    DEFAULT_TECH_COSTS: List[Dict[int, int]] = [
        {0: 8, 1: 4},           # epoch 0 → 1: 8 basic + 4 advanced
        {0: 10, 1: 6, 2: 3},    # epoch 1 → 2: 10 basic + 6 adv + 3 rare
    ]
    # Тиры стартовых рук (длина = n_arms_init)
    DEFAULT_INITIAL_TIERS: Sequence[int] = (0, 0, 0, 1, 1)
    # Тиры разблокируемых рук на каждом advancement
    DEFAULT_UNLOCK_TIERS: List[Sequence[int]] = [
        (1, 2, 2),   # epoch 0→1: +1 advanced, +2 rare
        (2, 3, 3),   # epoch 1→2: +1 rare, +2 legendary
    ]
    DEFAULT_AMPLIFIER_BASE_COST: Dict[int, int] = {0: 5, 1: 3}

    def __init__(
        self,
        n_arms_init: int = 5,
        n_levels: int = 3,
        info_per_multiplier: int = 5,
        mult_per_arm_base: int = 3,
        arm_cost_increase: int = 1,
        mult_bonus: float = 0.5,
        arm_bonus: float = 0.5,
        arm_bonus_increase: float = 0.1,
        max_arms: int = 50,
        # --- новые параметры экономики данных ---
        initial_tiers: Optional[Sequence[int]] = None,
        unlock_tiers: Optional[List[Sequence[int]]] = None,
        tech_costs: Optional[List[Dict[int, int]]] = None,
        tech_multiplier_bonus: float = 1.0,
        tech_arm_mean_boost: float = 0.15,
        amplifier_base_cost: Optional[Dict[int, int]] = None,
        amplifier_mult_bonus: float = 0.3,
        amplifier_cost_growth: float = 1.5,
        # --- общие ---
        arm_mean_sampler: Optional[Callable] = None,
        recombination_sigma: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.arm_mean_sampler = arm_mean_sampler or (lambda rng: rng.uniform(0, 1))
        self.n_arms_init = int(n_arms_init)
        self.n_levels = int(n_levels)
        self.info_per_multiplier = int(info_per_multiplier)
        self.mult_per_arm_base = int(mult_per_arm_base)
        self.arm_cost_increase = int(arm_cost_increase)
        self.mult_bonus = float(mult_bonus)
        self.arm_bonus = float(arm_bonus)
        self.arm_bonus_increase = float(arm_bonus_increase)
        self.max_arms = int(max_arms)
        self.tech_multiplier_bonus = float(tech_multiplier_bonus)
        self.tech_arm_mean_boost = float(tech_arm_mean_boost)
        self.amplifier_mult_bonus = float(amplifier_mult_bonus)
        self.amplifier_cost_growth = float(amplifier_cost_growth)
        self.recombination_sigma = float(recombination_sigma)

        self.initial_tiers = tuple(
            (initial_tiers or self.DEFAULT_INITIAL_TIERS)[:self.n_arms_init]
        )
        self.unlock_tiers = list(unlock_tiers or self.DEFAULT_UNLOCK_TIERS)
        self.tech_costs = list(tech_costs or self.DEFAULT_TECH_COSTS)
        self.amplifier_base_cost = dict(
            amplifier_base_cost or self.DEFAULT_AMPLIFIER_BASE_COST
        )
        self.reset()

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------
    def reset(self) -> None:
        # --- стартовый набор рук (разные тиры) ---
        means_list, tiers_list = [], []
        for i in range(self.n_arms_init):
            base_mean = float(self.arm_mean_sampler(self.rng))
            tier = int(self.initial_tiers[i]) if i < len(self.initial_tiers) else 0
            mean = float(np.clip(
                base_mean + self.tech_arm_mean_boost * tier, 0.0, 1.0
            ))
            means_list.append(mean)
            tiers_list.append(tier)
        self.means = np.array(means_list)
        self.arm_tiers = np.array(tiers_list, dtype=int)
        self.n_arms = int(self.n_arms_init)

        # --- общая экономика ---
        self.info_units = 0
        self.data: Dict[int, int] = {}
        self.n_multipliers = 0
        self.arms_bought = 0
        self.total_arm_bonus = 0.0
        self.total_points = 0.0
        self.t = 0

        # --- метацикл ---
        self.epoch = 0
        self.techs_crafted = 0
        self.amplifiers_crafted = 0
        self.tech_multiplier_total = 0.0
        self.amplifier_multiplier_total = 0.0

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        # K рук + 4 мета-действия (info→mult, buy_arm, craft_gate, craft_amplifier)
        return self.n_arms + 4

    @property
    def optimal_mean(self) -> float:
        return float(self.means.max())

    @property
    def current_arm_cost(self) -> int:
        return self.mult_per_arm_base + self.arms_bought * self.arm_cost_increase

    @property
    def current_multiplier(self) -> float:
        diminishing = self.mult_bonus * np.log1p(self.n_multipliers)
        return (
            1.0
            + diminishing
            + self.total_arm_bonus
            + self.tech_multiplier_total
            + self.amplifier_multiplier_total
        )

    @property
    def current_tech_cost(self) -> Optional[Dict[int, int]]:
        """Рецепт для advancement (None — если финальная эпоха)."""
        if self.epoch >= self.n_levels - 1:
            return None
        idx = min(self.epoch, len(self.tech_costs) - 1)
        return dict(self.tech_costs[idx])

    @property
    def current_amplifier_cost(self) -> Dict[int, int]:
        factor = self.amplifier_cost_growth ** self.amplifiers_crafted
        return {t: max(1, int(np.ceil(q * factor))) for t, q in self.amplifier_base_cost.items()}

    @property
    def can_craft_gate(self) -> bool:
        cost = self.current_tech_cost
        if cost is None:
            return False
        return all(self.data.get(t, 0) >= q for t, q in cost.items())

    @property
    def can_craft_amplifier(self) -> bool:
        cost = self.current_amplifier_cost
        return all(self.data.get(t, 0) >= q for t, q in cost.items())

    # ------------------------------------------------------------------
    # Действия
    # ------------------------------------------------------------------
    def pull(self, action: int) -> float:
        self.t += 1
        if 0 <= action < self.n_arms:
            return self._pull_arm(action)
        if action == self.n_arms:
            return self._convert_info()
        if action == self.n_arms + 1:
            return self._buy_arm()
        if action == self.n_arms + 2:
            return self._craft_gate()
        if action == self.n_arms + 3:
            return self._craft_amplifier()
        raise ValueError(f"Invalid action {action}")

    def _pull_arm(self, action: int) -> float:
        tier = int(self.arm_tiers[action])
        base = float(self.rng.random() < self.means[action])
        # Базовый (0/1) сигнал — для обучения агента; множитель — только в счёт.
        self.total_points += base * self.current_multiplier
        self.info_units += 1
        self.data[tier] = self.data.get(tier, 0) + 1
        return base

    def _convert_info(self) -> float:
        if self.info_units >= self.info_per_multiplier:
            self.info_units -= self.info_per_multiplier
            self.n_multipliers += 1
        return 0.0

    def _buy_arm(self) -> float:
        cost = self.current_arm_cost
        if self.n_multipliers >= cost and self.n_arms < self.max_arms:
            self.n_multipliers -= cost
            self.total_arm_bonus += (
                self.arm_bonus + self.arm_bonus_increase * self.arms_bought
            )
            self.arms_bought += 1
            # Синергия топ-2 + рекомбинация Вайцмана
            top = np.argsort(self.means)[-2:]
            p1, p2 = float(self.means[top[0]]), float(self.means[top[1]])
            synergy = 1.0 - (1.0 - p1) * (1.0 - p2)
            new_mean = float(np.clip(
                synergy + self.rng.normal(0, self.recombination_sigma), 0.0, 1.0
            ))
            new_tier = int(max(self.arm_tiers[top[0]], self.arm_tiers[top[1]]))
            self.means = np.append(self.means, new_mean)
            self.arm_tiers = np.append(self.arm_tiers, new_tier)
            self.n_arms += 1
        return 0.0

    def _craft_gate(self) -> float:
        """Комбинирует данные разных tier'ов в технологию и продвигает эпоху.
        Немедленной награды не даёт — только future multiplier + новые руки."""
        cost = self.current_tech_cost
        if cost is None:
            return 0.0
        if not all(self.data.get(t, 0) >= q for t, q in cost.items()):
            return 0.0
        for t, q in cost.items():
            self.data[t] -= q

        self.techs_crafted += 1
        self.epoch += 1
        self.tech_multiplier_total += self.tech_multiplier_bonus

        # Разблокируем новые руки более высокого тира
        idx = self.epoch - 1
        if idx < len(self.unlock_tiers):
            for tier in self.unlock_tiers[idx]:
                if self.n_arms >= self.max_arms:
                    break
                base_mean = float(self.arm_mean_sampler(self.rng))
                new_mean = float(np.clip(
                    base_mean + self.tech_arm_mean_boost * int(tier), 0.0, 1.0
                ))
                self.means = np.append(self.means, new_mean)
                self.arm_tiers = np.append(self.arm_tiers, int(tier))
                self.n_arms += 1
        return 0.0

    def _craft_amplifier(self) -> float:
        """Побочная ветка: стоимость растёт экспоненциально, эффект линейный.
        Тоже без немедленной награды."""
        cost = self.current_amplifier_cost
        if not all(self.data.get(t, 0) >= q for t, q in cost.items()):
            return 0.0
        for t, q in cost.items():
            self.data[t] -= q
        self.amplifiers_crafted += 1
        self.amplifier_multiplier_total += self.amplifier_mult_bonus
        return 0.0