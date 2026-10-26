"""``LayeredPyramidBandit`` — основная среда гипотезы.

Отличия от ``PyramidBandit``
----------------------------
1. Данные разной ценности (tiers): рука тира ``t`` даёт данные тира ``t``.
2. Технологии («ворота» эпох) требуют *комбинации* разных тиров — то есть
   Exploration обязан быть разнообразным, а не сводиться к одной лучшей руке.
3. Амплификаторы — побочная ветка с растущей стоимостью и линейным эффектом.
4. Всё это даёт **отложенную выгоду**: мета-действия не приносят очков сразу.
5. Информационный поток отделён от награды: ``info_per_pull``,
   ``useful_info_prob`` и ``junk_feeds_multiplier`` задают перегрузку (H6),
   не меняя ни награду, ни очки за вытягивание. Розыгрыш полезности идёт по
   отдельному потоку RNG, поэтому шум не сдвигает последовательность наград.

Важно: ``reward`` (сигнал для обучения агента) — это базовый 0/1 успех руки.
Множитель попадает только в счёт очков (``points_gained``). Иначе агент
оптимизировал бы мета-экономику вместо качества выбора, и сравнение политик
по награде теряло бы смысл.

Идентификаторы действий стабильны (см. ``bandit.envs.base``): руки
``0 .. n_arms-1``, мета-действия ``10000 + вид``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import ActionSpace, MetaAction, StepEnvironment, meta_action_kind


class LayeredPyramidBandit(StepEnvironment):
    """Пирамида с данными разной ценности, эпохами и амплификаторами."""

    #: Рецепты технологий: ``tech_costs[i]`` — стоимость перехода epoch i → i+1.
    DEFAULT_TECH_COSTS: List[Dict[int, int]] = [
        {0: 8, 1: 4},           # epoch 0 → 1: 8 basic + 4 advanced
        {0: 10, 1: 6, 2: 3},    # epoch 1 → 2: 10 basic + 6 advanced + 3 rare
    ]
    #: Тиры стартовых рук (длина = ``n_arms_init``).
    DEFAULT_INITIAL_TIERS: Sequence[int] = (0, 0, 0, 1, 1)
    #: Тиры рук, разблокируемых на каждом advancement.
    DEFAULT_UNLOCK_TIERS: List[Sequence[int]] = [
        (1, 2, 2),   # epoch 0 → 1: +1 advanced, +2 rare
        (2, 3, 3),   # epoch 1 → 2: +1 rare, +2 legendary
    ]
    DEFAULT_AMPLIFIER_BASE_COST: Dict[int, int] = {0: 5, 1: 3}

    META: Tuple[MetaAction, ...] = (
        MetaAction.INFO_TO_MULTIPLIER,
        MetaAction.BUY_ARM,
        MetaAction.CRAFT_GATE,
        MetaAction.CRAFT_AMPLIFIER,
    )

    def __init__(
        self,
        n_arms_init: int = 5,
        n_levels: int = 3,
        info_per_multiplier: int = 5,
        info_per_pull: int = 1,
        useful_info_prob: float = 1.0,
        junk_feeds_multiplier: float = 1.0,
        mult_per_arm_base: int = 3,
        arm_cost_increase: int = 1,
        mult_bonus: float = 0.5,
        arm_bonus: float = 0.5,
        arm_bonus_increase: float = 0.1,
        max_arms: int = 50,
        initial_tiers: Optional[Sequence[int]] = None,
        unlock_tiers: Optional[List[Sequence[int]]] = None,
        tech_costs: Optional[List[Dict[int, int]]] = None,
        tech_multiplier_bonus: float = 1.0,
        tech_arm_mean_boost: float = 0.15,
        amplifier_base_cost: Optional[Dict[int, int]] = None,
        amplifier_mult_bonus: float = 0.3,
        amplifier_cost_growth: float = 1.5,
        arm_mean_sampler: Optional[Callable] = None,
        recombination_sigma: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        # Отдельный поток для розыгрыша полезности единиц: шум не должен
        # сдвигать последовательность наград, иначе перегрузка меняла бы очки
        # даже у политики `none` (нарушение «reward не зависит от H6»).
        self._noise_rng = np.random.default_rng(
            None if seed is None else int(seed) + 1
        )
        self.arm_mean_sampler = arm_mean_sampler or (lambda rng: rng.uniform(0, 1))
        self.n_arms_init = int(n_arms_init)
        self.n_levels = int(n_levels)
        self.info_per_multiplier = int(info_per_multiplier)
        self.info_per_pull = int(info_per_pull)
        self.useful_info_prob = float(useful_info_prob)
        self.junk_feeds_multiplier = float(junk_feeds_multiplier)
        if self.info_per_pull < 1:
            raise ValueError("info_per_pull должно быть не меньше 1")
        if not 0.0 <= self.useful_info_prob <= 1.0:
            raise ValueError("useful_info_prob должен лежать в [0, 1]")
        if self.junk_feeds_multiplier < 0.0:
            raise ValueError("junk_feeds_multiplier не может быть отрицательным")
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
            (initial_tiers or self.DEFAULT_INITIAL_TIERS)[: self.n_arms_init]
        )
        self.unlock_tiers = list(unlock_tiers or self.DEFAULT_UNLOCK_TIERS)
        # Ключи рецептов приводим к int: в манифесте ``<name>.json`` словари
        # ``{0: 8}`` сериализуются как ``{"0": 8}``, и после перезапуска из
        # JSON сравнение с ``data[tier]`` молча не находило бы совпадений —
        # ворота и амплификаторы переставали собираться.
        self.tech_costs = [
            {int(tier): int(amount) for tier, amount in dict(recipe).items()}
            for recipe in (tech_costs or self.DEFAULT_TECH_COSTS)
        ]
        self.amplifier_base_cost = {
            int(tier): int(amount)
            for tier, amount in dict(
                amplifier_base_cost or self.DEFAULT_AMPLIFIER_BASE_COST
            ).items()
        }
        self.reset()

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Сгенерировать стартовые руки и обнулить экономику.

        Порядок обращений к RNG сохранён таким же, как в версии 0.1: это
        позволяет golden-тестам зафиксировать численные значения.
        """
        means_list, tiers_list = [], []
        for i in range(self.n_arms_init):
            base_mean = float(self.arm_mean_sampler(self.rng))
            tier = int(self.initial_tiers[i]) if i < len(self.initial_tiers) else 0
            mean = float(np.clip(base_mean + self.tech_arm_mean_boost * tier, 0.0, 1.0))
            means_list.append(mean)
            tiers_list.append(tier)
        self.means = np.array(means_list)
        self.arm_tiers = np.array(tiers_list, dtype=int)
        self.n_arms = int(self.n_arms_init)

        self.info_units = 0
        self.data: Dict[int, int] = {}
        self.n_multipliers = 0
        self.arms_bought = 0
        self.total_arm_bonus = 0.0
        self.points = 0.0
        self.t = 0

        self.epoch = 0
        self.techs_crafted = 0
        self.amplifiers_crafted = 0
        self.tech_multiplier_total = 0.0
        self.amplifier_multiplier_total = 0.0

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(self.n_arms, self.max_arms, self.META)

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
        """Рецепт перехода в следующую эпоху (``None`` — финальная эпоха)."""
        if self.epoch >= self.n_levels - 1:
            return None
        idx = min(self.epoch, len(self.tech_costs) - 1)
        return dict(self.tech_costs[idx])

    @property
    def current_amplifier_cost(self) -> Dict[int, int]:
        factor = self.amplifier_cost_growth ** self.amplifiers_crafted
        return {t: max(1, int(np.ceil(q * factor))) for t, q in self.amplifier_base_cost.items()}

    @property
    def can_convert_info(self) -> bool:
        return self.info_units >= self.info_per_multiplier

    @property
    def can_buy_arm(self) -> bool:
        return self.n_multipliers >= self.current_arm_cost and self.n_arms < self.max_arms

    @property
    def can_craft_gate(self) -> bool:
        cost = self.current_tech_cost
        if cost is None:
            return False
        return all(self.data.get(t, 0) >= q for t, q in cost.items())

    @property
    def can_craft_amplifier(self) -> bool:
        return all(self.data.get(t, 0) >= q for t, q in self.current_amplifier_cost.items())

    # ------------------------------------------------------------------
    # Действия
    # ------------------------------------------------------------------
    def _apply_action(self, action: int) -> Tuple[float, float]:
        kind = meta_action_kind(action)
        if kind is None:
            return self._pull_arm(action)
        if kind is MetaAction.INFO_TO_MULTIPLIER:
            return self._convert_info()
        if kind is MetaAction.BUY_ARM:
            return self._buy_arm()
        if kind is MetaAction.CRAFT_GATE:
            return self._craft_gate()
        if kind is MetaAction.CRAFT_AMPLIFIER:
            return self._craft_amplifier()
        raise ValueError(f"LayeredPyramidBandit не поддерживает {kind.name}")  # pragma: no cover

    def _pull_arm(self, action: int) -> Tuple[float, float]:
        """Вытянуть руку: базовый 0/1 сигнал + информационный поток шага.

        Поток информации (H6) отделён от награды и очков: перегрузка меняет
        только учёт данных. Каждое вытягивание возвращает ``info_per_pull``
        единиц; каждая единица с вероятностью ``useful_info_prob`` полезна —
        даёт данные тира и топливо для конвертации. Остальное — «мусор»: он
        попадает в ``info_gained`` (знаменатель доли сигнала), а в
        ``info_units`` — только с весом ``junk_feeds_multiplier`` (произведение
        округляется до целого, ``info_units`` остаётся счётчиком).

        При значениях по умолчанию (1 единица, ``useful_info_prob = 1``) поток
        совпадает со старым поведением: ровно одна полезная единица и ни одного
        дополнительного обращения к RNG (golden-тесты остаются в силе).
        """
        tier = int(self.arm_tiers[action])
        base = float(self.rng.random() < self.means[action])
        # Базовый (0/1) сигнал — для обучения агента; множитель — только в счёт.
        gained = base * self.current_multiplier
        self.points += gained

        units = self.info_per_pull
        if self.useful_info_prob >= 1.0:
            useful = units
        elif self.useful_info_prob <= 0.0:
            useful = 0
        else:
            useful = int(
                sum(
                    float(self._noise_rng.random()) < self.useful_info_prob
                    for _ in range(units)
                )
            )
        junk = units - useful
        self.info_units += useful + int(round(junk * self.junk_feeds_multiplier))
        self.data[tier] = self.data.get(tier, 0) + useful
        self._last_useful_gained = float(useful)
        self._last_info_gained = float(units)
        return base, gained

    def _convert_info(self) -> Tuple[float, float]:
        if self.info_units >= self.info_per_multiplier:
            self.info_units -= self.info_per_multiplier
            self.n_multipliers += 1
        return 0.0, 0.0

    def _buy_arm(self) -> Tuple[float, float]:
        cost = self.current_arm_cost
        if self.n_multipliers >= cost and self.n_arms < self.max_arms:
            self.n_multipliers -= cost
            self.total_arm_bonus += self.arm_bonus + self.arm_bonus_increase * self.arms_bought
            self.arms_bought += 1
            # Синергия топ-2 + рекомбинация Вайцмана.
            top = np.argsort(self.means)[-2:]
            p1, p2 = float(self.means[top[0]]), float(self.means[top[1]])
            synergy = 1.0 - (1.0 - p1) * (1.0 - p2)
            new_mean = float(
                np.clip(synergy + self.rng.normal(0, self.recombination_sigma), 0.0, 1.0)
            )
            new_tier = int(max(self.arm_tiers[top[0]], self.arm_tiers[top[1]]))
            self.means = np.append(self.means, new_mean)
            self.arm_tiers = np.append(self.arm_tiers, new_tier)
            self.n_arms += 1
        return 0.0, 0.0

    def _craft_gate(self) -> Tuple[float, float]:
        """Скомбинировать данные разных тиров в технологию и продвинуть эпоху.

        Немедленной награды не даёт: только future multiplier и новые руки.
        """
        cost = self.current_tech_cost
        if cost is None:
            return 0.0, 0.0
        if not all(self.data.get(t, 0) >= q for t, q in cost.items()):
            return 0.0, 0.0
        for t, q in cost.items():
            self.data[t] -= q

        self.techs_crafted += 1
        self.epoch += 1
        self.tech_multiplier_total += self.tech_multiplier_bonus

        idx = self.epoch - 1
        if idx < len(self.unlock_tiers):
            for tier in self.unlock_tiers[idx]:
                if self.n_arms >= self.max_arms:
                    break
                base_mean = float(self.arm_mean_sampler(self.rng))
                new_mean = float(
                    np.clip(base_mean + self.tech_arm_mean_boost * int(tier), 0.0, 1.0)
                )
                self.means = np.append(self.means, new_mean)
                self.arm_tiers = np.append(self.arm_tiers, int(tier))
                self.n_arms += 1
        return 0.0, 0.0

    def _craft_amplifier(self) -> Tuple[float, float]:
        """Побочная ветка: стоимость растёт экспоненциально, эффект линейный."""
        cost = self.current_amplifier_cost
        if not all(self.data.get(t, 0) >= q for t, q in cost.items()):
            return 0.0, 0.0
        for t, q in cost.items():
            self.data[t] -= q
        self.amplifiers_crafted += 1
        self.amplifier_multiplier_total += self.amplifier_mult_bonus
        return 0.0, 0.0

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return (
            f"LayeredPyramidBandit(n_arms={self.n_arms}, epoch={self.epoch}, "
            f"t={self.t}, points={self.points:.1f})"
        )
