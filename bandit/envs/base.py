"""Базовые контракты сред: стабильные id действий, наблюдение, шаг.

Зачем нужны стабильные id
-------------------------
В версии 0.1 мета-действия кодировались как ``n_arms + k``. Поскольку
``n_arms`` меняется по ходу эпизода (покупка рук, advancement), один и тот же
``int`` на разных шагах означал разные действия. Следствия: траектории нельзя
анализировать постфактум, а обучаемая мета-политика над объединённым
пространством действий невозможна.

Теперь id стабильны на весь эпизод:

* руки — ``0 .. n_arms-1`` (новые руки только дописываются в конец);
* мета-действия — ``META_ACTION_BASE + kind`` (``10000 .. 10003``).

Старый способ по-прежнему доступен явно: ``env.pull_legacy(action)`` или
``legacy_action_to_stable(action, n_arms)``. Молчаливой интерпретации нет:
``env.step(n_arms)`` поднимает ``ValueError`` с подсказкой.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np

META_ACTION_BASE = 10_000
"""Первый id мета-действия; руки занимают ``0 .. META_ACTION_BASE - 1``."""


class MetaAction(IntEnum):
    """Виды мета-действий метацикла."""

    INFO_TO_MULTIPLIER = 0
    BUY_ARM = 1
    CRAFT_GATE = 2
    CRAFT_AMPLIFIER = 3


ACTION_KIND_ARM = 0
"""Код «действие — вытягивание руки» в массиве ``RunRecords.kind``."""

_KIND_LABELS: Dict[int, str] = {
    ACTION_KIND_ARM: "arm",
    int(MetaAction.INFO_TO_MULTIPLIER) + 1: "info_to_multiplier",
    int(MetaAction.BUY_ARM) + 1: "buy_arm",
    int(MetaAction.CRAFT_GATE) + 1: "craft_gate",
    int(MetaAction.CRAFT_AMPLIFIER) + 1: "craft_amplifier",
}

ACTION_KIND_LABELS: Tuple[str, ...] = tuple(_KIND_LABELS[k] for k in sorted(_KIND_LABELS))

TierCounts = Tuple[Tuple[int, int], ...]
"""Неизменяемое представление ``Dict[int, int]`` (тир → количество)."""


def meta_action_id(kind: MetaAction | int) -> int:
    """Стабильный id мета-действия данного вида."""
    return META_ACTION_BASE + int(kind)


def is_meta_action(action: int) -> bool:
    """``True``, если id относится к диапазону мета-действий."""
    return int(action) >= META_ACTION_BASE


def meta_action_kind(action: int) -> Optional[MetaAction]:
    """Вид мета-действия по id или ``None`` (в т.ч. для неизвестных id)."""
    if not is_meta_action(action):
        return None
    offset = int(action) - META_ACTION_BASE
    if offset not in MetaAction._value2member_map_:
        return None
    return MetaAction(offset)


def action_kind(action: int) -> int:
    """Код вида действия: 0 — рука, 1..4 — мета-действия."""
    kind = meta_action_kind(action)
    return ACTION_KIND_ARM if kind is None else int(kind) + 1


def action_kind_label(action: int) -> str:
    """Человекочитаемый вид действия (для CSV/графиков)."""
    return _KIND_LABELS[action_kind(action)]


def legacy_action_to_stable(action: int, n_arms: int) -> int:
    """Перевод legacy-индекса ``n_arms + k`` в стабильный id."""
    action = int(action)
    if action < int(n_arms):
        if action < 0:
            raise ValueError(f"Недопустимый legacy-индекс {action}")
        return action
    offset = action - int(n_arms)
    if offset not in MetaAction._value2member_map_:
        raise ValueError(
            f"Недопустимый legacy-индекс {action} при n_arms={n_arms}: "
            f"мета-действий всего {len(MetaAction)}"
        )
    return meta_action_id(offset)


def stable_action_to_legacy(action: int, n_arms: int) -> int:
    """Обратный перевод: стабильный id → legacy-индекс ``n_arms + k``."""
    action = int(action)
    kind = meta_action_kind(action)
    if kind is None:
        return action
    return int(n_arms) + int(kind)


@dataclass(frozen=True)
class ActionSpace:
    """Текущее пространство действий среды (только чтение).

    ``n_actions`` — мощность пространства; ``all_ids()`` — сами id в
    каноническом порядке (сначала руки, затем мета-действия).
    """

    n_arms: int
    max_arms: int
    meta: Tuple[MetaAction, ...] = ()

    def __post_init__(self) -> None:
        if self.n_arms < 0:
            raise ValueError("n_arms не может быть отрицательным")
        if self.max_arms < self.n_arms:
            raise ValueError("max_arms не может быть меньше n_arms")

    @property
    def n_actions(self) -> int:
        return int(self.n_arms) + len(self.meta)

    @property
    def supports_meta(self) -> bool:
        return bool(self.meta)

    @property
    def n_meta(self) -> int:
        return len(self.meta)

    def arm_ids(self) -> range:
        return range(int(self.n_arms))

    def meta_ids(self) -> Tuple[int, ...]:
        return tuple(meta_action_id(kind) for kind in self.meta)

    def all_ids(self) -> Tuple[int, ...]:
        return tuple(range(int(self.n_arms))) + self.meta_ids()

    def contains(self, action: int) -> bool:
        try:
            action = int(action)
        except (TypeError, ValueError):
            return False
        kind = meta_action_kind(action)
        if kind is not None:
            return kind in self.meta
        return 0 <= action < int(self.n_arms)

    def label(self, action: int) -> str:
        action = int(action)
        if is_meta_action(action):
            kind = meta_action_kind(action)
            return f"meta:{kind.name.lower()}" if kind is not None else f"meta:{action}"
        return f"arm:{action}"

    def __str__(self) -> str:  # pragma: no cover - только для отладки
        meta = ", ".join(kind.name.lower() for kind in self.meta) or "нет"
        return f"ActionSpace(arms=0..{self.n_arms - 1}, meta=[{meta}])"


class State(NamedTuple):
    """Снимок состояния среды (неизменяемый)."""

    t: int
    n_arms: int
    epoch: int
    n_levels: int
    info_units: int
    n_multipliers: int
    points: float
    multiplier: float
    techs_crafted: int
    amplifiers_crafted: int
    data: TierCounts


class Observation(NamedTuple):
    """То, что видит политика/агент.

    Истинные средние рук (``means``) сюда не входят: агент знает только то,
    что «добыл» (``data``), и правила экономики (рецепты, стоимости).
    ``best_arm`` / ``best_mean`` заполняются только при
    ``env.expose_oracle = True`` — это нужно исключительно для оракульных
    верхних границ, а не для «полу-оракульных» основных политик.
    """

    t: int
    n_arms: int
    max_arms: int
    n_actions: int
    epoch: int
    n_levels: int
    info_units: int
    info_per_multiplier: int
    n_multipliers: int
    current_arm_cost: int
    current_multiplier: float
    points: float
    techs_crafted: int
    amplifiers_crafted: int
    data: TierCounts
    tech_cost: TierCounts
    amplifier_cost: TierCounts
    can_convert_info: bool
    can_buy_arm: bool
    can_craft_gate: bool
    can_craft_amplifier: bool
    arm_tiers: Tuple[int, ...] = ()
    """Тир каждой руки (публичная структура среды, не скрытая информация)."""
    arm_means: Tuple[float, ...] = ()
    """Истинные средние; заполняются только при ``expose_oracle=True``."""
    best_arm: Optional[int] = None
    best_mean: Optional[float] = None

    def tier_data(self, tier: int) -> int:
        for t, count in self.data:
            if t == tier:
                return count
        return 0

    def missing_tech_tiers(self) -> Tuple[int, ...]:
        """Тиры, по которым «воротам» ещё не хватает данных (порядок рецепта)."""
        return tuple(tier for tier, need in self.tech_cost if self.tier_data(tier) < need)


@dataclass(frozen=True)
class StepInfo:
    """Полная запись одного шага: что было, что сделали, что стало."""

    action: int
    kind_code: int
    reward: float
    points_gained: float
    before: State
    after: State
    best_arm: int
    best_mean: float
    chosen_mean: float
    arm_regret: float
    useful_gained: float = 0.0
    """Сколько *полезных* единиц информации принёс шаг (H6).

    Полезная единица — та, что дала данные тира и топливо для конвертации.
    У мета-действий и у сред без экономики информации всегда 0.
    """
    info_gained: float = 0.0
    """Сколько единиц информации принёс шаг всего (полезные + мусор).

    ``useful_gained / info_gained`` — доля сигнала в потоке; при перегрузке
    (``LayeredPyramidBandit``) она падает, хотя число шагов не меняется.
    """

    @property
    def is_arm_pull(self) -> bool:
        return self.kind_code == ACTION_KIND_ARM

    @property
    def kind_label(self) -> str:
        return _KIND_LABELS[self.kind_code]

    @property
    def was_effective(self) -> bool:
        """Изменила ли действие состояние экономики (для мета-действий)."""
        if self.is_arm_pull:
            return True
        before = (
            self.before.n_arms,
            self.before.epoch,
            self.before.info_units,
            self.before.n_multipliers,
            self.before.techs_crafted,
            self.before.amplifiers_crafted,
        )
        after = (
            self.after.n_arms,
            self.after.epoch,
            self.after.info_units,
            self.after.n_multipliers,
            self.after.techs_crafted,
            self.after.amplifiers_crafted,
        )
        return before != after


class StepEnvironment(ABC):
    """Среда с явным ``step`` и наблюдаемым состоянием.

    Подклассы реализуют ``reset``, ``_apply_action`` и ``action_space``.
    """

    means: np.ndarray

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.expose_oracle: bool = False
        self._init_common_state()

    # ------------------------------------------------------------------
    # Состояние
    # ------------------------------------------------------------------
    def _init_common_state(self) -> None:
        self.n_arms: int = 0
        self.max_arms: int = 0
        self.epoch: int = 0
        self.n_levels: int = 1
        self.info_units: int = 0
        self.info_per_multiplier: int = 1
        self.n_multipliers: int = 0
        self.points: float = 0.0
        self.techs_crafted: int = 0
        self.amplifiers_crafted: int = 0
        self.data: Dict[int, int] = {}
        self.t: int = 0
        # Информационный поток последнего шага (H6): заполняется средами,
        # которые различают полезные данные и мусор.
        self._last_useful_gained: float = 0.0
        self._last_info_gained: float = 0.0

    @property
    def total_points(self) -> float:
        """Совместимость с версией 0.1 (там поле называлось ``total_points``)."""
        return float(self.points)

    @abstractmethod
    def reset(self) -> None:
        """Вернуть среду в начальное состояние (RNG не пересоздаётся)."""

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """Текущее пространство действий."""

    @property
    def n_actions(self) -> int:
        return self.action_space.n_actions

    @property
    def optimal_mean(self) -> float:
        return float(np.max(self.means))

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.means))

    # ------------------------------------------------------------------
    # Экономика: значения по умолчанию (стационарные среды)
    # ------------------------------------------------------------------
    @property
    def current_multiplier(self) -> float:
        return 1.0

    @property
    def current_arm_cost(self) -> int:
        return 0

    @property
    def current_tech_cost(self) -> Optional[Dict[int, int]]:
        return None

    @property
    def current_amplifier_cost(self) -> Dict[int, int]:
        return {}

    @property
    def can_convert_info(self) -> bool:
        return False

    @property
    def can_buy_arm(self) -> bool:
        return False

    @property
    def can_craft_gate(self) -> bool:
        return False

    @property
    def can_craft_amplifier(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Наблюдение
    # ------------------------------------------------------------------
    def observation(self) -> Observation:
        space = self.action_space
        tech_cost = self.current_tech_cost or {}
        best_arm: Optional[int] = None
        best_mean: Optional[float] = None
        arm_means: Tuple[float, ...] = ()
        if self.expose_oracle:
            best_arm = self.optimal_arm
            best_mean = self.optimal_mean
            arm_means = tuple(float(mu) for mu in self.means)
        return Observation(
            t=self.t,
            n_arms=self.n_arms,
            max_arms=self.max_arms,
            n_actions=space.n_actions,
            epoch=self.epoch,
            n_levels=self.n_levels,
            info_units=self.info_units,
            info_per_multiplier=self.info_per_multiplier,
            n_multipliers=self.n_multipliers,
            current_arm_cost=self.current_arm_cost,
            current_multiplier=float(self.current_multiplier),
            points=float(self.points),
            techs_crafted=self.techs_crafted,
            amplifiers_crafted=self.amplifiers_crafted,
            data=tuple(sorted(self.data.items())),
            tech_cost=tuple(sorted(tech_cost.items())),
            amplifier_cost=tuple(sorted(self.current_amplifier_cost.items())),
            can_convert_info=bool(self.can_convert_info),
            can_buy_arm=bool(self.can_buy_arm),
            can_craft_gate=bool(self.can_craft_gate),
            can_craft_amplifier=bool(self.can_craft_amplifier),
            arm_tiers=tuple(int(tier) for tier in getattr(self, "arm_tiers", ())),
            arm_means=arm_means,
            best_arm=best_arm,
            best_mean=best_mean,
        )

    def _make_state(self) -> State:
        return State(
            t=self.t,
            n_arms=self.n_arms,
            epoch=self.epoch,
            n_levels=self.n_levels,
            info_units=self.info_units,
            n_multipliers=self.n_multipliers,
            points=float(self.points),
            multiplier=float(self.current_multiplier),
            techs_crafted=self.techs_crafted,
            amplifiers_crafted=self.amplifiers_crafted,
            data=tuple(sorted(self.data.items())),
        )

    # ------------------------------------------------------------------
    # Шаг
    # ------------------------------------------------------------------
    def step(self, action: int) -> StepInfo:
        """Выполнить действие и вернуть полную запись шага."""
        action = self._validate_action(action)
        best_arm = self.optimal_arm
        best_mean = self.optimal_mean
        before = self._make_state()
        is_arm = not is_meta_action(action)
        chosen_mean = float(self.means[action]) if is_arm else float("nan")
        self._last_useful_gained = 0.0
        self._last_info_gained = 0.0
        self.t += 1
        reward, gained = self._apply_action(action)
        after = self._make_state()
        arm_regret = (best_mean - chosen_mean) if is_arm else float("nan")
        return StepInfo(
            action=action,
            kind_code=action_kind(action),
            reward=float(reward),
            points_gained=float(gained),
            before=before,
            after=after,
            best_arm=best_arm,
            best_mean=best_mean,
            chosen_mean=chosen_mean,
            arm_regret=float(arm_regret),
            useful_gained=float(self._last_useful_gained),
            info_gained=float(self._last_info_gained),
        )

    def pull(self, action: int) -> float:
        """Совместимость с версией 0.1: вернуть только сигнал награды."""
        return self.step(action).reward

    def pull_legacy(self, action: int) -> float:
        """Действие в старом кодировании ``n_arms + k`` (явно, не молча)."""
        return self.pull(legacy_action_to_stable(action, self.n_arms))

    @abstractmethod
    def _apply_action(self, action: int) -> Tuple[float, float]:
        """Применить провалидированное действие: ``(reward, points_gained)``."""

    def _validate_action(self, action: int) -> int:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError(
                f"Действие должно быть целым числом, получено {type(action).__name__}: {action!r}"
            )
        action = int(action)
        kind = meta_action_kind(action)
        if kind is not None:
            if kind not in self.action_space.meta:
                raise ValueError(f"Среда не поддерживает мета-действие {kind.name}")
            return action
        if 0 <= action < self.n_arms:
            return action
        hint = ""
        if self.n_arms <= action < META_ACTION_BASE:
            hint = (
                " Похоже на legacy-индекс мета-действия: используйте "
                f"meta_action_id(...) или env.pull_legacy({action})."
            )
        raise ValueError(f"Недопустимое действие {action} (n_arms={self.n_arms}).{hint}")


class BanditEnvironment(StepEnvironment):
    """Стационарный бандит: ``n_arms`` фиксировано, мета-действий нет."""

    def __init__(self, n_arms: int, seed: Optional[int] = None) -> None:
        if int(n_arms) <= 0:
            raise ValueError("n_arms должно быть положительным")
        self._n_arms_fixed = int(n_arms)
        super().__init__(seed)

    def reset(self) -> None:
        self.n_arms = self._n_arms_fixed
        self.max_arms = self._n_arms_fixed
        self.epoch = 0
        self.n_levels = 1
        self.info_units = 0
        self.n_multipliers = 0
        self.techs_crafted = 0
        self.amplifiers_crafted = 0
        self.points = 0.0
        self.t = 0
        self.data = {}

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(self.n_arms, self.max_arms, ())

    def _apply_action(self, action: int) -> Tuple[float, float]:
        reward = float(self._draw(action))
        self.points += reward
        return reward, reward

    @abstractmethod
    def _draw(self, action: int) -> float:
        """Реализовать розыгрыш награды руки."""

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"{type(self).__name__}(n_arms={self._n_arms_fixed})"
