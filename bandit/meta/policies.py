"""Мета-политики (предписания метацикла).

Все основные политики используют только публичное наблюдение
(``Observation``): свои данные, свои множители, рецепты и стоимости. Истинные
средние рук им недоступны — иначе сравнение с агентом теряет смысл.

Оракульные политики (``oracle_*``) помечены ``uses_oracle_info = True`` и
служат только верхними границами; в отчётах они выносятся отдельно.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from ..envs.base import MetaAction, Observation, meta_action_id
from .base import BasePolicy
from .learned import LearnedMeta

__all__ = [
    "AmpMeta",
    "AmpOnly",
    "BasePolicy",
    "FullCycle",
    "GateOnly",
    "GreedyMeta",
    "LearnedMeta",
    "MultOnly",
    "NoMeta",
    "OracleFullCycle",
    "OracleNoMeta",
    "POLICY_CLASSES",
    "make_policy",
]


class NoMeta(BasePolicy):
    """Метацикл выключен: агент всегда тянет руку (контроль)."""

    name = "none"

    def __call__(self, obs: Observation) -> Optional[int]:
        return None


class MultOnly(BasePolicy):
    """Только конвертация данных в множитель (минимум метацикла)."""

    name = "mult_only"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        return None


class GreedyMeta(BasePolicy):
    """Сначала множители, затем покупка рук; технологии не трогает."""

    name = "greedy"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        if obs.can_buy_arm:
            return self._meta(MetaAction.BUY_ARM)
        return None


class AmpMeta(BasePolicy):
    """GreedyMeta + амплификаторы (побочная ветка экономики)."""

    name = "amp"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_craft_amplifier:
            return self._meta(MetaAction.CRAFT_AMPLIFIER)
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        if obs.can_buy_arm:
            return self._meta(MetaAction.BUY_ARM)
        return None


class FullCycle(BasePolicy):
    """Полный метацикл: технологии → амплификаторы → множители → руки.

    Порядок приоритетов важен: «ворота» открывают новые тиры и эпохи, без них
    рецепты следующих уровней недостижимы. Амплификаторы дешевле на ранних
    эпохах, поэтому идут раньше конвертации.
    """

    name = "full_cycle"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_craft_gate:
            return self._meta(MetaAction.CRAFT_GATE)
        if obs.can_craft_amplifier:
            return self._meta(MetaAction.CRAFT_AMPLIFIER)
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        if obs.can_buy_arm:
            return self._meta(MetaAction.BUY_ARM)
        return None


class GateOnly(BasePolicy):
    """Абляция: только технологии (плюс множители и покупка рук)."""

    name = "gate_only"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_craft_gate:
            return self._meta(MetaAction.CRAFT_GATE)
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        if obs.can_buy_arm:
            return self._meta(MetaAction.BUY_ARM)
        return None


class AmpOnly(BasePolicy):
    """Абляция: только амплификаторы (плюс множители и покупка рук)."""

    name = "amp_only"

    def __call__(self, obs: Observation) -> Optional[int]:
        if obs.can_craft_amplifier:
            return self._meta(MetaAction.CRAFT_AMPLIFIER)
        if obs.can_convert_info:
            return self._meta(MetaAction.INFO_TO_MULTIPLIER)
        if obs.can_buy_arm:
            return self._meta(MetaAction.BUY_ARM)
        return None


class OracleNoMeta(BasePolicy):
    """Верхняя граница выбора рук при выключенном метацикле."""

    name = "oracle_none"
    uses_oracle_info = True

    def __call__(self, obs: Observation) -> int:
        if obs.best_arm is None:
            raise RuntimeError(
                "OracleNoMeta требует env.expose_oracle=True "
                "(иначе политика не видит истинные средние)."
            )
        return int(obs.best_arm)


class OracleFullCycle(FullCycle):
    """Оракул: идеальный выбор руки + полный метацикл (верхняя граница).

    «Идеальный» здесь означает не только максимальное среднее: «воротам» нужны
    данные *всех* тиров рецепта. Жадный оракул (всегда лучшая рука) собирает
    однородные данные и не открывает эпохи — поэтому он проигрывает обычному
    UCB1 по очкам. Этот оракул сначала добирает недостающие тиры, а внутри
    нужного тира берёт руку с максимальным истинным средним.
    """

    name = "oracle_full_cycle"
    uses_oracle_info = True

    def __call__(self, obs: Observation) -> int:
        action = super().__call__(obs)
        if action is not None:
            return action
        if obs.best_arm is None:
            raise RuntimeError(
                "OracleFullCycle требует env.expose_oracle=True "
                "(иначе политика не видит истинные средние)."
            )
        needed = set(obs.missing_tech_tiers())
        if needed and obs.arm_means and len(obs.arm_tiers) == len(obs.arm_means):
            candidates = [
                index for index, tier in enumerate(obs.arm_tiers) if tier in needed
            ]
            if candidates:
                return max(candidates, key=lambda index: obs.arm_means[index])
        return int(obs.best_arm)


POLICY_CLASSES: Dict[str, Type[BasePolicy]] = {
    "none": NoMeta,
    "mult_only": MultOnly,
    "greedy": GreedyMeta,
    "amp": AmpMeta,
    "full_cycle": FullCycle,
    "gate_only": GateOnly,
    "amp_only": AmpOnly,
    "learned": LearnedMeta,
    "oracle_none": OracleNoMeta,
    "oracle_full_cycle": OracleFullCycle,
}


def make_policy(kind: Optional[str], **params: Any) -> BasePolicy:
    """Создать политику по имени (``None`` → ``NoMeta``)."""
    if kind is None:
        return NoMeta()
    key = str(kind).strip().lower()
    if key not in POLICY_CLASSES:
        raise KeyError(
            f"Неизвестная мета-политика {kind!r}. Доступны: {', '.join(sorted(POLICY_CLASSES))}"
        )
    return POLICY_CLASSES[key](**params)
