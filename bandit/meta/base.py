"""Слой мета-политик: *когда* инвестировать вместо *что* тянуть.

Политика получает ``Observation`` (без истинных средних рук) и возвращает
либо стабильный id действия, либо ``None`` — «делегирую выбор агенту».
Такое разделение — ключевое для проверяемости гипотезы: вклад метацикла
измеряется при фиксированном агенте, а не смешивается с ним.

Здесь же живёт ``BasePolicy`` — общий минимум для всех реализаций
(``meta/policies.py`` и ``meta/learned.py``). Вынесен в базовый модуль,
чтобы обучаемая политика не импортировала ``policies`` (цикл импортов).
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..envs.base import MetaAction, Observation, meta_action_id


@runtime_checkable
class Policy(Protocol):
    """Протокол мета-политики."""

    name: str
    uses_oracle_info: bool

    def __call__(self, obs: Observation) -> Optional[int]:
        """Вернуть id действия или ``None`` (выбор за агентом)."""


class BasePolicy:
    """Общая логика мета-политик: имя, флаг оракула, id мета-действий."""

    name: str = "base"
    uses_oracle_info: bool = False

    def __call__(self, obs: Observation) -> Optional[int]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    @staticmethod
    def _meta(kind: MetaAction) -> int:
        return meta_action_id(kind)


__all__ = ["BasePolicy", "Policy"]
