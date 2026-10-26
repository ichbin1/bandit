"""Метацикл обучения на многоруком бандите.

Структура пакета
----------------
* ``bandit.envs`` — среды: стационарные бандиты, ``PyramidBandit`` (legacy)
  и ``LayeredPyramidBandit`` с экономикой данных и метациклом;
* ``bandit.agents`` — алгоритмы выбора действий;
* ``bandit.meta`` — мета-политики (метацикл как предписание, включая оракул);
* ``bandit.runner`` — воспроизводимый прогон, ``RunConfig`` / ``RunRecords``;
* ``bandit.metrics`` — метрики гипотезы (сожаление, эффективность, адаптация);
* ``bandit.stats`` — парная статистика (bootstrap, Wilcoxon, знаки, Холм);
* ``bandit.report`` — отчёт по прогону: таблицы, сравнения, вердикты H1–H6;
* ``bandit.plots`` — графики;
* ``bandit.configs`` — зарегистрированные конфиги экспериментов;
* ``bandit.cli`` — командная строка (``python -m bandit.cli``).

Ключевое отличие от версии 0.1
------------------------------
Идентификаторы действий стабильны на протяжении эпизода: руки занимают
``0 .. n_arms-1``, мета-действия — ``META_ACTION_BASE + вид``. Раньше
мета-действие кодировалось как ``n_arms + k``, поэтому при добавлении руки
один и тот же ``int`` начинал означать другое действие: сохранённые
траектории были невоспроизводимы, а обучение мета-политики невозможно.
"""
from __future__ import annotations

__version__ = "0.4.0"

from .envs.base import (  # noqa: F401
    META_ACTION_BASE,
    ActionSpace,
    MetaAction,
    Observation,
    State,
    StepEnvironment,
    StepInfo,
    is_meta_action,
    meta_action_id,
)

__all__ = [
    "__version__",
    "META_ACTION_BASE",
    "ActionSpace",
    "MetaAction",
    "Observation",
    "State",
    "StepEnvironment",
    "StepInfo",
    "is_meta_action",
    "meta_action_id",
]
