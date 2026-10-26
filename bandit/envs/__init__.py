"""Среды (многорукие бандиты).

* ``stationary`` — классические стационарные бандиты (Bernoulli / Gaussian);
* ``pyramid`` — ``PyramidBandit``: простая экономика множителей (legacy);
* ``layered`` — ``LayeredPyramidBandit``: данные разной ценности, технологии
  (эпохи), амплификаторы, отложенная выгода — основная среда гипотезы.
"""
from __future__ import annotations

from .base import (  # noqa: F401
    META_ACTION_BASE,
    ActionSpace,
    BanditEnvironment,
    MetaAction,
    Observation,
    State,
    StepEnvironment,
    StepInfo,
    action_kind,
    action_kind_label,
    is_meta_action,
    legacy_action_to_stable,
    meta_action_id,
    meta_action_kind,
    stable_action_to_legacy,
)
from .layered import LayeredPyramidBandit  # noqa: F401
from .pyramid import PyramidBandit  # noqa: F401
from .stationary import BernoulliBandit, GaussianBandit  # noqa: F401

ENVIRONMENTS = {
    "bernoulli": BernoulliBandit,
    "gaussian": GaussianBandit,
    "pyramid": PyramidBandit,
    "layered": LayeredPyramidBandit,
}

__all__ = [
    "META_ACTION_BASE",
    "ActionSpace",
    "BanditEnvironment",
    "BernoulliBandit",
    "GaussianBandit",
    "LayeredPyramidBandit",
    "MetaAction",
    "Observation",
    "PyramidBandit",
    "State",
    "StepEnvironment",
    "StepInfo",
    "ENVIRONMENTS",
    "action_kind",
    "action_kind_label",
    "is_meta_action",
    "legacy_action_to_stable",
    "meta_action_id",
    "meta_action_kind",
    "stable_action_to_legacy",
]
