"""Реестр агентов: ``make_agent(kind, n_actions, seed, **params)``.

Реестр нужен, чтобы конфиги экспериментов были сериализуемыми словарями
(``{"kind": "ucb1", "params": {"c": 2.0}}``) и работали в
``ProcessPoolExecutor`` на Windows (spawn) без проблем с pickle лямбд.
"""
from __future__ import annotations

from typing import Any, Dict, Type

from .base import Agent
from .baselines import EpsilonGreedy, GreedyAgent, RandomAgent, RescorlaWagnerAgent
from .thompson import ThompsonSamplingBernoulli, ThompsonSamplingGaussian
from .ucb import UCB1

AGENT_CLASSES: Dict[str, Type[Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "epsilon_greedy": EpsilonGreedy,
    "ucb1": UCB1,
    "thompson_bernoulli": ThompsonSamplingBernoulli,
    "thompson_gaussian": ThompsonSamplingGaussian,
    "rescorla_wagner": RescorlaWagnerAgent,
}

AGENT_ALIASES: Dict[str, str] = {
    "eps": "epsilon_greedy",
    "epsilon": "epsilon_greedy",
    "ts": "thompson_bernoulli",
    "thompson": "thompson_bernoulli",
    "ts_gaussian": "thompson_gaussian",
    "rw": "rescorla_wagner",
    "rescorla-wagner": "rescorla_wagner",
}


def resolve_agent_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    key = AGENT_ALIASES.get(key, key)
    if key not in AGENT_CLASSES:
        raise KeyError(
            f"Неизвестный агент {kind!r}. Доступны: {', '.join(sorted(AGENT_CLASSES))}"
        )
    return key


def make_agent(kind: str, n_actions: int, seed: int | None = None, **params: Any) -> Agent:
    """Создать агента по имени с параметрами."""
    cls = AGENT_CLASSES[resolve_agent_kind(kind)]
    return cls(n_actions=n_actions, seed=seed, **params)


__all__ = [
    "AGENT_CLASSES",
    "AGENT_ALIASES",
    "Agent",
    "EpsilonGreedy",
    "GreedyAgent",
    "RandomAgent",
    "RescorlaWagnerAgent",
    "ThompsonSamplingBernoulli",
    "ThompsonSamplingGaussian",
    "UCB1",
    "make_agent",
    "resolve_agent_kind",
]
