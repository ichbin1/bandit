"""Агенты: корректность выбора, рост числа рук, RNG-паритет с v0.1."""
from __future__ import annotations

import numpy as np
import pytest

from bandit.agents import (
    AGENT_CLASSES,
    EpsilonGreedy,
    GreedyAgent,
    RandomAgent,
    RescorlaWagnerAgent,
    ThompsonSamplingBernoulli,
    ThompsonSamplingGaussian,
    UCB1,
    make_agent,
    resolve_agent_kind,
)


def _ucb1_sequence(seed: int, n_actions: int = 5) -> list:
    agent = UCB1(n_actions, c=2.0, seed=seed)
    actions = []
    for _ in range(50):
        action = agent.select_action()
        actions.append(action)
        agent.update(action, float(action % 2))
    return actions


def test_ucb1_tries_every_arm_before_exploiting():
    agent = UCB1(5, c=2.0, seed=1)
    first = []
    for _ in range(5):
        action = agent.select_action()
        first.append(action)
        agent.update(action, 0.0)
    assert sorted(first) == [0, 1, 2, 3, 4]


def test_ucb1_is_deterministic_for_same_seed():
    assert _ucb1_sequence(11) == _ucb1_sequence(11)


def test_grow_preserves_statistics_and_shrink_is_rejected():
    agent = UCB1(3, seed=0)
    agent.update(0, 1.0)
    agent.update(1, 0.5)
    agent.grow_to(5)
    assert agent.n_actions == 5
    assert agent.counts.tolist() == [1, 1, 0, 0, 0]
    assert agent.values[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        agent.set_n_actions(2)


class _SpyRng:
    """Обёртка над ``Generator``: считает вызовы ``choice`` и делегирует всё."""

    def __init__(self, rng):
        self._rng = rng
        self.choice_calls = 0

    def choice(self, *args, **kwargs):
        self.choice_calls += 1
        return self._rng.choice(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_argmax_random_always_calls_choice():
    """Паритет с v0.1: rng.choice вызывается даже при единственном максимуме."""
    agent = RandomAgent(3, seed=5)
    spy = _SpyRng(agent.rng)
    agent.rng = spy
    assert agent._argmax_random(np.array([0.0, 1.0, 0.0])) == 1
    assert spy.choice_calls == 1
    assert agent._argmax_random(np.array([2.0, 2.0, 0.0])) in (0, 1)
    assert spy.choice_calls == 2


def test_random_agent_is_roughly_uniform():
    agent = RandomAgent(4, seed=3)
    counts = np.bincount([agent.select_action() for _ in range(8000)], minlength=4)
    assert counts.min() > 1700
    assert counts.max() < 2300


def test_epsilon_greedy_zero_epsilon_is_greedy():
    agent = EpsilonGreedy(3, epsilon=0.0, initial_value=0.5, seed=2)
    agent.update(2, 1.0)
    assert {agent.select_action() for _ in range(20)} == {2}


def test_epsilon_decay_reaches_floor():
    agent = EpsilonGreedy(2, epsilon=0.5, epsilon_decay=0.5, epsilon_min=0.05, seed=0)
    for _ in range(20):
        action = agent.select_action()
        agent.update(action, 0.0)
    assert agent.epsilon == pytest.approx(0.05)


def test_greedy_agent_sticks_to_first_arm():
    agent = GreedyAgent(4, initial_value=0.0, seed=4)
    first = agent.select_action()
    for _ in range(10):
        agent.update(first, 1.0)
        assert agent.select_action() == first


def test_rescorla_wagner_moves_value_toward_reward():
    agent = RescorlaWagnerAgent(2, alpha=0.5, epsilon=0.0, initial_value=0.0, seed=0)
    agent.update(1, 1.0)
    assert agent.values[1] == pytest.approx(0.5)
    agent.update(1, 1.0)
    assert agent.values[1] == pytest.approx(0.75)


def test_thompson_updates_posterior_and_grows():
    agent = ThompsonSamplingBernoulli(2, seed=0)
    agent.update(0, 1.0)
    agent.update(1, 0.0)
    assert agent.alpha.tolist() == [2.0, 1.0]
    assert agent.beta.tolist() == [1.0, 2.0]
    agent.grow_to(4)
    assert agent.alpha.tolist() == [2.0, 1.0, 1.0, 1.0]


def test_thompson_gaussian_runs_and_prefers_promising_arm():
    agent = ThompsonSamplingGaussian(3, sigma=0.5, prior_std=0.1, seed=1)
    for _ in range(200):
        agent.update(0, 1.0)
    picks = [agent.select_action() for _ in range(100)]
    assert picks.count(0) > 90


def test_make_agent_aliases_and_errors():
    agent = make_agent("ts", n_actions=3, seed=0)
    assert isinstance(agent, ThompsonSamplingBernoulli)
    assert resolve_agent_kind("ucb1") == "ucb1"
    with pytest.raises(KeyError):
        make_agent("nope", n_actions=3, seed=0)
    assert "epsilon_greedy" in AGENT_CLASSES
