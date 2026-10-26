"""Обучаемая мета-политика (P4): детерминизм, окно, награда, отчётность."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bandit.configs import get_config
from bandit.envs import LayeredPyramidBandit, MetaAction, meta_action_id
from bandit.meta import (
    ABSTAIN,
    CONTEXT_LABELS,
    POLICY_CLASSES,
    LearnedMeta,
    make_policy,
)
from bandit.runner import run_config

INFO = meta_action_id(MetaAction.INFO_TO_MULTIPLIER)
BUY = meta_action_id(MetaAction.BUY_ARM)
GATE = meta_action_id(MetaAction.CRAFT_GATE)
AMP = meta_action_id(MetaAction.CRAFT_AMPLIFIER)

ALLOWED = {
    INFO: "can_convert_info",
    BUY: "can_buy_arm",
    GATE: "can_craft_gate",
    AMP: "can_craft_amplifier",
}


def _obs(env, t, points, **flags):
    """Наблюдение с подменёнными ``t``/``points``/флагами доступности."""
    fields = dict(
        t=int(t),
        points=float(points),
        can_convert_info=True,
        can_buy_arm=False,
        can_craft_gate=False,
        can_craft_amplifier=False,
    )
    fields.update(flags)
    return env.observation()._replace(**fields)


def test_registry_exposes_learned_policy():
    assert POLICY_CLASSES["learned"] is LearnedMeta
    policy = make_policy("learned", window=3, c=2.0)
    assert isinstance(policy, LearnedMeta)
    assert (policy.window, policy.c) == (3, 2.0)
    assert make_policy("learned").window == 5


@pytest.mark.parametrize(
    "params",
    [{"window": 0}, {"window": -3}, {"c": -1.0}, {"decay": 0.0}, {"decay": 1.5}],
)
def test_invalid_params_are_rejected(params):
    with pytest.raises(ValueError):
        LearnedMeta(**params)


def test_first_decision_is_exploration_and_window_blocks_the_rest():
    env = LayeredPyramidBandit(seed=3)
    policy = LearnedMeta(window=4)
    assert policy(_obs(env, 0, 0.0)) == INFO  # неисследованная опция
    for step in range(1, 4):
        assert policy(_obs(env, step, float(step))) is None  # окно открыто
    # Окно закрылось, награда 0, но бонус исследования > 0 → снова инвестиция.
    assert policy(_obs(env, 4, 4.0)) == INFO


def test_policy_returns_only_actions_the_env_allows():
    env = LayeredPyramidBandit(seed=11)
    policy = LearnedMeta(window=3, c=0.5)
    seen = set()
    for _ in range(300):
        obs = env.observation()
        action = policy(obs)
        if action is None:
            env.step(0)
            continue
        assert action in ALLOWED
        assert getattr(obs, ALLOWED[action]), obs
        seen.add(action)
        env.step(action)
    assert seen, "политика ни разу не инвестировала"


def test_reward_is_log_ratio_of_adjacent_window_yields():
    env = LayeredPyramidBandit(seed=3)
    policy = LearnedMeta(window=5, c=0.0, min_trials=2)
    # Окно [0, 5): 1 очко за шаг. Окно [5, 10): 2 очка за шаг.
    points = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0,
              5: 5.0, 6: 7.0, 7: 9.0, 8: 11.0, 9: 13.0, 10: 15.0}

    assert policy(_obs(env, 0, points[0])) == INFO
    for step in range(1, 5):
        assert policy(_obs(env, step, points[step])) is None
    # Первое окно калибровочное (нет базы) → награда 0, но счётчик растёт.
    assert policy(_obs(env, 5, points[5])) == INFO
    for step in range(6, 10):
        assert policy(_obs(env, step, points[step])) is None
    assert policy(_obs(env, 10, points[10])) == INFO

    table = policy.table()
    key = ("epoch 0", CONTEXT_LABELS[1])
    stats = table[key][f"meta {INFO}"]
    expected = math.log((2.0 + 1e-3) / (1.0 + 1e-3))
    assert stats[0] == 2.0  # два закрытых окна
    assert stats[1] == pytest.approx(0.5 * expected)

    summary = policy.summary()
    assert summary["contexts"] == 1.0
    assert summary["invested"] >= 2.0
    assert 0.0 < summary["invest_share"] <= 1.0


@pytest.mark.slow
def test_learned_is_deterministic_and_beats_no_meta():
    learned = run_config(
        get_config("pyramid-ucb1-learned"), n_runs=2, base_seed=7, n_steps=800, progress=False
    )
    again = run_config(
        get_config("pyramid-ucb1-learned"), n_runs=2, base_seed=7, n_steps=800, progress=False
    )
    none = run_config(
        get_config("pyramid-ucb1-none"), n_runs=2, base_seed=7, n_steps=800, progress=False
    )

    assert np.array_equal(learned.points, again.points)
    assert np.array_equal(learned.actions, again.actions)
    assert np.all(learned.points[:, -1] > none.points[:, -1])
    assert np.all(learned.epoch_after[:, -1] > none.epoch_after[:, -1])


@pytest.mark.slow
def test_learned_close_to_greedy_full_cycle_at_full_horizon():
    """Обучаемая политика догоняет ручной жадный метацикл (а не «проигрывает ему»)."""
    learned = run_config(
        get_config("pyramid-ucb1-learned"), n_runs=3, base_seed=7, progress=False
    )
    greedy = run_config(
        get_config("pyramid-ucb1-full_cycle"), n_runs=3, base_seed=7, progress=False
    )
    ratio = learned.points[:, -1].mean() / greedy.points[:, -1].mean()
    assert ratio > 0.9


def test_constants_are_exported():
    assert ABSTAIN == -1
    assert set(CONTEXT_LABELS) == {1, 2}
