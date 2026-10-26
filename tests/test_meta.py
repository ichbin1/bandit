"""Мета-политики: приоритеты, абляции, оракулы."""
from __future__ import annotations

import pytest

from bandit.envs import LayeredPyramidBandit, MetaAction, meta_action_id
from bandit.meta import (
    POLICY_CLASSES,
    FullCycle,
    GateOnly,
    GreedyMeta,
    MultOnly,
    NoMeta,
    OracleFullCycle,
    OracleNoMeta,
    make_policy,
)

INFO = meta_action_id(MetaAction.INFO_TO_MULTIPLIER)
BUY = meta_action_id(MetaAction.BUY_ARM)
GATE = meta_action_id(MetaAction.CRAFT_GATE)
AMP = meta_action_id(MetaAction.CRAFT_AMPLIFIER)


@pytest.fixture()
def env():
    env = LayeredPyramidBandit(seed=3)
    env.expose_oracle = True
    return env


def test_no_meta_never_acts(env):
    assert NoMeta()(env.observation()) is None


def test_mult_only_waits_for_enough_data(env):
    policy = MultOnly()
    assert policy(env.observation()) is None
    env.info_units = env.info_per_multiplier
    assert policy(env.observation()) == INFO


def test_greedy_meta_priorities(env):
    policy = GreedyMeta()
    assert policy(env.observation()) is None
    env.n_multipliers = env.current_arm_cost
    assert policy(env.observation()) == BUY
    env.info_units = env.info_per_multiplier
    assert policy(env.observation()) == INFO  # конверсия приоритетнее покупки


def test_full_cycle_priorities(env):
    policy = FullCycle()
    env.data = {0: 8, 1: 4}
    assert policy(env.observation()) == GATE
    env.data = {0: 5, 1: 3}
    assert policy(env.observation()) == AMP
    env.data = {}
    env.info_units = env.info_per_multiplier
    assert policy(env.observation()) == INFO
    env.info_units = 0
    env.n_multipliers = env.current_arm_cost
    assert policy(env.observation()) == BUY


def test_gate_only_ignores_amplifier_branch(env):
    policy = GateOnly()
    env.data = {0: 5, 1: 3}  # хватает на амплификатор, но не на «ворота»
    assert env.can_craft_amplifier
    assert policy(env.observation()) is None
    assert policy(env.observation()) != AMP


def test_oracle_policies(env):
    best = int(env.means.argmax())
    assert OracleNoMeta()(env.observation()) == best

    env.data = {0: 8, 1: 4}
    assert OracleFullCycle()(env.observation()) == GATE
    env.data = {}
    assert OracleFullCycle()(env.observation()) == best


def test_oracle_full_cycle_collects_missing_tiers(env):
    """Жадный оракул не открывает эпохи: нужны данные всех тиров рецепта."""
    env.data = {0: 8, 1: 0}  # «воротам» не хватает только tier 1
    action = OracleFullCycle()(env.observation())
    assert env.arm_tiers[action] == 1


def test_oracle_requires_exposed_means():
    env = LayeredPyramidBandit(seed=3)
    assert env.observation().best_arm is None
    with pytest.raises(RuntimeError):
        OracleNoMeta()(env.observation())
    with pytest.raises(RuntimeError):
        OracleFullCycle()(env.observation())


def test_make_policy_and_registry():
    assert isinstance(make_policy(None), NoMeta)
    assert isinstance(make_policy("full_cycle"), FullCycle)
    assert set(POLICY_CLASSES) >= {"none", "full_cycle", "gate_only", "amp_only", "oracle_none"}
    with pytest.raises(KeyError):
        make_policy("nope")
