"""Контракт среды: стабильные id, мета-действия, запись шага, экономика."""
from __future__ import annotations

import numpy as np
import pytest

from bandit.envs import (
    BernoulliBandit,
    LayeredPyramidBandit,
    MetaAction,
    is_meta_action,
    legacy_action_to_stable,
    meta_action_id,
    stable_action_to_legacy,
)


@pytest.fixture()
def env():
    return LayeredPyramidBandit(seed=1)


def test_action_space_ids_are_stable(env):
    space = env.action_space
    assert list(space.arm_ids()) == list(range(5))
    assert space.meta_ids() == (
        meta_action_id(MetaAction.INFO_TO_MULTIPLIER),
        meta_action_id(MetaAction.BUY_ARM),
        meta_action_id(MetaAction.CRAFT_GATE),
        meta_action_id(MetaAction.CRAFT_AMPLIFIER),
    )
    assert all(is_meta_action(a) for a in space.meta_ids())
    assert space.n_actions == env.n_arms + 4
    assert space.contains(0) and not space.contains(5)


def test_meta_ids_do_not_shift_when_arms_grow(env):
    before = env.action_space.meta_ids()
    env.n_multipliers = 100  # хватает на покупку
    info = env.step(meta_action_id(MetaAction.BUY_ARM))
    assert info.was_effective
    assert env.n_arms == 6
    assert env.action_space.meta_ids() == before
    # старый id руки остаётся рукой, новая рука получает id 5
    assert env.step(4).is_arm_pull
    assert env.step(5).is_arm_pull


def test_legacy_index_requires_explicit_conversion(env):
    with pytest.raises(ValueError, match="legacy"):
        env.step(env.n_arms)
    assert env.pull_legacy(env.n_arms) == 0.0
    assert legacy_action_to_stable(env.n_arms, env.n_arms) == meta_action_id(
        MetaAction.INFO_TO_MULTIPLIER
    )
    assert stable_action_to_legacy(meta_action_id(MetaAction.CRAFT_GATE), env.n_arms) == 7


def test_invalid_actions_are_rejected(env):
    with pytest.raises(TypeError):
        env.step(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        env.step(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(10_004)  # неизвестный вид мета-действия


def test_arm_pull_reward_is_base_signal_and_points_use_multiplier(env):
    info = env.step(0)
    assert info.is_arm_pull
    assert info.reward in (0.0, 1.0)
    assert info.points_gained == pytest.approx(info.reward * info.before.multiplier)
    assert info.kind_label == "arm"
    assert info.after.points == pytest.approx(info.before.points + info.points_gained)
    assert info.after.t == info.before.t + 1


def test_data_has_tier_of_pulled_arm(env):
    assert env.arm_tiers[3] == 1
    env.step(3)
    assert env.data[1] == 1
    assert env.data.get(0, 0) == 0
    assert env.info_units == 1


def test_conversion_needs_enough_data(env):
    idle = env.step(meta_action_id(MetaAction.INFO_TO_MULTIPLIER))
    assert not idle.was_effective
    assert env.n_multipliers == 0

    for _ in range(env.info_per_multiplier):
        env.step(0)
    converted = env.step(meta_action_id(MetaAction.INFO_TO_MULTIPLIER))
    assert converted.was_effective
    assert env.n_multipliers == 1
    assert env.info_units == 0
    assert converted.points_gained == 0.0


def test_gate_and_amplifier_are_unaffordable_at_start(env):
    for kind in (MetaAction.CRAFT_GATE, MetaAction.CRAFT_AMPLIFIER):
        info = env.step(meta_action_id(kind))
        assert not info.was_effective
        assert info.points_gained == 0.0
    assert env.epoch == 0
    assert env.techs_crafted == 0


def test_oracle_is_hidden_by_default(env):
    assert env.observation().best_arm is None
    env.expose_oracle = True
    observation = env.observation()
    assert observation.best_arm == int(np.argmax(env.means))
    assert observation.best_mean == pytest.approx(float(env.means.max()))
    assert observation.arm_means == tuple(float(mu) for mu in env.means)


def test_observation_reports_tiers_and_missing_recipe(env):
    observation = env.observation()
    assert observation.arm_tiers == tuple(int(t) for t in env.arm_tiers)
    assert len(observation.arm_tiers) == env.n_arms
    assert observation.tech_cost == ((0, 8), (1, 4))
    assert observation.missing_tech_tiers() == (0, 1)
    env.data = {0: 8, 1: 1}
    assert env.observation().missing_tech_tiers() == (1,)
    env.data = {0: 8, 1: 4}
    assert env.observation().missing_tech_tiers() == ()


def test_reset_is_deterministic_for_same_seed():
    first = LayeredPyramidBandit(seed=123)
    second = LayeredPyramidBandit(seed=123)
    assert first.means.tolist() == second.means.tolist()
    assert first.arm_tiers.tolist() == second.arm_tiers.tolist()


def test_stationary_environment_has_no_meta_actions():
    env = BernoulliBandit(n_arms=3, seed=0)
    assert env.action_space.meta == ()
    assert env.action_space.n_actions == 3
    with pytest.raises(ValueError):
        env.step(meta_action_id(MetaAction.BUY_ARM))
    info = env.step(1)
    assert info.is_arm_pull
    assert info.arm_regret == pytest.approx(info.best_mean - info.chosen_mean)
