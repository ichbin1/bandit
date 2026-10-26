"""Реестр конфигов и наборы."""
from __future__ import annotations

import json

import pytest

from bandit.agents import resolve_agent_kind
from bandit.configs import (
    AGENTS,
    INFO_REGIMES,
    LEARNED_AGENT_KEYS,
    LEARNED_POLICY_KEYS,
    OVERLOAD_AGENT_POLICIES,
    OVERLOAD_BASE_REGIME,
    OVERLOAD_ORACLE_AGENT,
    POLICIES,
    PYRAMID_AGENT_KEYS,
    PYRAMID_POLICY_KEYS,
    PYRAMID_STEPS,
    REGISTRY,
    SUITES,
    get_config,
    resolve,
)
from bandit.envs import ENVIRONMENTS
from bandit.meta import POLICY_CLASSES
from bandit.report import split_config_name


def test_registry_size_and_uniqueness():
    base = len(PYRAMID_AGENT_KEYS) * len(PYRAMID_POLICY_KEYS)
    learned = len(LEARNED_AGENT_KEYS) * len(LEARNED_POLICY_KEYS)
    overload = len(INFO_REGIMES) * (len(OVERLOAD_AGENT_POLICIES) + 1)
    assert len(REGISTRY) == base + 2 + len(PYRAMID_AGENT_KEYS) + learned + overload
    assert len(REGISTRY) == 108
    assert len(set(REGISTRY)) == len(REGISTRY)


def test_learned_configs_exist_with_expected_params():
    for agent in LEARNED_AGENT_KEYS:
        for policy in LEARNED_POLICY_KEYS:
            name = f"pyramid-{agent}-{policy}"
            assert name in REGISTRY
            config = REGISTRY[name]
            assert config.policy["kind"] == "learned"
            assert config.n_steps == PYRAMID_STEPS
            assert config.policy["params"] == POLICIES[policy]["params"]
    assert REGISTRY["pyramid-ucb1-learned"].policy["params"]["window"] == 5
    assert REGISTRY["pyramid-ucb1-learned_slow"].policy["params"]["window"] == 8


def test_every_pyramid_combination_exists():
    for agent in PYRAMID_AGENT_KEYS:
        for policy in PYRAMID_POLICY_KEYS:
            name = f"pyramid-{agent}-{policy}"
            assert name in REGISTRY
            config = REGISTRY[name]
            assert config.env["kind"] == "layered"
            assert config.n_steps == 3000
            assert config.policy is not None and config.policy["kind"] == POLICIES[policy]["kind"]


def test_all_specs_are_resolvable():
    for name, config in REGISTRY.items():
        assert config.env["kind"] in ENVIRONMENTS, name
        assert resolve_agent_kind(config.agent["kind"]) in AGENTS_KEYS, name
        if config.policy is not None:
            assert config.policy["kind"] in POLICY_CLASSES, name


AGENTS_KEYS = {"random", "greedy", "epsilon_greedy", "ucb1", "thompson_bernoulli",
               "thompson_gaussian", "rescorla_wagner"}


def test_suites_are_consistent():
    assert set(SUITES) == {
        "pyramid",
        "pyramid-agents",
        "pyramid-policies",
        "pyramid-ablation",
        "pyramid-learned",
        "pyramid-hypotheses",
        "pyramid-overload",
        "stationary",
        "demo",
    }
    for suite, names in SUITES.items():
        assert names, suite
        assert len(set(names)) == len(names), suite
        for name in names:
            assert name in REGISTRY, (suite, name)
    assert len(SUITES["pyramid"]) == 58
    assert len(SUITES["pyramid-learned"]) == 7
    assert SUITES["pyramid-learned"][2] == "pyramid-ucb1-learned"
    assert len(SUITES["pyramid-hypotheses"]) == 13
    assert len(SUITES["pyramid-overload"]) == 36
    for name in ("pyramid-ucb1-gate_only", "pyramid-random-none", "pyramid-rw-full_cycle"):
        assert name in SUITES["pyramid-hypotheses"]
    assert "pyramid-oracle-full_cycle" in SUITES["pyramid-ablation"]
    assert "pyramid-oracle-none" not in SUITES["pyramid-agents"]


def test_overload_configs_cover_all_regimes():
    """H6: на каждый режим — 5 пар агент×политика и оракул с теми же параметрами."""
    assert OVERLOAD_BASE_REGIME in INFO_REGIMES
    assert len(INFO_REGIMES) == 6
    names = SUITES["pyramid-overload"]
    assert len(names) == len(INFO_REGIMES) * (len(OVERLOAD_AGENT_POLICIES) + 1)

    for name in names:
        parsed = split_config_name(name)
        assert parsed is not None, name
        env, agent, policy = parsed
        regime = env.split("pyramid-overload-", 1)[1]
        assert regime in INFO_REGIMES, name
        config = REGISTRY[name]
        for key, value in INFO_REGIMES[regime].items():
            assert config.env["params"][key] == value, (name, key)
        assert config.env["kind"] == "layered"
        assert config.n_steps == PYRAMID_STEPS
        if agent == OVERLOAD_ORACLE_AGENT:
            assert config.expose_oracle and policy == "full_cycle"
        else:
            assert not config.expose_oracle

    for regime in INFO_REGIMES:
        oracle = REGISTRY[f"pyramid-overload-{regime}-{OVERLOAD_ORACLE_AGENT}-full_cycle"]
        assert oracle.expose_oracle
        assert oracle.policy is not None and oracle.policy["kind"] == "oracle_full_cycle"


def test_stationary_configs():
    for name, config in REGISTRY.items():
        if not name.startswith("stationary-"):
            continue
        assert config.env["kind"] == "bernoulli"
        assert config.env["params"]["n_arms"] == 10
        assert config.n_steps == 1000
        assert config.policy["kind"] == "none"
    assert REGISTRY["stationary-ts"].agent["kind"] == "thompson_gaussian"


def test_oracle_configs_expose_means():
    assert REGISTRY["pyramid-oracle-none"].expose_oracle
    assert REGISTRY["pyramid-oracle-full_cycle"].expose_oracle
    assert REGISTRY["pyramid-ucb1-none"].expose_oracle is False


def test_resolve_selection_ordering_and_errors():
    names = [config.name for config in resolve(suite="pyramid-ablation")]
    assert names[0] == "pyramid-ucb1-none"
    assert len(names) == 7

    both = resolve(["pyramid-ucb1-none"], "pyramid-ablation")
    assert len(both) == 7  # дубликат не повторяется

    with pytest.raises(ValueError):
        resolve()
    with pytest.raises(KeyError):
        resolve(["nope"])
    with pytest.raises(KeyError):
        resolve(suite="nope")
    with pytest.raises(KeyError):
        get_config("nope")


def test_configs_are_json_serialisable():
    payload = json.dumps([config.to_dict() for config in REGISTRY.values()])
    assert len(payload) > 1000
