"""Схема веб-интерфейса: поля конструктора и сборка конфигов из спецификации."""
from __future__ import annotations

import pytest

from bandit.configs import AGENTS, POLICIES
from bandit.envs import ENVIRONMENTS
from bandit.web import schema


def _spec(**overrides):
    spec = {
        "env": {"kind": "layered", "params": {}},
        "agents": [{"key": "ucb1", "params": {}}],
        "policies": [{"key": "none", "params": {}}],
        "steps": 64,
        "runs": 2,
        "seed": 7,
        "add_oracle": False,
    }
    spec.update(overrides)
    return spec


def test_ui_schema_lists_every_environment_agent_and_policy():
    ui = schema.ui_schema()
    assert {env["kind"] for env in ui["environments"]} == set(ENVIRONMENTS)
    assert {agent["key"] for agent in ui["agents"]} == set(AGENTS)
    assert {policy["key"] for policy in ui["policies"]} == set(POLICIES)
    assert ui["defaults"]["env"]["kind"] == "layered"
    assert ui["estimate"]["steps_per_second"] > 0
    assert "final_points" in ui["metrics"]
    assert any(suite["name"] == "pyramid-ablation" for suite in ui["suites"])


def test_layered_fields_expose_defaults_and_skip_callable():
    ui = schema.ui_schema()
    layered = next(env for env in ui["environments"] if env["kind"] == "layered")
    fields = {field["name"]: field for field in layered["fields"]}
    assert {"n_arms_init", "n_levels", "max_arms", "initial_tiers", "tech_costs"} <= set(fields)
    assert "arm_mean_sampler" in layered["ignored"]
    assert fields["initial_tiers"]["type"] == "json"
    assert fields["initial_tiers"]["default"] == [0, 0, 0, 1, 1]
    assert fields["tech_costs"]["default"][0] == {"0": 8, "1": 4}
    assert fields["junk_feeds_multiplier"]["group"] == "advanced"
    assert fields["n_arms_init"]["group"] == "base"
    assert layered["n_meta_actions"] == 4


def test_layered_presets_include_overload_regimes():
    ui = schema.ui_schema()
    layered = next(env for env in ui["environments"] if env["kind"] == "layered")
    keys = {preset["key"] for preset in layered["presets"]}
    assert "default" in keys
    assert {"regime-base", "regime-noisy", "regime-starved"} <= keys
    noisy = next(preset for preset in layered["presets"] if preset["key"] == "regime-noisy")
    assert noisy["params"]["info_per_pull"] == 4


def test_stationary_means_field_is_json_and_nullable():
    ui = schema.ui_schema()
    bernoulli = next(env for env in ui["environments"] if env["kind"] == "bernoulli")
    means = next(field for field in bernoulli["fields"] if field["name"] == "means")
    assert means["type"] == "json"
    assert means["default"] is None
    assert bernoulli["n_meta_actions"] == 0


def test_agent_fields_use_config_values():
    ui = schema.ui_schema()
    eps = next(agent for agent in ui["agents"] if agent["key"] == "eps_decay")
    values = {field["name"]: field["default"] for field in eps["fields"]}
    assert values["epsilon"] == pytest.approx(0.3)
    assert values["epsilon_decay"] == pytest.approx(0.999)
    assert values["epsilon_min"] == pytest.approx(0.01)
    ucb1 = next(agent for agent in ui["agents"] if agent["key"] == "ucb1")
    assert {field["name"] for field in ucb1["fields"]} == {"c"}


def test_policy_fields_only_for_learned():
    ui = schema.ui_schema()
    learned = next(policy for policy in ui["policies"] if policy["key"] == "learned")
    assert {field["name"] for field in learned["fields"]} == {"c", "window"}
    none = next(policy for policy in ui["policies"] if policy["key"] == "none")
    assert none["fields"] == []

def test_build_configs_makes_agent_policy_matrix_with_oracle():
    configs = schema.build_configs(
        _spec(
            agents=[{"key": "ucb1"}, {"key": "random"}],
            policies=[{"key": "none"}, {"key": "full_cycle"}],
            add_oracle=True,
        )
    )
    assert [config.name for config in configs] == [
        "ui-layered-ucb1-none",
        "ui-layered-ucb1-full_cycle",
        "ui-layered-random-none",
        "ui-layered-random-full_cycle",
        "ui-layered-oracle-full_cycle",
    ]
    oracle = configs[-1]
    assert oracle.expose_oracle
    assert oracle.policy["kind"] == "oracle_full_cycle"
    assert oracle.n_steps == 64
    # Параметры среды берутся из LAYERED_PARAMS, а не из «сырых» defaults класса.
    assert configs[0].env["params"]["n_levels"] == 3


def test_build_configs_applies_param_overrides():
    configs = schema.build_configs(
        _spec(
            env={"kind": "layered", "params": {"n_levels": 1, "max_arms": 5}},
            agents=[{"key": "ucb1", "params": {"c": 0.5}}],
            policies=[{"key": "learned", "params": {"window": 7}}],
        )
    )
    config = configs[0]
    assert config.env["params"]["n_levels"] == 1
    assert config.env["params"]["max_arms"] == 5
    assert config.agent["params"]["c"] == pytest.approx(0.5)
    assert config.policy["params"] == {"c": 1.0, "window": 7}


def test_oracle_policy_gets_expose_oracle_automatically():
    configs = schema.build_configs(_spec(policies=[{"key": "oracle_none"}]))
    assert configs[0].expose_oracle
    assert configs[0].policy["kind"] == "oracle_none"


def test_build_configs_rejects_unknown_keys_and_empty_selection():
    with pytest.raises(KeyError):
        schema.build_configs(_spec(env={"kind": "nope"}))
    with pytest.raises(KeyError):
        schema.build_configs(_spec(agents=[{"key": "nope"}]))
    with pytest.raises(KeyError):
        schema.build_configs(_spec(policies=[{"key": "nope"}]))
    with pytest.raises(ValueError):
        schema.build_configs(_spec(agents=[]))
    with pytest.raises(ValueError):
        schema.build_configs(_spec(policies=[]))
    with pytest.raises(ValueError):
        schema.build_configs(_spec(steps=0))


def test_build_configs_deduplicates_pairs():
    configs = schema.build_configs(_spec(policies=[{"key": "none"}, {"key": "none"}]))
    assert [config.name for config in configs] == ["ui-layered-ucb1-none"]


def test_config_name_is_path_safe():
    name = schema.config_name("layered", "eps 0.1/x", "full cycle")
    assert name == "ui-layered-eps-0.1-x-full-cycle"
    assert "/" not in name and " " not in name


def test_preview_estimates_work_from_measured_speed():
    configs = schema.build_configs(
        _spec(policies=[{"key": "none"}, {"key": "full_cycle"}], steps=1000)
    )
    data = schema.preview(configs, {"runs": 4})
    assert data["n_configs"] == 2
    assert data["n_episodes"] == 8
    assert data["total_steps"] == 8000
    assert data["est_seconds"] == pytest.approx(
        8000 / schema.EST_STEPS_PER_SECOND, rel=1e-6
    )


def test_validate_spec_runs_smoke_and_reports_env_errors():
    good = schema.validate_spec(_spec())
    assert good == {"ok": True, "n_configs": 1, "errors": []}

    bad = schema.validate_spec(
        _spec(
            env={"kind": "bernoulli", "params": {"n_arms": 3, "means": [0.1, 0.2]}},
            policies=[{"key": "none"}],
        )
    )
    assert bad["ok"] is False
    assert bad["n_configs"] == 1
    assert "means" in bad["errors"][0]


def test_validate_spec_reports_unknown_agent_without_raising():
    result = schema.validate_spec(_spec(agents=[{"key": "nope"}]))
    assert result["ok"] is False
    assert result["n_configs"] == 0
    assert "nope" in result["errors"][0]

