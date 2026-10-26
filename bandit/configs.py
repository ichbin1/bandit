"""Зарегистрированные конфиги экспериментов (сериализуемые описания).

Имя конфига = ``<среда>-<агент>-<мета-политика>``. Реестр строится
детерминированно, поэтому его можно использовать в ``--jobs > 1`` и сохранять
в манифест прогона.

Никаких «магических» констант в CLI: всё, что влияет на результат, лежит
здесь и попадает в ``results/<run_id>/manifest.json``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .runner import RunConfig

#: Параметры ``LayeredPyramidBandit``, совпадающие с демо версии 0.1.
LAYERED_PARAMS: Dict[str, object] = {
    "n_arms_init": 5,
    "n_levels": 3,
    "info_per_multiplier": 5,
    "mult_per_arm_base": 3,
    "arm_cost_increase": 1,
    "mult_bonus": 0.5,
    "arm_bonus": 0.5,
    "arm_bonus_increase": 0.1,
    "max_arms": 50,
    "tech_multiplier_bonus": 1.0,
    "tech_arm_mean_boost": 0.15,
    "amplifier_mult_bonus": 0.3,
    "amplifier_cost_growth": 1.5,
    "recombination_sigma": 0.1,
}

#: Параметры ``PyramidBandit`` (малая среда без уровней).
PYRAMID_PARAMS: Dict[str, object] = {
    "n_arms_init": 5,
    "info_per_multiplier": 5,
    "mult_per_arm_base": 3,
    "arm_cost_increase": 1,
    "mult_bonus": 0.5,
    "arm_bonus": 0.5,
    "max_arms": 50,
    "recombination_sigma": 0.1,
}

STATIONARY_PARAMS: Dict[str, object] = {"n_arms": 10}

#: Режимы информационной перегрузки (H6): параметры ``LayeredPyramidBandit``.
#:
#: Ось «объём × качество»: ``info_per_pull`` — сколько единиц приносит одно
#: вытягивание, ``useful_info_prob`` — доля полезных среди них,
#: ``junk_feeds_multiplier`` — вес мусора в топливе для конвертации данных
#: (0 — мусор только считается в знаменателе доли сигнала).
INFO_REGIMES: Dict[str, Dict[str, object]] = {
    "base": {"info_per_pull": 1, "useful_info_prob": 1.0, "junk_feeds_multiplier": 1.0},
    "abundant": {"info_per_pull": 4, "useful_info_prob": 1.0, "junk_feeds_multiplier": 1.0},
    "noisy": {"info_per_pull": 4, "useful_info_prob": 0.5, "junk_feeds_multiplier": 0.0},
    "noisy_fed": {"info_per_pull": 4, "useful_info_prob": 0.5, "junk_feeds_multiplier": 1.0},
    "starved": {"info_per_pull": 1, "useful_info_prob": 0.25, "junk_feeds_multiplier": 0.0},
    "flood": {"info_per_pull": 10, "useful_info_prob": 1.0, "junk_feeds_multiplier": 1.0},
}

#: Человекочитаемое описание режима (попадает в ``description`` конфига).
INFO_REGIME_NOTES: Dict[str, str] = {
    "base": "контроль: 1 единица, вся полезная",
    "abundant": "поток ×4, вся полезная: информация дешёвая",
    "noisy": "поток ×4, полезных 50%, мусор не кормит конвертацию",
    "noisy_fed": "поток ×4, полезных 50%, мусор кормит конвертацию",
    "starved": "поток ×1, полезных 25%: сигнал редкий",
    "flood": "поток ×10: конвертация почти каждый шаг",
}

#: Пары «агент × политика» внутри каждого режима перегрузки.
#: ``ucb1`` даёт пару для H6a, остальные — проверку устойчивости выигрыша
#: к качеству агента под шумом.
OVERLOAD_AGENT_POLICIES: List[tuple[str, str]] = [
    ("ucb1", "none"),
    ("ucb1", "full_cycle"),
    ("random", "full_cycle"),
    ("eps_decay", "full_cycle"),
    ("rw", "full_cycle"),
]

#: Имя «агента» в конфигах оракула (как у ``pyramid-oracle-*``).
OVERLOAD_ORACLE_AGENT = "oracle"

#: Базовый (контрольный) режим перегрузки.
OVERLOAD_BASE_REGIME = "base"

#: Агенты: ключ → описание для ``create_agent``.
AGENTS: Dict[str, Dict[str, object]] = {
    "random": {"kind": "random"},
    "greedy": {"kind": "greedy"},
    "eps0.01": {"kind": "epsilon_greedy", "params": {"epsilon": 0.01}},
    "eps0.1": {"kind": "epsilon_greedy", "params": {"epsilon": 0.1}},
    "eps_decay": {
        "kind": "epsilon_greedy",
        "params": {"epsilon": 0.3, "epsilon_decay": 0.999, "epsilon_min": 0.01},
    },
    "ucb1": {"kind": "ucb1", "params": {"c": 2.0}},
    "ts": {"kind": "thompson_bernoulli"},
    "rw": {
        "kind": "rescorla_wagner",
        "params": {"alpha": 0.1, "epsilon": 0.1, "initial_value": 0.5},
    },
}

#: Мета-политики: ключ → описание для ``create_policy``.
POLICIES: Dict[str, Dict[str, object]] = {
    "none": {"kind": "none"},
    "mult_only": {"kind": "mult_only"},
    "greedy": {"kind": "greedy"},
    "amp": {"kind": "amp"},
    "full_cycle": {"kind": "full_cycle"},
    "gate_only": {"kind": "gate_only"},
    "amp_only": {"kind": "amp_only"},
    "learned": {"kind": "learned", "params": {"c": 1.0, "window": 5}},
    "learned_slow": {"kind": "learned", "params": {"c": 1.0, "window": 8}},
    "oracle_none": {"kind": "oracle_none"},
    "oracle_full_cycle": {"kind": "oracle_full_cycle"},
}

PYRAMID_AGENT_KEYS: List[str] = [
    "random", "greedy", "eps0.01", "eps0.1", "eps_decay", "ucb1", "ts", "rw",
]
PYRAMID_POLICY_KEYS: List[str] = [
    "none", "mult_only", "greedy", "amp", "full_cycle", "gate_only", "amp_only",
]

LEARNED_AGENT_KEYS: List[str] = ["ucb1", "eps0.1", "random"]
"""Агенты для обучаемой политики: сильный, шумный и слабый (контраст)."""

LEARNED_POLICY_KEYS: List[str] = ["learned", "learned_slow"]
"""Ключи обучаемой политики: базовый горизонт и абляция окна измерения."""

PYRAMID_STEPS = 3000
STATIONARY_STEPS = 1000


def _layered_config(
    agent_key: str,
    policy_key: str,
    n_steps: int = PYRAMID_STEPS,
    expose_oracle: bool = False,
    env_params: Optional[Dict[str, object]] = None,
    name: Optional[str] = None,
    note: Optional[str] = None,
) -> RunConfig:
    params = dict(LAYERED_PARAMS)
    if env_params:
        params.update(env_params)
    return RunConfig(
        name=name or f"pyramid-{agent_key}-{policy_key}",
        env={"kind": "layered", "params": params},
        agent=dict(AGENTS[agent_key]),
        policy=dict(POLICIES[policy_key]),
        n_steps=int(n_steps),
        expose_oracle=bool(expose_oracle),
        description=(
            f"LayeredPyramidBandit, агент={agent_key}, мета-политика={policy_key}"
            + (", оракул" if expose_oracle else "")
            + (f"; {note}" if note else "")
        ),
    )


def _overload_config(regime: str, agent_key: str, policy_key: str) -> RunConfig:
    """Конфиг одного режима перегрузки (H6).

    Имя ``pyramid-overload-<режим>-<агент>-<политика>`` разбирается отчётом как
    ``<среда>-<агент>-<политика>`` с ``env = pyramid-overload-<режим>``, поэтому
    вердикты H1–H5 (чистая среда) и H6 (режимы) не смешиваются.
    """
    if regime not in INFO_REGIMES:
        raise KeyError(f"Неизвестный режим перегрузки {regime!r}")
    is_oracle = agent_key == OVERLOAD_ORACLE_AGENT
    return _layered_config(
        "random" if is_oracle else agent_key,
        "oracle_full_cycle" if is_oracle else policy_key,
        expose_oracle=is_oracle,
        env_params=dict(INFO_REGIMES[regime]),
        name=f"pyramid-overload-{regime}-{agent_key}-{policy_key}",
        note=f"режим {regime}: {INFO_REGIME_NOTES[regime]}",
    )


def build_registry() -> Dict[str, RunConfig]:
    """Все конфиги: факториал агент × политика + оракулы + стационарные."""
    registry: Dict[str, RunConfig] = {}

    for agent_key in PYRAMID_AGENT_KEYS:
        for policy_key in PYRAMID_POLICY_KEYS:
            config = _layered_config(agent_key, policy_key)
            registry[config.name] = config

    # Верхние границы: оракул по рукам без метацикла и с полным метациклом.
    registry["pyramid-oracle-none"] = _layered_config(
        "random", "oracle_none", expose_oracle=True, name="pyramid-oracle-none"
    )
    registry["pyramid-oracle-full_cycle"] = _layered_config(
        "random", "oracle_full_cycle", expose_oracle=True, name="pyramid-oracle-full_cycle"
    )

    # Обучаемая мета-политика (P4) не входит в факториал: она дороже по
    # вычислениям и интересна только с несколькими агентами и абляцией горизонта.
    for agent_key in LEARNED_AGENT_KEYS:
        for policy_key in LEARNED_POLICY_KEYS:
            config = _layered_config(agent_key, policy_key)
            registry[config.name] = config

    # H6: режимы информационной перегрузки — 5 пар агент×политика + оракул.
    for regime in INFO_REGIMES:
        for agent_key, policy_key in OVERLOAD_AGENT_POLICIES:
            config = _overload_config(regime, agent_key, policy_key)
            registry[config.name] = config
        oracle = _overload_config(regime, OVERLOAD_ORACLE_AGENT, "full_cycle")
        registry[oracle.name] = oracle

    # Классическая песочница: там сожаление определено корректно.
    for agent_key in PYRAMID_AGENT_KEYS:
        agent_spec: Dict[str, object] = dict(AGENTS[agent_key])
        if agent_spec["kind"] == "thompson_bernoulli":
            agent_spec = {"kind": "thompson_gaussian"}
        name = f"stationary-{agent_key}"
        registry[name] = RunConfig(
            name=name,
            env={"kind": "bernoulli", "params": dict(STATIONARY_PARAMS)},
            agent=agent_spec,
            policy={"kind": "none"},
            n_steps=STATIONARY_STEPS,
            description=f"Bernoulli-бандит (10 рук), агент={agent_key}",
        )
    return registry


REGISTRY: Dict[str, RunConfig] = build_registry()


def _suite(names: List[str]) -> List[str]:
    missing = [name for name in names if name not in REGISTRY]
    if missing:
        raise KeyError(f"В реестре нет конфигов: {', '.join(missing)}")
    return names


#: Наборы для ``python -m bandit.cli run --suite ...``.
SUITES: Dict[str, List[str]] = {
    "pyramid": _suite(
        [
            f"pyramid-{agent}-{policy}"
            for agent in PYRAMID_AGENT_KEYS
            for policy in PYRAMID_POLICY_KEYS
        ]
        + ["pyramid-oracle-none", "pyramid-oracle-full_cycle"]
    ),
    "pyramid-agents": _suite(
        [f"pyramid-{agent}-full_cycle" for agent in PYRAMID_AGENT_KEYS]
        + ["pyramid-oracle-full_cycle"]
    ),
    "pyramid-policies": _suite(
        [f"pyramid-ucb1-{policy}" for policy in PYRAMID_POLICY_KEYS]
        + ["pyramid-oracle-none", "pyramid-oracle-full_cycle"]
    ),
    "pyramid-ablation": _suite(
        [
            "pyramid-ucb1-none",
            "pyramid-ucb1-mult_only",
            "pyramid-ucb1-gate_only",
            "pyramid-ucb1-amp_only",
            "pyramid-ucb1-amp",
            "pyramid-ucb1-full_cycle",
            "pyramid-oracle-full_cycle",
        ]
    ),
    "stationary": _suite([f"stationary-{agent}" for agent in PYRAMID_AGENT_KEYS]),
    "pyramid-learned": _suite(
        [
            "pyramid-ucb1-none",
            "pyramid-ucb1-full_cycle",
            "pyramid-ucb1-learned",
            "pyramid-ucb1-learned_slow",
            "pyramid-eps0.1-learned",
            "pyramid-random-learned",
            "pyramid-oracle-full_cycle",
        ]
    ),
    "pyramid-hypotheses": _suite(
        [
            # H2: абляция компонент при одном агенте.
            "pyramid-ucb1-none",
            "pyramid-ucb1-mult_only",
            "pyramid-ucb1-gate_only",
            "pyramid-ucb1-amp_only",
            "pyramid-ucb1-amp",
            "pyramid-ucb1-full_cycle",
            # H3: сильный и слабый агент.
            "pyramid-random-none",
            "pyramid-random-full_cycle",
            # H4: агенты с забыванием при той же политике.
            "pyramid-eps_decay-full_cycle",
            "pyramid-rw-full_cycle",
            "pyramid-eps0.1-full_cycle",
            # P4 и верхняя граница.
            "pyramid-ucb1-learned",
            "pyramid-oracle-full_cycle",
        ]
    ),
    "pyramid-overload": _suite(
        [
            f"pyramid-overload-{regime}-{agent}-{policy}"
            for regime in INFO_REGIMES
            for agent, policy in OVERLOAD_AGENT_POLICIES
        ]
        + [
            f"pyramid-overload-{regime}-{OVERLOAD_ORACLE_AGENT}-full_cycle"
            for regime in INFO_REGIMES
        ]
    ),
    "demo": _suite(
        [
            "pyramid-random-none",
            "pyramid-ucb1-none",
            "pyramid-eps0.1-none",
            "pyramid-ucb1-mult_only",
            "pyramid-ucb1-gate_only",
            "pyramid-ucb1-full_cycle",
            "pyramid-oracle-full_cycle",
        ]
    ),
}


def get_config(name: str) -> RunConfig:
    if name not in REGISTRY:
        raise KeyError(
            f"Неизвестный конфиг {name!r}. Посмотрите `python -m bandit.cli list`."
        )
    return REGISTRY[name]


def resolve(names: Optional[List[str]] = None, suite: Optional[str] = None) -> List[RunConfig]:
    """Собрать список конфигов по именам и/или имени набора."""
    selected: List[str] = []
    if suite:
        if suite not in SUITES:
            raise KeyError(
                f"Неизвестный набор {suite!r}. Доступны: {', '.join(sorted(SUITES))}"
            )
        selected.extend(SUITES[suite])
    if names:
        selected.extend(names)
    if not selected:
        raise ValueError("Укажите --config и/или --suite")
    seen = set()
    configs: List[RunConfig] = []
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        configs.append(get_config(name))
    return configs
