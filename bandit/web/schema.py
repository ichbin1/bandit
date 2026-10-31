"""Схема веб-интерфейса: что показывать и как собрать конфиги.

Модуль намеренно ничего не знает про FastAPI:

* ``ui_schema()`` — описание всех регулировок (среды, агенты, политики) для
  браузера: поля, границы, defaults, подсказки;
* ``build_configs(spec)`` — чистая функция «спецификация → ``RunConfig``»;
* ``preview()`` / ``validate_spec()`` — оценка объёма работы и дешёвый прогон
  по 32 шага, который ловит неверные параметры до запуска.

Такую же спецификацию можно сохранить в ``spec.json`` и воспроизвести из CLI
(``bandit.cli run --config-file results/<id>/configs.json``).
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .. import __version__
from ..agents import AGENT_CLASSES
from ..configs import (
    AGENTS,
    INFO_REGIME_NOTES,
    INFO_REGIMES,
    LAYERED_PARAMS,
    POLICIES,
    PYRAMID_PARAMS,
    STATIONARY_PARAMS,
    SUITES,
)
from ..envs import ENVIRONMENTS
from ..metrics import METRIC_DESCRIPTIONS
from ..runner import RunConfig, run_config

DEFAULT_STEPS = 3000
"""Горизонт нового эксперимента: как ``configs.PYRAMID_STEPS``."""

DEFAULT_RUNS = 30
DEFAULT_SEED = 42
SMOKE_STEPS = 32
"""Столько шагов считает ``validate_spec``: хватает, чтобы поймать битые параметры."""

EST_STEPS_PER_SECOND = 5000.0
"""Оценка скорости одного процесса (замеры ``adv-s1``/``adv-s2``/``_h6_10``)."""

#: Пресеты параметров среды: с них начинается редактирование.
_ENV_PRESETS: Dict[str, Dict[str, Any]] = {
    "layered": dict(LAYERED_PARAMS),
    "pyramid": dict(PYRAMID_PARAMS),
    "bernoulli": dict(STATIONARY_PARAMS),
    "gaussian": dict(STATIONARY_PARAMS),
}

_ENV_LABELS: Dict[str, str] = {
    "layered": "Многоуровневая пирамида (основная среда)",
    "pyramid": "Пирамида 0.1 — малая среда",
    "bernoulli": "Стационарный Bernoulli",
    "gaussian": "Стационарный Gaussian",
}

_AGENT_LABELS: Dict[str, str] = {
    "random": "Случайный выбор",
    "greedy": "Жадный (чистая эксплуатация)",
    "eps0.01": "ε-жадный, ε = 0.01",
    "eps0.1": "ε-жадный, ε = 0.1",
    "eps_decay": "ε-жадный с затуханием",
    "ucb1": "UCB1",
    "ts": "Thompson Sampling (Bernoulli)",
    "rw": "Рескорла–Вагнер",
}

_POLICY_LABELS: Dict[str, str] = {
    "none": "Без метацикла (контроль)",
    "mult_only": "Только конвертация данных",
    "greedy": "Жадная: множители → руки",
    "amp": "Жадная + амплификаторы",
    "full_cycle": "Полный метацикл",
    "gate_only": "Абляция: только технологии",
    "amp_only": "Абляция: только амплификаторы",
    "learned": "Обучаемая (UCB-мета, окно 5)",
    "learned_slow": "Обучаемая (окно 8)",
    "oracle_none": "Оракул по рукам, без метацикла",
    "oracle_full_cycle": "Оракул по рукам + метацикл",
}

#: Подсказки к полям сред: подпись, группа, границы, смысл.
_ENV_FIELD_HINTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "layered": {
        "n_arms_init": {
            "label": "Стартовых рук", "min": 1, "max": 200, "step": 1,
            "help": "Сколько рук доступно на старте; длина списка тиров.",
        },
        "n_levels": {
            "label": "Эпох (уровней)", "min": 1, "max": 6, "step": 1,
            "help": "1 — развитие выключено: ворота не открывают новые эпохи.",
        },
        "info_per_multiplier": {
            "label": "Данных на множитель", "min": 1, "max": 200, "step": 1,
            "help": "Сколько единиц информации стоят одного множителя.",
        },
        "arm_bonus_increase": {
            "label": "Рост бонуса за новую руку", "min": 0.0, "max": 5.0, "step": 0.05,
            "help": "Главный источник выигрыша в S1-оси engine: бонус растёт с покупками.",
        },
        "max_arms": {
            "label": "Потолок рук", "min": 1, "max": 500, "step": 1,
            "help": "Ограничивает и покупки, и ворота (cap5 запрещает покупки).",
        },
        "info_per_pull": {
            "label": "Информации за вытягивание", "group": "advanced",
            "min": 1, "max": 200, "step": 1, "help": "Ось H6: объём потока данных.",
        },
        "useful_info_prob": {
            "label": "Доля полезной информации", "group": "advanced",
            "min": 0.0, "max": 1.0, "step": 0.05, "help": "Ось H6: качество потока.",
        },
        "junk_feeds_multiplier": {
            "label": "Мусор идёт в топливо", "group": "advanced",
            "min": 0.0, "max": 1.0, "step": 1.0,
            "help": "1 — мусор тоже кормит конвертацию, 0 — только в знаменателе доли.",
        },
        "mult_per_arm_base": {
            "label": "Множителей за новую руку", "group": "advanced",
            "min": 1, "max": 200, "step": 1,
        },
        "arm_cost_increase": {
            "label": "Рост цены руки", "group": "advanced", "min": 0, "max": 20, "step": 1,
        },
        "mult_bonus": {
            "label": "Бонус множителя за эпоху", "group": "advanced",
            "min": 0.0, "max": 10.0, "step": 0.1,
        },
        "arm_bonus": {
            "label": "Стартовый бонус новой руки", "group": "advanced",
            "min": 0.0, "max": 10.0, "step": 0.1,
        },
        "tech_multiplier_bonus": {
            "label": "Множитель за технологию", "group": "advanced",
            "min": 0.0, "max": 10.0, "step": 0.1,
        },
        "tech_arm_mean_boost": {
            "label": "Усиление рук технологией", "group": "advanced",
            "min": 0.0, "max": 1.0, "step": 0.05,
        },
        "amplifier_mult_bonus": {
            "label": "Бонус амплификатора", "group": "advanced",
            "min": 0.0, "max": 10.0, "step": 0.1,
        },
        "amplifier_cost_growth": {
            "label": "Рост цены амплификатора", "group": "advanced",
            "min": 1.0, "max": 5.0, "step": 0.1,
        },
        "recombination_sigma": {
            "label": "σ рекомбинации", "group": "advanced", "min": 0.0, "max": 2.0, "step": 0.05,
        },
        "initial_tiers": {
            "label": "Тиры стартовых рук", "type": "json", "group": "advanced",
            "help": "Список целых (обрезается до n_arms_init), например [0, 0, 0, 1, 1].",
        },
        "unlock_tiers": {
            "label": "Тиры, открываемые эпохами", "type": "json", "group": "advanced",
            "help": "Список списков: что добавляет каждый переход эпохи.",
        },
        "tech_costs": {
            "label": "Рецепты технологий", "type": "json", "group": "advanced",
            "help": "Список словарей {тир: количество}; длина = n_levels - 1.",
        },
        "amplifier_base_cost": {
            "label": "Цена амплификатора", "type": "json", "group": "advanced",
            "help": "Словарь {тир: количество}.",
        },
    },
    "pyramid": {
        "n_arms_init": {"label": "Стартовых рук", "min": 1, "max": 200, "step": 1},
        "info_per_multiplier": {"label": "Данных на множитель", "min": 1, "max": 200, "step": 1},
        "mult_per_arm_base": {"label": "Множителей за новую руку", "min": 1, "max": 200, "step": 1},
        "arm_cost_increase": {"label": "Рост цены руки", "group": "advanced", "min": 0, "max": 20, "step": 1},
        "mult_bonus": {"label": "Бонус множителя", "group": "advanced", "min": 0.0, "max": 10.0, "step": 0.1},
        "arm_bonus": {"label": "Бонус новой руки", "group": "advanced", "min": 0.0, "max": 10.0, "step": 0.1},
        "max_arms": {"label": "Потолок рук", "group": "advanced", "min": 1, "max": 500, "step": 1},
        "recombination_sigma": {"label": "σ рекомбинации", "group": "advanced", "min": 0.0, "max": 2.0, "step": 0.05},
    },
    "bernoulli": {
        "n_arms": {"label": "Рук", "min": 1, "max": 200, "step": 1},
        "means": {
            "label": "Вероятности рук", "type": "json",
            "help": "Список чисел в [0, 1] длиной n_arms; null — разыграть из сида.",
        },
    },
    "gaussian": {
        "n_arms": {"label": "Рук", "min": 1, "max": 200, "step": 1},
        "means": {
            "label": "Средние рук", "type": "json",
            "help": "Список чисел длиной n_arms; null — разыграть из сида.",
        },
        "std": {"label": "σ шума", "min": 0.01, "max": 10.0, "step": 0.05},
    },
}

#: Подсказки к параметрам агентов и обучаемых политик.
_FIELD_HINTS: Dict[str, Dict[str, Any]] = {
    "epsilon": {"label": "ε (доля исследования)", "min": 0.0, "max": 1.0, "step": 0.01,
                "help": "Вероятность случайной руки вместо жадной."},
    "epsilon_decay": {"label": "Затухание ε за шаг", "min": 0.9, "max": 1.0, "step": 0.001,
                      "help": "1.0 — постоянное ε; 0.999 — медленное затухание."},
    "epsilon_min": {"label": "Нижняя граница ε", "min": 0.0, "max": 1.0, "step": 0.001},
    "initial_value": {"label": "Начальная оценка", "min": -10.0, "max": 10.0, "step": 0.05},
    "c": {"label": "c (ширина бонуса)", "min": 0.0, "max": 10.0, "step": 0.1,
          "help": "UCB: value + sqrt(c · ln t / n)."},
    "alpha": {"label": "α (скорость обучения)", "min": 0.0, "max": 1.0, "step": 0.01},
    "window": {"label": "Окно измерения, шагов", "min": 1, "max": 100, "step": 1,
               "help": "Сколько шагов политика ждёт, чтобы оценить инвестицию."},
}

#: Значения JSON-полей по умолчанию берём у самой среды, а не из ``None``.
_JSON_DEFAULTS: Dict[Tuple[str, str], Any] = {
    ("layered", "initial_tiers"): lambda cls: list(cls.DEFAULT_INITIAL_TIERS),
    ("layered", "unlock_tiers"): lambda cls: [list(tiers) for tiers in cls.DEFAULT_UNLOCK_TIERS],
    ("layered", "tech_costs"): lambda cls: [
        {str(tier): int(amount) for tier, amount in recipe.items()}
        for recipe in cls.DEFAULT_TECH_COSTS
    ],
    ("layered", "amplifier_base_cost"): lambda cls: {
        str(tier): int(amount) for tier, amount in cls.DEFAULT_AMPLIFIER_BASE_COST.items()
    },
}

_SCALAR_TYPES = {"int": "int", "float": "float", "bool": "bool", "str": "str"}


def _field_type(annotation: Any) -> Optional[str]:
    """Тип поля для UI; ``None`` — параметр нельзя задать из браузера."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    text = str(text).replace("typing.", "").strip()
    if "Callable" in text:
        return None
    if text in _SCALAR_TYPES:
        return _SCALAR_TYPES[text]
    optional = re.fullmatch(r"Optional\[(.+)\]", text)
    if optional:
        return _field_type(optional.group(1))
    return "json"


def _signature(cls: type) -> Dict[str, inspect.Parameter]:
    """Параметры конструктора без ``self``/``seed`` (сид задаётся экспериментом)."""
    return {
        name: param
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name not in ("self", "seed")
    }


def _default_value(kind: str, name: str, default: Any) -> Any:
    factory = _JSON_DEFAULTS.get((kind, name))
    if factory is not None:
        return factory(ENVIRONMENTS[kind])
    if default is inspect.Parameter.empty:
        return None
    return default

def _env_fields(kind: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Поля среды + список параметров, недоступных из UI (callable)."""
    hints = _ENV_FIELD_HINTS.get(kind, {})
    fields: List[Dict[str, Any]] = []
    ignored: List[str] = []
    for name, param in _signature(ENVIRONMENTS[kind]).items():
        hint = dict(hints.get(name, {}))
        field_type = hint.get("type") or _field_type(param.annotation)
        if field_type is None:
            ignored.append(name)
            continue
        fields.append({
            "name": name,
            "label": hint.get("label") or name.replace("_", " "),
            "type": field_type,
            "default": _default_value(kind, name, param.default),
            "group": hint.get("group", "base"),
            "min": hint.get("min"),
            "max": hint.get("max"),
            "step": hint.get("step"),
            "help": hint.get("help", ""),
        })
    return fields, ignored


def _agent_fields(agent_key: str) -> List[Dict[str, Any]]:
    """Поля агента: значения по умолчанию = то, что реально пойдёт в прогон."""
    config = AGENTS[agent_key]
    current = dict(config.get("params") or {})
    fields: List[Dict[str, Any]] = []
    for name, param in _signature(AGENT_CLASSES[str(config["kind"])]).items():
        if name == "n_actions":
            continue
        field_type = _field_type(param.annotation)
        if field_type is None:
            continue
        hint = _FIELD_HINTS.get(name, {})
        default = current.get(name, param.default)
        fields.append({
            "name": name,
            "label": hint.get("label") or name.replace("_", " "),
            "type": hint.get("type", field_type),
            "default": None if default is inspect.Parameter.empty else default,
            "group": "base",
            "min": hint.get("min"),
            "max": hint.get("max"),
            "step": hint.get("step"),
            "help": hint.get("help", ""),
        })
    return fields


def _policy_fields(policy_key: str) -> List[Dict[str, Any]]:
    """Поля политики: только те, что она принимает (у ``learned`` это c и window)."""
    fields: List[Dict[str, Any]] = []
    for name, default in dict(POLICIES[policy_key].get("params") or {}).items():
        hint = _FIELD_HINTS.get(name, {})
        is_int = isinstance(default, int) and not isinstance(default, bool)
        fields.append({
            "name": name,
            "label": hint.get("label") or name.replace("_", " "),
            "type": hint.get("type", "int" if is_int else "float"),
            "default": default,
            "group": "base",
            "min": hint.get("min"),
            "max": hint.get("max"),
            "step": hint.get("step", 1 if is_int else 0.1),
            "help": hint.get("help", ""),
        })
    return fields


def _env_presets(kind: str) -> List[Dict[str, Any]]:
    """Пресеты среды: ``default`` плюс режимы перегрузки H6 (только layered)."""
    presets: List[Dict[str, Any]] = [{
        "key": "default",
        "label": "По умолчанию",
        "params": dict(_ENV_PRESETS.get(kind, {})),
    }]
    if kind == "layered":
        for regime, params in INFO_REGIMES.items():
            presets.append({
                "key": f"regime-{regime}",
                "label": f"Инфопоток: {INFO_REGIME_NOTES.get(regime, regime)}",
                "params": dict(params),
            })
    return presets

def ui_schema() -> Dict[str, Any]:
    """Полное описание конструктора: среды, агенты, политики, наборы, defaults."""
    environments: List[Dict[str, Any]] = []
    for kind, cls in ENVIRONMENTS.items():
        fields, ignored = _env_fields(kind)
        meta = tuple(getattr(cls, "META", ()))
        environments.append({
            "kind": kind,
            "label": _ENV_LABELS.get(kind, kind),
            "n_meta_actions": len(meta),
            "meta_actions": [getattr(action, "name", str(action)) for action in meta],
            "supports_epochs": bool(hasattr(cls, "DEFAULT_UNLOCK_TIERS")),
            "presets": _env_presets(kind),
            "fields": fields,
            "ignored": ignored,
        })
    agents = [
        {
            "key": key,
            "label": _AGENT_LABELS.get(key, key),
            "kind": str(config["kind"]),
            "fields": _agent_fields(key),
        }
        for key, config in AGENTS.items()
    ]
    policies = [
        {
            "key": key,
            "label": _POLICY_LABELS.get(key, key),
            "kind": str(config["kind"]),
            "oracle": str(config["kind"]).startswith("oracle"),
            "fields": _policy_fields(key),
        }
        for key, config in POLICIES.items()
    ]
    return {
        "version": __version__,
        "defaults": {
            "title": "",
            "env": {"kind": "layered", "params": dict(LAYERED_PARAMS)},
            "agents": ["ucb1"],
            "policies": ["none", "full_cycle"],
            "runs": DEFAULT_RUNS,
            "seed": DEFAULT_SEED,
            "steps": DEFAULT_STEPS,
            "jobs": 1,
            "plots": True,
            "add_oracle": True,
        },
        "estimate": {"steps_per_second": EST_STEPS_PER_SECOND},
        "environments": environments,
        "agents": agents,
        "policies": policies,
        "suites": [
            {"name": name, "n_configs": len(names), "configs": list(names)}
            for name, names in SUITES.items()
        ],
        "metrics": dict(METRIC_DESCRIPTIONS),
    }

#: Префикс имён конфигов из UI: видно происхождение в ``runs.csv`` и отчёте.
NAME_PREFIX = "ui"

_SAFE_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def _slug(text: str) -> str:
    return _SAFE_RE.sub("-", str(text)).strip("-.") or "x"


def config_name(env_kind: str, agent_key: str, policy_key: str) -> str:
    """Имя конфига: ``ui-<среда>-<агент>-<политика>``."""
    return f"{NAME_PREFIX}-{_slug(env_kind)}-{_slug(agent_key)}-{_slug(policy_key)}"


def _spec_env(spec: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Среда из спецификации: ``kind`` + параметры поверх пресета."""
    env_spec = dict(spec.get("env") or {})
    kind = str(env_spec.get("kind") or "layered")
    if kind not in ENVIRONMENTS:
        raise KeyError(
            f"Неизвестная среда {kind!r}. Доступны: {', '.join(sorted(ENVIRONMENTS))}"
        )
    params = dict(_ENV_PRESETS.get(kind, {}))
    params.update(env_spec.get("params") or {})
    return kind, params


def build_configs(spec: Mapping[str, Any]) -> List[RunConfig]:
    """Собрать ``RunConfig`` из спецификации: агенты × политики (+ оракул).

    Параметры среды/агента/политики накладываются на значения из
    ``configs.py``, поэтому пустой ``params`` означает «как в реестре».
    Все проверки делает конструктор ``RunConfig``/среды — здесь только имена
    ключей, чтобы ошибка была понятной.
    """
    kind, env_params = _spec_env(spec)
    raw_steps = spec.get("steps")
    steps = DEFAULT_STEPS if raw_steps is None else int(raw_steps)
    if steps <= 0:
        raise ValueError("steps должен быть положительным")
    agent_items = list(spec.get("agents") or [])
    policy_items = list(spec.get("policies") or [])
    if not agent_items:
        raise ValueError("Выберите хотя бы одного агента")
    if not policy_items:
        raise ValueError("Выберите хотя бы одну мета-политику")
    configs: List[RunConfig] = []
    seen: set[str] = set()
    for agent_item in agent_items:
        agent_key = str((agent_item or {}).get("key") or "")
        if agent_key not in AGENTS:
            raise KeyError(
                f"Неизвестный агент {agent_key!r}. Доступны: {', '.join(AGENTS)}"
            )
        agent_params = dict(AGENTS[agent_key].get("params") or {})
        agent_params.update((agent_item or {}).get("params") or {})
        for policy_item in policy_items:
            policy_key = str((policy_item or {}).get("key") or "")
            if policy_key not in POLICIES:
                raise KeyError(
                    f"Неизвестная мета-политика {policy_key!r}. "
                    f"Доступны: {', '.join(POLICIES)}"
                )
            policy_params = dict(POLICIES[policy_key].get("params") or {})
            policy_params.update((policy_item or {}).get("params") or {})
            policy_kind = str(POLICIES[policy_key]["kind"])
            name = config_name(kind, agent_key, policy_key)
            if name in seen:
                continue
            seen.add(name)
            configs.append(RunConfig(
                name=name,
                env={"kind": kind, "params": dict(env_params)},
                agent={"kind": AGENTS[agent_key]["kind"], "params": agent_params},
                policy={"kind": policy_kind, "params": policy_params},
                n_steps=steps,
                expose_oracle=policy_kind.startswith("oracle"),
                description=f"UI: {kind} · {agent_key} · {policy_key}",
            ))
    if bool(spec.get("add_oracle")):
        name = config_name(kind, "oracle", "full_cycle")
        if name not in seen:
            seen.add(name)
            configs.append(RunConfig(
                name=name,
                env={"kind": kind, "params": dict(env_params)},
                agent={"kind": AGENTS["random"]["kind"]},
                policy={"kind": POLICIES["oracle_full_cycle"]["kind"]},
                n_steps=steps,
                expose_oracle=True,
                description=f"UI: {kind} · оракул (верхняя граница)",
            ))
    if not configs:
        raise ValueError("Не удалось собрать ни одного конфига")
    return configs


def preview(configs: Sequence[RunConfig], spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Оценка объёма: число конфигов, эпизодов, шагов и секунд (по замеру)."""
    runs = int(spec.get("runs") or DEFAULT_RUNS)
    n_steps = int(configs[0].n_steps) if configs else 0
    total_steps = len(configs) * runs * n_steps
    return {
        "configs": [config.name for config in configs],
        "n_configs": len(configs),
        "n_episodes": len(configs) * runs,
        "total_steps": total_steps,
        "est_seconds": round(total_steps / EST_STEPS_PER_SECOND, 2),
    }


def validate_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Дешёвая проверка: по 32 шага на конфиг, ошибки — как их видит среда."""
    try:
        configs = build_configs(spec)
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "n_configs": 0, "errors": [_error_text(exc)]}
    seed = int(spec.get("seed") or DEFAULT_SEED)
    errors: List[str] = []
    for config in configs:
        try:
            run_config(
                config,
                n_runs=1,
                base_seed=seed,
                jobs=1,
                n_steps=min(SMOKE_STEPS, config.n_steps),
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 — показываем текст конструктора среды
            errors.append(f"{config.name}: {_error_text(exc)}")
    return {"ok": not errors, "n_configs": len(configs), "errors": errors}


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc) or type(exc).__name__





