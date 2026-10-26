"""Создание сред, агентов и политик из сериализуемых описаний.

Описания — обычные словари (``{"kind": ..., "params": {...}}``). Это важно для
Windows: ``ProcessPoolExecutor`` использует spawn и pickle, поэтому конфиги не
должны содержать лямбд и замыканий.

ВАЖНО: ``dtype`` и порядок полей фиксированы — они определяют формат
сохранённых ``.npz``. Новые поля добавляйте только в конец.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .agents import make_agent
from .agents.base import Agent
from .envs import ENVIRONMENTS
from .envs.base import StepEnvironment
from .meta import make_policy
from .meta.base import Policy

RECORD_FIELDS: Tuple[str, ...] = (
    "actions",
    "kind",
    "effective",
    "reward",
    "points_gained",
    "points",
    "n_arms_before",
    "n_arms_after",
    "epoch_before",
    "epoch_after",
    "multiplier_before",
    "multiplier_after",
    "info_before",
    "best_mean",
    "best_arm",
    "chosen_mean",
    "arm_regret",
    "useful_gained",
    "info_gained",
)

RECORD_DTYPES: Dict[str, Any] = {
    "actions": np.int32,
    "kind": np.int8,
    "effective": np.int8,
    "reward": np.float64,
    "points_gained": np.float64,
    "points": np.float64,
    "n_arms_before": np.int32,
    "n_arms_after": np.int32,
    "epoch_before": np.int16,
    "epoch_after": np.int16,
    "multiplier_before": np.float64,
    "multiplier_after": np.float64,
    "info_before": np.int16,
    "best_mean": np.float64,
    "best_arm": np.int32,
    "chosen_mean": np.float64,
    "arm_regret": np.float64,
    "useful_gained": np.int16,
    "info_gained": np.int16,
}

RECORD_DESCRIPTIONS: Dict[str, str] = {
    "actions": "стабильный id действия (руки 0..K-1, мета 10000+вид)",
    "kind": "0 — рука, 1..4 — мета-действие",
    "effective": "1, если действие изменило состояние экономики",
    "reward": "сигнал обучения (0/1 для рук, 0 для мета)",
    "points_gained": "прибавка очков на шаге (с множителем)",
    "points": "кумулятивные очки после шага",
    "n_arms_before": "рук до действия",
    "n_arms_after": "рук после действия",
    "epoch_before": "эпоха до действия",
    "epoch_after": "эпоха после действия",
    "multiplier_before": "множитель до действия",
    "multiplier_after": "множитель после действия",
    "info_before": "неконвертированные данные до действия",
    "best_mean": "истинное среднее лучшей руки (для метрик)",
    "best_arm": "id лучшей руки (для метрик)",
    "chosen_mean": "истинное среднее выбранной руки (nan для мета)",
    "arm_regret": "best_mean - chosen_mean (nan для мета)",
    "useful_gained": "полезные единицы информации на шаге (H6)",
    "info_gained": "все единицы информации на шаге, включая мусор (H6)",
}


@dataclass(frozen=True)
class RunConfig:
    """Полное описание одного прогона: среда + агент + политика + горизонт."""

    name: str
    env: Mapping[str, Any]
    agent: Mapping[str, Any]
    policy: Optional[Mapping[str, Any]] = None
    n_steps: int = 3000
    expose_oracle: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if int(self.n_steps) <= 0:
            raise ValueError("n_steps должен быть положительным")
        if "kind" not in self.env:
            raise ValueError("env должен содержать ключ 'kind'")
        if "kind" not in self.agent:
            raise ValueError("agent должен содержать ключ 'kind'")
        if self.policy is not None and "kind" not in self.policy:
            raise ValueError("policy должен содержать ключ 'kind'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env": dict(self.env),
            "agent": dict(self.agent),
            "policy": None if self.policy is None else dict(self.policy),
            "n_steps": int(self.n_steps),
            "expose_oracle": bool(self.expose_oracle),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        return cls(
            name=str(data["name"]),
            env=dict(data["env"]),
            agent=dict(data["agent"]),
            policy=None if data.get("policy") is None else dict(data["policy"]),
            n_steps=int(data.get("n_steps", 3000)),
            expose_oracle=bool(data.get("expose_oracle", False)),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class SeedPair:
    """Пара сидов: среда и агент. Одинаковая для всех конфигов (paired design)."""

    env_seed: int
    agent_seed: int


def create_env(spec: Mapping[str, Any], seed: int) -> StepEnvironment:
    kind = str(spec["kind"]).strip().lower()
    if kind not in ENVIRONMENTS:
        raise KeyError(
            f"Неизвестная среда {kind!r}. Доступны: {', '.join(sorted(ENVIRONMENTS))}"
        )
    params = dict(spec.get("params") or {})
    return ENVIRONMENTS[kind](seed=seed, **params)


def create_agent(spec: Mapping[str, Any], n_actions: int, seed: int) -> Agent:
    params = dict(spec.get("params") or {})
    return make_agent(str(spec["kind"]), n_actions=n_actions, seed=seed, **params)


def create_policy(spec: Optional[Mapping[str, Any]]) -> Policy:
    if spec is None:
        return make_policy(None)
    params = dict(spec.get("params") or {})
    return make_policy(str(spec["kind"]), **params)


def make_seed_pairs(base_seed: int, n_runs: int) -> List[SeedPair]:
    """Порядок сидов фиксирован и одинаков для всех конфигов.

    Это даёт paired design: различия между конфигами считаются по одним и тем
    же средам, поэтому дисперсия разности ниже, чем у независимых выборок.
    """
    if int(n_runs) <= 0:
        raise ValueError("n_runs должен быть положительным")
    rng = np.random.default_rng(base_seed)
    draws = rng.integers(0, 2**31 - 1, size=(int(n_runs), 2))
    return [SeedPair(int(a), int(b)) for a, b in draws]


def run_episode(
    config: RunConfig,
    env_seed: int,
    agent_seed: int,
    n_steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Один эпизод: цикл «политика → агент → среда → запись».

    Записывается всё, что нужно для метрик гипотезы: сожаление по рукам,
    эффективность относительно оракула, время до эпох, состояние экономики.
    """
    steps = int(config.n_steps if n_steps is None else n_steps)
    env = create_env(config.env, int(env_seed))
    env.expose_oracle = bool(config.expose_oracle)
    policy = create_policy(config.policy)
    agent = create_agent(config.agent, n_actions=env.n_arms, seed=int(agent_seed))

    arrays = {name: np.zeros(steps, dtype=RECORD_DTYPES[name]) for name in RECORD_FIELDS}

    for t in range(steps):
        action = policy(env.observation())
        delegated = action is None
        if delegated:
            agent.grow_to(env.n_arms)
            action = int(agent.select_action())
            if not 0 <= action < env.n_arms:
                raise RuntimeError(
                    f"Агент {type(agent).__name__} вернул недопустимую руку {action} "
                    f"(n_arms={env.n_arms})"
                )
        info = env.step(action)
        if delegated and info.is_arm_pull:
            agent.update(info.action, info.reward)

        arrays["actions"][t] = info.action
        arrays["kind"][t] = info.kind_code
        arrays["effective"][t] = int(info.was_effective)
        arrays["reward"][t] = info.reward
        arrays["points_gained"][t] = info.points_gained
        arrays["points"][t] = info.after.points
        arrays["n_arms_before"][t] = info.before.n_arms
        arrays["n_arms_after"][t] = info.after.n_arms
        arrays["epoch_before"][t] = info.before.epoch
        arrays["epoch_after"][t] = info.after.epoch
        arrays["multiplier_before"][t] = info.before.multiplier
        arrays["multiplier_after"][t] = info.after.multiplier
        arrays["info_before"][t] = info.before.info_units
        arrays["best_mean"][t] = info.best_mean
        arrays["best_arm"][t] = info.best_arm
        arrays["chosen_mean"][t] = info.chosen_mean
        arrays["arm_regret"][t] = info.arm_regret
        arrays["useful_gained"][t] = info.useful_gained
        arrays["info_gained"][t] = info.info_gained

    return arrays


def _run_task(task: Tuple[RunConfig, SeedPair]) -> Dict[str, np.ndarray]:
    """Точка входа для воркеров (обязательно модульного уровня — pickle)."""
    config, pair = task
    return run_episode(config, pair.env_seed, pair.agent_seed)


@dataclass
class RunRecords:
    """Результаты прогона: ``arrays[field]`` имеет форму ``(n_runs, n_steps)``."""

    config: Dict[str, Any]
    seeds: List[SeedPair]
    arrays: Dict[str, np.ndarray]

    def __getattr__(self, name: str) -> np.ndarray:
        if name in RECORD_FIELDS:
            return object.__getattribute__(self, "arrays")[name]
        raise AttributeError(f"RunRecords не содержит поля {name!r}")

    @property
    def n_runs(self) -> int:
        return len(self.seeds)

    @property
    def n_steps(self) -> int:
        return int(self.arrays["points"].shape[1])

    @property
    def name(self) -> str:
        return str(self.config["name"])

    def run(self, index: int) -> "RunRecords":
        """Подвыборка из одного прогона (удобно для отладки/графиков)."""
        return RunRecords(
            config=self.config,
            seeds=[self.seeds[index]],
            arrays={k: v[index : index + 1] for k, v in self.arrays.items()},
        )

    def save(self, directory: Path | str) -> Path:
        """Сохранить прогон в ``<directory>/<name>.npz`` + манифест."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.npz"
        np.savez_compressed(path, **self.arrays)
        manifest = {
            "config": self.config,
            "n_runs": self.n_runs,
            "n_steps": self.n_steps,
            "fields": list(RECORD_FIELDS),
            "seeds": [[s.env_seed, s.agent_seed] for s in self.seeds],
        }
        (directory / f"{self.name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: Path | str, name: str) -> "RunRecords":
        directory = Path(directory)
        manifest = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        with np.load(directory / f"{name}.npz") as data:
            arrays = {field: data[field] for field in manifest["fields"]}
        return cls(
            config=manifest["config"],
            seeds=[SeedPair(int(a), int(b)) for a, b in manifest["seeds"]],
            arrays=arrays,
        )

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"RunRecords(name={self.name!r}, n_runs={self.n_runs}, n_steps={self.n_steps})"


def run_config(
    config: RunConfig,
    n_runs: int = 100,
    base_seed: int = 42,
    jobs: int = 1,
    n_steps: Optional[int] = None,
    progress: bool = True,
) -> RunRecords:
    """Прогнать конфиг ``n_runs`` раз (пары сидов общие для всех конфигов)."""
    steps = int(config.n_steps if n_steps is None else n_steps)
    if steps != config.n_steps:
        config = replace(config, n_steps=steps)
    pairs = make_seed_pairs(base_seed, n_runs)
    jobs = max(1, int(jobs))

    if jobs == 1:
        results = []
        for i, pair in enumerate(pairs, start=1):
            results.append(run_episode(config, pair.env_seed, pair.agent_seed))
            if progress and (i == len(pairs) or i % max(1, len(pairs) // 10) == 0):
                print(f"  {config.name}: {i}/{len(pairs)}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = list(
                executor.map(_run_task, [(config, pair) for pair in pairs], chunksize=1)
            )
        if progress:
            print(f"  {config.name}: {len(pairs)}/{len(pairs)} (jobs={jobs})", flush=True)

    arrays = {
        field: np.stack([result[field] for result in results]) for field in RECORD_FIELDS
    }
    return RunRecords(config=config.to_dict(), seeds=pairs, arrays=arrays)
