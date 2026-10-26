"""Стадии проверки преимущества метацикла: S0, S1, S2, S6.

Скрипт не считает эпизоды сам: он описывает конфиги, регистрирует их в
``bandit.configs.REGISTRY`` и передаёт в ``bandit.cli run``. Поэтому формат
результатов (``*.npz``, ``runs.csv``, ``metrics.csv``, ``manifest.json``)
совпадает с обычными прогонами, а разбирает их
``scripts/advantage_report.py``.

Имя конфига: ``pyramid-adv-<ось>-<уровень>-<агент>-<политика>``. Уровни
пишутся без дефисов (``inc005``, ``cap5``, ``lv1``), поэтому имя однозначно
разбирается как ``<среда>-<агент>-<политика>``.

Стадии
------
``s0`` — протокол: два конфига и мало прогонов, дешёвая проверка конвейера.
``s1`` — экономика: где именно создаётся выигрыш. Оси:

       * ``engine`` — расписание бонуса за руку (``arm_bonus_increase``);
       * ``cap`` — потолок числа рук (``max_arms``);
       * ``lv`` — число эпох (``n_levels``).

``s2`` — абляция ветвей на дефолтных параметрах: ``none`` / ``mult_only`` /
       ``gate_only`` / ``amp_only`` / ``amp`` / ``full_cycle`` + оракулы.
``s6`` — длинный горизонт (T = 10 000): ось ``long`` даёт профиль Δ(t) и
       момент t*, ось ``agent`` — устойчивость выигрыша к качеству агента.

Запуск::

    python scripts/advantage_sweep.py --list
    python scripts/advantage_sweep.py --stage s2 --runs 5 --out results/_adv_dry
    python scripts/advantage_sweep.py --all --runs 30
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bandit import configs as cfgs  # noqa: E402
from bandit.cli import main as cli_main  # noqa: E402
from bandit.runner import RunConfig  # noqa: E402

ADV_PREFIX = "pyramid-adv"
"""Префикс имён: отделяет стадии от обычных конфигов ``pyramid-*``."""

ORACLE_AGENT = "oracle"
"""Имя «агента» в конфигах оракула (как в ``pyramid-overload-*``)."""

#: Горизонты стадий. ``s6`` длинный: только он даёт профиль Δ(t) без обрезки.
STAGE_STEPS: Dict[str, int] = {
    "s0": cfgs.PYRAMID_STEPS,
    "s1": cfgs.PYRAMID_STEPS,
    "s2": cfgs.PYRAMID_STEPS,
    "s6": 10_000,
}

#: Число прогонов по умолчанию: длинная стадия дороже, ей хватает 20 пар сидов.
STAGE_RUNS: Dict[str, int] = {"s0": 5, "s1": 30, "s2": 30, "s6": 20}


@dataclass(frozen=True)
class Stage:
    """Одна стадия: набор конфигов и параметры прогона."""

    key: str
    title: str
    n_steps: int
    n_runs: int
    configs: Tuple[RunConfig, ...]

    @property
    def out_dir(self) -> Path:
        return ROOT / "results" / f"adv-{self.key}"


def _adv_config(
    axis: str,
    level: str,
    agent: str,
    policy: str,
    *,
    n_steps: int,
    env_params: Optional[Dict[str, object]] = None,
    note: str = "",
) -> RunConfig:
    """Конфиг стадии: ``pyramid-adv-<ось>-<уровень>-<агент>-<политика>``.

    ``agent == "oracle"`` — это конфиг верхней границы: он получает политику
    ``oracle_<policy>`` и флаг ``expose_oracle``, а агент у него фиктивный
    (``random``), как в ``pyramid-overload-*``.
    """
    params = dict(cfgs.LAYERED_PARAMS)
    if env_params:
        params.update(env_params)
    is_oracle = agent == ORACLE_AGENT
    policy_key = f"oracle_{policy}" if is_oracle else policy
    if policy_key not in cfgs.POLICIES:
        raise KeyError(f"Неизвестная мета-политика {policy_key!r}")
    agent_key = "random" if is_oracle else agent
    if agent_key not in cfgs.AGENTS:
        raise KeyError(f"Неизвестный агент {agent_key!r}")
    return RunConfig(
        name=f"{ADV_PREFIX}-{axis}-{level}-{agent}-{policy}",
        env={"kind": "layered", "params": params},
        agent=dict(cfgs.AGENTS[agent_key]),
        policy=dict(cfgs.POLICIES[policy_key]),
        n_steps=int(n_steps),
        expose_oracle=is_oracle,
        description=(
            f"{ADV_PREFIX}: ось {axis}={level}, агент={agent}, политика={policy}"
            + (f"; {note}" if note else "")
        ),
    )


def _contrast(
    axis: str,
    levels: Sequence[Tuple[str, Dict[str, object]]],
    *,
    n_steps: int,
    policies: Sequence[str] = ("none", "full_cycle"),
    agent: str = "ucb1",
    note: str = "",
) -> List[RunConfig]:
    """Уровни одной оси × контрастные политики при одном агенте."""
    return [
        _adv_config(
            axis,
            level,
            agent,
            policy,
            n_steps=n_steps,
            env_params=params,
            note=note,
        )
        for level, params in levels
        for policy in policies
    ]


#: Ось «расписание бонуса за руку»: квадратичный рост проверяется по уровням.
ENGINE_LEVELS: Sequence[Tuple[str, Dict[str, object]]] = (
    ("inc0", {"arm_bonus_increase": 0.0}),
    ("inc005", {"arm_bonus_increase": 0.05}),
    ("inc01", {"arm_bonus_increase": 0.1}),
    ("inc02", {"arm_bonus_increase": 0.2}),
)

#: Ось «потолок рук»: ``cap5`` — покупки запрещены, выигрыш может идти только
#: от ворот; ``cap50`` — дефолт (покупка рук разрешена).
CAP_LEVELS: Sequence[Tuple[str, Dict[str, object]]] = (
    ("cap5", {"max_arms": 5}),
    ("cap10", {"max_arms": 10}),
    ("cap50", {"max_arms": 50}),
)

#: Ось «число эпох»: ``lv1`` отключает ворота, оставляя конверсии и покупки.
LEVEL_LEVELS: Sequence[Tuple[str, Dict[str, object]]] = (
    ("lv1", {"n_levels": 1}),
    ("lv2", {"n_levels": 2}),
    ("lv3", {"n_levels": 3}),
)

#: Политики абляции ветвей (H2): по одной ветви и полный цикл.
ABLATION_POLICIES: Sequence[str] = (
    "none",
    "mult_only",
    "gate_only",
    "amp_only",
    "amp",
    "full_cycle",
)

#: Политики длинного горизонта: ``amp_only`` нужен только для дешёвой абляции.
LONG_POLICIES: Sequence[str] = ("none", "mult_only", "gate_only", "full_cycle")


def _stage_s0() -> Stage:
    n_steps = STAGE_STEPS["s0"]
    configs = _contrast(
        "smoke",
        (("base", {}),),
        n_steps=n_steps,
        note="проверка конвейера",
    )
    return Stage(
        key="s0",
        title="S0: протокол (дешёвый дым на дефолтной среде)",
        n_steps=n_steps,
        n_runs=STAGE_RUNS["s0"],
        configs=tuple(configs),
    )


def _stage_s1() -> Stage:
    n_steps = STAGE_STEPS["s1"]
    configs = (
        _contrast("engine", ENGINE_LEVELS, n_steps=n_steps, note="расписание бонуса за руку")
        + _contrast("cap", CAP_LEVELS, n_steps=n_steps, note="потолок числа рук")
        + _contrast("lv", LEVEL_LEVELS, n_steps=n_steps, note="число эпох")
    )
    return Stage(
        key="s1",
        title="S1: экономика метацикла (расписание, потолок рук, эпохи)",
        n_steps=n_steps,
        n_runs=STAGE_RUNS["s1"],
        configs=tuple(configs),
    )


def _stage_s2() -> Stage:
    n_steps = STAGE_STEPS["s2"]
    configs = _contrast(
        "abl",
        (("base", {}),),
        n_steps=n_steps,
        policies=ABLATION_POLICIES,
        note="абляция ветвей",
    )
    # Верхние границы: полный метацикл и только выбор рук.
    configs.append(_adv_config("abl", "base", ORACLE_AGENT, "full_cycle", n_steps=n_steps))
    configs.append(_adv_config("abl", "base", ORACLE_AGENT, "none", n_steps=n_steps))
    return Stage(
        key="s2",
        title="S2: абляция ветвей на дефолтных параметрах",
        n_steps=n_steps,
        n_runs=STAGE_RUNS["s2"],
        configs=tuple(configs),
    )


def _stage_s6() -> Stage:
    n_steps = STAGE_STEPS["s6"]
    configs = _contrast(
        "long",
        (("t10k", {}),),
        n_steps=n_steps,
        policies=LONG_POLICIES,
        note="длинный горизонт",
    )
    configs.append(_adv_config("long", "t10k", ORACLE_AGENT, "full_cycle", n_steps=n_steps))
    # Ось «устойчивость к агенту»: у каждого уровня свой контраст none/full_cycle,
    # поэтому Δ(агент) сравнивается внутри уровня и не смешивает агентов.
    for agent in ("ucb1", "greedy", "random"):
        configs.extend(
            _adv_config(
                "agent",
                agent,
                agent,
                policy,
                n_steps=n_steps,
                note="устойчивость к агенту",
            )
            for policy in ("none", "full_cycle")
        )
    return Stage(
        key="s6",
        title="S6: длинный горизонт (T = 10 000) и устойчивость к агенту",
        n_steps=n_steps,
        n_runs=STAGE_RUNS["s6"],
        configs=tuple(configs),
    )


STAGES: Dict[str, Stage] = {
    stage.key: stage
    for stage in (_stage_s0(), _stage_s1(), _stage_s2(), _stage_s6())
}


def register(configs: Sequence[RunConfig]) -> List[str]:
    """Добавить конфиги в ``bandit.configs.REGISTRY`` и вернуть имена.

    ``cli run --config NAME`` ищет имя именно в реестре, поэтому стадии
    регистрируются до вызова CLI. Повторная регистрация того же конфига
    допустима (удобно для тестов), а конфликт параметров — ошибка.
    """
    names: List[str] = []
    for config in configs:
        existing = cfgs.REGISTRY.get(config.name)
        if existing is not None and existing.to_dict() != config.to_dict():
            raise KeyError(
                f"Конфиг {config.name!r} уже зарегистрирован с другими параметрами"
            )
        cfgs.REGISTRY[config.name] = config
        names.append(config.name)
    return names


def run_stage(
    stage: Stage,
    *,
    runs: Optional[int] = None,
    seed: int = 42,
    jobs: int = 1,
    out: Optional[Path] = None,
    steps: Optional[int] = None,
) -> int:
    """Запустить стадию через ``bandit.cli run`` (тот же формат результатов)."""
    names = register(stage.configs)
    n_runs = int(stage.n_runs if runs is None else runs)
    out_dir = Path(out) if out is not None else stage.out_dir
    horizon = int(stage.n_steps if steps is None else steps)
    print(
        f"\n[{stage.key}] {stage.title}\n"
        f"  конфигов: {len(names)}, прогонов: {n_runs}, горизонт: {horizon} шагов "
        f"(эпизодов: {len(names) * n_runs})",
        flush=True,
    )
    argv = ["run"]
    for name in names:
        argv.extend(["--config", name])
    argv.extend(
        [
            "--runs",
            str(n_runs),
            "--seed",
            str(int(seed)),
            "--jobs",
            str(int(jobs)),
            "--out",
            str(out_dir),
            "--no-plots",
            "--quiet",
        ]
    )
    if steps is not None:
        argv.extend(["--steps", str(int(steps))])
    return cli_main(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Стадии проверки преимущества метацикла (S0/S1/S2/S6)"
    )
    parser.add_argument("--stage", action="append", choices=sorted(STAGES), help="стадия")
    parser.add_argument("--all", action="store_true", help="все стадии по порядку")
    parser.add_argument("--runs", type=int, default=None, help="прогонов на конфиг")
    parser.add_argument("--seed", type=int, default=42, help="базовый сид (по умолчанию 42)")
    parser.add_argument("--jobs", type=int, default=1, help="процессов (Windows: spawn)")
    parser.add_argument("--steps", type=int, default=None, help="переопределить горизонт")
    parser.add_argument("--out", type=Path, default=None, help="каталог (для одной стадии)")
    parser.add_argument("--list", action="store_true", help="показать стадии и конфиги")
    args = parser.parse_args(argv)

    keys = sorted(STAGES) if args.all else list(args.stage or [])
    if args.list:
        for key in sorted(STAGES):
            stage = STAGES[key]
            print(
                f"\n{key}: {stage.title}\n"
                f"  горизонт {stage.n_steps}, прогонов {stage.n_runs}, "
                f"конфигов {len(stage.configs)}"
            )
            for config in stage.configs:
                print(f"    {config.name:<48} {config.description}")
        return 0
    if not keys:
        parser.error("укажите --stage или --all")
    if args.out is not None and len(keys) > 1:
        parser.error("--out допустим только для одной стадии")

    for key in keys:
        code = run_stage(
            STAGES[key],
            runs=args.runs,
            seed=args.seed,
            jobs=args.jobs,
            out=args.out,
            steps=args.steps,
        )
        if code != 0:
            return code
    print("\nГотово. Отчёт: python scripts/advantage_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


