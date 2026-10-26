"""Быстрая демонстрация: один конфиг, несколько прогонов, дашборд.

Запуск::

    python -m bandit.demo
    python -m bandit.demo --config pyramid-ucb1-full_cycle --runs 10 --steps 3000

Для полноценных экспериментов используйте ``python -m bandit.cli``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from . import metrics as m, plots
from .cli import configure_output
from .configs import get_config
from .runner import run_config


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_output()
    parser = argparse.ArgumentParser(description="Демонстрационный прогон одного конфига.")
    parser.add_argument("--config", default="pyramid-ucb1-full_cycle", help="имя конфига")
    parser.add_argument("--runs", type=int, default=3, help="число прогонов")
    parser.add_argument("--steps", type=int, default=1500, help="горизонт")
    parser.add_argument("--seed", type=int, default=42, help="базовый сид")
    parser.add_argument(
        "--out", type=Path, default=Path("results") / "demo", help="каталог результатов"
    )
    args = parser.parse_args(argv)

    config = get_config(args.config)
    print(f"[demo] {config.name}: {args.runs} прогонов × {args.steps} шагов")
    records = run_config(config, n_runs=args.runs, base_seed=args.seed, n_steps=args.steps)

    points = m.aggregate(m.final_points(records))
    regret = m.aggregate(m.total_arm_regret(records))
    print(
        f"очки {points['mean']:.1f} ± {points['sem']:.1f} | "
        f"arm_regret {regret['mean']:.1f} | "
        f"эпоха {m.aggregate(m.final_epoch(records))['mean']:.2f} | "
        f"рук {m.aggregate(m.final_n_arms(records))['mean']:.1f}"
    )

    records.save(args.out)
    path = plots.save_dashboard(
        {config.name: records}, args.out / "dashboard.png", title=config.name
    )
    print(f"Сохранено: {args.out.resolve()} (дашборд: {path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
