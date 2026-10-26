"""Проверка паритета: код версии 0.1 (файлы в корне) против пакета ``bandit``.

Сравниваются не только итоговые очки, но и число рук/эпох: любой сдвиг в
порядке обращений к RNG проявится сразу. Именно эти числа закреплены в
``tests/test_golden.py``.

Запуск::

    python scripts/check_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomAgent, UCB1  # noqa: E402  legacy v0.1
from environment import LayeredPyramidBandit  # noqa: E402  legacy v0.1
from experiment import run_pyramid_experiment  # noqa: E402  legacy v0.1

from bandit.configs import get_config  # noqa: E402
from bandit.runner import run_config  # noqa: E402

SEED = 1000
STEPS = 3000
RUNS = 1


def legacy_env(seed: int) -> LayeredPyramidBandit:
    return LayeredPyramidBandit(
        n_arms_init=5,
        n_levels=3,
        info_per_multiplier=5,
        mult_per_arm_base=3,
        arm_cost_increase=1,
        mult_bonus=0.5,
        arm_bonus=0.5,
        arm_bonus_increase=0.1,
        max_arms=50,
        tech_multiplier_bonus=1.0,
        tech_arm_mean_boost=0.15,
        amplifier_mult_bonus=0.3,
        amplifier_cost_growth=1.5,
        seed=seed,
    )


def legacy_no_meta(env) -> None:
    return None


def legacy_full_cycle(env):
    if env.can_craft_gate:
        return env.n_arms + 2
    if env.can_craft_amplifier:
        return env.n_arms + 3
    if env.info_units >= env.info_per_multiplier:
        return env.n_arms
    if env.n_multipliers >= env.current_arm_cost and env.n_arms < env.max_arms:
        return env.n_arms + 1
    return None


AGENTS = {
    "random": lambda seed: RandomAgent(5, seed=seed),
    "ucb1": lambda seed: UCB1(5, c=2.0, seed=seed),
}
POLICIES = {
    "none": legacy_no_meta,
    "full_cycle": legacy_full_cycle,
}


def main() -> int:
    ok = True
    for agent_name, agent_factory in AGENTS.items():
        for policy_name, policy in POLICIES.items():
            name = f"pyramid-{agent_name}-{policy_name}"
            legacy = run_pyramid_experiment(
                env_factory=legacy_env,
                agent_factory=agent_factory,
                meta_policy=policy,
                n_steps=STEPS,
                n_runs=RUNS,
                seed=SEED,
            )
            new = run_config(
                get_config(name), n_runs=RUNS, base_seed=SEED, n_steps=STEPS, progress=False
            )
            same_points = np.array_equal(legacy.total_points[:, -1], new.points[:, -1])
            same_arms = np.array_equal(legacy.n_arms[:, -1], new.n_arms_after[:, -1])
            same_epochs = np.array_equal(legacy.epochs[:, -1], new.epoch_after[:, -1])
            ok = ok and same_points and same_arms and same_epochs
            print(
                f"{name:<32} points={float(new.points[0, -1]):>18.8f} "
                f"epoch={int(new.epoch_after[0, -1])} arms={int(new.n_arms_after[0, -1])} "
                f"| совпадает: очки={same_points}, руки={same_arms}, эпохи={same_epochs}"
            )
    print("\nПАРИТЕТ: " + ("OK" if ok else "РАСХОЖДЕНИЕ"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
