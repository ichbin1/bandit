"""Golden-тесты: точные значения очков, совпадающие с версией 0.1.

Числа получены прогоном legacy-кода (``environment.py`` + ``experiment.py``,
``n_runs=1``, ``seed=1000``, 3000 шагов) и проверены на побитовое равенство с
новым пакетом. Любое изменение порядка обращений к RNG их сломает — это и есть
защита воспроизводимости.

Пересчитать при осознанном изменении модели среды: ``python scripts/check_parity.py``
(сравнивает код версии 0.1 из корня репозитория с пакетом).
"""
from __future__ import annotations

import pytest

from bandit.configs import get_config
from bandit.runner import run_config

SEED = 1000
STEPS = 3000

GOLDEN = {
    "pyramid-random-none": {
        "final": 1548.0,
        "checks": {99: 54.0, 499: 259.0, 999: 524.0, 1999: 1054.0},
        "epoch": 0,
        "arms": 5,
    },
    "pyramid-ucb1-none": {
        "final": 2824.0,
        "checks": {99: 73.0, 499: 439.0, 999: 910.0, 1999: 1866.0},
        "epoch": 0,
        "arms": 5,
    },
    "pyramid-random-full_cycle": {
        "final": 72278.18939731,
        "checks": {99: 205.848077, 499: 3028.689964, 999: 10109.52973, 1999: 34125.126396},
        "action_sum": 5354702,
        "epoch": 2,
        "arms": 40,
    },
    "pyramid-ucb1-full_cycle": {
        "final": 74231.49402523,
        "checks": {99: 180.007706, 499: 2888.412692, 999: 9887.328417, 1999: 34695.001912},
        "action_sum": 5321185,
        "epoch": 2,
        "arms": 40,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_points_match_version_0_1(name):
    expected = GOLDEN[name]
    config = get_config(name)
    records = run_config(config, n_runs=1, base_seed=SEED, n_steps=STEPS, progress=False)

    assert records.points[0, -1] == pytest.approx(expected["final"], rel=1e-9)
    for step, value in expected["checks"].items():
        # Проверки округлены до 6 знаков — отсюда абсолютный допуск.
        assert records.points[0, step] == pytest.approx(value, rel=1e-9, abs=1e-3), (
            f"шаг {step + 1}"
        )

    assert int(records.epoch_after[0, -1]) == expected["epoch"]
    assert int(records.n_arms_after[0, -1]) == expected["arms"]
    if "action_sum" in expected:
        assert int(records.actions.sum()) == expected["action_sum"]


@pytest.mark.slow
def test_meta_cycle_beats_no_meta_in_golden_run():
    """H1 на одном сиде: полный цикл даёт на порядок больше очков (для UCB1)."""
    none = run_config(
        get_config("pyramid-ucb1-none"), n_runs=1, base_seed=SEED, n_steps=STEPS, progress=False
    )
    full = run_config(
        get_config("pyramid-ucb1-full_cycle"),
        n_runs=1,
        base_seed=SEED,
        n_steps=STEPS,
        progress=False,
    )
    assert full.points[0, -1] > none.points[0, -1]
    assert full.epoch_after[0, -1] > none.epoch_after[0, -1]
    assert full.n_arms_after[0, -1] > none.n_arms_after[0, -1]
