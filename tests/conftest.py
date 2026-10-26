"""Общие фикстуры тестов.

Пакет не обязан быть установленным: добавляем корень репозитория в ``sys.path``.
Бэкенд matplotlib — ``Agg``, чтобы тесты графиков не открывали окна.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def layered_env():
    """Свежая ``LayeredPyramidBandit`` с фиксированным сидом (без оракула)."""
    from bandit.envs import LayeredPyramidBandit

    return LayeredPyramidBandit(seed=7)


@pytest.fixture(scope="session")
def short_records():
    """Короткий, но настоящий прогон: 2 сида × 200 шагов."""
    from bandit.configs import get_config
    from bandit.runner import run_config

    return run_config(
        get_config("pyramid-ucb1-full_cycle"), n_runs=2, n_steps=200, progress=False
    )
