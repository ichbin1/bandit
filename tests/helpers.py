"""Помощники для тестов: синтетические ``RunRecords`` без прогонов."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from bandit.runner import RECORD_DTYPES, RECORD_FIELDS, RunRecords, SeedPair


def make_records(**fields: Any) -> RunRecords:
    """Собрать ``RunRecords`` из переданных полей (остальные — нули).

    Каждое поле — массив формы ``(n_runs, n_steps)`` или ``(n_steps,)``
    (тогда считается одним прогоном). Все поля приводятся к ``RECORD_DTYPES``.
    """
    prepared: Dict[str, np.ndarray] = {}
    n_runs = n_steps = None
    for name, value in fields.items():
        if name not in RECORD_DTYPES:
            raise KeyError(f"Неизвестное поле записи: {name}")
        array = np.asarray(value)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Поле {name} должно быть 1D или 2D, получено {array.shape}")
        if n_runs is None:
            n_runs, n_steps = array.shape
        elif array.shape != (n_runs, n_steps):
            raise ValueError(
                f"Поле {name} имеет форму {array.shape}, ожидалась {(n_runs, n_steps)}"
            )
        prepared[name] = array.astype(RECORD_DTYPES[name])

    if n_runs is None:
        n_runs, n_steps = 1, 1

    arrays = {
        name: np.zeros((n_runs, n_steps), dtype=RECORD_DTYPES[name]) for name in RECORD_FIELDS
    }
    arrays.update(prepared)
    return RunRecords(
        config={"name": "synthetic", "env": {"kind": "layered"}, "agent": {"kind": "random"}},
        seeds=[SeedPair(0, 0) for _ in range(n_runs)],
        arrays=arrays,
    )
