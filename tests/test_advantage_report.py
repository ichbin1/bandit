"""Стадии проверки метацикла: имена, профиль Δ(t), CSV и инварианты стадий."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bandit import metrics as m
from bandit.configs import get_config
from bandit.runner import run_config

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> Any:
    """Импортировать ``scripts/<name>.py`` как модуль (без установки пакета).

    Модуль обязательно регистрируется в ``sys.modules``: иначе ``@dataclass``
    внутри скрипта не находит свой модуль при разборе аннотаций.
    """
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"adv_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = load_script("advantage_report")
sweep = load_script("advantage_sweep")


def make_row(name: str, records: Any, stage: str = "s2") -> Any:
    """``ConfigRow`` из реальных записей: так тест ловит рассинхрон полей."""
    axis, level, agent, policy = report.parse_config_name(name)
    return report.ConfigRow(
        directory=ROOT / "results" / "_test",
        stage=stage,
        name=name,
        axis=axis,
        level=level,
        agent=agent,
        policy=policy,
        n_steps=int(records.n_steps),
        points=np.asarray(records.points, dtype=float),
        values=m.summarize(records),
    )


def test_parse_config_name_splits_axis_level_and_policy():
    assert report.parse_config_name("pyramid-adv-cap-cap5-ucb1-full_cycle") == (
        "cap",
        "cap5",
        "ucb1",
        "full_cycle",
    )
    # Обычные конфиги не стадий: ось — всё до агента, уровень пуст.
    assert report.parse_config_name("pyramid-ucb1-none") == ("pyramid", "", "ucb1", "none")


def test_first_stable_positive_requires_sign_to_hold():
    assert report.first_stable_positive([-1.0, -2.0, 0.5, 0.1]) == 3.0
    assert report.first_stable_positive([1.0, 2.0]) == 1.0
    assert report.first_stable_positive([1.0, -1.0, 1.0]) == 3.0
    assert np.isnan(report.first_stable_positive([1.0, -1.0, -1.0]))


def test_build_nodes_pairs_by_environment_and_seeds():
    steps = 200
    full = run_config(
        get_config("pyramid-ucb1-full_cycle"), n_runs=2, n_steps=steps, progress=False
    )
    plain = run_config(
        get_config("pyramid-ucb1-none"), n_runs=2, n_steps=steps, progress=False
    )
    rows = [
        make_row("pyramid-adv-abl-base-ucb1-none", plain, stage="s9"),
        make_row("pyramid-adv-abl-base-ucb1-full_cycle", full, stage="s9"),
    ]
    nodes = report.build_nodes(rows)
    assert len(nodes) == 2
    node = report.find_node(nodes, "s9", "abl", "base", "full_cycle", agent="ucb1")
    assert node is not None and node.reference == "pyramid-adv-abl-base-ucb1-none"

    expected = float(
        np.mean(full.points[:, -1] - plain.points[:, -1])
    )
    assert node.delta is not None
    assert node.delta["diff"] == pytest.approx(expected, abs=1e-6)
    assert node.diffs is not None and node.diffs.shape == (2, steps)
    assert node.curve is not None and node.curve.shape == (steps,)

    mean, sem = report.horizon_delta(node, steps)
    assert mean == pytest.approx(expected, abs=1e-6)
    assert np.isfinite(sem)
    assert np.isnan(report.horizon_delta(node, steps + 1)[0])

    # Стадии s1/s6 в этом наборе нет: вердикты честно «н/д».
    assert {item.status for item in report.compute_verdicts(nodes)} == {report.STATUS_NA}


def test_write_csv_keeps_horizon_columns(tmp_path):
    full = run_config(
        get_config("pyramid-ucb1-full_cycle"), n_runs=2, n_steps=120, progress=False
    )
    plain = run_config(
        get_config("pyramid-ucb1-none"), n_runs=2, n_steps=120, progress=False
    )
    rows = [
        make_row("pyramid-adv-abl-base-ucb1-none", plain),
        make_row("pyramid-adv-abl-base-ucb1-full_cycle", full),
    ]
    nodes = report.build_nodes(rows)
    path = report.write_csv(tmp_path / "advantage.csv", nodes, horizons=(100, 200))
    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    header = list(parsed[0])
    assert "delta@100" in header and "delta@200" in header
    assert "multiplier_from_arms" in header
    assert len(parsed) == 2
    # За горизонтом T значения nan, внутри — числа.
    cells = {row["config"]: row for row in parsed}
    full_cells = cells["pyramid-adv-abl-base-ucb1-full_cycle"]
    assert full_cells["delta@200"] == "nan"
    assert float(full_cells["delta@100"]) != 0.0
    # Базлайн сравнивается сам с собой: Δ ровно нулевое.
    assert float(cells["pyramid-adv-abl-base-ucb1-none"]["delta@100"]) == 0.0


def test_sweep_stages_are_unique_and_oracle_gets_oracle_policy():
    names = [config.name for stage in sweep.STAGES.values() for config in stage.configs]
    assert len(names) == len(set(names))
    assert set(sweep.STAGES) == {"s0", "s1", "s2", "s6"}

    oracle = [
        config
        for stage in sweep.STAGES.values()
        for config in stage.configs
        if config.name.endswith("-oracle-full_cycle")
    ]
    assert oracle, "в стадиях должен быть оракул: без него нет КПД"
    for config in oracle:
        assert config.policy["kind"] == "oracle_full_cycle"
        assert config.expose_oracle

    # В каждой группе (ось, уровень, агент) есть контраст none → full_cycle:
    # иначе Δ считать не с чем. Группы оракула не в счёт.
    for stage in sweep.STAGES.values():
        groups: dict = {}
        for config in stage.configs:
            axis, level, agent, policy = report.parse_config_name(config.name)
            groups.setdefault((axis, level, agent), set()).add(policy)
        for key, policies in groups.items():
            if key[2] == sweep.ORACLE_AGENT:
                continue
            assert {"none", "full_cycle"} <= policies, key
