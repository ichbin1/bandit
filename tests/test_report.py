"""Отчёт P5: разбор имён, вердикты H1–H6, markdown/JSON-артефакты."""
from __future__ import annotations

import csv
import json

import pytest

from bandit import report as rp

N_RUNS = 8


def _write_runs(path, values, metrics):
    """Записать runs.csv: ``values[name][metric] = список по прогонам``."""
    sizes = {len(table[metric]) for table in values.values() for metric in metrics}
    if len(sizes) != 1:
        raise ValueError(f"Разная длина рядов: {sorted(sizes)}")
    n_runs = sizes.pop()
    columns = list(rp.st.CSV_INDEX_COLUMNS) + list(metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, table in values.items():
            for run in range(n_runs):
                writer.writerow(
                    {
                        "config": name,
                        "run": run,
                        "env_seed": 100 + run,
                        "agent_seed": 200 + run,
                        **{metric: table[metric][run] for metric in metrics},
                    }
                )
    return path


def _scenario(tight_oracle: bool = True):
    """Синтетический прогон, в котором все пять гипотез подтверждаются."""
    base = [1000.0 + 10.0 * run for run in range(N_RUNS)]
    values = {
        "pyramid-ucb1-none": {
            "final_points": base,
            "final_epoch": [0.0] * N_RUNS,
            "steps_to_epoch_1": [float("nan")] * N_RUNS,
            "points_after_epoch_1": [float("nan")] * N_RUNS,
            "efficiency": [0.05 + 0.001 * run for run in range(N_RUNS)],
        },
        "pyramid-ucb1-full_cycle": {
            "final_points": [x + 900.0 + 3.0 * run for run, x in enumerate(base)],
            "final_epoch": [2.0] * N_RUNS,
            "steps_to_epoch_1": [120.0 + run for run in range(N_RUNS)],
            "points_after_epoch_1": [100.0 + 5.0 * run for run in range(N_RUNS)],
            "efficiency": [0.7 + 0.002 * run for run in range(N_RUNS)],
        },
        "pyramid-ucb1-gate_only": {
            "final_points": [x + 500.0 + 2.0 * run for run, x in enumerate(base)],
            "final_epoch": [1.0] * N_RUNS,
            "steps_to_epoch_1": [400.0 + run for run in range(N_RUNS)],
            "points_after_epoch_1": [50.0 + 2.0 * run for run in range(N_RUNS)],
            "efficiency": [0.4 + 0.001 * run for run in range(N_RUNS)],
        },
        "pyramid-ucb1-mult_only": {
            "final_points": [x + 100.0 + 1.0 * run for run, x in enumerate(base)],
            "final_epoch": [0.0] * N_RUNS,
            "steps_to_epoch_1": [float("nan")] * N_RUNS,
            "points_after_epoch_1": [float("nan")] * N_RUNS,
            "efficiency": [0.1 + 0.001 * run for run in range(N_RUNS)],
        },
        "pyramid-random-none": {
            "final_points": base,
            "final_epoch": [0.0] * N_RUNS,
            "steps_to_epoch_1": [float("nan")] * N_RUNS,
            "points_after_epoch_1": [float("nan")] * N_RUNS,
            "efficiency": [0.04 + 0.001 * run for run in range(N_RUNS)],
        },
        "pyramid-random-full_cycle": {
            "final_points": [x + 20.0 + 2.0 * run for run, x in enumerate(base)],
            "final_epoch": [1.0] * N_RUNS,
            "steps_to_epoch_1": [800.0 + run for run in range(N_RUNS)],
            "points_after_epoch_1": [10.0 + 1.0 * run for run in range(N_RUNS)],
            "efficiency": [0.2 + 0.001 * run for run in range(N_RUNS)],
        },
        "pyramid-eps_decay-full_cycle": {
            "final_points": [x + 400.0 + 4.0 * run for run, x in enumerate(base)],
            "final_epoch": [2.0] * N_RUNS,
            "steps_to_epoch_1": [110.0 + run for run in range(N_RUNS)],
            "points_after_epoch_1": [130.0 + 6.0 * run for run in range(N_RUNS)],
            "efficiency": [0.6 + 0.002 * run for run in range(N_RUNS)],
        },
        "pyramid-oracle-full_cycle": {
            "final_points": [x + 2000.0 for x in base],
            "final_epoch": [2.0] * N_RUNS,
            "steps_to_epoch_1": [90.0] * N_RUNS,
            "points_after_epoch_1": [300.0] * N_RUNS,
            "efficiency": [1.25 if not tight_oracle else 1.0] * N_RUNS,
        },
    }
    metrics = [
        "final_points",
        "final_epoch",
        "steps_to_epoch_1",
        "points_after_epoch_1",
        "efficiency",
    ]
    return values, metrics


def test_split_and_group_config_names():
    assert rp.split_config_name("pyramid-ucb1-full_cycle") == (
        "pyramid",
        "ucb1",
        "full_cycle",
    )
    assert rp.split_config_name("stationary-ucb1") is None
    assert rp.split_config_name("pyramid-overload-noisy-ucb1-full_cycle") == (
        "pyramid-overload-noisy",
        "ucb1",
        "full_cycle",
    )
    grouped = rp.group_configs(
        ["pyramid-ucb1-none", "pyramid-ucb1-learned", "pyramid-random-none", "odd"]
    )
    assert set(grouped["pyramid"]) == {"ucb1", "random"}
def test_build_report_marks_all_hypotheses_supported(tmp_path):
    values, metrics = _scenario()
    _write_runs(tmp_path / "runs.csv", values, metrics)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "synthetic", "bandit_version": "0.3.0", "suite": "unit"}),
        encoding="utf-8",
    )

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    statuses = {item["code"]: item["status"] for item in report.payload["hypotheses"]}
    expected = {code: rp.SUPPORTED for code in ("H1", "H2", "H3", "H4", "H5")}
    expected["H6"] = rp.NO_DATA  # режимов `pyramid-overload-*` в сценарии нет
    assert statuses == expected
    assert report.payload["reference"] == "pyramid-random-none"
    assert report.payload["n_runs"] == N_RUNS
    assert len(report.payload["comparisons"]) == len(values)
    assert {"среда", "агент", "политика"} <= set(report.payload["summary"][0])

    h1 = next(item for item in report.payload["hypotheses"] if item["code"] == "H1")
    assert {pair["agent"] for pair in h1["evidence"]["pairs"]} == {"ucb1", "random"}
    for pair in h1["evidence"]["pairs"]:
        assert pair["p_adjusted"] >= pair["p_value"] - 1e-12

    assert "## 3. Гипотезы" in report.markdown
    assert "**Статус:** подтверждена." in report.markdown
    assert "pyramid-ucb1-full_cycle" in report.markdown


def test_h5_is_refuted_when_efficiency_exceeds_one(tmp_path):
    values, metrics = _scenario(tight_oracle=False)
    _write_runs(tmp_path / "runs.csv", values, metrics)

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h5 = next(item for item in report.payload["hypotheses"] if item["code"] == "H5")
    assert h5["status"] == rp.REFUTED
    assert h5["evidence"]["over_one"]
    assert "КПД > 1" in h5["detail"]


def test_missing_metrics_lead_to_no_data(tmp_path):
    values = {
        "pyramid-ucb1-none": {"final_points": [1.0, 2.0, 3.0, 4.0]},
        "pyramid-ucb1-full_cycle": {"final_points": [10.0, 11.0, 12.0, 13.0]},
        "pyramid-random-none": {"final_points": [1.0, 2.0, 3.0, 4.0]},
        "pyramid-random-full_cycle": {"final_points": [2.0, 3.0, 4.0, 5.0]},
    }
    _write_runs(tmp_path / "runs.csv", values, ["final_points"])

    report = rp.build_report(tmp_path, n_boot=100, seed=5)
    statuses = {item["code"]: item["status"] for item in report.payload["hypotheses"]}
    assert statuses["H1"] == rp.SUPPORTED
    assert statuses["H2"] == rp.NO_DATA
    assert statuses["H4"] == rp.NO_DATA
    assert statuses["H5"] == rp.NO_DATA
    assert statuses["H6"] == rp.NO_DATA
    assert "пересоберите прогон" in next(
        item for item in report.payload["hypotheses"] if item["code"] == "H4"
    )["detail"]


def test_write_report_creates_serialisable_artifacts(tmp_path):
    values, metrics = _scenario()
    _write_runs(tmp_path / "runs.csv", values, metrics)
    report = rp.build_report(tmp_path, n_boot=100, seed=5)

    markdown_path, json_path = rp.write_report(tmp_path, report)
    assert markdown_path.name == "report.md" and markdown_path.stat().st_size > 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == tmp_path.name
    assert [item["code"] for item in payload["hypotheses"]] == [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ]
    assert payload["comparisons"][0]["metric"] == "final_points"
    assert payload["n_boot"] == 100
    assert payload["learned"] is None


def test_non_significant_negative_gives_not_supported(tmp_path):
    values = {
        "pyramid-ucb1-none": {
            "final_points": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "points_after_epoch_1": [float("nan")] * 6,
        },
        "pyramid-ucb1-full_cycle": {
            "final_points": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "points_after_epoch_1": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        },
        "pyramid-eps_decay-full_cycle": {
            "final_points": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "points_after_epoch_1": [110.0, 190.0, 310.0, 390.0, 510.0, 590.0],
        },
    }
    _write_runs(
        tmp_path / "runs.csv",
        values,
        ["final_points", "points_after_epoch_1"],
    )

    report = rp.build_report(tmp_path, n_boot=200, seed=5)
    h4 = next(item for item in report.payload["hypotheses"] if item["code"] == "H4")
    # Разности +10/−10 чередуются: средний эффект ноль, значимости нет.
    assert h4["status"] == rp.NOT_SUPPORTED
    assert h4["evidence"]["pairs"][0]["diff"] == pytest.approx(0.0)


def test_report_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        rp.build_report(tmp_path)

    values, metrics = _scenario()
    _write_runs(tmp_path / "runs.csv", values, metrics)
    with pytest.raises(KeyError):
        rp.build_report(tmp_path, reference="nope", n_boot=50)


def test_small_number_of_runs_marks_verdicts_as_preliminary(tmp_path):
    values, metrics = _scenario()
    _write_runs(tmp_path / "runs.csv", values, metrics)
    report = rp.build_report(tmp_path, n_boot=100, seed=5)
    assert report.payload["n_runs_warning"] is True
    assert "вердикты предварительные" in report.markdown


def test_learned_section_appears_when_config_present(tmp_path):
    values = {
        "pyramid-ucb1-none": {"final_points": [1.0, 2.0, 3.0, 4.0]},
        "pyramid-ucb1-full_cycle": {"final_points": [10.0, 11.0, 12.0, 13.0]},
        "pyramid-ucb1-learned": {"final_points": [9.0, 10.0, 11.0, 12.0]},
    }
    _write_runs(tmp_path / "runs.csv", values, ["final_points"])

    report = rp.build_report(tmp_path, n_boot=100, seed=5)
    learned = report.payload["learned"]
    assert learned is not None
    row = learned["rows"][0]
    assert row["agent"] == "ucb1"
    assert 0.0 < row["ratio_vs_full_cycle"] < 1.0
    assert row["comparisons"]["none"]["diff"] > 0
    assert "## 4. Обучаемая мета-политика (P4)" in report.markdown
