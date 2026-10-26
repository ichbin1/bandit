"""CLI: артефакты прогона, сводки, коды возврата."""
from __future__ import annotations

import csv
import json

import pytest

from bandit import cli


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_list_output(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "bandit v" in out
    assert "pyramid-ablation" in out
    assert "ucb1" in out

    assert cli.main(["list", "--verbose"]) == 0
    assert "pyramid-oracle-full_cycle" in capsys.readouterr().out


def test_run_creates_artifacts(tmp_path):
    code = cli.main(
        [
            "run",
            "--config",
            "pyramid-random-none",
            "--config",
            "pyramid-oracle-full_cycle",
            "--config",
            "pyramid-ucb1-full_cycle",
            "--runs",
            "2",
            "--steps",
            "60",
            "--seed",
            "7",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == 0

    for name in ("pyramid-random-none", "pyramid-oracle-full_cycle", "pyramid-ucb1-full_cycle"):
        assert (tmp_path / f"{name}.npz").exists()
        assert (tmp_path / f"{name}.json").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_runs"] == 2
    assert manifest["n_steps_override"] == 60
    assert len(manifest["configs"]) == 3
    assert manifest["suite"] is None
    assert manifest["bandit_version"]

    runs = _read_csv(tmp_path / "runs.csv")
    assert len(runs) == 6
    assert {"config", "run", "env_seed", "agent_seed", "final_points"} <= set(runs[0])

    metrics = _read_csv(tmp_path / "metrics.csv")
    assert any(row["metric"] == "efficiency" for row in metrics)
    efficiency = [row for row in metrics if row["metric"] == "efficiency"]
    assert {row["config"] for row in efficiency} >= {"pyramid-ucb1-full_cycle"}
    assert (tmp_path / "dashboard.png").stat().st_size > 0
    assert (tmp_path / "final_points.png").stat().st_size > 0


def test_run_without_oracle_has_no_efficiency(tmp_path):
    code = cli.main(
        [
            "run",
            "--config",
            "pyramid-random-none",
            "--runs",
            "1",
            "--steps",
            "30",
            "--out",
            str(tmp_path),
            "--no-plots",
            "--quiet",
        ]
    )
    assert code == 0
    assert not (tmp_path / "dashboard.png").exists()
    metrics = _read_csv(tmp_path / "metrics.csv")
    assert all(row["metric"] != "efficiency" for row in metrics)


def test_show_reads_saved_run(tmp_path, capsys):
    assert (
        cli.main(
            [
                "run",
                "--config",
                "pyramid-ucb1-none",
                "--runs",
                "1",
                "--steps",
                "40",
                "--out",
                str(tmp_path),
                "--no-plots",
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["show", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "pyramid-ucb1-none" in out
    assert "очки" in out


def test_show_reports_missing_directory(tmp_path, capsys):
    assert cli.main(["show", str(tmp_path / "nope")]) == 2
    assert "Нет такой папки" in capsys.readouterr().err

    assert cli.main(["show", str(tmp_path)]) == 2
    assert "нет сохранённых прогонов" in capsys.readouterr().err


def test_run_without_selection_returns_error(capsys):
    assert cli.main(["run", "--runs", "1"]) == 2
    assert "Ошибка" in capsys.readouterr().err


def test_unknown_config_returns_error(capsys):
    assert cli.main(["run", "--config", "nope", "--runs", "1"]) == 2
    assert "Неизвестный конфиг" in capsys.readouterr().err


def test_demo_command(tmp_path):
    from bandit import demo

    code = demo.main(
        [
            "--config",
            "pyramid-ucb1-mult_only",
            "--runs",
            "1",
            "--steps",
            "40",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "dashboard.png").exists()
    assert (tmp_path / "pyramid-ucb1-mult_only.npz").exists()


def test_stats_command_writes_results_csv(tmp_path, capsys):
    assert (
        cli.main(
            [
                "run",
                "--config",
                "pyramid-ucb1-none",
                "--config",
                "pyramid-ucb1-full_cycle",
                "--runs",
                "8",
                "--steps",
                "100",
                "--out",
                str(tmp_path),
                "--no-plots",
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()

    code = cli.main(
        [
            "stats",
            str(tmp_path / "runs.csv"),
            "--reference",
            "pyramid-ucb1-none",
            "--metric",
            "final_points",
            "--n-boot",
            "200",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Метрика: final_points" in out
    assert "pyramid-ucb1-full_cycle" in out

    rows = _read_csv(tmp_path / "results.csv")
    assert {row["config"] for row in rows} == {"pyramid-ucb1-none", "pyramid-ucb1-full_cycle"}
    assert all(row["method"] == "bootstrap" for row in rows)
    significant = [row for row in rows if row["config"] == "pyramid-ucb1-full_cycle"][0]
    assert float(significant["diff"]) > 0
    assert float(significant["p_adjusted"]) < 0.05


def test_stats_reports_errors(tmp_path, capsys):
    assert cli.main(["stats", str(tmp_path / "nope.csv")]) == 2
    assert "Нет файла" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "run",
                "--config",
                "pyramid-ucb1-none",
                "--runs",
                "2",
                "--steps",
                "30",
                "--out",
                str(tmp_path),
                "--no-plots",
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()
    runs = tmp_path / "runs.csv"

    assert cli.main(["stats", str(runs), "--reference", "nope"]) == 2
    assert "Нет базлайна" in capsys.readouterr().err

    assert cli.main(["stats", str(runs), "--metric", "no_such_metric"]) == 2
    assert "нет в runs.csv" in capsys.readouterr().err


def test_report_command_writes_markdown_and_json(tmp_path, capsys):
    assert (
        cli.main(
            [
                "run",
                "--config",
                "pyramid-ucb1-none",
                "--config",
                "pyramid-ucb1-full_cycle",
                "--runs",
                "4",
                "--steps",
                "120",
                "--out",
                str(tmp_path),
                "--no-plots",
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert cli.main(["report", str(tmp_path), "--n-boot", "200", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "Вердикты" in out
    assert (tmp_path / "report.md").stat().st_size > 0

    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["reference"] == "pyramid-ucb1-none"
    assert [item["code"] for item in payload["hypotheses"]] == [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ]
    h1 = payload["hypotheses"][0]
    assert h1["status"] in {"подтверждена", "частично", "опровергнута", "нет данных"}
    assert "points_after_epoch_1" in payload["metrics"]

    assert cli.main(["report", str(tmp_path / "nope")]) == 2
    assert "Нет runs.csv" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["list", "run", "show", "stats", "report", "demo"])
def test_parser_has_all_commands(command):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--help"])
