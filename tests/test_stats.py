"""Парная статистика: тесты, bootstrap, поправки и CSV-обмен (P3).

Проверки намеренно «ручные»: для маленьких выборок ожидаемые значения
считаются по формулам прямо в тесте, поэтому тест поймает ошибку в
реализации, а не просто зафиксирует текущий вывод.
"""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from bandit import stats as s


def test_bootstrap_mean_ci_is_reproducible_and_brackets_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    first = s.bootstrap_mean_ci(values, n_boot=500, seed=11)
    second = s.bootstrap_mean_ci(values, n_boot=500, seed=11)
    assert first == second
    assert first["n"] == 10
    assert first["mean"] == pytest.approx(5.5)
    assert first["ci_lo"] < first["mean"] < first["ci_hi"]


def test_bootstrap_mean_ci_degenerate_samples():
    empty = s.bootstrap_mean_ci([np.nan, np.nan])
    assert empty["n"] == 0
    assert math.isnan(empty["mean"])
    assert math.isnan(empty["ci_lo"])

    single = s.bootstrap_mean_ci([2.5])
    assert single["mean"] == 2.5
    assert math.isnan(single["ci_lo"]) and math.isnan(single["ci_hi"])


def test_paired_bootstrap_p_value_is_never_zero():
    values = [2.0] * 8
    reference = [1.0] * 8
    result = s.paired_bootstrap(values, reference, n_boot=200, seed=3)
    assert result["diff"] == pytest.approx(1.0)
    assert result["ci_lo"] == pytest.approx(1.0)
    assert result["ci_hi"] == pytest.approx(1.0)
    assert 0.0 < result["p_value"] < 0.05
    assert (result["wins"], result["losses"], result["ties"]) == (8, 0, 0)


def test_paired_bootstrap_drops_nan_pairs_and_checks_lengths():
    values = [1.0, np.nan, 3.0, 4.0]
    reference = [0.0, 0.0, np.nan, 0.0]
    result = s.paired_bootstrap(values, reference, n_boot=100, seed=5)
    assert result["n"] == 2
    assert result["diff"] == pytest.approx(2.5)

    with pytest.raises(ValueError):
        s.paired_bootstrap([1.0, 2.0], [1.0])


def test_wilcoxon_detects_consistent_gain():
    values = np.arange(1.0, 11.0)
    zeros = np.zeros_like(values)
    result = s.wilcoxon_signed_rank(values, zeros)
    assert result["n"] == 10
    assert result["statistic"] == pytest.approx(55.0)
    assert result["p_value"] < 0.01


def test_wilcoxon_drops_zero_differences():
    values = [3.0, 3.0, 3.0]
    result = s.wilcoxon_signed_rank(values, values)
    assert result == {"statistic": 0.0, "p_value": 1.0, "n": 0}


def test_wilcoxon_applies_tie_correction():
    values = [1.0, 1.0, 1.0, 1.0]
    zeros = [0.0] * 4
    result = s.wilcoxon_signed_rank(values, zeros)
    # Ранги всех |d| совпадают: дисперсия уменьшается на (4^3-4)/48 = 1.25.
    variance = 4 * 5 * 9 / 24.0 - 1.25
    z = (10.0 - 5.0) / math.sqrt(variance)
    assert result["n"] == 4
    assert result["statistic"] == pytest.approx(10.0)
    assert result["p_value"] == pytest.approx(math.erfc(abs(z) / math.sqrt(2.0)))


def test_sign_test_is_exact_for_small_samples():
    absolute = s.sign_test([1.0] * 6, [0.0] * 6)
    assert absolute["n"] == 6 and absolute["wins"] == 6
    assert absolute["p_value"] == pytest.approx(2.0 / 2**6)

    ties = s.sign_test([1.0, 1.0, 0.0, 0.0], [0.0] * 4)
    assert ties["n"] == 2 and ties["p_value"] == pytest.approx(0.5)

    nothing = s.sign_test([2.0, 2.0], [2.0, 2.0])
    assert nothing == {"p_value": 1.0, "n": 0, "wins": 0, "losses": 0}


def test_cohens_dz_handles_constant_and_tiny_samples():
    assert s.cohens_dz([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert math.isinf(s.cohens_dz([2.0, 2.0], [0.0, 0.0]))
    assert s.cohens_dz([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert math.isnan(s.cohens_dz([1.0], [0.0]))


def test_holm_monotone_and_order_preserving():
    adjusted = s.holm([0.01, 0.04, 0.03])
    # p(1)*3 = 0.03 → p(3)*2 = 0.06 → p(2)*1 = 0.04, но running max = 0.06.
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_nan():
    assert s.holm([0.9, 0.9]) == pytest.approx([1.0, 1.0])
    adjusted = s.holm([float("nan"), 0.01])
    assert math.isnan(adjusted[0])
    assert adjusted[1] == pytest.approx(0.02)


def test_holm_dict_keeps_keys():
    adjusted = s.holm_dict({"b": 0.02, "a": 0.01})
    assert list(adjusted) == ["b", "a"]
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.02)


def test_paired_comparison_methods_differ_but_share_ci():
    values = [5.0, 6.0, 7.0, 8.0, 9.0]
    reference = [1.0, 2.0, 3.0, 4.0, 5.0]
    bootstrap = s.paired_comparison(values, reference, n_boot=500, seed=7)
    wilcoxon = s.paired_comparison(values, reference, method="wilcoxon")
    sign = s.paired_comparison(values, reference, method="sign")
    assert bootstrap["diff"] == pytest.approx(4.0)
    assert bootstrap["mean"] == pytest.approx(7.0)
    assert bootstrap["reference_mean"] == pytest.approx(3.0)
    assert bootstrap["ci_lo"] == wilcoxon["ci_lo"] == sign["ci_lo"]
    assert sign["p_value"] == pytest.approx(2.0 / 2**5)
    assert set(bootstrap) == {
        "n",
        "mean",
        "reference_mean",
        "diff",
        "ci_lo",
        "ci_hi",
        "p_value",
        "effect_size",
        "wins",
        "losses",
        "ties",
    }
    with pytest.raises(KeyError):
        s.paired_comparison(values, reference, method="t-test")


def test_compare_values_adjusts_with_holm_and_keeps_reference_row():
    values_by_name = {
        "base": {"final_points": [10.0, 11.0, 12.0, 13.0, 14.0]},
        "better": {"final_points": [20.0, 21.0, 22.0, 23.0, 24.0]},
        "worse": {"final_points": [0.0, 1.0, 2.0, 3.0, 4.0]},
    }
    results = s.compare_values(values_by_name, "base", n_boot=500, seed=1)
    assert [item.config for item in results] == ["base", "better", "worse"]
    by_name = {item.config: item for item in results}
    assert by_name["base"].p_adjusted == pytest.approx(1.0)
    assert not by_name["base"].significant
    assert by_name["better"].diff == pytest.approx(10.0)
    assert by_name["worse"].diff == pytest.approx(-10.0)
    assert by_name["better"].p_adjusted >= by_name["better"].p_value
    assert by_name["better"].significant

    again = s.compare_values(values_by_name, "base", n_boot=500, seed=1)
    assert [item.as_dict() for item in again] == [item.as_dict() for item in results]


def test_compare_values_requires_reference_and_metric():
    values_by_name = {"only": {"final_points": [1.0, 2.0, 3.0]}}
    with pytest.raises(KeyError):
        s.compare_values(values_by_name, "missing")
    with pytest.raises(KeyError):
        s.compare_values(values_by_name, "only", metric="efficiency")


def test_compare_all_metrics_skips_metrics_without_full_coverage():
    values_by_name = {
        "base": {"final_points": [1.0, 2.0], "efficiency": [0.5, 0.6]},
        "cand": {"final_points": [3.0, 4.0]},
    }
    out = s.compare_all_metrics(values_by_name, "base", n_boot=200, seed=2)
    assert set(out) == {"final_points"}
    assert [item.config for item in out["final_points"]] == ["base", "cand"]


def _write_runs_csv(path, rows, metrics=("final_points", "final_epoch")):
    columns = list(s.CSV_INDEX_COLUMNS) + list(metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_runs_csv_restores_run_order_and_nan(tmp_path):
    path = tmp_path / "runs.csv"
    _write_runs_csv(
        path,
        [
            {"config": "a", "run": 1, "env_seed": 20, "agent_seed": 21,
             "final_points": "7", "final_epoch": "2"},
            {"config": "a", "run": 0, "env_seed": 10, "agent_seed": 11,
             "final_points": "5", "final_epoch": "nan"},
            {"config": "b", "run": 0, "env_seed": 10, "agent_seed": 11,
             "final_points": "", "final_epoch": "1"},
            {"config": "", "run": 0, "env_seed": 10, "agent_seed": 11,
             "final_points": "9", "final_epoch": "0"},
        ],
    )
    values, metrics = s.load_runs_csv(path)
    assert metrics == ["final_points", "final_epoch"]
    assert list(values) == ["a", "b"]
    assert values["a"]["final_points"].tolist() == [5.0, 7.0]
    assert math.isnan(values["a"]["final_epoch"][0])
    assert values["a"]["final_epoch"][1] == 2.0
    assert math.isnan(values["b"]["final_points"][0])


def test_load_runs_csv_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        s.load_runs_csv(path)


def test_write_results_csv_round_trip(tmp_path):
    values_by_name = {
        "base": {"final_points": [10.0, 11.0, 12.0, 13.0]},
        "cand": {"final_points": [20.0, 21.0, 22.0, 23.0]},
    }
    results = s.compare_values(values_by_name, "base", n_boot=300, seed=4)
    path = s.write_results_csv(tmp_path / "results.csv", results)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["config"] == "base"
    assert rows[0]["metric"] == "final_points"
    assert rows[0]["reference"] == "base"
    assert float(rows[0]["p_adjusted"]) == pytest.approx(1.0)
    assert float(rows[1]["diff"]) == pytest.approx(10.0)
    assert rows[0]["method"] == "bootstrap"


def _result(**overrides):
    base = dict(
        config="cand",
        metric="final_points",
        reference="base",
        method="bootstrap",
        n=10,
        mean=20.0,
        reference_mean=10.0,
        diff=10.0,
        ci_lo=5.0,
        ci_hi=15.0,
        p_value=0.00001,
        p_adjusted=0.00002,
        effect_size=1.234,
        wins=9,
        losses=1,
        ties=0,
    )
    base.update(overrides)
    return s.PairedResult(**base)


def test_format_rows_marks_significance_and_handles_nan():
    rows = s.format_rows([_result(), _result(config="flat", diff=0.0, p_adjusted=0.5)])
    significant, plain = rows
    assert significant["config"] == "cand*"
    assert significant["Δ"] == "10.00"
    assert significant["95% ДИ"] == "[5.00; 15.00]"
    assert significant["p"] == "<1e-4"
    assert significant["dz"] == "1.23"
    assert significant["+/−"] == "9/1"
    assert plain["config"] == "flat"
    assert plain["p Холм"] == "0.5000"

    nan_row = s.format_rows([_result(effect_size=float("nan"), ci_lo=float("nan"))])[0]
    assert nan_row["dz"] == "—"
    assert nan_row["95% ДИ"] == "[—; 15.00]"


def test_comparison_columns_match_format_rows():
    row = s.format_rows([_result()])[0]
    assert tuple(row) == s.COMPARISON_COLUMNS
