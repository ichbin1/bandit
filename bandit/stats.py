"""Парная статистика над сохранёнными прогонами: тесты, bootstrap, поправки.

Зачем парные тесты
------------------
``run_config`` прогоняет все конфиги одного набора на одних и тех же парах
сидов ``(env_seed, agent_seed)``. Значит, значения метрик двух конфигов
образуют *пары*: разность считается по одинаковым средам, и её дисперсия
существенно ниже, чем у разности независимых выборок (в ``runs.csv`` есть
колонки сидов). Поэтому основной инструмент — ``paired_bootstrap`` и
``wilcoxon_signed_rank``, а не сравнение двух средних «в лоб».

Что считается
-------------
* ``bootstrap_mean_ci`` — перцентильный интервал среднего (робастен к
  ненормальности: очки здесь с тяжёлым правым хвостом);
* ``paired_bootstrap`` — интервал и p-value для средней разности. p-value
  двусторонний: ``2·min(P(Δ*≤0), P(Δ*≥0))`` по bootstrap-распределению, с
  поправкой ``(k+1)/(n_boot+1)``, чтобы p никогда не был ровно нулём;
* ``wilcoxon_signed_rank`` — критерий знаковых рангов (нормальное
  приближение с поправкой на связи): не требует нормальности и устойчив к
  выбросам;
* ``sign_test`` — точный биномиальный тест «кто чаще выигрывает» (удобен,
  когда важна только доля побед, а не величина разрыва);
* ``holm`` — поправка на множественные сравнения (контроль FWER).

Зависимостей кроме numpy нет: ``scipy`` в ``requirements.txt`` опционален, а
всё перечисленное занимает несколько десятков строк и/или выражается через
``math.erfc``. Bootstrap инициализируется явным сидом, поэтому таблицы
воспроизводятся побитово.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_N_BOOT = 4000
"""Число bootstrap-выборок по умолчанию (хватает для 3 значащих цифр p)."""

DEFAULT_SEED = 20240501
"""Сид bootstrap по умолчанию: результаты не зависят от порядка запусков."""

METHODS: Tuple[str, ...] = ("bootstrap", "wilcoxon", "sign")
"""Доступные p-value-тесты для парных сравнений."""


def _paired(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Выровнять две выборки по парам и выбросить нечисловые пары."""
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    if left.size != right.size:
        raise ValueError(
            f"Парные выборки должны быть одной длины: {left.size} != {right.size}"
        )
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(n_boot), int(n)))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Перцентильный bootstrap-интервал для среднего.

    Возвращает ``mean``, ``ci_lo``/``ci_hi`` (уровень ``1-alpha``) и ``n``.
    Для ``n < 2`` интервал не определён (nan): одна точка не даёт разброса.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if n == 1:
        return {
            "mean": float(array[0]),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 1,
        }
    means = array[_bootstrap_indices(n, n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(array.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def paired_bootstrap(
    values: Sequence[float],
    reference: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Bootstrap для средней парной разности ``values - reference``.

    Пересэмплируются *пары* целиком (а не выборки по отдельности): это
    сохраняет корреляцию между конфигами, которая и есть источник выигрыша
    парного дизайна.
    """
    left, right = _paired(values, reference)
    n = int(left.size)
    empty = {
        "diff": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "p_value": float("nan"),
        "n": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
    }
    if n == 0:
        return empty
    diff = left - right
    wins = int(np.count_nonzero(diff > 0))
    losses = int(np.count_nonzero(diff < 0))
    ties = int(np.count_nonzero(diff == 0))
    if n == 1:
        return {
            "diff": float(diff[0]),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_value": float("nan"),
            "n": 1,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    means = diff[_bootstrap_indices(n, n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    correction = float(n_boot) + 1.0
    p_low = (float(np.count_nonzero(means <= 0.0)) + 1.0) / correction
    p_high = (float(np.count_nonzero(means >= 0.0)) + 1.0) / correction
    return {
        "diff": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(min(1.0, 2.0 * min(p_low, p_high))),
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _tie_sizes(values: np.ndarray) -> List[int]:
    """Размеры групп равных значений (для поправки на связи)."""
    values = np.sort(np.asarray(values, dtype=float))
    sizes: List[int] = []
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or values[index] != values[start]:
            sizes.append(index - start)
            start = index
    return sizes


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ранги 1..n со средними рангами для связей."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        stop = start
        while stop + 1 < array.size and array[order[stop + 1]] == array[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def wilcoxon_signed_rank(
    values: Sequence[float], reference: Sequence[float]
) -> Dict[str, float]:
    """Критерий знаковых рангов Уилкоксона (нормальное приближение).

    Нулевые разности отбрасываются (стандартная практика), связи по
    ``|d|`` уменьшают дисперсию. Возвращает ``statistic`` (сумма рангов
    положительных разностей), ``p_value`` и число использованных пар ``n``.
    """
    left, right = _paired(values, reference)
    diff = left - right
    diff = diff[diff != 0.0]
    n = int(diff.size)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}
    abs_diff = np.abs(diff)
    ranks = _average_ranks(abs_diff)
    w_plus = float(ranks[diff > 0.0].sum())
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    for size in _tie_sizes(abs_diff):
        if size > 1:
            variance -= (size**3 - size) / 48.0
    if variance <= 0.0:
        return {"statistic": w_plus, "p_value": 1.0, "n": n}
    z = (w_plus - mean) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {"statistic": w_plus, "p_value": float(min(1.0, p_value)), "n": n}


def sign_test(
    values: Sequence[float], reference: Sequence[float]
) -> Dict[str, float]:
    """Точный двусторонний критерий знаков (биномиальный).

    Считает только направления разностей: нули отбрасываются, поэтому тест
    отвечает на вопрос «какой конфиг чаще выигрывает», не глядя на величину
    выигрыша. При малом числе пар это единственный тест с корректным
    уровнем значимости (bootstrap опирается на асимптотику).
    """
    left, right = _paired(values, reference)
    diff = left - right
    diff = diff[diff != 0.0]
    wins = int(np.count_nonzero(diff > 0.0))
    losses = int(np.count_nonzero(diff < 0.0))
    n = wins + losses
    if n == 0:
        return {"p_value": 1.0, "n": 0, "wins": 0, "losses": 0}
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return {
        "p_value": float(min(1.0, 2.0 * tail)),
        "n": n,
        "wins": wins,
        "losses": losses,
    }


def cohens_dz(values: Sequence[float], reference: Sequence[float]) -> float:
    """Размер эффекта ``dz`` для парных данных: ``mean(d) / sd(d)``.

    При нулевом разбросе разностей величина не определена: возвращается
    ``0``, если средняя разность тоже нулевая, и ``±inf`` иначе.
    """
    left, right = _paired(values, reference)
    diff = left - right
    n = int(diff.size)
    if n < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    mean = float(diff.mean())
    if sd == 0.0:
        return 0.0 if mean == 0.0 else math.copysign(float("inf"), mean)
    return mean / sd


def holm(pvalues: Sequence[float]) -> List[float]:
    """Поправка Холма–Бонферрони: скорректированные p-value.

    Сохраняет порядок входа. NaN остаются NaN и не участвуют в семействе.
    """
    array = np.asarray(pvalues, dtype=float).reshape(-1)
    n = int(array.size)
    order = np.argsort(np.where(np.isfinite(array), array, np.inf), kind="mergesort")
    adjusted = np.full(n, np.nan)
    running = 0.0
    rank = 0
    for index in order:
        if not np.isfinite(array[index]):
            adjusted[index] = np.nan
            continue
        running = max(running, (n - rank) * float(array[index]))
        adjusted[index] = min(1.0, running)
        rank += 1
    return [float(value) for value in adjusted]


def holm_dict(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """``holm`` для словаря «имя → p» с сохранением ключей."""
    names = list(pvalues)
    adjusted = holm([pvalues[name] for name in names])
    return dict(zip(names, adjusted))


@dataclass(frozen=True)
class PairedResult:
    """Одно парное сравнение конфига с базлайном по одной метрике."""

    config: str
    metric: str
    reference: str
    method: str
    n: int
    mean: float
    reference_mean: float
    diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_adjusted: float
    effect_size: float
    wins: int
    losses: int
    ties: int
    #: Уровень значимости, с которым сравнивается ``p_adjusted``.
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        """Значимо ли отличие на уровне ``alpha`` (используется CLI/отчётом)."""
        return bool(np.isfinite(self.p_adjusted) and self.p_adjusted < self.alpha)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "metric": self.metric,
            "reference": self.reference,
            "method": self.method,
            "n": self.n,
            "mean": self.mean,
            "reference_mean": self.reference_mean,
            "diff": self.diff,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }


def paired_comparison(
    values: Sequence[float],
    reference: Sequence[float],
    method: str = "bootstrap",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Полный набор статистик для одной пары выборок.

    Доверительный интервал всегда bootstrap-перцентильный (он же даёт
    ``diff`` и счёт побед), а ``p_value`` считается выбранным методом:
    ``bootstrap`` (по умолчанию), ``wilcoxon`` или ``sign``.
    """
    key = str(method).strip().lower()
    if key not in METHODS:
        raise KeyError(f"Неизвестный метод {method!r}. Доступны: {', '.join(METHODS)}")
    boot = paired_bootstrap(values, reference, n_boot=n_boot, alpha=alpha, seed=seed)
    p_value = float(boot["p_value"])
    if key == "wilcoxon":
        p_value = float(wilcoxon_signed_rank(values, reference)["p_value"])
    elif key == "sign":
        p_value = float(sign_test(values, reference)["p_value"])
    left, right = _paired(values, reference)
    return {
        "n": int(boot["n"]),
        "mean": float(left.mean()) if left.size else float("nan"),
        "reference_mean": float(right.mean()) if right.size else float("nan"),
        "diff": float(boot["diff"]),
        "ci_lo": float(boot["ci_lo"]),
        "ci_hi": float(boot["ci_hi"]),
        "p_value": p_value,
        "effect_size": cohens_dz(values, reference),
        "wins": int(boot["wins"]),
        "losses": int(boot["losses"]),
        "ties": int(boot["ties"]),
    }


def compare_values(
    values_by_name: Mapping[str, Mapping[str, Sequence[float]]],
    reference: str,
    metric: str = "final_points",
    method: str = "bootstrap",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> List[PairedResult]:
    """Сравнить каждый конфиг с базлайном ``reference`` по метрике ``metric``.

    ``values_by_name`` — «имя конфига → {метрика: массив по прогонам}», т.е.
    ровно результат ``metrics.summarize``. Пары выравниваются по индексу
    прогона: он одинаков для всех конфигов благодаря ``make_seed_pairs``.

    p-value корректируются методом Холма по всему семейству сравнений
    (базлайн в семейство не входит: у него ``p = 1``). Сид bootstrap
    сдвигается на позицию конфига, чтобы выборки разных сравнений не были
    скоррелированы между собой.
    """
    if reference not in values_by_name:
        raise KeyError(
            f"Базлайн {reference!r} отсутствует в прогоне. "
            f"Доступны: {', '.join(sorted(values_by_name))}"
        )
    if metric not in values_by_name[reference]:
        raise KeyError(
            f"У базлайна {reference!r} нет метрики {metric!r}. "
            "Она считается только при наличии оракула в прогоне."
        )
    baseline = np.asarray(values_by_name[reference][metric], dtype=float)
    results: List[PairedResult] = []
    pvalues: List[float] = []
    for position, (name, table) in enumerate(values_by_name.items()):
        if metric not in table:
            raise KeyError(f"У конфига {name!r} нет метрики {metric!r}")
        stats = paired_comparison(
            table[metric],
            baseline,
            method=method,
            n_boot=n_boot,
            alpha=alpha,
            seed=int(seed) + position,
        )
        results.append(
            PairedResult(
                config=name,
                metric=metric,
                reference=reference,
                method=str(method).strip().lower(),
                alpha=float(alpha),
                p_adjusted=float(stats["p_value"]),
                **stats,
            )
        )
        pvalues.append(1.0 if name == reference else stats["p_value"])
    adjusted = holm(pvalues)
    return [replace(item, p_adjusted=float(p)) for item, p in zip(results, adjusted)]


def compare_all_metrics(
    values_by_name: Mapping[str, Mapping[str, Sequence[float]]],
    reference: str,
    metrics: Optional[Iterable[str]] = None,
    method: str = "bootstrap",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[PairedResult]]:
    """``compare_values`` для набора метрик (пропуская отсутствующие)."""
    names = list(metrics) if metrics is not None else [
        metric for metric in next(iter(values_by_name.values()))
    ]
    out: Dict[str, List[PairedResult]] = {}
    for offset, metric in enumerate(names):
        if any(metric not in table for table in values_by_name.values()):
            continue
        out[metric] = compare_values(
            values_by_name,
            reference,
            metric=metric,
            method=method,
            n_boot=n_boot,
            alpha=alpha,
            seed=int(seed) + 1000 * (offset + 1),
        )
    return out


def _to_float(text: Any) -> float:
    """Разобрать число из CSV; пустое/``nan`` → nan (а не исключение)."""
    if text is None:
        return float("nan")
    value = str(text).strip()
    if not value or value.lower() in {"nan", "none", "null", "—"}:
        return float("nan")
    return float(value)


CSV_INDEX_COLUMNS: Tuple[str, ...] = ("config", "run", "env_seed", "agent_seed")
"""Служебные колонки ``runs.csv`` (всё остальное — метрики)."""


def load_runs_csv(
    path: Path | str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Прочитать ``runs.csv`` в «конфиг → {метрика: массив по прогонам}».

    Порядок прогонов восстанавливается по колонке ``run``, поэтому файл
    можно сортировать/править вручную, не ломая парность.
    """
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Пустой CSV: {path}")
        metric_names = [
            name for name in reader.fieldnames if name not in CSV_INDEX_COLUMNS
        ]
        grouped: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
        for row in reader:
            name = str(row.get("config", "")).strip()
            if not name:
                continue
            run_index = int(_to_float(row.get("run", 0)))
            table = grouped.setdefault(name, {metric: [] for metric in metric_names})
            for metric in metric_names:
                table[metric].append((run_index, _to_float(row.get(metric))))
    values: Dict[str, Dict[str, np.ndarray]] = {}
    for name, table in grouped.items():
        values[name] = {
            metric: np.asarray(
                [value for _, value in sorted(pairs, key=lambda item: item[0])],
                dtype=float,
            )
            for metric, pairs in table.items()
        }
    return values, metric_names


def write_results_csv(path: Path | str, results: Sequence[PairedResult]) -> Path:
    """Сохранить сравнения в CSV (по строке на конфиг)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "config",
        "metric",
        "reference",
        "method",
        "n",
        "mean",
        "reference_mean",
        "diff",
        "ci_lo",
        "ci_hi",
        "p_value",
        "p_adjusted",
        "effect_size",
        "wins",
        "losses",
        "ties",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in results:
            writer.writerow({name: item.as_dict()[name] for name in columns})
    return path


def _fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


COMPARISON_COLUMNS: Tuple[str, ...] = (
    "config",
    "n",
    "Δ",
    "95% ДИ",
    "p",
    "p Холм",
    "dz",
    "+/−",
)
"""Заголовки таблицы сравнений для CLI и отчёта."""


def format_rows(
    results: Sequence[PairedResult],
    significant_mark: str = "*",
) -> List[Dict[str, str]]:
    """Строки для текстовой таблицы (значения уже отформатированы)."""
    rows: List[Dict[str, str]] = []
    for item in results:
        rows.append(
            {
                "config": item.config + (significant_mark if item.significant else ""),
                "n": str(item.n),
                "Δ": _fmt(item.diff, 2),
                "95% ДИ": f"[{_fmt(item.ci_lo, 2)}; {_fmt(item.ci_hi, 2)}]",
                "p": _fmt_p(item.p_value),
                "p Холм": _fmt_p(item.p_adjusted),
                "dz": _fmt(item.effect_size, 2),
                "+/−": f"{item.wins}/{item.losses}",
            }
        )
    return rows


__all__ = [
    "COMPARISON_COLUMNS",
    "CSV_INDEX_COLUMNS",
    "DEFAULT_N_BOOT",
    "DEFAULT_SEED",
    "METHODS",
    "PairedResult",
    "bootstrap_mean_ci",
    "cohens_dz",
    "compare_all_metrics",
    "compare_values",
    "format_rows",
    "holm",
    "holm_dict",
    "load_runs_csv",
    "paired_bootstrap",
    "paired_comparison",
    "sign_test",
    "wilcoxon_signed_rank",
    "write_results_csv",
]





