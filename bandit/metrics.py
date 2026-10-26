"""Метрики гипотезы: очки, сожаление, эффективность, скорость адаптации.

Про «настоящее» сожаление
-------------------------
В ``LayeredPyramidBandit`` среда меняется действиями агента (покупка рук,
технологии), поэтому классическое сожаление ``T·μ* − Σ r_t`` не определено:
«лучшая рука» — движущаяся цель. Поэтому отчёт даёт три величины:

* ``arm_regret`` — недобор из-за выбора руки: ``μ*(t) − μ(a_t)`` в момент
  выбора (по снимку ``means`` до действия). Мета-шаги исключены: они не
  выбирают руку, их цена видна в ``efficiency``;
* ``efficiency`` — ``points_T / oracle_points_T`` при тех же сидах: доля
  выхлопа идеальной политики (оракул тоже платит за метацикл). Оракул обязан
  собирать данные всех тиров рецепта, иначе он не открывает эпохи; если
  ``efficiency > 1``, значит оракул задан неверно (не верхняя граница).
* ``point_regret`` — ``oracle_points(t) − points(t)`` по шагам (для графиков).
* ``useful_info_share`` / ``junk_info_share`` — доля полезных единиц и мусора в
  информационном потоке. Это характеристика *режима* (H6), а не политики:
  среда задаёт её вероятностью полезной единицы.

Это ограничение зафиксировано в ``docs/HYPOTHESIS.md``: метрики сопоставимы
между политиками, но не являются сожалением в смысле теории бандитов.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .runner import RunRecords

METRIC_DESCRIPTIONS: Dict[str, str] = {
    "final_points": "очки к концу горизонта",
    "cumulative_arm_regret": "суммарный недобор по выбору рук (мета-шаги не входят)",
    "mean_arm_regret": "средний недобор на одно вытягивание руки",
    "optimal_arm_rate": "доля вытягиваний лучшей (истинной) руки",
    "arm_pull_share": "доля шагов, потраченных на руки (остальное — мета)",
    "meta_step_share": "доля шагов, потраченных на метацикл",
    "conversions": "число результативных конверсий данных в множитель",
    "arm_buys": "число купленных рук",
    "gate_crafts": "число собранных ворот (переходов эпох)",
    "amplifier_crafts": "число собранных амплификаторов",
    "multiplier_from_conversions": "вклад конверсий в множитель (логарифмический бонус)",
    "multiplier_from_arms": (
        "чистый вклад покупок рук в множитель: бонус за руку минус потеря "
        "логарифмического бонуса за потраченные конверсии"
    ),
    "multiplier_from_gates": "вклад ворот в множитель",
    "multiplier_from_amplifiers": "вклад амплификаторов в множитель",
    "points_per_pull": "очки на одно вытягивание руки",
    "auc_normalized": "площадь под кривой очков / (T · итог): 1 = всё сразу",
    "steps_to_50pct": "шаг, на котором достигнуто 50% собственного итога",
    "steps_to_90pct": "шаг, на котором достигнуто 90% собственного итога",
    "steps_to_epoch_1": "шаг выхода в эпоху 1 (nan — не вышли)",
    "steps_to_epoch_2": "шаг выхода в эпоху 2 (nan — не вышли)",
    "final_epoch": "эпоха к концу горизонта",
    "final_n_arms": "рук к концу горизонта",
    "final_multiplier": "множитель очков к концу горизонта",
    "points_after_epoch_1": "очки после выхода в эпоху 1 (nan — эпоха не достигнута)",
    "efficiency": "очки / очки оракула при тех же сидах",
    "useful_info_share": "доля полезных единиц в информационном потоке (H6)",
    "junk_info_share": "доля мусора в информационном потоке (H6)",
}


#: Ветви метацикла: имя метрики → код вида действия в ``RunRecords.kind``.
#:
#: Коды заданы ``bandit.envs.base.action_kind``: 0 — рука, 1 — конверсия
#: данных, 2 — покупка руки, 3 — ворота, 4 — амплификатор.
META_BRANCH_KINDS: Dict[str, int] = {
    "conversions": 1,
    "arm_buys": 2,
    "gate_crafts": 3,
    "amplifier_crafts": 4,
}

#: Метрика вклада в множитель → ветвь метацикла (ключ ``META_BRANCH_KINDS``).
#:
#: Имена метрик короче счётчиков, потому что они читаются в отчёте рядом с
#: ``final_multiplier``: ``multiplier_from_arms`` понятнее, чем
#: ``multiplier_from_arm_buys``.
MULTIPLIER_METRICS: Dict[str, str] = {
    "multiplier_from_conversions": "conversions",
    "multiplier_from_arms": "arm_buys",
    "multiplier_from_gates": "gate_crafts",
    "multiplier_from_amplifiers": "amplifier_crafts",
}


def arm_pull_mask(records: RunRecords) -> np.ndarray:
    """Маска шагов, где действие — вытягивание руки."""
    return records.kind == 0


def cumulative_points(records: RunRecords) -> np.ndarray:
    return np.asarray(records.points, dtype=float)


def final_points(records: RunRecords) -> np.ndarray:
    return np.asarray(records.points[:, -1], dtype=float)


def cumulative_arm_regret(records: RunRecords) -> np.ndarray:
    """Накопленный недобор по рукам (мета-шаги дают 0 прироста)."""
    return np.nancumsum(np.asarray(records.arm_regret, dtype=float), axis=1)


def total_arm_regret(records: RunRecords) -> np.ndarray:
    return np.nansum(np.asarray(records.arm_regret, dtype=float), axis=1)


def mean_arm_regret(records: RunRecords) -> np.ndarray:
    pulls = arm_pull_mask(records).sum(axis=1)
    total = total_arm_regret(records)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pulls > 0, total / pulls, np.nan)


def arm_pull_share(records: RunRecords) -> np.ndarray:
    return arm_pull_mask(records).mean(axis=1)


def meta_mask(records: RunRecords) -> np.ndarray:
    """Маска шагов, где действие — мета-действие."""
    return records.kind != 0


def meta_step_share(records: RunRecords) -> np.ndarray:
    """Доля шагов, потраченных на метацикл (дополнение к ``arm_pull_share``)."""
    return meta_mask(records).mean(axis=1)


def meta_action_counts(records: RunRecords) -> Dict[str, np.ndarray]:
    """Число *результативных* мета-действий каждого вида за прогон.

    Считаются только шаги с ``effective == 1``: попытка купить руку без
    множителей или собрать ворота без полного рецепта экономику не меняет, и в
    бюджете ветвей такие шаги были бы шумом. Имена ключей совпадают с
    метриками (``conversions``, ``arm_buys``, ``gate_crafts``,
    ``amplifier_crafts``).
    """
    kind = np.asarray(records.kind)
    effective = np.asarray(records.effective) == 1
    return {
        name: ((kind == code) & effective).sum(axis=1).astype(float)
        for name, code in META_BRANCH_KINDS.items()
    }


def multiplier_increment(records: RunRecords) -> np.ndarray:
    """Пошаговое изменение множителя: ``multiplier_after - multiplier_before``."""
    return np.asarray(records.multiplier_after, dtype=float) - np.asarray(
        records.multiplier_before, dtype=float
    )


def multiplier_from_branch(records: RunRecords, code: int) -> np.ndarray:
    """Вклад шагов вида ``code`` в множитель (сумма пошаговых изменений)."""
    increment = multiplier_increment(records)
    mask = np.asarray(records.kind) == int(code)
    return np.where(mask, increment, 0.0).sum(axis=1)


def multiplier_from_conversions(records: RunRecords) -> np.ndarray:
    return multiplier_from_branch(records, META_BRANCH_KINDS["conversions"])


def multiplier_from_arms(records: RunRecords) -> np.ndarray:
    """Чистый вклад покупок рук: бонус за руку минус плата конверсиями.

    Покупка списывает ``n_multipliers``, а логарифмический бонус за конверсии
    зависит от их числа, поэтому изменение множителя на шаге покупки включает
    и потерю. Это честная цена действия, а не только его выгода.
    """
    return multiplier_from_branch(records, META_BRANCH_KINDS["arm_buys"])


def multiplier_from_gates(records: RunRecords) -> np.ndarray:
    return multiplier_from_branch(records, META_BRANCH_KINDS["gate_crafts"])


def multiplier_from_amplifiers(records: RunRecords) -> np.ndarray:
    return multiplier_from_branch(records, META_BRANCH_KINDS["amplifier_crafts"])


def multiplier_budget(records: RunRecords) -> Dict[str, np.ndarray]:
    """Разложение прироста множителя по ветвям метацикла.

    Множитель меняется только на мета-шагах, а ``multiplier_before/after``
    записываются на каждом шаге, поэтому вклады ветвей складываются
    телескопически: их сумма равна ``final_multiplier - 1`` (стартовый
    множитель). Это и есть бюджет ветвей — проверка в тестах.
    """
    return {
        metric: multiplier_from_branch(records, META_BRANCH_KINDS[branch])
        for metric, branch in MULTIPLIER_METRICS.items()
    }


def multiplier_budget_residual(records: RunRecords) -> np.ndarray:
    """Невязка бюджета ветвей: ``Σ вкладов - (final_multiplier - 1)``.

    Должна быть нулём с точностью до плавающей арифметики; ненулевое значение
    означает, что множитель менялся на шаге, не отнесённом ни к одной ветви.
    """
    total = sum(multiplier_budget(records).values())
    final = final_multiplier(records)
    initial = np.asarray(records.multiplier_before, dtype=float)[:, 0]
    return total - (final - initial)


def useful_info_share(records: RunRecords) -> np.ndarray:
    """Доля полезных единиц в информационном потоке прогона (H6).

    Знаменатель — ``info_gained`` (все единицы, включая мусор), числитель —
    ``useful_gained``. Величина описывает *режим*, а не политику: она задана
    средой, поэтому служит шкалой перегрузки в отчёте. Если среда не ведёт
    учёт информации (например, ``BernoulliBandit``), результат — ``nan``.
    """
    useful = np.asarray(records.useful_gained, dtype=float).sum(axis=1)
    total = np.asarray(records.info_gained, dtype=float).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, useful / total, np.nan)


def junk_info_share(records: RunRecords) -> np.ndarray:
    """Доля мусора в информационном потоке прогона (дополнение к 1, H6)."""
    useful = np.asarray(records.useful_gained, dtype=float).sum(axis=1)
    total = np.asarray(records.info_gained, dtype=float).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (total - useful) / total, np.nan)


def optimal_arm_indicator(records: RunRecords) -> np.ndarray:
    """Пошаговый индикатор «выбрана лучшая рука» (nan на мета-шагах)."""
    mask = arm_pull_mask(records)
    hit = (records.actions == records.best_arm) & mask
    return np.where(mask, hit.astype(float), np.nan)


def optimal_arm_rate(records: RunRecords) -> np.ndarray:
    mask = arm_pull_mask(records)
    hits = ((records.actions == records.best_arm) & mask).sum(axis=1)
    pulls = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pulls > 0, hits / pulls, np.nan)


def points_per_pull(records: RunRecords) -> np.ndarray:
    pulls = arm_pull_mask(records).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pulls > 0, final_points(records) / pulls, np.nan)


def normalized_auc(records: RunRecords) -> np.ndarray:
    """Площадь под кривой очков, нормированная на прямоугольник ``T · итог``."""
    points = cumulative_points(records)
    final = points[:, -1]
    t = np.arange(points.shape[1], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        area = np.trapezoid(points, t, axis=1)
        return np.where(final > 0, area / (points.shape[1] * final), np.nan)


def _first_step(reached: np.ndarray) -> np.ndarray:
    any_reached = reached.any(axis=1)
    steps = np.argmax(reached, axis=1) + 1
    out = steps.astype(float)
    out[~any_reached] = np.nan
    return out


def steps_to_fraction(
    records: RunRecords, reference: Sequence[float], fraction: float
) -> np.ndarray:
    """Первый шаг (1-based), где очки достигли ``fraction · reference``."""
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if reference.size != records.n_runs:
        raise ValueError(
            f"reference должен иметь длину n_runs={records.n_runs}, получено {reference.size}"
        )
    target = float(fraction) * reference
    reached = cumulative_points(records) >= target[:, None]
    return _first_step(reached)


def steps_to_epoch(records: RunRecords, epoch: int) -> np.ndarray:
    reached = np.asarray(records.epoch_after) >= int(epoch)
    return _first_step(reached)


def efficiency(records: RunRecords, oracle_points: Sequence[float]) -> np.ndarray:
    oracle_points = np.asarray(oracle_points, dtype=float).reshape(-1)
    if oracle_points.size != records.n_runs:
        raise ValueError(
            f"oracle_points должен иметь длину n_runs={records.n_runs}, "
            f"получено {oracle_points.size}"
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(oracle_points > 0, final_points(records) / oracle_points, np.nan)


def point_regret(records: RunRecords, oracle_points: Sequence[float]) -> np.ndarray:
    """Пошаговый недобор относительно траектории оракула."""
    oracle = np.asarray(oracle_points, dtype=float)
    if oracle.ndim == 1:
        oracle = oracle[:, None]
    if oracle.shape[0] != records.n_runs:
        raise ValueError("oracle_points должен иметь форму (n_runs,) или (n_runs, n_steps)")
    return oracle - cumulative_points(records)


def final_epoch(records: RunRecords) -> np.ndarray:
    return np.asarray(records.epoch_after[:, -1], dtype=float)


def final_n_arms(records: RunRecords) -> np.ndarray:
    return np.asarray(records.n_arms_after[:, -1], dtype=float)


def final_multiplier(records: RunRecords) -> np.ndarray:
    return np.asarray(records.multiplier_after[:, -1], dtype=float)


def points_after_epoch(records: RunRecords, epoch: int = 1) -> np.ndarray:
    """Очки, заработанные *после* выхода в ``epoch`` (nan — эпоха не достигнута).

    Метрика для H4: сравнивать алгоритмы «после смены состава лучших рук»
    честнее по очкам, накопленным уже в новой эпохе, чем по итогу, где
    доминирует ранний участок.
    """
    reached = np.asarray(records.epoch_after) >= int(epoch)
    index = np.argmax(reached, axis=1)
    points = cumulative_points(records)
    before = np.take_along_axis(points, np.maximum(index - 1, 0)[:, None], axis=1)[:, 0]
    before = np.where(index > 0, before, 0.0)
    out = points[:, -1] - before
    out[~reached.any(axis=1)] = np.nan
    return out


def summarize(
    records: RunRecords,
    oracle_final: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Все скалярные метрики одного прогона (по массиву на прогон)."""
    final = final_points(records)
    metrics: Dict[str, np.ndarray] = {
        "final_points": final,
        "cumulative_arm_regret": total_arm_regret(records),
        "mean_arm_regret": mean_arm_regret(records),
        "optimal_arm_rate": optimal_arm_rate(records),
        "arm_pull_share": arm_pull_share(records),
        "meta_step_share": meta_step_share(records),
        "points_per_pull": points_per_pull(records),
        "auc_normalized": normalized_auc(records),
        "steps_to_50pct": steps_to_fraction(records, final, 0.5),
        "steps_to_90pct": steps_to_fraction(records, final, 0.9),
        "steps_to_epoch_1": steps_to_epoch(records, 1),
        "steps_to_epoch_2": steps_to_epoch(records, 2),
        "final_epoch": final_epoch(records),
        "final_n_arms": final_n_arms(records),
        "final_multiplier": final_multiplier(records),
        "points_after_epoch_1": points_after_epoch(records, 1),
        "useful_info_share": useful_info_share(records),
        "junk_info_share": junk_info_share(records),
    }
    # Бюджет ветвей метацикла: счётчики действий и их вклад в множитель.
    metrics.update(meta_action_counts(records))
    metrics.update(multiplier_budget(records))
    if oracle_final is not None:
        metrics["efficiency"] = efficiency(records, oracle_final)
    return metrics


def aggregate(values: Sequence[float]) -> Dict[str, float]:
    """Среднее, стандартная ошибка и 95% ДИ (нормальное приближение).

    Полноценные интервалы (bootstrap) и тесты — в ``bandit.stats`` (P3);
    здесь нужна только быстрая сводка для таблиц.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    n = finite.size
    if n == 0:
        return {"mean": float("nan"), "sem": float("nan"), "ci95_lo": float("nan"),
                "ci95_hi": float("nan"), "n": 0.0}
    mean = float(finite.mean())
    if n == 1:
        return {"mean": mean, "sem": float("nan"), "ci95_lo": float("nan"),
                "ci95_hi": float("nan"), "n": 1.0}
    sem = float(finite.std(ddof=1) / np.sqrt(n))
    return {
        "mean": mean,
        "sem": sem,
        "ci95_lo": mean - 1.96 * sem,
        "ci95_hi": mean + 1.96 * sem,
        "n": float(n),
    }
