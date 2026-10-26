"""Отчёт по стадиям проверки метацикла: ``advantage.md`` + ``advantage.csv``.

Скрипт читает каталоги, созданные ``scripts/advantage_sweep.py`` (``*.npz`` и
манифест от ``bandit.cli run``), и отвечает на вопросы, поставленные к
метациклу:

* **ветви** — очки, Δ к политике ``none`` в том же (ось, уровень, агент),
  Wilcoxon со знаковыми рангами + поправка Холма, ``dz``, счёт побед, КПД
  относительно оракула;
* **бюджет** — сколько шагов ушло в метацикл, сколько было конверсий, покупок
  рук, ворот и амплификаторов и какой вклад каждая ветвь дала в множитель
  (сумма компонент равна ``final_multiplier − 1``, см. ``bandit.metrics``);
* **профиль по горизонтам** — Δ(t) на сетке горизонтов и момент t*, после
  которого знак Δ больше не меняется (выигрыш «поздний» или «ранний»);
* **вердикты** — проверки M1..M8 со статусом и числами, на которых он получен.

Запуск::

    python scripts/advantage_report.py
    python scripts/advantage_report.py results/adv-s2 --out results/adv-s2.md
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bandit import metrics as m  # noqa: E402
from bandit import stats as st  # noqa: E402
from bandit.cli import _oracle_finals  # noqa: E402  единая логика поиска оракула
from bandit.runner import RunRecords  # noqa: E402

ADV_PREFIX = "pyramid-adv"
"""Префикс имён стадий (см. ``scripts/advantage_sweep.py``)."""

REFERENCE_POLICY = "none"
"""Политика-базлайн внутри группы: без метацикла агент только тянет руки."""

PRIMARY_POLICY = "full_cycle"
"""Политика, для которой считаются t* и профиль по горизонтам."""

DEFAULT_DIRS: Tuple[str, ...] = ("adv-s1", "adv-s2", "adv-s6")
"""Каталоги стадий по умолчанию (несуществующие пропускаются)."""

DEFAULT_HORIZONS: Tuple[int, ...] = (1000, 3000, 5000, 10_000)
"""Сетка горизонтов для профиля Δ(t)."""


@dataclass(frozen=True)
class ConfigRow:
    """Один конфиг стадии: разобранное имя и все метрики по прогонам."""

    directory: Path
    stage: str
    name: str
    axis: str
    level: str
    agent: str
    policy: str
    n_steps: int
    points: np.ndarray
    values: Dict[str, np.ndarray]

    def mean(self, metric: str) -> float:
        """Среднее метрики (nan, если её нет)."""
        array = self.values.get(metric)
        if array is None:
            return float("nan")
        finite = np.asarray(array, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(finite.mean()) if finite.size else float("nan")


@dataclass
class Node:
    """Строка анализа: конфиг + его сравнение с базлайном группы."""

    row: ConfigRow
    reference: Optional[str] = None
    delta: Optional[Dict[str, float]] = None
    p_adjusted: float = float("nan")
    t_star: float = float("nan")
    curve: Optional[np.ndarray] = None
    diffs: Optional[np.ndarray] = None

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_adjusted) and self.p_adjusted < 0.05)


def parse_config_name(name: str) -> Tuple[str, str, str, str]:
    """Разобрать имя на ``(ось, уровень, агент, политика)``.

    Стадийные имена имеют вид ``pyramid-adv-<ось>-<уровень>-<агент>-<политика>``
    (уровни без дефисов). Для прочих конфигов осью считается всё, что стоит
    перед агентом: так отчёт не падает на обычных ``pyramid-*`` прогонах.
    """
    parts = name.split("-")
    if len(parts) >= 6 and parts[0] == "pyramid" and parts[1] == "adv":
        return parts[2], parts[3], parts[4], "-".join(parts[5:])
    if len(parts) >= 3:
        return "-".join(parts[:-2]), "", parts[-2], parts[-1]
    return name, "", "", ""


def stage_key(directory: Path) -> str:
    """Ключ стадии из имени каталога: ``adv-s2`` → ``s2``."""
    name = directory.name
    return name[4:] if name.startswith("adv-") else name


def load_rows(directory: Path) -> List[ConfigRow]:
    """Прочитать все прогоны каталога и посчитать метрики с КПД."""
    names = sorted(path.stem for path in directory.glob("*.npz"))
    if not names:
        raise FileNotFoundError(f"В {directory} нет сохранённых прогонов (*.npz)")
    records: Dict[str, RunRecords] = {
        name: RunRecords.load(directory, name) for name in names
    }
    summaries = {
        name: m.summarize(item, oracle_final=_oracle_finals(records, item))
        for name, item in records.items()
    }
    stage = stage_key(directory)
    rows: List[ConfigRow] = []
    for name in names:
        axis, level, agent, policy = parse_config_name(name)
        rows.append(
            ConfigRow(
                directory=directory,
                stage=stage,
                name=name,
                axis=axis,
                level=level,
                agent=agent,
                policy=policy,
                n_steps=int(records[name].n_steps),
                points=np.asarray(records[name].points, dtype=float),
                values=summaries[name],
            )
        )
    return rows


def group_key(row: ConfigRow) -> Tuple[str, str, str, str]:
    """Группа сравнения: стадия, ось, уровень, агент."""
    return row.stage, row.axis, row.level, row.agent


def first_stable_positive(curve: Sequence[float]) -> float:
    """Первый шаг (1-based), начиная с которого Δ(t) > 0 до конца горизонта.

    ``nan`` — преимущество так и не стало устойчивым (или исчезло к концу):
    именно этот случай делает вывод «метацикл полезен» условным.
    """
    values = np.asarray(curve, dtype=float)
    positive = values > 0
    if values.size == 0 or not positive[-1]:
        return float("nan")
    non_positive = np.flatnonzero(~positive)
    if non_positive.size == 0:
        return 1.0
    return float(non_positive[-1] + 2)


def build_nodes(
    rows: Sequence[ConfigRow],
    *,
    method: str = "wilcoxon",
    n_boot: int = st.DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = st.DEFAULT_SEED,
) -> List[Node]:
    """Сравнить каждый конфиг с ``none`` в его группе (парные сиды).

    Группа — (стадия, ось, уровень, агент): сравнивать разные уровни между
    собой нельзя, у них разные среды. p-value корректируются Холмом по всем
    политикам группы, поэтому таблица читается как семейство сравнений.
    """
    groups: Dict[Tuple[str, str, str, str], List[ConfigRow]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    nodes: List[Node] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.policy)
        references = [row for row in group if row.policy == REFERENCE_POLICY]
        reference = references[0] if references else None
        group_nodes: List[Node] = []
        pvalues: Dict[str, float] = {}
        for row in group:
            node = Node(row=row, reference=reference.name if reference else None)
            if reference is not None:
                stats = st.paired_comparison(
                    row.values["final_points"],
                    reference.values["final_points"],
                    method=method,
                    n_boot=n_boot,
                    alpha=alpha,
                    seed=seed,
                )
                node.delta = stats
                node.diffs = row.points - reference.points
                node.curve = np.nanmean(node.diffs, axis=0)
                node.t_star = first_stable_positive(node.curve)
                if row.policy != REFERENCE_POLICY:
                    pvalues[row.name] = float(stats["p_value"])
            group_nodes.append(node)
        adjusted = st.holm_dict(pvalues) if pvalues else {}
        for node in group_nodes:
            node.p_adjusted = float(adjusted.get(node.row.name, float("nan")))
        nodes.extend(group_nodes)
    return nodes


def horizon_delta(node: Node, horizon: int) -> Tuple[float, float]:
    """Δ(t) и его стандартная ошибка в точке ``horizon`` (nan, если t > T)."""
    if node.diffs is None:
        return float("nan"), float("nan")
    index = int(horizon) - 1
    if index < 0 or index >= node.diffs.shape[1]:
        return float("nan"), float("nan")
    column = node.diffs[:, index]
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    sem = float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else float("nan")
    return mean, sem


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


def _fmt_int(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.1f}"


#: Статусы вердиктов: строка попадает в markdown как есть.
STATUS_YES = "да"
STATUS_NO = "нет"
STATUS_NA = "н/д"


@dataclass(frozen=True)
class Verdict:
    """Один вердикт: код, вопрос, статус и числа, на которых он получен."""

    code: str
    question: str
    status: str
    detail: str


def _status(condition: Optional[bool]) -> str:
    if condition is None:
        return STATUS_NA
    return STATUS_YES if condition else STATUS_NO


def _delta_text(node: Optional[Node]) -> str:
    """Короткая запись парного сравнения для колонки «числа»."""
    if node is None or node.delta is None:
        return "нет данных"
    stats = node.delta
    wins, losses = int(stats["wins"]), int(stats["losses"])
    return (
        f"Δ={_fmt(stats['diff'], 1)} [{_fmt(stats['ci_lo'], 1)}; {_fmt(stats['ci_hi'], 1)}], "
        f"p_Холм={_fmt_p(node.p_adjusted)}, побед {wins}/{wins + losses}"
    )


def _diff(node: Optional[Node]) -> float:
    if node is None or node.delta is None:
        return float("nan")
    return float(node.delta["diff"])


def _mean(node: Optional[Node], metric: str) -> float:
    return float("nan") if node is None else node.row.mean(metric)


def _verdict_m1(nodes: Sequence[Node]) -> Verdict:
    full = find_node(nodes, "s2", "abl", "base", PRIMARY_POLICY, agent="ucb1")
    condition: Optional[bool] = None
    if full is not None and full.delta is not None:
        condition = bool(_diff(full) > 0 and full.p_adjusted < 0.05)
    return Verdict(
        code="M1",
        question="Полный метацикл даёт больше очков, чем none",
        status=_status(condition),
        detail=_delta_text(full),
    )


def _verdict_m2(nodes: Sequence[Node]) -> Verdict:
    """Метацикл выгоден даже при запрете покупки рук (``cap5``).

    ``gate_only``/``amp_only`` в этой среде тоже покупают руки, поэтому решает
    именно ``cap5``: там ``max_arms`` = ``n_arms_init`` и покупка невозможна —
    остаётся только конверсия, ворота и амплификаторы.
    """
    gate_only = find_node(nodes, "s2", "abl", "base", "gate_only", agent="ucb1")
    cap5 = find_node(nodes, "s1", "cap", "cap5", PRIMARY_POLICY, agent="ucb1")
    condition: Optional[bool] = None
    if cap5 is not None and cap5.delta is not None:
        condition = bool(_diff(cap5) > 0 and cap5.p_adjusted < 0.05)
    return Verdict(
        code="M2",
        question="Выигрыш есть и без покупки рук (запрет на cap5)",
        status=_status(condition),
        detail=f"cap5: {_delta_text(cap5)}; gate_only (с покупками): {_delta_text(gate_only)}",
    )


def _verdict_m3(nodes: Sequence[Node]) -> Verdict:
    """Амплификаторы — побочная ветка: слабее ворот и по бюджету, и по абляции."""
    full = find_node(nodes, "s2", "abl", "base", PRIMARY_POLICY, agent="ucb1")
    gate_only = find_node(nodes, "s2", "abl", "base", "gate_only", agent="ucb1")
    amp_only = find_node(nodes, "s2", "abl", "base", "amp_only", agent="ucb1")
    budget_gates = _mean(full, "multiplier_from_gates")
    budget_amps = _mean(full, "multiplier_from_amplifiers")
    ablation_ok: Optional[bool] = None
    if gate_only is not None and amp_only is not None:
        if gate_only.delta is not None and amp_only.delta is not None:
            ablation_ok = bool(_diff(amp_only) < _diff(gate_only))
    budget_ok: Optional[bool] = None
    if np.isfinite(budget_gates) and np.isfinite(budget_amps):
        budget_ok = bool(budget_amps < budget_gates)
    if ablation_ok is None and budget_ok is None:
        status = STATUS_NA
    else:
        status = _status(bool(ablation_ok is not False and budget_ok is not False))
    return Verdict(
        code="M3",
        question="Амплификаторы слабее ворот (побочная ветка)",
        status=status,
        detail=(
            f"amp_only: {_delta_text(amp_only)}; gate_only: {_delta_text(gate_only)}; "
            f"вклад в множитель: ампы {_fmt(budget_amps, 2)}, ворота {_fmt(budget_gates, 2)}"
        ),
    )


def _verdict_m4(nodes: Sequence[Node]) -> Verdict:
    """Расписание бонуса за руку: усиливает выигрыш или создаёт его?

    Проверяется монотонность Δ по уровням ``arm_bonus_increase`` и то, что при
    нуле выигрыш не исчезает. Квадратичность заранее не предполагается: бонус
    за руку растёт как ``arm_bonus + arm_bonus_increase · buys``, но число
    покупок ограничено бюджетом множителей, поэтому фактический рост Δ может
    быть медленнее квадратичного.
    """
    inc0 = find_node(nodes, "s1", "engine", "inc0", PRIMARY_POLICY, agent="ucb1")
    inc01 = find_node(nodes, "s1", "engine", "inc01", PRIMARY_POLICY, agent="ucb1")
    inc02 = find_node(nodes, "s1", "engine", "inc02", PRIMARY_POLICY, agent="ucb1")
    uniform = [_diff(inc0), _diff(inc01), _diff(inc02)]
    base = uniform[1]
    ratio = uniform[2] / base if np.isfinite(base) and base > 0 else float("nan")
    flat = uniform[0] / base if np.isfinite(base) and base > 0 else float("nan")
    condition: Optional[bool] = None
    if all(np.isfinite(value) for value in uniform):
        condition = bool(uniform[0] > 0 and uniform[0] < uniform[1] < uniform[2])
    return Verdict(
        code="M4",
        question="Расписание бонуса за руку усиливает выигрыш, но не создаёт его",
        status=_status(condition),
        detail=(
            f"Δ(inc0)={_fmt(uniform[0], 1)}, Δ(inc01)={_fmt(uniform[1], 1)}, "
            f"Δ(inc02)={_fmt(uniform[2], 1)}; "
            f"Δ(inc02)/Δ(inc01)={_fmt(ratio)}, Δ(inc0)/Δ(inc01)={_fmt(flat)}"
        ),
    )


def _verdict_m5(nodes: Sequence[Node]) -> Verdict:
    """Выигрыш поздний: к 10% горизонта набрана малая доля итогового Δ."""
    long_full = find_node(nodes, "s6", "long", "t10k", PRIMARY_POLICY, agent="ucb1")
    auc = _mean(long_full, "auc_normalized")
    early_share = float("nan")
    condition: Optional[bool] = None
    if long_full is not None and long_full.diffs is not None:
        horizon = long_full.row.n_steps
        early, _ = horizon_delta(long_full, max(1, horizon // 10))
        total, _ = horizon_delta(long_full, horizon)
        if np.isfinite(total) and total > 0:
            early_share = early / total
            condition = bool(early_share < 0.25 and np.isfinite(long_full.t_star))
    return Verdict(
        code="M5",
        question="Преимущество накапливается с горизонтом (выигрыш поздний)",
        status=_status(condition),
        detail=(
            f"Δ(10%T)/Δ(T)={_fmt(early_share, 3)}, "
            f"t*={_fmt(long_full.t_star if long_full else float('nan'), 0)}, "
            f"AUC={_fmt(auc, 3)}, 50%={_fmt(_mean(long_full, 'steps_to_50pct'), 0)}, "
            f"90%={_fmt(_mean(long_full, 'steps_to_90pct'), 0)}"
        ),
    )


def _verdict_m6(nodes: Sequence[Node]) -> Verdict:
    """Преимущество не зависит от силы агента."""
    greedy = find_node(nodes, "s6", "agent", "greedy", PRIMARY_POLICY, agent="greedy")
    random = find_node(nodes, "s6", "agent", "random", PRIMARY_POLICY, agent="random")
    checks: List[Optional[bool]] = []
    for node in (greedy, random):
        checks.append(None if node is None or node.delta is None else bool(_diff(node) > 0))
    if all(item is None for item in checks):
        status = STATUS_NA
    else:
        status = _status(bool(all(bool(item) for item in checks)))
    return Verdict(
        code="M6",
        question="Выигрыш сохраняется у слабых агентов",
        status=status,
        detail=f"greedy: {_delta_text(greedy)}; random: {_delta_text(random)}",
    )


def _verdict_m7(nodes: Sequence[Node]) -> Verdict:
    """До оракула далеко: экономика остаётся узким местом."""
    full = find_node(nodes, "s2", "abl", "base", PRIMARY_POLICY, agent="ucb1")
    eff = _mean(full, "efficiency")
    condition: Optional[bool] = None
    if np.isfinite(eff):
        condition = bool(eff < 1.0)
    return Verdict(
        code="M7",
        question="КПД относительно оракула ниже единицы",
        status=_status(condition),
        detail=f"КПД={_fmt(eff, 3)}; до верхней границы остаётся запас",
    )


def _verdict_m8(nodes: Sequence[Node]) -> Verdict:
    """Основной выигрыш даёт покупка рук: без неё Δ падает в разы."""
    cap5 = find_node(nodes, "s1", "cap", "cap5", PRIMARY_POLICY, agent="ucb1")
    cap50 = find_node(nodes, "s1", "cap", "cap50", PRIMARY_POLICY, agent="ucb1")
    full = find_node(nodes, "s2", "abl", "base", PRIMARY_POLICY, agent="ucb1")
    mult_only = find_node(nodes, "s2", "abl", "base", "mult_only", agent="ucb1")
    base = _diff(cap50)
    share = _diff(cap5) / base if np.isfinite(base) and base > 0 else float("nan")
    condition: Optional[bool] = None
    if np.isfinite(share):
        condition = bool(share < 0.25)
    full_points = float(full.delta["mean"]) if full is not None and full.delta else float("nan")
    mult_points = (
        float(mult_only.delta["mean"])
        if mult_only is not None and mult_only.delta
        else float("nan")
    )
    return Verdict(
        code="M8",
        question="Основной выигрыш даёт покупка рук",
        status=_status(condition),
        detail=(
            f"Δ(cap5)={_fmt(_diff(cap5), 1)}, Δ(cap50)={_fmt(base, 1)}, "
            f"доля без покупок={_fmt(share, 2)}; очки mult_only/full_cycle="
            f"{_fmt(mult_points, 0)}/{_fmt(full_points, 0)}"
        ),
    )


def compute_verdicts(nodes: Sequence[Node]) -> List[Verdict]:
    """Проверки M1..M8 по уже посчитанным сравнениям.

    Коды привязаны к ячейкам стадий из ``scripts/advantage_sweep.py``; если
    стадии в отчёте нет, вердикт получает статус «н/д» и не притворяется
    результатом.
    """
    return [
        _verdict_m1(nodes),
        _verdict_m2(nodes),
        _verdict_m3(nodes),
        _verdict_m4(nodes),
        _verdict_m5(nodes),
        _verdict_m6(nodes),
        _verdict_m7(nodes),
        _verdict_m8(nodes),
    ]


def find_node(
    nodes: Iterable[Node],
    stage: str,
    axis: str,
    level: str,
    policy: str,
    agent: Optional[str] = None,
) -> Optional[Node]:
    """Найти строку анализа по стадии, оси, уровню и политике."""
    for node in nodes:
        row = node.row
        if row.stage != stage or row.axis != axis or row.level != level:
            continue
        if row.policy != policy:
            continue
        if agent is not None and row.agent != agent:
            continue
        return node
    return None


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _branch_rows(nodes: Sequence[Node]) -> List[List[str]]:
    """Таблица «очки и Δ к none»."""
    rows: List[List[str]] = []
    for node in nodes:
        row = node.row
        stats = node.delta
        mean = float(stats["mean"]) if stats else row.mean("final_points")
        ref_mean = float(stats["reference_mean"]) if stats else float("nan")
        ratio = mean / ref_mean if np.isfinite(ref_mean) and ref_mean else float("nan")
        interval = (
            f"[{_fmt(stats['ci_lo'], 1)}; {_fmt(stats['ci_hi'], 1)}]" if stats else "—"
        )
        rows.append(
            [
                row.stage,
                row.axis,
                row.level,
                row.agent,
                row.policy,
                _fmt(mean, 0),
                _fmt(_diff(node), 1),
                interval,
                _fmt_p(node.p_adjusted),
                _fmt(float(stats["effect_size"]) if stats else float("nan"), 2),
                f"{int(stats['wins'])}/{int(stats['losses'])}" if stats else "—",
                _fmt(ratio, 2) if np.isfinite(ratio) else "—",
                _fmt(row.mean("efficiency"), 3),
            ]
        )
    return rows


def _budget_rows(nodes: Sequence[Node]) -> List[List[str]]:
    """Таблица «бюджет ветвей»: шаги, действия и вклад в множитель."""
    rows: List[List[str]] = []
    for node in nodes:
        row = node.row
        rows.append(
            [
                row.stage,
                row.axis,
                row.level,
                row.agent,
                row.policy,
                _fmt(row.mean("meta_step_share"), 3),
                _fmt(row.mean("conversions"), 1),
                _fmt(row.mean("arm_buys"), 1),
                _fmt(row.mean("gate_crafts"), 1),
                _fmt(row.mean("amplifier_crafts"), 1),
                _fmt(row.mean("final_multiplier"), 2),
                _fmt(row.mean("multiplier_from_conversions"), 2),
                _fmt(row.mean("multiplier_from_arms"), 2),
                _fmt(row.mean("multiplier_from_gates"), 2),
                _fmt(row.mean("multiplier_from_amplifiers"), 2),
            ]
        )
    return rows


def _profile_rows(nodes: Sequence[Node], horizons: Sequence[int]) -> List[List[str]]:
    """Таблица «профиль по горизонтам» для политик с Δ(t)."""
    rows: List[List[str]] = []
    for node in nodes:
        if node.diffs is None:
            continue
        row = node.row
        cells = [
            row.stage,
            row.axis,
            row.level,
            row.agent,
            row.policy,
            _fmt(node.t_star, 0),
            _fmt(row.mean("auc_normalized"), 3),
            _fmt(row.mean("steps_to_50pct"), 0),
            _fmt(row.mean("steps_to_90pct"), 0),
        ]
        for horizon in horizons:
            if row.n_steps < horizon:
                cells.append("—")
                continue
            mean, sem = horizon_delta(node, horizon)
            if not np.isfinite(mean):
                cells.append("—")
            elif np.isfinite(sem):
                cells.append(f"{_fmt(mean, 1)}±{_fmt(sem, 1)}")
            else:
                cells.append(_fmt(mean, 1))
        rows.append(cells)
    return rows


def build_markdown(
    nodes: Sequence[Node],
    verdicts: Sequence[Verdict],
    sources: Sequence[Path],
    horizons: Sequence[int],
) -> str:
    """Собрать ``advantage.md``: ветви, бюджет, профиль, вердикты, как читать."""
    stages = sorted({node.row.stage for node in nodes})
    n_runs = max(
        (
            int(np.isfinite(np.asarray(node.row.values["final_points"], dtype=float)).sum())
            for node in nodes
        ),
        default=0,
    )
    lines: List[str] = [
        "# Преимущество метацикла: стадии S0/S1/S2/S6",
        "",
        "Δ — парная разность итоговых очков относительно политики `none` в той же "
        "среде, с теми же сидами и горизонтом (знаковые ранги Уилкоксона, поправка "
        "Холма по семейству политик группы). «×» — отношение средних, КПД — "
        "очки / очки оракула при тех же сидах.",
        "",
        f"Источники: {', '.join(f'`{path.name}`' for path in sources)}; "
        f"стадии: {', '.join(stages)}; прогонов в конфиге: до {n_runs}.",
        "",
        "## 1. Ветви: очки и Δ к `none`",
        "",
        _table(
            [
                "стадия", "ось", "уровень", "агент", "политика", "очки", "Δ",
                "95% ДИ", "p Холм", "dz", "+/−", "×", "КПД",
            ],
            _branch_rows(nodes),
        ),
        "",
        "## 2. Бюджет ветвей",
        "",
        "Вклад в множитель — изменение множителя на шагах данной ветви. Сумма "
        "четырёх вкладов равна `final_multiplier − 1`, поэтому таблица показывает, "
        "из чего собран выигрыш, а не только его величину.",
        "",
        _table(
            [
                "стадия", "ось", "уровень", "агент", "политика", "мета-шаги",
                "конверсии", "руки", "ворота", "ампы", "множитель", "из конверсий",
                "из рук", "из ворот", "из ампов",
            ],
            _budget_rows(nodes),
        ),
        "",
        "## 3. Профиль по горизонтам",
        "",
        "`t*` — первый шаг, начиная с которого Δ(t) > 0 до конца горизонта (`—` — "
        "устойчивого преимущества нет). AUC нормирован на `T · итог`: 0.5 — "
        "линейный рост, меньше — выигрыш приходит позже.",
        "",
        _table(
            ["стадия", "ось", "уровень", "агент", "политика", "t*", "AUC", "50%", "90%"]
            + [f"Δ@{horizon}" for horizon in horizons],
            _profile_rows(nodes, horizons),
        ),
        "",
        "## 4. Вердикты",
        "",
        _table(
            ["код", "проверка", "статус", "числа"],
            [[item.code, item.question, item.status, item.detail] for item in verdicts],
        ),
        "",
        "## 5. Как читать",
        "",
        "* `M4` проверяет не «квадратичность», а роль расписания: бонус за руку "
        "равен `arm_bonus + arm_bonus_increase · buys`, но число покупок "
        "ограничено бюджетом множителей. Поэтому Δ растёт с "
        "`arm_bonus_increase` медленнее, чем квадратично, и остаётся "
        "положительным даже при нуле: расписание усиливает выигрыш, но не "
        "создаёт его.",
        "* `M7` напоминает, что даже полный цикл не дотягивает до оракула: запас "
        "КПД — это и есть цена метацикла в очках.",
        "* Бюджет ветвей — путевая величина, а не причинная: покупка руки "
        "списывает множители и тем самым «сбрасывает» логарифмический бонус "
        "конверсий, поэтому вклады конверсий и рук переплетены. Сумма всегда "
        "равна `final_multiplier − 1`, но причинный вывод даёт абляция "
        "(`mult_only` против `full_cycle`), а не доли бюджета.",
        "* `amp` и `amp_only` в этой среде совпадают по порядку действий "
        "(амплификатор → конверсия → покупка), поэтому их Δ равны: это проверка "
        "консистентности, а не два независимых свидетельства.",
        "* Вклад покупок рук в множитель может быть отрицательным: каждая покупка "
        "списывает множители и сбрасывает логарифмический бонус конверсий. При "
        "`arm_bonus_increase = 0` фиксированной прибавки 0.5 не хватает, и ветвь "
        "рук становится убыточной (в бюджете `s1/engine/inc0` вклад рук < 0).",
        "* Сравнения идут внутри (ось, уровень, агент): уровни меняют среду, поэтому "
        "переносить вывод между уровнями нельзя.",
        "",
    ]
    return "\n".join(lines)


#: Колонки ``advantage.csv``: по строке на конфиг, всё числом (для Excel/pandas).
BASE_CSV_COLUMNS: Tuple[str, ...] = (
    "stage",
    "axis",
    "level",
    "agent",
    "policy",
    "config",
    "n_steps",
    "n_runs",
    "points_mean",
    "reference_mean",
    "ratio",
    "delta",
    "ci_lo",
    "ci_hi",
    "p_value",
    "p_holm",
    "dz",
    "wins",
    "losses",
    "efficiency",
    "arm_pull_share",
    "meta_step_share",
    "conversions",
    "arm_buys",
    "gate_crafts",
    "amplifier_crafts",
    "final_multiplier",
    "multiplier_from_conversions",
    "multiplier_from_arms",
    "multiplier_from_gates",
    "multiplier_from_amplifiers",
    "auc_normalized",
    "t_star",
    "steps_to_50pct",
    "steps_to_90pct",
)

PROFILE_COLUMNS: Tuple[str, ...] = (
    "stage",
    "axis",
    "level",
    "agent",
    "policy",
    "horizon",
    "delta",
    "sem",
)


def _num(value: float) -> str:
    """Число для CSV: ``nan`` остаётся явным, чтобы пропуски было видно."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{number:.6g}"


def _row_values(node: Node) -> Dict[str, str]:
    """Строка CSV: сравнение и метрики конфига в одном словаре."""
    row = node.row
    stats = node.delta or {}
    mean = float(stats["mean"]) if stats else row.mean("final_points")
    ref_mean = float(stats["reference_mean"]) if stats else float("nan")
    ratio = mean / ref_mean if np.isfinite(ref_mean) and ref_mean else float("nan")
    finite = np.asarray(row.values["final_points"], dtype=float)
    values: Dict[str, str] = {
        "stage": row.stage,
        "axis": row.axis,
        "level": row.level,
        "agent": row.agent,
        "policy": row.policy,
        "config": row.name,
        "n_steps": str(row.n_steps),
        "n_runs": str(int(np.isfinite(finite).sum())),
        "points_mean": _num(mean),
        "reference_mean": _num(ref_mean),
        "ratio": _num(ratio),
        "delta": _num(stats.get("diff", float("nan"))),
        "ci_lo": _num(stats.get("ci_lo", float("nan"))),
        "ci_hi": _num(stats.get("ci_hi", float("nan"))),
        "p_value": _num(stats.get("p_value", float("nan"))),
        "p_holm": _num(node.p_adjusted),
        "dz": _num(stats.get("effect_size", float("nan"))),
        "wins": str(int(stats["wins"])) if stats else "0",
        "losses": str(int(stats["losses"])) if stats else "0",
        "t_star": _num(node.t_star),
    }
    for metric in BASE_CSV_COLUMNS:
        if metric not in values:
            values[metric] = _num(row.mean(metric))
    return values


def write_csv(path: Path, nodes: Sequence[Node], horizons: Sequence[int]) -> Path:
    """Сохранить по строке на конфиг (включая Δ на сетке горизонтов)."""
    columns = list(BASE_CSV_COLUMNS) + [f"delta@{horizon}" for horizon in horizons]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for node in nodes:
            values = _row_values(node)
            for horizon in horizons:
                mean, _ = horizon_delta(node, horizon)
                values[f"delta@{horizon}"] = _num(mean)
            writer.writerow({name: values.get(name, "nan") for name in columns})
    return path


def write_profile_csv(
    path: Path, nodes: Sequence[Node], horizons: Sequence[int]
) -> Path:
    """Δ(t) на сетке горизонтов: по строке на (конфиг, горизонт)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROFILE_COLUMNS))
        writer.writeheader()
        for node in nodes:
            if node.diffs is None:
                continue
            row = node.row
            for horizon in horizons:
                if row.n_steps < horizon:
                    continue
                mean, sem = horizon_delta(node, horizon)
                writer.writerow(
                    {
                        "stage": row.stage,
                        "axis": row.axis,
                        "level": row.level,
                        "agent": row.agent,
                        "policy": row.policy,
                        "horizon": str(horizon),
                        "delta": _num(mean),
                        "sem": _num(sem),
                    }
                )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Отчёт по стадиям проверки преимущества метацикла"
    )
    parser.add_argument("dirs", nargs="*", type=Path, help="каталоги стадий")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "advantage.md", help="путь к .md"
    )
    parser.add_argument("--method", choices=list(st.METHODS), default="wilcoxon")
    parser.add_argument("--n-boot", type=int, default=st.DEFAULT_N_BOOT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=st.DEFAULT_SEED)
    parser.add_argument(
        "--horizons",
        default=",".join(str(item) for item in DEFAULT_HORIZONS),
        help="сетка горизонтов через запятую",
    )
    parser.add_argument("--quiet", action="store_true", help="не печатать markdown")
    args = parser.parse_args(argv)

    directories = [Path(item) for item in args.dirs] or [
        ROOT / "results" / name for name in DEFAULT_DIRS
    ]
    available = [item for item in directories if any(item.glob("*.npz"))]
    for item in directories:
        if item not in available:
            print(f"Пропуск {item}: нет сохранённых прогонов", file=sys.stderr)
    if not available:
        print("Нет данных: сначала запустите scripts/advantage_sweep.py", file=sys.stderr)
        return 2

    rows: List[ConfigRow] = []
    for directory in available:
        rows.extend(load_rows(directory))
    nodes = build_nodes(
        rows,
        method=args.method,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )
    verdicts = compute_verdicts(nodes)
    horizons = sorted(
        {int(part) for part in str(args.horizons).split(",") if part.strip()}
    )

    out = Path(args.out)
    markdown = build_markdown(nodes, verdicts, available, horizons)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    csv_path = write_csv(out.with_suffix(".csv"), nodes, horizons)
    profile_path = write_profile_csv(
        out.with_name(f"{out.stem}_profile.csv"), nodes, horizons
    )

    summary = ", ".join(f"{item.code}: {item.status}" for item in verdicts)
    print(f"Вердикты — {summary}")
    if not args.quiet:
        print()
        print(markdown)
    print(f"\nОтчёт: {out.resolve()}")
    print(f"Таблица: {csv_path.resolve()}")
    print(f"Профиль: {profile_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





