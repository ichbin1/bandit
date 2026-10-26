"""Сводный отчёт по сохранённому прогону (P5).

Что делает
----------
``build_report(directory)`` читает ``runs.csv`` (и ``manifest.json``, если он
есть) и собирает два артефакта:

* ``report.md`` — отчёт для чтения: таблица «среда × агент × политика», парные
  сравнения с поправкой Холма и вердикты по H1–H6 с числами;
* ``report.json`` — те же данные машинно-читаемо (для проверок и графиков).

Вердикты считаются по данным прогона, а не пишутся руками: отчёт обязан сам
показывать, подтверждается гипотеза или нет и на каких именно числах. Если
конфигов или метрик не хватает, вердикт — «нет данных», а не «опровергнута»:
отсутствие эксперимента не является аргументом против гипотезы.

Как проверяются гипотезы (формулировки — в ``docs/HYPOTHESIS.md``)
------------------------------------------------------------------
H1  ``full_cycle`` > ``none`` при том же агенте (парно, по ``final_points``).
H2  порядок вклада компонентов: ``full_cycle > gate_only > mult_only > none``;
    ветка амплификаторов должна уступать «воротам». Строки считаются для
    каждой пары «среда × агент», где абляция вообще есть.
H3  разрыв ``full_cycle − none`` больше у сильного агента (``ucb1``), чем у
    слабого (``random``): метацикл не аддитивен к качеству агента.
H4  агенты с забыванием (``eps_decay``, ``rw``) обгоняют ``ucb1`` по очкам,
    накопленным уже *после* выхода в эпоху 1 (``points_after_epoch_1``).
H5  ``efficiency(full_cycle) > efficiency(none)`` и ни у кого
    ``efficiency > 1`` (иначе оракул задан неверно).
H6  метацикл сохраняет эффективность при информационной перегрузке:
    ``full_cycle > none`` в каждом режиме ``pyramid-overload-*`` (H6a) и
    выигрыш не «схлопывается» относительно контрольного режима ``base``, а
    эпохи продолжают открываться (H6b).

H1–H5 считаются только по «чистым» средам: режимы ``pyramid-overload-*``
описывают H6, иначе один и тот же эффект попадал бы в два вердикта и размывал
основные. H6, наоборот, игнорирует чистые среды.

Имя конфига разбирается как ``<среда>-<агент>-<политика>``, поэтому отчёт
работает с любым набором ``--suite``, а не только с полным факториалом.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import metrics as m
from . import stats as st

SUPPORTED = "подтверждена"
REFUTED = "опровергнута"
PARTIAL = "частично"
NOT_SUPPORTED = "не подтверждена"
NO_DATA = "нет данных"
"""Статусы вердиктов.

``NOT_SUPPORTED`` — «данные не показывают предсказанного эффекта, но и
значимого обратного эффекта тоже нет». Это честнее, чем «опровергнута»:
отсутствие значимости при малом числе прогонов не является доказательством
против гипотезы.
"""

POLICY_ORDER: Tuple[str, ...] = (
    "none",
    "mult_only",
    "gate_only",
    "amp_only",
    "amp",
    "full_cycle",
    "learned",
)
"""Порядок политик в таблице (от «ничего» к полному метациклу)."""

FORGETTING_AGENTS: Tuple[str, ...] = ("eps_decay", "rw", "eps0.1")
"""Агенты с забыванием/шумом, которые должны выигрывать после смены эпохи."""

STRONG_AGENT = "ucb1"
WEAK_AGENT = "random"

OVERLOAD_ENV_PREFIX = "pyramid-overload-"
"""Префикс «среды» в именах конфигов режимов информационной перегрузки (H6)."""

OVERLOAD_BASE_REGIME = "base"
"""Контрольный режим перегрузки: та же экономика, что и по умолчанию."""

OVERLOAD_RETENTION_MIN = 0.5
"""Минимальная доля выигрыша (от режима ``base``), которую H6 считает нормой."""

MIN_RUNS_FOR_VERDICT = 10
"""Порог, ниже которого вердикты помечаются как предварительные (мало пар)."""


def _is_overload_env(env: str) -> bool:
    return str(env).startswith(OVERLOAD_ENV_PREFIX)


def _regime_name(env: str) -> str:
    text = str(env)
    return text[len(OVERLOAD_ENV_PREFIX):] if _is_overload_env(text) else text


def _primary_grouped(data: "RunData") -> Dict[str, Dict[str, Dict[str, str]]]:
    """Группы «чистых» сред: H1–H5 не смешиваются с режимами H6."""
    return {
        env: agents for env, agents in data.grouped.items() if not _is_overload_env(env)
    }


def _overload_grouped(data: "RunData") -> Dict[str, Dict[str, Dict[str, str]]]:
    """Группы режимов перегрузки (H6)."""
    return {env: agents for env, agents in data.grouped.items() if _is_overload_env(env)}


@dataclass
class Verdict:
    """Вердикт по одной гипотезе."""

    code: str
    title: str
    status: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class Report:
    """Готовый отчёт: markdown для чтения, payload для машин."""

    markdown: str
    payload: Dict[str, Any]


def split_config_name(name: str) -> Optional[Tuple[str, str, str]]:
    """``<среда>-<агент>-<политика>`` → три части (``None`` для прочих имён)."""
    parts = str(name).rsplit("-", 2)
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def group_configs(names: Iterable[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Сгруппировать имена: ``среда → агент → политика → имя конфига``."""
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    for name in names:
        parsed = split_config_name(name)
        if parsed is None:
            continue
        env, agent, policy = parsed
        grouped.setdefault(env, {}).setdefault(agent, {})[policy] = name
    return grouped


def _mean(values: Optional[Sequence[float]]) -> float:
    """Среднее по конечным значениям (nan, если считать нечего)."""
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _fmt_p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


@dataclass(frozen=True)
class CompareConfig:
    """Параметры сравнений: тест, число bootstrap-выборок, alpha, сид."""

    method: str = "bootstrap"
    n_boot: int = st.DEFAULT_N_BOOT
    alpha: float = 0.05
    seed: int = st.DEFAULT_SEED


@dataclass
class RunData:
    """Прочитанный прогон: метрики по конфигам + манифест."""

    directory: Path
    values: Dict[str, Dict[str, np.ndarray]]
    metrics: List[str]
    manifest: Dict[str, Any]
    grouped: Dict[str, Dict[str, Dict[str, str]]]

    @property
    def names(self) -> List[str]:
        return list(self.values)

    @property
    def n_runs(self) -> int:
        for table in self.values.values():
            for array in table.values():
                return int(np.asarray(array).size)
        return 0

    def policy_name(self, env: str, agent: str, policy: str) -> Optional[str]:
        return self.grouped.get(env, {}).get(agent, {}).get(policy)


def load_run(directory: Path | str) -> RunData:
    """Прочитать ``runs.csv``/``manifest.json`` из каталога прогона."""
    directory = Path(directory)
    path = directory / "runs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Нет runs.csv в {directory}")
    values, metrics = st.load_runs_csv(path)
    manifest: Dict[str, Any] = {}
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return RunData(
        directory=directory,
        values=values,
        metrics=list(metrics),
        manifest=manifest,
        grouped=group_configs(values),
    )


def default_reference(data: RunData) -> Optional[str]:
    """Базовая линия по умолчанию: первый конфиг политики ``none``.

    Приоритет: «чистая» среда (H1–H5), затем контрольный режим перегрузки
    ``pyramid-overload-base`` (H6), затем любой ``none``. Так сравнение в
    прогоне ``pyramid-overload`` идёт с контролем, а не с алфавитно первым
    режимом.
    """
    none_names: List[Tuple[str, str]] = []
    for name in sorted(data.values):
        parsed = split_config_name(name)
        if parsed is not None and parsed[2] == "none":
            none_names.append((parsed[0], name))
    if not none_names:
        return sorted(data.values)[0] if data.values else None
    clean = [name for env, name in none_names if not _is_overload_env(env)]
    if clean:
        return clean[0]
    base = [
        name
        for env, name in none_names
        if _regime_name(env) == OVERLOAD_BASE_REGIME
    ]
    return (base or [name for _, name in none_names])[0]


def _paired(
    data: RunData,
    name: str,
    reference: str,
    metric: str,
    cfg: CompareConfig,
    seed_offset: int = 0,
) -> Optional[Dict[str, float]]:
    """Парное сравнение ``name`` с ``reference`` (``None``, если метрики нет)."""
    table = data.values.get(name, {})
    ref_table = data.values.get(reference, {})
    if metric not in table or metric not in ref_table:
        return None
    return st.paired_comparison(
        table[metric],
        ref_table[metric],
        method=cfg.method,
        n_boot=cfg.n_boot,
        alpha=cfg.alpha,
        seed=cfg.seed + seed_offset,
    )


def _pairs_of(data: RunData, policy_a: str, policy_b: str) -> List[Tuple[str, str, str, str]]:
    """Все тройки ``(среда, агент, конфиг A, конфиг B)`` для двух политик.

    Режимы ``pyramid-overload-*`` исключены: их разбирает H6, а H1–H5 должны
    оставаться ровно теми же утверждениями, что и до появления перегрузки.
    """
    out: List[Tuple[str, str, str, str]] = []
    for env, agents in sorted(_primary_grouped(data).items()):
        for agent, policies in sorted(agents.items()):
            if policy_a in policies and policy_b in policies:
                out.append((env, agent, policies[policy_a], policies[policy_b]))
    return out


def _h1(data: RunData, cfg: CompareConfig) -> Verdict:
    """H1: ``full_cycle`` > ``none`` при том же агенте и горизонте."""
    title = "Полный метацикл выгоднее отказа от инвестиций"
    rows: List[Dict[str, Any]] = []
    for env, agent, name, reference in _pairs_of(data, "full_cycle", "none"):
        result = _paired(data, name, reference, "final_points", cfg, len(rows))
        if result is None:
            continue
        rows.append({"env": env, "agent": agent, "config": name, "reference": reference, **result})
    if not rows:
        return Verdict("H1", title, NO_DATA,
                       "В прогоне нет пар `full_cycle`/`none` с метрикой `final_points`.")

    adjusted = st.holm([row["p_value"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(np.isfinite(p_adj) and p_adj < cfg.alpha)

    positive = [row for row in rows if row["diff"] > 0]
    significant_positive = [row for row in positive if row["significant"]]
    significant_negative = [row for row in rows if row["diff"] < 0 and row["significant"]]

    if significant_negative and significant_positive:
        status = PARTIAL
    elif significant_negative:
        status = REFUTED
    elif significant_positive and len(positive) == len(rows):
        status = SUPPORTED
    elif positive:
        status = PARTIAL
    else:
        status = NOT_SUPPORTED

    detail = "; ".join(
        f"`{row['agent']}`: Δ = {row['diff']:+.0f} очков "
        f"(p Холма {_fmt_p(row['p_adjusted'])})"
        for row in rows
    )
    return Verdict("H1", title, status, detail, {"pairs": rows})


def _ablation_targets(data: RunData) -> List[Tuple[str, str, Dict[str, str]]]:
    """Агенты «чистых» сред, у которых есть ``none`` + ``gate_only`` + ``full_cycle``."""
    out: List[Tuple[str, str, Dict[str, str]]] = []
    for env, agents in sorted(_primary_grouped(data).items()):
        for agent, policies in sorted(agents.items()):
            if {"none", "gate_only", "full_cycle"} <= set(policies):
                out.append((env, agent, policies))
    return out


def _h2(data: RunData, cfg: CompareConfig) -> Verdict:
    """H2: технологии дают больше, чем амплификаторы и конвертация.

    Строка считается для каждой пары «среда × агент», где есть полная абляция:
    иначе отчёт молча выбирал одного агента и не показывал, что в другом
    месте порядок компонентов нарушен.
    """
    title = "Вклад компонентов: технологии важнее амплификаторов и множителей"
    targets = _ablation_targets(data)
    if not targets:
        return Verdict(
            "H2",
            title,
            NO_DATA,
            "Нужны конфиги `none`, `gate_only` и `full_cycle` для одного агента "
            "(см. `--suite pyramid-ablation`).",
        )

    rows: List[Dict[str, Any]] = []
    for env, agent, policies in targets:
        order = [policy for policy in POLICY_ORDER if policy in policies]
        means = {
            policy: _mean(data.values[policies[policy]].get("final_points"))
            for policy in order
        }
        baseline = _paired(
            data, policies["full_cycle"], policies["none"], "final_points", cfg, len(rows)
        )
        full_best = bool(means["full_cycle"] >= max(means.values()) - 1e-9)
        gate_over_mult = (
            means["gate_only"] > means["mult_only"] if "mult_only" in means else None
        )
        amp_branch_weak = (
            means["gate_only"]
            > max(means.get("amp_only", -np.inf), means.get("amp", -np.inf))
            if {"amp_only", "amp"} & set(means)
            else None
        )
        significant = bool(
            baseline and baseline["diff"] > 0 and baseline["p_value"] < cfg.alpha
        )

        if gate_over_mult is False or not full_best:
            row_status = REFUTED
        elif gate_over_mult and significant:
            row_status = SUPPORTED
        else:
            row_status = PARTIAL

        ranked = sorted(
            ((policy, value) for policy, value in means.items() if policy != "learned"),
            key=lambda item: item[1],
            reverse=True,
        )
        chain = " > ".join(f"{policy} {_fmt(value, 0)}" for policy, value in ranked)
        detail = f"`{env}/{agent}`: {chain}"
        if "learned" in means:
            detail += f" (learned {_fmt(means['learned'], 0)} — P4, вне абляции)"
        detail += (
            f"; full_cycle vs none: Δ = {_fmt(baseline['diff'], 0) if baseline else '—'}"
            f" (p {_fmt_p(baseline['p_value']) if baseline else '—'})"
        )
        rows.append(
            {
                "env": env,
                "agent": agent,
                "status": row_status,
                "detail": detail,
                "means": {policy: float(means[policy]) for policy in order},
                "gate_over_mult": gate_over_mult,
                "gate_over_amp": amp_branch_weak,
                "full_cycle_best": full_best,
                "full_cycle_vs_none": baseline,
            }
        )

    statuses = {row["status"] for row in rows}
    if REFUTED in statuses:
        status = REFUTED
    elif statuses == {SUPPORTED}:
        status = SUPPORTED
    else:
        status = PARTIAL

    first = rows[0]
    return Verdict(
        "H2",
        title,
        status,
        "; ".join(row["detail"] for row in rows),
        {
            "rows": rows,
            "env": first["env"],
            "agent": first["agent"],
            "means": first["means"],
            "gate_over_mult": first["gate_over_mult"],
            "gate_over_amp": first["gate_over_amp"],
            "full_cycle_best": first["full_cycle_best"],
            "full_cycle_vs_none": first["full_cycle_vs_none"],
        },
    )


def _h3(data: RunData, cfg: CompareConfig) -> Verdict:
    """H3: выигрыш метацикла зависит от качества агента (не аддитивен)."""
    title = "Выигрыш метацикла зависит от качества агента"
    gaps: Dict[Tuple[str, str], Dict[str, float]] = {}
    for index, (env, agent, name, reference) in enumerate(_pairs_of(data, "full_cycle", "none")):
        result = _paired(data, name, reference, "final_points", cfg, index)
        if result is not None:
            gaps[(env, agent)] = result
    if not gaps:
        return Verdict("H3", title, NO_DATA, "Нет пар `full_cycle`/`none` для сравнения агентов.")

    rows: List[Dict[str, Any]] = []
    for env, agents in sorted(_primary_grouped(data).items()):
        strong = gaps.get((env, STRONG_AGENT))
        weak = gaps.get((env, WEAK_AGENT))
        if strong is None or weak is None:
            continue
        rows.append(
            {
                "env": env,
                "strong_agent": STRONG_AGENT,
                "strong_gap": strong["diff"],
                "weak_agent": WEAK_AGENT,
                "weak_gap": weak["diff"],
                "half_width_strong": 0.5 * (strong["ci_hi"] - strong["ci_lo"]),
            }
        )

    if not rows:
        return Verdict(
            "H3",
            title,
            NO_DATA,
            f"Нужны пары и для `{STRONG_AGENT}`, и для `{WEAK_AGENT}` "
            "с политиками `none`/`full_cycle`.",
        )

    all_greater = all(row["strong_gap"] > row["weak_gap"] for row in rows)
    weak_positive = all(row["weak_gap"] > 0 for row in rows)
    if all_greater and weak_positive:
        status = SUPPORTED
    elif all_greater:
        status = PARTIAL
    else:
        status = REFUTED

    detail = "; ".join(
        f"`{row['env']}`: Δ({row['strong_agent']}) = {row['strong_gap']:+.0f}, "
        f"Δ({row['weak_agent']}) = {row['weak_gap']:+.0f}"
        for row in rows
    )
    return Verdict(
        "H3",
        title,
        status,
        detail,
        {"pairs": rows, "weak_agent_gain_positive": weak_positive},
    )


def _h4(data: RunData, cfg: CompareConfig) -> Verdict:
    """H4: после смены эпохи выигрывают агенты с забыванием."""
    title = "После смены эпохи выигрывают алгоритмы с забыванием"
    metric = "points_after_epoch_1"
    if metric not in data.metrics:
        return Verdict(
            "H4",
            title,
            NO_DATA,
            f"В `runs.csv` нет метрики `{metric}` — пересоберите прогон текущей версией.",
        )

    rows: List[Dict[str, Any]] = []
    for env, agents in sorted(_primary_grouped(data).items()):
        if STRONG_AGENT not in agents:
            continue
        for agent in FORGETTING_AGENTS:
            if agent not in agents or agent == STRONG_AGENT:
                continue
            # Сравнивать можно только одинаковые политики: иначе разница
            # агентов смешана с разницей политик.
            common = sorted(set(agents[agent]) & set(agents[STRONG_AGENT]))
            if not common:
                continue
            policy = "full_cycle" if "full_cycle" in common else common[0]
            baseline = agents[STRONG_AGENT][policy]
            result = _paired(data, agents[agent][policy], baseline, metric, cfg, len(rows))
            if result is None:
                continue
            rows.append(
                {
                    "env": env,
                    "agent": agent,
                    "policy": policy,
                    "config": agents[agent][policy],
                    "reference": baseline,
                    "baseline_policy": policy,
                    **result,
                }
            )

    if not rows:
        return Verdict(
            "H4",
            title,
            NO_DATA,
            f"Нет пар `{STRONG_AGENT}` и агентов с забыванием ({', '.join(FORGETTING_AGENTS)}) "
            "при одинаковой политике.",
        )

    adjusted = st.holm([row["p_value"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(np.isfinite(p_adj) and p_adj < cfg.alpha)

    positive = [row for row in rows if row["diff"] > 0]
    significant_positive = [row for row in positive if row["significant"]]
    significant_negative = [row for row in rows if row["diff"] < 0 and row["significant"]]
    if significant_positive and significant_negative:
        status = PARTIAL
    elif significant_negative:
        status = REFUTED
    elif significant_positive and len(positive) == len(rows):
        status = SUPPORTED
    elif significant_positive:
        status = PARTIAL
    else:
        status = NOT_SUPPORTED

    detail = "; ".join(
        f"`{row['agent']}` vs `{STRONG_AGENT}` ({row['policy']}): "
        f"Δ = {row['diff']:+.0f} очков после эпохи 1 (p Холма {_fmt_p(row['p_adjusted'])})"
        for row in rows
    )
    return Verdict("H4", title, status, detail, {"pairs": rows, "metric": metric})


def _h5(data: RunData, cfg: CompareConfig) -> Verdict:
    """H5: ``efficiency`` растёт от метацикла и не превышает 1."""
    title = "Полный метацикл даёт большую долю от верхней границы (и не больше 1)"
    if "efficiency" not in data.metrics:
        return Verdict(
            "H5",
            title,
            NO_DATA,
            "В прогоне нет метрики `efficiency`: нужен конфиг с оракулом "
            "(`pyramid-oracle-*`) при тех же сидах.",
        )

    over_one: List[Dict[str, Any]] = []
    for name, table in sorted(data.values.items()):
        parsed = split_config_name(name)
        if parsed is not None and _is_overload_env(parsed[0]):
            continue  # режимы перегрузки относятся к H6, а не к H5
        value = _mean(table.get("efficiency"))
        if np.isfinite(value) and value > 1.0 + 1e-9:
            over_one.append({"config": name, "efficiency": float(value)})

    rows: List[Dict[str, Any]] = []
    for env, agent, name, reference in _pairs_of(data, "full_cycle", "none"):
        result = _paired(data, name, reference, "efficiency", cfg, len(rows))
        if result is None:
            continue
        rows.append({"env": env, "agent": agent, "config": name, "reference": reference, **result})

    if not rows and not over_one:
        return Verdict(
            "H5",
            title,
            NO_DATA,
            "Нет ни пар `full_cycle`/`none` с `efficiency`, ни превышений единицы.",
        )

    adjusted = st.holm([row["p_value"] for row in rows]) if rows else []
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(np.isfinite(p_adj) and p_adj < cfg.alpha)

    positive = [row for row in rows if row["diff"] > 0]
    significant_positive = [row for row in positive if row["significant"]]
    significant_negative = [row for row in rows if row["diff"] < 0 and row["significant"]]
    if over_one:
        status = REFUTED
    elif significant_positive:
        status = SUPPORTED
    elif significant_negative:
        status = REFUTED
    else:
        status = NOT_SUPPORTED

    parts = [
        f"`{row['agent']}`: Δ(КПД) = {row['diff']:+.3f} "
        f"(p Холма {_fmt_p(row['p_adjusted'])})"
        for row in rows
    ]
    if over_one:
        parts.append(
            "КПД > 1 у: " + ", ".join(f"`{item['config']}` ({item['efficiency']:.3f})" for item in over_one)
            + " — оракул задан неверно либо у конфигов разные сиды"
        )
    return Verdict(
        "H5",
        title,
        status,
        "; ".join(parts),
        {"pairs": rows, "over_one": over_one},
    )


def _finite_share(values: Optional[Sequence[float]]) -> float:
    """Доля конечных значений (например, доля прогонов, дошедших до эпохи)."""
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    return float(np.isfinite(array).mean())


def _h6(data: RunData, cfg: CompareConfig) -> Verdict:
    """H6: метацикл сохраняет эффективность при информационной перегрузке.

    Две части, обе считаются по данным:

    * **H6a (уровень)** — ``full_cycle`` лучше ``none`` в каждом режиме
      ``pyramid-overload-*`` (парно по ``final_points``, Холм по режимам);
    * **H6b (качество)** — выигрыш не «схлопывается» относительно контроля
      ``base`` (минимальная доля по режимам ≥ ``OVERLOAD_RETENTION_MIN``), и
      доля прогонов с открытой эпохой 1 не падает ниже половины этой доли в
      ``base``.

    ``useful_info_share`` — характеристика *режима* (среда задаёт долю
    полезных единиц), поэтому она печатается как шкала перегрузки, а не как
    достижение политики. Опровержение — значимое ухудшение ``full_cycle``
    относительно ``none`` или потеря эпох, которые в ``base`` открывались.
    """
    title = "Метацикл сохраняет эффективность при информационной перегрузке"
    regimes = _overload_grouped(data)
    if not regimes:
        return Verdict(
            "H6",
            title,
            NO_DATA,
            "Нужны конфиги `pyramid-overload-*` (см. `--suite pyramid-overload`).",
        )

    rows: List[Dict[str, Any]] = []
    for env, agents in sorted(regimes.items()):
        policies = agents.get(STRONG_AGENT, {})
        name = policies.get("full_cycle")
        reference = policies.get("none")
        if name is None or reference is None:
            continue
        result = _paired(data, name, reference, "final_points", cfg, len(rows))
        if result is None:
            continue
        table = data.values.get(name, {})
        rows.append(
            {
                "env": env,
                "regime": _regime_name(env),
                "config": name,
                "reference": reference,
                "useful_info_share": _mean(table.get("useful_info_share")),
                "junk_info_share": _mean(table.get("junk_info_share")),
                "reach_epoch_1": _finite_share(table.get("steps_to_epoch_1")),
                "reach_epoch_2": _finite_share(table.get("steps_to_epoch_2")),
                "final_epoch": _mean(table.get("final_epoch")),
                "efficiency": _mean(table.get("efficiency")),
                **result,
            }
        )

    if not rows:
        return Verdict(
            "H6",
            title,
            NO_DATA,
            "В режимах `pyramid-overload-*` нет пары `ucb1`: нужны конфиги "
            "`...-ucb1-none` и `...-ucb1-full_cycle`.",
        )

    adjusted = st.holm([row["p_value"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = float(p_adj)
        row["significant"] = bool(np.isfinite(p_adj) and p_adj < cfg.alpha)

    base = next((row for row in rows if row["regime"] == OVERLOAD_BASE_REGIME), None)
    overload = [row for row in rows if row["regime"] != OVERLOAD_BASE_REGIME]
    quality_measured = bool(
        base is not None
        and "steps_to_epoch_1" in data.metrics
        and np.isfinite(base["reach_epoch_1"])
    )

    for row in overload:
        if quality_measured and base is not None and base["diff"] > 0:
            row["retention"] = float(row["diff"] / base["diff"])
        else:
            row["retention"] = float("nan")
        row["lost_epochs"] = bool(
            quality_measured
            and base is not None
            and base["reach_epoch_1"] > 0
            and row["reach_epoch_1"] < 0.5 * base["reach_epoch_1"]
        )

    retentions = [row["retention"] for row in overload if np.isfinite(row["retention"])]
    median_retention = float(np.median(retentions)) if retentions else float("nan")
    min_retention = float(min(retentions)) if retentions else float("nan")
    negative = [row for row in overload if row["diff"] < 0 and row["significant"]]
    positive = [row for row in overload if row["diff"] > 0]
    significant_positive = [row for row in positive if row["significant"]]
    collapsed = [row for row in overload if row["lost_epochs"]]

    if not overload:
        status = NO_DATA
    elif negative or any(row["reach_epoch_1"] == 0 for row in collapsed):
        status = REFUTED
    elif (
        len(positive) == len(overload)
        and significant_positive
        and quality_measured
        and not collapsed
        and np.isfinite(min_retention)
        and min_retention >= OVERLOAD_RETENTION_MIN
    ):
        status = SUPPORTED
    elif positive:
        status = PARTIAL
    else:
        status = NOT_SUPPORTED

    parts: List[str] = []
    for row in rows:
        share = row["useful_info_share"]
        share_text = f", сигнала {share:.0%}" if np.isfinite(share) else ""
        reach = row["reach_epoch_1"]
        reach_text = f", эпоха 1 в {reach:.0%} прогонов" if np.isfinite(reach) else ""
        parts.append(
            f"`{row['regime']}`: Δ = {row['diff']:+.0f} очков "
            f"(p Холма {_fmt_p(row['p_adjusted'])}){share_text}{reach_text}"
        )
    if np.isfinite(median_retention):
        parts.append(
            f"минимальное сохранение выигрыша к `base`: {min_retention:.0%} "
            f"(медиана {median_retention:.0%})"
        )
    if not quality_measured:
        parts.append(
            "качество не измерено: нет режима `base` или метрики `steps_to_epoch_1` "
            "(пересоберите прогон текущей версией)"
        )
    if not overload:
        parts.append("кроме контрольного `base` режимов перегрузки в прогоне нет")

    return Verdict(
        "H6",
        title,
        status,
        "; ".join(parts),
        {
            "pairs": rows,
            "base_regime": OVERLOAD_BASE_REGIME,
            "median_retention": median_retention,
            "min_retention": min_retention,
            "retention_min": OVERLOAD_RETENTION_MIN,
            "quality_measured": quality_measured,
            "collapsed": [row["regime"] for row in collapsed],
            "metric": "final_points",
        },
    )


def learned_summary(data: RunData, cfg: CompareConfig) -> Optional[Dict[str, Any]]:
    """Сравнение обучаемой политики (P4) с ручными базлайнами."""
    rows: List[Dict[str, Any]] = []
    for env, agents in sorted(_primary_grouped(data).items()):
        for agent, policies in sorted(agents.items()):
            if "learned" not in policies:
                continue
            name = policies["learned"]
            table = data.values[name]
            row: Dict[str, Any] = {
                "env": env,
                "agent": agent,
                "config": name,
                "mean": _mean(table.get("final_points")),
                "epochs": _mean(table.get("final_epoch")),
                "ratio_vs_full_cycle": float("nan"),
                "comparisons": {},
            }
            for reference_policy in ("none", "full_cycle", "learned_slow"):
                reference = policies.get(reference_policy)
                if reference is None:
                    continue
                result = _paired(data, name, reference, "final_points", cfg, len(rows))
                if result is None:
                    continue
                row["comparisons"][reference_policy] = {"reference": reference, **result}
            full = row["comparisons"].get("full_cycle")
            if full and full["reference_mean"] > 0:
                row["ratio_vs_full_cycle"] = row["mean"] / full["reference_mean"]
            rows.append(row)

    if not rows:
        return None

    ratios = [
        row["ratio_vs_full_cycle"]
        for row in rows
        if np.isfinite(row["ratio_vs_full_cycle"])
    ]
    if not ratios:
        verdict = "В прогоне нет пары с `full_cycle` — сравнение ограничено `none`."
    elif min(ratios) >= 0.9:
        verdict = "Обучаемая политика догоняет ручной жадный метацикл."
    elif min(ratios) >= 0.6:
        verdict = "Обучаемая политика уступает жадной, но инвестирует осмысленно."
    else:
        verdict = "Обучаемая политика заметно слабее жадной: проверьте окно измерения."
    return {"rows": rows, "verdict": verdict}


def _median(values: Optional[Sequence[float]]) -> float:
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def summary_rows(data: RunData) -> List[Dict[str, str]]:
    """Строки таблицы «среда × агент × политика» (уже отформатированные)."""
    rows: List[Dict[str, str]] = []
    for name in sorted(data.values):
        parsed = split_config_name(name)
        env, agent, policy = parsed if parsed else ("—", "—", "—")
        table = data.values[name]
        points = m.aggregate(table.get("final_points", []))
        rows.append(
            {
                "среда": env,
                "агент": agent,
                "политика": policy,
                "очки (mean ± SEM)": f"{_fmt(points['mean'], 0)} ± {_fmt(points['sem'], 0)}",
                "эпоха": _fmt(_mean(table.get("final_epoch")), 1),
                "шаг эпохи 1": _fmt(_median(table.get("steps_to_epoch_1")), 0),
                "КПД": _fmt(_mean(table.get("efficiency")), 3),
            }
        )
    rows.sort(
        key=lambda row: (
            row["агент"] == "oracle",
            row["агент"],
            POLICY_ORDER.index(row["политика"])
            if row["политика"] in POLICY_ORDER
            else len(POLICY_ORDER),
            row["среда"],
        )
    )
    return rows


def _md_table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _markdown(data: RunData, payload: Dict[str, Any], results: Sequence[st.PairedResult]) -> str:
    """Собрать markdown-отчёт из payload и посчитанных сравнений."""
    manifest = data.manifest
    run_id = manifest.get("run_id") or data.directory.name
    lines: List[str] = [f"# Отчёт по прогону `{run_id}`", ""]

    version = manifest.get("bandit_version", "—")
    commit = manifest.get("git_commit")
    lines.append(f"* версия пакета: `{version}`" + (f", git `{commit}`" if commit else ""))
    lines.append(f"* создан: {manifest.get('created', '—')}")
    lines.append(
        f"* конфигов: {len(data.values)}, прогонов на конфиг: {data.n_runs}"
        + (
            f", шагов: {manifest['n_steps_override']}"
            if manifest.get("n_steps_override")
            else ""
        )
    )
    lines.append(f"* набор: `{manifest.get('suite') or 'custom'}`")
    lines.append(
        f"* сравнения: `{payload['method']}`, bootstrap = {payload['n_boot']}, "
        f"alpha = {payload['alpha']}, сид = {payload['seed']}, поправка Холма"
    )
    if manifest.get("command"):
        lines.append(f"* команда: `{manifest['command']}`")
    lines.append("")
    if data.n_runs < MIN_RUNS_FOR_VERDICT:
        lines.append(
            f"**Внимание:** прогонов на конфиг — {data.n_runs} "
            f"(< {MIN_RUNS_FOR_VERDICT}); вердикты предварительные, "
            "для решения нужно больше парных прогонов."
        )
        lines.append("")

    lines.append("## 1. Сводка «среда × агент × политика»")
    lines.append("")
    summary = summary_rows(data)
    if summary:
        lines.append(_md_table(list(summary[0]), summary))
    else:
        lines.append("_В прогоне нет конфигов вида `<среда>-<агент>-<политика>`._")
    lines.append("")
    lines.append(
        "`—` в колонке «КПД» означает, что метрика не считалась (в прогоне нет "
        "конфига с оракулом для тех же `env.params` и сидов). `oracle` — "
        "верхняя граница, а не участник рейтинга."
    )
    lines.append("")

    lines.append("## 2. Парные сравнения с базлайном")
    lines.append("")
    lines.append(f"Базлайн: `{payload['reference'] or '—'}`. Метрика: `final_points`.")
    lines.append("")
    if results:
        lines.append(_md_table(list(st.COMPARISON_COLUMNS), st.format_rows(results)))
        lines.append("")
        lines.append("Звёздочкой отмечены сравнения со скорректированным `p < alpha`.")
    else:
        lines.append("_Сравнения не посчитаны: нет общего базлайна._")
    lines.append("")

    lines.append("## 3. Гипотезы")
    lines.append("")
    for verdict in payload["hypotheses"]:
        lines.append(f"### {verdict['code']} — {verdict['title']}")
        lines.append("")
        lines.append(f"**Статус:** {verdict['status']}.")
        lines.append("")
        lines.append(verdict["detail"])
        lines.append("")

    if payload.get("learned"):
        learned = payload["learned"]
        lines.append("## 4. Обучаемая мета-политика (P4)")
        lines.append("")
        lines.append(learned["verdict"])
        lines.append("")
        rows = []
        for row in learned["rows"]:
            full = row["comparisons"].get("full_cycle")
            rows.append(
                {
                    "агент": row["agent"],
                    "очки": _fmt(row["mean"], 0),
                    "эпоха": _fmt(row["epochs"], 1),
                    "к full_cycle": _fmt(row["ratio_vs_full_cycle"], 3),
                    "Δ к none": _fmt(
                        row["comparisons"].get("none", {}).get("diff", float("nan")), 0
                    ),
                    "Δ к full_cycle": _fmt(full["diff"] if full else float("nan"), 0),
                }
            )
        lines.append(_md_table(list(rows[0]), rows))
        lines.append("")

    lines.append("## 5. Воспроизведение")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"python -m bandit.cli show {data.directory}")
    lines.append(
        f"python -m bandit.cli stats {data.directory / 'runs.csv'} "
        f"--reference {payload['reference'] or '—'} --method {payload['method']}"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_report(
    directory: Path | str,
    reference: Optional[str] = None,
    method: str = "bootstrap",
    n_boot: int = st.DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = st.DEFAULT_SEED,
) -> Report:
    """Построить отчёт по каталогу прогона (``runs.csv`` + ``manifest.json``)."""
    data = load_run(directory)
    cfg = CompareConfig(method=method, n_boot=n_boot, alpha=alpha, seed=seed)
    reference_name = reference or default_reference(data)
    if reference is not None and reference_name not in data.values:
        raise KeyError(
            f"Базлайна {reference!r} нет в прогоне. Доступны: {', '.join(sorted(data.values))}"
        )

    results: List[st.PairedResult] = []
    if reference_name is not None and "final_points" in data.metrics:
        results = st.compare_values(
            data.values,
            reference_name,
            metric="final_points",
            method=cfg.method,
            n_boot=cfg.n_boot,
            alpha=cfg.alpha,
            seed=cfg.seed,
        )

    verdicts = [
        _h1(data, cfg),
        _h2(data, cfg),
        _h3(data, cfg),
        _h4(data, cfg),
        _h5(data, cfg),
        _h6(data, cfg),
    ]
    payload: Dict[str, Any] = {
        "run_id": data.manifest.get("run_id") or data.directory.name,
        "bandit_version": data.manifest.get("bandit_version"),
        "created": data.manifest.get("created"),
        "suite": data.manifest.get("suite"),
        "n_configs": len(data.values),
        "n_runs": data.n_runs,
        "reference": reference_name,
        "method": cfg.method,
        "n_boot": int(cfg.n_boot),
        "alpha": float(cfg.alpha),
        "seed": int(cfg.seed),
        "configs": data.names,
        "metrics": sorted(data.metrics),
        "summary": summary_rows(data),
        "comparisons": [item.as_dict() for item in results],
        "hypotheses": [verdict.as_dict() for verdict in verdicts],
        "overload": {
            "env_prefix": OVERLOAD_ENV_PREFIX,
            "base_regime": OVERLOAD_BASE_REGIME,
            "retention_min": OVERLOAD_RETENTION_MIN,
        },
        "n_runs_warning": bool(data.n_runs < MIN_RUNS_FOR_VERDICT),
        "learned": learned_summary(data, cfg),
    }
    return Report(markdown=_markdown(data, payload, results), payload=payload)


def _json_default(value: Any) -> Any:
    """Преобразование numpy-значений для ``json.dumps``."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Не сериализуется в JSON: {type(value).__name__}")


def _sanitize(value: Any) -> Any:
    """Заменить нечисловые float на ``null``: ``Infinity`` — не валидный JSON.

    ``dz`` бывает бесконечным (нулевой разброс разностей), ``p`` — NaN при
    одной паре; такие значения должны читаться внешними парсерами.
    """
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_, np.ndarray)):
        return _sanitize(value.item() if not isinstance(value, np.ndarray) else value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(directory: Path | str, report: Report) -> Tuple[Path, Path]:
    """Сохранить ``report.md`` и ``report.json`` в каталог прогона."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "report.md"
    json_path = directory / "report.json"
    markdown_path.write_text(report.markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            _sanitize(report.payload),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return markdown_path, json_path


__all__ = [
    "NO_DATA",
    "NOT_SUPPORTED",
    "PARTIAL",
    "REFUTED",
    "Report",
    "RunData",
    "SUPPORTED",
    "Verdict",
    "build_report",
    "default_reference",
    "group_configs",
    "learned_summary",
    "load_run",
    "split_config_name",
    "summary_rows",
    "write_report",
]
