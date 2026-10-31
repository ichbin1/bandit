"""Командная строка: ``python -m bandit.cli <команда>``.

Команды
-------
``list``   — что вообще есть (среды, агенты, политики, наборы, конфиги);
``run``    — прогнать конфиги/набор, сохранить ``.npz`` + манифест + сводку;
``show``   — пересчитать сводку по сохранённому каталогу;
``stats``  — парные сравнения по ``runs.csv`` (bootstrap/Wilcoxon/знаки + Холм);
``report`` — сводный отчёт: таблицы, сравнения, вердикты H1–H6 (``report.md``);
``ui``     — веб-интерфейс: конструктор прогонов в браузере (``bandit.web``);
``demo``   — быстрая демонстрация (``--suite demo``) с графиками.

Примеры
-------
    python -m bandit.cli list
    python -m bandit.cli run --suite pyramid-ablation --runs 30 --jobs 4
    python -m bandit.cli run --config pyramid-ucb1-none --config pyramid-ucb1-full_cycle
    python -m bandit.cli run --config-file results/ui-.../configs.json
    python -m bandit.cli show results/20260101-120000
    python -m bandit.cli stats results/20260101-120000/runs.csv --method wilcoxon
    python -m bandit.cli report results/20260101-120000
    python -m bandit.cli ui --port 8000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import __version__, metrics as m, plots
from . import stats as st
from .configs import AGENTS, POLICIES, REGISTRY, SUITES, resolve
from .metrics import METRIC_DESCRIPTIONS
from .report import build_report, write_report
from .runner import RECORD_FIELDS, RunConfig, RunRecords, run_config

TABLE_METRICS = (
    "final_points",
    "cumulative_arm_regret",
    "optimal_arm_rate",
    "arm_pull_share",
    "steps_to_epoch_1",
    "efficiency",
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _median(values: Sequence[float]) -> float:
    """Медиана только по конечным значениям (без предупреждений про all-NaN)."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _print_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    """Печать простой текстовой таблицы без внешних зависимостей."""
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    line = "  ".join(col.ljust(widths[col]) for col in columns)
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def configure_output() -> None:
    """Не падать на символах, которых нет в текущей кодировке консоли.

    На Windows ``sys.stdout`` часто cp1251/cp866: кириллица печатается, а
    символы вроде ``×`` — нет. ``errors="replace"`` заменяет их на ``?``
    вместо ``UnicodeEncodeError``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - экзотические потоки
            pass


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - git может отсутствовать
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def cmd_list(args: argparse.Namespace) -> int:
    print(f"bandit v{__version__}")
    print(f"\nНаборы ({len(SUITES)}):")
    for name, configs in SUITES.items():
        print(f"  {name:<18} {len(configs)} конфигов")
    print(f"\nАгенты ({len(AGENTS)}): {', '.join(AGENTS)}")
    print(f"Политики ({len(POLICIES)}): {', '.join(POLICIES)}")
    if args.verbose:
        print(f"\nКонфиги ({len(REGISTRY)}):")
        for name, config in REGISTRY.items():
            print(f"  {name:<34} {config.n_steps:>5} шагов  {config.description}")
    else:
        print("\nВсе имена конфигов: --verbose")
    return 0


ORACLE_POLICY_ORDER: Sequence[str] = ("oracle_full_cycle", "oracle_none")
"""Порядок предпочтения оракулов: полный метацикл — более сильная граница."""


def _env_signature(config: Mapping[str, object]) -> str:
    """Канонический отпечаток спецификации среды: ``kind`` + ``params``."""
    env = config.get("env") or {}
    env = dict(env) if isinstance(env, Mapping) else {}
    params = dict(env.get("params") or {})
    return json.dumps(
        {"kind": env.get("kind"), "params": params},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _policy_kind(config: Mapping[str, object]) -> str:
    policy = config.get("policy") or {}
    if not isinstance(policy, Mapping):
        return ""
    return str(policy.get("kind") or "")


def _seed_signature(records: RunRecords) -> tuple:
    return tuple((int(pair.env_seed), int(pair.agent_seed)) for pair in records.seeds)


def _oracle_finals(
    records_by_name: Dict[str, RunRecords], records: RunRecords
) -> Optional[np.ndarray]:
    """Очки оракула при тех же сидах **и тех же параметрах среды**.

    Раньше имена оракулов были захардкожены (``pyramid-oracle-*``), поэтому в
    прогоне с режимами перегрузки (``pyramid-overload-*``) КПД молча не
    считался, а H6 не могла измерить сохранение качества. Теперь оракул
    ищется по спецификации: политика ``oracle_*``, совпадающие
    ``env.kind``/``env.params``, сиды и горизонт.
    """
    target_env = _env_signature(records.config)
    target_seeds = _seed_signature(records)
    for kind in ORACLE_POLICY_ORDER:
        for name in sorted(records_by_name):
            oracle = records_by_name[name]
            if _policy_kind(oracle.config) != kind:
                continue
            if oracle.n_steps != records.n_steps or oracle.n_runs != records.n_runs:
                continue
            if _env_signature(oracle.config) != target_env:
                continue
            if _seed_signature(oracle) != target_seeds:
                continue
            return m.final_points(oracle)
    return None


def _write_metrics_csv(path: Path, summaries: Dict[str, Dict[str, np.ndarray]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "metric", "mean", "sem", "ci95_lo", "ci95_hi", "n"])
        for name, values in summaries.items():
            for metric, array in values.items():
                stats = m.aggregate(array)
                writer.writerow(
                    [
                        name,
                        metric,
                        f"{stats['mean']:.6g}",
                        f"{stats['sem']:.6g}",
                        f"{stats['ci95_lo']:.6g}",
                        f"{stats['ci95_hi']:.6g}",
                        int(stats["n"]),
                    ]
                )


def _write_runs_csv(
    path: Path,
    summaries: Dict[str, Dict[str, np.ndarray]],
    records_by_name: Dict[str, RunRecords],
) -> None:
    """По одному прогону в строке — вход для парных тестов (``bandit.stats``)."""
    metric_names: List[str] = []
    for values in summaries.values():
        for metric in values:
            if metric not in metric_names and metric != "arm_pull_share":
                metric_names.append(metric)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "run", "env_seed", "agent_seed"] + metric_names)
        for name, values in summaries.items():
            records = records_by_name[name]
            for index, seed_pair in enumerate(records.seeds):
                row = [name, index, seed_pair.env_seed, seed_pair.agent_seed]
                row.extend(
                    f"{float(values[metric][index]):.6g}" if metric in values else "nan"
                    for metric in metric_names
                )
                writer.writerow(row)


def _summary_rows(summaries: Dict[str, Dict[str, np.ndarray]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name, values in summaries.items():
        points = m.aggregate(values["final_points"])
        row = {
            "config": name,
            "очки": f"{_fmt(points['mean'])}±{_fmt(points['sem'])}",
            "arm_regret": _fmt(m.aggregate(values["cumulative_arm_regret"])["mean"]),
            "лучшая_рука": _fmt(m.aggregate(values["optimal_arm_rate"])["mean"], 3),
            "доля_рук": _fmt(m.aggregate(values["arm_pull_share"])["mean"], 3),
            "шаг_эпохи1": _fmt(_median(values["steps_to_epoch_1"]), 0),
            "эпоха": _fmt(m.aggregate(values["final_epoch"])["mean"], 1),
        }
        if "efficiency" in values:
            row["КПД"] = _fmt(m.aggregate(values["efficiency"])["mean"], 3)
        rows.append(row)

    def _key(row: Dict[str, str]) -> float:
        text = row["очки"].split("±")[0]
        return -float(text) if text not in ("—", "nan", "") else float("inf")

    rows.sort(key=_key)
    return rows


def _error_text(exc: BaseException) -> str:
    """Текст ошибки для консоли: у ``KeyError`` полезное сообщение лежит в ``args[0]``."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def load_config_file(path: Path | str) -> List[RunConfig]:
    """Прочитать конфиги из JSON: ``[RunConfig, ...]`` или ``{"configs": [...]}``.

    Такой файл пишет веб-интерфейс (``results/<id>/configs.json``), поэтому
    прогон из браузера полностью воспроизводится командой из ``manifest.json``.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать {path}: {exc}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("configs") or []
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{path}: ожидался непустой список конфигов или {{'configs': [...]}}"
        )
    return [RunConfig.from_dict(item) for item in payload]


def _run_configs(args: argparse.Namespace) -> List[RunConfig]:
    """Собрать конфиги прогона: ``--suite``/``--config`` плюс ``--config-file``.

    Конфиги из файла регистрируются в ``configs.REGISTRY`` — так же, как это
    делает ``scripts/advantage_sweep.py``. Дальше ``cmd_run`` и отчёт работают
    с именами, а полное описание всё равно попадает в манифест.
    """
    configs: List[RunConfig] = []
    if args.suite or args.config:
        configs.extend(resolve(args.config, args.suite))
    config_file = getattr(args, "config_file", None)
    if config_file:
        for config in load_config_file(config_file):
            existing = REGISTRY.get(config.name)
            if existing is not None and existing.to_dict() != config.to_dict():
                raise KeyError(
                    f"Конфиг {config.name!r} уже зарегистрирован с другими параметрами"
                )
            REGISTRY[config.name] = config
            configs.append(config)
    if not configs:
        raise ValueError("Укажите --config, --suite и/или --config-file")
    unique: List[RunConfig] = []
    seen = set()
    for config in configs:
        if config.name in seen:
            continue
        seen.add(config.name)
        unique.append(config)
    return unique


def cmd_run(args: argparse.Namespace) -> int:
    try:
        configs = _run_configs(args)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Ошибка: {_error_text(exc)}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else Path("results") / f"{_timestamp()}-{args.suite or 'custom'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    records_by_name: Dict[str, RunRecords] = {}
    for index, config in enumerate(configs, start=1):
        n_steps = int(config.n_steps if args.steps is None else args.steps)
        print(
            f"[{index}/{len(configs)}] {config.name}: {n_steps} шагов, {args.runs} прогонов",
            flush=True,
        )
        records = run_config(
            config,
            n_runs=args.runs,
            base_seed=args.seed,
            jobs=args.jobs,
            n_steps=args.steps,
            progress=not args.quiet,
        )
        records.save(out_dir)
        records_by_name[config.name] = records

    summaries = {
        name: m.summarize(records, oracle_final=_oracle_finals(records_by_name, records))
        for name, records in records_by_name.items()
    }

    manifest = {
        "run_id": out_dir.name,
        "created": started.isoformat(timespec="seconds"),
        "elapsed_sec": round((datetime.now() - started).total_seconds(), 1),
        "bandit_version": __version__,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "n_runs": int(args.runs),
        "base_seed": int(args.seed),
        "jobs": int(args.jobs),
        "suite": args.suite,
        "config_file": (
            str(args.config_file) if getattr(args, "config_file", None) else None
        ),
        "n_steps_override": args.steps,
        "configs": [config.to_dict() for config in configs],
        "record_fields": list(RECORD_FIELDS),
        "metric_descriptions": METRIC_DESCRIPTIONS,
        "oracle_reference": (
            "конфиг с политикой oracle_* и теми же env.params, сидами и горизонтом"
        ),
        "table_metrics": list(TABLE_METRICS),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_metrics_csv(out_dir / "metrics.csv", summaries)
    _write_runs_csv(out_dir / "runs.csv", summaries, records_by_name)

    if args.plots:
        title = f"{args.suite or 'custom'}: {len(configs)} конфигов, {args.runs} прогонов"
        path = plots.save_dashboard(records_by_name, out_dir / "dashboard.png", title=title)
        print(f"График: {path}")
        path = plots.save_final_points_bar(
            records_by_name, out_dir / "final_points.png", title=title
        )
        print(f"График: {path}")

    print()
    _print_table(
        _summary_rows(summaries),
        ["config", "очки", "arm_regret", "лучшая_рука", "доля_рук", "шаг_эпохи1", "эпоха", "КПД"],
    )
    print(f"\nРезультаты: {out_dir.resolve()}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Нет такой папки: {directory}", file=sys.stderr)
        return 2
    names = sorted(path.stem for path in directory.glob("*.npz"))
    if not names:
        print(f"В {directory} нет сохранённых прогонов (*.npz)", file=sys.stderr)
        return 2

    records_by_name = {name: RunRecords.load(directory, name) for name in names}
    summaries = {
        name: m.summarize(records, oracle_final=_oracle_finals(records_by_name, records))
        for name, records in records_by_name.items()
    }
    _print_table(
        _summary_rows(summaries),
        ["config", "очки", "arm_regret", "лучшая_рука", "доля_рук", "шаг_эпохи1", "эпоха", "КПД"],
    )
    if args.plots:
        path = plots.save_dashboard(
            records_by_name, directory / "dashboard.png", title=directory.name
        )
        print(f"График: {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    demo_args = argparse.Namespace(
        config=None,
        suite="demo",
        runs=args.runs,
        seed=args.seed,
        jobs=args.jobs,
        steps=args.steps,
        out=args.out,
        plots=True,
        quiet=args.quiet,
    )
    return cmd_run(demo_args)


def cmd_stats(args: argparse.Namespace) -> int:
    """Парные сравнения по ``runs.csv``: таблица + ``results.csv``."""
    path = Path(args.runs)
    if not path.is_file():
        print(f"Нет файла: {path}", file=sys.stderr)
        return 2
    try:
        values, metric_names = st.load_runs_csv(path)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    if args.reference not in values:
        print(
            f"Нет базлайна {args.reference!r}. Доступны: {', '.join(sorted(values))}",
            file=sys.stderr,
        )
        return 2

    metrics = list(args.metric) if args.metric else [
        metric for metric in TABLE_METRICS if metric in metric_names
    ]
    if not metrics:
        print(f"В {path.name} нет табличных метрик: {', '.join(metric_names)}", file=sys.stderr)
        return 2

    results: List[st.PairedResult] = []
    for metric in metrics:
        if metric not in metric_names:
            print(f"Метрики {metric!r} нет в {path.name}", file=sys.stderr)
            return 2
        results.extend(
            st.compare_values(
                values,
                args.reference,
                metric=metric,
                method=args.method,
                n_boot=args.n_boot,
                alpha=args.alpha,
                seed=args.seed,
            )
        )

    print(f"{path.name}: {len(values)} конфигов, базлайн {args.reference}, метод {args.method}")
    for metric in metrics:
        print()
        print(f"Метрика: {metric}")
        rows = st.format_rows([item for item in results if item.metric == metric])
        _print_table(rows, list(st.COMPARISON_COLUMNS))

    out = Path(args.out) if args.out else path.parent / "results.csv"
    st.write_results_csv(out, results)
    print(f"\nСравнения: {out.resolve()}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Собрать ``report.md``/``report.json`` по каталогу прогона."""
    directory = Path(args.directory)
    if not (directory / "runs.csv").is_file():
        print(f"Нет runs.csv в {directory}", file=sys.stderr)
        return 2
    try:
        report = build_report(
            directory,
            reference=args.reference,
            method=args.method,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )
    except KeyError as exc:
        print(f"Ошибка: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2

    markdown_path, json_path = write_report(directory, report)
    verdicts = ", ".join(
        f"{item['code']}: {item['status']}" for item in report.payload["hypotheses"]
    )
    print(f"Вердикты — {verdicts}")
    if not args.quiet:
        print()
        print(report.markdown)
    print(f"\nОтчёт: {markdown_path.resolve()}")
    print(f"JSON:  {json_path.resolve()}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Запустить веб-интерфейс (``bandit.web``): конструктор прогонов в браузере."""
    try:
        from .web.app import serve
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        print(
            "Веб-интерфейс требует fastapi и uvicorn: pip install -e .[ui] "
            f"(не найдено: {getattr(exc, 'name', exc)})",
            file=sys.stderr,
        )
        return 2
    return serve(
        host=args.host,
        port=args.port,
        open_browser=args.browser,
        reload=args.reload,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bandit.cli",
        description="Эксперименты с бандитами и метациклом обучения.",
    )
    parser.add_argument("--version", action="version", version=f"bandit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="показать среды, агентов, политики и конфиги")
    p_list.add_argument("-v", "--verbose", action="store_true", help="перечислить все конфиги")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="прогнать конфиги и сохранить результаты")
    p_run.add_argument("--config", action="append", metavar="NAME", help="имя конфига (можно несколько раз)")
    p_run.add_argument("--config-file", type=Path, default=None, metavar="PATH",
                       help="JSON со списком конфигов (RunConfig) — так запускает веб-интерфейс")
    p_run.add_argument("--suite", choices=sorted(SUITES), help="набор конфигов")
    p_run.add_argument("--runs", type=int, default=20, help="число прогонов на конфиг (по умолчанию 20)")
    p_run.add_argument("--seed", type=int, default=42, help="базовый сид (по умолчанию 42)")
    p_run.add_argument("--steps", type=int, default=None, help="переопределить горизонт")
    p_run.add_argument("--jobs", type=int, default=1, help="процессов (Windows: spawn)")
    p_run.add_argument("--out", type=Path, default=None, help="каталог результатов")
    p_run.add_argument("--no-plots", dest="plots", action="store_false", help="не строить графики")
    p_run.add_argument("--quiet", action="store_true", help="без прогресса")
    p_run.set_defaults(func=cmd_run, plots=True)

    p_show = sub.add_parser("show", help="сводка по сохранённому каталогу")
    p_show.add_argument("directory", type=Path, help="каталог с *.npz/*.json")
    p_show.add_argument("--plots", action="store_true", help="перерисовать графики")
    p_show.set_defaults(func=cmd_show)

    p_demo = sub.add_parser("demo", help="быстрая демонстрация (набор demo)")
    p_demo.add_argument("--runs", type=int, default=5, help="число прогонов (по умолчанию 5)")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--steps", type=int, default=1500, help="горизонт (по умолчанию 1500)")
    p_demo.add_argument("--jobs", type=int, default=1)
    p_demo.add_argument("--out", type=Path, default=Path("results") / "demo")
    p_demo.add_argument("--quiet", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_ui = sub.add_parser("ui", help="веб-интерфейс: конструктор прогонов в браузере")
    p_ui.add_argument("--host", default="127.0.0.1",
                      help="адрес (по умолчанию только локальный: 127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    p_ui.add_argument("--no-browser", dest="browser", action="store_false",
                      help="не открывать браузер автоматически")
    p_ui.add_argument("--reload", action="store_true",
                      help="автоперезагрузка кода (только для разработки)")
    p_ui.set_defaults(func=cmd_ui, browser=True)

    p_stats = sub.add_parser("stats", help="парные сравнения по runs.csv")
    p_stats.add_argument(
        "runs",
        type=Path,
        nargs="?",
        default=Path("results") / "demo" / "runs.csv",
        help="путь к runs.csv (по умолчанию results/demo/runs.csv)",
    )
    p_stats.add_argument(
        "--reference",
        default="pyramid-ucb1-none",
        help="базлайн: имя конфига (по умолчанию pyramid-ucb1-none)",
    )
    p_stats.add_argument(
        "--metric", action="append", metavar="NAME", help="метрика (можно несколько раз)"
    )
    p_stats.add_argument("--method", choices=list(st.METHODS), default="bootstrap",
                         help="тест для p-value (по умолчанию bootstrap)")
    p_stats.add_argument("--n-boot", type=int, default=st.DEFAULT_N_BOOT,
                         help=f"bootstrap-выборок (по умолчанию {st.DEFAULT_N_BOOT})")
    p_stats.add_argument("--alpha", type=float, default=0.05, help="уровень значимости")
    p_stats.add_argument("--seed", type=int, default=st.DEFAULT_SEED,
                         help=f"сид bootstrap (по умолчанию {st.DEFAULT_SEED})")
    p_stats.add_argument("--out", type=Path, default=None,
                         help="куда сохранить results.csv (по умолчанию рядом с runs.csv)")
    p_stats.set_defaults(func=cmd_stats)

    p_report = sub.add_parser("report", help="сводный отчёт: таблицы, сравнения, H1–H6")
    p_report.add_argument("directory", type=Path, help="каталог прогона (с runs.csv)")
    p_report.add_argument("--reference", default=None, help="базлайн (по умолчанию — первый конфиг `none`)")
    p_report.add_argument("--method", choices=list(st.METHODS), default="bootstrap",
                          help="тест для p-value (по умолчанию bootstrap)")
    p_report.add_argument("--n-boot", type=int, default=st.DEFAULT_N_BOOT,
                          help=f"bootstrap-выборок (по умолчанию {st.DEFAULT_N_BOOT})")
    p_report.add_argument("--alpha", type=float, default=0.05, help="уровень значимости")
    p_report.add_argument("--seed", type=int, default=st.DEFAULT_SEED,
                          help=f"сид bootstrap (по умолчанию {st.DEFAULT_SEED})")
    p_report.add_argument("--quiet", action="store_true", help="не печатать markdown")
    p_report.set_defaults(func=cmd_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
