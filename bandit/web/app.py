"""FastAPI-приложение веб-интерфейса ``bandit.web``.

Маршруты:

* ``GET  /api/health`` — версия, каталог результатов, активные задания;
* ``GET  /api/schema`` — всё для конструктора (``web.schema.ui_schema``);
* ``POST /api/preview`` — сколько конфигов/шагов и оценка времени;
* ``POST /api/validate`` — дешёвый прогон 32 шага (ловит битые параметры);
* ``POST /api/jobs``, ``GET /api/jobs``, ``GET /api/jobs/{id}``,
  ``POST /api/jobs/{id}/cancel`` — очередь заданий;
* ``GET  /api/runs``, ``GET /api/runs/{id}``, ``POST /api/runs/{id}/report``,
  ``GET /api/runs/{id}/file/{name}`` — просмотр готовых прогонов.

Статика монтируется последней (``/``), иначе она перехватила бы ``/api/*``.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import threading
import webbrowser
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from . import schema
from .jobs import RESULTS_DIR, JobManager
from .models import (
    ExperimentSpec,
    Health,
    JobView,
    PreviewView,
    RunDetail,
    RunSummary,
    ValidateView,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
"""Собранная статика: ``index.html``, ``app.js``, ``style.css`` (без сборки)."""

FILES = frozenset({
    "dashboard.png",
    "final_points.png",
    "report.md",
    "report.json",
    "metrics.csv",
    "runs.csv",
    "manifest.json",
    "spec.json",
    "configs.json",
})
"""Что можно отдать браузеру: только известные файлы прогона."""

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _num(value: Any) -> Optional[float]:
    """Число для JSON: ``None`` вместо ``nan``/``inf`` (иначе ``JSON.parse`` падает)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics_rows(directory: Path) -> List[Dict[str, Any]]:
    """``metrics.csv`` → строки таблицы: конфиг + метрика → {mean, sem, n}."""
    path = directory / "metrics.csv"
    if not path.is_file():
        return []
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            config = str(row.get("config") or "")
            metric = str(row.get("metric") or "")
            if not config or not metric:
                continue
            entry = rows.setdefault(config, {"config": config, "values": {}})
            entry["values"][metric] = {
                "mean": _num(row.get("mean")),
                "sem": _num(row.get("sem")),
                "n": _num(row.get("n")),
            }
    return sorted(
        rows.values(),
        key=lambda item: -(item["values"].get("final_points", {}).get("mean") or 0.0),
    )


def _run_summary(directory: Path) -> Optional[RunSummary]:
    manifest = _read_json(directory / "manifest.json")
    if manifest is None:
        return None
    return RunSummary(
        id=directory.name,
        created=str(manifest.get("created") or ""),
        n_runs=int(manifest.get("n_runs") or 0),
        n_configs=len(manifest.get("configs") or []),
        suite=manifest.get("suite"),
        elapsed_sec=float(manifest.get("elapsed_sec") or 0.0),
        has_report=(directory / "report.json").is_file(),
        has_plots=(directory / "dashboard.png").is_file(),
    )


def _list_runs(results_dir: Path) -> List[RunSummary]:
    if not results_dir.is_dir():
        return []
    summaries = [
        summary
        for summary in (_run_summary(path) for path in results_dir.iterdir() if path.is_dir())
        if summary is not None
    ]
    return sorted(summaries, key=lambda item: item.created, reverse=True)


def _run_dir(results_dir: Path, run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Недопустимый id прогона")
    root = results_dir.resolve()
    path = (root / run_id).resolve()
    if path.parent != root:
        raise HTTPException(status_code=400, detail="Путь вне каталога результатов")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Нет прогона {run_id!r}")
    return path


def _run_detail(directory: Path) -> RunDetail:
    manifest = _read_json(directory / "manifest.json") or {}
    report = _read_json(directory / "report.json") or {}
    configs = [
        str(item.get("name"))
        for item in manifest.get("configs") or []
        if isinstance(item, dict) and item.get("name")
    ]
    n_steps = manifest.get("n_steps_override") or 0
    if not n_steps:
        first = next(
            (item for item in manifest.get("configs") or [] if isinstance(item, dict)),
            {},
        )
        n_steps = first.get("n_steps") or 0
    hypotheses = [
        {
            "code": str(item.get("code") or ""),
            "title": str(item.get("title") or ""),
            "status": str(item.get("status") or ""),
            "detail": str(item.get("detail") or ""),
        }
        for item in report.get("hypotheses") or []
        if isinstance(item, dict)
    ]
    return RunDetail(
        id=directory.name,
        created=str(manifest.get("created") or ""),
        n_runs=int(manifest.get("n_runs") or 0),
        base_seed=int(manifest.get("base_seed") or 0),
        n_steps=int(n_steps),
        elapsed_sec=float(manifest.get("elapsed_sec") or 0.0),
        configs=configs,
        metrics=_metrics_rows(directory),
        hypotheses=hypotheses,
        has_report=(directory / "report.json").is_file(),
        has_plots=(directory / "dashboard.png").is_file(),
    )


def _build_report(directory: Path) -> None:
    """Пересчитать ``report.md``/``report.json`` тем же кодом, что и CLI."""
    from ..cli import main as cli_main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_main(["report", str(directory)])
    if code != 0:
        tail = " | ".join(buffer.getvalue().strip().splitlines()[-3:])
        raise HTTPException(
            status_code=500,
            detail=f"bandit.cli report завершился с ошибкой: {tail}",
        )


def create_app(results_dir: Path | str | None = None) -> FastAPI:
    """Собрать приложение с собственным каталогом результатов (удобно в тестах)."""
    base = Path(results_dir) if results_dir is not None else RESULTS_DIR
    manager = JobManager(base)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="bandit UI",
        version=__version__,
        description="Конструктор прогонов: среда, агенты, мета-политики, метрики.",
        lifespan=lifespan,
    )

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        return Health(
            version=__version__,
            results_dir=str(base),
            jobs_running=manager.active(),
            steps_per_second=schema.EST_STEPS_PER_SECOND,
        )

    @app.get("/api/schema")
    def get_schema() -> Dict[str, Any]:
        return schema.ui_schema()

    @app.post("/api/preview", response_model=PreviewView)
    def preview(spec: ExperimentSpec) -> PreviewView:
        payload = spec.model_dump()
        try:
            configs = schema.build_configs(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PreviewView(**schema.preview(configs, payload))

    @app.post("/api/validate", response_model=ValidateView)
    def validate(spec: ExperimentSpec) -> ValidateView:
        return ValidateView(**schema.validate_spec(spec.model_dump()))

    @app.post("/api/jobs", response_model=JobView)
    def create_job(spec: ExperimentSpec) -> JobView:
        payload = spec.model_dump()
        try:
            configs = schema.build_configs(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        check = schema.validate_spec(payload)
        if not check["ok"]:
            raise HTTPException(status_code=400, detail="; ".join(check["errors"]))
        return manager.submit(payload, configs).view()

    @app.get("/api/jobs", response_model=List[JobView])
    def list_jobs() -> List[JobView]:
        return [job.view() for job in manager.jobs()]

    @app.get("/api/jobs/{job_id}", response_model=JobView)
    def get_job(job_id: str) -> JobView:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Нет задания {job_id!r}")
        return job.view()

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobView)
    def cancel_job(job_id: str) -> JobView:
        job = manager.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Нет задания {job_id!r}")
        return job.view()

    @app.get("/api/runs", response_model=List[RunSummary])
    def list_runs() -> List[RunSummary]:
        return _list_runs(base)

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        return _run_detail(_run_dir(base, run_id))

    @app.post("/api/runs/{run_id}/report", response_model=RunDetail)
    def make_report(run_id: str) -> RunDetail:
        directory = _run_dir(base, run_id)
        if not (directory / "runs.csv").is_file():
            raise HTTPException(status_code=400, detail="В прогоне нет runs.csv")
        _build_report(directory)
        return _run_detail(directory)

    @app.get("/api/runs/{run_id}/file/{name}")
    def get_run_file(run_id: str, name: str) -> FileResponse:
        directory = _run_dir(base, run_id)
        if name not in FILES:
            raise HTTPException(status_code=404, detail=f"Файл {name!r} не отдаётся")
        path = directory / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Нет файла {name!r}")
        return FileResponse(path)

    if STATIC_DIR.is_dir():
        # Монтируется последним: иначе StaticFiles перехватил бы /api/*.
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    reload: bool = False,
) -> int:
    """Запустить uvicorn; ``reload`` полезен при правке статики и кода."""
    import uvicorn

    url = f"http://{host}:{port}/"
    print(f"bandit UI: {url}  (Ctrl+C — остановить)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    if reload:
        uvicorn.run("bandit.web.app:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(serve())


