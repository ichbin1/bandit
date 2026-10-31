"""Менеджер заданий веб-интерфейса: очередь, подпроцесс, лог, отмена.

Задание — это обычный ``bandit.cli run`` в отдельном процессе. Так бесплатно
получаются три важные вещи:

* отмена: ``Popen.terminate`` действительно останавливает счёт;
* ``--jobs > 1`` работает как в CLI (никаких spawn-проблем на Windows);
* сервер не наследует состояние прогона, поэтому падение задания не уносит UI.

Спецификация и конфиги пишутся в ``results/<id>/`` (``spec.json``,
``configs.json``), так что прогон воспроизводим из командной строки.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

from ..runner import RunConfig
from .models import JobView

ROOT = Path(__file__).resolve().parents[2]
"""Корень репозитория: там лежат ``results/`` и пакет ``bandit``."""

RESULTS_DIR = Path(os.environ.get("BANDIT_RESULTS_DIR") or (ROOT / "results"))
"""Каталог результатов; ``BANDIT_RESULTS_DIR`` удобен для тестов и демо."""

LOG_LIMIT = 300
"""Сколько последних строк лога держим в памяти (для UI хватает)."""


@dataclass
class Job:
    """Одно задание: спецификация, прогресс, хвост лога, процесс."""

    id: str
    title: str
    out_dir: Path
    spec_path: Path
    configs_path: Path
    n_configs: int
    runs: int
    steps: int
    jobs: int
    seed: int
    plots: bool
    state: str = "queued"
    error: Optional[str] = None
    created: datetime = field(default_factory=datetime.now)
    started: Optional[datetime] = None
    finished: Optional[datetime] = None
    cancel_requested: bool = False
    process: Optional[subprocess.Popen] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LIMIT))

    def elapsed(self) -> float:
        end = self.finished or datetime.now()
        start = self.started or self.created
        return max(0.0, (end - start).total_seconds())

    def n_done(self) -> int:
        """Готовые конфиги: CLI пишет по ``<name>.npz`` на конфиг."""
        try:
            return len(list(self.out_dir.glob("*.npz")))
        except OSError:  # pragma: no cover - каталог могли удалить
            return 0

    def view(self) -> JobView:
        done = self.n_done()
        if self.state == "done" or self.n_configs <= 0:
            progress = 1.0 if self.state == "done" else 0.0
        else:
            progress = min(1.0, done / self.n_configs)
        return JobView(
            id=self.id,
            title=self.title,
            state=self.state,
            out_dir=str(self.out_dir),
            created=self.created.isoformat(timespec="seconds"),
            n_configs=self.n_configs,
            n_done=done,
            progress=round(progress, 4),
            runs=self.runs,
            steps=self.steps,
            jobs=self.jobs,
            elapsed_sec=round(self.elapsed(), 1),
            log=list(self.log),
            error=self.error,
        )


def _slug(text: str) -> str:
    """Безопасный кусок имени каталога (кириллицу заменяет на дефисы)."""
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "-", str(text)).strip("-.")
    return cleaned[:40] or "run"


def _tail(lines: Deque[str], limit: int = 8) -> str:
    return " | ".join(list(lines)[-limit:])


class JobManager:
    """Очередь заданий: один воркер, поэтому БД не нужна и CPU не делится.

    Параллелизм внутри задания задаётся ``--jobs`` (процессы CLI), а очередь
    остаётся последовательной: пользователь видит предсказуемый прогресс.
    """

    def __init__(self, results_dir: Path | str = RESULTS_DIR) -> None:
        self.results_dir = Path(results_dir)
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self._queue: "queue.Queue[Optional[Job]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._loop, name="bandit-ui-jobs", daemon=True
        )
        self._worker.start()

    # --- публичное API ---------------------------------------------------
    def submit(self, spec: Mapping[str, Any], configs: Sequence[RunConfig]) -> Job:
        """Создать задание и поставить его в очередь."""
        if not configs:
            raise ValueError("Нет конфигов для запуска")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slug(str(spec.get("title") or "") or "ui")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            job_id = f"ui-{stamp}-{slug}"
            counter = 2
            while job_id in self._jobs or (self.results_dir / job_id).exists():
                job_id = f"ui-{stamp}-{slug}-{counter}"
                counter += 1
            out_dir = self.results_dir / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            spec_path = out_dir / "spec.json"
            configs_path = out_dir / "configs.json"
            spec_path.write_text(
                json.dumps(dict(spec), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            configs_path.write_text(
                json.dumps(
                    [config.to_dict() for config in configs],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            job = Job(
                id=job_id,
                title=str(spec.get("title") or job_id),
                out_dir=out_dir,
                spec_path=spec_path,
                configs_path=configs_path,
                n_configs=len(configs),
                runs=int(spec.get("runs") or 30),
                steps=int(configs[0].n_steps),
                jobs=int(spec.get("jobs") or 1),
                seed=int(spec.get("seed") or 0),
                plots=bool(spec.get("plots", True)),
            )
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job)
        return job

    def jobs(self) -> List[Job]:
        """Все задания, новые сверху."""
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order)]

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        """Отменить задание: в очереди — сразу, в работе — ``terminate``."""
        job = self.get(job_id)
        if job is None:
            return None
        job.cancel_requested = True
        process = job.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:  # pragma: no cover - процесс уже умер
                pass
        elif job.state == "queued":
            job.state = "cancelled"
            job.finished = datetime.now()
            job.log.append("отменено до запуска")
        return job

    def active(self) -> int:
        """Сколько заданий сейчас выполняется (для ``/api/health``)."""
        return sum(1 for job in self.jobs() if job.state == "running")

    def shutdown(self) -> None:
        """Остановить воркер и активный процесс (вызывается на выходе из UI)."""
        for job in self.jobs():
            process = job.process
            if process is not None and process.poll() is None:
                job.cancel_requested = True
                try:
                    process.terminate()
                except OSError:  # pragma: no cover
                    pass
        self._queue.put(None)
        self._worker.join(timeout=5.0)

    # --- внутреннее ------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._run(job)
            except Exception as exc:  # noqa: BLE001 - задание не должно убивать воркер
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = datetime.now()
                job.log.append(job.error)
            finally:
                self._queue.task_done()

    def _command(self, job: Job) -> List[str]:
        """Команда запуска: тот же путь, что у пользователя в терминале."""
        command = [
            sys.executable, "-m", "bandit.cli", "run",
            "--config-file", str(job.configs_path),
            "--runs", str(job.runs),
            "--seed", str(job.seed),
            "--jobs", str(job.jobs),
            "--out", str(job.out_dir),
        ]
        if not job.plots:
            command.append("--no-plots")
        return command

    def _run(self, job: Job) -> None:
        if job.cancel_requested:
            job.state = "cancelled"
            job.finished = datetime.now()
            return
        job.state = "running"
        job.started = datetime.now()
        command = self._command(job)
        job.log.append(
            f"[{job.started:%H:%M:%S}] старт: {job.n_configs} конфигов × "
            f"{job.runs} прогонов × {job.steps} шагов (jobs={job.jobs})"
        )
        job.log.append("$ " + subprocess.list2cmdline(command))
        env = dict(os.environ)
        # Дочерний процесс должен писать UTF-8: при выводе в pipe Windows берёт
        # cp1251, и кириллица в логе превращается в мусор.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        job.process = process
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                job.log.append(text)
        code = process.wait()
        job.finished = datetime.now()
        if job.cancel_requested:
            job.state = "cancelled"
        elif code == 0:
            job.state = "done"
        else:
            job.state = "failed"
            job.error = f"код возврата {code}: {_tail(job.log)}"
        job.log.append(f"[{job.finished:%H:%M:%S}] {job.state}, {job.elapsed():.1f} с")


