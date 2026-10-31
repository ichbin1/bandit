"""Pydantic-модели веб-интерфейса: спецификация эксперимента и ответы API.

Модели — единственное место, где описан контракт с браузером. ``extra="forbid"``
выбран осознанно: опечатка в имени параметра (``max_arm`` вместо ``max_arms``)
должна давать 422 с путём до поля, а не молча игнорироваться.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..envs import ENVIRONMENTS


class EnvSpec(BaseModel):
    """Среда: ``kind`` + параметры конструктора (сливаются с пресетом)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "layered"
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        kind = str(value).strip().lower()
        if kind not in ENVIRONMENTS:
            raise ValueError(
                f"Неизвестная среда {value!r}. Доступны: {', '.join(sorted(ENVIRONMENTS))}"
            )
        return kind


class AgentSpec(BaseModel):
    """Агент: ключ из ``configs.AGENTS`` + переопределения параметров."""

    model_config = ConfigDict(extra="forbid")

    key: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PolicySpec(BaseModel):
    """Мета-политика: ключ из ``configs.POLICIES`` + переопределения параметров."""

    model_config = ConfigDict(extra="forbid")

    key: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    """Полная спецификация эксперимента (то, что редактирует пользователь)."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    env: EnvSpec = Field(default_factory=EnvSpec)
    agents: List[AgentSpec] = Field(default_factory=lambda: [AgentSpec(key="ucb1")])
    policies: List[PolicySpec] = Field(
        default_factory=lambda: [PolicySpec(key="none"), PolicySpec(key="full_cycle")]
    )
    runs: int = Field(30, ge=1, le=1000)
    seed: int = Field(42, ge=0)
    steps: int = Field(3000, ge=16, le=200_000)
    jobs: int = Field(1, ge=1, le=16)
    plots: bool = True
    add_oracle: bool = True


class PreviewView(BaseModel):
    """Ответ ``/api/preview``: сколько будет посчитано и сколько это займёт."""

    configs: List[str]
    n_configs: int
    n_episodes: int
    total_steps: int
    est_seconds: float


class ValidateView(BaseModel):
    """Ответ ``/api/validate``: дешёвый прогон по 32 шага на конфиг."""

    ok: bool
    n_configs: int
    errors: List[str]


class JobView(BaseModel):
    """Состояние задания: прогресс, хвост лога, ошибка."""

    id: str
    title: str
    state: str
    out_dir: str
    created: str
    n_configs: int
    n_done: int
    progress: float
    runs: int
    steps: int
    jobs: int
    elapsed_sec: float
    log: List[str]
    error: Optional[str] = None


class RunSummary(BaseModel):
    """Строка списка прогонов (читается из ``results/<id>/manifest.json``)."""

    id: str
    created: str
    n_runs: int
    n_configs: int
    suite: Optional[str] = None
    elapsed_sec: float
    has_report: bool
    has_plots: bool


class RunDetail(BaseModel):
    """Содержимое прогона: таблица метрик и вердикты гипотез."""

    id: str
    created: str
    n_runs: int
    base_seed: int
    n_steps: int
    elapsed_sec: float
    configs: List[str]
    metrics: List[Dict[str, Any]]
    hypotheses: List[Dict[str, Any]]
    has_report: bool
    has_plots: bool


class Health(BaseModel):
    """``/api/health``: версия, каталог результатов, активные задания."""

    version: str
    results_dir: str
    jobs_running: int
    steps_per_second: float
