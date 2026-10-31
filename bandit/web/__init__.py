"""Веб-интерфейс ``bandit``: конструктор прогонов, очередь заданий, просмотр метрик.

Пакет не импортирует FastAPI на уровне модуля пакета: ``bandit.cli`` остаётся
работоспособным без веб-зависимостей, а ``pip install -e .[ui]`` добавляет
``fastapi``/``uvicorn``. ``create_app``/``serve`` подгружаются лениво.
"""
from __future__ import annotations

from typing import Any

__all__ = ["create_app", "serve"]


def __getattr__(name: str) -> Any:
    """Ленивый доступ к фабрике приложения, чтобы импорт пакета был дёшев."""
    if name in ("create_app", "serve"):
        from .app import create_app, serve

        return {"create_app": create_app, "serve": serve}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
