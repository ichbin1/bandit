"""``python -m bandit.web`` — то же, что ``bandit.cli ui``, но без остальных команд."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .app import serve


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bandit.web",
        description="Веб-интерфейс bandit: конструктор прогонов в браузере.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    parser.add_argument("--no-browser", dest="browser", action="store_false",
                        help="не открывать браузер автоматически")
    args = parser.parse_args(argv)
    return serve(host=args.host, port=args.port, open_browser=args.browser)


if __name__ == "__main__":  # pragma: no cover - ручной запуск
    raise SystemExit(main())
