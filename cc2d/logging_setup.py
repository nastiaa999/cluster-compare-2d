"""Настройка журнала работы (п. 4.2.5 ТЗ, решение 12.12)."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: Размер файла журнала до ротации и число старых файлов.
MAX_BYTES = 1024 * 1024
BACKUP_COUNT = 3


def log_path() -> Path:
    """Путь к журналу: %LOCALAPPDATA%\\CC2D\\cc2d.log (папка программы может быть закрыта для записи)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "CC2D" / "cc2d.log"


def configure(path: Path | None = None) -> Path:
    """Подключает запись журнала в файл с ротацией и возвращает путь к нему."""
    path = path or log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return path
