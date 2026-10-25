"""Запись файлов через временный файл с последующим переименованием (п. 4.2.3 ТЗ)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from .errors import ExportError


def atomic_write(path: str | Path, write: Callable[[str], None]) -> None:
    """Записывает файл атомарно.

    Функция write получает путь временного файла в той же папке и записывает
    в него содержимое. Затем временный файл заменяет целевой. При ошибке
    временный файл удаляется, а прежний файл с тем же именем остаётся целым.

    :raises ExportError: если записать файл не удалось
    """
    path = Path(path)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        write(tmp)
        os.replace(tmp, path)
        tmp = None
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise ExportError(f"Не удалось записать файл {path.name}: {reason}") from exc
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)
