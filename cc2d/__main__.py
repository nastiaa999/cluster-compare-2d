"""Точка входа: ``python -m cc2d`` открывает главное окно программы."""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from cc2d.gui.app import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
