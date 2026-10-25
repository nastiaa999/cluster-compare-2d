"""Настройки Sphinx для документации исходного кода CC2D (п. 5.2 ТЗ).

Документация формируется из строк документации модулей: autosummary с
параметром recursive сам находит все модули пакета cc2d, поэтому новый
модуль попадает в документацию без правки этого файла.
Сборка: ``.venv\Scripts\python -m sphinx -b html docs docs/_build/html``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import cc2d  # noqa: E402

project = "ClusterCompare 2D"
author = "А. С. Касатикова"
copyright = "2026, А. С. Касатикова"
version = release = cc2d.__version__
language = "ru"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # Реэкспорт из __init__ не документируется повторно: класс описан в своём модуле.
    "ignore-module-all": True,
}
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_title = "ClusterCompare 2D — документация исходного кода"
