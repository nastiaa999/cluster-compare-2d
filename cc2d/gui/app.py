"""Запуск приложения: журнал, перевод стандартных диалогов, оформление, главное окно."""

from __future__ import annotations

import logging
import sys
import traceback

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import __version__
from ..logging_setup import configure
from .main_window import MainWindow
from .style import STYLESHEET

log = logging.getLogger(__name__)


def _install_excepthook() -> None:
    """Непредвиденная ошибка пишется в журнал, пользователь получает сообщение (п. 4.2.5 ТЗ)."""

    def hook(exc_type, exc, tb):
        log.critical("Необработанное исключение:\n%s",
                     "".join(traceback.format_exception(exc_type, exc, tb)))
        QMessageBox.critical(None, "Ошибка программы",
                             f"Произошла непредвиденная ошибка: {exc}\n\n"
                             "Подробности записаны в журнал работы.")

    sys.excepthook = hook


def _fit_to_screen(window: MainWindow) -> None:
    """Размер окна по экрану: на экранах от 1366 × 768 окно помещается целиком."""
    screen = QGuiApplication.primaryScreen().availableGeometry()
    width, height = min(1400, screen.width()), min(860, screen.height())
    window.resize(width, height)
    if screen.width() <= 1400 or screen.height() <= 860:
        window.showMaximized()
    else:
        window.show()


def run(argv: list[str] | None = None) -> int:
    """Запускает программу и возвращает код завершения."""
    log_file = configure()
    log.info("Запуск ClusterCompare 2D %s, журнал: %s", __version__, log_file)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("ClusterCompare 2D")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    QLocale.setDefault(QLocale(QLocale.Language.Russian, QLocale.Country.Russia))
    translator = QTranslator(app)
    if translator.load("qtbase_ru", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(translator)
    _install_excepthook()
    window = MainWindow()
    _fit_to_screen(window)
    return app.exec()
