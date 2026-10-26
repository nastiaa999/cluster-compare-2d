"""Выполнение расчёта в фоновом потоке (п. 4.1.4.7 ТЗ)."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from ..core.errors import CancelledError
from ..core.run_manager import RunManager

log = logging.getLogger(__name__)


class RunWorker(QObject):
    """Вызывает операцию «запустить» менеджера запуска в отдельном потоке.

    Результат и ошибки передаются в поток интерфейса сигналами Qt.
    """

    #: Прогресс: процент и текст для строки состояния.
    progress = Signal(int, str)
    #: Расчёт завершён, передаётся RunOutput.
    finished = Signal(object)
    #: Расчёт завершился ошибкой, передаётся исключение.
    failed = Signal(object)
    #: Расчёт отменён пользователем.
    cancelled = Signal()

    def __init__(self, manager: RunManager):
        super().__init__()
        self.manager = manager

    @Slot()
    def run(self) -> None:
        try:
            output = self.manager.run(self.progress.emit)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # ошибка передаётся в окно и пишется в журнал
            log.exception("Расчёт завершился ошибкой")
            self.failed.emit(exc)
        else:
            self.finished.emit(output)


def start_worker(manager: RunManager, parent: QObject) -> tuple[QThread, RunWorker]:
    """Создаёт поток и исполнителя; поток завершается сам после любого исхода расчёта."""
    thread = QThread(parent)
    worker = RunWorker(manager)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # quit вызывается прямо в потоке расчёта: главный поток может в этот момент
    # ждать завершения потока в wait(), и сообщение в его очередь не дошло бы.
    for signal in (worker.finished, worker.failed, worker.cancelled):
        signal.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    return thread, worker
