"""Класс «Менеджер запуска»: порядок обработки при запуске кластеризации."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from .errors import AllAlgorithmsFailedError, CancelledError, ParameterError
from .evaluator import QualityEvaluator
from .experiment import Experiment
from .preprocessing import PreparedData, Preprocessor
from .result import ClusteringResult
from .summary import SummaryTable

log = logging.getLogger(__name__)

IDLE, RUNNING, DONE, CANCELLED, FAILED = "ожидание", "расчёт", "завершён", "отменён", "ошибка"

#: Функция прогресса: (процент 0–100, текст для строки состояния).
ProgressCallback = Callable[[int, str], None]


@dataclass
class RunOutput:
    """Что менеджер возвращает главному окну (сообщение 1a на рис. 2.6).

    :param results: результаты выполненных алгоритмов
    :param data: подготовленные точки и эталонные метки
    :param table: сводная таблица с копией эксперимента
    :param failures: алгоритмы, завершившиеся ошибкой: (название, исключение)
    """

    results: list[ClusteringResult]
    data: PreparedData
    table: SummaryTable
    failures: list = field(default_factory=list)


class RunManager:
    """Менеджер запуска (рис. 2.5, 2.6).

    Владеет экспериментом (композиция). Метод :meth:`run` выполняется в
    фоновом потоке; :meth:`cancel` можно вызвать из потока интерфейса.
    Ошибка одного алгоритма не прерывает остальные (п. 4.2.4 ТЗ).

    :param preprocessor: предобработчик, связанный с текущим набором данных
    """

    def __init__(self, preprocessor: Preprocessor):
        self.experiment = Experiment()
        self.preprocessor = preprocessor
        self.evaluator = QualityEvaluator()
        self.status = IDLE
        self.progress = 0
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Просит остановить расчёт («отменить»)."""
        self._cancel.set()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise CancelledError()

    def _report(self, callback: ProgressCallback | None, percent: int, text: str) -> None:
        self.progress = percent
        if callback:
            callback(percent, text)

    def prepare(self) -> None:
        """Готовит запуск: сбрасывает прежнюю просьбу об отмене.

        Интерфейс вызывает этот метод в своём потоке до старта фонового потока,
        поэтому «Отменить», нажатая сразу после «Запустить», не теряется.
        """
        self._cancel.clear()
        self.status = RUNNING

    def run(self, progress: ProgressCallback | None = None) -> RunOutput:
        """Выполняет расчёт («запустить»).

        Если :meth:`prepare` не вызывался, он вызывается здесь.

        :raises CancelledError: если расчёт отменён
        :raises AllAlgorithmsFailedError: если не выполнился ни один алгоритм
        :raises DataPreparationError: если набор не подготовлен
        """
        if self.status != RUNNING:
            self.prepare()
        try:
            output = self._run(progress)
        except CancelledError:
            self.status = CANCELLED
            log.info("Расчёт отменён")
            raise
        except Exception:
            self.status = FAILED
            raise
        self.status = DONE
        return output

    def _run(self, progress: ProgressCallback | None) -> RunOutput:
        snapshot = self.experiment.copy()
        if not snapshot.algorithms:
            raise ParameterError("Не отмечен ни один алгоритм")
        log.info("Запуск расчёта: %s", self._describe(snapshot))

        self._report(progress, 0, "Подготовка данных")
        self.preprocessor.method = snapshot.preprocessing
        data = self.preprocessor.process(snapshot.x_column, snapshot.y_column,
                                         snapshot.label_column)
        self._check_cancel()

        results, failures = [], []
        count = len(snapshot.algorithms)
        for i, algorithm in enumerate(snapshot.algorithms):
            self._report(progress, 5 + 85 * i // count, f"Выполняется {algorithm.title}")
            try:
                result = algorithm.run(data.points, snapshot.seed, self._check_cancel)
            except CancelledError:
                raise
            except Exception as exc:
                log.exception("Ошибка алгоритма %s", algorithm.title)
                failures.append((algorithm.title, exc))
                continue
            log.info("%s: %.3f с, кластеров %d, шум %d", algorithm.title, result.elapsed,
                     result.n_clusters, result.n_noise)
            results.append(result)

        self._report(progress, 90, "Вычисление метрик")
        evaluated = []
        for result in results:
            self._check_cancel()
            try:
                self.evaluator.evaluate(result, data.points, data.reference_labels)
            except Exception as exc:
                log.exception("Ошибка вычисления метрик для %s", result.algorithm)
                failures.append((result.algorithm, exc))
                continue
            evaluated.append(result)
        if not evaluated:
            raise AllAlgorithmsFailedError(failures)

        table = SummaryTable()
        for result in evaluated:
            table.add(result)
        table.highlight_best()
        table.set_params(snapshot)
        self._report(progress, 100, "Расчёт завершён")
        return RunOutput(evaluated, data, table, failures)

    @staticmethod
    def _describe(experiment: Experiment) -> str:
        algorithms = ", ".join(f"{a.title} {a.parsed_params()}" for a in experiment.algorithms)
        return (f"X={experiment.x_column}, Y={experiment.y_column}, "
                f"метки={experiment.label_column}, предобработка={experiment.preprocessing}, "
                f"seed={experiment.seed}, алгоритмы: {algorithms}")
