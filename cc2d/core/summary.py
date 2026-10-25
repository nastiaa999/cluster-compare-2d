"""Класс «Сводная таблица»."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .evaluator import (ARI, CALINSKI_HARABASZ, DAVIES_BOULDIN, EXTERNAL_METRICS, NMI,
                        SILHOUETTE)
from .result import ClusteringResult

if TYPE_CHECKING:
    from .experiment import Experiment

TIME = "time"

#: Столбцы таблицы: (ключ, заголовок), порядок по ТЗ п. 4.1.1.4 и рис. 2.14.
COLUMNS = (
    ("algorithm", "Алгоритм"),
    ("n_clusters", "Кластеров"),
    ("n_noise", "Шум"),
    (SILHOUETTE, "Силуэт"),
    (DAVIES_BOULDIN, "Дэвис–Болдин"),
    (CALINSKI_HARABASZ, "Калински–Харабаш"),
    (ARI, "ARI"),
    (NMI, "NMI"),
    (TIME, "Время, с"),
    ("message", "Замечание"),
)

#: Направление выбора лучшего значения (табл. 1.1 РПЗ): True — максимум.
BEST_IS_MAX = {SILHOUETTE: True, DAVIES_BOULDIN: False, CALINSKI_HARABASZ: True,
               ARI: True, NMI: True, TIME: False}

#: Число знаков после запятой на экране.
DISPLAY_DECIMALS = {SILHOUETTE: 2, DAVIES_BOULDIN: 2, CALINSKI_HARABASZ: 1,
                    ARI: 2, NMI: 2, TIME: 2}

#: Обозначение неопределённой внутренней метрики.
UNDEFINED = "–"


def format_decimal(value: float, decimals: int) -> str:
    """Число с заданным числом знаков и десятичной запятой."""
    return f"{value:.{decimals}f}".replace(".", ",")


class SummaryTable:
    """Сводная таблица (рис. 2.5): строки результатов и параметры расчёта.

    Параметры расчёта — копия эксперимента, с которой получены результаты;
    её записывает менеджер запуска, а окно экспорта сохраняет в JSON.
    """

    def __init__(self):
        self.rows: list[ClusteringResult] = []
        self.params: Experiment | None = None
        #: Номера строк с лучшим значением по каждому столбцу метрик.
        self.best: dict[str, set[int]] = {}

    def add(self, result: ClusteringResult) -> None:
        """Добавляет оценённый результат."""
        self.rows.append(result)

    def set_params(self, experiment_copy: Experiment) -> None:
        """Записывает копию эксперимента («записатьПараметры»)."""
        self.params = experiment_copy

    def get_params(self) -> Experiment | None:
        """Возвращает копию эксперимента («получитьПараметры»)."""
        return self.params

    def value(self, row: int, column: str):
        """Значение ячейки.

        Для метрик: число; None, если внутренняя метрика не определена;
        отсутствующее значение внешней метрики тоже None (см. :meth:`is_computed`).
        """
        result = self.rows[row]
        if column == "algorithm":
            return result.algorithm
        if column == "n_clusters":
            return result.n_clusters
        if column == "n_noise":
            return result.n_noise
        if column == TIME:
            return result.elapsed
        if column == "message":
            return result.message
        return result.metrics.get(column)

    def is_computed(self, row: int, column: str) -> bool:
        """Вычислялась ли метрика: внешние метрики без эталонных меток не вычисляются."""
        return column not in EXTERNAL_METRICS or column in self.rows[row].metrics

    def display(self, row: int, column: str) -> str:
        """Текст ячейки на экране: «–» для неопределённой метрики, пусто без меток."""
        value = self.value(row, column)
        if column in DISPLAY_DECIMALS:
            if not self.is_computed(row, column):
                return ""
            if value is None:
                return UNDEFINED
            return format_decimal(value, DISPLAY_DECIMALS[column])
        return str(value)

    def highlight_best(self) -> None:
        """Отмечает лучшие значения («выделитьЛучшее»).

        Сравниваются значения, округлённые до отображаемых знаков, поэтому при
        равенстве выделяются все строки с лучшим значением.
        """
        self.best = {}
        for column, is_max in BEST_IS_MAX.items():
            values = {}
            for row in range(len(self.rows)):
                value = self.value(row, column)
                if value is not None and self.is_computed(row, column):
                    values[row] = round(value, DISPLAY_DECIMALS[column])
            if values:
                target = max(values.values()) if is_max else min(values.values())
                self.best[column] = {row for row, v in values.items() if v == target}

    def is_best(self, row: int, column: str) -> bool:
        """Выделено ли значение ячейки как лучшее."""
        return row in self.best.get(column, set())
