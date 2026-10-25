"""Абстрактный класс «Алгоритм кластеризации»."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..errors import ParameterError
from ..result import ClusteringResult

#: Функция проверки отмены: выбрасывает CancelledError, если расчёт отменён.
CancelCheck = Callable[[], None]


def _no_cancel() -> None:
    """Проверка отмены по умолчанию: расчёт никогда не отменяется."""


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Квадраты евклидовых расстояний между строками a и b (матрица len(a) × len(b))."""
    d = (a[:, None, 0] - b[None, :, 0]) ** 2
    d += (a[:, None, 1] - b[None, :, 1]) ** 2
    return d


class ClusteringAlgorithm(ABC):
    """Алгоритм кластеризации (абстрактный класс, рис. 2.5).

    Наследник задаёт :attr:`key`, :attr:`title`, описания параметров
    :attr:`PARAMS` и реализует :meth:`_fit`. Значения параметров хранятся в
    :attr:`params` в том виде, в каком их ввёл пользователь (строки или числа),
    и разбираются при проверке.
    """

    #: Короткое имя для файлов и файла эксперимента.
    key: str = ""
    #: Название для пользователя.
    title: str = ""
    #: Описания параметров {имя: описание из модуля params}.
    PARAMS: dict = {}

    def __init__(self, **params):
        self.params: dict = {}
        self.reset_params()
        for name, value in params.items():
            if name not in self.PARAMS:
                raise ParameterError(f"{self.title}: неизвестный параметр «{name}»")
            self.params[name] = value

    def reset_params(self) -> None:
        """Возвращает параметрам значения по умолчанию (кнопка «По умолчанию»)."""
        self.params = {name: p.default for name, p in self.PARAMS.items()}

    def errors(self, n_points: int | None = None) -> dict[str, str]:
        """Возвращает ошибки параметров {имя: причина}; пусто, если всё верно.

        :param n_points: число точек набора; если задано, число кластеров не
            должно его превышать
        """
        result = {}
        for name, spec in self.PARAMS.items():
            try:
                value = spec.parse(self.params[name])
            except ParameterError as exc:
                result[name] = str(exc)
                continue
            if name == "n_clusters" and n_points is not None and value > n_points:
                result[name] = f"Число кластеров больше числа точек ({n_points})"
        return result

    def validate_params(self, n_points: int | None = None) -> None:
        """Проверяет параметры (входит в условие П4).

        :raises ParameterError: с причиной первой найденной ошибки
        """
        errors = self.errors(n_points)
        if errors:
            raise ParameterError(next(iter(errors.values())))

    def parsed_params(self) -> dict:
        """Разобранные значения параметров (после успешной проверки)."""
        return {name: spec.parse(self.params[name]) for name, spec in self.PARAMS.items()}

    def run(self, points: np.ndarray, seed: int,
            cancel_check: CancelCheck | None = None) -> ClusteringResult:
        """Выполняет кластеризацию и создаёт результат.

        :param points: точки после предобработки, массив n × 2
        :param seed: начальное значение генератора случайных чисел
        :param cancel_check: функция проверки отмены
        """
        points = np.asarray(points, dtype=float)
        self.validate_params(len(points))
        start = time.perf_counter()
        labels = self._fit(points, seed, self.parsed_params(), cancel_check or _no_cancel)
        elapsed = time.perf_counter() - start
        return ClusteringResult(self.title, self.key, labels, elapsed)

    @abstractmethod
    def _fit(self, points: np.ndarray, seed: int, params: dict,
             cancel_check: CancelCheck) -> np.ndarray:
        """Возвращает метки кластеров с нуля, шум −1."""

    def to_dict(self) -> dict:
        """Описание алгоритма для файла эксперимента."""
        return {"name": self.key, "params": self.parsed_params()}
