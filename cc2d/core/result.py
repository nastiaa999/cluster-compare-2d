"""Класс «Результат кластеризации»."""

from __future__ import annotations

import numpy as np

#: Метка точки шума в массиве меток (табл. 2.2 РПЗ).
NOISE = -1


class ClusteringResult:
    """Результат кластеризации (рис. 2.5).

    Метки хранятся с нуля, шум обозначается −1. Пользователю номера кластеров
    показываются с единицы (решение 12.5).

    :param algorithm: название алгоритма для пользователя
    :param key: короткое имя алгоритма (для имён файлов)
    :param labels: метки кластеров, по одной на точку рабочего набора
    :param elapsed: время выполнения алгоритма, с
    """

    def __init__(self, algorithm: str, key: str, labels: np.ndarray, elapsed: float):
        self.algorithm = algorithm
        self.key = key
        self._labels = np.asarray(labels, dtype=int)
        self.n_clusters = int(np.unique(self._labels[self._labels != NOISE]).size)
        self.n_noise = int((self._labels == NOISE).sum())
        self.elapsed = float(elapsed)
        self.message = ""
        self.metrics: dict[str, float | None] = {}

    @property
    def n_points(self) -> int:
        """Число точек, к которым применён алгоритм."""
        return int(self._labels.size)

    def set_metrics(self, metrics: dict[str, float | None]) -> None:
        """Записывает метрики качества (вызывает оценщик качества)."""
        self.metrics = dict(metrics)

    def set_message(self, message: str) -> None:
        """Записывает замечание о возможной нерепрезентативности результата."""
        self.message = message

    def get_labels(self) -> np.ndarray:
        """Возвращает копию меток кластеров."""
        return self._labels.copy()
