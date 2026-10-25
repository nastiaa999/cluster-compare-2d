"""Алгоритм K-Means (метод Ллойда)."""

from __future__ import annotations

import numpy as np

from ..params import IntParam
from .base import CancelCheck, ClusteringAlgorithm, squared_distances


class KMeans(ClusteringAlgorithm):
    """K-Means: разбиение на k кластеров с минимальной суммой квадратов расстояний до центров.

    Начальные центры — k различных точек набора, выбранных случайно по seed.
    Итерация приписывает точки к ближайшему центру и пересчитывает центры;
    расчёт заканчивается, когда разбиение перестаёт меняться или достигнуто
    максимальное число итераций. Если кластер опустел, его центр переносится
    в точку, наиболее удалённую от своего центра, поэтому кластеров всегда k.
    """

    key = "kmeans"
    title = "K-Means"
    PARAMS = {
        "n_clusters": IntParam("n_clusters", "Число кластеров", 2, 20, 3),
        "max_iter": IntParam("max_iter", "Макс. итераций", 1, 1000, 500),
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._initial_centers: np.ndarray | None = None
        #: Центры кластеров после последнего выполнения.
        self.centers_: np.ndarray | None = None

    def run(self, points, seed, cancel_check=None, initial_centers=None):
        """Выполняет кластеризацию.

        :param initial_centers: начальные центры k × 2; используется только
            в тестах для сверки с эталонной реализацией
        """
        self._initial_centers = None if initial_centers is None else np.asarray(
            initial_centers, dtype=float)
        try:
            return super().run(points, seed, cancel_check)
        finally:
            self._initial_centers = None

    def _fit(self, points: np.ndarray, seed: int, params: dict,
             cancel_check: CancelCheck) -> np.ndarray:
        k, max_iter = params["n_clusters"], params["max_iter"]
        if self._initial_centers is not None:
            centers = self._initial_centers.copy()
        else:
            rng = np.random.default_rng(seed)
            centers = points[rng.choice(len(points), size=k, replace=False)].copy()
        labels = None
        for _ in range(max_iter):
            cancel_check()
            distances = squared_distances(points, centers)
            new_labels = distances.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centers = self._update_centers(points, labels, centers, distances)
        labels = squared_distances(points, centers).argmin(axis=1)
        self.centers_ = centers
        return labels

    @staticmethod
    def _update_centers(points, labels, centers, distances) -> np.ndarray:
        k = len(centers)
        counts = np.bincount(labels, minlength=k)
        new_centers = centers.copy()
        filled = counts > 0
        for axis in range(2):
            sums = np.bincount(labels, weights=points[:, axis], minlength=k)
            new_centers[filled, axis] = sums[filled] / counts[filled]
        empty = np.flatnonzero(~filled)
        if empty.size:
            own = distances[np.arange(len(points)), labels].copy()
            for j in empty:
                far = int(own.argmax())
                new_centers[j] = points[far]
                own[far] = -1.0
        return new_centers
