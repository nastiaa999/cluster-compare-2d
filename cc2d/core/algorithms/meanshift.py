"""Алгоритм Mean Shift с плоским ядром."""

from __future__ import annotations

import math

import numpy as np

from ..errors import ParameterError
from ..params import AUTO, FloatOrAutoParam
from .base import CancelCheck, ClusteringAlgorithm, squared_distances

_BLOCK = 500
#: Доля точек, по которой оценивается размер окна в режиме «авто».
QUANTILE = 0.3
#: Наибольшее число итераций сдвига одной точки.
MAX_ITER = 300
#: Порог остановки: сдвиг меньше этой доли размера окна.
STOP_FACTOR = 1e-3


def estimate_bandwidth(points: np.ndarray, quantile: float = QUANTILE) -> float:
    """Оценивает размер окна по данным.

    Для каждой точки берётся расстояние до её ⌈quantile·n⌉-го ближайшего
    соседа (сама точка считается первым соседом), размер окна равен среднему
    этих расстояний.
    """
    n = len(points)
    k = max(1, math.ceil(quantile * n))
    total = 0.0
    for start in range(0, n, _BLOCK):
        d = squared_distances(points[start:start + _BLOCK], points)
        total += np.sqrt(np.partition(d, k - 1, axis=1)[:, k - 1]).sum()
    return total / n


class MeanShift(ClusteringAlgorithm):
    """Mean Shift: сдвиг точек к локальным максимумам плотности.

    Каждая точка набора служит начальным положением. На каждой итерации
    положение заменяется средним точек, попавших в окно радиуса h (плоское
    ядро). Сдвиг заканчивается, когда он меньше 10⁻³·h или выполнено 300
    итераций. Найденные центры упорядочиваются по числу точек в окне, и
    центр, лежащий ближе h к уже принятому, отбрасывается. Каждая точка
    приписывается ближайшему центру, шума нет.
    """

    key = "meanshift"
    title = "Mean Shift"
    PARAMS = {
        "bandwidth": FloatOrAutoParam("bandwidth", "Размер окна", 0.01, 100.0, AUTO),
    }

    def __init__(self, **params):
        super().__init__(**params)
        #: Размер окна, использованный при последнем выполнении.
        self.bandwidth_: float | None = None
        #: Центры кластеров после последнего выполнения.
        self.centers_: np.ndarray | None = None

    def _fit(self, points: np.ndarray, seed: int, params: dict,
             cancel_check: CancelCheck) -> np.ndarray:
        h = params["bandwidth"]
        if h == AUTO:
            h = estimate_bandwidth(points)
            if h <= 0:
                raise ParameterError(
                    "Размер окна не удалось оценить: слишком много совпадающих точек")
        self.bandwidth_ = h
        means, counts = self._shift_all(points, h, cancel_check)
        centers = self._merge_centers(means, counts, h)
        self.centers_ = centers
        return np.concatenate([squared_distances(points[s:s + _BLOCK], centers).argmin(axis=1)
                               for s in range(0, len(points), _BLOCK)])

    @staticmethod
    def _shift_all(points: np.ndarray, h: float, cancel_check: CancelCheck):
        """Сдвигает все начальные положения до сходимости.

        :return: конечные положения и число точек в окне у каждого (0 — положение отброшено)
        """
        n = len(points)
        h2, stop = h * h, STOP_FACTOR * h
        means = points.copy()
        final = np.zeros_like(points)
        counts = np.zeros(n, dtype=int)
        iterations = np.zeros(n, dtype=int)
        active = np.arange(n)
        while active.size:
            cancel_check()
            still_active = []
            for start in range(0, active.size, _BLOCK):
                idx = active[start:start + _BLOCK]
                inside = squared_distances(means[idx], points) <= h2
                count = inside.sum(axis=1)
                new = (inside @ points) / np.maximum(count, 1)[:, None]
                shift = np.sqrt(((new - means[idx]) ** 2).sum(axis=1))
                done = (shift <= stop) | (iterations[idx] == MAX_ITER) | (count == 0)
                final[idx[done]] = new[done]
                counts[idx[done]] = count[done]
                means[idx] = new
                iterations[idx] += 1
                still_active.append(idx[~done])
            active = np.concatenate(still_active)
        return final, counts

    @staticmethod
    def _merge_centers(means: np.ndarray, counts: np.ndarray, h: float) -> np.ndarray:
        """Оставляет центры, удалённые друг от друга больше чем на h."""
        intensity = {}
        for mean, count in zip(map(tuple, means), counts):
            if count:
                intensity[mean] = int(count)
        ordered = sorted(intensity.items(), key=lambda item: (item[1], item[0]), reverse=True)
        centers = np.array([c for c, _ in ordered])
        unique = np.ones(len(centers), dtype=bool)
        h2 = h * h
        for i in range(len(centers)):
            if unique[i]:
                near = squared_distances(centers[i:i + 1], centers)[0] <= h2
                unique[near] = False
                unique[i] = True
        return centers[unique]
