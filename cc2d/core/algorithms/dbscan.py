"""Алгоритм DBSCAN."""

from __future__ import annotations

import numpy as np

from ..params import FloatParam, IntParam
from ..result import NOISE
from .base import CancelCheck, ClusteringAlgorithm, squared_distances

_BLOCK = 500


class DBSCAN(ClusteringAlgorithm):
    """DBSCAN: кластеры как области высокой плотности точек.

    Окрестность точки — точки на расстоянии не больше ε, включая саму точку.
    Точка корневая, если в окрестности не меньше minPts точек. Кластер
    образуют корневые точки, достижимые друг из друга через окрестности, и
    граничные точки рядом с ними; граничная точка, соседняя с несколькими
    кластерами, относится к кластеру с меньшим номером. Кластеры нумеруются
    в порядке первой корневой точки. Остальные точки — шум.
    Соседи ищутся перебором всех пар, O(n²); матрица смежности 5000 × 5000
    занимает 25 МБ.
    """

    key = "dbscan"
    title = "DBSCAN"
    PARAMS = {
        "eps": FloatParam("eps", "Радиус окрестности", 0.01, 100.0, 0.5),
        "min_samples": IntParam("min_samples", "Мин. число точек", 1, 100, 5),
    }

    def _fit(self, points: np.ndarray, seed: int, params: dict,
             cancel_check: CancelCheck) -> np.ndarray:
        n = len(points)
        eps2 = params["eps"] ** 2
        adjacent = np.empty((n, n), dtype=bool)
        for start in range(0, n, _BLOCK):
            block = squared_distances(points[start:start + _BLOCK], points)
            adjacent[start:start + _BLOCK] = block <= eps2
        core = adjacent.sum(axis=1) >= params["min_samples"]

        labels = np.full(n, NOISE)
        cluster = 0
        for seed_point in np.flatnonzero(core):
            if labels[seed_point] != NOISE:
                continue
            cancel_check()
            labels[seed_point] = cluster
            frontier = np.array([seed_point])
            while frontier.size:
                reached = adjacent[frontier].any(axis=0) & core & (labels == NOISE)
                frontier = np.flatnonzero(reached)
                labels[frontier] = cluster
            cluster += 1

        border = np.flatnonzero(~core)
        if border.size and cluster:
            core_labels = np.where(core, labels, cluster)
            for start in range(0, border.size, _BLOCK):
                idx = border[start:start + _BLOCK]
                nearest = np.where(adjacent[idx], core_labels[None, :], cluster).min(axis=1)
                labels[idx] = np.where(nearest < cluster, nearest, NOISE)
        return labels
