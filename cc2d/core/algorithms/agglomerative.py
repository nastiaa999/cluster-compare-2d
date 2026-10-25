"""Агломеративная иерархическая кластеризация со связью Уорда."""

from __future__ import annotations

import numpy as np

from ..params import ChoiceParam, IntParam
from .base import CancelCheck, ClusteringAlgorithm, squared_distances

#: Число строк матрицы расстояний, вычисляемых за один раз.
_BLOCK = 500
#: Как часто (в объединениях) проверять отмену расчёта.
_CANCEL_EVERY = 100


class Agglomerative(ClusteringAlgorithm):
    """Агломеративная кластеризация: последовательное объединение ближайших кластеров.

    Расстояние между кластерами по Уорду пересчитывается после каждого
    объединения формулой Ланса–Уильямса для квадратов расстояний

    d(k, i∪j) = ((nᵢ + nₖ)·d(k, i) + (nⱼ + nₖ)·d(k, j) − nₖ·d(i, j)) / (nᵢ + nⱼ + nₖ),

    а пары для объединения ищутся цепочкой ближайших соседей. Связь Уорда
    удовлетворяет условию сводимости, поэтому цепочка находит те же
    объединения, что и полный перебор, за O(n²) операций вместо O(n³).
    Матрица квадратов расстояний n × n занимает около 200 МБ при n = 5000.
    """

    key = "agglomerative"
    title = "Агломеративная"
    PARAMS = {
        "n_clusters": IntParam("n_clusters", "Число кластеров", 2, 20, 3),
        "linkage": ChoiceParam("linkage", "Тип связи", ("ward",), "ward"),
    }

    def _fit(self, points: np.ndarray, seed: int, params: dict,
             cancel_check: CancelCheck) -> np.ndarray:
        n, k = len(points), params["n_clusters"]
        merges = self._nn_chain(points, cancel_check)
        merges.sort(key=lambda m: m[0])
        parent = np.arange(n)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for _, i, j in merges[:n - k]:
            parent[find(j)] = find(i)
        roots = np.array([find(i) for i in range(n)])
        # Кластеры нумеруются в порядке первой точки каждого кластера.
        _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
        rank = np.empty(first.size, dtype=int)
        rank[np.argsort(first)] = np.arange(first.size)
        return rank[inverse]

    @staticmethod
    def _nn_chain(points: np.ndarray, cancel_check: CancelCheck) -> list:
        """Возвращает список объединений (высота, i, j); кластер i∪j хранится под номером i."""
        n = len(points)
        dist = np.empty((n, n))
        for start in range(0, n, _BLOCK):
            dist[start:start + _BLOCK] = squared_distances(points[start:start + _BLOCK], points)
        np.fill_diagonal(dist, np.inf)
        size = np.ones(n)
        active = np.ones(n, dtype=bool)
        merges = []
        chain: list[int] = []
        while len(merges) < n - 1:
            if len(merges) % _CANCEL_EVERY == 0:
                cancel_check()
            if not chain:
                chain.append(int(np.flatnonzero(active)[0]))
            a = chain[-1]
            row = dist[a]
            b = int(row.argmin())
            if len(chain) > 1 and row[chain[-2]] <= row[b]:
                b = chain[-2]
            if len(chain) > 1 and b == chain[-2]:
                chain.pop()
                chain.pop()
                i, j = min(a, b), max(a, b)
                height = dist[i, j]
                ni, nj = size[i], size[j]
                new_row = ((ni + size) * dist[i] + (nj + size) * dist[j]
                           - size * height) / (ni + nj + size)
                dist[i] = new_row
                dist[:, i] = new_row
                dist[i, i] = np.inf
                dist[j] = np.inf
                dist[:, j] = np.inf
                size[i] = ni + nj
                active[j] = False
                merges.append((height, i, j))
            else:
                chain.append(b)
        return merges
