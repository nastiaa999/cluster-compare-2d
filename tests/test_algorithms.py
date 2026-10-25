"""Параметры и поведение алгоритмов (тесты 10, 31–48, 70–73, 79)."""

import numpy as np
import pytest

from cc2d.core.algorithms import DBSCAN, Agglomerative, KMeans, MeanShift
from cc2d.core.errors import ParameterError
from cc2d.core.params import AUTO


@pytest.mark.parametrize("cls, name, value", [
    (KMeans, "n_clusters", 1), (KMeans, "n_clusters", 21), (KMeans, "n_clusters", "2,5"),
    (Agglomerative, "n_clusters", 1), (Agglomerative, "n_clusters", 21),
    (Agglomerative, "n_clusters", "2,5"),
    (KMeans, "max_iter", 0), (KMeans, "max_iter", 1001), (KMeans, "max_iter", "10,5"),
    (DBSCAN, "eps", "0,009"), (DBSCAN, "eps", "100,1"), (DBSCAN, "eps", "abc"),
    (DBSCAN, "min_samples", 0), (DBSCAN, "min_samples", 101), (DBSCAN, "min_samples", "2,5"),
    (MeanShift, "bandwidth", "0,009"), (MeanShift, "bandwidth", "100,1"),
    (MeanShift, "bandwidth", "abc"),
])
def test_invalid_params(cls, name, value):
    """Тесты 31–48: значение отклоняется, причина относится к полю."""
    algorithm = cls(**{name: value})
    assert set(algorithm.errors()) == {name}
    with pytest.raises(ParameterError):
        algorithm.validate_params()


@pytest.mark.parametrize("cls, params", [
    (KMeans, {"n_clusters": 2, "max_iter": 1}),
    (KMeans, {"n_clusters": 20, "max_iter": 1000}),
    (Agglomerative, {"n_clusters": 20}),
    (DBSCAN, {"eps": "0,01", "min_samples": 1}),
    (DBSCAN, {"eps": 100, "min_samples": 100}),
    (MeanShift, {"bandwidth": "0,01"}),
    (MeanShift, {"bandwidth": 100}),
    (MeanShift, {"bandwidth": "авто"}),
])
def test_boundary_params_valid(cls, params):
    """Тесты 1–4: значения на границах диапазонов допустимы."""
    assert cls(**params).errors() == {}


def test_reset_to_defaults():
    """Тест 10: «По умолчанию» восстанавливает 3 кластера и 500 итераций."""
    algorithm = KMeans(n_clusters=7, max_iter=10)
    algorithm.reset_params()
    assert algorithm.params == {"n_clusters": 3, "max_iter": 500}
    assert MeanShift().params == {"bandwidth": AUTO}
    assert DBSCAN().params == {"eps": 0.5, "min_samples": 5}
    assert Agglomerative().params == {"n_clusters": 3, "linkage": "ward"}


def test_clusters_more_than_points():
    """Тест 79: 10 точек и 11 кластеров — параметр некорректен."""
    algorithm = KMeans(n_clusters=11)
    assert algorithm.errors() == {}
    assert "больше числа точек" in algorithm.errors(n_points=10)["n_clusters"]


def ten_points():
    return np.random.default_rng(0).normal(size=(10, 2))


def test_agglomerative_n_minus_one():
    """Тест 70: 10 точек, 9 кластеров."""
    assert Agglomerative(n_clusters=9).run(ten_points(), 0).n_clusters == 9


def test_agglomerative_each_point():
    """Тест 71: 10 точек, 10 кластеров — каждая точка свой кластер."""
    labels = Agglomerative(n_clusters=10).run(ten_points(), 0).get_labels()
    assert sorted(labels) == list(range(10))


def test_dbscan_no_noise(blobs500):
    """Тест 72: минимальное число точек 1 — все точки корневые."""
    assert DBSCAN(min_samples=1).run(blobs500[0], 0).n_noise == 0


def test_dbscan_all_noise(blobs500):
    """Тест 73: радиус 0,01 и 100 точек — весь набор шум."""
    result = DBSCAN(eps=0.01, min_samples=100).run(blobs500[0], 0)
    assert result.n_noise == 500 and result.n_clusters == 0


def test_dbscan_one_cluster(blobs500):
    """Тест 68: радиус 100 — один кластер."""
    assert DBSCAN(eps=100).run(blobs500[0], 0).n_clusters == 1


def test_kmeans_reproducible_with_seed(blobs500):
    a = KMeans(n_clusters=3).run(blobs500[0], 11).get_labels()
    b = KMeans(n_clusters=3).run(blobs500[0], 11).get_labels()
    assert np.array_equal(a, b)


def test_kmeans_always_k_clusters():
    """Решение 12.4: опустевший кластер получает новый центр, кластеров ровно k."""
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(10, 1, (20, 2))])
    init = [[0, 0], [10, 10], [50, 50], [60, 60]]
    result = KMeans(n_clusters=4).run(points, 0, initial_centers=init)
    assert result.n_clusters == 4


def test_meanshift_auto_bandwidth(blobs500):
    algorithm = MeanShift()
    result = algorithm.run(blobs500[0], 0)
    assert algorithm.bandwidth_ > 0
    assert result.n_noise == 0 and result.n_clusters >= 1


def test_result_fields(blobs500):
    result = DBSCAN().run(blobs500[0], 0)
    assert result.algorithm == "DBSCAN" and result.key == "dbscan"
    assert result.elapsed >= 0
    assert result.n_points == 500
