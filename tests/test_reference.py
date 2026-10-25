"""Сверка ручных алгоритмов с эталонными реализациями scikit-learn (тесты 100–103).

scikit-learn используется здесь только как эталон, в программе его алгоритмы
кластеризации не вызываются.
"""

import numpy as np
from sklearn import cluster as reference
from sklearn.metrics import adjusted_rand_score

from cc2d.core.algorithms import DBSCAN, Agglomerative, KMeans, MeanShift


def test_kmeans_matches_sklearn(blobs500):
    """Тест 100: одинаковые начальные центры → одинаковые разбиения и центры."""
    points, _ = blobs500
    init = points[[0, 1, 2]]
    algorithm = KMeans(n_clusters=3, max_iter=500)
    labels = algorithm.run(points, 0, initial_centers=init).get_labels()
    # tol=0: эталон, как и ручная реализация, останавливается, когда разбиение не меняется.
    ref = reference.KMeans(3, init=init, n_init=1, max_iter=500, tol=0,
                           algorithm="lloyd").fit(points)
    assert adjusted_rand_score(labels, ref.labels_) == 1
    assert np.abs(algorithm.centers_ - ref.cluster_centers_).max() < 1e-3


def test_agglomerative_matches_sklearn(blobs500):
    """Тест 101: связь Уорда, три кластера."""
    points, _ = blobs500
    labels = Agglomerative(n_clusters=3).run(points, 0).get_labels()
    ref = reference.AgglomerativeClustering(3, linkage="ward").fit(points)
    assert adjusted_rand_score(labels, ref.labels_) == 1


def test_dbscan_matches_sklearn(blobs500):
    """Тест 102: радиус 0,5, минимальное число точек 5."""
    points, _ = blobs500
    labels = DBSCAN(eps=0.5, min_samples=5).run(points, 0).get_labels()
    ref = reference.DBSCAN(eps=0.5, min_samples=5).fit(points)
    assert np.array_equal(labels == -1, ref.labels_ == -1)
    assert adjusted_rand_score(labels, ref.labels_) >= 0.99


def test_meanshift_matches_sklearn(blobs500):
    """Тест 103: размер окна 1,0."""
    points, _ = blobs500
    result = MeanShift(bandwidth=1.0).run(points, 0)
    ref = reference.MeanShift(bandwidth=1.0).fit(points)
    assert result.n_clusters == len(ref.cluster_centers_)
    assert adjusted_rand_score(result.get_labels(), ref.labels_) == 1


def test_bandwidth_estimate_matches_sklearn(blobs500):
    """Оценка окна «авто» совпадает с estimate_bandwidth при целом 0,3·n."""
    from cc2d.core.algorithms.meanshift import estimate_bandwidth
    points, _ = blobs500
    assert abs(estimate_bandwidth(points) - reference.estimate_bandwidth(points)) < 1e-9
