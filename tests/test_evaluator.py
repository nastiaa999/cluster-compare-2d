"""Оценщик качества и замечания (тесты 66–69, 71, 73, 74)."""

import numpy as np
import pytest
from sklearn import metrics as skm

from cc2d.core.algorithms import DBSCAN, Agglomerative, KMeans
from cc2d.core.evaluator import (ARI, MSG_ALL_NOISE, MSG_EACH_POINT, MSG_NOISE,
                                 MSG_ONE_CLUSTER, MSG_WEAK, NMI, SILHOUETTE,
                                 QualityEvaluator)
from cc2d.core.result import ClusteringResult

from conftest import three_groups


def evaluate(result, points, labels=None):
    QualityEvaluator().evaluate(result, points, labels)
    return result


def test_all_metrics_match_sklearn(blobs500):
    """Тест 66: П10 = 1, П11 = 1 → пять метрик, совпадают с scikit-learn."""
    points, ref = blobs500
    result = evaluate(KMeans(n_clusters=3).run(points, 1), points, ref)
    labels = result.get_labels()
    expected = {
        "silhouette": skm.silhouette_score(points, labels),
        "davies_bouldin": skm.davies_bouldin_score(points, labels),
        "calinski_harabasz": skm.calinski_harabasz_score(points, labels),
        "ari": skm.adjusted_rand_score(ref, labels),
        "nmi": skm.normalized_mutual_info_score(ref, labels),
    }
    for name, value in expected.items():
        assert result.metrics[name] == pytest.approx(value, abs=1e-3)
    assert result.message == ""


def test_no_labels_no_external(blobs500):
    """Тест 67: без столбца меток ARI и NMI не вычисляются."""
    points, _ = blobs500
    result = evaluate(KMeans(n_clusters=3).run(points, 1), points)
    assert ARI not in result.metrics and NMI not in result.metrics
    assert result.metrics[SILHOUETTE] is not None


def test_one_cluster_with_labels(blobs500):
    """Тест 68: один кластер — внутренних метрик нет, ARI = NMI = 0, замечание."""
    points, ref = blobs500
    result = evaluate(DBSCAN(eps=100).run(points, 0), points, ref)
    assert result.metrics[SILHOUETTE] is None
    assert result.metrics[ARI] == pytest.approx(0)
    assert result.metrics[NMI] == pytest.approx(0)
    assert result.message == MSG_ONE_CLUSTER


def test_one_cluster_without_labels(blobs500):
    """Тест 69: П10 = 0, П11 = 0."""
    points, _ = blobs500
    result = evaluate(DBSCAN(eps=100).run(points, 0), points)
    assert result.metrics[SILHOUETTE] is None and ARI not in result.metrics


def test_each_point_own_cluster():
    """Тест 71: число кластеров n — прочерк и замечание."""
    points = np.random.default_rng(0).normal(size=(10, 2))
    result = evaluate(Agglomerative(n_clusters=10).run(points, 0), points)
    assert result.metrics[SILHOUETTE] is None
    assert result.message == MSG_EACH_POINT


def test_all_noise(blobs500):
    """Тест 73: весь набор шум."""
    points, _ = blobs500
    result = evaluate(DBSCAN(eps=0.01, min_samples=100).run(points, 0), points)
    assert result.metrics[SILHOUETTE] is None
    assert result.message == MSG_ALL_NOISE


def test_most_points_noise(blobs500):
    """Решение 12.2: шум больше половины точек, но не все."""
    points, _ = blobs500
    result = evaluate(DBSCAN(eps=0.05, min_samples=8).run(points, 0), points)
    assert 250 < result.n_noise < 500
    assert MSG_NOISE in result.message


def test_perfect_partition():
    """Тест 74: ARI = 1 и NMI = 1."""
    points, ref = three_groups()
    result = evaluate(KMeans(n_clusters=3).run(points, 0), points, ref)
    assert result.metrics[ARI] == pytest.approx(1)
    assert result.metrics[NMI] == pytest.approx(1)


def test_weak_silhouette_message():
    """Решение 12.2: силуэт меньше 0,25."""
    points = np.random.default_rng(0).uniform(size=(200, 2))
    labels = np.random.default_rng(1).integers(0, 3, size=200)
    result = evaluate(ClusteringResult("Тест", "test", labels, 0.0), points)
    assert result.metrics[SILHOUETTE] < 0.25
    assert result.message == MSG_WEAK


def test_noise_excluded_from_internal_metrics(blobs500):
    points, ref = blobs500
    result = evaluate(DBSCAN(eps=0.2, min_samples=10).run(points, 0), points, ref)
    labels = result.get_labels()
    mask = labels != -1
    assert result.metrics[SILHOUETTE] == pytest.approx(
        skm.silhouette_score(points[mask], labels[mask]))
    assert result.metrics[ARI] == pytest.approx(skm.adjusted_rand_score(ref, labels))
