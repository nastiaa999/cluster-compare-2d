"""Сводная таблица: выделение лучших значений и вид ячеек."""

import numpy as np

from cc2d.core.result import ClusteringResult
from cc2d.core.summary import TIME, UNDEFINED, SummaryTable


def result(name, metrics, elapsed=0.1):
    r = ClusteringResult(name, name.lower(), np.array([0, 1, 1]), elapsed)
    r.set_metrics(metrics)
    return r


def table_of(*results):
    table = SummaryTable()
    for r in results:
        table.add(r)
    table.highlight_best()
    return table


def test_best_by_direction():
    table = table_of(
        result("A", {"silhouette": 0.6, "davies_bouldin": 0.5, "calinski_harabasz": 100.0}, 0.3),
        result("B", {"silhouette": 0.7, "davies_bouldin": 0.4, "calinski_harabasz": 50.0}, 0.1),
    )
    assert table.is_best(1, "silhouette")
    assert table.is_best(1, "davies_bouldin")
    assert table.is_best(0, "calinski_harabasz")
    assert table.is_best(1, TIME) and not table.is_best(0, TIME)


def test_ties_after_rounding_all_highlighted():
    """Решение 12.7: равные после округления значения выделяются все."""
    table = table_of(result("A", {"silhouette": 0.701}), result("B", {"silhouette": 0.699}))
    assert table.best["silhouette"] == {0, 1}


def test_undefined_and_not_computed_cells():
    table = table_of(result("A", {"silhouette": None, "davies_bouldin": None,
                                  "calinski_harabasz": None}),
                     result("B", {"silhouette": 0.5, "davies_bouldin": 1.0,
                                  "calinski_harabasz": 3.0, "ari": 0.9, "nmi": 0.8}))
    assert table.display(0, "silhouette") == UNDEFINED
    assert table.display(0, "ari") == ""
    assert table.display(1, "ari") == "0,90"
    assert table.display(1, "calinski_harabasz") == "3,0"
    assert table.best["silhouette"] == {1}
    assert table.best["ari"] == {1}


def test_params_copy_stored():
    table = SummaryTable()
    marker = object()
    table.set_params(marker)
    assert table.get_params() is marker
