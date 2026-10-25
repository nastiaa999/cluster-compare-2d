"""Временные характеристики ядра на 5000 точках (тесты 93–95).

Запускаются отдельно: ``pytest -m slow``. Окончательные измерения выполняются
на компьютере минимальной конфигурации (п. 4.4.2 ТЗ).
"""

import time

import pytest

from cc2d.core.algorithms import DBSCAN, Agglomerative, KMeans, MeanShift
from cc2d.core.preprocessing import Preprocessor
from cc2d.core.sources import DataGenerator

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def points5000():
    p = Preprocessor()
    p.dataset = DataGenerator(n_points=5000, n_clusters=10).get_data(1)
    return p.process("x", "y").points


def test_generation_time():
    """Тест 93: генерация 5000 точек не дольше 1 с."""
    start = time.perf_counter()
    DataGenerator(n_points=5000, n_clusters=10).get_data(1)
    assert time.perf_counter() - start <= 1


@pytest.mark.parametrize("cls", [KMeans, Agglomerative, DBSCAN, MeanShift])
def test_algorithm_time(points5000, cls):
    """Тест 94: каждый алгоритм с параметрами по умолчанию не дольше 10 с."""
    assert cls().run(points5000, 1).elapsed <= 10
