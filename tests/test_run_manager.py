"""Менеджер запуска: порядок обработки, изоляция ошибок, отмена (тесты 2, 60, 85, 88)."""

import numpy as np
import pytest

from cc2d.core.algorithms import DBSCAN, Agglomerative, KMeans, MeanShift
from cc2d.core.algorithms.base import ClusteringAlgorithm
from cc2d.core.errors import AllAlgorithmsFailedError, CancelledError, ParameterError
from cc2d.core.experiment import Experiment
from cc2d.core.preprocessing import Preprocessor
from cc2d.core.run_manager import CANCELLED, DONE, RunManager
from cc2d.core.sources import DataGenerator


class Broken(ClusteringAlgorithm):
    """Тестовая заглушка, вызывающая исключение (тест 85)."""

    key = "broken"
    title = "Заглушка"
    PARAMS = {}

    def _fit(self, points, seed, params, cancel_check):
        raise RuntimeError("сбой внутри алгоритма")


@pytest.fixture
def manager():
    gen = DataGenerator(n_points=300, n_clusters=3)
    preprocessor = Preprocessor()
    preprocessor.dataset = gen.get_data(10)
    m = RunManager(preprocessor)
    m.experiment = Experiment(gen, "x", "y", "label",
                              algorithms=[KMeans(), Agglomerative(), DBSCAN(), MeanShift()],
                              seed=10)
    return m


def test_full_run(manager):
    """Тест 2: четыре алгоритма, таблица из четырёх строк с выделенными лучшими значениями."""
    steps = []
    output = manager.run(lambda percent, text: steps.append(percent))
    assert [r.key for r in output.results] == ["kmeans", "agglomerative", "dbscan", "meanshift"]
    assert len(output.table.rows) == 4
    assert output.table.best
    assert all("ari" in r.metrics for r in output.results)
    assert steps[0] == 0 and steps[-1] == 100
    assert manager.status == DONE
    assert output.data.points.shape == (300, 2)


def test_table_keeps_params_of_run(manager):
    """Тест 88: в таблице копия параметров расчёта, а не изменённые позже."""
    output = manager.run()
    manager.experiment.algorithms[0].params["n_clusters"] = 5
    assert output.table.get_params().algorithms[0].params["n_clusters"] == 3
    assert output.table.get_params() is not manager.experiment


def test_failed_algorithm_isolated(manager, caplog):
    """Тест 85: остальные алгоритмы завершаются, исключение записано в журнал."""
    manager.experiment.algorithms[1] = Broken()
    output = manager.run()
    assert [r.key for r in output.results] == ["kmeans", "dbscan", "meanshift"]
    assert output.failures[0][0] == "Заглушка"
    assert "сбой внутри алгоритма" in caplog.text


def test_all_failed(manager):
    manager.experiment.algorithms = [Broken()]
    with pytest.raises(AllAlgorithmsFailedError, match="Заглушка"):
        manager.run()


def test_cancel(manager):
    """Тест 60: отмена во время расчёта останавливает его, результаты не выдаются."""

    def progress(percent, text):
        if percent > 0:
            manager.cancel()

    with pytest.raises(CancelledError):
        manager.run(progress)
    assert manager.status == CANCELLED


def test_run_after_cancel_works(manager):
    manager.cancel()
    assert len(manager.run().results) == 4


def test_cancel_right_after_prepare(manager):
    """«Отменить» сразу после «Запустить», пока поток не начал работу, не теряется."""
    manager.prepare()
    manager.cancel()
    with pytest.raises(CancelledError):
        manager.run()


def test_no_algorithms(manager):
    manager.experiment.algorithms = []
    with pytest.raises(ParameterError):
        manager.run()


def test_rerun_replaces_results(manager):
    """Тест 84: повторный запуск с другой парой признаков даёт новую таблицу."""
    first = manager.run()
    manager.experiment.y_column = "label"
    manager.experiment.label_column = None
    second = manager.run()
    assert second.table is not first.table
    assert not np.array_equal(second.data.points, first.data.points)
