"""Предобработчик: исключение строк и нормализация (тесты 5, 18, 76, 78, 82)."""

import numpy as np
import pytest

from cc2d.core.errors import DataPreparationError
from cc2d.core.preprocessing import MINMAX, STANDARD, Preprocessor

from conftest import make_dataset


def prep(dataset, method=STANDARD):
    p = Preprocessor(method)
    p.dataset = dataset
    return p


def test_excluded_rows_counted_and_removed():
    """Тест 5: в X три пустые ячейки, в Y две ячейки с текстом → исключено 5 строк."""
    x = [str(i) for i in range(100)]
    y = [str(i * 2) for i in range(100)]
    x[3] = x[10] = x[50] = ""
    y[20], y[30] = "abc", "нет"
    p = prep(make_dataset({"x": x, "y": y}))
    assert p.count_excluded("x", "y") == 5
    data = p.process("x", "y")
    assert len(data.points) == 95
    assert data.excluded == 5
    assert data.reference_labels is None


def test_count_does_not_change_dataset():
    dataset = make_dataset({"x": ["1", ""] * 10, "y": ["2", "3"] * 10})
    before = dataset.get_all()
    prep(dataset).count_excluded("x", "y")
    assert dataset.get_all().equals(before)


def test_empty_label_excluded():
    """Тест 82: пустая ячейка в столбце текстовых меток исключает строку."""
    labels = ["a", "b"] * 10
    labels[4] = " "
    p = prep(make_dataset({"x": range(20), "y": range(20, 40), "c": labels}))
    assert p.count_excluded("x", "y", "c") == 1
    assert p.count_excluded("x", "y") == 0
    data = p.process("x", "y", "c")
    assert len(data.reference_labels) == 19


def test_less_than_10_rows_after_exclusion():
    """Тест 78: 12 строк, в трёх пропуски → расчёт запрещён (П1 = 0)."""
    x = [str(i) for i in range(12)]
    x[0] = x[1] = x[2] = ""
    p = prep(make_dataset({"x": x, "y": range(12)}))
    assert "не меньше 10" in p.readiness_problem("x", "y")
    with pytest.raises(DataPreparationError):
        p.process("x", "y")


def test_constant_feature():
    """Тест 18, решение 12.9: постоянный признак делает набор неподготовленным."""
    p = prep(make_dataset({"x": ["5"] * 20, "y": range(20)}))
    assert "постоянен" in p.readiness_problem("x", "y")


def test_standardization():
    p = prep(make_dataset({"x": range(20), "y": [i ** 2 for i in range(20)]}))
    points = p.process("x", "y").points
    assert np.allclose(points.mean(axis=0), 0)
    assert np.allclose(points.std(axis=0), 1)


def test_minmax_and_comma_decimal():
    p = prep(make_dataset({"x": [f"{i},5" for i in range(20)], "y": range(20)}), MINMAX)
    data = p.process("x", "y")
    assert data.points.min() == 0 and data.points.max() == 1
    assert data.raw_points[0, 0] == 0.5


def test_first_and_last_rows_excluded_keep_order():
    """Тест 76: пустые ячейки X в первой и последней строках из 50."""
    x = [str(i) for i in range(50)]
    x[0] = x[-1] = ""
    data = prep(make_dataset({"x": x, "y": range(50)})).process("x", "y")
    assert len(data.points) == 48
    assert data.row_index[0] == 1 and data.row_index[-1] == 48
    assert data.raw_points[0, 0] == 1 and data.raw_points[-1, 0] == 48
