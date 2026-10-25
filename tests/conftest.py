"""Общие данные для тестов.

Номера тестов в docstring соответствуют таблицам приложения Г РПЗ.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from cc2d.core.dataset import SOURCE_CSV, Dataset

import pandas as pd


@pytest.fixture
def write_csv(tmp_path):
    """Возвращает функцию, записывающую файл CSV и возвращающую путь к нему."""

    def write(text: str, name: str = "data.csv", encoding: str = "utf-8") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode(encoding))
        return path

    return write


def csv_text(header, rows) -> str:
    """Собирает текст CSV с разделителем «;»."""
    lines = [";".join(header)] + [";".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


@pytest.fixture
def blobs500():
    """Набор из теста 66: 500 точек трёх кластеров (стандартизованных) и эталонные метки."""
    points, labels = make_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=7)
    return StandardScaler().fit_transform(points), labels


def make_dataset(columns: dict) -> Dataset:
    """Набор данных из словаря столбцов (значения хранятся строками, как из CSV)."""
    frame = pd.DataFrame({k: [str(v) for v in vals] for k, vals in columns.items()},
                         dtype=object)
    return Dataset(frame, SOURCE_CSV)


def three_groups():
    """Набор из теста 74: три группы по 10 точек вокруг (0; 0), (10; 0) и (0; 10)."""
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    points = np.vstack([c + rng.normal(0, 0.1, size=(10, 2)) for c in centers])
    labels = np.repeat([1, 2, 3], 10)
    return points, labels
