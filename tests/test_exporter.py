"""Экспортёр и визуализатор (тесты 7, 8, 53, 76, 77)."""

import csv
import os

import numpy as np
import pytest
from matplotlib.image import imread

from cc2d.core.algorithms import DBSCAN, KMeans
from cc2d.core.errors import ExportError
from cc2d.core.evaluator import QualityEvaluator
from cc2d.core.exporter import CSV_ENCODING, Exporter
from cc2d.core.preprocessing import Preprocessor
from cc2d.core.sources import DataGenerator
from cc2d.core.summary import SummaryTable
from cc2d.visualization import Visualizer, grid_shape

from conftest import make_dataset


def read(path):
    return path.read_text(encoding=CSV_ENCODING).splitlines()


def read_rows(path):
    with open(path, encoding=CSV_ENCODING, newline="") as f:
        return list(csv.reader(f, delimiter=";"))


@pytest.fixture
def run_data():
    """Результаты K-Means и DBSCAN на сгенерированном наборе без меток."""
    dataset = DataGenerator(n_points=200).get_data(5)
    p = Preprocessor()
    p.dataset = dataset
    data = p.process("x", "y")
    results = [KMeans().run(data.points, 1), DBSCAN(eps=0.01, min_samples=100).run(data.points, 1)]
    table = SummaryTable()
    for r in results:
        QualityEvaluator().evaluate(r, data.points)
        table.add(r)
    table.highlight_best()
    return dataset, data, results, table


def test_table_csv(run_data, tmp_path):
    """Тест 7 и решение 12.5: «;», запятая, «–» для неопределённой метрики, пусто без меток."""
    _, _, _, table = run_data
    path = tmp_path / "summary.csv"
    Exporter().export_table(table, path)
    assert read(path)[0].startswith("Алгоритм;Кластеров;Шум;Силуэт")
    header, kmeans, dbscan = read_rows(path)
    assert kmeans[0] == "K-Means" and kmeans[1] == "3"
    assert "," in kmeans[3] and len(kmeans[3].split(",")[1]) == 4
    assert kmeans[6] == "" and kmeans[7] == ""
    assert len(kmeans[8].split(",")[1]) == 3
    assert dbscan[3] == "–" and dbscan[9].startswith("Все точки")


def test_labels_csv_first_and_last_rows(tmp_path):
    """Тест 76: 50 строк, первая и последняя исключены → 48 строк в исходном порядке."""
    x = [str(i + 0.25) for i in range(50)]
    x[0] = x[-1] = ""
    p = Preprocessor()
    p.dataset = make_dataset({"длина": x, "ширина": [str(i * 2) for i in range(50)]})
    data = p.process("длина", "ширина")
    result = KMeans(n_clusters=2).run(data.points, 0)
    path = tmp_path / "labels_kmeans.csv"
    Exporter().export_result(result, data, path)
    lines = read(path)
    assert lines[0] == "длина;ширина;cluster"
    assert len(lines) == 49
    assert lines[1].startswith("1,25;2;") and lines[-1].startswith("48,25;96;")
    assert {line.split(";")[2] for line in lines[1:]} == {"1", "2"}


def test_noise_label_minus_one(run_data, tmp_path):
    _, data, results, _ = run_data
    path = tmp_path / "labels_dbscan.csv"
    Exporter().export_result(results[1], data, path)
    assert {line.split(";")[2] for line in read(path)[1:]} == {"-1"}


def test_dataset_csv(run_data, tmp_path):
    """Тест 8: экспорт сгенерированного набора с теми же значениями."""
    dataset, *_ = run_data
    path = tmp_path / "dataset.csv"
    Exporter().export_dataset(dataset.get_all(), path)
    lines = read(path)
    assert lines[0] == "x;y;label" and len(lines) == 201
    first = dataset.get_rows(1).iloc[0]
    x, y, label = lines[1].split(";")
    assert float(x.replace(",", ".")) == first["x"] and int(label) == first["label"]


def test_png_size(run_data, tmp_path):
    """Тест 77: диаграмма 1200 × 800 пикселей."""
    _, data, results, _ = run_data
    figure = Visualizer().build_single(results[0], data.points)
    path = tmp_path / "plot_kmeans.png"
    Exporter().save_figure(figure, path, (1200, 800))
    assert imread(path).shape[:2] == (800, 1200)


def test_write_error_keeps_old_file(run_data, tmp_path, monkeypatch):
    """Тест 53: ошибка записи — прежний файл цел, временные файлы удалены."""
    _, _, _, table = run_data
    path = tmp_path / "summary.csv"
    path.write_text("старое содержимое", encoding="utf-8")

    def deny(src, dst):
        raise PermissionError(13, "Отказано в доступе")

    monkeypatch.setattr(os, "replace", deny)
    with pytest.raises(ExportError, match="summary.csv"):
        Exporter().export_table(table, path)
    assert path.read_text(encoding="utf-8") == "старое содержимое"
    assert os.listdir(tmp_path) == ["summary.csv"]


def test_missing_folder(run_data, tmp_path):
    _, _, _, table = run_data
    with pytest.raises(ExportError):
        Exporter().export_table(table, tmp_path / "нет" / "summary.csv")


@pytest.mark.parametrize("count, shape", [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2))])
def test_grid(count, shape):
    """Решение 12.14: сетка диаграмм."""
    assert grid_shape(count) == shape


def test_build_three_hides_fourth(run_data):
    _, data, results, _ = run_data
    figure = Visualizer().build(results + [results[0]], data.points)
    axes = figure.axes
    assert sum(ax.get_visible() for ax in axes) == 3
    assert axes[1].get_title() == "DBSCAN (кластеров: 0, шум: 200)"
