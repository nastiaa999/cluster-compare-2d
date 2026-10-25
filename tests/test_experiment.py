"""Эксперимент: файл параметров JSON (тесты 6, 49–51, 87–89)."""

import json

import numpy as np
import pytest

from cc2d.core.algorithms import DBSCAN, KMeans, MeanShift
from cc2d.core.errors import ExperimentFileError
from cc2d.core.experiment import Experiment
from cc2d.core.preprocessing import MINMAX, Preprocessor
from cc2d.core.sources import CsvLoader, DataGenerator

from conftest import csv_text


@pytest.fixture
def csv_experiment(write_csv):
    rng = np.random.default_rng(0)
    data = [[f"{a:.3f}".replace(".", ","), f"{b:.3f}", "А" if a > 0 else "Б"]
            for a, b in rng.normal(size=(60, 2))]
    path = write_csv(csv_text(["длина", "ширина", "вид"], data))
    return Experiment(CsvLoader(path), "длина", "ширина", "вид", MINMAX,
                      [KMeans(n_clusters=4, max_iter=100), MeanShift()], seed=12345)


def run_kmeans(experiment):
    dataset = experiment.restore_dataset()
    p = Preprocessor(experiment.preprocessing)
    p.dataset = dataset
    data = p.process(experiment.x_column, experiment.y_column, experiment.label_column)
    return experiment.algorithms[0].run(data.points, experiment.seed).get_labels()


def test_save_load_roundtrip(csv_experiment, tmp_path):
    """Тест 6: параметры восстанавливаются, метки кластеров совпадают."""
    path = tmp_path / "experiment.json"
    csv_experiment.save(path)
    before = run_kmeans(csv_experiment)

    loaded = Experiment()
    loaded.load(path)
    assert (loaded.x_column, loaded.y_column, loaded.label_column) == ("длина", "ширина", "вид")
    assert loaded.preprocessing == MINMAX and loaded.seed == 12345
    assert [a.key for a in loaded.algorithms] == ["kmeans", "meanshift"]
    assert loaded.algorithms[0].params == {"n_clusters": 4, "max_iter": 100}
    assert loaded.algorithms[1].params == {"bandwidth": "авто"}
    assert np.array_equal(run_kmeans(loaded), before)


def test_file_is_utf8_json(csv_experiment, tmp_path):
    path = tmp_path / "e.json"
    csv_experiment.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["source"]["type"] == "csv"
    assert data["columns"] == {"x": "длина", "y": "ширина", "labels": "вид"}
    assert data["seed"] == 12345


def test_generated_source_restored(tmp_path):
    """Тест 87: набор генерируется заново с сохранённым seed и совпадает с исходным."""
    gen = DataGenerator(n_points=300, n_clusters=4, spread=0.5, noise=0.1)
    experiment = Experiment(gen, "x", "y", "label", algorithms=[DBSCAN()], seed=777)
    original = gen.get_data(777).get_all()
    path = tmp_path / "gen.json"
    experiment.save(path)
    loaded = Experiment()
    loaded.load(path)
    assert loaded.restore_dataset().get_all().equals(original)


def test_broken_json_keeps_params(csv_experiment, tmp_path):
    """Тест 49: нарушена структура JSON — параметры не меняются."""
    path = tmp_path / "bad.json"
    path.write_text("{ не json", encoding="utf-8")
    with pytest.raises(ExperimentFileError, match="JSON"):
        csv_experiment.load(path)
    assert csv_experiment.seed == 12345 and csv_experiment.x_column == "длина"


def test_value_out_of_range(csv_experiment, tmp_path):
    """Тест 50: число кластеров K-Means равно 50."""
    path = tmp_path / "e.json"
    csv_experiment.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["algorithms"][0]["params"]["n_clusters"] = 50
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    target = Experiment(seed=1)
    with pytest.raises(ExperimentFileError, match="Недопустимое значение"):
        target.load(path)
    assert target.seed == 1 and target.algorithms == []


def test_missing_csv(csv_experiment, tmp_path):
    """Тест 51: файл параметров указывает на удалённый файл CSV."""
    path = tmp_path / "e.json"
    csv_experiment.save(path)
    csv_path = csv_experiment.source.path
    import os
    os.remove(csv_path)
    with pytest.raises(ExperimentFileError, match="не найден"):
        Experiment().load(path)


def test_not_an_experiment(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ExperimentFileError, match="структура"):
        Experiment().load(path)


def test_copy_is_independent(csv_experiment):
    """Тест 88: копия не меняется при изменении параметров эксперимента."""
    snapshot = csv_experiment.copy()
    csv_experiment.algorithms[0].params["n_clusters"] = 5
    csv_experiment.x_column = "ширина"
    assert snapshot.algorithms[0].params["n_clusters"] == 4
    assert snapshot.x_column == "длина"


def test_restore_checks_columns(csv_experiment, tmp_path):
    csv_experiment.x_column = "нет такого"
    with pytest.raises(ExperimentFileError, match="нет числового столбца"):
        csv_experiment.restore_dataset()
