"""Загрузчик CSV и генератор данных (тесты 1–4, 12–17, 19–30, 80, 81, 83, 87)."""

import numpy as np
import pytest

from cc2d.core.errors import DataFormatError, ParameterError
from cc2d.core.sources import GENERATOR_PARAMS, CsvLoader, DataGenerator

from conftest import csv_text


def rows(n, x="1.5", y="2.5", label="1"):
    return [[x, y, label] for _ in range(n)]


def test_utf8_point_10_rows(write_csv):
    """Тест 1: UTF-8, десятичная точка, 10 строк, два числовых столбца и целые метки."""
    data = [[f"{i}.5", f"{i * 2}.25", i % 2] for i in range(10)]
    path = write_csv(csv_text(["a", "b", "class"], data))
    dataset = CsvLoader(path).get_data()
    assert dataset.row_count == 10
    assert dataset.numeric_columns[:2] == ["a", "b"]
    assert dataset.default_columns() == ("a", "b", None)
    assert len(dataset.get_rows(100)) == 10
    assert not dataset.is_generated


def test_cp1251_comma_5000_rows(write_csv):
    """Тест 2: Windows-1251, десятичная запятая, 5000 строк, текстовые метки."""
    data = [[f"{i},5", f"{i % 7},25", "вид А" if i % 2 else "вид Б"] for i in range(5000)]
    path = write_csv(csv_text(["длина", "ширина", "вид"], data), encoding="cp1251")
    loader = CsvLoader(path)
    dataset = loader.get_data()
    assert loader.encoding == "Windows-1251"
    assert dataset.row_count == 5000
    assert dataset.numeric_columns == ["длина", "ширина"]
    assert len(dataset.get_rows(100)) == 100


def test_comma_separated_rejected(write_csv):
    """Тест 12: поля разделены запятой."""
    path = write_csv("a,b,c\n" + "1,2,3\n" * 10)
    with pytest.raises(DataFormatError, match="структура"):
        CsvLoader(path).get_data()


def test_binary_rejected(tmp_path):
    """Тест 13: двоичный файл с расширением .csv."""
    path = tmp_path / "image.csv"
    path.write_bytes(bytes(range(256)) * 20)
    with pytest.raises(DataFormatError, match="текстовым"):
        CsvLoader(path).get_data()


def test_utf16_rejected(write_csv):
    """Тест 14: кодировка UTF-16."""
    path = write_csv(csv_text(["a", "b"], rows(10)), encoding="utf-16")
    with pytest.raises(DataFormatError, match="Кодировка"):
        CsvLoader(path).get_data()


@pytest.mark.parametrize("n", [9, 5001])
def test_row_count_limits(write_csv, n):
    """Тесты 15, 16: 9 и 5001 строка данных."""
    path = write_csv(csv_text(["a", "b", "c"], rows(n)))
    with pytest.raises(DataFormatError, match="строк данных"):
        CsvLoader(path).get_data()


def test_less_than_two_numeric_columns(write_csv):
    """Тест 17: один числовой и один текстовый столбец."""
    path = write_csv(csv_text(["a", "name"], [["1", "текст"]] * 10))
    with pytest.raises(DataFormatError, match="меньше двух числовых"):
        CsvLoader(path).get_data()


def test_bom_removed_from_first_column(write_csv):
    """Тест 80: UTF-8 с меткой порядка байтов."""
    path = write_csv(csv_text(["x", "y", "c"], rows(10)), encoding="utf-8-sig")
    dataset = CsvLoader(path).get_data()
    assert dataset.columns[0] == "x"
    assert "x" in dataset.numeric_columns


def test_extra_field_reports_line(write_csv):
    """Тест 81: в строке на одно поле больше, чем в заголовке."""
    data = rows(10)
    data[2] = data[2] + ["лишнее"]
    path = write_csv(csv_text(["a", "b", "c"], data))
    with pytest.raises(DataFormatError, match="в строке 4"):
        CsvLoader(path).get_data()


def test_cyrillic_path(write_csv):
    """Тест 83: кириллица и пробелы в пути к файлу."""
    path = write_csv(csv_text(["a", "b", "c"], rows(10)), name="Мои данные/набор точек.csv")
    assert CsvLoader(path).get_data().row_count == 10


def test_missing_file(tmp_path):
    with pytest.raises(DataFormatError, match="не найден"):
        CsvLoader(tmp_path / "нет.csv").get_data()


def test_column_with_few_bad_values_stays_numeric(write_csv):
    """Решение 12.8: столбец с отдельными пропусками и текстом остаётся числовым (тест 5)."""
    data = [[str(i), str(i), "a"] for i in range(100)]
    data[0][0] = ""
    data[1][1] = "ошибка"
    path = write_csv(csv_text(["x", "y", "c"], data))
    assert CsvLoader(path).get_data().numeric_columns == ["x", "y"]


# --- генератор ---------------------------------------------------------------

def test_generator_lower_bounds():
    """Тест 3: 50 точек, 2 кластера, разброс 0,1, шум 0,01."""
    dataset = DataGenerator(n_points=50, n_clusters=2, spread="0,1", noise="0,01").get_data(1)
    assert dataset.row_count == 50
    assert dataset.is_generated


def test_generator_upper_bounds_and_auto_columns():
    """Тест 4: 5000 точек, 10 кластеров, разброс 5,0, шум 0,5; столбцы выбраны сразу."""
    dataset = DataGenerator(n_points=5000, n_clusters=10, spread=5.0, noise=0.5).get_data(2)
    assert dataset.row_count == 5000
    assert dataset.default_columns() == ("x", "y", "label")
    labels = dataset.get_columns(["label"])["label"].to_numpy()
    assert (labels == -1).sum() == 2500
    assert set(labels[labels != -1]) == set(range(1, 11))


def test_generator_noise_share():
    """Решение 12.1: число шумовых точек round(уровень × число точек)."""
    dataset = DataGenerator(n_points=500, noise=0.05).get_data(3)
    labels = dataset.get_columns(["label"])["label"].to_numpy()
    assert (labels == -1).sum() == 25


def test_generator_same_seed_same_data():
    """Тест 87: повторная генерация с тем же seed даёт тот же набор."""
    a = DataGenerator().get_data(42).get_all()
    b = DataGenerator().get_data(42).get_all()
    c = DataGenerator().get_data(43).get_all()
    assert a.equals(b)
    assert not a.equals(c)


@pytest.mark.parametrize("name, value", [
    ("n_points", 49), ("n_points", 5001), ("n_points", "100,5"),
    ("n_clusters", 1), ("n_clusters", 11), ("n_clusters", "2,5"),
    ("spread", "0,09"), ("spread", "5,1"), ("spread", "abc"),
    ("noise", "0,009"), ("noise", "0,51"), ("noise", "abc"),
])
def test_generator_invalid_params(name, value):
    """Тесты 19–30: неверные параметры генерации отклоняются с причиной."""
    gen = DataGenerator(**{name: value})
    errors = gen.errors()
    assert set(errors) == {name}
    assert GENERATOR_PARAMS[name].label in errors[name]
    with pytest.raises(ParameterError):
        gen.get_data(1)


def test_generator_points_inside_noise_box():
    dataset = DataGenerator(n_points=1000, noise=0.2).get_data(5)
    frame = dataset.get_all()
    assert np.isfinite(frame[["x", "y"]].to_numpy()).all()
