"""Класс «Эксперимент»: параметры запуска и файл параметров JSON."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from .algorithms import BY_KEY, ClusteringAlgorithm
from .dataset import Dataset
from .errors import CC2DError, ExperimentFileError, ParameterError
from .fileio import atomic_write
from .preprocessing import METHODS, STANDARD
from .sources import DISTRIBUTIONS, GENERATOR_PARAMS, CsvLoader, DataGenerator, DataSource

#: Отметка формата в файле параметров.
FORMAT = "CC2D experiment"
FORMAT_VERSION = 1
#: Диапазон значений seed: 0…2³¹ − 1.
SEED_LIMIT = 2 ** 31
#: Наибольшее число алгоритмов в эксперименте (п. 4.1.1.2 ТЗ).
MAX_ALGORITHMS = 4


def new_seed() -> int:
    """Возвращает новое случайное значение seed."""
    return random.randrange(SEED_LIMIT)


class Experiment:
    """Эксперимент (рис. 2.5): источник данных, столбцы, предобработка, алгоритмы и seed.

    Значение seed пользователь не вводит: программа задаёт его при запуске и
    при каждой генерации набора, записывает в файл и берёт из файла при
    загрузке (п. 4.2.6 ТЗ, решение 12.16).

    :param source: источник данных, из которого получен текущий набор
    """

    def __init__(self, source: DataSource | None = None, x_column: str | None = None,
                 y_column: str | None = None, label_column: str | None = None,
                 preprocessing: str = STANDARD,
                 algorithms: list[ClusteringAlgorithm] | None = None,
                 seed: int | None = None):
        self.source = source
        self.x_column = x_column
        self.y_column = y_column
        self.label_column = label_column
        self.preprocessing = preprocessing
        self.algorithms: list[ClusteringAlgorithm] = list(algorithms or [])
        self.seed = new_seed() if seed is None else seed

    def copy(self) -> "Experiment":
        """Создаёт независимую копию («копировать»)."""
        return copy.deepcopy(self)

    # --- запись -----------------------------------------------------------

    def to_dict(self) -> dict:
        """Параметры эксперимента в виде словаря для JSON (рис. 2.8–2.10).

        :raises ParameterError: если параметры алгоритмов некорректны
        """
        if isinstance(self.source, CsvLoader):
            source = {"type": "csv", "path": str(Path(self.source.path).resolve())}
        elif isinstance(self.source, DataGenerator):
            self.source.validate()
            source = {"type": "generated", **self.source.params()}
        else:
            raise ParameterError("Источник данных не задан")
        for algorithm in self.algorithms:
            algorithm.validate_params()
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "source": source,
            "columns": {"x": self.x_column, "y": self.y_column, "labels": self.label_column},
            "preprocessing": self.preprocessing,
            "algorithms": [a.to_dict() for a in self.algorithms],
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        """Записывает параметры в файл JSON (UTF-8) через временный файл."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

        def write(tmp: str) -> None:
            Path(tmp).write_text(text, encoding="utf-8")

        atomic_write(path, write)

    # --- чтение -----------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Читает параметры из файла JSON.

        Файл сначала полностью разбирается и проверяется; текущие параметры
        меняются, только если ошибок нет.

        :raises ExperimentFileError: с причиной отказа
        """
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise ExperimentFileError(f"Не удалось прочитать файл параметров: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ExperimentFileError(
                "Файл параметров повреждён: нарушена структура JSON") from None
        try:
            fields = self._parse(data)
        except ExperimentFileError:
            raise
        except ParameterError as exc:
            raise ExperimentFileError(f"Недопустимое значение параметра. {exc}") from exc
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ExperimentFileError(
                "Нарушена структура файла параметров: это не файл эксперимента CC2D") from None
        for name, value in fields.items():
            setattr(self, name, value)

    @staticmethod
    def _parse(data: dict) -> dict:
        if data.get("format") != FORMAT:
            raise ExperimentFileError(
                "Нарушена структура файла параметров: это не файл эксперимента CC2D")
        src = data["source"]
        if src["type"] == "csv":
            path = str(src["path"])
            if not Path(path).is_file():
                raise ExperimentFileError(f"Файл данных не найден: {path}")
            source: DataSource = CsvLoader(path)
        elif src["type"] == "generated":
            if src["distribution"] not in DISTRIBUTIONS:
                raise ParameterError(f"Неизвестный тип распределения: {src['distribution']}")
            source = DataGenerator(src["distribution"],
                                   **{name: src[name] for name in GENERATOR_PARAMS})
            source.validate()
        else:
            raise ValueError(src["type"])

        columns = data["columns"]
        x, y, labels = columns["x"], columns["y"], columns["labels"]
        if not isinstance(x, str) or not isinstance(y, str) or not (
                labels is None or isinstance(labels, str)):
            raise ValueError("columns")

        preprocessing = data["preprocessing"]
        if preprocessing not in METHODS:
            raise ParameterError(f"Неизвестный способ предобработки: {preprocessing}")

        items = data["algorithms"]
        if not isinstance(items, list) or len(items) > MAX_ALGORITHMS:
            raise ValueError("algorithms")
        algorithms = []
        for item in items:
            cls = BY_KEY[item["name"]]
            params = item["params"]
            if set(params) != set(cls.PARAMS):
                raise ValueError("params")
            algorithm = cls(**params)
            algorithm.validate_params()
            algorithms.append(algorithm)
        if len({a.key for a in algorithms}) != len(algorithms):
            raise ValueError("duplicates")

        seed = data["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < SEED_LIMIT:
            raise ParameterError("seed: нужно целое число от 0 до 2147483647")

        return {"source": source, "x_column": x, "y_column": y, "label_column": labels,
                "preprocessing": preprocessing, "algorithms": algorithms, "seed": seed}

    def restore_dataset(self) -> Dataset:
        """Получает набор данных по сохранённому источнику и проверяет столбцы.

        Для сгенерированного набора используется сохранённый seed, поэтому
        набор совпадает с исходным.

        :raises ExperimentFileError: если набор нельзя получить или в нём нет
            нужных столбцов
        """
        try:
            dataset = self.source.get_data(self.seed)
        except CC2DError as exc:
            raise ExperimentFileError(f"Не удалось получить набор данных: {exc}") from exc
        for name in (self.x_column, self.y_column):
            if name not in dataset.numeric_columns:
                raise ExperimentFileError(f"В наборе данных нет числового столбца «{name}»")
        if self.label_column is not None and self.label_column not in dataset.columns:
            raise ExperimentFileError(f"В наборе данных нет столбца «{self.label_column}»")
        return dataset
