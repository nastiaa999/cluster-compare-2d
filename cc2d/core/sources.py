"""Источники данных: абстрактный класс, загрузчик CSV и генератор данных."""

from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from .dataset import SOURCE_CSV, SOURCE_GENERATED, Dataset
from .errors import DataFormatError, ParameterError
from .params import FloatParam, IntParam

#: Допустимое число строк данных в файле CSV (п. 4.1.2.1 ТЗ).
MIN_ROWS, MAX_ROWS = 10, 5000
#: Минимальное число числовых столбцов.
MIN_NUMERIC_COLUMNS = 2


class DataSource(ABC):
    """Источник данных (абстрактный класс, рис. 2.5)."""

    @abstractmethod
    def get_data(self, seed: int | None = None) -> Dataset:
        """Создаёт набор данных.

        :param seed: начальное значение генератора случайных чисел
        """

    @abstractmethod
    def validate(self) -> None:
        """Проверяет параметры источника; при ошибке выбрасывает исключение."""


class CsvLoader(DataSource):
    """Загрузчик CSV: читает файл формата п. 4.5.2.1 ТЗ.

    Кодировка UTF-8 (в том числе с меткой порядка байтов) или Windows-1251,
    разделитель полей «;», заголовок в первой строке, десятичный разделитель
    точка или запятая.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.encoding: str | None = None

    def validate(self) -> None:
        if not Path(self.path).is_file():
            raise DataFormatError(f"Файл не найден: {self.path}")

    def get_data(self, seed: int | None = None) -> Dataset:
        """Читает и проверяет файл. Значение seed не используется."""
        self.validate()
        try:
            raw = Path(self.path).read_bytes()
        except OSError as exc:
            raise DataFormatError(f"Не удалось прочитать файл: {exc.strerror}") from exc
        text = self._decode(raw)
        header, rows = self._parse(text)
        if not MIN_ROWS <= len(rows) <= MAX_ROWS:
            raise DataFormatError(
                f"В файле {len(rows)} строк данных, допустимо от {MIN_ROWS} до {MAX_ROWS}")
        data = pd.DataFrame(rows, columns=header, dtype=object)
        dataset = Dataset(data, SOURCE_CSV, path=self.path)
        if len(dataset.numeric_columns) < MIN_NUMERIC_COLUMNS:
            raise DataFormatError(
                "В файле меньше двух числовых столбцов, а для расчёта нужны два признака")
        return dataset

    def _decode(self, raw: bytes) -> str:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or self._looks_like_utf16(raw):
            raise DataFormatError(
                "Кодировка файла не поддерживается: нужна UTF-8 или Windows-1251")
        if b"\x00" in raw:
            raise DataFormatError("Неверный формат файла: файл не является текстовым")
        try:
            text = raw.decode("utf-8-sig")
            self.encoding = "UTF-8"
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1251")
            except UnicodeDecodeError:
                raise DataFormatError(
                    "Кодировка файла не поддерживается: нужна UTF-8 или Windows-1251") from None
            self.encoding = "Windows-1251"
        control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\r\n")
        if control > 0.01 * max(len(text), 1):
            raise DataFormatError("Неверный формат файла: файл не является текстовым")
        if not text.strip():
            raise DataFormatError("Файл пуст")
        return text

    @staticmethod
    def _looks_like_utf16(raw: bytes) -> bool:
        """UTF-16 без метки порядка байтов: каждый второй байт нулевой."""
        sample = raw[:2000]
        if len(sample) < 4:
            return False
        zeros_even = sample[0::2].count(0) / len(sample[0::2])
        zeros_odd = sample[1::2].count(0) / len(sample[1::2])
        return max(zeros_even, zeros_odd) > 0.4

    @staticmethod
    def _parse(text: str) -> tuple[list[str], list[list[str]]]:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=";")
        try:
            header = next(reader, None)
            if not header:
                raise DataFormatError("Файл пуст")
            header = [name.strip() for name in header]
            if len(header) < 2:
                raise DataFormatError(
                    "Неверная структура файла: в заголовке одно поле. "
                    "Поля должны разделяться точкой с запятой («;»)")
            if any(not name for name in header):
                raise DataFormatError("Неверная структура файла: в заголовке есть столбец без имени")
            duplicates = sorted({name for name in header if header.count(name) > 1})
            if duplicates:
                raise DataFormatError(
                    f"Неверная структура файла: повторяется имя столбца «{duplicates[0]}»")
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise DataFormatError(
                        f"Неверная структура файла: в строке {reader.line_num} "
                        f"полей {len(row)}, а в заголовке {len(header)}")
                rows.append(row)
        except csv.Error as exc:
            raise DataFormatError(f"Неверная структура файла: {exc}") from exc
        return header, rows


# --- Генератор данных ---------------------------------------------------------

#: Параметры генерации (п. 4.1.2.2 ТЗ).
GENERATOR_PARAMS = {
    "n_points": IntParam("n_points", "Число точек", 50, 5000, 500),
    "n_clusters": IntParam("n_clusters", "Число кластеров", 2, 10, 3),
    "spread": FloatParam("spread", "Разброс", 0.1, 5.0, 1.0),
    "noise": FloatParam("noise", "Уровень шума", 0.01, 0.5, 0.05),
}

#: Метка шумовых точек в столбце меток генератора.
NOISE_LABEL = -1
#: Запас прямоугольника для шумовых точек по каждой оси (доля размаха облаков).
NOISE_MARGIN = 0.1


@dataclass(frozen=True)
class DistributionType:
    """Тип распределения точек генератора.

    :param generate: функция (генератор, seed) → (точки n×2, метки)
    :param fields: параметры генерации, которые использует этот тип
    """

    generate: Callable[["DataGenerator", int], tuple[np.ndarray, np.ndarray]]
    fields: tuple[str, ...]


def _gaussian_blobs(gen: "DataGenerator", seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Гауссовы облака с долей равномерного шума.

    Число шумовых точек равно округлённому произведению уровня шума на число
    точек, общее число точек не меняется. Шум равномерно распределён
    в прямоугольнике, охватывающем облака, с запасом 10 % по каждой оси.
    Метки облаков 1…k, шума −1.
    """
    n_noise = int(gen.noise * gen.n_points + 0.5)
    n_blobs = gen.n_points - n_noise
    points, labels = make_blobs(n_samples=n_blobs, centers=gen.n_clusters,
                                cluster_std=gen.spread, random_state=seed)
    rng = np.random.default_rng(seed)
    low, high = points.min(axis=0), points.max(axis=0)
    margin = NOISE_MARGIN * (high - low)
    noise = rng.uniform(low - margin, high + margin, size=(n_noise, 2))
    points = np.vstack([points, noise])
    labels = np.concatenate([labels + 1, np.full(n_noise, NOISE_LABEL)])
    order = rng.permutation(gen.n_points)
    return points[order], labels[order]


#: Название типа «Гауссовы облака».
GAUSSIAN_BLOBS = "Гауссовы облака"

#: Реестр типов распределения: новый тип добавляется функцией и записью здесь.
DISTRIBUTIONS: dict[str, DistributionType] = {
    GAUSSIAN_BLOBS: DistributionType(_gaussian_blobs,
                                     ("n_points", "n_clusters", "spread", "noise")),
}


class DataGenerator(DataSource):
    """Генератор данных: создаёт синтетический набор с метками генератора.

    Значения параметров могут быть строками, введёнными пользователем;
    :meth:`validate` разбирает их и проверяет диапазоны.
    """

    #: Имена столбцов сгенерированного набора.
    COLUMNS = ("x", "y", "label")

    def __init__(self, distribution: str = GAUSSIAN_BLOBS, n_points=500, n_clusters=3,
                 spread=1.0, noise=0.05):
        self.distribution = distribution
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.spread = spread
        self.noise = noise

    def active_fields(self) -> tuple[str, ...]:
        """Параметры, которые использует выбранный тип распределения."""
        return DISTRIBUTIONS[self.distribution].fields

    def errors(self) -> dict[str, str]:
        """Возвращает ошибки параметров {имя параметра: причина}; пусто, если всё верно."""
        if self.distribution not in DISTRIBUTIONS:
            return {"distribution": f"Неизвестный тип распределения: {self.distribution}"}
        result = {}
        for name in self.active_fields():
            try:
                GENERATOR_PARAMS[name].parse(getattr(self, name))
            except ParameterError as exc:
                result[name] = str(exc)
        return result

    def validate(self) -> None:
        """Проверяет и нормализует параметры (условие П8)."""
        errors = self.errors()
        if errors:
            raise ParameterError(next(iter(errors.values())))
        for name in self.active_fields():
            setattr(self, name, GENERATOR_PARAMS[name].parse(getattr(self, name)))

    def params(self) -> dict:
        """Параметры генерации для файла эксперимента."""
        return {"distribution": self.distribution, "n_points": self.n_points,
                "n_clusters": self.n_clusters, "spread": self.spread, "noise": self.noise}

    def get_data(self, seed: int | None = None) -> Dataset:
        """Генерирует набор; один и тот же seed даёт один и тот же набор."""
        self.validate()
        if seed is None:
            raise ParameterError("Для генерации нужно значение seed")
        points, labels = DISTRIBUTIONS[self.distribution].generate(self, int(seed))
        x, y, label = self.COLUMNS
        data = pd.DataFrame({x: points[:, 0], y: points[:, 1], label: labels.astype(int)})
        return Dataset(data, SOURCE_GENERATED, label_column=label)
