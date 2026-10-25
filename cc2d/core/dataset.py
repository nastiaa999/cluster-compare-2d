"""Класс «Набор данных»."""

from __future__ import annotations

import pandas as pd

from .parsing import is_numeric_column

#: Вид источника: набор загружен из файла CSV.
SOURCE_CSV = "csv"
#: Вид источника: набор получен генерацией.
SOURCE_GENERATED = "generated"


class Dataset:
    """Набор данных (рис. 2.5): исходная таблица, имена числовых столбцов и эталонные метки.

    Объект создаётся источником данных и в ходе работы не изменяется
    (п. 4.2.3 ТЗ): операции возвращают копии.

    :param data: исходная таблица; для файла CSV значения хранятся строками
        в том виде, в каком записаны в файле
    :param source_kind: :data:`SOURCE_CSV` или :data:`SOURCE_GENERATED`
    :param label_column: столбец эталонных меток, который предлагается выбрать
        сразу (у сгенерированного набора это метки генератора)
    :param path: путь к файлу, если набор загружен из файла
    """

    def __init__(self, data: pd.DataFrame, source_kind: str,
                 label_column: str | None = None, path: str | None = None):
        self._data = data
        self.source_kind = source_kind
        self.label_column = label_column
        self.path = path
        self.numeric_columns = [c for c in data.columns if is_numeric_column(data[c])]

    @property
    def columns(self) -> list[str]:
        """Имена всех столбцов в порядке файла."""
        return list(self._data.columns)

    @property
    def row_count(self) -> int:
        """Число строк данных."""
        return len(self._data)

    @property
    def is_generated(self) -> bool:
        """Набор получен генерацией (условие П9)."""
        return self.source_kind == SOURCE_GENERATED

    def default_columns(self) -> tuple[str, str, str | None]:
        """Столбцы, выбираемые сразу после загрузки: первые два числовых и метки."""
        x, y = self.numeric_columns[0], self.numeric_columns[1]
        return x, y, self.label_column

    def get_rows(self, n: int) -> pd.DataFrame:
        """Возвращает первые n строк для показа пользователю."""
        return self._data.head(n).copy()

    def get_columns(self, names) -> pd.DataFrame:
        """Возвращает столбцы с заданными именами (повторы имён допускаются один раз)."""
        unique = list(dict.fromkeys(names))
        return self._data[unique].copy()

    def get_all(self) -> pd.DataFrame:
        """Возвращает копию всей таблицы (для экспорта сгенерированного набора)."""
        return self._data.copy()
