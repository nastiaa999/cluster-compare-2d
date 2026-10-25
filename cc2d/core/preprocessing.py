"""Класс «Предобработчик»: исключение некорректных строк и нормализация признаков."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .dataset import Dataset
from .errors import DataPreparationError
from .parsing import is_empty, to_numeric

#: Стандартизация (z-оценка), способ по умолчанию (п. 4.1.2.3 ТЗ).
STANDARD = "standard"
#: Масштабирование в диапазон [0; 1].
MINMAX = "minmax"
#: Способы предобработки и их названия для интерфейса.
METHODS = {
    STANDARD: "Стандартизация (z-оценка)",
    MINMAX: "Масштабирование в диапазон [0; 1]",
}
#: Наименьшее число строк, участвующих в расчёте.
MIN_ROWS_FOR_RUN = 10


@dataclass
class PreparedData:
    """Подготовленные данные: результат операции «обработать».

    :param points: точки после предобработки, массив n × 2
    :param raw_points: те же точки до предобработки (для файла меток, решение 12.5)
    :param reference_labels: эталонные метки или None, если столбец не выбран
    :param row_index: номера исходных строк, вошедших в расчёт
    :param x_name: имя столбца X
    :param y_name: имя столбца Y
    :param excluded: число исключённых строк
    """

    points: np.ndarray
    raw_points: np.ndarray
    reference_labels: np.ndarray | None
    row_index: np.ndarray
    x_name: str
    y_name: str
    excluded: int


class Preprocessor:
    """Предобработчик (рис. 2.5).

    Строка исключается, если в столбце X или Y пропуск или нечисловое значение
    либо если в выбранном столбце меток пропуск (п. 4.1.1.1 ТЗ).

    :param method: :data:`STANDARD` или :data:`MINMAX`
    """

    def __init__(self, method: str = STANDARD):
        self.method = method
        self.dataset: Dataset | None = None
        self.excluded_count = 0

    def _valid_mask(self, x: str, y: str, label_column: str | None) -> np.ndarray:
        if self.dataset is None:
            raise DataPreparationError("Набор данных не загружен")
        frame = self.dataset.get_columns([c for c in (x, y, label_column) if c])
        mask = np.isfinite(to_numeric(frame[x])) & np.isfinite(to_numeric(frame[y]))
        if label_column:
            mask &= ~is_empty(frame[label_column])
        return mask

    def count_excluded(self, x: str, y: str, label_column: str | None = None) -> int:
        """Подсчитывает строки, которые будут исключены; данные не изменяются."""
        return int((~self._valid_mask(x, y, label_column)).sum())

    def readiness_problem(self, x: str, y: str, label_column: str | None = None) -> str | None:
        """Проверяет, подготовлен ли набор для расчёта (условие П1).

        :return: причина, по которой расчёт невозможен, или None
        """
        if self.dataset is None:
            return "Набор данных не загружен"
        mask = self._valid_mask(x, y, label_column)
        remaining = int(mask.sum())
        if remaining < MIN_ROWS_FOR_RUN:
            return (f"После исключения строк осталось {remaining}, "
                    f"а для расчёта нужно не меньше {MIN_ROWS_FOR_RUN}")
        frame = self.dataset.get_columns([x, y])
        for name in dict.fromkeys((x, y)):
            values = to_numeric(frame[name])[mask]
            if np.ptp(values) == 0:
                return f"Признак «{name}» постоянен: все его значения одинаковы"
        return None

    def process(self, x: str, y: str, label_column: str | None = None) -> PreparedData:
        """Исключает некорректные строки и выполняет предобработку.

        :raises DataPreparationError: если набор не подготовлен (условие П1)
        """
        problem = self.readiness_problem(x, y, label_column)
        if problem:
            raise DataPreparationError(problem)
        mask = self._valid_mask(x, y, label_column)
        frame = self.dataset.get_columns([c for c in (x, y, label_column) if c])
        raw = np.column_stack([to_numeric(frame[x]), to_numeric(frame[y])])[mask]
        scaler = StandardScaler() if self.method == STANDARD else MinMaxScaler()
        points = scaler.fit_transform(raw)
        labels = None
        if label_column:
            labels = frame[label_column].astype(str).str.strip().to_numpy()[mask]
        self.excluded_count = int((~mask).sum())
        return PreparedData(points=points, raw_points=raw, reference_labels=labels,
                            row_index=np.flatnonzero(mask), x_name=x, y_name=y,
                            excluded=self.excluded_count)
