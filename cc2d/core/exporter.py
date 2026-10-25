"""Класс «Экспортёр»: запись результатов в файлы CSV и PNG."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluator import CALINSKI_HARABASZ, DAVIES_BOULDIN, SILHOUETTE
from .fileio import atomic_write
from .preprocessing import PreparedData
from .result import NOISE, ClusteringResult
from .summary import COLUMNS, TIME, UNDEFINED, SummaryTable, format_decimal

SUMMARY_FILE = "summary.csv"
EXPERIMENT_FILE = "experiment.json"
DATASET_FILE = "dataset.csv"
#: Заголовок столбца меток в файле меток кластеров.
CLUSTER_COLUMN = "cluster"

#: Размеры диаграмм в пикселях (не меньше 1200 × 800, п. 4.5.3.1 ТЗ).
PLOT_SIZES = ((1200, 800), (1800, 1200), (2400, 1600))
DEFAULT_PLOT_SIZE = (1800, 1200)
PLOT_DPI = 100

#: Знаков после запятой в CSV: метрики 4, время 3 (решение 12.5).
CSV_DECIMALS = {SILHOUETTE: 4, DAVIES_BOULDIN: 4, CALINSKI_HARABASZ: 4,
                "ari": 4, "nmi": 4, TIME: 3}
#: Кодировка CSV: UTF-8 с меткой порядка байтов, чтобы русский Excel
#: правильно показывал кириллицу; программа сама читает такие файлы.
CSV_ENCODING = "utf-8-sig"


def labels_file(key: str) -> str:
    """Имя файла меток кластеров алгоритма."""
    return f"labels_{key}.csv"


def plot_file(key: str) -> str:
    """Имя файла диаграммы алгоритма."""
    return f"plot_{key}.png"


def format_coordinate(value: float) -> str:
    """Координата без округления (кратчайшая точная запись) с десятичной запятой."""
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", ",")


def user_cluster_number(label: int) -> int:
    """Номер кластера для пользователя: с единицы, шум остаётся −1."""
    return NOISE if label == NOISE else label + 1


class Exporter:
    """Экспортёр (рис. 2.5).

    Файлы CSV записываются в UTF-8 с разделителем «;» и десятичной запятой,
    все файлы — через временный файл с переименованием.
    """

    def __init__(self):
        self.format = "csv"

    @staticmethod
    def _write_csv(path: str | Path, header: list[str], rows) -> None:
        def write(tmp: str) -> None:
            with open(tmp, "w", encoding=CSV_ENCODING, newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(header)
                writer.writerows(rows)

        atomic_write(path, write)

    def export_table(self, table: SummaryTable, path: str | Path) -> None:
        """Записывает сводную таблицу (рис. 2.14).

        Неопределённая внутренняя метрика записывается как «–», ARI и NMI без
        эталонных меток — пустой ячейкой.
        """
        header = [title for _, title in COLUMNS]
        rows = []
        for row in range(len(table.rows)):
            cells = []
            for column, _ in COLUMNS:
                value = table.value(row, column)
                if column in CSV_DECIMALS:
                    if not table.is_computed(row, column):
                        cells.append("")
                    elif value is None:
                        cells.append(UNDEFINED)
                    else:
                        cells.append(format_decimal(value, CSV_DECIMALS[column]))
                else:
                    cells.append(value)
            rows.append(cells)
        self._write_csv(path, header, rows)

    def export_result(self, result: ClusteringResult, data: PreparedData,
                      path: str | Path) -> None:
        """Записывает метки кластеров (рис. 2.13).

        Координаты исходные, до предобработки; строки в порядке исходного
        файла без исключённых строк; кластеры с единицы, шум −1.
        """
        labels = result.get_labels()
        rows = ([format_coordinate(x), format_coordinate(y), user_cluster_number(int(lab))]
                for (x, y), lab in zip(data.raw_points, labels))
        self._write_csv(path, [data.x_name, data.y_name, CLUSTER_COLUMN], rows)

    def export_dataset(self, rows: pd.DataFrame, path: str | Path) -> None:
        """Записывает сгенерированный набор (столбцы x; y; label)."""
        out = []
        for record in rows.itertuples(index=False):
            out.append([format_coordinate(v) if isinstance(v, (float, np.floating)) else v
                        for v in record])
        self._write_csv(path, list(rows.columns), out)

    def save_figure(self, figure, path: str | Path,
                    size: tuple[int, int] = DEFAULT_PLOT_SIZE) -> None:
        """Сохраняет диаграмму Matplotlib в PNG заданного размера в пикселях."""
        width, height = size
        figure.set_size_inches(width / PLOT_DPI, height / PLOT_DPI)

        def write(tmp: str) -> None:
            figure.savefig(tmp, dpi=PLOT_DPI, format="png")

        atomic_write(path, write)
