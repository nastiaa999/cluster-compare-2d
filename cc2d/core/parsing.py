"""Разбор значений таблицы: числа с точкой или запятой и пустые ячейки."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_numeric(column: pd.Series) -> np.ndarray:
    """Переводит столбец в массив float.

    Десятичный разделитель может быть точкой или запятой (п. 4.5.2.1 ТЗ).
    Пустые ячейки, текст и бесконечности дают NaN.
    """
    if pd.api.types.is_numeric_dtype(column):
        values = column.to_numpy(dtype=float, copy=True)
    else:
        text = column.astype(str).str.strip().str.replace(",", ".", regex=False)
        values = np.array(pd.to_numeric(text, errors="coerce"), dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def is_empty(column: pd.Series) -> np.ndarray:
    """Возвращает маску пустых ячеек столбца."""
    if pd.api.types.is_numeric_dtype(column):
        return column.isna().to_numpy()
    return (column.isna() | (column.astype(str).str.strip() == "")).to_numpy()


def is_numeric_column(column: pd.Series) -> bool:
    """Проверяет, считается ли столбец числовым.

    Столбец числовой, если больше половины его непустых значений разбираются
    как числа. Поэтому столбец с отдельными пропусками или опечатками остаётся
    числовым, а столбец текстовых меток нет.
    """
    filled = ~is_empty(column)
    count = int(filled.sum())
    if count == 0:
        return False
    numeric = int(np.isfinite(to_numeric(column))[filled].sum())
    return numeric * 2 > count
