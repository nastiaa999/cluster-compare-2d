"""Описания параметров с проверкой типа и диапазона.

Параметры генерации (п. 4.1.2.2 ТЗ) и алгоритмов (п. 4.1.2.4 ТЗ) описываются
объектами этого модуля. Каждое описание умеет разобрать значение, введённое
пользователем (строку с точкой или запятой либо число), и сообщить причину
отказа, которую интерфейс выводит рядом с полем.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ParameterError

#: Значение размера окна Mean Shift, при котором окно оценивается по данным.
AUTO = "авто"


def format_number(value: float) -> str:
    """Возвращает число в русской записи: десятичная запятая, без лишних нулей."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value)).replace(".", ",")


def _to_float(raw: Any) -> float:
    """Разбирает вещественное число из строки или числа.

    :raises ValueError: если значение не является конечным числом
    """
    if isinstance(raw, bool):
        raise ValueError(raw)
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().replace(",", ".")
        if not text:
            raise ValueError(raw)
        value = float(text)
    if not math.isfinite(value):
        raise ValueError(raw)
    return value


@dataclass(frozen=True)
class FloatParam:
    """Вещественный параметр с диапазоном [low; high]."""

    name: str
    label: str
    low: float
    high: float
    default: float

    def range_text(self) -> str:
        return f"от {format_number(self.low)} до {format_number(self.high)}"

    def parse(self, raw: Any) -> float:
        """Возвращает разобранное значение или выбрасывает ParameterError."""
        try:
            value = _to_float(raw)
        except ValueError:
            raise ParameterError(f"{self.label}: нужно число {self.range_text()}") from None
        if not self.low <= value <= self.high:
            raise ParameterError(f"{self.label}: значение должно быть {self.range_text()}")
        return value


@dataclass(frozen=True)
class IntParam:
    """Целочисленный параметр с диапазоном [low; high]."""

    name: str
    label: str
    low: int
    high: int
    default: int

    def range_text(self) -> str:
        return f"от {self.low} до {self.high}"

    def parse(self, raw: Any) -> int:
        """Возвращает разобранное значение или выбрасывает ParameterError."""
        try:
            value = _to_float(raw)
        except ValueError:
            raise ParameterError(
                f"{self.label}: нужно целое число {self.range_text()}") from None
        if not value.is_integer():
            raise ParameterError(
                f"{self.label}: нужно целое число {self.range_text()}")
        if not self.low <= value <= self.high:
            raise ParameterError(f"{self.label}: значение должно быть {self.range_text()}")
        return int(value)


@dataclass(frozen=True)
class FloatOrAutoParam(FloatParam):
    """Вещественный параметр, который может принимать значение «авто»."""

    def parse(self, raw: Any) -> float | str:
        if isinstance(raw, str) and raw.strip().lower() == AUTO:
            return AUTO
        try:
            value = _to_float(raw)
        except ValueError:
            raise ParameterError(
                f"{self.label}: нужно число {self.range_text()} или «{AUTO}»") from None
        if not self.low <= value <= self.high:
            raise ParameterError(
                f"{self.label}: значение должно быть {self.range_text()} или «{AUTO}»")
        return value


@dataclass(frozen=True)
class ChoiceParam:
    """Параметр, выбираемый из списка значений."""

    name: str
    label: str
    choices: tuple
    default: Any

    def parse(self, raw: Any) -> Any:
        if raw not in self.choices:
            allowed = ", ".join(str(c) for c in self.choices)
            raise ParameterError(f"{self.label}: допустимые значения: {allowed}")
        return raw
