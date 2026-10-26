"""Виджеты главного окна: поле с проверкой, карточка алгоритма, модель таблицы данных."""

from __future__ import annotations

import numbers

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..core.algorithms.base import ClusteringAlgorithm
from ..core.params import ChoiceParam, format_number
from .style import COLORS


def value_text(value) -> str:
    """Текст значения параметра для поля ввода (десятичная запятая)."""
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return format_number(value)
    return str(value)


def set_invalid(edit: QLineEdit, invalid: bool) -> None:
    """Включает или снимает красную рамку поля с ошибкой."""
    if edit.property("invalid") != invalid:
        edit.setProperty("invalid", invalid)
        edit.style().unpolish(edit)
        edit.style().polish(edit)


class ValidatedField(QWidget):
    """Поле ввода с причиной ошибки под ним (п. 4.2.1 ТЗ: причина выводится рядом с полем).

    :param placeholder: подсказка в пустом поле
    """

    #: Текст поля изменён пользователем.
    edited = Signal()

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.error = QLabel()
        self.error.setObjectName("FieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.edit)
        layout.addWidget(self.error)
        self.edit.textEdited.connect(lambda _: self.edited.emit())

    def text(self) -> str:
        return self.edit.text()

    def set_text(self, text: str) -> None:
        self.edit.setText(text)

    def set_error(self, message: str | None) -> None:
        """Показывает причину ошибки или скрывает её, если message пусто."""
        invalid = bool(message)
        set_invalid(self.edit, invalid)
        self.error.setText(message or "")
        self.error.setVisible(invalid)


class AlgorithmCard(QFrame):
    """Карточка алгоритма: флажок, поля параметров и кнопка «По умолчанию».

    Значения полей записываются в параметры алгоритма как введённый текст,
    проверку выполняет сам алгоритм (операция «проверитьПараметры»).

    :param algorithm: объект алгоритма, параметры которого редактирует карточка
    """

    #: Изменён флажок или значение параметра.
    changed = Signal()

    def __init__(self, algorithm: ClusteringAlgorithm, parent=None):
        super().__init__(parent)
        self.algorithm = algorithm
        self.setObjectName("AlgorithmCard")
        self.check = QCheckBox(algorithm.title)
        self.check.setObjectName("AlgorithmTitle")
        self.defaults = QPushButton("По умолчанию")
        self.defaults.setObjectName("Link")
        self.defaults.setToolTip("Вернуть значения по умолчанию (п. 4.1.2.4 ТЗ)")
        self.fields: dict[str, QWidget] = {}

        # Одна строка, как на рис. 2.16: флажок, пары «подпись — поле», кнопка сброса.
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(8)
        self.check.setMinimumWidth(150)
        grid.addWidget(self.check, 0, 0, Qt.AlignmentFlag.AlignTop)
        for i, (name, spec) in enumerate(algorithm.PARAMS.items()):
            label = QLabel(spec.label)
            label.setObjectName("Muted")
            label.setFixedWidth(120)
            if isinstance(spec, ChoiceParam):
                widget = QComboBox()
                widget.addItems([str(c) for c in spec.choices])
                widget.currentTextChanged.connect(self._on_edit)
            else:
                widget = QLineEdit()
                widget.setToolTip(self._range_hint(spec))
                widget.textEdited.connect(self._on_edit)
            widget.setFixedWidth(110)
            grid.addWidget(label, 0, 1 + 2 * i, Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(widget, 0, 2 + 2 * i, Qt.AlignmentFlag.AlignVCenter)
            self.fields[name] = widget
        grid.setColumnStretch(5, 1)
        grid.addWidget(self.defaults, 0, 6, Qt.AlignmentFlag.AlignVCenter)
        # Причины ошибок выводятся одной строкой под полями карточки.
        self.error = QLabel()
        self.error.setObjectName("FieldError")
        self.error.setWordWrap(True)
        self.error.hide()
        grid.addWidget(self.error, 1, 1, 1, 6)

        self.check.toggled.connect(self._on_toggle)
        self.defaults.clicked.connect(self.reset)
        self.load_from_algorithm()
        self._on_toggle(False)

    @staticmethod
    def _range_hint(spec) -> str:
        text = f"Допустимо {spec.range_text()}"
        if spec.name == "bandwidth":
            text += " или «авто» (размер окна оценивается по данным)"
        return text

    def load_from_algorithm(self) -> None:
        """Заполняет поля значениями параметров алгоритма."""
        for name, widget in self.fields.items():
            text = value_text(self.algorithm.params[name])
            if isinstance(widget, QComboBox):
                widget.setCurrentText(text)
            else:
                widget.setText(text)

    def _on_edit(self, *_):
        for name, widget in self.fields.items():
            text = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            self.algorithm.params[name] = text
        self.changed.emit()

    def _on_toggle(self, checked: bool) -> None:
        for widget in self.fields.values():
            widget.setEnabled(checked)
        self.defaults.setEnabled(checked)
        self.setProperty("active", checked)
        self.style().unpolish(self)
        self.style().polish(self)
        self.changed.emit()

    def reset(self) -> None:
        """Кнопка «По умолчанию» (операция «сброситьПараметры»)."""
        self.algorithm.reset_params()
        self.load_from_algorithm()
        self.changed.emit()

    def is_checked(self) -> bool:
        return self.check.isChecked()

    def set_checked(self, checked: bool) -> None:
        self.check.setChecked(checked)

    def show_errors(self, errors: dict[str, str]) -> None:
        """Выделяет поля с ошибками и выводит причины под ними.

        У неотмеченного алгоритма ошибки не показываются: его параметры не
        участвуют в условии запуска.
        """
        shown = errors if self.is_checked() else {}
        for name, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                set_invalid(widget, name in shown)
        self.error.setText("; ".join(shown.values()))
        self.error.setVisible(bool(shown))


def cell_text(value) -> str:
    """Текст ячейки таблицы данных: строки как в файле, числа с десятичной запятой."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(",.").replace(".", ",") or "0"
    return str(value)


class DataTableModel(QAbstractTableModel):
    """Модель таблицы первых 100 строк набора с подсветкой выбранных столбцов."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._xy: set[str] = set()
        self._labels: str | None = None

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame.reset_index(drop=True)
        self.endResetModel()

    def set_highlight(self, x: str | None, y: str | None, labels: str | None) -> None:
        """Подсвечивает столбцы X и Y и столбец меток."""
        self._xy = {c for c in (x, y) if c}
        self._labels = labels
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(self.rowCount() - 1, self.columnCount() - 1))
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def _column_color(self, column: int) -> QColor | None:
        name = self._frame.columns[column]
        if name in self._xy:
            return QColor(COLORS["xy"])
        if name == self._labels:
            return QColor(COLORS["labels"])
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return cell_text(self._frame.iat[index.row(), index.column()])
        if role == Qt.ItemDataRole.BackgroundRole:
            return self._column_color(index.column())
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._frame.columns[section])
            return str(section + 1)
        if role == Qt.ItemDataRole.BackgroundRole and orientation == Qt.Orientation.Horizontal:
            return self._column_color(section)
        return None
