"""Окно сравнения результатов (рис. 2.18 РПЗ)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import Qt, Signal  # noqa: E402
from PySide6.QtGui import QColor, QFont  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,  # noqa: E402
                               QLabel, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..core.summary import COLUMNS, SummaryTable  # noqa: E402
from ..visualization import Visualizer  # noqa: E402
from .style import COLORS  # noqa: E402

#: Во сколько раз меняется масштаб за один шаг колеса мыши.
ZOOM_STEP = 1.2


class PlotCanvas(FigureCanvasQTAgg):
    """Холст диаграмм: масштаб колесом мыши, перемещение левой кнопкой, сброс вида.

    Оси всех диаграмм общие (решение 12.14), поэтому масштаб и перемещение
    одной диаграммы применяются ко всем.
    """

    def __init__(self):
        super().__init__(Figure())
        self._home: list[tuple] = []
        self._drag = None
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def remember_view(self) -> None:
        """Запоминает исходный вид диаграмм для кнопки «Сбросить вид»."""
        self._home = [(ax, ax.get_xlim(), ax.get_ylim()) for ax in self.figure.axes]

    def reset_view(self) -> None:
        """Возвращает исходный масштаб и положение."""
        for ax, xlim, ylim in self._home:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.draw_idle()

    def _on_scroll(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        factor = 1 / ZOOM_STEP if event.button == "up" else ZOOM_STEP
        for limits, setter, center in ((ax.get_xlim(), ax.set_xlim, event.xdata),
                                       (ax.get_ylim(), ax.set_ylim, event.ydata)):
            low, high = limits
            setter(center - (center - low) * factor, center + (high - center) * factor)
        self.draw_idle()

    def _on_press(self, event) -> None:
        if event.button == 1 and event.inaxes is not None:
            ax = event.inaxes
            self._drag = (ax, event.x, event.y, ax.get_xlim(), ax.get_ylim())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event) -> None:
        if self._drag is None:
            return
        ax, x0, y0, (xl, xh), (yl, yh) = self._drag
        dx = (event.x - x0) * (xh - xl) / ax.bbox.width
        dy = (event.y - y0) * (yh - yl) / ax.bbox.height
        ax.set_xlim(xl - dx, xh - dx)
        ax.set_ylim(yl - dy, yh - dy)
        self.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()


class CompareWindow(QWidget):
    """Окно сравнения: диаграммы всех запущенных алгоритмов и сводная таблица.

    Окно немодальное и не имеет родителя, поэтому у него своя кнопка на панели
    задач. Кнопка закрытия в заголовке только сворачивает окно; закрывается
    оно вместе с главным окном.
    """

    #: Нажата кнопка «Экспорт…».
    export_requested = Signal()
    #: Нажата кнопка «В главное окно».
    main_requested = Signal()

    def __init__(self, visualizer: Visualizer | None = None):
        super().__init__()
        self.setObjectName("CompareWindow")
        self.setWindowTitle("ClusterCompare 2D: сравнение результатов")
        self.resize(1200, 860)
        self.visualizer = visualizer or Visualizer()
        self.allow_close = False

        reset = QPushButton("Сбросить вид")
        reset.setToolTip("Вернуть исходный масштаб диаграмм")
        hint = QLabel("Колесо мыши: масштаб. Перетаскивание левой кнопкой: перемещение.")
        hint.setObjectName("Muted")
        top = QHBoxLayout()
        top.addWidget(reset)
        top.addWidget(hint)
        top.addStretch()

        self.canvas = PlotCanvas()
        self.canvas.setMinimumHeight(300)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels([title for _, title in COLUMNS])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        table_title = QLabel("<b>Сводная таблица</b> (лучшие значения выделены)")
        table_box = QWidget()
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 6, 0, 0)
        table_layout.addWidget(table_title)
        table_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(table_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        export = QPushButton("Экспорт…")
        to_main = QPushButton("В главное окно")
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(export)
        bottom.addWidget(to_main)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addLayout(bottom)

        reset.clicked.connect(self.canvas.reset_view)
        export.clicked.connect(self.export_requested)
        to_main.clicked.connect(self.main_requested)

    def show_results(self, results, points, table: SummaryTable) -> None:
        """Показывает новые результаты вместо прежних."""
        self.visualizer.build(results, points, self.canvas.figure)
        self.canvas.draw()
        self.canvas.remember_view()
        self._fill_table(table)

    def _fill_table(self, table: SummaryTable) -> None:
        self.table.setRowCount(len(table.rows))
        bold = QFont()
        bold.setBold(True)
        for row in range(len(table.rows)):
            for col, (key, _) in enumerate(COLUMNS):
                item = QTableWidgetItem(table.display(row, key))
                if key in ("algorithm", "message"):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if table.is_best(row, key):
                    item.setBackground(QColor(COLORS["best"]))
                    item.setFont(bold)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        # Таблица всегда показывает все строки (их не больше четырёх) без прокрутки.
        height = (self.table.horizontalHeader().sizeHint().height()
                  + sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
                  + 2 * self.table.frameWidth() + 2)
        self.table.setFixedHeight(height)

    def closeEvent(self, event) -> None:
        """Крестик в заголовке сворачивает окно (решение пользователя, РПЗ 2.8)."""
        if self.allow_close:
            event.accept()
        else:
            event.ignore()
            self.showMinimized()
