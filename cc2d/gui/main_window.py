"""Главное окно (рис. 2.16 РПЗ), обработчики событий по диаграмме состояний (рис. 2.17)."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QStackedWidget, QTableView, QVBoxLayout, QWidget)

from ..core.algorithms import ALGORITHM_CLASSES
from ..core.dataset import Dataset
from ..core.errors import AllAlgorithmsFailedError, CC2DError
from ..core.experiment import Experiment, new_seed
from ..core.exporter import EXPERIMENT_FILE, Exporter
from ..core.preprocessing import METHODS, MINMAX, STANDARD, Preprocessor
from ..core.run_manager import RunManager, RunOutput
from ..core.sources import (DISTRIBUTIONS, GENERATOR_PARAMS, CsvLoader, DataGenerator,
                            DataSource)
from ..visualization import Visualizer
from .compare_window import CompareWindow
from .export_dialog import ExportDialog, RunResults
from .help_window import HelpWindow
from .state import Conditions
from .style import COLORS
from .widgets import AlgorithmCard, DataTableModel, ValidatedField, value_text

log = logging.getLogger(__name__)

#: Сколько строк набора показывается в таблице (п. 4.1.1.1 ТЗ).
PREVIEW_ROWS = 100
#: Пункт списка столбца меток, означающий, что столбец не выбран.
NO_LABELS = "не выбран"
#: Сколько миллисекунд показывается временное сообщение в строке состояния.
FLASH_MS = 4000


def _swatch(color: str) -> QLabel:
    """Цветной квадрат для легенды таблицы."""
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(color))
    label = QLabel()
    label.setPixmap(pixmap)
    return label


class MainWindow(QMainWindow):
    """Главное окно (граничный класс «Главное окно», рис. 2.5).

    Окно хранит набор данных, четыре объекта алгоритмов (параметры которых
    редактируют карточки) и результаты последнего расчёта. Параметры
    записываются в эксперимент, которым владеет менеджер запуска. После
    каждого события вызывается :meth:`refresh`, который пересчитывает условия
    доступности по диаграмме состояний (рис. 2.17).
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClusterCompare 2D")
        self.preprocessor = Preprocessor()
        self.manager = RunManager(self.preprocessor)
        self.visualizer = Visualizer()
        self.exporter = Exporter()
        self.algorithms = [cls() for cls in ALGORITHM_CLASSES]
        self.dataset: Dataset | None = None
        self.results: RunResults | None = None
        self.conditions = Conditions()
        self.running = False
        self.compare_window: CompareWindow | None = None
        self.help_window: HelpWindow | None = None
        self._thread = None
        self._worker = None
        self._run_dataset: Dataset | None = None
        self._cancelling = False
        self._updating = False
        self._flash_text = ""
        documents = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)
        self.folder = str(Path(documents) / "ClusterCompare 2D")

        self._build_menu()
        self._build_ui()
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)
        self.refresh()

    @property
    def experiment(self) -> Experiment:
        """Эксперимент, которым владеет менеджер запуска."""
        return self.manager.experiment

    # --- построение окна ---------------------------------------------------

    def _build_menu(self) -> None:
        def action(text, slot, shortcut=None):
            act = QAction(text, self)
            act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            return act

        self.act_open = action("Открыть файл CSV…", self.browse_csv, "Ctrl+O")
        self.act_load = action("Загрузить параметры…", self.load_params)
        self.act_save = action("Сохранить параметры…", self.save_params, "Ctrl+S")
        self.act_export = action("Экспорт…", lambda: self.open_export(self), "Ctrl+E")
        self.act_exit = action("Выход", self.close, "Ctrl+Q")
        self.act_help = action("Справка", self.open_help, "F1")
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_load)
        file_menu.addAction(self.act_save)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)
        self.menuBar().addMenu("Справка").addAction(self.act_help)

    def _build_ui(self) -> None:
        left = QVBoxLayout()
        left.addWidget(self._build_source_group())
        left.addWidget(self._build_features_group())
        left.addWidget(self._build_preprocessing_group())
        left.addWidget(self._build_params_group())
        left.addStretch()
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(410)
        left.setContentsMargins(0, 0, 0, 0)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self._build_data_group(), 1)
        right.addWidget(self._build_algorithms_group())

        columns = QHBoxLayout()
        columns.addWidget(left_widget)
        columns.addLayout(right, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 6, 12, 10)
        layout.addLayout(columns, 1)
        layout.addWidget(self._build_bottom_bar())
        self.setCentralWidget(central)

    def _build_source_group(self) -> QGroupBox:
        box = QGroupBox("Источник данных")
        self.radio_csv = QRadioButton("Файл CSV")
        self.radio_gen = QRadioButton("Генерация синтетического набора")
        self.radio_csv.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.radio_csv)
        group.addButton(self.radio_gen)

        self.path = QLineEdit()
        self.path.setReadOnly(True)
        self.path.setPlaceholderText("Файл не выбран")
        self.browse = QPushButton("Обзор…")
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(self.browse)

        self.distribution = QComboBox()
        self.distribution.addItems(list(DISTRIBUTIONS))
        self.gen_fields: dict[str, ValidatedField] = {}
        form = QFormLayout()
        form.setContentsMargins(22, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Тип распределения", self.distribution)
        for name, spec in GENERATOR_PARAMS.items():
            field = ValidatedField()
            field.set_text(value_text(spec.default))
            field.edit.setToolTip(f"Допустимо {spec.range_text()}")
            field.edited.connect(self.refresh)
            self.gen_fields[name] = field
            form.addRow(spec.label, field)
        self.generate = QPushButton("Сгенерировать набор")
        form.addRow("", self.generate)

        layout = QVBoxLayout(box)
        layout.addWidget(self.radio_csv)
        layout.addLayout(path_row)
        layout.addWidget(self.radio_gen)
        layout.addLayout(form)

        self.radio_gen.toggled.connect(self.refresh)
        self.browse.clicked.connect(self.browse_csv)
        self.generate.clicked.connect(self.generate_dataset)
        self.distribution.currentTextChanged.connect(self.refresh)
        return box

    def _build_features_group(self) -> QGroupBox:
        box = QGroupBox("Признаки")
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.label_combo = QComboBox()
        form = QFormLayout()
        form.addRow("X", self.x_combo)
        form.addRow("Y", self.y_combo)
        form.addRow("Столбец меток", self.label_combo)
        self.features_error = QLabel()
        self.features_error.setObjectName("GroupError")
        self.features_error.setWordWrap(True)
        self.features_error.hide()
        self.excluded = QLabel("Будет исключено строк: 0")
        self.excluded.setObjectName("Muted")
        layout = QVBoxLayout(box)
        layout.addLayout(form)
        layout.addWidget(self.features_error)
        layout.addWidget(self.excluded)
        for combo in (self.x_combo, self.y_combo, self.label_combo):
            combo.currentTextChanged.connect(self._on_columns_changed)
        return box

    def _build_preprocessing_group(self) -> QGroupBox:
        box = QGroupBox("Предобработка")
        self.radio_standard = QRadioButton(METHODS[STANDARD])
        self.radio_minmax = QRadioButton(METHODS[MINMAX])
        self.radio_standard.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.radio_standard)
        group.addButton(self.radio_minmax)
        layout = QVBoxLayout(box)
        layout.addWidget(self.radio_standard)
        layout.addWidget(self.radio_minmax)
        self.radio_standard.toggled.connect(self._on_preprocessing_changed)
        return box

    def _build_params_group(self) -> QGroupBox:
        box = QGroupBox("Параметры эксперимента")
        self.load_button = QPushButton("Загрузить параметры…")
        self.save_button = QPushButton("Сохранить параметры…")
        layout = QHBoxLayout(box)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        self.load_button.clicked.connect(self.load_params)
        self.save_button.clicked.connect(self.save_params)
        return box

    def _build_data_group(self) -> QGroupBox:
        self.data_box = QGroupBox("Данные")
        self.table_model = DataTableModel(self)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_view.horizontalHeader().setMinimumSectionSize(70)
        self.table_view.verticalHeader().setDefaultSectionSize(24)
        placeholder = QLabel("Откройте файл CSV или сгенерируйте набор,\n"
                             "и здесь появятся первые 100 строк данных.")
        placeholder.setObjectName("Placeholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.data_stack = QStackedWidget()
        self.data_stack.addWidget(placeholder)
        self.data_stack.addWidget(self.table_view)

        legend = QHBoxLayout()
        for color, text in ((COLORS["xy"], "столбцы X и Y"), (COLORS["labels"], "столбец меток")):
            legend.addWidget(_swatch(color))
            caption = QLabel(text)
            caption.setObjectName("Muted")
            legend.addWidget(caption)
            legend.addSpacing(16)
        legend.addStretch()
        layout = QVBoxLayout(self.data_box)
        layout.addWidget(self.data_stack, 1)
        layout.addLayout(legend)
        return self.data_box

    def _build_algorithms_group(self) -> QGroupBox:
        box = QGroupBox("Алгоритмы кластеризации")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)
        self.cards = []
        for algorithm in self.algorithms:
            card = AlgorithmCard(algorithm)
            card.changed.connect(self._on_algorithms_changed)
            layout.addWidget(card)
            self.cards.append(card)
        return box

    def _build_bottom_bar(self) -> QWidget:
        bar = QFrame()
        self.status = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.run_button = QPushButton("Запустить")
        self.run_button.setObjectName("Primary")
        self.run_button.setShortcut(QKeySequence("F5"))
        self.run_button.setToolTip("Запустить расчёт (F5)")
        self.cancel_button = QPushButton("Отменить")
        self.export_button = QPushButton("Экспорт…")
        self.help_button = QPushButton("Справка")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(self.status, 1)
        layout.addWidget(self.progress)
        layout.addSpacing(12)
        for button in (self.run_button, self.cancel_button, self.export_button, self.help_button):
            button.setMinimumWidth(110)
            layout.addWidget(button)
        self.run_button.clicked.connect(self.start_run)
        self.cancel_button.clicked.connect(self.cancel_run)
        self.export_button.clicked.connect(lambda: self.open_export(self))
        self.help_button.clicked.connect(self.open_help)
        return bar

    # --- сообщения (вынесены в методы, чтобы их можно было подменить в тестах) ---

    def inform(self, title: str, text: str) -> None:
        """Информационное сообщение."""
        QMessageBox.information(self, title, text)

    def warn(self, title: str, text: str) -> None:
        """Сообщение об ошибке или предупреждение."""
        QMessageBox.warning(self, title, text)

    def flash(self, text: str) -> None:
        """Временное сообщение в строке состояния."""
        self._flash_text = text
        self._flash_timer.start(FLASH_MS)
        self.refresh()

    def _end_flash(self) -> None:
        self._flash_text = ""
        self.refresh()

    # --- текущие значения полей ---------------------------------------------

    def selected_columns(self) -> tuple[str | None, str | None, str | None]:
        """Выбранные столбцы X, Y и меток (None, если не выбран)."""
        if self.dataset is None:
            return None, None, None
        label = self.label_combo.currentText()
        return (self.x_combo.currentText() or None, self.y_combo.currentText() or None,
                None if label in ("", NO_LABELS) else label)

    def generator_from_fields(self) -> DataGenerator:
        """Генератор с параметрами из полей формы."""
        return DataGenerator(self.distribution.currentText(),
                             **{name: field.text() for name, field in self.gen_fields.items()})

    def _remaining_rows(self) -> int | None:
        x, y, labels = self.selected_columns()
        if self.dataset is None or not x or not y:
            return None
        return self.dataset.row_count - self.preprocessor.count_excluded(x, y, labels)

    # --- доступность элементов -------------------------------------------------

    def refresh(self) -> None:
        """Пересчитывает условия и доступность элементов (рис. 2.17)."""
        if self._updating:
            return
        generation = self.radio_gen.isChecked()
        generator = self.generator_from_fields()
        gen_errors = generator.errors()
        active = set(DISTRIBUTIONS[self.distribution.currentText()].fields)
        for name, field in self.gen_fields.items():
            field.set_error(gen_errors.get(name) if generation and name in active else None)

        n_points = self._remaining_rows()
        params_valid = True
        for card in self.cards:
            errors = card.algorithm.errors(n_points)
            card.show_errors(errors)
            if card.is_checked() and errors:
                params_valid = False

        x, y, labels = self.selected_columns()
        loaded = self.dataset is not None
        problem = self.preprocessor.readiness_problem(x, y, labels) if loaded and x and y else None
        cond = Conditions(
            dataset_loaded=loaded,
            dataset_problem=problem,
            columns_differ=bool(x and y and x != y),
            algorithm_selected=any(card.is_checked() for card in self.cards),
            params_valid=params_valid,
            has_results=self.results is not None,
            running=self.running,
            generation_mode=generation,
            generation_valid=not gen_errors,
        )
        self.conditions = cond
        self._apply(cond, generation, active)

    def _apply(self, cond: Conditions, generation: bool, active: set) -> None:
        inputs = cond.inputs_enabled
        self.radio_csv.setEnabled(inputs)
        self.radio_gen.setEnabled(inputs)
        self.path.setEnabled(inputs and not generation)
        self.browse.setEnabled(inputs and not generation)
        self.distribution.setEnabled(inputs and generation)
        for name, field in self.gen_fields.items():
            field.setEnabled(inputs and generation and name in active)
        self.generate.setEnabled(cond.can_generate)
        for combo in (self.x_combo, self.y_combo, self.label_combo):
            combo.setEnabled(inputs and cond.dataset_loaded)
        self.radio_standard.setEnabled(inputs)
        self.radio_minmax.setEnabled(inputs)
        for card in self.cards:
            card.setEnabled(inputs)
        self.load_button.setEnabled(cond.can_load_params)
        self.save_button.setEnabled(cond.can_save_params)
        self.run_button.setEnabled(cond.can_run)
        self.cancel_button.setEnabled(cond.can_cancel and not self._cancelling)
        self.export_button.setEnabled(cond.can_export)
        self.act_open.setEnabled(inputs)
        self.act_load.setEnabled(cond.can_load_params)
        self.act_save.setEnabled(cond.can_save_params)
        self.act_export.setEnabled(cond.can_export)

        error = None
        if cond.dataset_loaded and not cond.columns_differ:
            error = "X и Y должны быть разными столбцами"
        elif cond.dataset_problem and cond.dataset_loaded:
            error = cond.dataset_problem
        self.features_error.setText(error or "")
        self.features_error.setVisible(bool(error))

        if self.running:
            return
        if self._flash_text:
            self.status.setText(self._flash_text)
        else:
            hint = cond.hint()
            self.status.setText(f"<b>{cond.state}</b>" + (f": {hint}" if hint else ""))

    # --- данные -------------------------------------------------------------------

    def browse_csv(self) -> None:
        """«Обзор…» и пункт меню «Открыть файл CSV…»."""
        start = str(Path(self.path.text()).parent) if self.path.text() else ""
        path, _ = QFileDialog.getOpenFileName(self, "Открыть файл CSV", start,
                                              "Файлы CSV (*.csv);;Все файлы (*)")
        if path:
            self.load_csv(path)

    def load_csv(self, path: str) -> bool:
        """Загружает файл; при ошибке прежний набор остаётся (решение 8)."""
        loader = CsvLoader(path)
        try:
            dataset = loader.get_data()
        except CC2DError as exc:
            log.error("Ошибка загрузки файла %s: %s", path, exc)
            self.warn("Ошибка файла", str(exc))
            return False
        log.info("Загружен файл %s (%s): %d строк", path, loader.encoding, dataset.row_count)
        self._updating = True
        self.radio_csv.setChecked(True)
        self.path.setText(str(Path(path)))
        self._updating = False
        self.set_dataset(dataset, loader)
        self.flash(f"Загружен файл {Path(path).name}: {dataset.row_count} строк")
        return True

    def generate_dataset(self) -> None:
        """«Сгенерировать набор»: новый seed и новый набор (решение 12.16)."""
        generator = self.generator_from_fields()
        try:
            generator.validate()
            seed = new_seed()
            dataset = generator.get_data(seed)
        except CC2DError as exc:
            self.warn("Ошибка генерации", str(exc))
            return
        self.experiment.seed = seed
        log.info("Сгенерирован набор: %s, seed=%d", generator.params(), seed)
        self.set_dataset(dataset, generator)
        self.flash(f"Сгенерирован набор из {dataset.row_count} точек")

    def set_dataset(self, dataset: Dataset, source: DataSource,
                    columns: tuple | None = None, warn: bool = True) -> None:
        """Делает набор текущим и подставляет столбцы X, Y и меток."""
        self.dataset = dataset
        self.preprocessor.dataset = dataset
        self.experiment.source = source
        x, y, labels = columns or dataset.default_columns()
        self._updating = True
        self.x_combo.clear()
        self.y_combo.clear()
        self.label_combo.clear()
        self.x_combo.addItems(dataset.numeric_columns)
        self.y_combo.addItems(dataset.numeric_columns)
        self.label_combo.addItems([NO_LABELS] + dataset.columns)
        self.x_combo.setCurrentText(x)
        self.y_combo.setCurrentText(y)
        self.label_combo.setCurrentText(labels or NO_LABELS)
        self._updating = False
        self.table_model.set_frame(dataset.get_rows(PREVIEW_ROWS))
        shown = min(PREVIEW_ROWS, dataset.row_count)
        self.data_box.setTitle(f"Данные (первые {shown} строк из {dataset.row_count})")
        self.data_stack.setCurrentIndex(1)
        self._on_columns_changed(warn=warn)

    def _on_columns_changed(self, *_, warn: bool = True) -> None:
        if self._updating or self.dataset is None:
            return
        x, y, labels = self.selected_columns()
        self.experiment.x_column, self.experiment.y_column = x, y
        self.experiment.label_column = labels
        self.table_model.set_highlight(x, y, labels)
        excluded = self.preprocessor.count_excluded(x, y, labels) if x and y else 0
        self.excluded.setText(f"Будет исключено строк: {excluded}")
        self.refresh()
        if warn and excluded:
            self.inform("Исключение строк",
                        "В выбранных столбцах есть пропуски или нечисловые значения.\n"
                        f"При расчёте будет исключено строк: {excluded}.")

    def _on_preprocessing_changed(self, *_) -> None:
        self.experiment.preprocessing = STANDARD if self.radio_standard.isChecked() else MINMAX
        self.refresh()

    def _on_algorithms_changed(self) -> None:
        self.experiment.algorithms = [c.algorithm for c in self.cards if c.is_checked()]
        self.refresh()

    # --- параметры эксперимента -------------------------------------------------

    def save_params(self) -> None:
        """«Сохранить параметры…»: записывает текущие значения полей."""
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить параметры эксперимента",
                                              str(Path(self.folder) / EXPERIMENT_FILE),
                                              "Параметры эксперимента (*.json)")
        if path:
            self.save_params_to(path)

    def save_params_to(self, path: str) -> bool:
        try:
            self.experiment.save(path)
        except CC2DError as exc:
            self.warn("Параметры не сохранены", str(exc))
            return False
        log.info("Параметры эксперимента сохранены: %s", path)
        self.flash(f"Параметры сохранены в {Path(path).name}")
        return True

    def load_params(self) -> None:
        """«Загрузить параметры…»."""
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить параметры эксперимента",
                                              self.folder, "Параметры эксперимента (*.json)")
        if path:
            self.load_params_from(path)

    def load_params_from(self, path: str) -> bool:
        """Читает файл и восстанавливает источник, столбцы, предобработку и алгоритмы.

        Если файл некорректен, текущие данные и параметры не меняются.
        """
        loaded = Experiment()
        try:
            loaded.load(path)
            dataset = loaded.restore_dataset()
        except CC2DError as exc:
            log.error("Ошибка загрузки параметров %s: %s", path, exc)
            self.warn("Параметры не загружены", str(exc))
            return False

        self._updating = True
        if isinstance(loaded.source, CsvLoader):
            self.radio_csv.setChecked(True)
            self.path.setText(loaded.source.path)
        else:
            self.radio_gen.setChecked(True)
            self.distribution.setCurrentText(loaded.source.distribution)
            for name, field in self.gen_fields.items():
                field.set_text(value_text(getattr(loaded.source, name)))
        if loaded.preprocessing == MINMAX:
            self.radio_minmax.setChecked(True)
        else:
            self.radio_standard.setChecked(True)
        by_key = {a.key: a for a in loaded.algorithms}
        for card in self.cards:
            if card.algorithm.key in by_key:
                card.algorithm.params = dict(by_key[card.algorithm.key].params)
                card.load_from_algorithm()
            card.set_checked(card.algorithm.key in by_key)
        self._updating = False

        self.experiment.seed = loaded.seed
        self.experiment.preprocessing = loaded.preprocessing
        self.experiment.algorithms = [c.algorithm for c in self.cards if c.is_checked()]
        self.set_dataset(dataset, loaded.source,
                         (loaded.x_column, loaded.y_column, loaded.label_column), warn=False)
        log.info("Загружены параметры эксперимента: %s", path)
        self.flash(f"Параметры загружены из {Path(path).name}")
        return True

    # --- расчёт ---------------------------------------------------------------

    def start_run(self) -> None:
        """«Запустить»: расчёт в фоновом потоке (сообщение 1 на рис. 2.6)."""
        if not self.conditions.can_run:
            return
        from .worker import start_worker
        self.running = True
        self._cancelling = False
        self._run_dataset = self.dataset
        self.manager.prepare()
        self._thread, self._worker = start_worker(self.manager, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("<b>Идёт расчёт</b>")
        self.refresh()
        self._thread.start()

    def cancel_run(self) -> None:
        """«Отменить»: прежние результаты сохраняются (решение 4)."""
        if self.running:
            self._cancelling = True
            self.manager.cancel()
            self.status.setText("<b>Идёт расчёт</b>: отмена…")
            self.refresh()

    def _on_progress(self, percent: int, text: str) -> None:
        self.progress.setValue(percent)
        if not self._cancelling:
            self.status.setText(f"<b>Идёт расчёт</b>: {text}")

    def _end_run(self) -> None:
        self.running = False
        self._cancelling = False
        self.progress.hide()
        if self._thread is not None:
            self._thread.wait()
            self._thread.deleteLater()
        self._thread = self._worker = None

    def _on_finished(self, output: RunOutput) -> None:
        self._end_run()
        self.results = RunResults(output.results, output.data, output.table, self._run_dataset)
        self.show_compare()
        self.flash("Расчёт завершён")
        if output.failures:
            details = "\n".join(f"{name}: {exc}" for name, exc in output.failures)
            self.warn("Ошибка алгоритма",
                      "Не все алгоритмы выполнены, остальные результаты показаны.\n\n" + details)

    def _on_failed(self, exc: Exception) -> None:
        self._end_run()
        if isinstance(exc, AllAlgorithmsFailedError):
            details = "\n".join(f"{name}: {err}" for name, err in exc.failures)
            self.warn("Расчёт не выполнен", f"{exc}.\n\n{details}")
        else:
            self.warn("Расчёт не выполнен", str(exc))
        self.refresh()

    def _on_cancelled(self) -> None:
        self._end_run()
        self.flash("Расчёт отменён")

    # --- другие окна -------------------------------------------------------------

    def show_compare(self) -> None:
        """Открывает окно сравнения с последними результатами."""
        if self.results is None:
            return
        if self.compare_window is None:
            self.compare_window = CompareWindow(self.visualizer)
            self.compare_window.export_requested.connect(
                lambda: self.open_export(self.compare_window))
            self.compare_window.main_requested.connect(self.show_main)
        run = self.results
        self.compare_window.show_results(run.results, run.data.points, run.table)
        self.compare_window.showNormal()
        self.compare_window.raise_()
        self.compare_window.activateWindow()

    def show_main(self) -> None:
        """Кнопка «В главное окно»: окно сравнения не закрывается."""
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def open_export(self, parent) -> None:
        """Окно экспорта; после закрытия возвращает в вызвавшее окно."""
        if self.results is None or self.running:
            return
        dialog = ExportDialog(self.results, self.folder, self.exporter, self.visualizer, parent)
        dialog.exec()
        self.folder = dialog.folder.text().strip() or self.folder

    def open_help(self) -> None:
        """Окно справки (доступно и во время расчёта)."""
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()

    def closeEvent(self, event) -> None:
        """Закрытие главного окна завершает работу; идущий расчёт сначала останавливается."""
        if self.running and self._thread is not None:
            self.manager.cancel()
            self._thread.wait()
        if self.compare_window is not None:
            self.compare_window.allow_close = True
            self.compare_window.close()
        if self.help_window is not None:
            self.help_window.close()
        log.info("Работа программы завершена")
        event.accept()
