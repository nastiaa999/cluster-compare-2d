"""Окно экспорта результатов (рис. 2.19 РПЗ)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout)

from ..core.dataset import Dataset
from ..core.errors import CC2DError
from ..core.exporter import (DATASET_FILE, DEFAULT_PLOT_SIZE, EXPERIMENT_FILE, PLOT_SIZES,
                             SUMMARY_FILE, Exporter, labels_file, plot_file)
from ..core.preprocessing import PreparedData
from ..core.result import ClusteringResult
from ..core.summary import SummaryTable
from ..visualization import Visualizer

log = logging.getLogger(__name__)


@dataclass
class RunResults:
    """Результаты расчёта, которые хранит главное окно до следующего запуска.

    :param dataset: набор, на котором выполнен расчёт (нужен для экспорта
        сгенерированного набора, даже если потом загружен другой)
    """

    results: list[ClusteringResult]
    data: PreparedData
    table: SummaryTable
    dataset: Dataset


class ExportDialog(QDialog):
    """Окно экспорта: выбор файлов, папка и список создаваемых файлов.

    «Сохранить» записывает файлы и оставляет окно открытым, «Закрыть»
    возвращает в окно, из которого вызван экспорт.

    :param folder: папка, предложенная по умолчанию
    """

    def __init__(self, run: RunResults, folder: str, exporter: Exporter | None = None,
                 visualizer: Visualizer | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт результатов")
        self.setMinimumWidth(640)
        self.run = run
        self.exporter = exporter or Exporter()
        self.visualizer = visualizer or Visualizer()

        self.summary = QCheckBox("Сводная таблица метрик (CSV)")
        self.labels = QCheckBox("Метки кластеров (CSV, отдельный файл для каждого алгоритма)")
        self.label_checks: dict[str, QCheckBox] = {}
        label_row = QHBoxLayout()
        label_row.setContentsMargins(24, 0, 0, 0)
        for result in run.results:
            check = QCheckBox(result.algorithm)
            check.setChecked(True)
            self.label_checks[result.key] = check
            label_row.addWidget(check)
        label_row.addStretch()

        self.plots = QCheckBox("Диаграммы (PNG, отдельный файл для каждого алгоритма)")
        self.size = QComboBox()
        for w, h in PLOT_SIZES:
            self.size.addItem(f"{w} × {h}", (w, h))
        self.size.setCurrentIndex(PLOT_SIZES.index(DEFAULT_PLOT_SIZE))
        size_row = QHBoxLayout()
        size_row.setContentsMargins(24, 0, 0, 0)
        size_row.addWidget(QLabel("Размер изображения"))
        size_row.addWidget(self.size)
        size_hint = QLabel("не меньше 1200 × 800")
        size_hint.setObjectName("Muted")
        size_row.addWidget(size_hint)
        size_row.addStretch()

        self.dataset_check = QCheckBox("Сгенерированный набор данных (CSV)")
        dataset_note = QLabel("недоступно: данные загружены из файла")
        dataset_note.setObjectName("Muted")
        dataset_note.setContentsMargins(24, 0, 0, 0)
        generated = run.dataset.is_generated
        self.dataset_check.setEnabled(generated)
        dataset_note.setVisible(not generated)

        self.params = QCheckBox("Параметры эксперимента (JSON, вместе с seed)")
        params_note = QLabel("позволяют повторить эксперимент с теми же результатами")
        params_note.setObjectName("Muted")
        params_note.setContentsMargins(24, 0, 0, 0)

        select_all = QPushButton("Выбрать всё")
        clear_all = QPushButton("Снять выбор")
        select_row = QHBoxLayout()
        select_row.addWidget(select_all)
        select_row.addWidget(clear_all)
        select_row.addStretch()

        what = QGroupBox("Что сохранить")
        what_layout = QVBoxLayout(what)
        what_layout.addWidget(self.summary)
        what_layout.addWidget(self.labels)
        what_layout.addLayout(label_row)
        what_layout.addWidget(self.plots)
        what_layout.addLayout(size_row)
        what_layout.addWidget(self.dataset_check)
        what_layout.addWidget(dataset_note)
        what_layout.addWidget(self.params)
        what_layout.addWidget(params_note)
        what_layout.addLayout(select_row)

        self.folder = QLineEdit(folder)
        browse = QPushButton("Обзор…")
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Папка"))
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(browse)
        self.files = QLabel()
        self.files.setObjectName("Muted")
        self.files.setWordWrap(True)
        where = QGroupBox("Куда сохранить")
        where_layout = QVBoxLayout(where)
        where_layout.addLayout(folder_row)
        where_layout.addWidget(QLabel("Будут созданы файлы:"))
        where_layout.addWidget(self.files)

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.save = QPushButton("Сохранить")
        self.save.setObjectName("Primary")
        close = QPushButton("Закрыть")
        bottom = QHBoxLayout()
        bottom.addWidget(self.result_label, 1)
        bottom.addWidget(self.save)
        bottom.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addWidget(what)
        layout.addWidget(where)
        layout.addLayout(bottom)

        for check in (self.summary, self.labels, self.plots, self.dataset_check, self.params,
                      *self.label_checks.values()):
            check.toggled.connect(self._update)
        self.folder.textChanged.connect(self._update)
        select_all.clicked.connect(lambda: self._set_all(True))
        clear_all.clicked.connect(lambda: self._set_all(False))
        browse.clicked.connect(self._browse)
        self.save.clicked.connect(self.save_files)
        close.clicked.connect(self.reject)
        self._set_all(True)

    def _set_all(self, checked: bool) -> None:
        for check in (self.summary, self.labels, self.plots, self.params,
                      *self.label_checks.values()):
            check.setChecked(checked)
        if self.dataset_check.isEnabled():
            self.dataset_check.setChecked(checked)
        self._update()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Папка для сохранения", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def planned(self) -> list[tuple[str, Callable[[Path], None]]]:
        """Файлы, которые будут созданы: (имя файла, функция записи по пути)."""
        run, exporter = self.run, self.exporter
        plan = []
        if self.summary.isChecked():
            plan.append((SUMMARY_FILE, lambda p: exporter.export_table(run.table, p)))
        for result in run.results:
            if self.labels.isChecked() and self.label_checks[result.key].isChecked():
                plan.append((labels_file(result.key),
                             lambda p, r=result: exporter.export_result(r, run.data, p)))
        if self.plots.isChecked():
            size = self.size.currentData()
            for result in run.results:
                plan.append((plot_file(result.key), lambda p, r=result: exporter.save_figure(
                    self.visualizer.build_single(r, run.data.points), p, size)))
        if self.dataset_check.isEnabled() and self.dataset_check.isChecked():
            plan.append((DATASET_FILE, lambda p: exporter.export_dataset(run.dataset.get_all(), p)))
        if self.params.isChecked() and run.table.get_params() is not None:
            plan.append((EXPERIMENT_FILE, lambda p: run.table.get_params().save(p)))
        return plan

    def _update(self, *_) -> None:
        for check in self.label_checks.values():
            check.setEnabled(self.labels.isChecked())
        self.size.setEnabled(self.plots.isChecked())
        folder = Path(self.folder.text().strip()) if self.folder.text().strip() else None
        names = []
        for name, _ in self.planned():
            exists = folder is not None and (folder / name).exists()
            names.append(f"{name} (будет заменён)" if exists else name)
        self.files.setText(", ".join(names) if names else "ничего не отмечено")
        self.save.setEnabled(bool(names) and folder is not None)

    def save_files(self) -> int:
        """Записывает отмеченные файлы; возвращает число сохранённых файлов."""
        folder = Path(self.folder.text().strip())
        errors, saved = [], 0
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Не удалось создать папку: {exc.strerror or exc}")
        else:
            for name, write in self.planned():
                try:
                    write(folder / name)
                    saved += 1
                except CC2DError as exc:
                    errors.append(str(exc))
        if errors:
            log.error("Ошибка экспорта в %s: %s", folder, "; ".join(errors))
            self.result_label.setStyleSheet("color: #c62828;")
            self.result_label.setText("Ошибка записи. " + " ".join(errors))
        else:
            log.info("Экспорт: %d файлов в %s", saved, folder)
            self.result_label.setStyleSheet("color: #2e7d32;")
            self.result_label.setText(f"Сохранено файлов: {saved}")
        self._update()
        return saved
