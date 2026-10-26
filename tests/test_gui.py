"""Системные сценарии через интерфейс без экрана (тесты 1, 5, 52, 60–65, 84, 86, 88–90)."""

import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from cc2d.gui.main_window import NO_LABELS, MainWindow  # noqa: E402
from cc2d.gui.export_dialog import ExportDialog  # noqa: E402
from cc2d.gui.state import READY  # noqa: E402

from conftest import csv_text  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = MainWindow()
    w.messages = []
    w.inform = lambda title, text: w.messages.append(("info", text))
    w.warn = lambda title, text: w.messages.append(("warn", text))
    yield w
    w.close()


def wait(app, condition, timeout=60.0):
    start = time.time()
    while not condition():
        app.processEvents()
        time.sleep(0.005)
        if time.time() - start > timeout:
            raise TimeoutError
    app.processEvents()


def blob_csv(write_csv, n=200, name="data.csv"):
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [6, 0], [0, 6]])
    rows = []
    for i in range(n):
        x, y = centers[i % 3] + rng.normal(0, 0.7, 2)
        rows.append([f"{x:.4f}".replace(".", ","), f"{y:.4f}", f"класс {i % 3}"])
    return write_csv(csv_text(["длина", "ширина", "вид"], rows), name=name)


def check(window, *indexes):
    for i in indexes:
        window.cards[i].set_checked(True)


def run_and_wait(app, window):
    window.start_run()
    wait(app, lambda: not window.running)


def test_start_state(window):
    """Тест 54: программа только что запущена — «Запустить» недоступна."""
    c = window.conditions
    assert not c.can_run and c.can_load_params and not c.can_save_params
    assert not window.run_button.isEnabled() and window.load_button.isEnabled()
    assert not any(card.is_checked() for card in window.cards)


def test_load_csv_auto_columns(window, write_csv):
    """Тест 4 (часть): после загрузки X и Y выбраны автоматически, метки не выбраны."""
    assert window.load_csv(str(blob_csv(write_csv)))
    assert window.selected_columns() == ("длина", "ширина", None)
    assert window.label_combo.currentText() == NO_LABELS
    assert window.table_model.rowCount() == 100
    assert "из 200" in window.data_box.title()
    assert window.save_button.isEnabled() and not window.run_button.isEnabled()


def test_same_column_blocks_run(window, write_csv):
    """Тест 55: X и Y — один столбец, рядом выводится причина."""
    window.load_csv(str(blob_csv(write_csv)))
    check(window, 0)
    assert window.run_button.isEnabled()
    window.y_combo.setCurrentText("длина")
    assert not window.run_button.isEnabled()
    assert "разными" in window.features_error.text()


def test_invalid_param_blocks_run(window, write_csv):
    """Тесты 57, 58: K-Means с 21 кластером → недоступна, с 3 → доступна."""
    window.load_csv(str(blob_csv(write_csv)))
    check(window, 0)
    card = window.cards[0]
    card.fields["n_clusters"].setText("21")
    card._on_edit()
    assert not window.run_button.isEnabled()
    assert "от 2 до 20" in card.error.text()
    card.reset()
    assert window.run_button.isEnabled() and window.conditions.state == READY


def test_excluded_rows_warning(window, write_csv):
    """Тест 5: предупреждение об исключённых строках."""
    rows = [[str(i), str(i * 2)] for i in range(100)]
    rows[3][0] = rows[7][0] = rows[9][0] = ""
    rows[20][1] = rows[30][1] = "abc"
    window.load_csv(str(write_csv(csv_text(["x", "y"], rows))))
    assert window.excluded.text() == "Будет исключено строк: 5"
    assert any("исключено строк: 5" in text for kind, text in window.messages)


def test_bad_file_keeps_dataset(window, write_csv):
    """Тест 90: некорректный файл не сбрасывает загруженный набор."""
    window.load_csv(str(blob_csv(write_csv)))
    bad = write_csv("a,b\n" + "1,2\n" * 20, name="bad.csv")
    assert not window.load_csv(str(bad))
    assert window.messages[-1][0] == "warn"
    assert window.selected_columns() == ("длина", "ширина", None)
    assert window.table_model.rowCount() == 100


def test_generation_selects_labels(window):
    """Тест 4 и решение 12.16: сгенерированный набор, метки генератора выбраны, новый seed."""
    window.radio_gen.setChecked(True)
    seed = window.experiment.seed
    window.generate_dataset()
    assert window.selected_columns() == ("x", "y", "label")
    assert window.experiment.seed != seed or window.dataset.row_count == 500


def test_generate_button_conditions(window):
    """Тесты 63, 64: уровень шума 0,6 — кнопка недоступна, 0,05 — доступна."""
    assert not window.generate.isEnabled()
    window.radio_gen.setChecked(True)
    window.gen_fields["noise"].set_text("0,6")
    window.refresh()
    assert not window.generate.isEnabled()
    assert window.gen_fields["noise"].error.isVisibleTo(window)
    window.gen_fields["noise"].set_text("0,05")
    window.refresh()
    assert window.generate.isEnabled()


def test_full_run_and_results_kept(app, window, write_csv):
    """Тесты 1, 2, 62: расчёт, окно сравнения; после правки полей результаты остаются."""
    window.load_csv(str(blob_csv(write_csv)))
    window.label_combo.setCurrentText("вид")
    check(window, 0, 1, 2, 3)
    run_and_wait(app, window)
    assert window.results is not None and len(window.results.results) == 4
    assert window.compare_window is not None and window.compare_window.isVisible()
    assert window.compare_window.table.rowCount() == 4
    assert window.export_button.isEnabled()
    window.cards[0].fields["n_clusters"].setText("5")
    window.cards[0]._on_edit()
    window.y_combo.setCurrentText("длина")
    assert window.export_button.isEnabled() and window.results is not None


def test_export_json_has_run_params(app, window, write_csv, tmp_path):
    """Тест 88: JSON при экспорте содержит параметры выполненного расчёта."""
    window.load_csv(str(blob_csv(write_csv)))
    check(window, 0)
    run_and_wait(app, window)
    window.cards[0].fields["n_clusters"].setText("5")
    window.cards[0]._on_edit()
    dialog = ExportDialog(window.results, str(tmp_path / "out"), parent=window)
    assert not dialog.dataset_check.isEnabled()          # тест 65: С5 = 0 для файла
    assert dialog.save_files() == 4                      # таблица, метки, PNG, JSON
    data = json.loads((tmp_path / "out" / "experiment.json").read_text(encoding="utf-8"))
    assert data["algorithms"][0]["params"]["n_clusters"] == 3
    dialog._set_all(False)
    assert not dialog.save.isEnabled()                   # тест 52


def test_export_generated_dataset_enabled(app, window):
    """Тест 63: после расчёта на сгенерированном наборе флажок набора доступен (С5 = 1)."""
    window.radio_gen.setChecked(True)
    window.generate_dataset()
    check(window, 2)
    run_and_wait(app, window)
    dialog = ExportDialog(window.results, "", parent=window)
    assert dialog.dataset_check.isEnabled()


def test_cancel_without_results(app, window):
    """Тест 60: отмена при отсутствии результатов — результатов нет, «Готово к запуску»."""
    window.radio_gen.setChecked(True)
    window.gen_fields["n_points"].set_text("5000")
    window.refresh()
    window.generate_dataset()
    check(window, 1, 3)
    window.start_run()
    assert window.cancel_button.isEnabled() and not window.export_button.isEnabled()
    assert not window.load_button.isEnabled() and not window.cards[0].isEnabled()
    assert window.help_button.isEnabled()
    window.cancel_run()
    wait(app, lambda: not window.running)
    assert window.results is None
    assert window.conditions.state == READY


def test_cancel_keeps_previous_results(app, window):
    """Тест 61: отмена нового расчёта сохраняет прежние результаты."""
    window.radio_gen.setChecked(True)
    window.gen_fields["n_points"].set_text("5000")
    window.refresh()
    window.generate_dataset()
    check(window, 0)
    run_and_wait(app, window)
    previous = window.results
    check(window, 1, 3)
    window.start_run()
    window.cancel_run()
    wait(app, lambda: not window.running)
    assert window.results is previous and window.export_button.isEnabled()


def test_rerun_replaces_results(app, window, write_csv):
    """Тест 84: повторный запуск полностью заменяет прежние результаты."""
    window.load_csv(str(blob_csv(write_csv)))
    check(window, 0, 2)
    run_and_wait(app, window)
    first = window.results
    window.cards[2].set_checked(False)
    run_and_wait(app, window)
    assert window.results is not first and len(window.results.results) == 1
    assert window.compare_window.table.rowCount() == 1


def test_save_and_load_params(app, window, write_csv, tmp_path):
    """Тесты 6, 89: параметры сохраняются и загружаются сразу после запуска программы."""
    path = blob_csv(write_csv)
    window.load_csv(str(path))
    window.label_combo.setCurrentText("вид")
    window.radio_minmax.setChecked(True)
    check(window, 0, 3)
    window.cards[0].fields["max_iter"].setText("77")
    window.cards[0]._on_edit()
    params = tmp_path / "experiment.json"
    assert window.save_params_to(str(params))

    fresh = MainWindow()
    fresh.inform = fresh.warn = lambda *a: None
    assert fresh.load_params_from(str(params))
    assert fresh.selected_columns() == ("длина", "ширина", "вид")
    assert fresh.radio_minmax.isChecked()
    assert [c.is_checked() for c in fresh.cards] == [True, False, False, True]
    assert fresh.cards[0].fields["max_iter"].text() == "77"
    assert fresh.experiment.seed == window.experiment.seed
    assert fresh.run_button.isEnabled()
    fresh.close()


def test_bad_params_file_keeps_state(window, write_csv, tmp_path):
    """Тест 49: файл параметров с нарушенной структурой — ничего не меняется."""
    window.load_csv(str(blob_csv(write_csv)))
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert not window.load_params_from(str(bad))
    assert window.selected_columns() == ("длина", "ширина", None)


def test_compare_window_close_minimizes(app, window, write_csv):
    """Крестик окна сравнения только сворачивает его."""
    window.load_csv(str(blob_csv(write_csv)))
    check(window, 0)
    run_and_wait(app, window)
    compare = window.compare_window
    compare.close()
    app.processEvents()
    assert compare.isVisible()
    canvas = compare.canvas
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()
    ax.set_xlim(0, 0.1)
    canvas.reset_view()
    assert ax.get_xlim() == home


def test_close_during_run(app, window):
    """Тест 86: закрытие главного окна во время расчёта останавливает расчёт."""
    window.radio_gen.setChecked(True)
    window.gen_fields["n_points"].set_text("5000")
    window.refresh()
    window.generate_dataset()
    check(window, 1, 3)
    window.start_run()
    thread = window._thread
    window.close()
    assert thread.isFinished()


def test_help_available_while_running(app, window):
    window.radio_gen.setChecked(True)
    window.generate_dataset()
    check(window, 3)
    window.start_run()
    assert window.help_button.isEnabled()
    window.open_help()
    assert window.help_window.isVisible()
    window.cancel_run()
    wait(app, lambda: not window.running)
