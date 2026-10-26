"""Таблицы истинности доступности элементов (тесты 54–59, 63–65; рис. 2.17)."""

import pytest

from cc2d.gui.state import DATA_LOADED, NO_DATA, READY, RUNNING, Conditions

READY_ALL = dict(dataset_loaded=True, columns_differ=True, algorithm_selected=True,
                 params_valid=True)


@pytest.mark.parametrize("changes, expected", [
    (dict(dataset_loaded=False), False),                        # тест 54: П1 = 0
    (dict(dataset_problem="постоянен"), False),                 # П1 = 0 (решение 12.9)
    (dict(columns_differ=False), False),                        # тест 55: П2 = 0
    (dict(algorithm_selected=False), False),                    # тест 56: П3 = 0
    (dict(params_valid=False), False),                          # тест 57: П4 = 0
    ({}, True),                                                 # тест 58: все условия
])
def test_run_button(changes, expected):
    """Таблица 3.1: С1 = П1 ∧ П2 ∧ П3 ∧ П4."""
    assert Conditions(**{**READY_ALL, **changes}).can_run is expected


def test_nothing_without_results():
    """Тест 59: П5 = 0, П6 = 0, П7 = 0 → С2 = С3 = С4 = 0."""
    c = Conditions(**READY_ALL)
    assert not c.can_cancel and not c.can_export and not c.can_generate


def test_running_blocks_everything_but_cancel():
    """Тест 60: во время расчёта доступны только «Отменить» (и «Справка»)."""
    c = Conditions(**READY_ALL, has_results=True, running=True, generation_mode=True)
    assert c.can_cancel
    assert not (c.can_run or c.can_export or c.can_generate or c.can_load_params
                or c.can_save_params or c.inputs_enabled)
    assert c.state == RUNNING


def test_results_kept_export_available():
    """Тест 62: П5 = 1, П6 = 0 → С1 = 1, С3 = 1."""
    c = Conditions(**READY_ALL, has_results=True)
    assert c.can_run and c.can_export


def test_generate_button():
    """Тесты 63, 64: С4 = П7 ∧ П8."""
    assert not Conditions(generation_mode=True, generation_valid=False).can_generate
    assert Conditions(generation_mode=True, generation_valid=True).can_generate
    assert not Conditions(generation_mode=False).can_generate


def test_params_buttons():
    """Решение 7: «Загрузить параметры…» всегда, кроме расчёта; «Сохранить…» при наборе."""
    assert Conditions().can_load_params and not Conditions().can_save_params
    assert Conditions(dataset_loaded=True).can_save_params


def test_states_and_hints():
    assert Conditions().state == NO_DATA
    assert "откройте файл" in Conditions().hint()
    loaded = Conditions(dataset_loaded=True, columns_differ=True)
    assert loaded.state == DATA_LOADED and "алгоритм" in loaded.hint()
    assert Conditions(**READY_ALL).state == READY and Conditions(**READY_ALL).hint() == ""
