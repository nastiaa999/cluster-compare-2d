"""Доступность элементов главного окна (диаграмма состояний, рис. 2.17 РПЗ).

Модуль не зависит от Qt: главное окно собирает текущие условия в объект
:class:`Conditions`, а тот вычисляет, какие кнопки доступны и в каком
состоянии находится окно. Обозначения условий совпадают с подразделом 3.1 и
таблицей Г.3 РПЗ.
"""

from __future__ import annotations

from dataclasses import dataclass

NO_DATA = "Нет данных"
DATA_LOADED = "Данные загружены"
READY = "Готово к запуску"
RUNNING = "Идёт расчёт"


@dataclass(frozen=True)
class Conditions:
    """Условия, от которых зависит доступность элементов окна.

    :param dataset_loaded: набор данных загружен или сгенерирован
    :param dataset_problem: причина, по которой набор не подготовлен, или None
        (вместе с dataset_loaded даёт условие П1)
    :param columns_differ: в качестве X и Y выбраны разные столбцы (П2)
    :param algorithm_selected: отмечен хотя бы один алгоритм (П3)
    :param params_valid: параметры всех отмеченных алгоритмов корректны (П4)
    :param has_results: есть результаты расчёта (П5)
    :param running: идёт расчёт (П6)
    :param generation_mode: выбран режим генерации (П7)
    :param generation_valid: параметры генерации корректны (П8)
    """

    dataset_loaded: bool = False
    dataset_problem: str | None = None
    columns_differ: bool = False
    algorithm_selected: bool = False
    params_valid: bool = True
    has_results: bool = False
    running: bool = False
    generation_mode: bool = False
    generation_valid: bool = True

    @property
    def dataset_ready(self) -> bool:
        """П1: набор подготовлен."""
        return self.dataset_loaded and self.dataset_problem is None

    @property
    def launch_conditions(self) -> bool:
        """Условия запуска выполнены: П1 ∧ П2 ∧ П3 ∧ П4."""
        return (self.dataset_ready and self.columns_differ
                and self.algorithm_selected and self.params_valid)

    @property
    def can_run(self) -> bool:
        """С1: кнопка «Запустить»."""
        return self.launch_conditions and not self.running

    @property
    def can_cancel(self) -> bool:
        """С2: кнопка «Отменить»."""
        return self.running

    @property
    def can_export(self) -> bool:
        """С3: кнопка «Экспорт…»."""
        return self.has_results and not self.running

    @property
    def can_generate(self) -> bool:
        """С4: кнопка «Сгенерировать набор»."""
        return self.generation_mode and self.generation_valid and not self.running

    @property
    def can_load_params(self) -> bool:
        """«Загрузить параметры…» доступна во всех состояниях, кроме расчёта."""
        return not self.running

    @property
    def can_save_params(self) -> bool:
        """«Сохранить параметры…» доступна, когда набор данных загружен."""
        return self.dataset_loaded and not self.running

    @property
    def inputs_enabled(self) -> bool:
        """Поля, флажки и списки окна доступны везде, кроме расчёта."""
        return not self.running

    @property
    def state(self) -> str:
        """Простое состояние окна по рис. 2.17."""
        if self.running:
            return RUNNING
        if not self.dataset_loaded:
            return NO_DATA
        return READY if self.launch_conditions else DATA_LOADED

    def hint(self) -> str:
        """Подсказка для строки состояния: что сделать, чтобы запустить расчёт."""
        if self.running or (self.dataset_loaded and self.launch_conditions):
            return ""
        if not self.dataset_loaded:
            return "откройте файл CSV или сгенерируйте набор"
        if self.dataset_problem:
            return self.dataset_problem
        if not self.columns_differ:
            return "выберите разные столбцы X и Y"
        if not self.algorithm_selected:
            return "отметьте хотя бы один алгоритм"
        return "исправьте параметры отмеченных алгоритмов"
