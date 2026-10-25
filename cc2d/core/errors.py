"""Исключения ядра CC2D.

Текст каждого исключения предназначен для пользователя: интерфейс выводит его
в сообщении или рядом с полем ввода без изменений (п. 4.2.1 ТЗ).
"""


class CC2DError(Exception):
    """Базовое исключение программы."""


class DataFormatError(CC2DError):
    """Файл данных не соответствует формату или структуре (п. 4.5.2.1 ТЗ)."""


class DataPreparationError(CC2DError):
    """Выбранные данные нельзя подготовить к расчёту.

    Например, после исключения строк осталось меньше 10 строк или признак
    постоянен.
    """


class ParameterError(CC2DError):
    """Значение параметра имеет неверный тип или выходит за допустимый диапазон."""


class ExperimentFileError(CC2DError):
    """Файл параметров эксперимента нельзя прочитать или он некорректен."""


class ExportError(CC2DError):
    """Ошибка записи файла экспорта."""


class CancelledError(CC2DError):
    """Расчёт отменён пользователем."""

    def __init__(self, message: str = "Расчёт отменён пользователем"):
        super().__init__(message)


class AllAlgorithmsFailedError(CC2DError):
    """Ни один из отмеченных алгоритмов не выполнился.

    :param failures: список пар (название алгоритма, исключение)
    """

    def __init__(self, failures):
        self.failures = list(failures)
        names = ", ".join(name for name, _ in self.failures)
        super().__init__(f"Не выполнен ни один алгоритм: {names}")
