"""Окно справки: описание алгоритмов со ссылками на открытые источники (п. 4.1.1.6 ТЗ)."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

SKLEARN = "https://scikit-learn.org/stable/modules/clustering.html"

HELP_HTML = f"""
<h2>ClusterCompare 2D</h2>
<p>Программа применяет к одному набору двумерных данных до четырёх алгоритмов
кластеризации и сравнивает результаты на диаграммах и по метрикам качества.</p>

<h3>Порядок работы</h3>
<ol>
<li>Откройте файл CSV (поля разделены «;») или сгенерируйте синтетический набор.</li>
<li>Проверьте столбцы X и Y; если в файле есть эталонные классы, выберите столбец меток.</li>
<li>Отметьте алгоритмы и при необходимости измените их параметры.</li>
<li>Нажмите «Запустить». Результаты откроются в окне сравнения.</li>
</ol>

<h3>K-Means</h3>
<p>Разбивает точки на заданное число кластеров так, чтобы сумма квадратов
расстояний от точек до центров их кластеров была наименьшей. Алгоритм
поочерёдно приписывает точки к ближайшему центру и пересчитывает центры.
Хорошо выделяет компактные выпуклые кластеры близкого размера.
Параметры: число кластеров и максимальное число итераций.</p>
<p>Источники: <a href="{SKLEARN}#k-means">документация scikit-learn</a>;
Lloyd S. Least squares quantization in PCM,
<a href="https://doi.org/10.1109/TIT.1982.1056489">IEEE Trans. Inf. Theory, 1982</a>.</p>

<h3>Агломеративная кластеризация (связь Уорда)</h3>
<p>Начинает с кластеров из одной точки и на каждом шаге объединяет два
кластера, при объединении которых внутрикластерная дисперсия растёт меньше
всего, пока не останется заданное число кластеров. Не зависит от случайного
выбора. Параметры: число кластеров и тип связи.</p>
<p>Источники: <a href="{SKLEARN}#hierarchical-clustering">документация scikit-learn</a>;
Ward J.H. Hierarchical grouping to optimize an objective function,
<a href="https://doi.org/10.1080/01621459.1963.10500845">JASA, 1963</a>.</p>

<h3>DBSCAN</h3>
<p>Считает кластером область, где точки расположены плотно. Точка корневая,
если в круге заданного радиуса вокруг неё не меньше заданного числа точек.
Кластеры могут иметь любую форму, их число определяется по данным, а точки
вне плотных областей помечаются как шум (крестики на диаграмме).
Параметры: радиус окрестности и минимальное число точек.</p>
<p>Источники: <a href="{SKLEARN}#dbscan">документация scikit-learn</a>;
Ester M. et al. A density-based algorithm for discovering clusters in large
spatial databases with noise,
<a href="https://cdn.aaai.org/KDD/1996/KDD96-037.pdf">KDD-96</a>.</p>

<h3>Mean Shift</h3>
<p>Сдвигает каждую точку к ближайшему максимуму плотности, усредняя точки
внутри окна заданного размера. Точки, пришедшие к одному максимуму, образуют
кластер, число кластеров определяется автоматически. При значении «авто»
размер окна оценивается по данным.</p>
<p>Источники: <a href="{SKLEARN}#mean-shift">документация scikit-learn</a>;
Comaniciu D., Meer P. Mean shift: a robust approach toward feature space analysis,
<a href="https://doi.org/10.1109/34.1000236">IEEE TPAMI, 2002</a>.</p>

<h3>Метрики в сводной таблице</h3>
<ul>
<li><b>Силуэт</b> (от −1 до 1, больше — лучше): насколько точки ближе к своему кластеру, чем к соседнему.</li>
<li><b>Дэвис–Болдин</b> (от 0, меньше — лучше): отношение разброса внутри кластеров к расстоянию между ними.</li>
<li><b>Калински–Харабаш</b> (от 0, больше — лучше): отношение межкластерной дисперсии к внутрикластерной.</li>
<li><b>ARI</b> и <b>NMI</b> (до 1, больше — лучше): совпадение с эталонными метками;
вычисляются, только если выбран столбец меток.</li>
</ul>
<p>Лучшее значение в каждом столбце выделено. Прочерк означает, что метрика не
определена: найден один кластер, все точки отнесены к шуму или каждая точка
образует свой кластер. Подробнее: <a href="https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation">оценка качества кластеризации</a>.</p>
"""


class HelpWindow(QDialog):
    """Окно справки (немодальное, доступно и во время расчёта)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HelpWindow")
        self.setWindowTitle("Справка — ClusterCompare 2D")
        self.resize(720, 640)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(HELP_HTML)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(browser)
        layout.addWidget(buttons)
