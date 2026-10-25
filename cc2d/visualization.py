"""Класс «Визуализатор»: диаграммы рассеяния результатов кластеризации.

Диаграммы строятся объектами Matplotlib без pyplot, поэтому их можно
встроить в окно Qt и сохранить в файл без графического окна.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .core.result import NOISE, ClusteringResult

#: Формы маркеров кластеров; шум обозначается крестиком.
_SHAPES = ("o", "s", "^", "D", "v", "p", "<", ">", "h", "*")
#: Заливки маркеров: кластеры различимы и при печати без цвета.
_FILLS = ("white", "black", "0.6")
_NOISE_STYLE = {"marker": "x", "color": "black", "linewidths": 0.8}
_SIZE = 22


def cluster_style(index: int) -> dict:
    """Форма и заливка маркера кластера с номером index (с нуля).

    Первые три кластера — белый круг, чёрный квадрат и серый треугольник, как
    на рис. 2.18. Сдвиг заливки на каждом круге форм делает сочетания
    неповторяющимися для 30 кластеров (у алгоритмов их не больше 20).
    """
    shape = _SHAPES[index % len(_SHAPES)]
    fill = _FILLS[(index // len(_SHAPES) + index) % len(_FILLS)]
    return {"marker": shape, "facecolors": fill, "edgecolors": "black", "linewidths": 0.6}


def grid_shape(count: int) -> tuple[int, int]:
    """Сетка диаграмм: 1 → 1×1, 2 → 1×2, 3–4 → 2×2 (решение 12.14)."""
    return {1: (1, 1), 2: (1, 2)}.get(count, (2, 2))


def plot_title(result: ClusteringResult) -> str:
    """Заголовок диаграммы, например «DBSCAN (кластеров: 3, шум: 18)»."""
    text = f"{result.algorithm} (кластеров: {result.n_clusters}"
    if result.key == "dbscan":
        text += f", шум: {result.n_noise}"
    return text + ")"


class Visualizer:
    """Визуализатор (рис. 2.5)."""

    def __init__(self):
        #: Вид диаграммы.
        self.view = "scatter"

    def build(self, results: list[ClusteringResult], points: np.ndarray,
              figure: Figure | None = None) -> Figure:
        """Строит диаграммы всех результатов рядом с общими осями.

        :param figure: фигура для рисования (например, холста Qt); если не
            задана, создаётся новая
        """
        figure = figure or Figure()
        figure.clear()
        rows, cols = grid_shape(len(results))
        axes = figure.subplots(rows, cols, sharex=True, sharey=True, squeeze=False).ravel()
        for ax, result in zip(axes, results):
            self._draw(ax, result, points)
        for ax in axes[len(results):]:
            ax.set_visible(False)
        self._legend(figure, results)
        figure.tight_layout(rect=(0, 0.06, 1, 1))
        return figure

    def build_single(self, result: ClusteringResult, points: np.ndarray) -> Figure:
        """Строит отдельную диаграмму одного результата (для файла PNG)."""
        figure = Figure()
        ax = figure.subplots()
        self._draw(ax, result, points)
        self._legend(figure, [result])
        figure.tight_layout(rect=(0, 0.08, 1, 1))
        return figure

    @staticmethod
    def _draw(ax, result: ClusteringResult, points: np.ndarray) -> None:
        labels = result.get_labels()
        for cluster in range(labels.max() + 1 if labels.size else 0):
            mask = labels == cluster
            ax.scatter(points[mask, 0], points[mask, 1], s=_SIZE, **cluster_style(cluster))
        noise = labels == NOISE
        if noise.any():
            ax.scatter(points[noise, 0], points[noise, 1], s=_SIZE, **_NOISE_STYLE)
        ax.set_title(plot_title(result))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    @staticmethod
    def _legend(figure: Figure, results: list[ClusteringResult]) -> None:
        clusters = max((r.n_clusters for r in results), default=0)
        handles = []
        for i in range(clusters):
            style = cluster_style(i)
            handles.append(Line2D([], [], linestyle="none", marker=style["marker"],
                                  markerfacecolor=style["facecolors"],
                                  markeredgecolor="black", label=f"кластер {i + 1}"))
        if any(r.n_noise for r in results):
            handles.append(Line2D([], [], linestyle="none", marker="x", color="black",
                                  label="шум"))
        if handles:
            figure.legend(handles=handles, loc="lower center",
                          ncol=min(len(handles), 11), frameon=False, fontsize="small")
