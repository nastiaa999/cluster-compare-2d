"""Класс «Оценщик качества»: метрики и замечание о нерепрезентативности."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)

from .result import NOISE, ClusteringResult

SILHOUETTE = "silhouette"
DAVIES_BOULDIN = "davies_bouldin"
CALINSKI_HARABASZ = "calinski_harabasz"
ARI = "ari"
NMI = "nmi"

#: Внутренние метрики: эталонные метки не нужны.
INTERNAL_METRICS = (SILHOUETTE, DAVIES_BOULDIN, CALINSKI_HARABASZ)
#: Внешние метрики: вычисляются, только если выбран столбец меток.
EXTERNAL_METRICS = (ARI, NMI)

#: Ниже этого значения силуэта устойчивой кластерной структуры нет
#: (шкала Кауфмана и Руссеу).
WEAK_SILHOUETTE = 0.25
#: Доля точек шума, выше которой выводится замечание.
NOISE_SHARE = 0.5

MSG_ONE_CLUSTER = "Найден один кластер"
MSG_ALL_NOISE = "Все точки отнесены к шуму"
MSG_EACH_POINT = "Каждая точка образует отдельный кластер"
MSG_WEAK = "Кластеры выражены слабо (силуэт меньше 0,25)"
MSG_NOISE = "Больше половины точек отнесено к шуму"


class QualityEvaluator:
    """Оценщик качества (рис. 2.5).

    Внутренние метрики вычисляются только при числе кластеров от 2 до n − 1,
    где n — число точек без шума; точки шума при этом исключаются. Внешние
    метрики вычисляются по всем точкам, шум считается отдельной группой.
    Неопределённая внутренняя метрика записывается как None, а внешние метрики
    без эталонных меток в словарь не попадают.

    Замечание составляется из сработавших правил в таком порядке: внутренние
    метрики не определены; силуэт меньше 0,25; больше половины точек — шум.
    Если шумом оказались все точки, выводится только первая фраза.
    """

    def __init__(self):
        self.metrics = INTERNAL_METRICS + EXTERNAL_METRICS

    def evaluate(self, result: ClusteringResult, points: np.ndarray,
                 reference_labels: np.ndarray | None = None) -> None:
        """Вычисляет метрики и замечание и записывает их в результат."""
        labels = result.get_labels()
        valid = labels != NOISE
        n_valid = int(valid.sum())
        k = result.n_clusters
        metrics: dict[str, float | None] = dict.fromkeys(INTERNAL_METRICS)
        notes = []

        defined = 2 <= k <= n_valid - 1
        if defined:
            pts, lab = points[valid], labels[valid]
            metrics[SILHOUETTE] = float(silhouette_score(pts, lab))
            metrics[DAVIES_BOULDIN] = float(davies_bouldin_score(pts, lab))
            metrics[CALINSKI_HARABASZ] = float(calinski_harabasz_score(pts, lab))
        elif n_valid == 0:
            notes.append(MSG_ALL_NOISE)
        elif k < 2:
            notes.append(MSG_ONE_CLUSTER)
        else:
            notes.append(MSG_EACH_POINT)

        if reference_labels is not None:
            metrics[ARI] = float(adjusted_rand_score(reference_labels, labels))
            metrics[NMI] = float(normalized_mutual_info_score(reference_labels, labels))

        if defined and metrics[SILHOUETTE] < WEAK_SILHOUETTE:
            notes.append(MSG_WEAK)
        if n_valid > 0 and result.n_noise > NOISE_SHARE * result.n_points:
            notes.append(MSG_NOISE)

        result.set_metrics(metrics)
        result.set_message("; ".join(notes))
