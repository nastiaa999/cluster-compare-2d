"""Алгоритмы кластеризации, реализованные вручную на NumPy."""

from .agglomerative import Agglomerative
from .base import ClusteringAlgorithm
from .dbscan import DBSCAN
from .kmeans import KMeans
from .meanshift import MeanShift

#: Все алгоритмы в порядке показа на форме и в сводной таблице.
ALGORITHM_CLASSES = (KMeans, Agglomerative, DBSCAN, MeanShift)
#: Классы алгоритмов по короткому имени.
BY_KEY = {cls.key: cls for cls in ALGORITHM_CLASSES}

__all__ = ["ClusteringAlgorithm", "KMeans", "Agglomerative", "DBSCAN", "MeanShift",
           "ALGORITHM_CLASSES", "BY_KEY"]
