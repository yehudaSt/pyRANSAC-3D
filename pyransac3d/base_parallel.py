from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from random import Random
from typing import List, Tuple

import numpy as np


class BaseParallelRansac(ABC):
    def __init__(self, seed=None, n_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.random = Random(seed)
        self.inliers = []
        self.equation = []

    def fit(
        self, points: np.ndarray, thresh: float = 0.05, max_iteration: int = 5000
    ) -> Tuple[List[int], List[int]]:
        """
        :param points: A numpy array of points, of shape (# points, 3)
        :param thresh: The distance threshold to include points as inliers
        :param max_iteration: How many (parallel) Ransac iterations to run
        :returns:
            best_eq: A list of integers representing the best 'equation' for the primitive shape.
            best_inliers: A list of indices of points that fit the shape.
        """
        jobs = ((self.random, points, float(thresh)) for _ in range(max_iteration))
        for eq, point_id_inliers in self.executor.map(self.iteration, *zip(*jobs)):
            if len(point_id_inliers) > len(self.best_inliers):
                self.best_eq = eq
                self.best_inliers = point_id_inliers
        return self.best_eq, self.best_inliers

    @abstractmethod
    def iteration(
        self, random: Random, pts: np.ndarray, thresh: float
    ) -> Tuple[List[int], List[int]]:
        pass
