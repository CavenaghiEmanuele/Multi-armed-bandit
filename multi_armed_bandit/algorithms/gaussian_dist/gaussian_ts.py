from . import GaussianAlgo

import numpy as np
from numpy.random import normal


class GaussianThompsonSampling(GaussianAlgo):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def select_action(self) -> int:
        samples = [normal(loc=self._mu[a], scale=self._dev_std[a]) for a in range(self._n_arms)]
        return np.argmax(samples)
