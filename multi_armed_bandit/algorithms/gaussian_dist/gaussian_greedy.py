from . import GaussianAlgo

import numpy as np


class GaussianGreedy(GaussianAlgo):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def select_action(self) -> int:
        return np.argmax(self._mu)
