from . import GaussianAlgo

import numpy as np


class GaussianGreedy(GaussianAlgo):

    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms, decay_rate)

    def __repr__(self):
        return "Greedy gaussian, decay rate: " + str(self._decay_rate)

    def select_action(self) -> int:
        return np.argmax(self._mu)
