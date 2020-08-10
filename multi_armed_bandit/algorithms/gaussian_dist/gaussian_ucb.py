from . import GaussianAlgo

import numpy as np
from math import log, sqrt
from typing import List


class GaussianUCB(GaussianAlgo):

    _action_taken: int
    _action_selection: List
    _c: float

    def __init__(self, n_arms: int, c: float=1, decay_rate: float = 0.01):
        super().__init__(n_arms)
        self._action_taken = 0
        self._action_selection = [0 for _ in range(n_arms)]
        self._c = c
        
    def __repr__(self):
        return "UCB gaussian, decay rate: " + str(self._decay_rate)

    def select_action(self) -> int:
        self._action_taken += 1
        estimates = []
        for a in range(self._n_arms):
            if self._action_selection[a] == 0:
                estimates.append(float("inf"))
            else:
                estimates.append(self._mu[a] + self._c * sqrt(log(self._action_taken) / self._action_selection[a]))

        action = np.argmax(estimates)
        self._action_selection[action] += 1
        return action
