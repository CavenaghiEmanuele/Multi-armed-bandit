from .bernoulli_algo import BernoulliAlgo

import numpy as np
from math import log, sqrt
from typing import List


class BernoulliUCB(BernoulliAlgo):

    _action_taken: int
    _action_selection: List
    _c: float

    def __init__(self, n_arms: int, c: float=1, store_estimates:bool=True):
        super().__init__(n_arms, store_estimates=store_estimates)
        self._action_taken = 0
        self._action_selection = [0 for _ in range(n_arms)]
        self._c = c
    
    def __repr__(self):
        return "UCB Bernoulli"

    def select_action(self, context: np.array) -> int:
        self._action_taken += 1
        estimates = []
        for a in range(self._n_arms):
            if self._action_selection[a] == 0:
                estimates.append(float("inf"))
            else:
                estimates.append(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]) + 
                                 self._c * sqrt(log(self._action_taken) / self._action_selection[a]))

        action = np.argmax(estimates)
        self._action_selection[action] += 1
        return action
