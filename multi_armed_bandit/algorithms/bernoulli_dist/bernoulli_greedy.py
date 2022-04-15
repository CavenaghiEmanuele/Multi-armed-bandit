from .bernoulli_algo import BernoulliAlgo

import numpy as np


class BernoulliGreedy(BernoulliAlgo):

    def __init__(self, n_arms: int, store_estimates:bool=True):
        super().__init__(n_arms, store_estimates=store_estimates)
    
    def __repr__(self):
        return "Greedy"

    def select_action(self, context: np.array) -> int:
        estimates = [self._betas[a][0] / (self._betas[a][0] + self._betas[a][1])
                     for a in range(self._n_arms)]
        return np.argmax(estimates)
