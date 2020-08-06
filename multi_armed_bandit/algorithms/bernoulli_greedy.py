from .bernoulli_algo import BernoulliAlgo

import numpy as np


class BernoulliGreedy(BernoulliAlgo):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def select_action(self) -> int:
        estimates = [self._betas[a][0] / (self._betas[a][0] + self._betas[a][1])
                     for a in range(self._n_arms)]
        return np.argmax(estimates)
