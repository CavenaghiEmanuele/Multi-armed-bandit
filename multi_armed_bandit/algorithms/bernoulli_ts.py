from .bernoulli_algo import BernoulliAlgo

import numpy as np
from numpy.random import beta


class BernoulliThompsonSampling(BernoulliAlgo):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)

    def select_action(self) -> int:
        samples = [beta(a=self._betas[a][0], b=self._betas[a][1])
                   for a in range(self._n_arms)]
        return np.argmax(samples)
