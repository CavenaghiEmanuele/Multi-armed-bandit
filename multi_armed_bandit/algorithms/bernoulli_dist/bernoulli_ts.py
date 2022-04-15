from .bernoulli_algo import BernoulliAlgo

import numpy as np
from numpy.random import beta


class BernoulliThompsonSampling(BernoulliAlgo):

    def __init__(self, n_arms: int, store_estimates:bool=True):
        super().__init__(n_arms, store_estimates=store_estimates)
    
    def __repr__(self):
        return "Thompson Sampling"

    def select_action(self, context: np.array) -> int:
        samples = [beta(a=self._betas[a][0], b=self._betas[a][1])
                   for a in range(self._n_arms)]
        return np.argmax(samples)
