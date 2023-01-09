from .bernoulli_agent import BernoulliAgent

import numpy as np
from numpy.random import beta


class TSBernoulli(BernoulliAgent):

    def __init__(self, id:str, n_arms: int):
        super().__init__(id, n_arms)
 
    def select_action(self, state:int) -> int:
        samples = [beta(a=self._betas[a][0], b=self._betas[a][1])
                   for a in range(self._n_arms)]
        return np.argmax(samples)
