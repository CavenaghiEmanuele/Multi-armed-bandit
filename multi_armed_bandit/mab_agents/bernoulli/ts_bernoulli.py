import numpy as np

from numpy.random import beta
from typing import List

from .bernoulli_agent import BernoulliAgent

class TSBernoulli(BernoulliAgent):

    def __init__(self, id:str, actions: List[str]):
        super().__init__(id, actions)
 
    def select_action(self, state:int) -> str:
        samples = {a:beta(a=self._betas[a][0], b=self._betas[a][1])
                   for a in self._actions}
        return max(samples, key=samples.get)
