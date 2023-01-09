import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, binomial
from typing import List

from . import Environment


class BernoulliEnvironment(Environment):

    _probabilities: float
    _best_action: int
    _state: int
    
    def __init__(self, n_arms: int, probabilities: List[float] = None):

        super().__init__(n_arms)

        if probabilities == None:
            self._probabilities = [uniform(0, 1) for _ in range(n_arms)]
        elif probabilities != None and n_arms == len(probabilities):
            self._probabilities = probabilities
        elif probabilities != None and n_arms != len(probabilities):
            raise Exception(
                'Length of probabilities vector must be the same of number of arms')
        
        self._best_action = np.argmax(self._probabilities)

    def __repr__(self):
        return 'Bernoulli Environment'
        
    def do_action(self, action: int):
        return binomial(size=1, n=1, p= self._probabilities[action])[0]

    def get_state(self):
        self._state = 1
        return self._state
