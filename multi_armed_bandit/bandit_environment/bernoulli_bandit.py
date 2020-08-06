import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, binomial
from typing import List

from . import MultiArmedBandit


class BernoulliBandit(MultiArmedBandit):

    _n_arms: int
    _probabilities: float
    _best_action_mean: float
    
    def __init__(self, n_arms: int, probabilities: List[float] = None):

        super().__init__(n_arms)

        if probabilities == None:
            self._probabilities = [uniform(0, 1) for _ in range(n_arms)]
        elif probabilities != None and n_arms == len(probabilities):
            self._probabilities = probabilities
        elif probabilities != None and n_arms != len(probabilities):
            raise Exception(
                "Length of probabilities vector must be the same of number of arms")
        
        self._best_action_mean = np.max(self._probabilities)


    def __repr__(self):
        return "Bernoulli Multi-armed bandit\n" + \
            "Probabilities = " + str(self._probabilities)
    
    def plot_arms(self):
        for a in range(self._n_arms):
            plt.bar(a, self._probabilities[a], label="Action: " + str(a) + ", prob: " + str(self._probabilities[a]))

        plt.legend()
        plt.show()
                    
    def do_action(self, action: int):
        return binomial(size=1, n=1, p= self._probabilities[action])
    
    def best_action_mean(self):
        return self._best_action_mean
    
    def action_mean(self, action: int):
        return self._probabilities[action]
