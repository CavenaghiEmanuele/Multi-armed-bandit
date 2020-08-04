import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats
from scipy.stats import bernoulli
from numpy.random import uniform, binomial
from typing import List

from . import MultiArmedBandit


class BernoulliBandit(MultiArmedBandit):

    _n_arms: int
    _probabilities: float
    
    def __init__(self, n_arms: int, probabilities: List[float] = None):

        super().__init__(n_arms)

        if probabilities == None:
            self._probabilities = [uniform(0, 1) for _ in range(n_arms)]
        elif probabilities != None and n_arms == len(probabilities):
            self._probabilities = probabilities
        elif probabilities != None and n_arms != len(probabilities):
            raise Exception(
                "Length of probabilities vector must be the same of number of arms")


    def __repr__(self):
        return "Bernoulli Multi-armed bandit\n" + \
            "Probabilities = " + str(self._probabilities)
    
    def plot_arms(self):
        
        for a in range(self._n_arms):
            x = scipy.linspace(0,1)
            pmf = scipy.stats.binom.pmf(x, 1, self._probabilities[a])
            plt.plot(x, pmf, label="prob: " + str(self._probabilities[a]))

        plt.legend()
        plt.show()
                    
    def select_action(self, action: int):
        return binomial(size=1, n=1, p= self._probabilities[action])
    
    def select_best_action(self):
        best_action = np.argmax(self._probabilities)
        return self.select_action(best_action)
