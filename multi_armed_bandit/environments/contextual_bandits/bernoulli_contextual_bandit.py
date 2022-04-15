from matplotlib.style import context
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, binomial
from typing import List

from .. import MultiArmedBandit


class BernoulliContextualBandit(MultiArmedBandit):

    _probabilities: np.array # 2D matrix -> row=arms, columns=weights
    
    def __init__(self, n_arms: int, context_dim: int, probabilities: np.array = None):

        super().__init__(n_arms)

        if probabilities == None:
            self._probabilities = np.random.rand(n_arms, context_dim)
        elif probabilities != None and n_arms == len(probabilities):
            self._probabilities = probabilities
        elif probabilities != None and n_arms != len(probabilities.shape[0]):
            raise Exception(
                "Length of probabilities vector must be the same of number of arms")
        
        self._context = np.random.rand(context_dim)
        self._best_action = np.argmax([1/(1 + np.exp(sum(self._probabilities[a] * self._context))) for a in range(n_arms)])
       

    def __repr__(self):
        return "Bernoulli Multi-armed bandit\n" + \
            "Probabilities = " + str(self._probabilities)

    def plot_arms(self, render: bool = True):
       return

    def do_action(self, action: int):
        f_x = sum(self._probabilities[action] * self._context)
        return binomial(size=1, n=1, p=1/(1 + np.exp(-f_x)))

    def best_action_mean(self):
        f_x = sum(self._probabilities[self._best_action] * self._context)
        return 1/(1 + np.exp(-f_x))

    def get_best_action(self):
        return self._best_action

    def action_mean(self, action: int):
        f_x = sum(self._probabilities[action] * self._context)
        return 1/(1 + np.exp(-f_x))

    def update_context(self):
        self._context = np.random.rand(self._probabilities.shape[1])
        self._best_action = np.argmax(
            [
                1/(1 + np.exp(-sum(self._probabilities[a] * self._context))) 
                for a in range(self._probabilities.shape[0])
            ])
        return
