import numpy as np
from numpy.random import uniform, binomial

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
        
        self._context = uniform(-1,1,context_dim)
        self._best_action = np.argmax(
            [
                max(-np.cos(-sum(self._probabilities[a] * self._context)), np.cos(-sum(self._probabilities[a] * self._context)))
                for a in range(self._probabilities.shape[0])
            ])
       

    def __repr__(self):
        return "Bernoulli Multi-armed bandit\n" + \
            "Probabilities = " + str(self._probabilities)

    def plot_arms(self, render: bool = True):
       return

    def do_action(self, action: int):
        f_x = sum(self._probabilities[action] * self._context)
        return binomial(size=1, n=1, p=max(-np.cos(f_x), np.cos(f_x)))

    def best_action_mean(self):
        f_x = sum(self._probabilities[self._best_action] * self._context)
        return max(-np.cos(f_x), np.cos(f_x))

    def get_best_action(self):
        return self._best_action

    def action_mean(self, action: int):
        f_x = sum(self._probabilities[action] * self._context)
        return max(-np.cos(f_x), np.cos(f_x))

    def update_context(self):
        self._context = self._context = uniform(-1,1,self._probabilities.shape[1])
        self._best_action = np.argmax(
            [
                max(-np.cos(-sum(self._probabilities[a] * self._context)), np.cos(-sum(self._probabilities[a] * self._context)))
                for a in range(self._probabilities.shape[0])
            ])
        return
