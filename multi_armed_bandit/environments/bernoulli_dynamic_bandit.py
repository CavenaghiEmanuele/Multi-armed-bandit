import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, binomial
from typing import List, Dict

from . import BernoulliBandit


class BernoulliDynamicBandit(BernoulliBandit):

    _probability_of_change: float
    _action_selection: Dict
    _fixed_actions: List = []

    def __init__(self, n_arms: int, probabilities: List[float] = None, 
                 probability_of_change: float = 0.001, fixed_action_prob: float = None):
        super().__init__(n_arms, probabilities)
        self._probability_of_change = probability_of_change
        self._action_selection = {a:[self._probabilities[a]] for a in range(n_arms)}
        
        if fixed_action_prob != None:
            for i in range(n_arms):
                if uniform(0, 1) < fixed_action_prob:
                    self._fixed_actions.append(i)
                    
    def plot_arms(self, render: bool = True):
        plt.figure()
        for a in range(self._n_arms):
            plt.plot(self._action_selection[a], label="Action: " +
                        str(a) + ", prob: " + str(self._probabilities[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def _change_action_probabilities(self):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._probability_of_change):
                self._probabilities[action] = uniform(0, 1)
                self._best_action_mean = np.max(self._probabilities)
            self._action_selection[action].append(self._probabilities[action])

    def do_action(self, action: int):
        reward = binomial(size=1, n=1, p=self._probabilities[action])
        self._change_action_probabilities()
        return reward
