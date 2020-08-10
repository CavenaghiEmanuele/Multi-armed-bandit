import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from typing import List, Dict

from . import GaussianBandit


class GaussianDynamicBandit(GaussianBandit):

    _probability_of_change: float
    _action_selection: Dict

    def __init__(self, n_arms: int, mean: List[float] = None, std_dev: List[float] = None, probability_of_change: float = 0.001):
        super().__init__(n_arms, mean, std_dev)
        self._probability_of_change = probability_of_change
        self._action_selection = {a:[self._mean[a]] for a in range(n_arms)}

    def plot_arms(self, render: bool = True):  
        plt.figure()
        for a in range(self._n_arms):
            plt.plot(self._action_selection[a], 
                     label="Action: " + str(a) + ", Mean: " + str(self._mean[a]) + ", std_dev: " + str(self._std_dev[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def _change_action_probabilities(self):
        for action in range(self._n_arms):
            if uniform(0, 1) < self._probability_of_change:
                self._mean[action] = uniform(0, 1)
                self._best_action_mean = np.max(self._mean)
            self._action_selection[action].append(self._mean[action])

    def do_action(self, action: int):
        reward = normal(loc=self._mean[action], scale=self._std_dev[action])
        self._change_action_probabilities()
        return reward
