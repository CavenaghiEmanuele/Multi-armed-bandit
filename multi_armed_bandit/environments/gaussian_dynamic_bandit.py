import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, normal
from typing import List

from . import GaussianBandit, DynamicMultiArmedBandit


class GaussianDynamicBandit(DynamicMultiArmedBandit, GaussianBandit):

    def __init__(self, n_arms: int, mean: List[float] = None, std_dev: List[float] = None,
                 prob_of_change: float = 0.001, fixed_action_prob: float = None):

        DynamicMultiArmedBandit.__init__(self, n_arms=n_arms, prob_of_change=prob_of_change, fixed_action_prob=fixed_action_prob)
        GaussianBandit.__init__(self, n_arms=n_arms,
                                mean=mean, std_dev=std_dev)

        self._action_value_trace = {a: [self._mean[a]] for a in range(n_arms)}

    def plot_arms(self, render: bool = True):
        plt.figure()
        for a in range(self._n_arms):
            plt.plot(self._action_value_trace[a],
                     label="Action: " + str(a) + ", Mean: " + str(self._mean[a]) + ", std_dev: " + str(self._std_dev[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def change_action_prob(self):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                self._mean[action] = uniform(0, 1)
                self._best_action = np.argmax(self._mean)
            self._action_value_trace[action].append(self._mean[action])
