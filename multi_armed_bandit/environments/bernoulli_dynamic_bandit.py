import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, binomial
from typing import List
from copy import deepcopy

from . import BernoulliBandit, DynamicMultiArmedBandit


class BernoulliDynamicBandit(DynamicMultiArmedBandit, BernoulliBandit):

    def __init__(self, n_arms: int, probabilities: List[float] = None, 
                 prob_of_change: float = 0.001, fixed_action_prob: float = None, save_replay: bool = False):
        
        DynamicMultiArmedBandit.__init__(self, n_arms=n_arms, prob_of_change=prob_of_change, fixed_action_prob=fixed_action_prob, save_replay=save_replay)
        BernoulliBandit.__init__(self, n_arms=n_arms, probabilities=probabilities)
        
        self._action_value_trace = {a:[self._probabilities[a]] for a in range(n_arms)}
        if save_replay:
            self._replays.update({"probabilities": deepcopy(self._probabilities)})


    def plot_arms(self, render: bool = True):
        plt.figure()
        for a in range(self._n_arms):
            plt.plot(self._action_value_trace[a], label="Action: " +
                        str(a) + ", prob: " + str(self._probabilities[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def change_action_prob(self, step: int):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                self._probabilities[action] = uniform(0, 1)
                self._best_action = np.argmax(self._probabilities)
                if self._save_replay:
                    try:
                        self._replays[step].append((action, self._probabilities[action]))
                    except:
                        self._replays.update({step : [(action, self._probabilities[action])]})
            self._action_value_trace[action].append(self._probabilities[action])
