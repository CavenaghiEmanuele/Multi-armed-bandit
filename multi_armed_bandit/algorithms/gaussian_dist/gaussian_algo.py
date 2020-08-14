import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from typing import List
from abc import ABC

from ..algorithm import Algorithm


class GaussianAlgo(Algorithm, ABC):

    _mu: List[float]
    _std_dev: List[float]
    _n_action_taken: List[int]

    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms=n_arms)
        self._mu = np.ones(n_arms)
        self._std_dev = np.ones(n_arms)
        self._decay_rate = decay_rate
        self._n_action_taken = np.zeros(n_arms)
        
    def reset_agent(self):
        self._mu = np.ones(self._n_arms)
        self._std_dev = np.ones(self._n_arms)
        self._n_action_taken = np.zeros(self._n_arms)

    def update_estimates(self, action: int, reward: int) -> None:
        self._n_action_taken[action] += 1
        self._std_dev[action] *= self._decay_rate
        n = self._n_action_taken[action]
        if n == 0:
            self._mu[action] = reward
            return
        self._mu[action] += (1 / n) * (reward - self._mu[action])
    
    def plot_estimates(self):  
        fig = plt.figure()
        for a in range(self._n_arms):
            x = np.linspace(self._mu[a] - 3*self._std_dev[a], self._mu[a] + 3*self._std_dev[a])
            _ = plt.plot(x, 
                     stats.norm.pdf(x, self._mu[a], self._std_dev[a]), 
                     label="Action: " + str(a) + ", Mean: " + str(self._mu[a]) + ", std_dev: " + str(self._std_dev[a]))
        fig.suptitle("Action's estimates")
        fig.legend()
        fig.show()
