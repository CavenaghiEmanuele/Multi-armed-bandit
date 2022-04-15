import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from abc import ABC

from ..algorithm import Algorithm


class GaussianAlgo(Algorithm, ABC):

    _mu: List[float]
    _std_dev: List[float]
    _n_action_taken: List[int]
    _mean_trace: Dict


    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms=n_arms)
        self._mu = np.ones(n_arms)/2
        self._std_dev = np.ones(n_arms)
        self._decay_rate = decay_rate
        self._n_action_taken = np.zeros(n_arms)
        self._mean_trace = {action : [1/2] for action in range(n_arms)}
        
    def reset_agent(self):
        self._mu = np.ones(self._n_arms)/2
        self._std_dev = np.ones(self._n_arms)
        self._n_action_taken = np.zeros(self._n_arms)
        self._mean_trace = {action : [1/2] for action in range(self._n_arms)}

    def update_estimates(self, action: int, context: np.array, reward: int) -> None:
        self._n_action_taken[action] += 1
        self._std_dev[action] *= self._decay_rate
        n = self._n_action_taken[action]
        if n == 0:
            self._mu[action] = reward
            return
        self._mu[action] += (1 / n) * (reward - self._mu[action])
        for a in range(self._n_arms):
            self._mean_trace[a].append(self._mu[a])
    
    def plot_estimates(self, render: bool = True):  
        fig = plt.figure()
        for a in range(self._n_arms):
            _ = plt.plot(self._mean_trace[a], label="Action: " + str(a))
        fig.suptitle("Action's estimates")
        fig.legend()
        if render:
            fig.show()
