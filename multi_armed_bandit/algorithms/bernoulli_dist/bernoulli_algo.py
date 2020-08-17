import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict
from abc import ABC

from ..algorithm import Algorithm


class BernoulliAlgo(Algorithm, ABC):

    _betas: np.ndarray
    _mean_trace: Dict

    def __init__(self, n_arms: int):
        super().__init__(n_arms=n_arms)
        self._betas = np.ones(shape=(n_arms, 2))
        self._mean_trace = {action : [1/2] for action in range(n_arms)}

    def reset_agent(self):
        self._betas = np.ones(shape=(self._n_arms, 2))
        self._mean_trace = {action : [1/2] for action in range(self._n_arms)}

    def update_estimates(self, action: int, reward: int) -> None:
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha
        for a in range(self._n_arms):
            self._mean_trace[a].append(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]))

    def plot_estimates(self, render: bool = True):
        fig = plt.figure()
        for a in range(self._n_arms):
            _ = plt.plot(self._mean_trace[a], label="Action: " + str(a))

        fig.suptitle("Action's estimates")
        fig.legend()
        if render:
            fig.show()
