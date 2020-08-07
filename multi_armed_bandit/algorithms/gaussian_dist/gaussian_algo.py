import numpy as np

from typing import List, Tuple
from abc import ABC, abstractmethod


class GaussianAlgo(ABC):

    _n_arms: int
    _mu: List[float]
    _dev_std: List[float]
    _n_action_taken: int

    def __init__(self, n_arms: int, decay_rate: float = 0.01):
        super().__init__()

        self._n_arms = n_arms
        self._mu = np.zeros(n_arms)
        self._dev_std = np.ones(n_arms)
        self._decay_rate = decay_rate
        self._n_action_taken = 0

    def update_estimates(self, action: int, reward: int) -> None:
        self._n_action_taken += 1
        n = self._n_action_taken
        old_mean = self._mu[action]
        self._mu[action] += (1 / n) * (reward - old_mean)
        self._dev_std[action] *= self._decay_rate
        

    @abstractmethod
    def select_action(self) -> int:
        pass
