import matplotlib.pyplot as plt
import numpy as np

from numpy.random import beta
from typing import Dict
from copy import deepcopy

from . import BernoulliAlgo


class MyBernoulliTS(BernoulliAlgo):

    _last_reward_trace: Dict
    _n: int
    _alpha_0: float
    _beta_0: float
    _gamma: float

    def __init__(self, n_arms: int, gamma: float = 0.9, alpha_0: float = 1, beta_0: float = 1, threshold: float = 0.1, n: int = 10):
        super().__init__(n_arms=n_arms)
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._threshold = threshold
        self._last_reward_trace = {a : [] for a in range(n_arms)}
        self._n = n
        self._gamma = gamma

    def __repr__(self):
        return "My Thompson Sampling Bernoulli"

    def update_estimates(self, action: int, reward: int) -> None:
        self._betas *= self._gamma

        self.update_last_reward_trace(action, reward)
        tmp_mean = self._last_reward_trace[action].count(1) / len(self._last_reward_trace[action])
        distance = abs(self._betas[action][0] / (self._betas[action][0] + self._betas[action][1]) - tmp_mean)

        if distance > self._threshold:
            self._betas[action][0] = 1 + self._last_reward_trace[action].count(1) # Reward == 1
            self._betas[action][1] = 1 + self._last_reward_trace[action].count(0) # Reward == 0
        else:
            if reward == 0:
                self._betas[action][1] += 1  # Update beta
            else:  # Reward == 1
                self._betas[action][0] += 1  # Update alpha
 
        for a in range(self._n_arms):
            self._mean_trace[a].append(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]))

    def update_last_reward_trace(self, action: int, reward: int):
        if len(self._last_reward_trace[action]) >= self._n:
            self._last_reward_trace[action].pop(0)
        self._last_reward_trace[action].append(reward)

    def select_action(self) -> int:
        samples = [beta(a=self._betas[a][0] + self._alpha_0, b=self._betas[a][1] + self._beta_0)
                   for a in range(self._n_arms)]
        return np.argmax(samples)
