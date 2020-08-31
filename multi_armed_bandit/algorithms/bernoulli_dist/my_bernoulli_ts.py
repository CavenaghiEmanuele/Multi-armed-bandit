import matplotlib.pyplot as plt
import numpy as np

from numpy.random import beta
from typing import Dict

from . import BernoulliAlgo


class MyBernoulliTS(BernoulliAlgo):

    _last_reward_trace: Dict
    _tmp_betas: np.ndarray
    _n: int
    _gamma: float

    def __init__(self, n_arms: int, gamma: float = 0.9, threshold: float = 0.1, n: int = 10):
        super().__init__(n_arms=n_arms)
        self._threshold = threshold
        self._last_reward_trace = {a : [] for a in range(n_arms)}
        self._tmp_betas = np.ones(shape=(n_arms, 2))
        self._n = n
        self._gamma = gamma

    def __repr__(self):
        return "My Thompson Sampling Bernoulli"

    def update_estimates(self, action: int, reward: int) -> None:
        self._betas *= self._gamma

        self.update_tmp_betas(action, reward)
        tmp_mean = self._tmp_betas[action][1] / (self._tmp_betas[action][1] + self._tmp_betas[action][0])
        long_mean = self._betas[action][1] / (self._betas[action][1] + self._betas[action][0])
        distance = abs(long_mean - tmp_mean)

        if distance > self._threshold and len(self._last_reward_trace[action]) >= self._n:
            self._betas[action] = self._tmp_betas[action]
        else:
            self._betas[action][reward] += 1

        for a in range(self._n_arms):
            self._mean_trace[a].append(self._betas[a][1] / (self._betas[a][1] + self._betas[a][0]))

    def update_tmp_betas(self, action: int, reward: int):
        if len(self._last_reward_trace[action]) >= self._n:
            tmp_reward = self._last_reward_trace[action].pop(0)
            self._tmp_betas[action][tmp_reward] -= 1

        self._tmp_betas[action][reward] += 1
        self._last_reward_trace[action].append(reward)

    def select_action(self) -> int:
        samples = [beta(a=self._betas[a][1], b=self._betas[a][0])
                   for a in range(self._n_arms)]
        tmp_samples = [beta(a=self._tmp_betas[a][1], b=self._tmp_betas[a][0])
                   for a in range(self._n_arms)]
        
        return np.argmax([max(samples[a], tmp_samples[a]) for a in range(self._n_arms)])
