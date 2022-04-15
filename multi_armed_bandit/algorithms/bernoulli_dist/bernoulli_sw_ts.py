from .bernoulli_algo import BernoulliAlgo

import numpy as np
from numpy.random import beta
from typing import List, Tuple


'''
Sliding-Window Thompson Sampling Algorithm
From Paper: "Sliding-Window Thompson Sampling for Non-Stationary Settings"
'''
class BernoulliSlidingWindowTS(BernoulliAlgo):

    _n: int
    _last_n_action: List[Tuple]

    def __init__(self, n_arms: int, n: int, store_estimates:bool=True):
        super().__init__(n_arms, store_estimates=store_estimates)
        self._n = n
        self._last_n_action = []
    
    def __repr__(self):
        return "Sliding Window TS"

    def update_estimates(self, action: int, context: np.array, reward: int) -> None:
        self.update_last_n_action_trace(action, reward)
        
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha
        if self._store_estimates:
            for a in range(self._n_arms):
                self._mean_trace[a].append(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]))

    def update_last_n_action_trace(self, action: int, reward: int):
        if len(self._last_n_action) >= self._n:
            tmp_action, tmp_reward = self._last_n_action.pop(0)
            if tmp_reward == 0 and self._betas[tmp_action][1] > 1:
                self._betas[tmp_action][1] -= 1  # Update beta
            elif tmp_reward == 1 and self._betas[tmp_action][0] > 1:  # Reward == 1
                self._betas[tmp_action][0] -= 1  # Update alpha
        
        self._last_n_action.append((action, reward))

    def select_action(self, context: np.array) -> int:
        samples = [beta(a=self._betas[a][0], b=self._betas[a][1])
                   for a in range(self._n_arms)]
        return np.argmax(samples)
