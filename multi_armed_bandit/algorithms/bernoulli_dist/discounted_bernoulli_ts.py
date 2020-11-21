import matplotlib.pyplot as plt
import numpy as np

from numpy.random import beta

from . import BernoulliAlgo


'''
From Paper: "Taming Non-stationary Bandits: A Bayesian Approach"
'''
class DiscountedBernoulliTS(BernoulliAlgo):

    _alpha_0: float
    _beta_0: float
    _gamma: float

    def __init__(self, n_arms: int, gamma: float = 0.9, alpha_0: float = 1, beta_0: float = 1, store_estimates:bool=True):
        super().__init__(n_arms=n_arms, store_estimates=store_estimates)
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._gamma = gamma

    def __repr__(self):
        return "Discounted TS"

    def update_estimates(self, action: int, reward: int) -> None:
        self._betas *= self._gamma
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha
        if self._store_estimates:
            for a in range(self._n_arms):
                self._mean_trace[a].append(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]))

    def select_action(self) -> int:
        samples = [beta(a=self._betas[a][0] + self._alpha_0, b=self._betas[a][1] + self._beta_0)
                   for a in range(self._n_arms)]
        return np.argmax(samples)
