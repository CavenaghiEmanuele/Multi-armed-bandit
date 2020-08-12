import matplotlib.pyplot as plt
import numpy as np

from numpy.random import beta
from abc import ABC

from . import BernoulliAlgo

'''
From Paper: "Taming Non-stationary Bandits: A Bayesian Approach"
'''
class DynamicBernoulliTS(BernoulliAlgo, ABC):
    
    _alpha_0: float
    _beta_0: float
    _gamma: float

    def __init__(self, n_arms: int, gamma: float = 0.9, alpha_0: float = 1, beta_0: float = 1):
        super().__init__(n_arms=n_arms)
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._gamma = gamma
        
    def __repr__(self):
        return "Dynamic Thompson Sampling Bernoulli"

    def update_estimates(self, action: int, reward: int) -> None:
        self._betas *= self._gamma
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha
            
            
    def select_action(self) -> int:
        samples = [beta(a=self._betas[a][0] + self._alpha_0, b=self._betas[a][1] + self._beta_0)
                   for a in range(self._n_arms)]
        return np.argmax(samples)

    def plot_estimates(self):  
        fig = plt.figure()
        for a in range(self._n_arms):
            _ = plt.bar(a, self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]), label="Action: " + str(a) + 
                        ", Mean: " + str(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1])) + 
                        ", Parms: " + str(self._betas[a]))
                        
        fig.suptitle("Action's estimates")
        fig.legend()
        fig.show()
