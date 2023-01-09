import numpy as np

from numpy.random import beta
from abc import ABC, abstractmethod
from typing import List


from ..agent import Agent


class BernoulliAgent(Agent, ABC):

    _n_arms: int
    _id: str
    _betas: np.ndarray

    def __init__(self, id:str, n_arms: int):
        super().__init__(id, n_arms)
        self._betas = np.ones(shape=(n_arms, 2))

    def reset_agent(self):
        self._betas = np.ones(shape=(self._n_arms, 2))
        return
    
    def update_estimates(self, state:int, action: int, reward: int) -> None:
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha

    @abstractmethod
    def select_action(self, state:int) -> int:
        pass