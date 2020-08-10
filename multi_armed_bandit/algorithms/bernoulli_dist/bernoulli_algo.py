import matplotlib.pyplot as plt

from  uuid import uuid1
from typing import List, Tuple
from abc import ABC, abstractmethod


class BernoulliAlgo(ABC):

    _n_arms: int
    _betas: List[Tuple]

    def __init__(self, n_arms: int):
        super().__init__()
        self._id = uuid1()
        self._n_arms = n_arms
        self._betas = [[1, 1] for _ in range(n_arms)]

    def update_estimates(self, action: int, reward: int) -> None:
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha

    @abstractmethod
    def select_action(self) -> int:
        pass
    
    def get_id(self):
        return self._id
    
    def plot_estimates(self):  
        fig = plt.figure()
        for a in range(self._n_arms):
            _ = plt.bar(a, self._betas[a][0] / (self._betas[a][0] + self._betas[a][1]), label="Action: " + str(a) + 
                        ", Mean: " + str(self._betas[a][0] / (self._betas[a][0] + self._betas[a][1])) + 
                        ", Parms: " + str(self._betas[a]))
                        
        fig.suptitle("Action's estimates")
        fig.legend()
        fig.show()
