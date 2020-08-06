from typing import List, Tuple
from abc import ABC, abstractmethod


class BernoulliAlgo(ABC):

    _n_arms: int
    _betas: List[Tuple]

    def __init__(self, n_arms: int):
        super().__init__()

        self._n_arms = n_arms
        self._betas = [[1, 1] for _ in range(n_arms)]

    def update_betas(self, action: int, reward: int) -> None:
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha

    @abstractmethod
    def select_action(self) -> int:
        pass
