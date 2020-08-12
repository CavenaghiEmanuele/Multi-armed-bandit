from uuid import uuid1
from abc import ABC, abstractmethod


class Algorithm(ABC):

    _n_arms: int

    def __init__(self, n_arms: int):
        self._id = uuid1()
        self._n_arms = n_arms

    def get_id(self):
        return self._id
    
    @abstractmethod
    def update_estimates(self, action: int, reward: int) -> None:
        pass

    @abstractmethod
    def select_action(self) -> int:
        pass
    
    @abstractmethod
    def plot_estimates(self):  
        pass
