import numpy as np
from abc import ABC, abstractmethod


class MultiArmedBandit(ABC):

    _n_arms: int

    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    @abstractmethod
    def plot_arms(self):
        pass

    @abstractmethod
    def select_action(self, action: int):
        pass
    
    @abstractmethod
    def select_best_action(self):
        pass