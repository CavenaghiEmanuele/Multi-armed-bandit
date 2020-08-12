from abc import ABC, abstractmethod


class MultiArmedBandit(ABC):

    _n_arms: int
    _best_action_mean: float

    def __init__(self, n_arms: int):
        self._n_arms = n_arms
    
    def get_n_arms(self):
        return self._n_arms

    @abstractmethod
    def plot_arms(self):
        pass

    @abstractmethod
    def do_action(self, action: int):
        pass
    
    @abstractmethod
    def best_action_mean(self):
        pass
    
    @abstractmethod
    def action_mean(self, action: int):
        pass
    