from abc import ABC, abstractmethod


class Environment(ABC):

    _n_arms: int
    _best_action: int

    def __init__(self, n_arms: int):
        self._n_arms = n_arms
    
    def get_n_arms(self):
        return self._n_arms

    def get_best_action(self, state:int):
        return self._best_action

    @abstractmethod
    def do_action(self, action: int):
        pass

    @abstractmethod
    def get_state(self):
        pass
