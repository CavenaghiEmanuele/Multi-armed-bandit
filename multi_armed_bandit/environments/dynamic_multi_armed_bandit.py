from abc import ABC, abstractmethod
from numpy.random import uniform
from typing import List, Dict

from .multi_armed_bandit import MultiArmedBandit


class DiscountedMultiArmedBandit(MultiArmedBandit, ABC):

    _prob_of_change: float
    _action_value_trace: Dict
    _fixed_actions: List = []
    _save_replay: bool
    _replays: Dict

    def __init__(self, n_arms: int, prob_of_change: float = 0.001, fixed_action_prob: float = None, save_replay: bool = False):
        super().__init__(n_arms)
        self._prob_of_change = prob_of_change
        self._save_replay = save_replay
        
        if save_replay:
            self._replays = {}

        if fixed_action_prob != None:
            for i in range(n_arms):
                if uniform(0, 1) < fixed_action_prob:
                    self._fixed_actions.append(i)

    def get_replay(self):
        return self._replays
    
    @abstractmethod          
    def change_action_prob(self, step: int):
        pass
    