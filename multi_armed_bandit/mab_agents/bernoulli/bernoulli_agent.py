import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict


from ..agent import Agent


class BernoulliAgent(Agent, ABC):

    _actions: List[str]
    _id: str
    _betas: Dict

    def __init__(self, id:str, actions: List[str]):
        super().__init__(id, actions)
        self._betas = {action:[1,1] for action in actions}

    def reset_agent(self) -> None:
        self._betas = {action:[1,1] for action in self._actions}
        return
    
    def update_estimates(self, state:int, action: str, reward: int) -> None:
        if reward == 0:
            self._betas[action][1] += 1  # Update beta
        else:  # Reward == 1
            self._betas[action][0] += 1  # Update alpha

    @abstractmethod
    def select_action(self, state:int) -> str:
        pass