from abc import ABC, abstractmethod
from typing import List


class Environment(ABC):

    _actions: List[str]
    _best_action: str

    def __init__(self, actions: List[str]):
        self._actions = actions
    
    def __repr__(self) -> str:
        return type(self).__name__

    def get_actions(self) -> List[str]:
        return self._actions

    def get_best_action(self, state:int) -> str:
        return self._best_action

    @abstractmethod
    def do_action(self, action: str):
        pass

    @abstractmethod
    def get_state(self):
        pass
