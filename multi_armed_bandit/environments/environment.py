from abc import ABC, abstractmethod
from typing import List, Dict
from random import choice


class Environment(ABC):

    _actions: List[str]
    _states: Dict

    def __init__(self, actions: List[str], states: Dict):
        self._actions = actions
        self._states = states
    
    def __repr__(self) -> str:
        return type(self).__name__

    def get_actions(self) -> List[str]:
        return self._actions
    
    def get_states(self) -> Dict:
        return self._states

    def get_state(self) -> Dict:
        return self._state

    @abstractmethod
    def do_action(self, action: str):
        pass

    @abstractmethod
    def next_state(self) -> None:
        pass
