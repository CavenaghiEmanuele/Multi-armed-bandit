from abc import ABC, abstractmethod
from typing import List, Dict
from random import choice


class Environment(ABC):

    _actions: List[str]
    _states: Dict
    _available_actions: Dict

    def __init__(self, actions: List[str], states: Dict, available_actions:Dict=None):
        # available_actions is in the form of:
        # {dim: {value:[actions], value:[actions]}} 
        # {'city': {'milan':['hotel1', 'hotel2'], 'rome': ['hotel3', 'hotel4', 'hotel5']}}
        self._actions = actions
        self._states = states
        self._available_actions = available_actions
    
    def __repr__(self) -> str:
        return type(self).__name__

    def get_actions(self) -> List[str]:
        return self._actions
    
    def get_states(self) -> Dict:
        return self._states

    def get_state(self) -> Dict:
        return self._state
    
    def get_available_actions(self, state:Dict=None) -> List[str]:
        if self._available_actions == None:
            return self._actions
        # The _available_actions dict has only one entry
        if state != None:
            for dim, values in self._available_actions.items():
                return values[state[dim]]
        for dim, values in self._available_actions.items():
            return values[self._state[dim]]

    @abstractmethod
    def do_action(self, action: str):
        pass

    @abstractmethod
    def next_state(self) -> None:
        pass
