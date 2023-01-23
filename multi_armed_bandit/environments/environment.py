from abc import ABC, abstractmethod
from typing import List, Dict
from random import choice


class Environment(ABC):

    _actions: List[str]
    _contexts: Dict
    _available_actions: Dict

    def __init__(self, actions: List[str], contexts: Dict, available_actions:Dict=None):
        # available_actions is in the form of:
        # {dim: {value:[actions], value:[actions]}} 
        # {'city': {'milan':['hotel1', 'hotel2'], 'rome': ['hotel3', 'hotel4', 'hotel5']}}
        self._actions = actions
        self._contexts = contexts
        self._available_actions = available_actions
    
    def __repr__(self) -> str:
        return type(self).__name__

    def get_actions(self) -> List[str]:
        return self._actions
    
    def get_contexts(self) -> Dict:
        return self._contexts

    def get_context(self) -> Dict:
        return self._context
    
    def get_available_actions(self, context:Dict=None) -> List[str]:
        if self._available_actions == None:
            return self._actions
        # The _available_actions dict has only one entry
        if context != None:
            for dim, values in self._available_actions.items():
                return values[context[dim]]
        for dim, values in self._available_actions.items():
            return values[self._context[dim]]

    @abstractmethod
    def do_action(self, action: str) -> int:
        pass

    @abstractmethod
    def next_context(self) -> None:
        pass

    @abstractmethod
    def get_best_action(self) -> str:
        pass
