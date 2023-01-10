from abc import ABC, abstractmethod
from functools import total_ordering
from typing import List, Dict


@total_ordering
class Agent(ABC):

    _actions: List[str]
    _id: str

    def __init__(self, id:str, actions: List[str], states:Dict=None):
        self._id = type(self).__name__ + str(id)
        self._actions = actions
        self._states = states
    
    def __repr__(self):
        return self._id
    
    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        return self._id == other.get_id()

    def __lt__(self, other):
        return self._id < other.get_id()

    def get_id(self) -> str:
        return self._id
    
    @abstractmethod
    def reset_agent(self):
        pass
    
    @abstractmethod
    def update_estimates(self, state:int, action: str, reward: int) -> None:
        pass

    @abstractmethod
    def select_action(self, state:int, available_actions:List[str]) -> int:
        pass