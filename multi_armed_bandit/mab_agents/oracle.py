from typing import List

from .agent import Agent
from ..environments import Environment

class Oracle(Agent):

    _env: Environment
    actions: List[str]
    _id: str

    def __init__(self, env: Environment):
        self._id = 'OracleAgent'
        self._env = env
        self.actions = env.get_actions()
    
    def update_estimates(self, state:int, action: str, reward: int) -> None:
        pass

    def select_action(self, state:int, available_actions:List[str]) -> str:
        return self._env.get_best_action()
