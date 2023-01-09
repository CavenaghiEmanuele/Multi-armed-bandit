from abc import ABC, abstractmethod

from .agent import Agent
from ..environments import Environment

class Oracle(Agent):

    _env: Environment
    _n_arms: int
    _id: str

    def __init__(self, env: Environment):
        self._id = 'OracleAgent'
        self._env = env
        self._n_arms = env.get_n_arms()
    
    def reset_agent(self):
        pass
    
    def update_estimates(self, state:int, action: int, reward: int) -> None:
        pass

    def select_action(self, state:int) -> int:
        return self._env.get_best_action(state)
