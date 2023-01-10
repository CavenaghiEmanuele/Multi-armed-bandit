from random import choice
from typing import List, Dict

from .agent import Agent


class RandomAgent(Agent):

    def __init__(self, id:str, actions: List[str]):
        super().__init__(id, actions)

    def reset_agent(self):
        return

    def update_estimates(self, state:int, action: str, reward: int) -> None:
        return

    def select_action(self, state:int, available_actions:List[str]) -> str:
        return choice(available_actions)
