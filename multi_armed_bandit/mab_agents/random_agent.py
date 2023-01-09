from random import randint

from .agent import Agent


class RandomAgent(Agent):

    def __init__(self, id:str, n_arms: int):
        super().__init__(id=id, n_arms=n_arms)

    def reset_agent(self):
        return

    def update_estimates(self, state:int, action: int, reward: int) -> None:
        return

    def select_action(self, state:int) -> int:
        return randint(0, self._n_arms-1) 
