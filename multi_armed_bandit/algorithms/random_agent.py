from random import randint

from .algorithm import Algorithm


class RandomAlgo(Algorithm):

    def __init__(self, n_arms: int, store_estimates:bool=True):
        super().__init__(n_arms=n_arms)

    def __repr__(self):
        return "Random"

    def reset_agent(self):
        return

    def update_estimates(self, action: int, reward: int) -> None:
        return

    def select_action(self) -> int:
        return randint(0, self._n_arms-1) 

    def plot_estimates(self, render: bool = True):
        return
