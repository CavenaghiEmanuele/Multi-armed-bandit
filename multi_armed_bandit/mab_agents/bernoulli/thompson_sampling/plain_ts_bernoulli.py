import itertools
from typing import List, Dict
from numpy.random import beta

from ...agent import Agent
from ....utils import from_dict_to_str


class PlainTSBernoulli(Agent):

    _parameters: Dict

    def __init__(self, id:str, actions: List[str], states:Dict):
        super().__init__(id, actions, states)
        self._parameters = { 
            from_dict_to_str(state) : {action: [1, 1] for action in self._actions}
            for state in [dict(zip(self._states.keys(),items)) for items in itertools.product(*self._states.values())]
        }

    def update_estimates(self, state:int, action: str, reward: int) -> None:
        if reward == 0:
            self._parameters[from_dict_to_str(state)][action][1] += 1  # Update beta
        else:  # Reward == 1
            self._parameters[from_dict_to_str(state)][action][0] += 1  # Update alpha
 
    def select_action(self, state:int, available_actions:List[str]) -> str:
        samples = {a:beta(
            a=self._parameters[from_dict_to_str(state)][a][0], 
            b=self._parameters[from_dict_to_str(state)][a][1]
            )
            for a in available_actions}
        return max(samples, key=samples.get)
