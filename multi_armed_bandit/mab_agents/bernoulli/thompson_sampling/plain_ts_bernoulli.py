import itertools
from typing import List, Dict
from numpy.random import beta

from ...agent import Agent
from ....utils import from_dict_to_str


class PlainTSBernoulli(Agent):

    _parameters: Dict

    def __init__(self, id:str, actions: List[str], contexts:Dict):
        super().__init__(id, actions, contexts)
        self._parameters = { 
            from_dict_to_str(context) : {action: [1, 1] for action in self._actions}
            for context in [dict(zip(self._contexts.keys(),items)) for items in itertools.product(*self._contexts.values())]
        }

    def update_estimates(self, context:int, action: str, reward: int) -> None:
        if reward == 0:
            self._parameters[from_dict_to_str(context)][action][1] += 1  # Update beta
        else:  # Reward == 1
            self._parameters[from_dict_to_str(context)][action][0] += 1  # Update alpha
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {a:beta(
            a=self._parameters[from_dict_to_str(context)][a][0], 
            b=self._parameters[from_dict_to_str(context)][a][1]
            )
            for a in available_actions}
        return max(samples, key=samples.get)
