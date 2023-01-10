import itertools

from numpy.random import uniform, binomial, choice
from typing import List, Dict

from .. import Environment
from ...utils import from_dict_to_str


class PlainBernoulliEnvironment(Environment):

    _state: Dict
    _reward_function: Dict
    
    def __init__(self, actions:List[str], states:Dict, available_actions:Dict=None):
        super().__init__(actions, states, available_actions)
        self.next_state()
        self._reward_function = { 
            from_dict_to_str(state) : {action: uniform(0, 1) for action in self.get_available_actions(state)}
            for state in [dict(zip(states.keys(),items)) for items in itertools.product(*states.values())]
        }

    def do_action(self, action: str) -> int:
        if action not in self.get_available_actions():
            raise Exception('Action is not avaible')
        return binomial(n=1, p=self._reward_function[from_dict_to_str(self._state)][action])

    def get_best_action(self) -> str:
        prob_current_state = self._reward_function[from_dict_to_str(self._state)]
        return max(prob_current_state, key=prob_current_state.get)

    def next_state(self) -> None:
        self._state = {dim:choice(self._states[dim]) for dim in self._states.keys()}
