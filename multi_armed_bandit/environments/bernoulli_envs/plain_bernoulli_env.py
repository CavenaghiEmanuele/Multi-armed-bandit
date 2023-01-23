import itertools

from numpy.random import uniform, binomial, choice
from typing import List, Dict

from .. import Environment
from ...utils import from_dict_to_str


class PlainBernoulliEnvironment(Environment):

    _context: Dict
    _reward_function: Dict
    
    def __init__(self, actions:List[str], contexts:Dict, available_actions:Dict=None):
        super().__init__(actions, contexts, available_actions)
        self.next_context()
        self._reward_function = { 
            from_dict_to_str(context) : {action: uniform(0, 1) for action in self.get_available_actions(context)}
            for context in [dict(zip(contexts.keys(),items)) for items in itertools.product(*contexts.values())]
        }

    def do_action(self, action: str) -> int:
        if action not in self.get_available_actions():
            raise Exception('Action is not avaible')
        return binomial(n=1, p=self._reward_function[from_dict_to_str(self._context)][action])

    def get_best_action(self) -> str:
        context_probs = self._reward_function[from_dict_to_str(self._context)]
        return max(context_probs, key=context_probs.get)

    def next_context(self) -> None:
        self._context = {dim:choice(self._contexts[dim]) for dim in self._contexts.keys()}
