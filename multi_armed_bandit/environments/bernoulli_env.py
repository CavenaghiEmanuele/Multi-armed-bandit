from numpy.random import uniform, binomial
from typing import List, Dict

from . import Environment


class BernoulliEnvironment(Environment):

    _probabilities: Dict
    _best_action: str
    _state: int
    
    def __init__(self, actions: List[str], probabilities: Dict = None):

        super().__init__(actions)

        if probabilities == None:
            self._probabilities = {action: uniform(0, 1) for action in actions}
        elif probabilities != None and len(actions) == len(probabilities):
            self._probabilities = probabilities
        elif probabilities != None and len(actions) != len(probabilities):
            raise Exception(
                'Length of probabilities vector must be the same of number of arms')

        self._best_action = max(self._probabilities, key=self._probabilities.get)
        
    def do_action(self, action: str):
        return binomial(size=1, n=1, p= self._probabilities[action])[0]

    def get_state(self):
        # TODO
        self._state = 1
        return self._state
