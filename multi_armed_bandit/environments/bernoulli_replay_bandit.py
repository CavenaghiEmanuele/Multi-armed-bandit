import numpy as np
from typing import Dict
from copy import deepcopy
from numpy.random import uniform

from . import BernoulliDynamicBandit


class BernoulliReplayBandit(BernoulliDynamicBandit):

    _replay: Dict

    def __init__(self, replay:Dict = None, n_step:int = None, n_arms:int = None, prob_of_change: float = 0.001, fixed_action_prob: float = None):
            
        # If replay is None generate automatic replay of n_steps
        if replay is None:
            if n_step is None or n_arms is None:
                raise Exception("Must pass n_step and n_arms")
            
            BernoulliDynamicBandit.__init__(self, 
                                            n_arms=n_arms,
                                            probabilities=[uniform(0, 1) for _ in range(n_arms)], 
                                            prob_of_change=prob_of_change, 
                                            fixed_action_prob=fixed_action_prob
                                            )
            self._replay = {'probabilities' : self._probabilities}
            for i in range(n_step):
                self._make_replay(step=i)
        else:
            self._replay = deepcopy(replay)
            BernoulliDynamicBandit.__init__(self, n_arms=len(self._replay['probabilities']), probabilities=self._replay['probabilities'])        

    def reset_to_start(self):
        self._probabilities = deepcopy(self._replay["probabilities"])
        self._best_action = np.argmax(self._probabilities)
        self._action_value_trace = {a:[self._probabilities[a]] for a in range(self._n_arms)}

    def change_action_prob(self, step: int):
        try:
            for change in self._replay[step]:
                self._probabilities[change[0]] = change[1]
                self._best_action = np.argmax(self._probabilities)
        except:
            pass

        for action in range(self._n_arms):
            self._action_value_trace[action].append(self._probabilities[action])
    
    def _make_replay(self, step: int):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                self._probabilities[action] = uniform(0, 1)
                try:
                    self._replay[step].append((action, self._probabilities[action]))
                except:
                    self._replay.update({step : [(action, self._probabilities[action])]})
            self._action_value_trace[action].append(self._probabilities[action])
