import numpy as np
from typing import Dict
from copy import deepcopy

from . import BernoulliDynamicBandit


class BernoulliReplayBandit(BernoulliDynamicBandit):

    _replay: Dict

    def __init__(self, replay: Dict):
        self._replay = deepcopy(replay)
        BernoulliDynamicBandit.__init__(self, n_arms=len(replay["probabilities"]), probabilities=replay["probabilities"])

    def reset_to_start(self):
        self._probabilities = deepcopy(self._replay["probabilities"])
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
