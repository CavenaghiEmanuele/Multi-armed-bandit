import numpy as np
from typing import Dict

from . import BernoulliDynamicBandit


class BernoulliReplayBandit(BernoulliDynamicBandit):

    _action_value_trace: Dict
    _replay: Dict

    def __init__(self, replay: Dict):
        BernoulliDynamicBandit.__init__(self, n_arms=len(replay["probabilities"]), probabilities=replay["probabilities"])
        self._replay = replay

    def change_action_prob(self, step: int):
        try:
            for change in self._replay[step]:
                self._probabilities[change[0]] = change[1]
                self._best_action = np.argmax(self._probabilities)
        except:
            pass

        for action in range(self._n_arms):
            self._action_value_trace[action].append(self._probabilities[action])
