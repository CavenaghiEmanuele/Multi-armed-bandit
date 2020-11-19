import numpy as np
from typing import Dict
from copy import deepcopy

from . import GaussianDiscountedBandit


class GaussianReplayBandit(GaussianDiscountedBandit):

    _replay: Dict

    def __init__(self, replay: Dict):
        self._replay = deepcopy(replay)
        GaussianDiscountedBandit.__init__(self, n_arms=len(replay["mean"]), mean=replay["mean"], std_dev=replay["std_dev"])
        
    def reset_to_start(self):
        self._mean = deepcopy(self._replay["mean"])
        self._std_dev = deepcopy(self._replay["std_dev"])
        self._best_action = np.argmax(self._mean)
        self._action_value_trace = {a: [self._mean[a]] for a in range(self._n_arms)}

    def change_action_prob(self, step: int):
        try:
            for change in self._replay[step]:
                self._mean[change[0]] = change[1]
                self._best_action = np.argmax(self._mean)
        except:
            pass

        for action in range(self._n_arms):
            self._action_value_trace[action].append(self._mean[action])
