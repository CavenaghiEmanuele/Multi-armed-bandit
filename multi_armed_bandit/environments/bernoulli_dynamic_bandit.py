import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, randint
from typing import List, Dict

from . import BernoulliBandit, DynamicMultiArmedBandit


class BernoulliDynamicBandit(DynamicMultiArmedBandit, BernoulliBandit):
    
    _arm_change_lock: Dict

    def __init__(self, n_arms: int, probabilities: List[float] = None, 
                 prob_of_change: float = 0.001, fixed_action_prob: float = None, type_change:str='abrupt'):
        
        DynamicMultiArmedBandit.__init__(self, 
                                         n_arms=n_arms, 
                                         prob_of_change=prob_of_change, 
                                         fixed_action_prob=fixed_action_prob, 
                                         type_change=type_change)
        BernoulliBandit.__init__(self, n_arms=n_arms, probabilities=probabilities)
        
        self._action_value_trace = {a:[self._probabilities[a]] for a in range(n_arms)}
        self._arm_change_lock = {action : {'lock':False, 'step_size':0.0, 'remaining_steps':0} 
                                     for action in range(n_arms)}

    def plot_arms(self, render: bool = True):
        plt.figure()
        for a in range(self._n_arms):
            plt.plot(self._action_value_trace[a], label="Action: " +
                        str(a) + ", prob: " + str(self._probabilities[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def change_action_prob(self, step: int):
        if self._type_change == 'abrupt':
            self._change_action_prob_abrupt()
        elif self._type_change == 'incremental':
            self._change_action_prob_incremental()

    def _change_action_prob_abrupt(self):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                self._probabilities[action] = uniform(0, 1)
                self._best_action = np.argmax(self._probabilities)
            self._action_value_trace[action].append(self._probabilities[action])

    def _change_action_prob_incremental(self):
        for action in range(self._n_arms):
            if not self._arm_change_lock[action]['lock']:
                if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                    new_prob = uniform(0, 1)
                    steps_to_change = randint(20, 100)
                    self._arm_change_lock[action] = {
                        'lock':True,
                        'step_size': (new_prob-self._probabilities[action]) / steps_to_change,
                        'remaining_steps':steps_to_change-1}

                    self._probabilities[action] += self._arm_change_lock[action]['step_size']
                self._action_value_trace[action].append(self._probabilities[action])
                self._best_action = np.argmax(self._probabilities)

            else:
                self._probabilities[action] += self._arm_change_lock[action]['step_size']
                self._arm_change_lock[action]['remaining_steps'] -= 1
                if self._arm_change_lock[action]['remaining_steps'] == 0:
                    self._arm_change_lock[action] = {'lock':False, 'step_size':0.0, 'remaining_steps':0}
                self._action_value_trace[action].append(self._probabilities[action])
                self._best_action = np.argmax(self._probabilities)
