import numpy as np
from typing import Dict, List
from copy import deepcopy
from numpy.random import uniform, randint

from . import BernoulliDynamicBandit


class BernoulliReplayBandit(BernoulliDynamicBandit):

    _replay: Dict
    _arm_change_lock: Dict
    _start_probabilities: List

    def __init__(self, replay:Dict = None, n_step:int = None, n_arms:int = None, prob_of_change: float = 0.001, fixed_action_prob: float = None, type_change:str='abrupt'):
            
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
            self._start_probabilities = deepcopy(self._probabilities)
            self._arm_change_lock = {action : {'lock':False, 'step_size':0.0, 'remaining_steps':0} 
                                     for action in range(n_arms)}
            for i in range(n_step):
                self._make_replay(step=i, type=type_change)
        else:
            self._replay = deepcopy(replay)
            self._start_probabilities = deepcopy(self._replay['probabilities'])
            BernoulliDynamicBandit.__init__(self, n_arms=len(self._replay['probabilities']), probabilities=self._replay['probabilities'])        

    def reset_to_start(self):
        self._probabilities = deepcopy(self._start_probabilities)
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
    
    def _make_replay(self, step: int, type:str):
        if type == 'abrupt':
            self._make_replay_abrupt(step)
        elif type == 'incremental':
            self._make_replay_incremental(step)
    
    def _make_replay_abrupt(self, step: int):
        for action in range(self._n_arms):
            if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                self._probabilities[action] = uniform(0, 1)
                try:
                    self._replay[step].append((action, self._probabilities[action]))
                except:
                    self._replay.update({step : [(action, self._probabilities[action])]})
            self._action_value_trace[action].append(self._probabilities[action])
            self._best_action = np.argmax(self._probabilities)
    
    def _make_replay_incremental(self, step: int):
        for action in range(self._n_arms):
            if not self._arm_change_lock[action]['lock']:            
                if (not action in self._fixed_actions) and (uniform(0, 1) < self._prob_of_change):
                    new_prob = uniform(0, 1)
                    steps_to_change = randint(100, 1000)    
                    self._arm_change_lock[action] = {
                        'lock':True, 
                        'step_size': (new_prob-self._probabilities[action]) / steps_to_change, 
                        'remaining_steps':steps_to_change-1}

                    self._probabilities[action] += self._arm_change_lock[action]['step_size']
                    
                    try:
                        self._replay[step].append((action, self._probabilities[action]))
                    except:
                        self._replay.update({step : [(action, self._probabilities[action])]})
                self._action_value_trace[action].append(self._probabilities[action])
                self._best_action = np.argmax(self._probabilities)
                
            else:
                self._probabilities[action] += self._arm_change_lock[action]['step_size']
                self._arm_change_lock[action]['remaining_steps'] -= 1
                if self._arm_change_lock[action]['remaining_steps'] == 0:
                    self._arm_change_lock[action] = {'lock':False, 'step_size':0.0, 'remaining_steps':0}
                                
                try:
                    self._replay[step].append((action, self._probabilities[action]))
                except:
                    self._replay.update({step : [(action, self._probabilities[action])]})
                self._action_value_trace[action].append(self._probabilities[action])
                self._best_action = np.argmax(self._probabilities)
