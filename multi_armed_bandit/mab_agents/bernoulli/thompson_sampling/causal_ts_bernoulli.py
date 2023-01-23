import numpy as np
import networkx as nx
import pylab as plt

from typing import List, Dict
from collections import defaultdict
from copy import deepcopy
from pandas import DataFrame
from numpy.random import beta
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import CausalInference

from ...agent import Agent
from ....utils import from_dict_to_str


class CausalTSBernoulli(Agent):

    _bn: BayesianNetwork
    _inference_engine: CausalInference
    _n_observations: int
    _obs_for_context: defaultdict
    _adjustment_set: set

    def __init__(self, id:str, actions: List[str], states:Dict, bn:BayesianNetwork):
        super().__init__(id, actions, states)
        self._bn = deepcopy(bn)
        self._init_uniform_cpds()
        self._n_observations = 1
        self._obs_for_context = defaultdict(int)
        self._inference_engine = CausalInference(self._bn)
        self._adjustment_set = self._inference_engine.get_minimal_adjustment_set('X', 'Y')

    def update_estimates(self, state:int, action: str, reward: int) -> None:
        self._bn.fit_update(DataFrame([state|{'X':action, 'Y':reward}]), n_prev_samples=self._n_observations)
        self._n_observations += 1
        self._obs_for_context[from_dict_to_str(state|{'X': action})] += 1
 
    def select_action(self, state:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            prob = self._inference_engine.query(
                variables=['Y'], 
                do={'X':a}, 
                evidence=state, 
                adjustment_set=self._adjustment_set, 
                show_progress=False
                ).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_str(state|{'X': a})] + 1
            samples.update({a:beta(a=prob*obs, b=(1-prob)*obs)})
        return max(samples, key=samples.get)


    def plot_bn(self) -> None:
        pos = nx.circular_layout(self._bn)
        nx.draw(self._bn, node_color='#00b4d9', pos=pos, with_labels=True)
        plt.show()

    def print_cpds(self, node:str=None) -> None:
        if node != None:
            print(self._bn.get_cpds(node))
            return
        for cpd in self._bn.get_cpds():
            print(cpd)


    def _init_uniform_cpds(self):
        variables = self._states|{'X':self._actions, 'Y':[0,1]}

        for state in variables.keys():
            parents = list(self._bn.predecessors(state))

            if len(parents) == 0:
                values = np.ones((len(variables[state]), 1))
                values = values / np.sum(values, axis=0)
                self._bn.add_cpds(TabularCPD(
                    variable=state,
                    variable_card=len(variables[state]),
                    values=values,
                    state_names=variables,
                ))
            else:
                values = np.ones((len(variables[state]), np.product([len(variables[parent]) for parent in parents])))
                values = values / np.sum(values, axis=0)

                self._bn.add_cpds(TabularCPD(
                    variable=state,
                    variable_card=len(variables[state]),
                    values=values,
                    evidence=parents,
                    evidence_card=[len(variables[parent]) for parent in parents],
                    state_names=variables,
                ))
