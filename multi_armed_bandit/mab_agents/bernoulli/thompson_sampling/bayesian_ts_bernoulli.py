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
from pgmpy.inference import VariableElimination

from ...agent import Agent
from ....utils import from_dict_to_str


class BayesianTSBernoulli(Agent):

    _bn: BayesianNetwork
    _inference_engine: VariableElimination
    _n_observations: int
    _obs_for_context: defaultdict

    def __init__(self, id:str, actions: List[str], contexts:Dict, bn:BayesianNetwork):
        super().__init__(id, actions, contexts)
        self._bn = deepcopy(bn)
        self._init_uniform_cpds()
        self._n_observations = 1
        self._obs_for_context = defaultdict(int)
        self._inference_engine = VariableElimination(self._bn)

    def update_estimates(self, context:int, action: str, reward: int) -> None:
        self._bn.fit_update(DataFrame([context|{'X':action, 'Y':reward}]), n_prev_samples=self._n_observations)
        self._n_observations += 1
        self._obs_for_context[from_dict_to_str(context|{'X': action})] += 1
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            prob = self._inference_engine.query(variables=['Y'], evidence=context|{'X':a}).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_str(context|{'X': a})] + 1
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
        variables = self._contexts|{'X':self._actions, 'Y':[0,1]}

        for context in variables.keys():
            parents = list(self._bn.predecessors(context))

            if len(parents) == 0:
                values = np.ones((len(variables[context]), 1))
                values = values / np.sum(values, axis=0)
                self._bn.add_cpds(TabularCPD(
                    variable=context,
                    variable_card=len(variables[context]),
                    values=values,
                    state_names=variables,
                ))
            else:
                values = np.ones((len(variables[context]), np.product([len(variables[parent]) for parent in parents])))
                values = values / np.sum(values, axis=0)

                self._bn.add_cpds(TabularCPD(
                    variable=context,
                    variable_card=len(variables[context]),
                    values=values,
                    evidence=parents,
                    evidence_card=[len(variables[parent]) for parent in parents],
                    state_names=variables,
                ))
