import numpy as np
import networkx as nx
import pylab as plt

from typing import List, Dict
from pandas import DataFrame
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

from ...agent import Agent


class BayesianTSBernoulli(Agent):

    _bn: BayesianNetwork
    _n_observations: int

    def __init__(self, id:str, actions: List[str], states:Dict, bn:BayesianNetwork):
        super().__init__(id, actions, states)
        self.reset_agent(bn)

    def reset_agent(self, bn:BayesianNetwork) -> None:
        self._bn = bn
        self._init_uniform_cpds()
        self._n_observations = 0

        self.print_cpds()

    def update_estimates(self, state:int, action: str, reward: int) -> None:
        observation = {}
        observation.update(state)
        observation.update({'X':action, 'Y':reward})

        self._bn.fit_update(DataFrame(observation), n_prev_samples=self._n_observations)
        self._n_observations += 1
 
    def select_action(self, state:int, available_actions:List[str]) -> str:
        #TODO
        '''samples = {a:beta(
            a=self._parameters[from_dict_to_str(state)][a][0], 
            b=self._parameters[from_dict_to_str(state)][a][1]
            )
            for a in available_actions}
        return max(samples, key=samples.get)'''
        raise NotImplementedError

    
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
        variables = {}
        variables.update(self._states)
        variables.update({'X':self._actions, 'Y':['0','1']})

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
