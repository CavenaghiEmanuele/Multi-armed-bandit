import itertools
import random
import numpy as np
import networkx as nx
import pylab as plt
import pandas as pd

from typing import List, Dict
from collections import defaultdict
from copy import deepcopy
from pandas import DataFrame
from numpy.random import beta
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination

from ...agent import Agent
from ....utils import from_dict_to_str, from_pd_series_to_str


class BayesianTSBernoulli(Agent):

    _bn: BayesianNetwork
    _inference_engine: VariableElimination
    _n_observations: int
    _obs_for_context: defaultdict
    _by_feature: bool
    _actions_features: pd.DataFrame

    '''
    actions: List or Pandas DataFrame. List of actions or DataFrame where each row is an action 
    and columns are the values of the associated features
    '''
    def __init__(self, id:str, actions: List[str], contexts:Dict, bn:BayesianNetwork):
        super().__init__(id, actions, contexts)
        if isinstance(actions, List):
            self._by_feature = False
        if isinstance(actions, pd.DataFrame):
            self._by_feature = True
            self._actions_features = actions
            acts_feats_domain = {col:actions[col].unique() for col in actions}
            self._actions = [
                from_dict_to_str(action)
                for action in [dict(zip(acts_feats_domain.keys(),items)) for items in itertools.product(*acts_feats_domain.values())]
            ]
        self._bn = deepcopy(bn)
        self._init_uniform_cpds()
        self._n_observations = 1
        self._obs_for_context = defaultdict(int)
        self._inference_engine = VariableElimination(self._bn)

    def update_estimates(self, context:int, action: str, reward: int) -> None:
        if self._by_feature == True:
            action = from_pd_series_to_str(self._actions_features.loc[action])
        self._bn.fit_update(DataFrame([context|{'X':action, 'Y':reward}]), n_prev_samples=self._n_observations)
        self._n_observations += 1
        self._obs_for_context[from_dict_to_str(context|{'X': action})] += 1
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            if self._by_feature == True:
                a_features = from_pd_series_to_str(self._actions_features.loc[a])
            else:
                a_features = a
            prob = self._inference_engine.query(variables=['Y'], evidence=context|{'X':a_features}).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_str(context|{'X': a_features})] + 1
            samples.update({a:beta(a=prob*obs, b=(1-prob)*obs)})
        # randomly brake ties
        return random.choice([k for (k, v) in samples.items() if v == max(samples.values())])

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
