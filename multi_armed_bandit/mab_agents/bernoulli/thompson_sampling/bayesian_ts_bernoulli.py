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
from ....utils import from_dict_to_json, from_pd_series_to_json


class BayesianTSBernoulli(Agent):

    _bn: BayesianNetwork
    _inference_engine: VariableElimination
    _n_observations: int
    _obs_for_context: defaultdict
    _by_feature: bool
    _stored_obs: pd.DataFrame
    _update_delay: int

    '''
    actions: List or Pandas DataFrame. List of actions or DataFrame where each row is an action 
    and columns are the values of the associated features
    '''
    def __init__(self, id:str, actions: List[str], contexts:Dict, bn:BayesianNetwork, update_delay:int=1):
        super().__init__(id, actions, contexts)
        if isinstance(actions, List):
            self._by_feature = False
        if isinstance(actions, pd.DataFrame):
            self._by_feature = True

        self._bn = deepcopy(bn)
        self._init_uniform_cpds()
        self._n_observations = 1
        self._obs_for_context = defaultdict(int)
        self._inference_engine = VariableElimination(self._bn)
        self._stored_obs = DataFrame()
        self._update_delay = update_delay

    def update_estimates(self, context:int, action: str, reward: int) -> None:
        # If there are nodes to represent actions'features the actions are represented by their features
        if self._by_feature:
            extended_context = context|self._actions.loc[action].to_dict()
        else:
            extended_context = context|{'X':action}

        # Update number of observations
        self._n_observations += 1
        # Add number of observations for each context-action pair
        self._obs_for_context[from_dict_to_json(extended_context)] += 1
        # Extend context with reward and make the dataframe to update estimates
        extended_context = extended_context|{'Y':reward}
        self._stored_obs = pd.concat([self._stored_obs, DataFrame([extended_context])], ignore_index=True)        
    
        # Update estimates if we collected enought observations
        if len(self._stored_obs.index) >= self._update_delay:
            # Update bn
            self._bn.fit_update(
                self._stored_obs, 
                n_prev_samples=self._n_observations-self._update_delay
                )
            self._stored_obs = DataFrame()
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            if self._by_feature:
                extended_context = context|self._actions.loc[a].to_dict()
            else:
                extended_context = context|a
            prob = self._inference_engine.query(variables=['Y'], evidence=extended_context).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_json(extended_context)] + 1
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
        variables = self._contexts|{'Y':[0,1]}
        if self._by_feature:
            variables = variables|{col:list(self._actions[col].unique()) for col in self._actions}
        else:
            variables = variables|{'X':self._actions}
            
        for node in variables.keys():
            parents = list(self._bn.predecessors(node))

            if len(parents) == 0:
                values = np.ones((len(variables[node]), 1))
                values = values / np.sum(values, axis=0)
                self._bn.add_cpds(TabularCPD(
                    variable=node,
                    variable_card=len(variables[node]),
                    values=values,
                    state_names=variables,
                ))
            else:
                values = np.ones((len(variables[node]), np.product([len(variables[parent]) for parent in parents])))
                values = values / np.sum(values, axis=0)

                self._bn.add_cpds(TabularCPD(
                    variable=node,
                    variable_card=len(variables[node]),
                    values=values,
                    evidence=parents,
                    evidence_card=[len(variables[parent]) for parent in parents],
                    state_names=variables,
                ))
