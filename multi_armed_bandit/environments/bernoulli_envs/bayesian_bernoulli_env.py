import itertools
import networkx as nx
import pylab as plt
import pandas as pd
import numpy as np

from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from numpy.random import binomial
from typing import Dict, List
from copy import deepcopy

from .. import Environment
from ...utils import from_dict_to_json, from_pd_series_to_json


class BayesianBernoulliEnvironment(Environment):

    _context: Dict
    _bn: BayesianNetwork
    _actions_features: pd.DataFrame
    _by_feature: bool

    '''
    actions: List or Pandas DataFrame. List of actions or DataFrame where each row is an action 
    and columns are the values of the associated features
    '''
    def __init__(
        self, 
        actions: List[str], 
        contexts: Dict, 
        available_actions: Dict = None, 
        bn: BayesianNetwork = None
        ):

        super().__init__(actions, contexts, available_actions)
        if isinstance(actions, List):
            self._by_feature = False
        if isinstance(actions, pd.DataFrame):
            self._by_feature = True
            self._actions_features = actions
            self._actions = list(actions.index.values)
        self._bn = deepcopy(bn)
        if bn.get_cpds() == []:
            self._create_cpds()
        self.next_context()

    def do_action(self, action: str):
        if action not in self.get_available_actions():
            raise Exception('Action is not avaible')
        return binomial(n=1, p=self._get_reward_probability(action))

    def get_best_action(self) -> str:
        context_probs = {
            action:self._get_reward_probability(action) 
            for action in self.get_available_actions()
            }
        return max(context_probs, key=context_probs.get)

    def next_context(self) -> None:
        tmp = self._bn.simulate(n_samples=1, show_progress=False).to_dict('list')
        # Remove action and outcome as not part of the context
        del tmp['X'], tmp['Y']
        self._context = {key:value[0] for key, value in tmp.items()}


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


    def _create_cpds(self, min_reward:float=0, max_reward:float=1) -> None:
        variables = self._contexts|{'X':self._actions, 'Y':['0','1']}
        # For each node add a random cpd
        for node in variables.keys():
            self._bn.add_cpds(TabularCPD.get_random(
                variable=node,
                evidence=list(self._bn.predecessors(node)), #parents of the node 
                cardinality={node:len(variables[node]) for node in variables},
                state_names=variables
                )
            )

        # If the actions are described by features we have to generate the cpds for node Y
        # in a way that the reward is a combination of the actions'features as described by the bn
        if self._by_feature:
            parents = list(self._bn.predecessors('Y'))
            acts_feats_domain = {col:self._actions_features[col].unique() for col in self._actions_features}
            # Generate the probabilities (for every context) for every combination of the actions' features
            actions_by_features = {
                from_dict_to_json(action):np.random.rand(np.product([len(variables[parent]) for parent in parents if parent != 'X']))
                for action in [dict(zip(acts_feats_domain.keys(),items)) for items in itertools.product(*acts_feats_domain.values())]
            }
            # Extend the probabilities for each single action
            values = [
                list(actions_by_features[from_pd_series_to_json(self._actions_features.loc[action])])
                for action in self._actions
            ]
            # Flattening the list of lists
            values = np.array([item for sublist in values for item in sublist])
            # Create a 2-dim np.array by adding the 1-p probabilities
            values = np.array([values, np.ones(np.product([len(variables[parent]) for parent in parents])) - values])
            # Create the cpd
            self._bn.add_cpds(TabularCPD(
                variable='Y',
                variable_card=len(['0', '1']),
                values=values,
                evidence=parents,
                evidence_card=[len(variables[parent]) for parent in parents],
                state_names=variables
            ))

    def _get_reward_probability(self, action) -> float:
        y_cpd = self._bn.get_cpds(node='Y')
        extended_context = self._context|{'X':action}
        # If there are nodes to represent actions'features add them to the extended context
        if self._by_feature:
            extended_context = extended_context|self._actions_features.loc[action].to_dict()
        # Extract only the parents of the Y node 
        bn_context = {var:extended_context[var] for var in y_cpd.get_evidence()}|{'Y':1}
        return y_cpd.get_value(**bn_context)
