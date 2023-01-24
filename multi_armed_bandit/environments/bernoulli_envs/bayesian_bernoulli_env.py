import networkx as nx
import pylab as plt
import pandas as pd
import itertools

from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from numpy.random import binomial
from typing import Dict, List

from .. import Environment
from ...utils import from_dict_to_str


class BayesianBernoulliEnvironment(Environment):

    _context: Dict
    _bn: BayesianNetwork

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
            acts_feats_domain = {col:actions[col].unique() for col in actions}
            self._actions = [
                from_dict_to_str(action)
                for action in [dict(zip(acts_feats_domain.keys(),items)) for items in itertools.product(*acts_feats_domain.values())]
            ]
        self._bn = bn
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
        del tmp['X']
        del tmp['Y']
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


    def _create_cpds(self) -> None:
        variables = self._contexts|{'X':self._actions, 'Y':['0','1']}
        for context in variables.keys():
            self._bn.add_cpds(TabularCPD.get_random(
                variable=context,
                evidence=list(self._bn.predecessors(context)), #parents of the node 
                cardinality={context:len(variables[context]) for context in variables},
                state_names=variables
                )
            )

    def _get_reward_probability(self, action) -> float:
        y_cpd = self._bn.get_cpds(node='Y')
        bn_context = {var:self._context[var] for var in y_cpd.get_evidence() if var!='X'}
        bn_context.update({'X':action, 'Y':1})
        return y_cpd.get_value(**bn_context)
