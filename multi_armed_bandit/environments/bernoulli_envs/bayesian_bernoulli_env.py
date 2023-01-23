import networkx as nx
import pylab as plt

from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from numpy.random import binomial
from typing import Dict, List

from .. import Environment


class BayesianBernoulliEnvironment(Environment):

    _state: Dict
    _bn: BayesianNetwork

    def __init__(
        self, 
        actions: List[str], 
        states: Dict, 
        available_actions: Dict = None, 
        bn: BayesianNetwork = None
        ):

        super().__init__(actions, states, available_actions)
        self._bn = bn
        if bn.get_cpds() == []:
            self._create_cpds()
        self.next_state()

    def do_action(self, action: str):
        if action not in self.get_available_actions():
            raise Exception('Action is not avaible')
        return binomial(n=1, p=self._get_reward_probability(action))

    def get_best_action(self) -> str:
        prob_current_state = {
            action:self._get_reward_probability(action) 
            for action in self.get_available_actions()
            }
        return max(prob_current_state, key=prob_current_state.get)

    def next_state(self) -> None:
        tmp = self._bn.simulate(n_samples=1, show_progress=False).to_dict('list')
        del tmp['X']
        del tmp['Y']
        self._state = {key:value[0] for key, value in tmp.items()}


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
        variables = self._states|{'X':self._actions, 'Y':['0','1']}
        for state in variables.keys():
            self._bn.add_cpds(TabularCPD.get_random(
                variable=state,
                evidence=list(self._bn.predecessors(state)), #parents of the node 
                cardinality={state:len(variables[state]) for state in variables},
                state_names=variables
                )
            )

    def _get_reward_probability(self, action) -> float:
        y_cpd = self._bn.get_cpds(node='Y')
        bn_state = {var:self._state[var] for var in y_cpd.get_evidence() if var!='X'}
        bn_state.update({'X':action, 'Y':1})
        return y_cpd.get_value(**bn_state)
