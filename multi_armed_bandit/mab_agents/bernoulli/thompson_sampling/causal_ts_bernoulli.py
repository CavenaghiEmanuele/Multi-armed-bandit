import random
from typing import List, Dict
from numpy.random import beta
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.inference import CausalInference

from multi_armed_bandit.utils.context import from_pd_series_to_str

from .bayesian_ts_bernoulli import BayesianTSBernoulli
from ....utils import from_dict_to_str


class CausalTSBernoulli(BayesianTSBernoulli):

    _inference_engine: CausalInference
    _adjustment_set: set

    def __init__(self, id:str, actions: List[str], contexts:Dict, bn:BayesianNetwork):
        super().__init__(id, actions, contexts, bn)
        self._inference_engine = CausalInference(self._bn)
        self._adjustment_set = self._inference_engine.get_minimal_adjustment_set('X', 'Y')
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            if self._by_feature == True:
                a_features = from_pd_series_to_str(self._actions_features.loc[a])
            else:
                a_features = a
            prob = self._inference_engine.query(
                variables=['Y'], 
                do={'X':a_features}, 
                evidence=context, 
                adjustment_set=self._adjustment_set, 
                show_progress=False
                ).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_str(context|{'X': a_features})] + 1
            samples.update({a:beta(a=prob*obs, b=(1-prob)*obs)})
        # randomly brake ties
        return random.choice([k for (k, v) in samples.items() if v == max(samples.values())])
