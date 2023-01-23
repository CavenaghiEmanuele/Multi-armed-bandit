from typing import List, Dict
from numpy.random import beta
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.inference import CausalInference

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
            prob = self._inference_engine.query(
                variables=['Y'], 
                do={'X':a}, 
                evidence=context, 
                adjustment_set=self._adjustment_set, 
                show_progress=False
                ).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_str(context|{'X': a})] + 1
            samples.update({a:beta(a=prob*obs, b=(1-prob)*obs)})
        return max(samples, key=samples.get)
