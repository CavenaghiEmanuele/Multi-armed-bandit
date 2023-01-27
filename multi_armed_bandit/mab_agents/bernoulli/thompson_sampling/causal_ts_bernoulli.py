import random

from typing import List, Dict
from numpy.random import beta
from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.inference import CausalInference

from .bayesian_ts_bernoulli import BayesianTSBernoulli
from ....utils import from_dict_to_json


class CausalTSBernoulli(BayesianTSBernoulli):

    _inference_engine: CausalInference

    def __init__(self, id:str, actions: List[str], contexts:Dict, bn:BayesianNetwork, update_delay:int=1):
        super().__init__(id, actions, contexts, bn, update_delay)
        self._inference_engine = CausalInference(self._bn)
 
    def select_action(self, context:int, available_actions:List[str]) -> str:
        samples = {}
        for a in available_actions:
            if self._by_feature:
                action = self._actions.loc[a].to_dict()
            else:
                action = {'X':a}

            prob = self._inference_engine.query(
                variables=['Y'], 
                do=action, 
                evidence=context, 
                show_progress=False
                ).get_value(Y=1)
            obs = self._obs_for_context[from_dict_to_json(context|action)] + 1
            samples.update({a:beta(a=prob*obs, b=(1-prob)*obs)})
        # randomly brake ties
        return random.choice([k for (k, v) in samples.items() if v == max(samples.values())])
