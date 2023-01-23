import pandas as pd

from multiprocessing import Pool, cpu_count
from typing import List
from tqdm import trange
from copy import deepcopy

from ..environments import Environment
from ..mab_agents import Agent, Oracle, BayesianTSBernoulli
from ..utils import from_dict_to_str


class Session():

    _env: Environment
    _agents: List[Agent]

    def __init__(self, env: Environment, agents: List[Agent]):
        self._env = env
        self._agents = agents

    def _runner(self, env, agents, steps, experiment) -> None:
        oracle = Oracle(env)
        agents.append(oracle)

        results=[]
        for step in trange(steps):
            state = env.get_state()
            available_actions = env.get_available_actions()
            
            for agent in agents:
                action = agent.select_action(state, available_actions)
                reward = env.do_action(action)
                agent.update_estimates(state, action, reward)
                # statistics
                results.append({
                    'agent': repr(agent), 
                    'experiment': experiment, 
                    'step':step, 
                    'state':from_dict_to_str(state), 
                    'action':action, 
                    'reward':reward
                    })
            env.next_state()
        return pd.DataFrame(results)

    def run(self, steps: int, experiments:int=1):

        params = [
            (deepcopy(self._env), self._agents, steps, experiment) 
            for experiment in range(experiments)
            ]

        pool = Pool(cpu_count())
        data = pool.starmap(self._runner, params)
        pool.close()
        pool.join()

        return pd.concat(data, ignore_index=True) 
