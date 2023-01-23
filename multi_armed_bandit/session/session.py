import pandas as pd

from multiprocessing import Pool, cpu_count
from typing import List
from tqdm import trange

from ..environments import Environment
from ..mab_agents import Agent, Oracle
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
            context = env.get_context()
            available_actions = env.get_available_actions()
            
            for agent in agents:
                action = agent.select_action(context, available_actions)
                reward = env.do_action(action)
                agent.update_estimates(context, action, reward)
                # statistics
                results.append({
                    'agent': repr(agent), 
                    'experiment': experiment, 
                    'step':step, 
                    'context':from_dict_to_str(context), 
                    'action':action, 
                    'reward':reward
                    })
            env.next_context()
        return pd.DataFrame(results)

    def run(self, steps: int, experiments:int=1):

        params = [
            (self._env, self._agents, steps, experiment) 
            for experiment in range(experiments)
            ]

        pool = Pool(cpu_count())
        data = pool.starmap(self._runner, params)
        pool.close()
        pool.join()

        return pd.concat(data, ignore_index=True) 
