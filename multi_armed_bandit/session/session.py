import pandas as pd

from multiprocessing import Pool, cpu_count
from typing import List
from tqdm import trange
from copy import deepcopy

from ..environments import Environment
from ..mab_agents import Agent, Oracle


class Session():

    _env: Environment
    _agents: List[Agent]

    def __init__(self, env: Environment, agents: List[Agent]):
        oracle = Oracle(env=env)
        self._env = env
        self._agents = agents
        self._agents.append(oracle)
        
    def _runner(self, env, agent, steps, experiment) -> None:
        results=[]
        for step in trange(steps):
            state = env.get_state()
            action = agent.select_action(state)
            reward = env.do_action(action)
            agent.update_estimates(state, action, reward)
            
            # statistics
            results.append({'agent': agent, 'experiment': experiment, 'step':step, 'reward':reward})
        return pd.DataFrame(results)

    def run(self, steps: int, experiments:int=1):

        params = [
            (deepcopy(self._env), agent, steps, experiment) 
            for agent in self._agents 
            for experiment in range(experiments)
            ]

        pool = Pool(cpu_count())
        data = pool.starmap(self._runner, params)
        pool.close()
        pool.join()

        return pd.concat(data, ignore_index=True) 
