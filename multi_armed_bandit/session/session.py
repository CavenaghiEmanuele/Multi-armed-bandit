from typing import Dict, List
from tqdm import trange


class Session():

    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

    def run(self, n_step: int):
        regret = []
        for step in trange(n_step):
            action = self._agent.select_action()
            reward = self._env.do_action(action)
            self._agent.update_betas(action, reward)
            regret.append(self._env.best_action_mean() - self._env.action_mean(action))
        
        return regret
