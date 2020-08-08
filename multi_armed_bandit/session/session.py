from typing import Dict, List
from tqdm import trange
from collections import defaultdict 


class Session():

    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

    def run(self, n_step: int):
        regret = []
        action_selection = {action:[0] for action in range(self._env.get_n_arms())}
        for step in trange(n_step):
            action = self._agent.select_action()
            reward = self._env.do_action(action)
            self._agent.update_estimates(action, reward)
            regret.append(self._env.best_action_mean() - self._env.action_mean(action))
            for a in range(self._env.get_n_arms()):
                if a == action:
                    action_selection[action].append(action_selection[action][-1]+1)
                else:
                    action_selection[a].append(action_selection[a][-1])
        
        return {"regret":regret, "action_selection":action_selection}
