import matplotlib.pyplot as plt

from typing import Dict, List
from tqdm import trange

from ..environments import MultiArmedBandit, DynamicMultiArmedBandit
from ..algorithms import Algorithm


class Session():

    _regret: Dict
    _action_selection: Dict
    _real_rewards_sum: Dict
    _env: MultiArmedBandit
    _agents: List[Algorithm]

    def __init__(self, env, agent: List):
        self._env = env
        if isinstance(agent, List):
            self._agents = agent
        else:
            self._agents = [agent]

        # Save statistics
        self._regrets = {agent.get_id():[] for agent in self._agents}
        self._real_rewards_sum = {agent.get_id():[0] for agent in self._agents}
        self._action_selection = {agent.get_id(): {a:[0] for a in range(self._env.get_n_arms())} for agent in self._agents}

    def run(self, n_step: int) -> None:
        for step in trange(n_step):
            for agent in self._agents:
                action = agent.select_action()
                reward = self._env.do_action(action)
                agent.update_estimates(action, reward)
                
                # Update statistics
                self._regrets[agent.get_id()].append(self._env.best_action_mean() - self._env.action_mean(action))
                self._real_rewards_sum[agent.get_id()].append(self._env.action_mean(action) + self._real_rewards_sum[agent.get_id()][-1])
                for a in range(self._env.get_n_arms()):
                    if a == action:
                        self._action_selection[agent.get_id()][action].append(self._action_selection[agent.get_id()][action][-1]+1)
                    else:
                        self._action_selection[agent.get_id()][a].append(self._action_selection[agent.get_id()][a][-1])

            if isinstance(self._env, DynamicMultiArmedBandit):
                self._env.change_action_prob()

    def plot_regret(self, render: bool=True):
        plt.figure()
        for agent in self._agents:
            plt.plot(self._regrets[agent.get_id()], label=agent)
        plt.suptitle("Regret")
        plt.legend()
        if render:
            plt.show()
    
    def plot_action_selection(self, render: bool=True):
        fig, axs = plt.subplots(len(self._agents))
        for i, agent in enumerate(self._agents):
            for action in range(self._env.get_n_arms()):
                axs[i].plot(self._action_selection[agent.get_id()][action], label="Action: " + str(action))
                axs[i].set_title("Action selection: " + str(agent))
                axs[i].legend()
   
        fig.suptitle("Action selection")
        if render:
            plt.show()
    
    def plot_real_rewards_sum(self, render: bool=True):
        plt.figure()
        for agent in self._agents:
            plt.plot(self._real_rewards_sum[agent.get_id()], label=agent)
        plt.suptitle("Real rewards sum")
        plt.legend()
        if render:
            plt.show()
