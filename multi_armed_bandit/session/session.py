import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List
from tqdm import trange

from ..environments import MultiArmedBandit, DiscountedMultiArmedBandit
from ..algorithms import Algorithm


class Session():

    _regret: Dict
    _action_selection_trace: Dict
    _action_selection: Dict
    _real_reward_trace: Dict
    _real_reward: Dict
    _env: MultiArmedBandit
    _agents: List[Algorithm]

    def __init__(self, env: MultiArmedBandit, agent: List[Algorithm]):
        self._env = env
        if isinstance(agent, List):
            self._agents = agent
        else:
            self._agents = [agent]

    def run(self, n_step: int, n_test: int = 1, use_replay: bool = False) -> None:
        
        # Save statistics
        self._regrets = {agent: np.zeros(n_step) for agent in self._agents}

        self._real_reward_trace = {agent: np.zeros(n_step) for agent in self._agents}
        self._real_reward_trace.update({"Oracle": np.zeros(n_step)})

        self._action_selection_trace = {agent: {a:np.zeros(n_step) for a in range(self._env.get_n_arms())} for agent in self._agents}
        
        for test in trange(n_test):
            self._real_reward = {agent: 0 for agent in self._agents}
            self._real_reward.update({"Oracle": 0})
            self._action_selection = {agent: {a : 0 for a in range(self._env.get_n_arms())} for agent in self._agents}
            
            for step in range(n_step):
                # Oracle
                self._real_reward["Oracle"] += self._env.action_mean(self._env.get_best_action())
                self._real_reward_trace["Oracle"][step] += (1/(test+1)) * (self._real_reward["Oracle"] - self._real_reward_trace["Oracle"][step])
                
                for agent in self._agents:
                    action = agent.select_action()
                    reward = self._env.do_action(action)
                    agent.update_estimates(action, reward)    
                    self._update_statistic(test=test, step=step, id_agent=agent, action=action)

                if isinstance(self._env, DiscountedMultiArmedBandit):
                    self._env.change_action_prob(step=step)
            
            # Reset env and agent to start condition, then the changes will be those stored in the replay saved inside the env
            if (test < n_test-1) and use_replay:
                self._env.reset_to_start()
                for agent in self._agents:
                    agent.reset_agent()


    def _update_statistic(self, test, step, id_agent, action):
        reward = self._env.action_mean(action)

        self._regrets[id_agent][step] += (1/(test+1)) * (self._env.best_action_mean() - reward - self._regrets[id_agent][step])

        self._real_reward[id_agent] += reward
        self._real_reward_trace[id_agent][step] += (1/(test+1)) * (self._real_reward[id_agent] - self._real_reward_trace[id_agent][step])
        
        for a in range(self._env.get_n_arms()):
            if a == action:
                self._action_selection[id_agent][action] += 1
                self._action_selection_trace[id_agent][action][step] += (1/(test+1)) * (self._action_selection[id_agent][action] - self._action_selection_trace[id_agent][action][step])
            else:
                self._action_selection_trace[id_agent][a][step] += (1/(test+1)) * (self._action_selection[id_agent][a] - self._action_selection_trace[id_agent][a][step])
        


    def plot_regret(self, render: bool=True):
        plt.figure()
        for agent in self._agents:
            plt.plot(self._regrets[agent], label=agent)
        plt.suptitle("Regret")
        plt.legend()
        if render:
            plt.show()
    
    def plot_action_selection_trace(self, render: bool=True):
        if len(self._agents) > 1:
            fig, axs = plt.subplots(len(self._agents), sharey=True)
            for i, agent in enumerate(self._agents):
                for action in range(self._env.get_n_arms()):
                    axs[i].plot(self._action_selection_trace[agent][action], label="Action: " + str(action))
                    axs[i].set_title("Action selection: " + str(agent))
                    axs[i].legend()
                fig.suptitle("Action selection")
        else:
            fig = plt.figure()
            agent = self._agents[0]
            for action in range(self._env.get_n_arms()):
                plt.plot(self._action_selection_trace[agent][action], label="Action: " + str(action))
                plt.legend()
            plt.suptitle("Action selection")
        
        if render:
            plt.show()
    
    def plot_real_reward_trace(self, render: bool=True):
        plt.figure()
        plt.plot(self._real_reward_trace["Oracle"], label="Oracle")
        for agent in self._agents:
            plt.plot(self._real_reward_trace[agent], label=agent)
        plt.suptitle("Real rewards sum")
        plt.legend()
        if render:
            plt.show()

    def plot_all(self, render: bool=True):
        self.plot_regret(render=False)
        self.plot_action_selection_trace(render=False)
        self.plot_real_reward_trace(render=False)
        if render:
            plt.show()

    def get_reward_sum(self, agent: Algorithm):
        if agent == "Oracle":
            return self._real_reward_trace["Oracle"][-1]
        return self._real_reward_trace[agent][-1]
