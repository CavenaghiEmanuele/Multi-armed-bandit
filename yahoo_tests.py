import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from numpy import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange

import multi_armed_bandit as mab



class YahooSession():

    _dataset: pd.DataFrame
    _clusters: pd.DataFrame
    _n_step: int
    _n_test: int
    _n_arms: int
    _compression: int

    def __init__(self, day:int, n_arms:int, n_test:int=1) -> None:

        path = 'datasets/yahoo_r6/'
        self._dataset = pd.read_csv(path + 'day' + str(day-1) + '.csv')
        self._clusters = pd.read_csv(path + 'clusters_6.csv')
        self._n_step = len(self._dataset.index)
        self._n_test = n_test
        self._n_arms = n_arms

    def select_cluster(self, step:int):
        row = self._dataset.loc[step]
        click = self._dataset.loc[step]['click']
        cluster = self._clusters.loc[self._clusters['id_article']==row['id_article']]['cluster'].values[0]        
        return cluster, click

    def plot_reward_trace(self, results):
        traces = {str(agent): np.zeros(len(results[0]['random'])) for agent in results[0]}

        for result in results:
            for agent in result:
                traces[str(agent)] = traces[str(agent)] + result[agent]    
        
        plt.figure()
        for agent in traces:
            plt.plot(traces[agent], label=agent)
        plt.suptitle("Rewards trace")
        plt.legend()
        plt.show()

        return

    def save_reward_trace_to_csv(self, results, day) -> None:
        dataset = pd.DataFrame()
        for i in range(len(results)):            
            dataset = pd.concat([dataset, pd.DataFrame.from_dict(results[i])], axis=1)
        dataset.to_csv("results/Yahoo/reward_trace_day" + str(day) + ".csv", index=False)
    
    def save_reward_perc_to_csv(self, results, day) -> None:
        
        tmp = {str(agent):[] for agent in results[0]}
        for i in range(len(results)):
            for key, value in results[i].items():
                tmp[str(key)].append(value)
                   
        dataset = pd.DataFrame.from_dict(tmp)        
        dataset.to_csv("results/Yahoo/reward_perc_day" + str(day) + ".csv", index=False)

    def run(self, mod:str="standard", compression:int=1000, termination_step:int=10000) -> Dict:
        self._compression = compression
        self._termination_step = termination_step
        results = []
        pool = Pool(cpu_count())
        if mod == 'standard':
            results = pool.map(self._run, range(self._n_test))
        elif mod == 'modified':
            results = pool.map(self._run_mod, range(self._n_test))
        elif mod == 'only_ones':
            self._dataset.drop(self._dataset[self._dataset.click == 0].index, inplace=True)
            self._dataset = self._dataset.reset_index()
            self._n_step = len(self._dataset.index)
            results = pool.map(self._run, range(self._n_test))
            
        pool.close()
        pool.join()
        return results # (Reward_trace, reward_percentual)

    def _run(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)        
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=240000, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.99999, store_estimates=False)
        agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, ts, sw_ts, d_ts, "random"]

        np.random.seed()
        c = self._compression
        reward_trace = {agent: [0] for agent in agent_list}
        reward_sum = {agent: 0 for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                if agent == "random": action = random.randint(6)
                else: action = agent.select_action()
                
                cluster, click = self.select_cluster(step)

                if (cluster == action) and (click == 1): reward = 1
                else: reward = 0

                # Update statistics
                reward_sum[agent] += reward
                if step % c == 0:
                    reward_trace[agent].append(reward_trace[agent][-1] + reward/c)
                else:
                    reward_trace[agent][-1] += reward/c

                #Update agent estimates
                if agent != "random":
                    agent.update_estimates(action, reward)

        for agent in agent_list:
            reward_sum[agent] /= self._n_step
        return (reward_trace, reward_sum)

    def _run_mod(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=0.99999, n=2000, store_estimates=False)        
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=240000, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.99999, store_estimates=False)
        agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, ts, sw_ts, d_ts, "random"]

        np.random.seed()
        reward_trace = {agent: [0] for agent in agent_list}
        reward_sum = {agent: 0 for agent in agent_list}
        effective_steps = {agent: 0 for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                # Check if the agent is already reach the "utils iterations"
                if effective_steps[agent] < self._termination_step:

                    if agent == "random": action = random.randint(6)
                    else: action = agent.select_action()

                    cluster, click = self.select_cluster(step)
                        
                    reward = 0
                    if cluster == action:
                        effective_steps[agent] += 1
                        if click == 1: reward = 1
                        else: reward = 0

                        # Update statistics                       
                        reward_sum[agent] += reward
                        reward_trace[agent].append(reward_trace[agent][-1] + reward)
     
                    #Update agent estimates
                    if agent != "random":
                        agent.update_estimates(action, reward)
                else:
                    print(agent, step)
                    agent_list.remove(agent)

        for key in reward_sum:
            reward_sum[key] /= self._termination_step

        return (reward_trace, reward_sum)        
        
    
if __name__ == "__main__":

    day = 5
    #n_arms = 6 --> Six clusters are created
    session = YahooSession(n_arms=6, n_test=10, day=day)
    
    # standard, modified, only_ones
    results = session.run(mod='only_ones', compression=1000, termination_step=50000)
    reward_trace = [item[0] for item in results]
    reward_perc = [item[1] for item in results]
    
    #session.save_reward_trace_to_csv(reward_trace, day)
    #session.save_reward_perc_to_csv(reward_perc, day)

    session.plot_reward_trace(reward_trace)
