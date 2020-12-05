import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange

import multi_armed_bandit as mab


class BaltimoreCrimeSession():

    _dataset: pd.DataFrame
    _n_step: int
    _n_test: int
    _n_arms: int
    _districts: List

    def __init__(self, n_test:int=1) -> None:
        
        self._dataset = pd.read_csv('datasets/baltimore_crime/modified_baltimore_crime.csv', parse_dates=['CrimeDate'])
        self._dataset.drop('CrimeTime', axis=1, inplace=True)
        self._n_step = len(self._dataset.index)
        self._n_test = n_test
        self._districts = list(self._dataset.District.unique())
        self._n_arms = len(self._districts)


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

    def run(self) -> Dict:

        pool = Pool(cpu_count())
        results = pool.map(self._run, range(self._n_test))
        pool.close()
        pool.join()
        return results #(Reward_trace, reward_percentual)

    def _run(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.99, n=500, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=0.99, n=500, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=0.99, n=500, store_estimates=False)        
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=30000, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.99, store_estimates=False)
        agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, ts, sw_ts, d_ts, "random"]

        np.random.seed()
        reward_trace = {agent: [0] for agent in agent_list}
        reward_sum = {agent: 0 for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                if agent == "random": action = random.randint(6)
                else: action = agent.select_action()
                
                district = self._districts.index(self._dataset.loc[step]['District'])

                if (district == action): reward = 1
                else: reward = 0

                # Update statistics
                reward_sum[agent] += reward
                reward_trace[agent].append(reward_trace[agent][-1] + reward)

                #Update agent estimates
                if agent != "random":
                    agent.update_estimates(action, reward)

        for agent in agent_list:
            reward_sum[agent] /= self._n_step
        return (reward_trace, reward_sum)
    
    
if __name__ == "__main__":

    session = BaltimoreCrimeSession(n_test=10)
    
    results = session.run()
    reward_trace = [item[0] for item in results]
    reward_perc = [item[1] for item in results]
    
    #session.save_reward_trace_to_csv(reward_trace)
    #session.save_reward_perc_to_csv(reward_perc)

    session.plot_reward_trace(reward_trace)
