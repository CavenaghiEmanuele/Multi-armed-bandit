import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange

import multi_armed_bandit as mab


class InsectSession():

    _dataset: pd.DataFrame
    _n_step: int
    _n_test: int
    _n_arms: int
    _insect_class: List

    def __init__(self, type_of_change:str, balanced_imbalanced:str, n_test:int=1) -> None:
        filename = type_of_change + '_' + balanced_imbalanced
        self._dataset = pd.read_csv('datasets/insects/' + filename + '.csv')
        self._n_step = len(self._dataset.index)
        self._n_test = n_test
        self._insect_class = list(pd.unique(self._dataset['class']))
        self._n_arms = len(self._insect_class)

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

    def save_reward_trace_to_csv(self, results, path) -> None:
        dataset = pd.DataFrame()
        for i in range(len(results)):            
            dataset = pd.concat([dataset, pd.DataFrame.from_dict(results[i])], axis=1)
        dataset.to_csv(path + 'reward_trace.csv', index=False)
    
    def save_reward_perc_to_csv(self, results, path) -> None:    
        tmp = {str(agent):[] for agent in results[0]}
        for i in range(len(results)):
            for key, value in results[i].items():
                tmp[str(key)].append(value)
                   
        dataset = pd.DataFrame.from_dict(tmp)        
        dataset.to_csv(path + 'reward_perc.csv', index=False)

    def run(self, type_of_change:str, balanced_imbalanced:str) -> Dict:
        path = 'results/insects/tests/' + type_of_change + '/' + balanced_imbalanced + '/'
        pool = Pool(cpu_count())
        results = pool.map(self._run, range(self._n_test))
        pool.close()
        pool.join()
        
        self.save_reward_trace_to_csv(results=[item[0] for item in results], path=path)
        self.save_reward_perc_to_csv(results=[item[1] for item in results], path=path)
        return results #(Reward_trace, reward_percentual)

    def _run(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.999, n=800, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=0.99, n=200, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=0.999, n=800, store_estimates=False)        
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=3200, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.999, store_estimates=False)
        agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, ts, sw_ts, d_ts, "random"]

        np.random.seed()
        reward_trace = {agent: [0] for agent in agent_list}
        reward_sum = {agent: 0 for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                if agent == "random": action = random.randint(6)
                else: action = agent.select_action()
                
                insect_class = self._insect_class.index(self._dataset.loc[step]['class'])

                if (insect_class == action): reward = 1
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

    def find_params(self, type_of_change:str, balanced_imbalanced:str, test_size:int=20) -> None:
        self._n_step = int(len(self._dataset.index) / 100 * test_size)
        path = 'results/insects/find_params/' + type_of_change + '/' + balanced_imbalanced + '/'

        self._params = {
            'f_algo' : [
                (0.9, 100), (0.9, 200), (0.9, 400), (0.9, 800),
                (0.95, 100), (0.95, 200), (0.95, 400), (0.95, 800),
                (0.99, 100), (0.99, 200), (0.99, 400), (0.99, 800),
                (0.999, 800), (0.9999, 800), (0.99999, 800)
                ],
            'Sliding Window TS' : [
                25, 50, 100, 200, 
                400, 800, 1600, 3200, 
                6400, 12800, 25600, 51200,
                102400, 204800, 409600
                ],
            'Discounted TS' : [
                0.5, 0.6, 0.7, 0.8,
                0.9, 0.92, 0.95, 0.97, 
                0.98, 0.99, 0.999, 0.9999,
                0.99999, 0.999999, 0.9999999
                ]
        }
        # only to save results
        self._params.update({'Max d-sw TS':self._params['f_algo']}) 
        self._params.update({'Min d-sw TS':self._params['f_algo']})
        self._params.update({'Mean d-sw TS':self._params['f_algo']})
        
        for i in range(len(self._params['f_algo'])):
            params = [
                (self._params['f_algo'][i][0],
                 self._params['f_algo'][i][1],
                 self._params['Sliding Window TS'][i],
                 self._params['Discounted TS'][i]
                 ) 
                for _ in range(self._n_test)
                ]
            pool = Pool(cpu_count())
            results = pool.starmap(self._find_params, params)
            pool.close()
            pool.join()

            for agent in results[0]:
                tmp = {str(self._params[agent][i]) : [result[agent] for result in results]}
                dataset = pd.concat(
                    [pd.read_csv(path + agent + '.csv'), pd.DataFrame.from_dict(tmp)], 
                    axis=1, join='inner')
                dataset.to_csv(path + agent + '.csv', index=False)
        return
    
    def _find_params(self, f_gamma, f_n, sw_n, d_ts_gamma):
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=f_gamma, n=f_n, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=f_gamma, n=f_n, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=f_gamma, n=f_n, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=sw_n, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=d_ts_gamma, store_estimates=False)
        agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, sw_ts, d_ts]

        np.random.seed()
        reward_sum = {str(agent): 0 for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                if agent == "random": action = random.randint(6)
                else: action = agent.select_action()
                
                insect_class = self._insect_class.index(self._dataset.loc[step]['class'])

                if (insect_class == action): reward = 1
                else: reward = 0

                # Update statistics
                reward_sum[str(agent)] += reward

                #Update agent estimates
                if agent != "random":
                    agent.update_estimates(action, reward)

        for agent in agent_list:
            reward_sum[str(agent)] /= self._n_step
        return reward_sum
    
    
if __name__ == "__main__":
    
    type_of_change='incremental-reoccurring' # abrupt, gradual, incremental-abrupt, incremental, incremental-reoccurring, out-of-control
    balanced_imbalanced='imbalanced' # balanced, imbalanced
    
    session = InsectSession(type_of_change, balanced_imbalanced, n_test=10)
    
    print(type_of_change, balanced_imbalanced)
    #session.find_params(type_of_change, balanced_imbalanced, test_size=100)
    session.run(type_of_change, balanced_imbalanced)
