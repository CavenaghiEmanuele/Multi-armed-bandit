import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from numpy import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange

import multi_armed_bandit as mab


class RealDatasetSession():

    _dataset_name: str
    _dataset: pd.DataFrame
    _n_step: int
    _n_test: int
    _arms: List

    def __init__(self, dataset_name:str, type_of_change:str='', balanced_imbalanced:str='', n_test:int=1) -> None:
        
        self._dataset_name = dataset_name
        path = 'datasets/' + dataset_name + '/'
        if dataset_name == 'insects':
            dataset_name = type_of_change + '_' + balanced_imbalanced
        self._dataset = pd.read_csv(path + dataset_name + '.csv')
        self._n_step = len(self._dataset.index)
        self._n_test = n_test
        self._arms = set()
        for t in set(pd.unique(self._dataset[self._arms_column()])):
            self._arms.update(t.split(','))
        self._arms = list(self._arms)
        
        # for insects datasets
        self._type_of_change = type_of_change
        self._balanced_imbalanced = balanced_imbalanced


    def _arms_column(self) -> str:
        if self._dataset_name == 'insects':
            return 'class'
        elif self._dataset_name == 'baltimore_crime':
            return 'District'
        elif self._dataset_name == 'adige_news':
            return 'article_topic'

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

    def run(self, test_size:int=80) -> Dict:
        path = 'results/' + self._dataset_name + '/'        
        params = [(test_size,) for _ in range(self._n_test)]
        
        pool = Pool(cpu_count())
        results = pool.starmap(self._run, params)       
        pool.close()
        pool.join()
        
        self.save_reward_trace_to_csv(results=[item[0] for item in results], path=path)
        self.save_reward_perc_to_csv(results=[item[1] for item in results], path=path)
        return results #(Reward_trace, reward_percentual)

    def _run(self, test_size) -> Dict:
        n_arms = len(self._arms)
        agent_list = self._best_agents(n_arms=n_arms) #Build agents

        np.random.seed()
        reward_trace = {agent: [0] for agent in agent_list}
        reward_sum = {agent: 0 for agent in agent_list}
            
        for step in trange(self._n_step - int(self._n_step/100*test_size), self._n_step):
            for agent in agent_list:
                if agent == 'random': action = random.randint(6)
                else: action = agent.select_action()
                
                arms = self._dataset.loc[step][self._arms_column()].split(',')
                reward = 0
                for a in arms:
                    arm = self._arms.index(a)
                    if (arm == action):
                        reward = 1
                        break

                # Update statistics
                reward_sum[agent] += reward
                reward_trace[agent].append(reward_trace[agent][-1] + reward)

                #Update agent estimates
                if agent != 'random':
                    agent.update_estimates(action, reward)

        for agent in agent_list:
            reward_sum[agent] /= self._n_step
        return (reward_trace, reward_sum)

    def find_params(self, train_size:int=20) -> None:
        path = 'results/' + self._dataset_name + '/find_params/'
        if self._dataset_name == 'insects':
            path += self._type_of_change + '/' + self._balanced_imbalanced + '/'

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
                (
                    train_size,
                    self._params['f_algo'][i][0],
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

    def _find_params(self, train_size, f_gamma, f_n, sw_n, d_ts_gamma):
        n_arms = len(self._arms)
        ########## BUILD AGENTS ###########
        agent_list = [
            mab.MaxDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.MinDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.MeanDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=sw_n, store_estimates=False), 
            mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=d_ts_gamma, store_estimates=False)
            ]

        np.random.seed()
        reward_sum = {str(agent): 0 for agent in agent_list}
            
        for step in trange(int(len(self._dataset.index) / 100 * train_size)):
            for agent in agent_list:
                if agent == 'random': action = random.randint(6)
                else: action = agent.select_action()
                                    
                arms = self._dataset.loc[step][self._arms_column()].split(',')
                reward = 0
                for a in arms:
                    arm = self._arms.index(a)
                    if (arm == action):
                        reward = 1
                        break

                # Update statistics
                reward_sum[str(agent)] += reward

                #Update agent estimates
                if agent != 'random':
                    agent.update_estimates(action, reward)

        for agent in agent_list:
            reward_sum[str(agent)] /= self._n_step
        return reward_sum

    def _best_agents(self, n_arms) -> List:
        if self._dataset_name == 'adige_news':
            return [
                mab.MaxDSWTS(n_arms=n_arms, gamma=0.999, n=800, store_estimates=False),
                mab.MinDSWTS(n_arms=n_arms, gamma=0.95, n=800, store_estimates=False),
                mab.MeanDSWTS(n_arms=n_arms, gamma=0.99, n=400, store_estimates=False),
                mab.BernoulliThompsonSampling(n_arms=n_arms, store_estimates=False),
                mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=3200, store_estimates=False), 
                mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=0.999, store_estimates=False), 
                'random'
                ]
        elif self._dataset_name == 'baltimore_crime':
            return [
                mab.MaxDSWTS(n_arms=n_arms, gamma=0.9999, n=800, store_estimates=False),
                mab.MinDSWTS(n_arms=n_arms, gamma=0.999, n=800, store_estimates=False),
                mab.MeanDSWTS(n_arms=n_arms, gamma=0.9999, n=800, store_estimates=False),
                mab.BernoulliThompsonSampling(n_arms=n_arms, store_estimates=False),
                mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=12800, store_estimates=False),
                mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=0.9999, store_estimates=False),
                'random'
                ]
        elif self._dataset_name == 'insects':
            return [
                mab.MaxDSWTS(n_arms=n_arms, gamma=0.999, n=800, store_estimates=False),
                mab.MinDSWTS(n_arms=n_arms, gamma=0.99, n=200, store_estimates=False),
                mab.MeanDSWTS(n_arms=n_arms, gamma=0.999, n=800, store_estimates=False),
                mab.BernoulliThompsonSampling(n_arms=n_arms, store_estimates=False),
                mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=3200, store_estimates=False),
                mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=0.999, store_estimates=False),
                'random'
                ]


if __name__ == '__main__':

    type_of_change = 'abrupt' # abrupt, gradual, incremental-abrupt, incremental, incremental-reoccurring, out-of-control
    balanced_imbalanced = 'imbalanced' # balanced, imbalanced
    dataset_name = 'adige_news' # adige_news, baltimore_crime, insects

    session = RealDatasetSession(dataset_name=dataset_name, type_of_change=type_of_change, balanced_imbalanced=balanced_imbalanced, n_test=10)

    session.run(test_size=80)
    #session.find_params(train_size=20)
