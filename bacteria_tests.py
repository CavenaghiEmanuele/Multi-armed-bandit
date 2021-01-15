import pandas as pd
import numpy as np

from numpy import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange

import multi_armed_bandit as mab


class BacteriaDatasetSession():

    _test_env: pd.DataFrame
    _train_env: pd.DataFrame
    _n_test: int
    _n_step_train: int
    _n_step_test: int
    _fixed_step: int

    def __init__(self, n_test:int=10, fixed_step:int=1) -> None:
        self._n_test = n_test
        self._n_step_train = 61 * fixed_step
        self._n_step_test = 177 * fixed_step
        self._fixed_step = fixed_step
        self.build_env(train_test='train')
        self.build_env(train_test='test')

    def build_env(self, train_test:str):  
        if train_test == 'train':
            df = pd.read_csv('datasets/bacteria/train.csv')
        elif train_test == 'test':
            df = pd.read_csv('datasets/bacteria/test.csv')

        bacteries = list(df.bacteria.unique())
        
        # Build frequencies
        frequencies = {}
        for registration in df.columns:
            if registration != 'bacteria':
                frequencies.update(
                    {registration : [
                            (bacteries.index(row['bacteria']), row[registration] / df[registration].sum())
                            for _, row in df.iterrows()
                            ]
                    })
        # Build replay
        replay = {}
        for i, value in enumerate(frequencies.values()):
            if i == 0: replay.update({'probabilities' : [x[1] for x in value]})
            else: replay.update({i*self._fixed_step : value})

        if train_test == 'train':
            self._train_env = mab.BernoulliReplayBandit(replay=replay)
        elif train_test == 'test':
            self._test_env = mab.BernoulliReplayBandit(replay=replay)

    def plot_arms(self, train_test:str, plot_legend:bool = True):
        if train_test == 'train':
            session = mab.Session(env=self._train_env, agent=[])
            session.run(n_step=self._n_step_train, use_replay=True)
            self._train_env.plot_arms(render=True, plot_legend=plot_legend)

        elif train_test == 'test':
            session = mab.Session(env=self._test_env, agent=[])
            session.run(n_step=self._n_step_test, use_replay=True)
            self._test_env.plot_arms(render=True, plot_legend=plot_legend)

    def run(self) -> None:
        params = [(_,) for _ in range(self._n_test)]

        pool = Pool(cpu_count())
        results = pool.starmap(self._run, params)       
        pool.close()
        pool.join()

        tmp = {str(agent):[] for agent in results[0]}
        for i in range(len(results)):
            for key, value in results[i].items():
                tmp[str(key)].append(value)
            
        dataset = pd.DataFrame.from_dict(tmp)        
        dataset.to_csv('results/bacteria/reward_perc.csv', index=False)
        return 

    def _run(self, fake) -> Dict:
        agent_list = self._best_agents(n_arms=self._test_env._n_arms) #Build agents
        
        np.random.seed()
        session = mab.Session(env=self._test_env, agent=agent_list)
        session.run(n_step=self._n_step_test, n_test=1, use_replay=True)
        
        results = {str(agent): session.get_reward_sum(agent) for agent in agent_list}
        results.update({"Oracle" : session.get_reward_sum("Oracle")})
        return results

    def find_params(self) -> None:
        path = 'results/bacteria/find_params/'

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
                tmp = {str(self._params[str(agent)][i]) : [result[agent] for result in results]}
                dataset = pd.concat(
                    [pd.read_csv(path + agent + '.csv'), pd.DataFrame.from_dict(tmp)], 
                    axis=1, join='inner')
                dataset.to_csv(path + agent + '.csv', index=False)
        return

    def _find_params(self, f_gamma, f_n, sw_n, d_ts_gamma):
        n_arms = self._train_env._n_arms
        ########## BUILD AGENTS ###########
        agent_list = [
            mab.MaxDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.MinDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.MeanDSWTS(n_arms=n_arms, gamma=f_gamma, n=f_n, store_estimates=False), 
            mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=sw_n, store_estimates=False), 
            mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=d_ts_gamma, store_estimates=False)
            ]
        np.random.seed()
        session = mab.Session(env=self._train_env, agent=agent_list)
        session.run(n_step=self._n_step_train, n_test=1, use_replay=True)
        return {str(agent): session.get_reward_sum(agent) for agent in agent_list}

    def _best_agents(self, n_arms) -> List:
            return [
                mab.MaxDSWTS(n_arms=n_arms, gamma=0.9999, n=800, store_estimates=False),
                mab.MinDSWTS(n_arms=n_arms, gamma=0.99, n=800, store_estimates=False),
                mab.MeanDSWTS(n_arms=n_arms, gamma=0.9999, n=800, store_estimates=False),
                mab.BernoulliThompsonSampling(n_arms=n_arms, store_estimates=False),
                mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=12800, store_estimates=False), 
                mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=0.9999, store_estimates=False), 
                mab.RandomAlgo(n_arms=n_arms)
                ]


if __name__ == '__main__':

    session = BacteriaDatasetSession(n_test=10, fixed_step=1)
    #session.run()
    #session.find_params()
    session.plot_arms(plot_legend=False, train_test='train')
