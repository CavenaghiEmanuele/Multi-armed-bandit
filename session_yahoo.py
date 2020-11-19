from pickle import NONE, TRUE
from multi_armed_bandit.algorithms.bernoulli_dist import mean_dsw_ts
from multi_armed_bandit.algorithms.bernoulli_dist import min_dsw_ts
import matplotlib.pyplot as plt
from numpy import random
import pandas as pd
import pickle
import numpy as np
import multi_armed_bandit as mab
from numpy.core.fromnumeric import mean

from multiprocessing import Pool, cpu_count
from typing import Dict, List
from tqdm import trange


class YahooSession():

    _dataset: pd.DataFrame
    _clusters: pd.DataFrame
    _id_articles: List
    _n_step: int
    _n_test: int
    _n_arms: int
    _compression: int

    def __init__(self, day:int, n_arms:int, n_test:int=1, compression:int=1) -> None:

        filenames = [
            "ydata-fp-td-clicks-v1_0.20090501.csv", "ydata-fp-td-clicks-v1_0.20090502.csv", "ydata-fp-td-clicks-v1_0.20090503.csv",
            "ydata-fp-td-clicks-v1_0.20090504.csv", "ydata-fp-td-clicks-v1_0.20090505.csv", "ydata-fp-td-clicks-v1_0.20090506.csv",
            "ydata-fp-td-clicks-v1_0.20090507.csv", "ydata-fp-td-clicks-v1_0.20090508.csv", "ydata-fp-td-clicks-v1_0.20090509.csv",
            "ydata-fp-td-clicks-v1_0.20090510.csv",
        ]
        path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'

        self._dataset = pd.read_csv("/media/emanuele/860EFA500EFA392F/Dataset Yahoo!/R6/Dataset Yahoo modified/" + filenames[day-1])
        self._clusters = pd.read_csv(path + 'clusters_6.csv')
        with open(path + 'day' + str(day) + '/id_articles.txt', "rb") as fp:
            self._id_articles = pickle.load(fp)
        self._n_step = len(self._dataset.index)
        self._n_test = n_test
        self._n_arms = n_arms
        self._compression = compression

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
        dataset.to_csv("results/Yahoo/reward_trace_day" + str(day) + ".csv")
    
    def save_reward_perc_to_csv(self, results, day) -> None:
        
        tmp = {str(agent):[] for agent in results[0]}
        for i in range(len(results)):
            for key, value in results[i].items():
                tmp[str(key)].append(value)
                   
        dataset = pd.DataFrame.from_dict(tmp)        
        dataset.to_csv("results/Yahoo/reward_perc_day" + str(day) + ".csv")

    def plot_from_csv(self, day, grayscale:bool=False) -> None:
        path = 'results/Yahoo/reward_trace_day' + str(day) + '.csv'
        dataset = pd.read_csv(path)
        dataset = dataset.drop('Unnamed: 0', 1)
        dataset = dataset.add_suffix('')
        
        agent_list = ['Max d-sw TS Bernoulli', 'Min d-sw TS Bernoulli', 'Mean d-sw TS Bernoulli',
                  'Thompson Sampling Bernoulli', 'Sliding Window Thompson Sampling Bernoulli',
                  'Discounted Thompson Sampling Bernoulli', 'random']
        suffix_list = ['', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
        
        if grayscale:
            plt.style.use('grayscale')
        
        for agent in agent_list:
            plt.plot(np.mean([dataset[agent + suffix].values for suffix in suffix_list], axis=0), label=agent, linewidth=3)      

        plt.title('Reward trace', fontsize=24)
        plt.grid()
        plt.legend(prop={'size': 24})
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

        plt.show()
        
        
    def run(self) -> Dict:
        pool = Pool(cpu_count())
        results = pool.map(self._run, range(self._n_test))
        pool.close()
        pool.join()
        return results # (Reward_trace, reward_percentual)

    def _run(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.9999, n=2000, store_estimates=False)
        min_dsw_ts = mab.MinDSWTS(n_arms=self._n_arms, gamma=0.9999, n=2000, store_estimates=False)
        mean_dsw_ts = mab.MeanDSWTS(n_arms=self._n_arms, gamma=0.9999, n=2000, store_estimates=False)        
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=7500, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.9999, store_estimates=False)
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


if __name__ == "__main__":

    day = 2
    #n_arms = 6 --> Six clusters are created
    session = YahooSession(n_arms=6, n_test=10, compression=1000, day=day)
    
    results = session.run()
    reward_trace = [item[0] for item in results]
    reward_perc = [item[1] for item in results]

    session.save_reward_trace_to_csv(reward_trace, day)
    session.save_reward_perc_to_csv(reward_perc, day)
    
    session.plot_reward_trace(reward_trace)
    '''
    session.plot_from_csv(day=day, grayscale=False)
    '''
