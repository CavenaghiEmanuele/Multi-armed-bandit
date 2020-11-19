import matplotlib.pyplot as plt
from numpy import random
import pandas as pd
import pickle
import numpy as np
import multi_armed_bandit as mab

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

    def run(self) -> Dict:
        pool = Pool(cpu_count())
        results = pool.map(self._run, range(self._n_test))
        pool.close()
        pool.join()
        return results

    def _run(self, fake) -> Dict:
        ########## BUILD AGENTS ###########
        max_dsw_ts = mab.MaxDSWTS(n_arms=self._n_arms, gamma=0.9999, n=2000, store_estimates=False)
        ts = mab.BernoulliThompsonSampling(n_arms=self._n_arms, store_estimates=False)
        sw_ts = mab.BernoulliSlidingWindowTS(n_arms=self._n_arms, n=7500, store_estimates=False)
        d_ts = mab.DiscountedBernoulliTS(n_arms=self._n_arms, gamma=0.9999, store_estimates=False)
        agent_list = [max_dsw_ts, ts, sw_ts, d_ts, "random"]

        np.random.seed()
        c = self._compression
        reward_trace = {agent: [0] for agent in agent_list}
            
        for step in trange(self._n_step):
            for agent in agent_list:
                if agent == "random": action = random.randint(6)
                else: action = agent.select_action()
                
                cluster, click = self.select_cluster(step)

                if (cluster == action) and (click == 1): reward = 1
                else: reward = 0

                # Update statistics
                if step % c == 0:
                    reward_trace[agent].append(reward_trace[agent][-1] + reward/c)
                else:
                    reward_trace[agent][-1] += reward/c

                #Update agent estimates
                if agent != "random":
                    agent.update_estimates(action, reward)

        return reward_trace


if __name__ == "__main__":

    #n_arms = 6 --> Six clusters are created
    session = YahooSession(n_arms=6, n_test=10, compression=1000, day=9)
    session.plot_reward_trace(session.run())
