import matplotlib.pyplot as plt
from numpy import random
import pandas as pd
import pickle
import numpy as np
import multi_armed_bandit as mab

from multiprocessing import Pool, cpu_count
from typing import List, Dict
from tqdm import trange


def select_cluster(step:int, dataset, clusters):
    row = dataset.loc[step]
    click = dataset.loc[step]['click']
    cluster = clusters.loc[clusters['id_article']==row['id_article']]['cluster'].values[0]        
    return cluster, click

def plot_reward_trace(n_test, n_step, results):
    traces = {str(agent): np.zeros(n_step) for agent in results[0]}

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

def run(n_arms, n_step, dataset, clusters, agent_list) -> Dict:
    np.random.seed()
    reward_trace = {agent: [0] for agent in agent_list}
        
    for step in trange(n_step):
        for agent in agent_list:
            #action = agent.select_action()
            action = random.randint(6)
            cluster, click = select_cluster(step, dataset, clusters)

            if (cluster == action) and (click == 1): reward = 1
            else: reward = 0

            # Update statistics
            reward_trace[agent].append(reward_trace[agent][-1] + reward)

            #Update agent estimates
            agent.update_estimates(action, reward)

    return reward_trace


if __name__ == "__main__":
    
    ########## PARAMETERS ############
    n_arms = 6 # Six clusters are created
    n_test = 16
    day = 1
    
    ########## LOAD FILES ###########
    filenames = [
            "ydata-fp-td-clicks-v1_0.20090501.csv", "ydata-fp-td-clicks-v1_0.20090502.csv", "ydata-fp-td-clicks-v1_0.20090503.csv",
            "ydata-fp-td-clicks-v1_0.20090504.csv", "ydata-fp-td-clicks-v1_0.20090505.csv", "ydata-fp-td-clicks-v1_0.20090506.csv",
            "ydata-fp-td-clicks-v1_0.20090507.csv", "ydata-fp-td-clicks-v1_0.20090508.csv", "ydata-fp-td-clicks-v1_0.20090509.csv",
            "ydata-fp-td-clicks-v1_0.20090510.csv",
        ]
    path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'
    
    dataset = pd.read_csv("/media/emanuele/860EFA500EFA392F/Dataset Yahoo!/R6/Dataset Yahoo modified/" + filenames[day-1])
    clusters = pd.read_csv(path + 'clusters_6.csv')
    with open(path + 'day' + str(day) + '/id_articles.txt', "rb") as fp:
        id_articles = pickle.load(fp)    
    n_step = len(dataset.index)

    ########## BUILD AGENTS ###########
    max_dsw_ts = mab.MaxDSWTS(n_arms=n_arms, gamma=0.98, n=20)
    ts = mab.BernoulliThompsonSampling(n_arms=n_arms)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=75)
    d_ts = mab.DynamicBernoulliTS(n_arms=n_arms, gamma=0.98)
    agent_list = [max_dsw_ts, ts, sw_ts, d_ts]

    ########## RUN ENVS ###########
    parms = [(n_arms, n_step, dataset, clusters, agent_list) for _ in range(n_test)]

    pool = Pool(cpu_count())
    results = pool.starmap(run, parms)
    pool.close()
    pool.join()
    
    ########## PLOT ###########
    plot_reward_trace(n_test, n_step-1, results)
    