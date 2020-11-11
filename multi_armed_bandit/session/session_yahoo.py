import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from typing import Dict, List
from pandas.io.parsers import read_fwf
from tqdm import trange

from ..environments import DynamicMultiArmedBandit
from ..algorithms import Algorithm


class YahooSession():

    _n_step: int
    _dataset: pd.DataFrame
    _clusters: pd.DataFrame
    _id_articles: List
    _agents: List[Algorithm]  

    def __init__(self, agent: List[Algorithm], day: int):
        if isinstance(agent, List):
            self._agents = agent
        else:
            self._agents = [agent]
        
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


    def run(self, n_test: int = 1) -> None:

        for test in trange(n_test):
            for step in trange(self._n_step):
                for agent in self._agents:
                    action = agent.select_action()       
                    cluster, click = self.select_cluster(step=step)
                    
                    if (cluster == action) and (click == 1): reward = 1
                    else: reward = 0
                                    
                    agent.update_estimates(action, reward)    
            
            # Reset agent to start condition
            if (test < n_test-1):
                for agent in self._agents:
                    agent.reset_agent()
                    
        return

    def select_cluster(self, step:int):
        row = self._dataset.loc[step]
        click = self._dataset.loc[step]['click']
        cluster = self._clusters.loc[self._clusters['id_article']==row['id_article']]['cluster'].values[0]        
        return cluster, click
