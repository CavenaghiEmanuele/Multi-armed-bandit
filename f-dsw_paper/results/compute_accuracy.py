import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from statistics import mean, stdev


if __name__ == '__main__':

    agents = ['Max d-sw TS', 'Min d-sw TS', 'Mean d-sw TS', 'Discounted TS', 'Sliding Window TS', 'Thompson Sampling', 'random']
    suffix_list = ['', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
    dataset_name = 'insects' # local_news, baltimore_crime, insects
    type_change = '/incremental-reoccurring' # abrupt, incremental, incremental-abrupt, incremental-reoccurring
    path = dataset_name + '/tests' + type_change + '/reward_trace.csv'

    dataset = pd.read_csv(path)
    dataset = dataset.add_suffix('')
    
    agent_results = {
        agent : [dataset[agent+suffix][len(dataset.index)-1] for suffix in suffix_list] 
        for agent in agents
        }

    stats = {
        agent : (
            round((mean(agent_results[agent]) / (len(dataset.index)-1)*100), 2),
            #stdev(agent_results[agent]) / (len(dataset.index)-1)
            )
        for agent in agents
        }

    
    #print(dataset)
    print(stats)

