import csv
import pickle
import pandas as pd
import multi_armed_bandit as mab
import matplotlib.pyplot as plt

from numpy import loadtxt


if __name__ == '__main__':
    
    path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'
    day = ['alldays']
    
    with open(path + day[0] + '/id_articles.txt', "rb") as fp:
        id_articles = pickle.load(fp)
            
    dataset = pd.read_csv(path + day[0] + '/hour1.csv')
    probabilities = []
    for id in id_articles:
        try:
            probabilities.append(dataset[str(id)][0])
        except:
            probabilities.append(0.0)
        
        
    replay = {'probabilities': probabilities}
    for hour in range(2,231):
        dataset = pd.read_csv(path + day[0] + '/hour' + str(hour) + '.csv')
        update = []
        for id in id_articles:
            try:
                update.append((id_articles.index(id), dataset[str(id)][0]))
            except:
                update.append((id_articles.index(id), 0.0))
        
        replay.update({hour : update})

    
    n_arms = len(id_articles)
    
    replay_env = mab.BernoulliReplayBandit(replay=replay)
    session = mab.Session(replay_env, [])
    session.run(230, 1, use_replay=True)
    replay_env.plot_arms()
