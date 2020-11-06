import csv
import pickle
import pandas as pd
import multi_armed_bandit as mab
import matplotlib.pyplot as plt

from numpy import loadtxt
from statistics import mean
    
    
def build_env(day: int, steps_for_hour: int=1, clusters: int=0):

    path = '/home/emanuele/GoogleDrive/Thompson Sampling/yahoo_dataset/'
    cluster = pd.read_csv(path + 'clusters_6.csv')
    hours = 230
    if day != 'alldays':
        day = 'day' + str(day)
        hours = 23

    # Get id articles
    with open(path + day + '/id_articles.txt', "rb") as fp:
        id_articles = pickle.load(fp)

    replay = {}
    for hour in range(1, hours + 1):
        dataset = pd.read_csv(path + day + '/hour' + str(hour) + '.csv')

        if clusters != 0:
            prob = [[] for i in range(0,clusters)]
        else:
            prob = [[] for i in range(0,len(id_articles))]
                    
        for id in id_articles:
            index = id_articles.index(id)
            if clusters != 0:
                index = cluster.loc[cluster['id_article']==id]['cluster'].values[0]

            try:
                if hour == 1: 
                    prob[index].append(dataset[str(id)][0])
                else: 
                    prob[index].append((id_articles.index(id), dataset[str(id)][0]))
            except:
                if hour == 1: prob[index].append(0.0)
                else: prob[index].append((id_articles.index(id), 0.0))
        
        if hour == 1:
            if clusters != 0:
                prob = [mean(prob[i]) for i in range(0, clusters)]
            else:
                prob = [prob[i][0] for i in range(0, len(id_articles))]
            replay.update({'probabilities': prob})
        else:
            if clusters != 0:
                prob = [(i, mean([x[1] for x in prob[i]])) for i in range(0, clusters)]
            else:
                prob = [prob[i][0] for i in range(0, len(id_articles))]
            replay.update({hour*steps_for_hour : prob})
        
    return mab.BernoulliReplayBandit(replay=replay)

if __name__ == '__main__':
    
    ######## PARAMETERS #########
    day = 6
    steps_for_hour = 60
    clusters = 6
    ############################
    hours = 23
    if day == 'alldays':
        hours = 230
    env = build_env(day, steps_for_hour, clusters=clusters)
    
    n_arms = env.get_n_arms()
    
    # Build Agents
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.70)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=30)
    my_ts = mab.MaxDSWTS(n_arms, gamma=0.70, n=10)
    
    # Build session
    session = mab.Session(env, [ts, dynamic_ts, sw_ts, my_ts])
    
    # Run session
    session.run(n_step=(hours+1)*steps_for_hour, n_test=100, use_replay=True)
    
    #Plot results
    env.plot_arms(render=False)
    session.plot_all(render=False)
    plt.show()
    