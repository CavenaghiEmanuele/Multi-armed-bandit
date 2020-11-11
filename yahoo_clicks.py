import csv
from multi_armed_bandit.algorithms.bernoulli_dist import max_dsw_ts
import pickle
import pandas as pd
import multi_armed_bandit as mab

from numpy import loadtxt


if __name__ == "__main__":
    
    n_arms = 6 # Six clusters are created
    
    max_dsw_ts = mab.MaxDSWTS(n_arms=n_arms, gamma=0.98, n=20)
    session = mab.YahooSession(agent=[max_dsw_ts], day=1)
    session.run()
    