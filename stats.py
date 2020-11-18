import multi_armed_bandit as mab

import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple




def custom_environments(n_arms, n_test, test_number:int) -> Tuple:
    # Build environment
    replay = {}
    if test_number == 1:
        # TEST 1
        replay = {'probabilities': [0.9, 0.7, 0.1, 0.3], 
                250: [(0, 0.0)],
                500: [(1, 0.0)]
                }
        
    elif test_number == 2:
        # TEST 2
        replay = {'probabilities': [0.0, 0.0, 0.1, 0.3], 
                250: [(0, 0.7)],
                500: [(1, 0.9)]
                }
    
    elif test_number == 3:
        # TEST 3
        replay = {'probabilities': [0.2, 0.3, 0.4, 0.5]}
    
    
    replay_env = mab.BernoulliReplayBandit(replay=replay)

    # Build Agents
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.98)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=75)
    max_dsw_ts = mab.MaxDSWTS(n_arms, gamma=0.98, n=20)
    min_dsw_ts = mab.MinDSWTS(n_arms, gamma=0.98, n=20)
    mean_dsw_ts = mab.MeanDSWTS(n_arms, gamma=0.98, n=20)
    agents = [ts, dynamic_ts, sw_ts, max_dsw_ts, min_dsw_ts, mean_dsw_ts]

    # Build session
    replay_session = mab.Session(replay_env, agents)
    
    # Run session
    replay_session.run(n_step=1000, n_test=n_test, use_replay=True)

    return pd.DataFrame.from_dict(replay_session._regrets), pd.DataFrame.from_dict(replay_session._real_reward_trace)


def multiple_env(n_arms, n_step, n_test, n_envs, cpus:int=cpu_count()) -> pd.DataFrame:
    results = {}
    for prob_of_change in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02]:
        rewards = {"Oracle": 0,
                "Thompson Sampling Bernoulli": 0,
                "Dynamic Thompson Sampling Bernoulli": 0,
                "Sliding Window Thompson Sampling Bernoulli": 0,
                "Max d-sw TS Bernoulli": 0,
                "Min d-sw TS Bernoulli": 0,
                "Mean d-sw TS Bernoulli":0
                }
        parms = [(n_arms, n_step, n_test, prob_of_change) for _ in range(n_envs)]
        pool = Pool(cpus)
        result = pool.starmap(_multiple_env, parms)
        pool.close()
        pool.join()

        for res in result:
            for agent in res:
                rewards[str(agent)] += res[agent]
        results.update({prob_of_change:[rewards[item] for item in rewards]})

    return pd.DataFrame.from_dict(results, orient='index', columns=["Oracle",
                                                                    "Thompson Sampling Bernoulli",
                                                                    "Dynamic Thompson Sampling Bernoulli",
                                                                    "Sliding Window Thompson Sampling Bernoulli",
                                                                    "Max d-sw TS Bernoulli",
                                                                    "Min d-sw TS Bernoulli",
                                                                    "Mean d-sw TS Bernoulli"])

def _multiple_env(n_arms, n_step, n_test, prob_of_change):
    # Build environment
    env = mab.BernoulliDynamicBandit(n_arms, prob_of_change=prob_of_change, fixed_action_prob=0.0, save_replay=True)

    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=n_step)

    # Build Agents
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.98)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=75)
    max_dsw_ts = mab.MaxDSWTS(n_arms, gamma=0.98, n=20)
    min_dsw_ts = mab.MinDSWTS(n_arms, gamma=0.98, n=20)
    mean_dsw_ts = mab.MeanDSWTS(n_arms, gamma=0.98, n=20)
    agents = [ts, dynamic_ts, sw_ts, max_dsw_ts, min_dsw_ts, mean_dsw_ts]

    # Build Env with replay
    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())

    # Build session
    replay_session = mab.Session(replay_env, agents)

    # Run session
    replay_session.run(n_step=n_step, n_test=n_test, use_replay=True)
    results = {agent: replay_session.get_reward_sum(agent)/n_step for agent in agents}
    results.update({"Oracle" : replay_session.get_reward_sum("Oracle")/n_step})
    return results


def min_max_mean_comparison(n_arms, n_step, n_test, n_envs, cpus:int=cpu_count()) -> pd.DataFrame:
    results = {}
    
    for prob_of_change in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02]:
        rewards = {"Oracle": 0,
                "Min d-sw TS Bernoulli": 0,
                "Max d-sw TS Bernoulli": 0,
                "Mean d-sw TS Bernoulli": 0
                }
        parms = [(n_arms, n_step, n_test, prob_of_change) for _ in range(n_envs)]
        pool = Pool(cpus)
        result = pool.starmap(_min_max_comparison, parms)
        pool.close()
        pool.join()

        for res in result:
            for agent in res:
                rewards[str(agent)] += res[agent]
        
        results.update({prob_of_change:[rewards[item] for item in rewards]})

    return pd.DataFrame.from_dict(results, orient='index', columns=["Oracle", 
                                                                       "Min d-sw TS Bernoulli", 
                                                                       "Max d-sw TS Bernoulli", 
                                                                       "Mean d-sw TS Bernoulli"])

def _min_max_comparison(n_arms, n_step, n_test, prob_of_change):
    # Build environment
    env = mab.BernoulliDynamicBandit(n_arms, prob_of_change=prob_of_change, fixed_action_prob=0.0, save_replay=True)

    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=n_step)

    # Build Agents
    max_dsw_ts = mab.MaxDSWTS(n_arms, gamma=0.98, n=20)
    min_dsw_ts = mab.MinDSWTS(n_arms, gamma=0.98, n=20)
    mean_dsw_ts = mab.MeanDSWTS(n_arms, gamma=0.98, n=20)
    agents = [max_dsw_ts, min_dsw_ts, mean_dsw_ts]

    # Build Env with replay
    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())

    # Build session
    replay_session = mab.Session(replay_env, agents)

    # Run session
    replay_session.run(n_step=n_step, n_test=n_test, use_replay=True)
    results = {agent: replay_session.get_reward_sum(agent)/n_step for agent in agents}
    results.update({"Oracle" : replay_session.get_reward_sum("Oracle")/n_step})
    return results


if __name__ == "__main__":
    
    n_arms = 4
    n_step = 100 #1000
    n_test = 3 #30
    n_envs = 10 #1000
    
    '''
    result = multiple_env(n_arms, n_step, n_test, n_envs, cpus=4)
    result.to_csv("results/Multiple_env.csv")
    '''
    '''
    result = min_max_mean_comparison(n_arms, n_step, n_test, n_envs, cpus=4)
    result.to_csv("results/dsw_ts_comparison.csv")
    '''
    
    test_number = 1
    regret, real_reward_trace = custom_environments(n_arms, n_test=10 ,test_number=test_number)
    regret.to_csv("results/custom_test_" + str(test_number) + "_regret.csv")
    real_reward_trace.to_csv("results/custom_test_" + str(test_number) + "_real_reward_trace.csv")
    

