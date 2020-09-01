import multi_armed_bandit as mab

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt


def stationary_bernoulli():
    n_arms = 5
    # Build environment
    env = mab.BernoulliBandit(n_arms, probabilities=[0.8, 0.4, 0.6, 0.75, 0.7])
    env.plot_arms(render=True)
    
    # Build Agents
    greedy = mab.BernoulliGreedy(n_arms)
    ucb = mab.BernoulliUCB(n_arms, c = 1)
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.99)
    
    # Build session
    session = mab.Session(env=env, agent=[greedy, ucb, ts, dynamic_ts])
    
    # Run session
    session.run(n_step=2000, n_test=1000, use_replay=False)
    
    #Plot results
    env.plot_arms(render=False)
    session.plot_all(render=False)
    plt.show()
    

def non_stationary_bernoulli():
    n_arms = 5
    # Build environment
    env = mab.BernoulliDynamicBandit(n_arms, prob_of_change=0.003, fixed_action_prob=0.1, save_replay=True)
    
    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=1000)
    env.plot_arms(render=True)

    # Build Env with replay
    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())

    # Build Agents
    greedy = mab.BernoulliGreedy(n_arms)
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.99)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=100)
    my_ts = mab.MyBernoulliTS(n_arms, gamma=0.99, n=30)
    
    # Build session
    replay_session = mab.Session(replay_env, [greedy, ts, dynamic_ts, sw_ts, my_ts])
    
    # Run session
    replay_session.run(n_step=2000, n_test=100, use_replay=True)
    
    #Plot results
    my_ts.plot_estimates(render=False)
    replay_env.plot_arms(render=False)
    replay_session.plot_all(render=False)
    plt.show()


def non_stationary_bernoulli_paper():
    n_arms = 4
    # Build environment
    replay = {'probabilities': [0.0, 0.0, 0.0, 0.0], 
              50: [(0, 0.1)], 100: [(1, 0.37)], 150: [(2, 0.63)], 200: [(3, 0.9)],
              250:[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
              300: [(0, 0.1)], 350: [(1, 0.37)], 400: [(2, 0.63)], 450: [(3, 0.9)],
              500:[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
              550: [(0, 0.1)], 600: [(1, 0.37)], 650: [(2, 0.63)], 700: [(3, 0.9)],
              750:[(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)],
              800: [(0, 0.1)], 850: [(1, 0.37)], 900: [(2, 0.63)], 950: [(3, 0.9)],            
              }
    replay_env = mab.BernoulliReplayBandit(replay=replay)

    # Build Agents
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.98)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=100)
    my_ts = mab.MyBernoulliTS(n_arms, gamma=0.98, n=30)

    
    # Build session
    replay_session = mab.Session(replay_env, [ts, dynamic_ts, sw_ts, my_ts])
    
    # Run session
    replay_session.run(n_step=1000, n_test=100, use_replay=True)
    
    #Plot results
    replay_env.plot_arms(render=False)
    ts.plot_estimates(render=False)
    dynamic_ts.plot_estimates(render=False)
    sw_ts.plot_estimates(render=False)
    my_ts.plot_estimates(render=False)
    replay_session.plot_all(render=False)
    plt.show()


def multiple_env():
    n_arms = 5
    n_step = 1000
    n_test = 30
    n_envs = 1000

    rewards = {"Oracle": 0,
               "Greedy Bernoulli": 0,
               "Thompson Sampling Bernoulli": 0,
               "Dynamic Thompson Sampling Bernoulli": 0,
               "Sliding Window Thompson Sampling Bernoulli": 0,
               "My Thompson Sampling Bernoulli": 0
               }

    parms = [(n_arms, n_step, n_test) for _ in range(n_envs)]

    pool = Pool(cpu_count())
    results = pool.starmap(launch_session, parms)
    pool.close()
    pool.join()

    for result in results:
        for agent in result:
            rewards[str(agent)] += result[agent]

    print(rewards)


def launch_session(n_arms, n_step, n_test):
    # Build environment
    env = mab.BernoulliDynamicBandit(n_arms, prob_of_change=0.003, fixed_action_prob=0.0, save_replay=True)

    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=n_step)

    # Build Agents
    greedy = mab.BernoulliGreedy(n_arms)
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.98)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=75)
    my_ts = mab.MyBernoulliTS(n_arms, gamma=0.98, n=20)
    agents = [greedy, ts, dynamic_ts, sw_ts, my_ts]

    # Build Env with replay
    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())

    # Build session
    replay_session = mab.Session(replay_env, agents)

    # Run session
    replay_session.run(n_step=n_step, n_test=n_test, use_replay=True)
    results = {agent: replay_session.get_reward_sum(agent)/n_step for agent in agents}
    results.update({"Oracle" : replay_session.get_reward_sum("Oracle")/n_step})
    return results


def non_stationary_bernoulli_custom():
    n_arms = 4
    # Build environment
    
    '''
    replay = {'probabilities': [0.0, 0.0, 0.1, 0.3], 
              250: [(0, 0.7)],
              500: [(1, 0.9)]
              }
    '''
    replay = {'probabilities': [0.9, 0.7, 0.1, 0.3], 
              250: [(0, 0.0)],
              500: [(1, 0.0)]
              }
    #'''
    replay_env = mab.BernoulliReplayBandit(replay=replay)

    # Build Agents
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.98)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms, n=100)
    my_ts = mab.MyBernoulliTS(n_arms, gamma=0.98, n=30)

    # Build session
    replay_session = mab.Session(replay_env, [ts, dynamic_ts, sw_ts, my_ts])
    
    # Run session
    replay_session.run(n_step=1000, n_test=100, use_replay=True)
    
    #Plot results
    replay_env.plot_arms(render=False)
    #ts.plot_estimates(render=False)
    dynamic_ts.plot_estimates(render=False)
    sw_ts.plot_estimates(render=False)
    my_ts.plot_estimates(render=False)
    replay_session.plot_all(render=False)
    plt.show()


if __name__ == "__main__":
    
    #stationary_bernoulli()
    #non_stationary_bernoulli()
    
    #non_stationary_bernoulli_paper()
    
    multiple_env()
    #non_stationary_bernoulli_custom()
