import multi_armed_bandit as mab

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
    env = mab.BernoulliDynamicBandit(n_arms, prob_of_change=0.001, fixed_action_prob=0.1, save_replay=True)
    
    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=2000)
    env.plot_arms(render=True)

    # Build Env with replay
    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())

    # Build Agents
    greedy = mab.BernoulliGreedy(n_arms)
    ucb = mab.BernoulliUCB(n_arms, c = 1)
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.99)
    
    # Build session
    replay_session = mab.Session(replay_env, [greedy, ucb, ts, dynamic_ts])
    
    # Run session
    replay_session.run(n_step=2000, n_test=1000, use_replay=True)
    
    #Plot results
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
    greedy = mab.BernoulliGreedy(n_arms)
    ucb = mab.BernoulliUCB(n_arms, c = 1)
    ts = mab.BernoulliThompsonSampling(n_arms)
    dynamic_ts = mab.DynamicBernoulliTS(n_arms, gamma=0.95)
    
    # Build session
    replay_session = mab.Session(replay_env, [greedy, ucb, ts, dynamic_ts])
    
    # Run session
    replay_session.run(n_step=1000, n_test=10, use_replay=True)
    
    #Plot results
    replay_env.plot_arms(render=False)
    greedy.plot_estimates(render=False)
    ucb.plot_estimates(render=False)
    ts.plot_estimates(render=False)
    dynamic_ts.plot_estimates(render=False)
    replay_session.plot_all(render=False)
    plt.show()
    

if __name__ == "__main__":
    
    #stationary_bernoulli()
    #non_stationary_bernoulli()
    non_stationary_bernoulli_paper()
