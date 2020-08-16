import multi_armed_bandit as mab

import matplotlib.pyplot as plt


def stationary_bernoulli():
    n_arms = 5
    # Build environment
    env = mab.BernoulliBandit(n_arms)
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


if __name__ == "__main__":
    
    #stationary_bernoulli()
    non_stationary_bernoulli()