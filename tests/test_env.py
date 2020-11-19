import multi_armed_bandit as mab


def test_gaussian_bandit():
    mean = [1.2, 1.3, 1.4]
    std_dev = [0.2, 0.3, 0.4]
    bandit = mab.GaussianBandit(n_arms=3, mean=mean, std_dev=std_dev)
    #bandit.plot_arms()

def test_bernoulli_bandit():
    probabilities = [0.2, 0.3, 0.4]
    bandit = mab.BernoulliBandit(n_arms=3, probabilities=probabilities)
    #bandit.plot_arms()

def test_bernoulli_Discounted_bandit():
    env = mab.BernoulliDiscountedBandit(n_arms=5, prob_of_change=0.0005, fixed_action_prob=0.2)
    session = mab.Session(env=env, agent=[])
    session.run(n_step=1000)
    #env.plot_arms(render=True)

def test_gaussian_Discounted_bandit():
    env = mab.GaussianDiscountedBandit(n_arms=5, prob_of_change=0.0005, fixed_action_prob=0.2)
    session = mab.Session(env=env, agent=[])
    session.run(n_step=1000)
    #env.plot_arms(render=True)

def test_bernoulli_replay_bandit():
    env = mab.BernoulliDiscountedBandit(n_arms=5, prob_of_change=0.001, fixed_action_prob=0.2, save_replay=True)
    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=3000)

    replay_env = mab.BernoulliReplayBandit(replay=env.get_replay())
    replay_session = mab.Session(replay_env, [])
    replay_session.run(n_step=3000, n_test=100, use_replay=True)
    #replay_env.plot_arms(render=True)

def test_gaussian_replay_bandit():
    env = mab.GaussianDiscountedBandit(n_arms=5, prob_of_change=0.001, fixed_action_prob=0.2, save_replay=True)
    # Generate replay
    session = mab.Session(env, [])
    session.run(n_step=3000)

    replay_env = mab.GaussianReplayBandit(replay=env.get_replay())
    replay_session = mab.Session(replay_env, [])
    replay_session.run(n_step=3000)
    #replay_env.plot_arms(render=True)
    
    assert env._action_value_trace == replay_env._action_value_trace
        
