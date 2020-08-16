import multi_armed_bandit as mab


def test_bernoulli_algorithms():
    n_arms = 4
    env = mab.BernoulliBandit(n_arms)
      
    greedy_agent = mab.BernoulliGreedy(n_arms)    
    ts_agent = mab.BernoulliThompsonSampling(n_arms)
    ucb_agent = mab.BernoulliUCB(n_arms, c=1)
    dynamic_ts_agent = mab.DynamicBernoulliTS(n_arms, gamma=0.99)
 
    session = mab.Session(env, [greedy_agent, ts_agent, ucb_agent, dynamic_ts_agent])
    session.run(3000)
    '''
    env.plot_arms(render=False)
    session.plot_action_selection(render=False)
    session.plot_regret(render=False)
    session.plot_real_rewards_sum(render=False)        
    plt.show()
    '''
    
def test_gaussian_algorithms():
    n_arms = 4
    env = mab.GaussianBandit(n_arms, std_dev=0.7)
    
    greedy_agent = mab.GaussianGreedy(n_arms)
    ts_agent = mab.GaussianThompsonSampling(n_arms, decay_rate=0.90)
    ucb_agent = mab.GaussianUCB(n_arms, decay_rate=0.9, c=1)
    
    session = mab.Session(env, [greedy_agent, ts_agent, ucb_agent])
    session.run(3000)
    '''
    env.plot_arms(render=False)
    session.plot_action_selection(render=False)
    session.plot_regret(render=False)
    session.plot_real_rewards_sum(render=False)
    plt.show()
    '''
    