import multi_armed_bandit as mab



if __name__ == '__main__':

    n_arms = 10
    env = mab.environments.BernoulliContextualBandit(n_arms=n_arms, context_dim=10)
    #env = mab.environments.BernoulliBandit(n_arms=n_arms)
    

    random = mab.algorithms.RandomAlgo(n_arms=n_arms)
    ts = mab.algorithms.BernoulliThompsonSampling(n_arms=n_arms)
    greedy = mab.algorithms.BernoulliGreedy(n_arms=n_arms)

    session = mab.Session(env, [ts, random])

    session.run(1000, 100)

    session.plot_all()
