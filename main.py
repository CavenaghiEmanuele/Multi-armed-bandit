import multi_armed_bandit as mab


if __name__ == '__main__':

    n_arms=10

    agents = [mab.RandomAgent(id=str(i), n_arms=n_arms) for i in range(2)]
    agents.append(mab.TSBernoulli(id='0', n_arms=n_arms))

    env = mab.BernoulliEnvironment(n_arms=n_arms)
    session = mab.Session(env=env, agents=agents)

    results = session.run(steps=1000, experiments=10)
    
    mab.plot_cumulative_reward(results)
