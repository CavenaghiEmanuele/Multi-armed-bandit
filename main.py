import multi_armed_bandit as mab


if __name__ == '__main__':

    n_arms=10

    agents = [mab.RandomAgent(id=str(i), n_arms=n_arms) for i in range(2)]
    env = mab.BernoulliEnvironment(n_arms=n_arms)
    session = mab.Session(env=env, agents=agents)

    results = session.run(steps=5, experiments=2)
    
    mab.plot_cumulative_reward(results)
