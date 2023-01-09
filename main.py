import multi_armed_bandit as mab


if __name__ == '__main__':

    actions = [str(i) for i in range(10)]

    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(2)]
    agents.append(mab.TSBernoulli(id='0', actions=actions))

    env = mab.BernoulliEnvironment(actions=actions)
    session = mab.Session(env=env, agents=agents)

    results = session.run(steps=1000, experiments=10)
    
    mab.plot_cumulative_reward(results)
