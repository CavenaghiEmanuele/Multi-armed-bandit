import multi_armed_bandit as mab


if __name__ == '__main__':

    actions = [str(i) for i in range(10)]
    states = {'f1':[0,1], 'f2':[0,1,2,3]}
    #states = {'f1':[0,1]}

    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(1)]
    agents.append(mab.PlainTSBernoulli(id='0', actions=actions, states=states))

    env = mab.PlainBernoulliEnvironment(actions, states)
    session = mab.Session(env, agents)

    results = session.run(steps=1000, experiments=30)
    
    mab.plot_cumulative_reward(results)
