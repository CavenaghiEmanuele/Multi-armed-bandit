import multi_armed_bandit as mab


if __name__ == '__main__':

    actions = [str(i) for i in range(5)]
    available_actions = {'f1': {'0':['0','1','2'], '1':['3','4']}}

    states = {'f1':['0','1'], 'f2':['0','1','2','3']}
    #states = {'f1':['0','1']}

    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(1)]
    agents.append(mab.PlainTSBernoulli(id='0', actions=actions, states=states))

    env = mab.PlainBernoulliEnvironment(actions, states, available_actions)
    session = mab.Session(env, agents)

    results = session.run(steps=1000, experiments=10)
    
    #mab.plot_cumulative_reward(results, errorbar=('ci', 99))
