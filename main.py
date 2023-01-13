import multi_armed_bandit as mab

from pgmpy.models.BayesianNetwork import BayesianNetwork


def plain_bernoulli():
    actions = [str(i) for i in range(5)]
    available_actions = {'f1': {'0':['0','1','2'], '1':['3','4']}}
    states = {'f1':['0','1'], 'f2':['0','1','2']}

    env = mab.PlainBernoulliEnvironment(actions, states, available_actions)
    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(1)]
    agents.append(mab.PlainTSBernoulli(id='0', actions=actions, states=states))

    session = mab.Session(env, agents)
    results = session.run(steps=25, experiments=10)
    
    mab.plot_cumulative_reward(results, errorbar=('ci', 99))
    return

def bayesian_bernoulli():

    actions = ['action' + str(i) for i in range(3)]
    available_actions = None
    states = {'C':['C' + str(i) for i in range(2)], 'D':['D' + str(i) for i in range(2)]}
    bn = BayesianNetwork([('X', 'Y'), ('C', 'X'), ('C', 'Y'), ('D','C')])

    env = mab.BayesianBernoulliEnvironment(actions, states, available_actions, bn)
    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(1)]
    agents.append(mab.PlainTSBernoulli(id='0', actions=actions, states=states))
    
    session = mab.Session(env, agents)
    results = session.run(steps=1000, experiments=20)

    env.print_cpds()

    #mab.plot_cumulative_reward(results)
    #mab.plot_performed_actions(results)
    mab.plot_visited_states(results)
    #env.plot_bn()
    

    return


if __name__ == '__main__':

    #plain_bernoulli()
    bayesian_bernoulli()
    