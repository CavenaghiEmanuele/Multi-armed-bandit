import multi_armed_bandit as mab

import pandas as pd

from pgmpy.models.BayesianNetwork import BayesianNetwork


def plain_bernoulli():
    actions = [str(i) for i in range(5)]
    available_actions = {'f1': {'0':['0','1','2'], '1':['3','4']}}
    context = {'f1':['0','1'], 'f2':['0','1','2']}

    env = mab.PlainBernoulliEnvironment(actions, context, available_actions)
    agents = [mab.RandomAgent(id=str(i), actions=actions) for i in range(1)]
    agents.append(mab.PlainTSBernoulli(id='0', actions=actions, context=context))

    session = mab.Session(env, agents)
    results = session.run(steps=25, experiments=10)
    
    mab.plot_cumulative_reward(results, errorbar=('ci', 99))
    return

def bayesian_bernoulli():

    actions = ['action' + str(i) for i in range(2)]
    available_actions = None #{'C': {'C0':['action0','action1','action2'], 'C1':['action3','action4']}}
    contexts = {'C':['C' + str(i) for i in range(2)]}
    bn = BayesianNetwork([('X', 'Y'), ('C', 'X'), ('C', 'Y')])

    #random = mab.RandomAgent(id='0', actions=actions)
    #plain = mab.PlainTSBernoulli(id='0', actions=actions, contexts=contexts)
    bayesian = mab.BayesianTSBernoulli(id='0', actions=actions, contexts=contexts, bn=bn)
    #causal = mab.CausalTSBernoulli(id='0', actions=actions, contexts=contexts, bn=bn)

    env = mab.BayesianBernoulliEnvironment(actions, contexts, available_actions, bn)
    
    session = mab.Session(env, [bayesian]) #[random, plain, bayesian, causal])
    results = session.run(steps=1000, experiments=1)

    print('#################################', 'ENV', '#################################')
    env.print_cpds()


    #mab.plot_cumulative_reward(results)
    #mab.plot_performed_actions(results)
    #mab.plot_visited_contexts(results)
    #env.plot_bn()
    

    return


def bayesian_bernoulli_by_feature():

    actions = {
        'action1':['0','0'],
        'action2':['0','1'],
        'action3':['1','0'],
        'action4':['1','1'],       
    }
    actions = pd.DataFrame.from_dict(actions, orient='index',
                       columns=['Feature1', 'Feature2'])
    
    
    available_actions = None #{'C': {'C0':['action0','action1','action2'], 'C1':['action3','action4']}}
    contexts = {'C':['C' + str(i) for i in range(2)], 'D':['D' + str(i) for i in range(2)] }
    bn = BayesianNetwork(
        [
            ('X', 'Y'), 
            ('C', 'X'), ('C', 'Y'), 
            ('D', 'C'),
            ('Feature1', 'X'), ('Feature1', 'Y'),
            ('Feature2', 'X'), ('Feature2', 'Y'),
        ]
    )

    env = mab.BayesianBernoulliEnvironment(actions, contexts, available_actions, bn)

    random = mab.RandomAgent(id='0', actions=list(actions.index))
    plain = mab.PlainTSBernoulli(id='0', actions=list(actions.index), contexts=contexts)
    bayesian = mab.BayesianTSBernoulli(id='0', actions=actions, contexts=contexts, bn=bn)
    causal = mab.CausalTSBernoulli(id='0', actions=actions, contexts=contexts, bn=bn)

    env = mab.BayesianBernoulliEnvironment(actions, contexts, available_actions, bn)

    session = mab.Session(env, [random, plain, bayesian, causal])
    results = session.run(steps=2000, experiments=1)

    print('#################################', 'ENV', '#################################')
    env.print_cpds()


    mab.plot_cumulative_reward(results)
    #mab.plot_performed_actions(results)
    #mab.plot_visited_contexts(results)
    #env.plot_bn()

    return


if __name__ == '__main__':

    #plain_bernoulli()
    bayesian_bernoulli()
    #bayesian_bernoulli_by_feature()
