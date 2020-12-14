import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def multiple_envs_plot(type_change:str, grayscale:bool=False):
    path = 'results/multiple_envs/Multiple_env_' + type_change + '_scaled.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    plt.title('Relative to Oracle', fontsize=24)
    plt.ylabel('Relative to Oracle', fontsize=24)
    plt.xlabel('Probability of change', fontsize=24)
    x_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02]
    plt.xticks(list(range(len(x_values))), x_values)
    plt.grid()
    plt.legend(prop={'size': 24})
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)

    plt.show()


def custom_tests_plot_regret(test_number, grayscale:bool=False):
    path = 'results/custum_tests/custom_test_' + str(test_number) + '_regret.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    plt.title('Regret', fontsize=24)
    plt.xlim(-10, 1010)
    plt.ylim(-0.01, 0.71)
    plt.grid()
    plt.legend(prop={'size': 24})
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

    plt.show()
    
def custom_tests_plot_reward_trace(test_number, grayscale:bool=False):
    path = 'results/custum_tests/custom_test_' + str(test_number) + '_real_reward_trace.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    plt.title('Reward trace', fontsize=24)
    plt.xlim(-10, 1010)
    plt.grid()
    plt.legend(prop={'size': 24})
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)

    plt.show()


def yahoo_plot_reward_trace(day, grayscale:bool=False) -> None:
    path = 'results/Yahoo/day' + str(day) + '/reward_trace_day' + str(day) + '.csv'

    dataset = pd.read_csv(path)
    dataset = dataset.add_suffix('')
    
    agent_list = ['Max d-sw TS', 'Min d-sw TS', 'Mean d-sw TS',
            'Thompson Sampling', 'Sliding Window TS', 'Discounted TS', 'random']
    suffix_list = ['', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
    
    plt.figure()
    if grayscale: plt.style.use('grayscale')
    
    for agent in agent_list:
        plt.plot(np.mean([dataset[agent + suffix].values for suffix in suffix_list], axis=0), label=agent, linewidth=3)      

    plt.title('Reward trace', fontsize=24)
    plt.grid()
    plt.legend(prop={'size': 24})
    plt.xlabel('Iterations grouped by 1000', fontsize=20)
    plt.ylabel('Reward averaged over 1000 iteration', fontsize=20)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)
    plt.show()
    
def yahoo_plot_reward_perc(day, grayscale:bool=False) -> None:
    path = 'results/Yahoo/day' + str(day) + '/reward_perc_day' + str(day) + '.csv'
    dataset = pd.read_csv(path)
    
    if grayscale: plt.style.use('grayscale')
    dataset.plot.box()
    
    plt.title('% of correct suggested site', fontsize=24)
    plt.grid(axis='y')
    plt.xlabel('', fontsize=20)
    plt.ylabel('% of correct suggested site', fontsize=20)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()
    
def yahoo_plot_all_reward_perc(day) -> None:
    path = 'results/Yahoo/day' + str(day) +  '/all_reward_perc_day' + str(day) +'.csv'
    _ = sns.catplot(
            x="Session", 
            y="% of correct suggested site",
            palette="rocket",
            col="Agent",
            data=pd.read_csv(path), kind="box",
            height=4, aspect=.7)        
    plt.show()


def baltimore_crime_plot_parameter_tuning(grayscale:bool=False) -> None:
    agents = ['Discounted TS', 'Sliding Window TS', 'Max d-sw TS', 'Mean d-sw TS', 'Min d-sw TS']
    for agent in agents:
        path = 'results/baltimore_crime/find_params/' + agent + '.csv'
        dataset = pd.read_csv(path)
        
        if grayscale: plt.style.use('grayscale')
        dataset.plot.box()
        
        plt.title('Agent: ' + agent , fontsize=24)
        plt.grid(axis='y')
        plt.xlabel('', fontsize=20)
        plt.ylabel('% of cumulative reward', fontsize=20)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()


def insects_plot_reward_trace(type_of_change:str, balanced_imbalanced:str, grayscale:bool=False) -> None:
    path = 'results/insects/tests/' + type_of_change + '/' + balanced_imbalanced + '/reward_trace.csv'

    dataset = pd.read_csv(path)
    dataset = dataset.add_suffix('')
    
    agent_list = ['Max d-sw TS', 'Min d-sw TS', 'Mean d-sw TS',
            'Thompson Sampling', 'Sliding Window TS', 'Discounted TS', 'random']
    suffix_list = ['', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
    
    plt.figure()
    if grayscale: plt.style.use('grayscale')
    
    for agent in agent_list:
        plt.plot(np.mean([dataset[agent + suffix].values for suffix in suffix_list], axis=0), label=agent, linewidth=3)      

    plt.title('Cumulative Reward trace', fontsize=24)
    plt.grid()
    plt.legend(prop={'size': 24})
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Cumulative Reward (averaged over 10 runs)', fontsize=20)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)
    plt.show()
    
def insects_plot_reward_perc(type_of_change:str, balanced_imbalanced:str, grayscale:bool=False) -> None:
    path = 'results/insects/tests/' + type_of_change + '/' + balanced_imbalanced + '/reward_perc.csv'
    dataset = pd.read_csv(path)
    
    if grayscale: plt.style.use('grayscale')
    dataset.plot.box()
    
    plt.title('% of correct identified classes', fontsize=24)
    plt.grid(axis='y')
    plt.xlabel('', fontsize=20)
    plt.ylabel('% of correct identified classes', fontsize=20)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()

def insects_plot_parameter_tuning(type_of_change:str, balanced_imbalanced:str, grayscale:bool=False) -> None:
    agents = ['Discounted TS', 'Sliding Window TS', 'Max d-sw TS', 'Mean d-sw TS', 'Min d-sw TS']
    for agent in agents:
        path = 'results/insects/find_params/' + type_of_change + '/' + balanced_imbalanced + '/' + agent + '.csv'
        dataset = pd.read_csv(path)
        dataset.drop(columns='tmp', inplace=True)
        
        if grayscale: plt.style.use('grayscale')
        dataset.plot.box()
        
        plt.title('Agent: ' + agent , fontsize=24)
        plt.grid(axis='y')
        plt.xlabel('', fontsize=20)
        plt.ylabel('% of cumulative reward', fontsize=20)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()

if __name__ == "__main__":
    
    #########################################################
    # Plot multiple envs results
    #########################################################
    '''
    multiple_envs_plot(type_change='incremental', grayscale=False) # incremental or abrupt
    '''
    
    #########################################################
    # Plot custom tests
    #########################################################
    '''
    test_number = 2
    custom_tests_plot_regret(test_number=test_number, grayscale=False)
    custom_tests_plot_reward_trace(test_number=test_number, grayscale=False)
    '''
    
    #########################################################
    # Plot Yahoo! find parameters
    #########################################################
    '''
    yahoo_plot_all_reward_perc(day=2)
    '''
    
    #########################################################
    # Plot Yahoo! tests days
    #########################################################
    '''
    day=1
    yahoo_plot_reward_trace(day=day, grayscale=False)
    yahoo_plot_reward_perc(day=day, grayscale=True)
    '''

    #########################################################
    # Plot Baltimore Crime dataset parameter tuning
    #########################################################
    '''
    baltimore_crime_plot_parameter_tuning(grayscale=False)
    '''
    
    #########################################################
    # Plot insects tests 
    #########################################################
    
    type_of_change = 'abrupt' # abrupt, gradual, incremental-abrupt, incremental, incremental-reoccuring
    balanced_imbalanced = 'imbalanced' # balanced, imbalanced
    #insects_plot_reward_perc(type_of_change, balanced_imbalanced, grayscale=False)
    #insects_plot_reward_trace(type_of_change, balanced_imbalanced, grayscale=False)
    insects_plot_parameter_tuning(type_of_change, balanced_imbalanced, grayscale=False)
