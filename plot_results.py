import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def multiple_envs_parameter_tuning_plot(type_change:str, grayscale:bool=False):
    agents = ['Discounted TS', 'Sliding Window TS', 'Max d-sw TS', 'Mean d-sw TS', 'Min d-sw TS']
    for agent in agents:
        path = 'results/multiple_envs/find_params/' + type_change + '/' + agent + '.csv'
        dataset = pd.read_csv(path)
        
        if grayscale: plt.style.use('grayscale')
        dataset.plot.box()
        
        plt.title('Agent: ' + agent , fontsize=35)
        plt.grid(axis='y')
        #plt.xlabel('', fontsize=35)
        #plt.ylabel('% of cumulative reward', fontsize=35)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)
        plt.tick_params(axis='both', which='major', labelsize=24)
    plt.show()

def multiple_envs_plot(type_change:str, grayscale:bool=False):
    path = 'results/multiple_envs/multiple_env_' + type_change + '_scaled.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    plt.ylabel('Relative to Oracle', fontsize=26)
    plt.xlabel('Probability of change', fontsize=26)
    x_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 0.02]
    plt.xticks(list(range(len(x_values))), x_values)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True, prop={'size': 28})
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.07)

    plt.show()


def custom_tests_plot_regret(test_number, grayscale:bool=False):
    path = 'results/custom_tests/' + str(test_number) + '_regret.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    #plt.title('Regret', fontsize=24)
    plt.legend(prop={'size': 24})
    plt.grid(axis='y')
    plt.xlabel('Step', fontsize=24)
    plt.ylabel('Regret', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.show()
    
def custom_tests_plot_reward_trace(test_number, grayscale:bool=False):
    path = 'results/custom_tests/' + str(test_number) + '_real_reward_trace.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.drop('Unnamed: 0', 1)
    
    if grayscale:
        plt.style.use('grayscale')
    dataset.plot(linewidth=3)
    
    #plt.title('Reward trace', fontsize=24)
    plt.legend(prop={'size': 24})
    plt.grid(axis='y')
    plt.xlabel('Step', fontsize=24)
    plt.ylabel('Sum of Cumulative Reward', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.show()


def real_dataset_plot_parameter_tuning(dataset_name:str, type_of_change:str='', grayscale:bool=False) -> None:
    agents = ['Discounted TS', 'Sliding Window TS', 'Max d-sw TS', 'Mean d-sw TS', 'Min d-sw TS']
    
    base_path = 'results/' + dataset_name + '/find_params/'
    if dataset_name == 'insects':
        base_path += type_of_change + '/imbalanced/'
   
    for agent in agents:
        path = base_path + agent + '.csv'
        dataset = pd.read_csv(path)
        dataset.drop(columns='tmp', inplace=True)
        
        if grayscale: plt.style.use('grayscale')
        dataset.plot.box()
        
        plt.title('Agent: ' + agent , fontsize=24)
        plt.grid(axis='y')
        plt.xlabel('', fontsize=20)
        plt.ylabel('% of cumulative reward', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()

def real_dataset_plot_reward_trace(dataset_name:str, type_of_change:str='', grayscale:bool=False) -> None:
    path = 'results/' + dataset_name + '/tests/'
    if dataset_name == 'insects':
        path += type_of_change + '/'
    path += 'reward_trace.csv'

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
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)
    plt.show()

def real_dataset_plot_reward_perc(dataset_name:str, type_of_change:str='', grayscale:bool=False) -> None:
    agents = ['Max d-sw TS', 'Min d-sw TS', 'Mean d-sw TS', 'Discounted TS', 'Sliding Window TS', 'Thompson Sampling']
    suffix_list = ['', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
    
    path = 'results/' + dataset_name + '/tests/'
    figure_title = '% of correct identified classes'
    if dataset_name == 'insects':
        path += type_of_change + '/'
        figure_title += ': ' + type_of_change
    path += 'reward_trace.csv'
    
    dataset = pd.read_csv(path)
    agent_results = [
        [
        dataset[agent+suffix][len(dataset.index)-1] / (len(dataset.index)-1) 
            for suffix in suffix_list
            ]
        for agent in agents
        ]
        
    if grayscale: plt.style.use('grayscale')
    plt.boxplot(agent_results, labels=agents)

    #plt.title(figure_title, fontsize=24)
    plt.grid(axis='y')
    plt.xlabel('', fontsize=20)
    plt.ylabel('% of correct identified classes', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)
    plt.show()
    
def bacteria_plot_reward_perc(grayscale:bool=False) -> None:
    path = 'results/bacteria/reward_perc.csv'
    df = pd.read_csv(path)
    df[['Max d-sw TS','Min d-sw TS', 'Mean d-sw TS', 'Thompson Sampling', 'Sliding Window TS', 'Discounted TS', 'Random']] = \
        df[['Max d-sw TS','Min d-sw TS', 'Mean d-sw TS', 'Thompson Sampling', 'Sliding Window TS', 'Discounted TS', 'Random']].div(df['Oracle'],axis=0)
    df.drop('Oracle', axis=1, inplace=True)
    
    if grayscale: plt.style.use('grayscale')
    df.plot.box()
    
    #plt.title('% of correct identified classes', fontsize=24)
    plt.grid(axis='y')
    plt.xlabel('', fontsize=20)
    plt.ylabel('Relative to Oracle', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.07)    
    plt.show()


if __name__ == "__main__":

    #########################################################
    # Plot multiple envs
    #########################################################
    '''
    #multiple_envs_parameter_tuning_plot(type_change='abrupt', grayscale=True) # incremental or abrupt
    multiple_envs_plot(type_change='abrupt', grayscale=False) # incremental or abrupt
    '''

    #########################################################
    # Plot custom tests
    #########################################################
    '''
    test_number = 3
    custom_tests_plot_regret(test_number=test_number, grayscale=False)
    custom_tests_plot_reward_trace(test_number=test_number, grayscale=False)
    '''

    #########################################################
    # Plot real dataset results
    #########################################################
    
    type_of_change = 'gradual' # abrupt, gradual, incremental-abrupt, incremental, incremental-reoccurring, out-of-control
    dataset_name = 'bacteria' # local_news, baltimore_crime, insects, bacteria

    # Plot bacteria results
    bacteria_plot_reward_perc()

    #real_dataset_plot_reward_perc(dataset_name=dataset_name, type_of_change=type_of_change)
    #real_dataset_plot_reward_trace(dataset_name=dataset_name, type_of_change=type_of_change)
    #real_dataset_plot_parameter_tuning(dataset_name=dataset_name, type_of_change=type_of_change)

    
    
