import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cumulative_reward(data: pd.DataFrame, errorbar=('ci', 95)):
    data['cum_rew'] = data.groupby(['agent', 'experiment'])['reward'].cumsum()
    sns.lineplot(data=data, x='step', y='cum_rew', hue='agent', errorbar=errorbar)
    plt.show()

def plot_performed_actions(data: pd.DataFrame):
    # plus 1 becouse cumcount starts from 0
    data['cum_action'] = data.groupby(['agent', 'experiment', 'action'])['action'].cumcount()+1
    sns.relplot(data=data, x='step', y='cum_action', hue='action', row='agent', kind='line')
    plt.show()

def plot_visited_states(data: pd.DataFrame):
    # plus 1 becouse cumcount starts from 0
    data['cum_state'] = data.groupby(['agent', 'experiment', 'state'])['state'].cumcount()+1
    sns.relplot(data=data, x='step', y='cum_state', hue='state', kind='line')
    plt.show()

def plot_cumulative_regret(data: pd.DataFrame):
    raise NotImplementedError
    