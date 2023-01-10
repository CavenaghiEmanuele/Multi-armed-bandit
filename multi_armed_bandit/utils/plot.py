import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cumulative_reward(data: pd.DataFrame, errorbar=('ci', 95)):
    data['cum_rew'] = data.groupby(['agent', 'experiment'])['reward'].cumsum()
    sns.lineplot(data=data, x='step', y='cum_rew', hue='agent', errorbar=errorbar)
    plt.show()


def plot_cumulative_regret(data: pd.DataFrame):
    raise NotImplementedError
    