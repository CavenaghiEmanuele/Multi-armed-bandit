from multi_armed_bandit import session
import pandas as pd
import multi_armed_bandit as mab


if __name__ == '__main__':
    
    df = pd.read_csv('datasets/baltimore_crime/modified_baltimore_crime.csv', parse_dates=['CrimeDate'])
    df.drop('CrimeTime', axis=1, inplace=True)
    
    districts = list(df.District.unique())
    days = df.CrimeDate.unique()
    
    frequencies = {}
    for day in days:
        day_df = df.loc[df['CrimeDate'] == day].groupby('District').count().reset_index()
        
        frequencies.update(
            {day : [(districts.index(row['District']), row['CrimeDate'] / day_df['CrimeDate'].sum())
                 for _, row in day_df.iterrows()]
            }
        )
        
    replay = {}
    for i, value in enumerate(frequencies.values()):
        if i == 0: replay.update({'probabilities' : [x[1] for x in value]})
        else: replay.update({i : value})
    

    env = mab.BernoulliReplayBandit(replay=replay)
    session = mab.Session(env=env, agent=[])
    session.run(n_step=2523, use_replay=True)
    env.plot_arms(render=True)
    
