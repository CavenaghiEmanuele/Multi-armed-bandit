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
    n_arms = env.get_n_arms()
    max_dsw_ts = mab.MaxDSWTS(n_arms=n_arms, gamma=0.99, n=50, store_estimates=False)
    min_dsw_ts = mab.MinDSWTS(n_arms=n_arms, gamma=0.99, n=50, store_estimates=False)
    mean_dsw_ts = mab.MeanDSWTS(n_arms=n_arms, gamma=0.99, n=50, store_estimates=False)        
    ts = mab.BernoulliThompsonSampling(n_arms=n_arms, store_estimates=False)
    sw_ts = mab.BernoulliSlidingWindowTS(n_arms=n_arms, n=300, store_estimates=False)
    d_ts = mab.DiscountedBernoulliTS(n_arms=n_arms, gamma=0.99, store_estimates=False)
    agent_list = [max_dsw_ts, min_dsw_ts, mean_dsw_ts, ts, sw_ts, d_ts]
    
    session = mab.Session(env=env, agent=agent_list)
    session.run(n_step=2523, n_test=30, use_replay=True)
    env.plot_arms(render=True)
    
    session.plot_all()
    
