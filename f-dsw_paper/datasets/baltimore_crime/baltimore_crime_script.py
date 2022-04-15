import pandas as pd


if __name__ == '__main__':
    
    dataset = pd.read_csv('original_baltimore_crime.csv', parse_dates=['CrimeDate'], keep_default_na=False)
    dataset.drop(['CrimeCode', 'Premise', 'Description', 'Latitude', 'Longitude', 
                  'Location 1', 'vri_name1', 'Total Incidents', 'Post', 
                  'Neighborhood', 'Inside/Outside', 'Weapon', 'Location'], 
                 axis=1, 
                 inplace=True)
    
    dataset.drop(dataset[dataset.District == ''].index, inplace=True)
    reversed_df = dataset.iloc[::-1]
    reversed_df.to_csv('baltimore_crime.csv', index=False)
    