import pandas as pd


if __name__ == '__main__':
    
    dataset = pd.read_csv('baltimore_crime.csv', parse_dates=['CrimeDate'])
    dataset.drop(['CrimeCode', 'Premise', 'Description', 'Latitude', 'Longitude', 
                  'Location 1', 'vri_name1', 'Total Incidents', 'Post', 
                  'Neighborhood', 'Inside/Outside', 'Weapon', 'Location'], 
                 axis=1, 
                 inplace=True)
    
    dataset.to_csv('modified_baltimore_crime.csv', index=False)
    