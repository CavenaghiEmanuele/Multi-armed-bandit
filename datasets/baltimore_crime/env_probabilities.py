import pandas as pd


if __name__ == '__main__':
    
    dataset = pd.read_csv('modified_baltimore_crime.csv', parse_dates=['CrimeDate'])


    print(dataset)