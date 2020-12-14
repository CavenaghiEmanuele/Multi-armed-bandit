import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.io.arff import loadarff 


if __name__ == '__main__':
    dataset_list = [f for f in listdir('.') if isfile(join('.', f)) and f.endswith(".arff")]
    
    for dataset_name in dataset_list:
        df = pd.DataFrame(loadarff(dataset_name)[0])
        df.drop([c for c in df.columns if c != 'class'], 
                axis=1, 
                inplace=True)
        dataset_name = dataset_name.replace('INSECTS-', '')
        dataset_name = dataset_name.replace('_norm.arff', '')   
        df.to_csv(dataset_name + '.csv', index=False)
