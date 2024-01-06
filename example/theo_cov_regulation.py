from torchpm.module import *
from torchpm.data import *
import pandas as pd
import numpy as np

def make_dataset(rng : np.random.Generator):
    theo_df = pd.read_csv('example/THEO.csv', na_values='.')
    theo_df['BWT'] = (theo_df['BWT'] - theo_df['BWT'].mean()) / theo_df['BWT'].std()
    
    covariance_matrix = [[1,     0.6**2, 0.3 **2],
                         [0.6**2,  1,    0],
                         [0.3**2,  0,    1]]
    
    sample = rng.multivariate_normal([0,0,0], covariance_matrix, [5])
    print(sample)
    dataset_config = MixedEffectsTimeDatasetConfig(
        dv_column_names=['CONC'],
        iv_column_names=['BWT'],
        time_column_name='TIME',
        id_column_name='ID')
    dataset = MixedEffectsTimeDataset(theo_df, dataset_config)


if __name__ == "__main__": 
    rng = np.random.default_rng(42)

    for _ in range(10):
        make_dataset(rng)