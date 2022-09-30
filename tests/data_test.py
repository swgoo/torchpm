import unittest
import torch as tc
from torch import nn
from torchpm import lossfunc, models, ode, predfunc
from torchpm import data
from torchpm.data import PMDataset, EssentialColumns, OptimalDesignDataset
from torchpm.para import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__' :
    unittest.main()

class DatasetTest(unittest.TestCase) :
    def test_pmdataset(self) :
        dataframe = pd.read_csv('examples/THEO.csv')
        dataset = PMDataset(dataframe)
        return dataset 

    def test_essential_columns(self) :
        print(EssentialColumns.AMT.value)
        print(EssentialColumns.AMT.dtype)
        print(EssentialColumns.ID.value)
        print(EssentialColumns.ID.dtype)
        pass

    def test_optimal_design_dataset(self):
        odd = OptimalDesignDataset(
            mean_of_covariate={'BWT': 70.},
            is_infusion = False,
            dosing_interval = 12.,
            sampling_times_after_dosing_time=[0,0.5,1],
            target_trough_concentration=10.,
            include_trough_before_dose = False,
            include_last_trough=True)
        pass
