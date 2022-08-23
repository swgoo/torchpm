import unittest
import torch as tc
from torch import nn
from torchpm import covariate, models, ode, predfunc, loss
from torchpm import dataset
from torchpm.dataset import PMDataset, EssentialColumns
from torchpm.parameter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__' :
    unittest.main()

class DatasetTest(unittest.TestCase) :
    def test_pmdataset(self) :
        dataframe = pd.read_csv('examples/THEO.csv')
        dataset = PMDataset(dataframe)
        pass

    def test_essential_columns(self) :
        print(EssentialColumns.AMT.value)
        print(EssentialColumns.AMT.dtype)
        print(EssentialColumns.ID.value)
        print(EssentialColumns.ID.dtype)
        pass
