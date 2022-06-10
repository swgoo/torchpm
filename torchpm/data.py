from dataclasses import dataclass
import enum
from typing import Dict, Iterable, List, Optional, OrderedDict

import numpy as np
from pyrsistent import s
from regex import F
from sympy import Id
import torch as tc
from scipy import stats
# from .ode import DosageFormConfig   

class EssentialColumns(enum.Enum) :
    ID = 'ID'
    TIME = 'TIME'
    AMT = 'AMT'
    RATE = 'RATE'
    DV = 'DV'
    MDV = 'MDV'
    CMT = 'CMT'

    @classmethod
    def get_list(cls) -> List[str] :
        return [elem.value for elem in cls] 

@dataclass
class Record:
    column_names : List[str]
    covariates : OrderedDict[str, float]
    id : int = 1
    time : float = 0
    amt : float = 0
    rate : float = 0
    dv : float = 0
    mdv : int = 0
    cmt : int = 0
    

    def __post_init__(self):
        if len(set(EssentialColumns.get_list()) - set(self.column_names)) < 1 :
            raise Exception('column_names must contain EssentialColumns')

        for name in self.covariate_names :
            setattr(self, name, 0)
    def make_record_list(self):
        for name in self.column_names :
            att = getattr(self, name)

        return None

class CSVDataset(tc.utils.data.Dataset):  # type: ignore
    """
    Args:
        file_path: csv file path
        column_names: csv file's column names
        device: (optional) data loaded location 
    """
    
    def __init__(self, 
                 numpy_dataset : np.ndarray,
                 column_names : List[str],
                 device : tc.device = tc.device("cpu"),
                 normalization_column_names : Optional[List[str]] = None):

        self.column_names = column_names
        self.device = device
        
        self.normalization_column_names = normalization_column_names
        if self.normalization_column_names :
            for name in self.normalization_column_names :
                numpy_dataset[:, column_names.index(name)] = stats.zscore(numpy_dataset[:, column_names.index(name)])


        y_true_total = numpy_dataset[:,self.column_names.index(EssentialColumns.DV.value)]
        self.mean = {}
        for i in range(len(self.column_names)):
            self.mean[self.column_names[i]] = numpy_dataset[:,i].mean()
        self.std = {}
        for i in range(len(self.column_names)):
            self.std[self.column_names[i]] = numpy_dataset[:,i].std()
        ids, ids_start_idx = np.unique(numpy_dataset[:, column_names.index(EssentialColumns.ID.value)], return_index=True)
        ids_start_idx = ids_start_idx[1:]
        dataset_np = np.split(numpy_dataset, ids_start_idx)
        y_true_np = np.split(y_true_total, ids_start_idx)

        self.dataset = [tc.from_numpy(data_np).to(device) for data_np in dataset_np]
        self.y_true = [tc.from_numpy(y_true_cur).to(device) for y_true_cur in y_true_np]
        self.len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.y_true[index]

    def __len__(self):
        return self.len

class Partition(object):

    def __init__(self, data, index, device):
        self.data = data
        self.index = index
        self.device = device

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return (data.to(self.device) for data in self.data[data_idx])

class DataPartitioner(object):
    """
    Dataset for multiprocessing 
    Args:
        data: total dataset
        partitions: sizes for dividing dataset by a ID.
        device: data loaded locations
    """

    def __init__(self, data : CSVDataset, sizes : List[int], devices : List[tc.DeviceObjType]):
        
        if len(sizes) != len(devices) :
            raise Exception('sizes length must equal devices length.')

        self.data = data
        self.partitions = []
        self.devices = devices
        data_len = len(data)

        indexes = [x for x in range(0, data_len)]

        for size, device in zip(sizes, devices):
            part_len = int(size)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition_index : int):
        return Partition(self.data, self.partitions[partition_index], self.devices[partition_index])

    
class OptimalDesignDataset(CSVDataset):
    def __init__(self,  
                dosing_interval : float,
                observated_compartment_num : int,
                administrated_compartment_num : int,
                covariate_names: List[str], 
                device: tc.device = tc.device("cpu"),
                normalization_column_names: Optional[List[str]] = None):
        
        i=0
        while True :
            i+=1
            record = Record(EssentialColumns.get_list() + covariate_names, covariate_names=covariate_names)


        
        super().__init__(numpy_dataset, column_names, device, normalization_column_names)
 