from dataclasses import dataclass
import enum
from typing import Dict, Iterable, List, Optional, OrderedDict, Union

import numpy as np
from pyrsistent import s
from regex import F
from sympy import Id
import torch as tc
from scipy import stats

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
    ID : int = 1
    TIME : float = 0
    AMT : float = 0
    RATE : float = 0
    DV : float = 0
    MDV : int = 0
    CMT : int = 0
    
    def __post_init__(self):
        if len(set(EssentialColumns.get_list()) - set(self.column_names)) > 0 :
            raise Exception('column_names must contain EssentialColumns')

        for k, v in self.covariates.items() :
            setattr(self, k, v)

    def make_record_list(self) -> List[float]:
        return [float(getattr(self, col)) for col in self.column_names]

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
    from .ode import EquationConfig

    def __init__(self,  
                equation_config : EquationConfig,
                column_names : List[str],
                dosing_interval : float,
                target_trough_concentration : float = 0.,
                sampling_times_after_dosing_time : List[float] = [],
                device: tc.device = tc.device("cpu"),
                include_trough_before_dose : bool = False,
                include_last_trough : bool = False,
                repeats : int = 10,):
        covariate_names = set(column_names) - set(EssentialColumns.get_list())
        covariate_names = list(covariate_names)

        covariates = OrderedDict()
        for name in covariate_names :
            covariates[name] = 0.
        
        dataset : List[List[float]]= []
        for i in range(repeats) :
            dosing_time = dosing_interval*i
            trough_sampling_times_after_dose = dosing_interval * (i+1) - 1e-6
            
            record_dose = Record(
                    column_names = column_names,
                    covariates=covariates,
                    TIME = dosing_time,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    CMT=equation_config.administrated_compartment_num,
                    MDV=1)
            dataset.append(record_dose.make_record_list())            
            
            for sampling_time_after_dose in sampling_times_after_dosing_time :
                cur_time = dosing_time + sampling_time_after_dose
                if cur_time >= trough_sampling_times_after_dose :
                    break
                else :
                    record_sampling = Record(
                            column_names = column_names,
                            covariates=covariates,
                            TIME = cur_time,
                            ID = 1,
                            AMT = 0,
                            RATE = 1 if equation_config.is_infusion else 0,
                            CMT=equation_config.observed_compartment_num,
                            MDV=0)
                    dataset.append(record_sampling.make_record_list())
            if include_trough_before_dose and i < repeats - 1 :
                record_trough = Record(
                        column_names = column_names,
                        covariates = covariates,
                        ID = 1,
                        AMT = 0,
                        RATE = 1 if equation_config.is_infusion else 0,
                        TIME=trough_sampling_times_after_dose - 1e-6,
                        DV = target_trough_concentration,
                        CMT=equation_config.observed_compartment_num,
                        MDV=0)
                
                dataset.append(record_trough.make_record_list())
        if include_last_trough:
            record_trough = Record(
                    column_names = column_names,
                    covariates = covariates,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    TIME=dosing_interval*repeats,
                    DV = target_trough_concentration,
                    CMT=equation_config.observed_compartment_num,
                    MDV=0)
            dataset.append(record_trough.make_record_list())
        
        numpy_dataset = np.array(dataset)
        
        super().__init__(numpy_dataset, column_names, device)
 