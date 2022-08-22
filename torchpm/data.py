import enum
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, OrderedDict, Set

import numpy as np
import torch as tc
from scipy import stats
import pandas as pd

from torch.utils.data import Dataset


class EssentialColumns(enum.Enum) :
    ID = 'ID'
    TIME = 'TIME'
    AMT = 'AMT'
    RATE = 'RATE'
    DV = 'DV'
    MDV = 'MDV'
    CMT = 'CMT'

    @classmethod
    def get_set(cls) -> Set[str] :
        return set([elem.value for elem in cls])

    @classmethod
    def get_list(cls) -> List[str] :
        return list(cls.get_set())
    
    @classmethod
    def check_essential_column(cls, column_names: Iterable[str]) -> None:
        if len(set(EssentialColumns.get_set()) - set(column_names)) > 0 :
            raise Exception('column_names must contain EssentialColumns')

class PMDataset(Dataset):
    
    def __init__(self, 
                 dataframe : pd.DataFrame,
                 **kwargs):
        super().__init__(**kwargs)
        EssentialColumns.check_essential_column(dataframe.columns)       
        
        for col in dataframe.columns :
            if col == EssentialColumns.ID.value :
                dataframe[col] = dataframe[col].astype(str)
            else :
                dataframe[col] = dataframe[col].astype(float)
        
        self.ids = list(dataframe[EssentialColumns.ID.value].sort_values(0).unique())
        
        self.datasets_by_id : Dict[str, Dict[str, tc.Tensor]] = {}
        for id in self.ids :
            id_mask = dataframe[EssentialColumns.ID.value] == id
            dataset_by_id = dataframe.loc[id_mask]
            self.datasets_by_id[id] = {col: tc.tensor(dataset_by_id[col].values) for col in dataframe.columns}

        self.len = len(self.datasets_by_id.keys())

    def __getitem__(self, idx):
        id = self.ids[idx]
        return self.datasets_by_id[id]

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

    def __init__(self, data : PMDataset, sizes : List[int], devices : List[tc.DeviceObjType]):
        
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

class OptimalDesignDataset(PMDataset):
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
        covariate_names = set(column_names) - set(EssentialColumns.get_set())
        covariate_names = list(covariate_names)

        covariates = OrderedDict()
        for name in covariate_names :
            covariates[name] = 0.
        
        dataset : List[List[float]]= []
        for i in range(repeats) :
            dosing_time = dosing_interval*i
            trough_sampling_times_after_dose = dosing_interval * (i+1) - 1e-6
            
            record_dose = PMRecord(
                    column_names = column_names,
                    TIME = dosing_time,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    CMT=equation_config.administrated_compartment_num,
                    MDV=1,
                    **covariates)
            dataset.append(record_dose.make_record_list())            
            
            for sampling_time_after_dose in sampling_times_after_dosing_time :
                cur_time = dosing_time + sampling_time_after_dose
                if cur_time >= trough_sampling_times_after_dose :
                    break
                else :
                    record_sampling = PMRecord(
                            column_names = column_names,
                            TIME = cur_time,
                            ID = 1,
                            AMT = 0,
                            RATE = 1 if equation_config.is_infusion else 0,
                            CMT=equation_config.observed_compartment_num,
                            MDV=0,
                            **covariates)
                    dataset.append(record_sampling.make_record_list())
            if include_trough_before_dose and i < repeats - 1 :
                record_trough = PMRecord(
                        column_names = column_names,
                        ID = 1,
                        AMT = 0,
                        RATE = 1 if equation_config.is_infusion else 0,
                        TIME=trough_sampling_times_after_dose - 1e-6,
                        DV = target_trough_concentration,
                        CMT=equation_config.observed_compartment_num,
                        MDV=0,
                        **covariates)
                
                dataset.append(record_trough.make_record_list())
        if include_last_trough:
            record_trough = PMRecord(
                    column_names = column_names,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    TIME=dosing_interval*repeats,
                    DV = target_trough_concentration,
                    CMT=equation_config.observed_compartment_num,
                    MDV=0,
                    **covariates)
            dataset.append(record_trough.make_record_list())
        
        numpy_dataset = np.array(dataset, dtype=np.float32)
        
        super().__init__(numpy_dataset, column_names, device)
 