from copy import deepcopy
import enum
from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Literal, Optional, OrderedDict, Set, Tuple, Type

import numpy as np

import torch
import pandas as pd

from torch import Tensor, tensor
from torch.nn import functional as F

from torch.utils.data import Dataset

def get_id(dataset : Dict[str, Tensor]) -> str:
    return str(int(dataset[EssentialColumns.ID.value][0]))

class EssentialColumnDtypes(enum.Enum) :
    ID = int
    TIME = float
    AMT = float
    RATE = float
    DV = float
    MDV = int
    CMT = int

@enum.unique
class EssentialColumns(enum.Enum) :

    def __init__(self, value) :
        self.dtype = EssentialColumnDtypes[self.name].value

    ID = 'ID'
    TIME = 'TIME'
    AMT = 'AMT'
    RATE = 'RATE'
    DV = 'DV'
    MDV = 'MDV'
    CMT = 'CMT'

    @classmethod
    def get_name_set(cls) -> Set[str] :
        return set(cls.get_name_list())

    @classmethod
    def get_name_list(cls) -> List[str] :
        return [elem.value for elem in cls]
    
    @classmethod
    def check_inclusion_of_names(cls, column_names: Iterable[str]) -> None:
        if len(set(EssentialColumns.get_name_set()) - set(column_names)) > 0 :
            raise Exception('column_names must contain EssentialColumns')
    
    @classmethod
    def int_column_names(cls) -> List[str]:
        return list([elem.value for elem in cls if elem.dtype is int])

    @classmethod
    def float_column_names(cls):
        return list([elem.value for elem in cls if elem.dtype is float])

class PMDataset(Dataset):
    def __init__(self, 
                 dataframe : pd.DataFrame,
                 **kwargs):
        super().__init__()
        
        EssentialColumns.check_inclusion_of_names(dataframe.columns)
        self._covariate_names : Set[str] = set([name for name in dataframe.columns]) - EssentialColumns.get_name_set()

        self.column_names = list(dataframe.columns)
        self.mean_of_columns : Dict[str, float] = dict(dataframe.mean(axis=0, skipna=True))
        
        for col in dataframe.columns :
            if col in EssentialColumns.int_column_names() :
                dataframe[col] = dataframe[col].astype(int)
            else :
                dataframe[col] = dataframe[col].astype(np.float32)
        self._ids : List[int] = dataframe[EssentialColumns.ID.value].sort_values(axis = 0).unique().tolist()

        self.max_record_length = 0
        self.record_lengths : Dict[int, int] = {}
        self.datasets_by_id : Dict[int, Dict[str, Tensor]] = {}
        for id in self.ids :
            id_mask = dataframe[EssentialColumns.ID.value] == id
            dataset_by_id = dataframe.loc[id_mask]
            self.datasets_by_id[id] = {}
            length = len(dataset_by_id)
            self.max_record_length = max(length, self.max_record_length)
            self.record_lengths[id] = length

        for id in self.ids :
            for col in dataframe.columns :
                id_mask = dataframe[EssentialColumns.ID.value] == id
                dataset_by_id = dataframe.loc[id_mask]
                length = self.record_lengths[id]
                t = tensor(dataset_by_id[col].values)
                self.datasets_by_id[id][col] = F.pad(t, (0, self.max_record_length - length))
                self.datasets_by_id[id][col] = self.datasets_by_id[id][col]
        self.len = len(self.datasets_by_id.keys())

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        id = self.ids[idx]
        return self.datasets_by_id[id]

    def __len__(self):
        return self.len
    
    @property
    def covariate_names(self) -> Set[str]:
        return self._covariate_names

    @property
    def ids(self) :
        return self._ids


@dataclass
class PMRecord:
    ID : int = 1
    TIME : float = 0
    AMT : float = 0
    RATE : float = 0
    DV : float = 0
    MDV : int = 0
    CMT : int = 0

class OptimalDesignDataset(PMDataset):
    def __init__(self,  
                is_infusion : bool,
                dosing_interval : float,
                mean_of_covariate : Dict[str, float] = dict(),
                target_trough_concentration : float = 0.,
                administrated_compartment_num : int = 0,
                observed_compartment_num : int = 0,
                sampling_times_after_dosing_time : List[float] = [],
                include_trough_before_dose : bool = False,
                include_last_trough : bool = False,
                repeats : int = 10,
                **kwargs):

        covariate_name_list = set(mean_of_covariate.keys()) - EssentialColumns.get_name_set()
        covariate_name_list = list(covariate_name_list)
        df_columns = EssentialColumns.get_name_list() + covariate_name_list

        for k in mean_of_covariate.keys() :
            if k in EssentialColumns.get_name_set():
                del mean_of_covariate[k]
        
        df = pd.DataFrame(columns=df_columns)
        def add_row(record : PMRecord):
            record_dose_dict = deepcopy(mean_of_covariate) | asdict(record)
            return pd.concat([df,pd.DataFrame(record_dose_dict, index=[0])], ignore_index=True)
        
        for i in range(repeats) :
            dosing_time = dosing_interval*i
            trough_sampling_times_after_dose = dosing_interval * (i+1) - 1e-6
            
            record_dose = PMRecord(
                    TIME = dosing_time,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if is_infusion else 0,
                    CMT=administrated_compartment_num,
                    MDV=1,)
            df = add_row(record_dose)
            
            for sampling_time_after_dose in sampling_times_after_dosing_time :
                cur_time = dosing_time + sampling_time_after_dose
                if cur_time >= trough_sampling_times_after_dose :
                    break
                else :
                    record_sampling = PMRecord(
                            TIME = cur_time,
                            ID = 1,
                            AMT = 0,
                            RATE = 1 if is_infusion else 0,
                            CMT=observed_compartment_num,
                            MDV=0)
                    
                    df = add_row(record_sampling)
            if include_trough_before_dose and i < repeats - 1 :
                record_trough = PMRecord(
                    ID = 1,
                    AMT = 0,
                    RATE = 1 if is_infusion else 0,
                    TIME=trough_sampling_times_after_dose - 1e-6,
                    DV = target_trough_concentration,
                    CMT=observed_compartment_num,
                    MDV=0)
                df = add_row(record_trough)
                
        if include_last_trough:
            record_trough = PMRecord(
                ID = 1,
                AMT = 0,
                RATE = 1 if is_infusion else 0,
                TIME=dosing_interval*repeats,
                DV = target_trough_concentration,
                CMT=observed_compartment_num,
                MDV=0,)
            df = add_row(record_trough)
        
        super().__init__(df, **kwargs)