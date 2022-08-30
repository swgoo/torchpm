import enum
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Literal, Optional, OrderedDict, Set, Tuple, Type

import torch
from scipy import stats
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
        super().__init__(**kwargs)
        
        EssentialColumns.check_inclusion_of_names(dataframe.columns)

        self.column_names = list(dataframe.columns)
        
        for col in dataframe.columns :
            if col in EssentialColumns.int_column_names() :
                dataframe[col] = dataframe[col].astype(int)
            else :
                dataframe[col] = dataframe[col].astype(float)
        
        self.ids : List[int] = dataframe[EssentialColumns.ID.value].sort_values(axis = 0).unique().tolist()
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

@dataclass
class PMRecord:

    def __init__(self,
            ID : int = 1,
            TIME : float = 0,
            AMT : float = 0,
            RATE : float = 0,
            DV : float = 0,
            MDV : int = 0,
            CMT : int = 0,
            **covariates : float) -> None:

        self.ID = ID
        self.TIME = TIME
        self.AMT = AMT
        self.RATE = RATE
        self.DV = DV
        self.MDV = MDV
        self.CMT = CMT
        for name, value in covariates.items() :
            setattr(self, name, value)

class OptimalDesignDataset(PMDataset):
    from .ode import EquationConfig

    def __init__(self,  
                equation_config : EquationConfig,
                column_names : List[str],
                dosing_interval : float,
                target_trough_concentration : float = 0.,
                sampling_times_after_dosing_time : List[float] = [],
                include_trough_before_dose : bool = False,
                include_last_trough : bool = False,
                repeats : int = 10,
                **kwargs):

        covariate_name_list = set(column_names) - EssentialColumns.get_name_set()
        covariate_name_list = list(covariate_name_list)
        df_columns = EssentialColumns.get_name_list()+covariate_name_list

        covariates = OrderedDict()
        for name in covariate_name_list :
            covariates[name] = 0.
        
        df = pd.DataFrame(columns=df_columns)
        
        for i in range(repeats) :
            dosing_time = dosing_interval*i
            trough_sampling_times_after_dose = dosing_interval * (i+1) - 1e-6
            
            record_dose = PMRecord(
                    TIME = dosing_time,
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    CMT=equation_config.administrated_compartment_num,
                    MDV=1,
                    **covariates)
            df = df.append(asdict(record_dose), ignore_index=True)
            
            for sampling_time_after_dose in sampling_times_after_dosing_time :
                cur_time = dosing_time + sampling_time_after_dose
                if cur_time >= trough_sampling_times_after_dose :
                    break
                else :
                    record_sampling = PMRecord(
                            TIME = cur_time,
                            ID = 1,
                            AMT = 0,
                            RATE = 1 if equation_config.is_infusion else 0,
                            CMT=equation_config.observed_compartment_num,
                            MDV=0,
                            **covariates)
                    
                    df = df.append(asdict(record_sampling), ignore_index=True)
            if include_trough_before_dose and i < repeats - 1 :
                record_trough = PMRecord(
                        ID = 1,
                        AMT = 0,
                        RATE = 1 if equation_config.is_infusion else 0,
                        TIME=trough_sampling_times_after_dose - 1e-6,
                        DV = target_trough_concentration,
                        CMT=equation_config.observed_compartment_num,
                        MDV=0,
                        **covariates)
                df = df.append(asdict(record_trough), ignore_index=True)
                
        if include_last_trough:
            record_trough = PMRecord(
                    ID = 1,
                    AMT = 1,
                    RATE = 1 if equation_config.is_infusion else 0,
                    TIME=dosing_interval*repeats,
                    DV = target_trough_concentration,
                    CMT=equation_config.observed_compartment_num,
                    MDV=0,
                    **covariates)
            df = df.append(asdict(record_trough), ignore_index=True)
        
        super().__init__(df, **kwargs)
 