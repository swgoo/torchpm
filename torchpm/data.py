from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from queue import Queue
import re
from typing import Dict, List, Literal, Mapping, Sequence, Tuple
from lightning import LightningDataModule
import numpy as np
import pandas as pd
from torch import Tensor, tensor, zeros
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from re import Pattern

#TODO automatic TAD
def read_from_nonmem_csv(csv_path : Path, na_values: str = '.', as_same_time_limit: float = 0.11,) -> pd.DataFrame :
    ESSENTIAL_COLUMN_PATTERN : Pattern = r"^ID$|^TIME$|^DV$|^AMT$|^CMT$|^MDV$"
    
    data_frame = pd.read_csv(csv_path, na_values=na_values)
    data_frame.columns = data_frame.columns.str.upper()
    assert as_same_time_limit > 0
    essential_col_num = f"{ESSENTIAL_COLUMN_PATTERN}".count('|') + 1
    assert essential_col_num == data_frame.filter(regex=ESSENTIAL_COLUMN_PATTERN,axis=1).columns.__len__()
    max_cmt = max(data_frame['CMT'])
    
    mdv_mask = data_frame['MDV'] == 1
    data_frame.loc[mdv_mask, 'DV'] = pd.NA

    dv_scalar_tensor = torch.tensor(data_frame['DV']).unsqueeze(-1)
    dv_one_hot = torch.nn.functional.one_hot(torch.tensor((data_frame['CMT']-1).to_numpy()), num_classes = max_cmt).to(torch.float)
    dv_one_hot[dv_one_hot == 0] = torch.nan
    dv=(dv_scalar_tensor*dv_one_hot).t().numpy()
    dv_col_names = [f"DV{i}" for i in range(max_cmt)]
    data_frame = data_frame.drop(['DV','MDV','CMT'], axis=1)
    data_frame = data_frame.assign(**{k: v for k,v in zip(dv_col_names, dv)})

    def _individual_pm_record_to_pm_train_dataset(data:pd.DataFrame):
        data = data.reset_index(drop=True)
        for i, r in data[::-1].iterrows():
            if i == 0 : break
            pre_r = data.loc[i-1]
            if abs(r['TIME'] - pre_r['TIME']) < as_same_time_limit :
                cur_dv = r.filter(regex  = r"^DV\d+$")
                for row_index, v in cur_dv.items():
                    if not np.isnan(cur_dv[row_index]) :
                        data.loc[i-1, row_index] = v
                if np.isnan(data['AMT'][i-1]) :
                    data.loc[i-1, 'AMT'] = data['AMT'][i]
                data = data.drop(i, axis=0)
        return data

    data_frame_output = None
    for id,df in data_frame.groupby("ID", sort=True,) :

        dfo = _individual_pm_record_to_pm_train_dataset(df)
        if data_frame_output is None :
            data_frame_output = dfo
        else :
            data_frame_output = pd.concat([data_frame_output, dfo], axis=0, ignore_index=True)
    return data_frame_output

@dataclass
class MixedEffectsTimeData:
    id: Tensor
    dv: Tensor
    iv: Tensor
    time: Tensor
    init: Tensor

@dataclass
class MixedEffectsTimeDatasetConfig:
    dv_column_names: Tuple[str,...] = ('DV',)
    iv_column_names: Tuple[str,...] | None = None
    init_column_names: Tuple[str,...] = ('AMT',)
    id_column_name: str = 'ID'
    time_column_name: str = 'TIME'

    def __post_init__(self):
        assert len(self.dv_column_names) == len(self.init_column_names)

class MixedEffectsTimeDataset(Dataset):
    def __init__(
            self,
            data_frame: pd.DataFrame,
            config: MixedEffectsTimeDatasetConfig):  
        
        for f in fields(MixedEffectsTimeData):
            setattr(self, f.name, [])

        df_by_id = data_frame.groupby(config.id_column_name, sort=True)
        for _, d in df_by_id:
            d = d.reset_index(drop=True)

            id = tensor(d[config.id_column_name][0], dtype = torch.int64)
            self.id.append(id)

            time = tensor(d[config.time_column_name].to_numpy(), dtype=torch.float32)
            self.time.append(time)

            dv = d.filter(config.dv_column_names, axis=1).to_numpy()
            dv = tensor(dv, dtype=torch.float32)
            self.dv.append(dv)

            iv = d.filter(config.iv_column_names, axis=1).to_numpy()
            iv = tensor(iv, dtype=torch.float32).nan_to_num()
            self.iv.append(iv)

            init = d.filter(config.init_column_names, axis=1).to_numpy()
            init = tensor(init, dtype=torch.float32).nan_to_num()
            self.init.append(init)

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index) -> Mapping[str,Tensor]:
        return {f.name:getattr(self, f.name)[index] for f in fields(MixedEffectsTimeData)}

class MixedEffectsTimeDataCollator:
    @torch.no_grad()
    def __call__(self, samples: List[Dict[str, Tensor]]):
        batch : Dict[str, List[Tensor]] = {}
        for s in samples :
            for k, value in s.items():
                vs = batch.get(k, [])
                vs.append(value)
                batch.update({k: vs})
        output : Dict[str, Tensor] = {}
        for key, value in batch.items():
            match key:
                case "id" :
                    output[key] = torch.stack(value)
                case "time" :
                    time = torch.nn.utils.rnn.pad_sequence(value, True, padding_value=float('nan'))
                    max_time = time.max(dim=-1).values
                    for i, d_t in enumerate(time) :
                        start = d_t.isnan().logical_not().sum()
                        max_time = d_t.nan_to_num(0).max()
                        for j,k in enumerate(range(start,len(d_t))):
                            d_t[k] = max_time.clone().detach() + j*0.1 + 0.1
                        time[i] = d_t
                        output[key] = time # batch, time
                case _ :
                    output[key] = torch.nn.utils.rnn.pad_sequence(value, True, padding_value=0.)

        return output

class MixedEffectsTimeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: MixedEffectsTimeDatasetConfig,
        train_data: Path | pd.DataFrame | str,
        valid_data: Path | pd.DataFrame | str | None = None,
        pred_data: Path | pd.DataFrame | str | None = None,
        test_csv_path: Path | pd.DataFrame | str | None = None,
        batch_size: int = 100,
        num_workers: int = 0,
        na_values: str = '.',
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.num_workers = num_workers
        self.na_values = na_values
        self.train_df = self._load_dataset(train_data)
        self.valid_df = self._load_dataset(valid_data if valid_data else train_data)
        self.pred_df = self._load_dataset(pred_data if pred_data else train_data)
        self.test_df = self._load_dataset(test_csv_path if test_csv_path else train_data)
        self._collater = MixedEffectsTimeDataCollator()
        self.batch_size = batch_size
    
    def _load_dataset(self, data: Path | pd.DataFrame):
        if isinstance(data, (Path, str)) :
            return pd.read_csv(data, na_values=self.na_values, skip_blank_lines=True)
        elif isinstance(data, pd.DataFrame) :
            return data
        else :
            raise RuntimeError(f'Data Type must be {Path.__name__} or {pd.DataFrame.__name__}')

    def setup(self, stage: str) -> None:        
        match stage:
            case 'fit' | 'validate':
                self.train_dataset = MixedEffectsTimeDataset(self.train_df, self.dataset_config)
                self.valid_dataset = MixedEffectsTimeDataset(self.valid_df, self.dataset_config)
            case 'predict':
                self.pred_dataset = MixedEffectsTimeDataset(self.pred_df, self.dataset_config)
            case 'test':
                self.test_dataset = MixedEffectsTimeDataset(self.test_df, self.dataset_config)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collater)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collater)
    
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collater)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collater)


