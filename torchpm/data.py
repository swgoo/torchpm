from typing import Iterable

import torch as tc
import numpy as np

class CSVDataset(tc.utils.data.Dataset):
    """
    Args:
        file_path: csv file path
        column_names: csv file's column names
        device: (optional) data loaded location 
    """
    
    def __init__(self, 
                 file_path : str,
                 column_names : Iterable[str],
                 device : tc.DeviceObjType = tc.device("cpu")):
        self.file_path = file_path
        self.column_names = column_names
        self.device = device
        
        dataset_total = np.loadtxt(self.file_path, delimiter=',', dtype=np.float32, skiprows=1)
        y_true_total = dataset_total[:,self.column_names.index('DV')]
        
        ids, ids_start_idx = np.unique(dataset_total[:, column_names.index('ID')], return_index=True)
        ids_start_idx = ids_start_idx[1:]
        dataset_np = np.split(dataset_total, ids_start_idx)
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
    def __init__(self, data : tc.utils.data.Dataset, sizes : Iterable[int] = None, devices : Iterable[tc.DeviceObjType] = None):

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