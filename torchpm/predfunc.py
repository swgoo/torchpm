from abc import abstractmethod
from collections import ChainMap
from typing import Any, Dict, Set, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from torch.nn import Module
from torchdiffeq import odeint

from torchpm import data

from .misc import *
from .para import *

from .data import EssentialColumns
from torch.nn.parameter import Parameter
from torch.nn import ParameterDict, ModuleDict

T = TypeVar('T')

class PredictionFunction(Module):
    PRED_COLUMN_NAME : str = 'PRED'

    def __init__(self,
            dataset : data.PMDataset,
            **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self._column_names = dataset.column_names
        self._ids = dataset.ids
        self._record_lengths = dataset.record_lengths
        self._theta_boundary_mode : bool = True
        self.theta_boundaries = ModuleDict()

    def __setattr__(self, name: str, value : Any) -> None:
        with tc.no_grad() :
            if type(value) is ThetaInit :
                theta_boundary = value.boundary
                self.theta_boundaries.update({name: theta_boundary})
                value = value.theta
            elif type(value) is EtaDict :
                for id in self._ids : 
                    value.update({str(id): Eta(tensor(0.1))})
            elif type(value) is EpsDict :
                for id in self._ids :
                    value.update({str(id): Eps(tc.zeros(self._record_lengths[id], requires_grad=True))})
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        att = super().__getattr__(name)
        if type(att) is Theta \
                and self.theta_boundary_mode \
                and name in self.theta_boundaries.keys():
            theta_boundary = self.theta_boundaries[name]
            return theta_boundary(att)
        return att
    
    @property
    def theta_boundary_mode(self) :
        return self._theta_boundary_mode

    @theta_boundary_mode.setter
    @torch.no_grad()
    def theta_boundary_mode(self, value: bool):
        # turn on
        if not self._theta_boundary_mode and value :
            attribute_names = dir(self)
            for name in attribute_names:
                att = getattr(self, name)
                if type(att) is Theta and name in self.theta_boundaries.keys():
                    theta_boundary = self.theta_boundaries[name]
                    if type(theta_boundary) is ThetaBoundary:
                        lb = float(theta_boundary.lb)
                        iv = float(att)
                        ub = float(theta_boundary.ub)
                        self.theta_boundaries[name] = ThetaBoundary(lb, iv, ub)
                        setattr(self, name, Theta(tensor(0.1), fixed=att.fixed, requires_grad=att.requires_grad))
            self._theta_boundary_mode = True
        # turn off
        elif self._theta_boundary_mode and not value :
            attribute_names = dir(self)
            for name in attribute_names:
                att = getattr(self, name)
                if type(att) is Theta and name in self.theta_boundaries.keys():
                    theta_boundary = self.theta_boundaries[name]
                    theta_value = theta_boundary(att)
                    theta = Theta(tensor(theta_value), fixed=att.fixed, requires_grad=att.requires_grad)
                    setattr(self, name, theta)
            self._theta_boundary_mode = False

    def get_attr_dict(self, dtype: Type[T]) -> Dict[str, T]:
        attr_dict : Dict[str, dtype] = {}
        att_names = dir(self)
        for name in att_names :
            att = getattr(self, name) 
            if type(att) is dtype :
                attr_dict[name] = att
        return attr_dict

    def reset_epss(self) :
        attributes = dir(self)
        for att_name in attributes :
            att = getattr(self, att_name)
            with tc.no_grad() :
                if type(att) is EpsDict :
                    for id in self._ids :
                        eps_value = tc.zeros_like(att[str(id)], requires_grad=True, device=att[str(id)].device)
                        att.update({str(id): Parameter(eps_value)})

    def _get_amt_indice(self, dataset : Dict[str, Tensor]) :
        amts = dataset[EssentialColumns.AMT.value]
        end = amts.size()[0]
        start_index = tc.squeeze(amts.nonzero(), 1)

        if start_index.size()[0] == 0 :
            return tc.tensor([0], device=amts.device)

        if start_index[0] != 0 :
            start_index = tc.cat([tensor([0], device = amts.device), start_index], 0)

        if start_index[-1] != end - 1 :
            start_index = tc.cat([start_index, tensor([end-1], device = amts.device)] , 0)

        return start_index 

    def _reshape_dataset(self, dataset : Dict[str, Tensor]) -> Dict[str, Tensor]:
        id_tensor = dataset[EssentialColumns.ID.value]
        id = int(id_tensor[0])

        record_length = self._record_lengths[id]
    
        for key, para in dataset.items():
            dataset[key] = para.to(device=id_tensor.device)
            if para.dim() == 0 or para.size()[0] == 1 :
                dataset[key] = para.repeat([record_length])
        return dataset
    
    def _batch_to_datasets(self, batch) :
        datasets : List[Dict[str, Tensor]] = []
        with torch.no_grad() :
            for id in batch[EssentialColumns.ID.value] :
                datasets.append({})
            for col_name, values in batch.items() :
                for dataset, value in zip(datasets, values) :
                    dataset[col_name] = value
            for dataset in datasets :
                id = dataset[EssentialColumns.ID.value][0]
                length = self._record_lengths[int(id)]
                for col_name, value in dataset.items() :
                    dataset[col_name] = value[:length]   
        return datasets
    
    @abstractmethod
    def parameter_fuction(self, input_columns : Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass
    
    @abstractmethod
    def error_function(self, y_pred: tc.Tensor, parameters: Dict[str, tc.Tensor]) -> tc.Tensor:
        pass
    
    @abstractmethod
    def forward(self, dataset):
        pass

class SymbolicPredictionFunction(PredictionFunction):

    @abstractmethod
    def pred_function(self, t : tc.Tensor, dataset : Dict[str, tc.Tensor]) -> tc.Tensor:
        pass

    def forward(self, batch) :

        datasets = self._batch_to_datasets(batch)
        
        output = []
        for dataset in datasets :
            amt_indice = self._get_amt_indice(dataset)
            dataset = self.parameter_fuction(dataset)
            parameters = self._reshape_dataset(dataset)
            f = tc.zeros(dataset[EssentialColumns.ID.value].size()[0], device = dataset[EssentialColumns.ID.value].device)
            for i in range(len(amt_indice) - 1):
                start_time_index = amt_indice[i]

                dataset_pre = {key: value[:start_time_index] for key, value in dataset.items()}
                f_pre = tc.zeros(dataset_pre[EssentialColumns.ID.value].size()[0], device = dataset[EssentialColumns.ID.value].device)

                times = parameters[EssentialColumns.TIME.value][start_time_index:]
                start_time = times[0]

                amts = parameters[EssentialColumns.AMT.value][start_time_index].repeat(parameters[EssentialColumns.AMT.value][start_time_index:].size()[0])

                parameters_sliced = {k: v[start_time_index:] for k, v in parameters.items()}            
                parameters_sliced[EssentialColumns.TIME.value] = times
                parameters_sliced[EssentialColumns.AMT.value] = amts

                t = times - start_time
                f_cur = self.pred_function(t, parameters_sliced)
                f = f + tc.cat([f_pre, f_cur], 0)

            pred = self.error_function(f, parameters)
            post_forward_output = self._reshape_dataset(parameters)
            post_forward_output[self.PRED_COLUMN_NAME] = pred
            output.append(post_forward_output)

        return output


class NumericPredictionFunction(PredictionFunction):
    """
    ordinary equation solver
    Args:
        rtol: ratio tolerance about ordinary differential equation integration
        atol: absolute tolerance about ordinary differential equation integration
    """
    def __init__(
            self, 
            dataset: data.PMDataset, 
            rtol : float = 1e-2,
            atol : float = 1e-2,
            **kwargs):
        
        super().__init__(dataset, **kwargs)
        self.rtol = rtol
        self.atol = atol
    

    @abstractmethod
    def pred_function(self, t : Tensor, y: Tensor , parameters : Dict[str, Tensor]) -> Tensor:
        pass
    
    def _generate_ode_function(
            self, 
            times : Tensor, 
            parameters : Dict[str, Tensor], 
            infusion_rate : Tensor, 
            infusion_end_time : Tensor):
        if times.dim() != 1 :
            raise Exception("times' dim should be 1")
        def ode_function(t : Tensor, y : Tensor):
            index = (times < t).sum() - 1
            parameters_sliced = {k: v[index] for k, v in parameters.items()}
            return self.pred_function(t, y, parameters_sliced) + infusion_rate * (infusion_end_time > t)
        return ode_function
        

    def forward(self, batch : Dict[str, Tensor]) :

        datasets = self._batch_to_datasets(batch)
        output = []
        for dataset in datasets :
            max_cmt = int(dataset[EssentialColumns.CMT.value].max())

            infusion_rate = tc.zeros(max_cmt+1, device = dataset[EssentialColumns.ID.value].device)
            infusion_end_time = tc.zeros(max_cmt+1, device = dataset[EssentialColumns.ID.value].device)
            
            amt_indice = self._get_amt_indice(dataset)
            dataset = self.parameter_fuction(dataset)
            parameters = self._reshape_dataset(dataset)

            y_pred_arr = []
            y_init = tc.zeros(max_cmt+1, device = dataset[EssentialColumns.ID.value].device)

            for i in range(len(amt_indice) - 1):
                amt_slice = slice(amt_indice[i], amt_indice[i+1]+1)
                amt = parameters[EssentialColumns.AMT.value][amt_indice[i]]
                rate = parameters[EssentialColumns.RATE.value][amt_indice[i]]
                cmt  = parameters[EssentialColumns.CMT.value][amt_slice]
                times = parameters[EssentialColumns.TIME.value][amt_slice]

                if  rate == 0 :                    
                    bolus = tc.zeros(max_cmt + 1, device = dataset[EssentialColumns.ID.value].device)
                    bolus[cmt[0].to(tc.int64)] = amt
                    y_init = y_init + bolus

                else :
                    time = times[0]
                    mask = tc.ones(max_cmt +1, device = dataset[EssentialColumns.ID.value].device)
                    mask[cmt[0].to(tc.int64)] = 0
    
                    rate_vector = tc.zeros(max_cmt +1, device = dataset[EssentialColumns.ID.value].device)
                    rate_vector[cmt[0].to(tc.int64)] = rate

                    infusion_rate = infusion_rate * mask + rate_vector
                    infusion_end_time_vector = tc.zeros(max_cmt +1, device = dataset[EssentialColumns.ID.value].device)
                    infusion_end_time_vector[cmt[0].to(tc.int64)] = time + amt / rate
                    infusion_end_time = infusion_end_time * mask + infusion_end_time_vector
                    
                result = odeint(self._generate_ode_function(times, parameters, infusion_rate, infusion_end_time), y_init, times, rtol=self.rtol, atol=self.atol)
                y_integrated = result
                y_init = result[-1]

                cmt_mask = tc.nn.functional.one_hot(cmt.to(tc.int64)).to(dataset[EssentialColumns.ID.value].device)  # type: ignore
                y_integrated = y_integrated.masked_select(cmt_mask==1)  # type: ignore

                if amt_indice[i+1]+1 == dataset[EssentialColumns.ID.value].size()[0] :
                    y_pred_arr.append(y_integrated)
                else :
                    y_pred_arr.append(y_integrated[:-1])

            pred = self.error_function(tc.cat(y_pred_arr), parameters)

            post_forward_output = self._reshape_dataset(parameters)
            post_forward_output[self.PRED_COLUMN_NAME] = pred
            output.append(post_forward_output)
        return output
