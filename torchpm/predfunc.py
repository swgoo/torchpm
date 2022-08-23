from abc import abstractmethod
from collections import ChainMap
from typing import Any, Dict, Set, Tuple

import torch as tc
import torch.nn as nn
from torchdiffeq import odeint

from torchpm import dataset

from .misc import *
from .parameter import *

from .dataset import EssentialColumns
from torch.nn.parameter import Parameter
from torch.nn import ParameterDict


class PredictionFunction(tc.nn.Module):
    _THETA_BOUNDARY_PREFIX = '_theta_boundary_'

    def __init__(self,
            dataset : dataset.PMDataset,
            **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self._column_names = dataset.column_names
        self._ids = dataset.ids
        self._record_lengths : Dict[int, int] = {}
        self._max_record_length = 0
        self.has_boundary = True

        for id in self._ids:
            length = self.dataset.datasets_by_id[id][EssentialColumns.ID.value].size()[0]
            self._record_lengths[id] = length
            self._max_record_length = max(length, self._max_record_length)
    
    def __setattr__(self, name: str, value) -> None:
        with tc.no_grad() :
            if type(value) is Theta :
                super().__setattr__(name, value)
                if value.has_boundary :
                    theta_scaler = ThetaBoundary(*value.init_values)
                    super().__setattr__(self._THETA_BOUNDARY_PREFIX + name, theta_scaler)
                return
            elif type(value) is Eta :
                for id in self._ids : 
                    eta_value = tensor(0.1)
                    value.update({str(int(id)): Parameter(eta_value)})
                super().__setattr__(name, value)
                return
            elif type(value) is Eps :
                for id in self._ids :
                    eps_value = tc.zeros(self._record_lengths[id], requires_grad=True)
                    value[id] = eps_value
                super().__setattr__(name, value)
                return
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        att = super().__getattribute__(name)
        if self.has_boundary and type(att) is Theta:
            theta_boundary = super().__getattribute__(self._THETA_BOUNDARY_PREFIX+name)
            if theta_boundary is not None :
                return theta_boundary(att)
            else :
                return att
        return att

    def reset_epss(self) :
        attributes = dir(self)
        for att_name in attributes :
            att = getattr(self, att_name)
            with tc.no_grad() :
                if type(att) is Eps :
                    for id in self._ids :
                        eps_value = tc.zeros_like(att[str(id)], requires_grad=True, device=att[str(id)].device)
                        att[str(id)] = eps_value

    def _get_amt_indice(self, dataset : Dict[str, Tensor]) :
        amts = dataset[EssentialColumns.AMT.value]
        end = amts.size()[0]
        start_index = tc.squeeze(amts.nonzero(), 1)

        if start_index.size()[0] == 0 :
            return tc.tensor([0], device=amts.device)

        if start_index[0] != 0 :
            start_index = tc.cat([tc.tensor([0], device = amts.device), start_index], 0)

        if start_index[-1] != end - 1 :
            start_index = tc.cat([start_index, tc.tensor([end-1], device = amts.device)] , 0)

        return start_index 
    
    def remove_boundary(self):
        if not self.has_boundary :
            return self
        with tc.no_grad() :
            attribute_names = dir(self)
            for att_name in attribute_names:
                att = getattr(self, att_name)
                if type(att) is Theta :
                    theta_scaler = getattr(self, self._THETA_BOUNDARY_PREFIX+att_name, None)
                    if theta_scaler is None :
                        continue
                    else :
                        setattr(self, att_name, theta_scaler(att))
            self.has_boundary = False
        return self
    
    def set_boundary(self):
        if self.has_boundary : 
            return self
        with tc.no_grad() :
            attribute_names = dir(self)
            for att_name in attribute_names:
                att = getattr(self, att_name)
                if type(att) is Theta :
                    theta_scaler : Optional[ThetaBoundary] = getattr(self, self._THETA_BOUNDARY_PREFIX+att_name, None)
                    if theta_scaler is not None :
                        lb = float(theta_scaler.lb)
                        iv = float(att)
                        ub = float(theta_scaler.ub)
                        theta_init = Theta(lb, iv, ub, fixed=att.fixed, has_boundary = att.has_boundary, requires_grad=att.requires_grad)
                        setattr(self, att_name, theta_init)
                    else : 
                        setattr(self, att_name, att)
            self.has_boundary = True
        return self


    def _pre_forward(self, dataset : Dict[str, Tensor]):
        id = dataset[EssentialColumns.ID.value][0]

        self._calculate_parameters(str(id), dataset)
        record_length = dataset[EssentialColumns.ID.value].size()[0]

        for key, para in dataset.items():
            if para.dim() == 0 or para.size()[0] == 1 :
                dataset[key] = para.repeat([record_length])
        return dataset
    

    def _post_forward(self, dataset, parameters):
        record_length = dataset.size()[0]
        output_columns = {}

        for cov_name in self._output_column_names :
            p = parameters[cov_name]
            if p.dim() == 0 or p.size()[0] == 1 :
                output_columns[cov_name] = p.repeat([record_length])
            else :
                output_columns[cov_name] = parameters[cov_name]
        return {'etas': self.get_etas(), 'epss': self.get_epss(), 'output_columns': output_columns}
    
    @abstractmethod
    def _calculate_parameters(self, id : str, input_columns : Dict[str, tc.Tensor]) -> None:
        pass
    
    @abstractmethod
    def _calculate_error(self, id : str, y_pred: tc.Tensor, parameters: Dict[str, tc.Tensor]) -> tc.Tensor:
        pass
    
    @abstractmethod
    def forward(self, dataset):
        pass

class SymbolicPredictionFunction(PredictionFunction):

    @abstractmethod
    def _calculate_preds(self, id, t : tc.Tensor, parameters : Dict[str, tc.Tensor]) -> tc.Tensor:
        pass

    def forward(self, dataset) :
        parameters = self._pre_forward(dataset)
        f = tc.zeros(dataset.size()[0], device = dataset.device)
        amt_indice = self._get_amt_indice(dataset)

        for i in range(len(amt_indice) - 1):
            start_time_index = amt_indice[i]

            #누적하기 위해 앞부분 생성
            dataset_pre = dataset[:start_time_index, :]
            f_pre = tc.zeros(dataset_pre.size()[0], device = dataset.device)

            times = parameters[EssentialColumns.TIME.value][start_time_index:]
            start_time = times[0]

            amts = parameters[EssentialColumns.AMT.value][start_time_index].repeat(parameters[EssentialColumns.AMT.value][start_time_index:].size()[0])

            parameters_sliced = {k: v[start_time_index:] for k, v in parameters.items()}            
            parameters_sliced[EssentialColumns.TIME.value] = times
            parameters_sliced[EssentialColumns.AMT.value] = amts

            t = times - start_time
            f_cur = self._calculate_preds(t, parameters_sliced)
            f = f + tc.cat([f_pre, f_cur], 0)

        y_pred = self._calculate_error(f, parameters)
        mdv_mask = dataset[:,self._column_names.index(EssentialColumnNames.MDV.value)] == 0
        post_forward_output = self._post_forward(dataset, parameters)
        return ChainMap({'y_pred': y_pred, 'mdv_mask': mdv_mask}, post_forward_output)


class NumericPredictionFunction(PredictionFunction):
    """
    ordinary equation solver
    Args:
        rtol: ratio tolerance about ordinary differential equation integration
        atol: absolute tolerance about ordinary differential equation integration
    """
    rtol : float = 1e-2
    atol : float = 1e-2

    @abstractmethod
    def _calculate_preds(self, id, t : tc.Tensor, y: tc.Tensor , parameters : Dict[str, tc.Tensor]) -> tc.Tensor:
        pass

    def _ode_function(self, t, y):
        index = (self.t < t).sum() - 1
        parameters_sliced = {k: v[index] for k, v in self.parameter_values.items()}
        return self._calculate_preds(t, y, parameters_sliced) + \
            self.infusion_rate * (self.infusion_end_time > t)

    def forward(self, dataset) :
        if not hasattr(self, 'parameter_values'):
            self.parameter_values : Dict[str, tc.Tensor] = {}
        parameters = self._pre_forward(dataset)
        self.parameter_values = parameters

        self.max_cmt = int(dataset[:,self._column_names.index('CMT')].max())
        y_pred_arr = []
        parameters_result = {}
        y_init = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_rate = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_end_time = tc.zeros(self.max_cmt+1, device = dataset.device)
        amt_indice = self._get_amt_indice(dataset)

        for i in range(len(amt_indice) - 1):
            amt_slice = slice(amt_indice[i], amt_indice[i+1]+1)
            amt = parameters[EssentialColumns.AMT.value][amt_indice[i]]
            rate = parameters[EssentialColumns.RATE.value][amt_indice[i]]
            cmt  = parameters[EssentialColumns.CMT.value][amt_slice]
            times = parameters[EssentialColumns.TIME.value][amt_slice]

            if  rate == 0 :                    
                bolus = tc.zeros(self.max_cmt + 1, device = dataset.device)
                bolus[cmt[0].to(tc.int64)] = amt
                y_init = y_init + bolus

            else :
                time = times[0]
                mask = tc.ones(self.max_cmt +1, device = dataset.device)
                mask[cmt[0].to(tc.int64)] = 0
 
                rate_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                rate_vector[cmt[0].to(tc.int64)] = rate

                self.infusion_rate = self.infusion_rate * mask + rate_vector
                infusion_end_time_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                infusion_end_time_vector[cmt[0].to(tc.int64)] = time + amt / rate
                self.infusion_end_time = self.infusion_end_time * mask + infusion_end_time_vector
                
            self.t = times
            result = odeint(self._ode_function, y_init, self.t, rtol=self.rtol, atol=self.atol)
            y_integrated = result
            y_init = result[-1]

            cmt_mask = tc.nn.functional.one_hot(cmt.to(tc.int64)).to(dataset.device)  # type: ignore
            y_integrated = y_integrated.masked_select(cmt_mask==1)  # type: ignore

            # parameters_sliced = {k: v[amt_slice] for k, v in self.parameter_values.items()}
            
            # for k, v in parameters_sliced.items() :
            #     if k not in parameters_result.keys() :
            #         parameters_result[k] = v
            #     else :
            #         parameters_result[k] = tc.cat([parameters_result[k], parameters_sliced[k]])

            if amt_indice[i+1]+1 == dataset.size()[0] :
                y_pred_arr.append(y_integrated)
            else :
                y_pred_arr.append(y_integrated[:-1])

        y_pred = self._calculate_error(tc.cat(y_pred_arr), parameters)

        mdv_mask = dataset[:,self._column_names.index(EssentialColumnNames.MDV.value)] == 0
        post_forward_output = self._post_forward(dataset, parameters)
        return ChainMap({'y_pred': y_pred, 'mdv_mask': mdv_mask}, post_forward_output)
