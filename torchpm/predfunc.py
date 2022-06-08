from abc import abstractmethod
from collections import ChainMap
from typing import Any, Dict, Set

import torch as tc
import torch.nn as nn
from torchdiffeq import odeint

from torchpm import data

from .misc import *
from .parameter import *


class PredictionFunction(tc.nn.Module):

    def __init__(self,
            dataset : data.CSVDataset,
            output_column_names: List[str],
            **kwargs):
        super().__init__(**kwargs)
        self.theta_names = set()
        self.eta_names = set()
        self.eps_names = set()
        self.dataset = dataset
        self._column_names = dataset.column_names
        self._output_column_names = output_column_names
        self._ids = set()
        self._record_lengths : Dict[str, int] = {}
        self._max_record_length = 0

        for data in tc.utils.data.DataLoader(self.dataset, batch_size=None, shuffle=False, num_workers=0):  # type: ignore
            id = data[0][:, self._column_names.index('ID')][0]
            self._ids.add(int(id))
            self._record_lengths[str(int(id))] = data[0].size()[0]
            self._max_record_length = max(data[0].size()[0], self._max_record_length)
    
    def __setattr__(self, name: str, value) -> None:
        att_type = type(value)

        with tc.no_grad() :
            if att_type is Theta :
                self.theta_names.add(name)

            elif att_type is Eta :
                self.eta_names.add(name)

                for id in self._ids : # type: ignore
                    eta_value = tc.tensor(0.1, device=self.dataset.device) # type: ignore
                    value.parameter_values.update({str(int(id)): tc.nn.Parameter(eta_value)}) # type: ignore

            elif att_type is Eps :
                self.eps_names.add(name)

                for id in self._ids :
                    eps_value = tc.zeros(self._record_lengths[str(int(id))], requires_grad=True, device=self.dataset.device)
                    value.parameter_values[str(int(id))] = eps_value # type: ignore
        
        super().__setattr__(name, value)
        

    def _get_estimated_parameters(self, names) :
        dictionary : Dict[str, Any] = {}
        for name in names :
            att = getattr(self, name)
            dictionary[name] = att
        return dictionary

    def get_thetas(self) -> Dict[str, Theta]:
        return self._get_estimated_parameters(self.theta_names)

    def get_etas(self) -> Dict[str, Eta]:
        return self._get_estimated_parameters(self.eta_names)

    def get_epss(self) -> Dict[str, Eps]:
        return self._get_estimated_parameters(self.eps_names)
    
    def _get_estimated_parameter_values(self, names) -> Dict[str, Any]:
        dictionary : Dict[str, Any] = {}
        for name in names :
            att = getattr(self, name)
            parameter_att_list = dir(att)
            if isinstance(att, Theta) and 'parameter_value' in parameter_att_list:  # type: ignore
                dictionary[name] = att.parameter_value
            elif isinstance(att, (Eta, Eps, Omega, Sigma)) and 'parameter_values' in parameter_att_list:  # type: ignore
                dictionary[name] = att.parameter_values
        return dictionary
        
    def _get_estimated_values(self, names) -> Dict[str, Any]:
        dictionary : Dict[str, Any] = {}
        for name in names :
            att = getattr(self, name)
            parameter_att_list = dir(att)
            if isinstance(att, Theta) and 'parameter_value' in parameter_att_list:  # type: ignore
                dictionary[name] = att
            elif isinstance(att, (Eta, Eps, Omega, Sigma)) and 'parameter_values' in parameter_att_list:  # type: ignore
                dictionary[name] = att
        return dictionary

    def get_theta_values(self) -> Dict[str, Theta]:
        return self._get_estimated_values(self.theta_names)

    def get_theta_parameter_values(self) :
        return self._get_estimated_parameter_values(self.theta_names)
    
    def get_eta_parameter_values(self) -> Dict[str, nn.ParameterDict]:  # type: ignore
        return self._get_estimated_parameter_values(self.eta_names)
    
    def get_eps_parameter_values(self) -> Dict[str, Dict[str, nn.Parameter]]:  # type: ignore
        return self._get_estimated_parameter_values(self.eps_names)

    def reset_epss(self) :
        attributes = dir(self)
        for att_name in attributes :
            att = getattr(self, att_name)
            att_type = type(att)
            with tc.no_grad() :
                if att_type is Eps :
                    for id in self._ids :
                        eps_value = tc.zeros(self._record_lengths[str(int(id))], requires_grad=True, device=self.dataset.device)
                        att.parameter_values[str(int(id))] = eps_value

    def _get_amt_indice(self, dataset) :
        amts = dataset[:, self._column_names.index('AMT')]
        end = amts.size()[0]
        start_index = tc.squeeze(amts.nonzero(), 0)

        if start_index.size()[0] == 0 :
            return tc.tensor([0], device = dataset.device)

        if start_index[0] != 0 :
            start_index = tc.cat([tc.tensor([0], device = dataset.device), start_index], 0)

        if start_index[-1] != end - 1 :
            start_index = tc.cat([start_index, tc.tensor([end-1], device = dataset.device)] , 0)

        return start_index 
    
    def descale(self):
        with tc.no_grad() :
            attribute_names = dir(self)
            for att_name in attribute_names:
                att = getattr(self, att_name)
                if type(att) is Theta :
                    att.descale()
        return self
    
    def scale(self):
        with tc.no_grad() :
            attribute_names = dir(self)
            for att_name in attribute_names:
                att = getattr(self, att_name)
                if type(att) is Theta :
                    att.scale()
        return self


    def _pre_forward(self, dataset):
        id = str(int(dataset[:,self._column_names.index('ID')][0]))
        self._id = id

        for name in self.eta_names:
            att = getattr(self, name)
            att.id = id

        for name in self.eps_names:
            att = getattr(self, name)
            att.id = id

        input_columns = self._get_input_columns(dataset)
        self._calculate_parameters(input_columns)
        parameters = input_columns
        record_length = dataset.size()[0]

        for key, para in parameters.items():
            if para.dim() == 0 or para.size()[0] == 1 :
                parameters[key] = para.repeat([record_length])

        return parameters
    

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
    def _calculate_parameters(self, input_columns : Dict[str, tc.Tensor]) -> None:
        pass
    
    @abstractmethod
    def _calculate_error(self, y_pred: tc.Tensor, parameters: Dict[str, tc.Tensor]) -> tc.Tensor:
        pass
    
    @abstractmethod
    def forward(self, dataset):
        pass

    def _get_input_columns(self, dataset) :
        dataset = dataset.t()
        input_columns = {}
        for i, name in enumerate(self._column_names) :
            input_columns[name] = dataset[i]
        return input_columns

class SymbolicPredictionFunction(PredictionFunction):

    @abstractmethod
    def _calculate_preds(self, t : tc.Tensor, parameters : Dict[str, tc.Tensor]) -> tc.Tensor:
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

            times = parameters['TIME'][start_time_index:]
            start_time = times[0]

            amts = parameters['AMT'][start_time_index].repeat(parameters['AMT'][start_time_index:].size()[0])

            parameters_sliced = {k: v[start_time_index:] for k, v in parameters.items()}            
            parameters_sliced['TIME'] = times
            parameters_sliced['AMT'] = amts

            t = times - start_time
            f_cur = self._calculate_preds(t, parameters_sliced)
            f = f + tc.cat([f_pre, f_cur], 0)

        y_pred = self._calculate_error(f, parameters)
        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0
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
    def _calculate_preds(self, t : tc.Tensor, y: tc.Tensor , parameters : Dict[str, tc.Tensor]) -> tc.Tensor:
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
            amt = parameters['AMT'][amt_indice[i]]
            rate = parameters['RATE'][amt_indice[i]]
            cmt  = parameters['CMT'][amt_slice]
            times = parameters['TIME'][amt_slice]

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

            parameters_sliced = {k: v[amt_slice] for k, v in self.parameter_values.items()}
            y_pred = self._calculate_error(y_integrated, parameters_sliced)

            for k, v in parameters_sliced.items() :
                if k not in parameters_result.keys() :
                    parameters_result[k] = v
                else :
                    parameters_result[k] = tc.cat([parameters_result[k], parameters_sliced[k]])
            y_pred_arr.append(y_pred)

        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0
        post_forward_output = self._post_forward(dataset, parameters_result)
        return ChainMap({'y_pred': tc.cat(y_pred_arr), 'mdv_mask': mdv_mask}, post_forward_output)
