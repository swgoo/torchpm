from typing import Dict, Iterable, Set
import torch as tc
import torch.nn as nn
from torchdiffeq import odeint

from collections import ChainMap

from torchpm import data

from .estimated_parameter import *
from .misc import *

class PredictionFunctionModule(tc.nn.Module):
    ESSENTIAL_COLUMNS : Iterable[str] = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT']

    def __init__(self,
                dataset : tc.utils.data.Dataset,
                column_names : Iterable[str],
                output_column_names: Iterable[str],
                *args, **kwargs):
        super(PredictionFunctionModule, self).__init__(*args, **kwargs)
        self.dataset : tc.utils.data.Dataset = dataset
        self._column_names : Iterable[str] = column_names
        self._output_column_names = output_column_names

        self._ids = set()
        self._record_lengths : Dict[str, int] = {}
        self._max_record_length = 0
        for data in self.dataset :
            id = data[0][:, self._column_names.index('ID')][0]
            self._ids.add(int(id))
            self._record_lengths[str(int(id))] = data[0].size()[0]
            self._max_record_length = max(data[0].size()[0], self._max_record_length)
    
    def initialize(self):
        self._theta_names : Set[str] = set()
        self._eta_names : Set[str] = set()
        self._eps_names : Set[str] = set()
        
        attributes = dir(self)
        for att_name in attributes:
            att = getattr(self, att_name)
            att_type = type(att)
            with tc.no_grad() :
                if att_type is Theta :
                    self._theta_names.add(att_name.replace('theta_', ''))
                elif att_type is Eta :
                    self._eta_names.add(att_name.replace('eta_', ''))
                    for id in self._ids :
                        eta_value = tc.tensor(0.1, device=self.dataset.device) 
                        att.parameter_values.update({str(int(id)): tc.nn.Parameter(eta_value)})
                elif att_type is Eps :
                    self._eps_names.add(att_name.replace('eps_', ''))
                    for id in self._ids :
                        eps_value = tc.zeros(self._record_lengths[str(int(id))], requires_grad=True, device=self.dataset.device)
                        att.parameter_values[str(int(id))] = eps_value

    def _get_estimated_parameters(self, prefix, names) :
        dictionary : Dict[str, tc.Tensor] = {}
        for name in names :
            att = getattr(self, prefix + name)
            dictionary[name] = att()
        return dictionary

    def get_thetas(self) :
        return self._get_estimated_parameters('theta_', self._theta_names)
    
    def get_etas(self) :
        return self._get_estimated_parameters('eta_', self._eta_names)
    
    def get_epss(self) :
        return self._get_estimated_parameters('eps_', self._eps_names)
    
    def _get_estimated_parameter_values(self, prefix, names) :
        dictionary : Dict[str, tc.Tensor] = {}
        for name in names :
            att = getattr(self, prefix + name)
            parameter_att_list = dir(att)
            if 'parameter_value' in parameter_att_list:
                dictionary[name] = att.parameter_value
            elif 'parameter_values' in parameter_att_list:
                dictionary[name] = att.parameter_values
        return dictionary
    
    def get_theta_parameter_values(self) :
        return self._get_estimated_parameter_values('theta_', self._theta_names)
    
    def get_eta_parameter_values(self) :
        return self._get_estimated_parameter_values('eta_', self._eta_names)
    
    def get_eps_parameter_values(self) :
        return self._get_estimated_parameter_values('eps_', self._eps_names)

    def reset_epss(self) :
        attributes = dir(self)
        for att_name in attributes:
            att = getattr(self, att_name)
            att_type = type(att)
            with tc.no_grad() :
                if att_type is Eps :
                    self._eps_names.add(att_name.replace('eps_', ''))
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

    def _pre_forward(self, dataset):
        id = str(int(dataset[:,self._column_names.index('ID')][0]))
        self._id = id
        
        for name in self._eta_names:
            att = getattr(self, 'eta_' + name)
            att.id = id

        for name in self._eps_names:
            att = getattr(self, 'eps_' + name)
            att.id = id
        
        covariates = self._get_covariates(dataset)
        parameters = self._calculate_parameters(**covariates)
        record_length = dataset.size()[0]
        for key, para in parameters.items():
            if para.dim() == 0 or para.size()[0] == 1 :
                parameters[key] = para.repeat([record_length])
        return parameters
    
    def _post_forward(self, parameters):
        output_columns = {}
        for cov_name in self._output_column_names :
            output_columns[cov_name] = parameters[cov_name]
        return {'etas': self.get_etas(), 'epss': self.get_epss(), 'output_columns': output_columns}
    
    def _calculate_parameters(self, **covariates) -> Dict[str, tc.Tensor]:
        pass

    def _calculate_error(self, y_pred, **parameters) -> tc.Tensor:
        pass
    
    def forward(self, dataset):
        pass

    def _get_covariates(self, dataset) :
        dataset = dataset.t()
        
        covariates = {}
        for i, name in enumerate(self._column_names) :
            covariates[name] = dataset[i]
        
        return covariates


class PredictionFunctionByTime(PredictionFunctionModule):
    def __init__(self, dataset: tc.utils.data.Dataset, column_names: Iterable[str], output_column_names: Iterable[str], *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)
    
    def _calculate_preds(self, t, **parameters) -> tc.Tensor:
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
            parameters_sliced = {k: v[start_time_index:] for k, v in parameters.items()}
            
            t = times - start_time
            f_cur = self._calculate_preds(t, **parameters_sliced)
            f = f + tc.cat([f_pre, f_cur], 0)
        
        y_pred, parameters = self._calculate_error(f, **parameters)
        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0
        
        post_forward_output = self._post_forward(parameters)
        
        return ChainMap({'y_pred': y_pred, 'mdv_mask': mdv_mask}, post_forward_output)

class PredictionFunctionByODE(PredictionFunctionModule):
    """
    ordinary equation solver
    Args:
        rtol: ratio tolerance about ordinary differential equation integration
        atol: absolute tolerance about ordinary differential equation integration
    """
    rtol : float = 1e-2
    atol : float = 1e-2
 
    def ode_function(self, t, y):
        index = (self.t < t).sum() -1
        parameters_sliced = {k: v[index] for k, v in self.parameters.items()}
    
        return self._calculate_preds(t, y, **parameters_sliced) \
            + self.infusion_rate * (self.infusion_end_time > t)
    def forward(self, dataset) :
        parameters =self._pre_forward(dataset)
        self.parameters = parameters

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
                rate_vector[cmt[0]] = rate
                self.infusion_rate = self.infusion_rate * mask + rate_vector
 
                infusion_end_time_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                infusion_end_time_vector[cmt[0]] = time + amt / rate
                self.infusion_end_time = self.infusion_end_time * mask + infusion_end_time_vector
                
            self.t = times
            result = odeint(self.ode_function, y_init, self.t, rtol=self.rtol, atol=self.atol)
            y_integrated = result
            y_init = result[-1]
            
            cmt_mask = tc.nn.functional.one_hot(cmt.to(tc.int64)).to(dataset.device)
            y_integrated = y_integrated.masked_select(cmt_mask==1)

            parameters_sliced = {k: v[amt_slice] for k, v in self.parameters.items()}
            y_pred, parameters_sliced = self._calculate_error(y_integrated, **parameters_sliced)
            
            for k, v in parameters_sliced.items() :
                if k not in parameters_result.keys() :
                    parameters_result[k] = v
                else :
                    parameters_result[k] = tc.cat([parameters_result[k], parameters_sliced[k]])
            y_pred_arr.append(y_pred)

        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0

        post_forward_output = self._post_forward(parameters_result)
        
        return ChainMap({'y_pred': tc.cat(y_pred_arr), 'mdv_mask': mdv_mask}, post_forward_output)
