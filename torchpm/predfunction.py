from re import L
from typing import Dict, Iterable
import torch as tc
import torch.nn as nn
from torchdiffeq import odeint
from zmq import device

from . import estimated_parameter
from .misc import *

class PredictionFunctionModule(tc.nn.Module):

    def __init__(self,
                dataset : tc.utils.data.Dataset,
                column_names : Iterable[str],
                output_column_names: Iterable[str],
                *args, **kwargs):
        super(PredictionFunctionModule, self).__init__(*args, **kwargs)
        self._dataset : tc.utils.data.Dataset = dataset
        self._column_names : Iterable[str] = column_names
        self._output_column_names = output_column_names

        self._ids = set()
        self._record_lengths : Dict[str, int] = {}
        self._max_record_length = 0
        for data in self._dataset :
            id = data[0][:, self._column_names.index('ID')][0]
            self._ids.add(int(id))
            self._record_lengths[str(int(id))] = data[0].size()[0]
            self._max_record_length = max(data[0].size()[0], self._max_record_length)

        self._cov_indice = self._get_cov_indice(self._column_names)
    
    def initialize(self):
        attributes = dir(self)
        for att_name in attributes:
            att = getattr(self, att_name)
            att_type = type(att)
            with tc.no_grad() :
                if att_type is estimated_parameter.Eta :
                    for id in self._ids :
                        eta_value = tc.tensor(0.1, device=self._dataset.device) 
                        att.etas.update({str(int(id)): tc.nn.Parameter(eta_value)})
                elif att_type is estimated_parameter.Eps :
                    for id in self._ids :
                        eps_value = tc.zeros(self._record_lengths[str(int(id))], requires_grad=True, device=self._dataset.device)
                        att.epss[str(int(id))] = eps_value

    def _get_cov_indice(self, column_name) :
        ESSENTIAL_COLUMNS : Iterable[str] = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT']
        cov_indice = []
        for i, col_name in enumerate(column_name):
            if col_name not in ESSENTIAL_COLUMNS :
                cov_indice.append(i)
        return tc.tensor(cov_indice)
    
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
                if type(att) is estimated_parameter.Theta :
                    att.descale()
        return self

    def _pre_forward(self, dataset):
        id = str(int(dataset[:,self._column_names.index('ID')][0]))
        self.id = id
        attribute_names = dir(self)
        for att_name in attribute_names :
            att = getattr(self, att_name)
            if type(att) is estimated_parameter.Eps or type(att) is estimated_parameter.Eta :
                att.id = id
    
    def _set_covariate_totals(self, dataset) :
        dataset = dataset.t()
        for cov_name in self._output_column_names :
            cov = getattr(self, cov_name, tc.zeros_like(dataset[0], device=dataset.device))
            setattr(self, "_total_" + cov_name, cov)
        for index in self._cov_indice:
            cov_name = self._column_names[index]
            cov = dataset[index]
            setattr(self, "_total_" + cov_name, cov)
        for cov_name in ['CMT', 'AMT', 'RATE'] :
            cov = getattr(self, cov_name, dataset[self._column_names.index(cov_name)])
            setattr(self, "_total_" + cov_name, cov)
        
    def _set_covariates(self, start, end = None) :
        for cov_name in self._output_column_names :
            cov = getattr(self, "_total_"+cov_name)
            if cov.dim() == 1 and end is None :
                cov = cov[start:]
            elif cov.dim() == 1 and end is not None :
                cov = cov[start:end]
            setattr(self, cov_name, cov)
        for index in self._cov_indice:
            cov_name = self._column_names[index]
            cov = getattr(self, "_total_"+cov_name)
            if end is None :
                cov = cov[start:]
            else :
                cov = cov[start:end]
            setattr(self, cov_name, cov)
        for cov_name in ['CMT', 'AMT', 'RATE'] :
            cov = getattr(self, "_total_"+cov_name)
            if end is None :
                cov = cov[start:]
            else :
                cov = cov[start:end]
            setattr(self, cov_name, cov)

        
    def _save_covariates_to_covariate_totals(self, start, end = None):
        for cov_name in self._output_column_names :
            if cov_name in self._column_names: continue

            cov = getattr(self, cov_name)
            cov_total = getattr(self, "_total_"+cov_name)
            if cov.dim() == 1 and end is None :
                cov_total[start:] = cov 
            elif cov.dim() == 1 and end is not None :
                cov_total[start:end] = cov
            else :
                cov_total = cov


    def _post_forward(self):
        output_columns : Dict[str, tc.Tensor] = {}
        etas : Dict[str, tc.Tensor] = {}
        epss : Dict[str, tc.Tensor] = {}
        
        attribute_names = dir(self)
        for att_name in attribute_names:
            att = getattr(self, att_name)
            att_type = type(att)
            
            if att_name.startswith('_total_') and att_name.replace('_total_', '') in self._output_column_names :
                output_columns[att_name.replace('_total_', '')] = att

            if att_type is estimated_parameter.Eta:
                etas[att_name] = att()
            elif att_type is estimated_parameter.Eps:
                epss[att_name] = att()
        
        return {'etas': etas, 'epss': epss, 'output_columns': output_columns}
    
    def _calculate_parameters(self):
        pass

    def _calculate_preds(self, start, end = None) -> tc.Tensor:
        pass

    def _calculate_error(self, y_pred) -> tc.Tensor:
        pass
    
    def forward(self, dataset):
        pass


class PredictionFunctionByTime(PredictionFunctionModule):
    def __init__(self, dataset: tc.utils.data.Dataset, column_names: Iterable[str], output_column_names: Iterable[str], *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)
 
    def forward(self, dataset) :
        self._pre_forward(dataset)

        self._set_covariate_totals(dataset)
        self._set_covariates(0)
        self._calculate_parameters()
        self._set_covariate_totals(dataset)

        f = tc.zeros(dataset.size()[0], device = dataset.device)
        amt_indice = self._get_amt_indice(dataset)
        for i in range(len(amt_indice) - 1):
            start_time_index = amt_indice[i]
 
            #누적하기 위해 앞부분 생성
            dataset_pre = dataset[:start_time_index, :]
            f_pre = tc.zeros(dataset_pre.size()[0], device = dataset.device)

            
            self.amt = self._total_AMT[start_time_index]
            self.cmt = self._total_CMT[start_time_index:]
            self.rate = self._total_RATE[start_time_index]

            dataset_cur = dataset[start_time_index:, :]
            start_time = dataset_cur[0, self._column_names.index('TIME')]
            self._set_covariates(start_time_index)

            dataset_cur_tp = dataset_cur.transpose(0,1)
            times = dataset_cur_tp[self._column_names.index('TIME'), :]
            self.t = times - start_time
            
            
            f_cur = self._calculate_preds()
            f = f + tc.cat([f_pre, f_cur], 0)
            self._save_covariates_to_covariate_totals(start_time_index)
        
        y_pred = self._calculate_error(f)
        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0
        
        self._save_covariates_to_covariate_totals(0)
        post_forward_output = self._post_forward()
        
        return {'y_pred': y_pred, 'mdv_mask': mdv_mask} | post_forward_output



#TODO update
class PredictionFunctionByODE(PredictionFunctionModule):
    """
    ordinary equation solver
    Args:
        rtol: ratio tolerance about ordinary differential equation integration
        atol: absolute tolerance about ordinary differential equation integration
    """
    rtol : float = 1e-2
    atol : float = 1e-2
                
    def _get_element(self, data, name, index) :
        return data[index, self._column_names.index(name)]
 
    def ode_function(self, t, y):
        index = (self.t < t).sum() -1
        '''
        # for k, v in self.parameter_value.items():
        #     pk_cur[k] = v[index]
        # if self.theta_scale is not None :
        #     theta = self.theta_scale(self.theta)
        # else :
        #     theta = self.theta
        # return self.pred_fn(t, y, theta, self.cur_eta, cmt, None, None, pk_cur) + self.infusion_rate * (self.infusion_end_time > t)
        '''
        self._set_covariates(index, index+1)
        return self._calculate_preds(t, y)
    def forward(self, dataset) :
        self._pre_forward(dataset)

        self._set_covariate_totals(dataset)
        self._set_covariates(0)
        self._calculate_parameters()
        self._set_covariate_totals(dataset)

        self.max_cmt = int(dataset[:,self._column_names.index('CMT')].max())
        
        y_pred_arr = []
        y_init = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_rate = tc.zeros(self.max_cmt+1, device = dataset.device)
        self.infusion_end_time = tc.zeros(self.max_cmt+1, device = dataset.device)
        amt_indice = self._get_amt_indice(dataset)
        for i in range(len(amt_indice) - 1):
            amt_slice = slice(amt_indice[i], amt_indice[i+1]+1)
            dataset_cur = dataset[amt_slice, :]

            self._set_covariates(amt_indice[i], amt_indice[i+1]+1)

            if  self.RATE[0] == 0 :                    
                bolus = tc.zeros(self.max_cmt + 1, device = dataset.device)
                
                bolus[self.CMT[0].to(tc.int64)] = self.AMT[0]
                y_init = y_init + bolus
            else :
                time = self._get_element(dataset_cur, 'TIME', 0)
 
                mask = tc.ones(self.max_cmt +1, device = dataset.device)
                mask[self.CMT[0]] = 0
 
                rate_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                rate_vector[self.CMT[0]] = self.RATE[0]
 
                infusion_during_time_vector = tc.zeros(self.max_cmt +1, device = dataset.device)
                infusion_during_time_vector[self.CMT[0]] = time + self.AMT[0] / self.RATE[0]
 
                self.infusion_rate = self.infusion_rate * mask + rate_vector
                self.infusion_end_time = self.infusion_end_time * mask + infusion_during_time_vector
                
            self.t = dataset_cur.t()[self._column_names.index('TIME')]
            result = odeint(self.ode_function, y_init, self.t, rtol=self.rtol, atol=self.atol)
            self._set_covariates(amt_indice[i], amt_indice[i+1]+1)
            y_integrated = result
            y_init = result[-1]
            
            cmts_cur = dataset_cur.t()[self._column_names.index('CMT')]
            # cmts_cur = self.CMT.t()
            cmt_mask = tc.nn.functional.one_hot(cmts_cur.to(tc.int64)).to(dataset.device)
            y_integrated = y_integrated.masked_select(cmt_mask==1)
 
            y_pred = self._calculate_error(y_integrated)
            
            y_pred_arr.append(y_pred)

        mdv_mask = dataset[:,self._column_names.index('MDV')] == 0

        post_forward_output = self._post_forward()
        
        return {'y_pred': tc.cat(y_pred_arr), 'mdv_mask': mdv_mask} | post_forward_output
