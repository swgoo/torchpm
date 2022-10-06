from abc import abstractclassmethod
from ast import Call
from typing import Any, Callable, Dict, Set, Tuple, Type, TypeVar
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.nn import Module
from torchdiffeq import odeint

from torchpm import data

from .misc import *
from .para import *

from .data import EssentialColumns
from torch.nn.parameter import Parameter
from torch.nn import ParameterDict, ModuleDict, ModuleList

T = TypeVar('T')

RandomVariableInitFunction = Callable[[Any], None]

class ParameterFunction(Module) :
    def forward(
        self, 
        parameters : Dict[str, Tensor], 
        theta : Dict[str, Tensor], 
        eta : Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError()

class PredFormula(Module) :
    def forward(
        self, 
        y : Tensor, 
        t : Tensor, 
        parameters : Dict[str, Tensor], 
        theta : Dict[str, Tensor], 
        eta : Mapping [str, Tensor]) -> Tensor:
        raise NotImplementedError()
class ErrorFunction(Module) :
    def forward(
        self, 
        parameters : Dict[str, Tensor], 
        theta : Dict[str, Tensor], 
        eps : Dict[str, Tensor]) -> Tensor : 
        raise NotImplementedError()
        
class PredictionFunction(Module):
    PRED_COLUMN_NAME : str = 'PRED'

    def __init__(self,
            dataset : data.PMDataset,
            parameter_fuctions : Iterable[ParameterFunction],
            pred_formulae : Iterable[PredFormula],
            error_functions : Iterable[ErrorFunction],
            theta_Inits : Dict[str, ThetaInit],
            eta : Set[str],
            eps : Set[str],
            **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        # self._column_names = dataset.column_names
        # self._ids = dataset.ids
        # self._record_lengths = dataset.record_lengths
        self._theta_boundary_mode : bool = True
        self._theta_boundaries = ModuleDict()
        self._theta = ParameterDict()
        self.update_theta_boundaries(theta_Inits)
        
        self.eta : Dict[str, ParameterDict] = ModuleDict()  # type: ignore
        for name in eta : 
            self.eta[name] = ParameterDict()
            for id in self.dataset._ids :
                eta_dict = self.eta[name]
                if isinstance(eta_dict, ParameterDict): 
                    eta_dict.update({str(id): Eta(tensor(0.1))})
        
        self.eps : Dict[str, ParameterDict] = ModuleDict()  # type: ignore
        for name in eps : 
            self.eps[name] = ParameterDict()
            for id in self.dataset._ids :
                eps_dict = self.eps[name]
                if isinstance(eps_dict, ParameterDict): 
                    eps_dict.update({str(id): Eps(tc.zeros(self.dataset.record_lengths[id], requires_grad=True))})
        
        self.parameter_fuctions = ModuleList(parameter_fuctions)
        self.pred_formulae = ModuleList(pred_formulae)
        self.error_functions = ModuleList(error_functions)

    def update_theta_boundaries(self, theta_inits : Dict[str, Optional[ThetaInit]]):
        for k, v in theta_inits.items():
            self._theta[k] = Theta(data = tensor(0.1))
            if v is not None :
                self._theta_boundaries[k] = v.boundary
            else :
                del self._theta_boundaries[k]

    @property
    def theta(self):
        theta_dict : Dict[str, Tensor] = {}
        for k, v in self._theta.items() : 
            theta_dict[k] = self._theta_boundaries[k](v)
        return theta_dict

    def pred_function(
        self, 
        t : tc.Tensor, 
        y: Tensor, 
        parameters : Dict[str, tc.Tensor], 
        theta : Dict[str, Tensor], 
        eta : Dict[str, Tensor]) -> tc.Tensor:
        for pred_f in self.pred_formulae:
            y = pred_f(t, y, parameters, theta, eta)
        return y
    
    @property
    def theta_boundary_mode(self) :
        return self._theta_boundary_mode

    @theta_boundary_mode.setter
    @torch.no_grad()
    def theta_boundary_mode(self, value: bool):
        # turn on
        if not self._theta_boundary_mode and value :
            for k, v in self._theta.items() :
                if k in self._theta_boundaries.keys():
                    theta_boundary = self._theta_boundaries[k]
                    if type(v) is Theta and type(theta_boundary) is ThetaBoundary:
                        lb = float(theta_boundary.lb)
                        iv = float(v)
                        ub = float(theta_boundary.ub)
                        self._theta_boundaries[k] = ThetaBoundary(lb, iv, ub)
                        self._theta[k] = Theta(tensor(0.1), fixed=v.fixed, requires_grad=v.requires_grad)
            self._theta_boundary_mode = True
        # turn off
        elif self._theta_boundary_mode and not value :
            for k, v in self._theta.items():
                if type(v) is Theta and k in self._theta_boundaries.keys():
                    theta_boundary = self._theta_boundaries[k]
                    theta_value = theta_boundary(v)
                    theta = Theta(tensor(theta_value), fixed=v.fixed, requires_grad=v.requires_grad)
                    self._theta[k] = theta
            self._theta_boundary_mode = False

    # def reset_epss(self) :
    #     attributes = dir(self)
    #     for att_name in attributes :
    #         att = getattr(self, att_name)
    #         with tc.no_grad() :
    #             if type(att) is EpsDict :
    #                 for id in self._ids :
    #                     eps_value = tc.zeros_like(att[str(id)], requires_grad=True, device=att[str(id)].device)
    #                     att.update({str(id): Parameter(eps_value)})

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

        record_length = self.dataset.record_lengths[id]
    
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
                length = self.dataset.record_lengths[int(id)]
                for col_name, value in dataset.items() :
                    dataset[col_name] = value[:length]
        return datasets
    
    def get_random_variables(self, id: int) -> Tuple[Dict[str,Tensor],Dict[str,Tensor]]:
        eta : Dict[str, Parameter] = {}
        for k, v in self.eta.items() :
            eta[k] = v[str(id)]
        
        eps : Dict[str, Parameter] = {}
        for k, v in self.eps.items() :
            eps[k] = v[str(id)]

        return eta, eps
    
    def parameter_fuction(self, 
        parameters : Dict[str, Tensor],
        theta : Dict[str, Tensor],
        eta : Dict[str, Tensor],) -> Dict[str, Tensor]:
        for func in self.parameter_fuctions:
            parameters = func(parameters, theta, eta)
        return parameters
    
    def error_function(
        self,
        y_pred: Tensor, 
        parameters: Dict[str, Tensor], 
        theta : Dict[str, Tensor], 
        eta : Dict[str, Tensor],
        eps : Dict[str, Tensor]) -> tc.Tensor:
        for func in self.error_functions :
            y_pred = func(y_pred, parameters, theta, eta, eps)
        return y_pred
    
    @abstractmethod
    def forward(self, dataset):
        pass

class SymbolicPredictionFunction(PredictionFunction):   

    def __init__(self, dataset: data.PMDataset, parameter_fuctions: Iterable[ParameterFunction], pred_formulae: Iterable[PredFormula], error_functions: Iterable[ErrorFunction], theta_Inits: Dict[str, ThetaInit], eta: Set[str], eps: Set[str], **kwargs):
        super().__init__(dataset, parameter_fuctions, pred_formulae, error_functions, theta_Inits, eta, eps, **kwargs)
    
    def forward(self, batch) :

        datasets = self._batch_to_datasets(batch)
        theta : Dict[str, Tensor] = dict()
            
        output = []
        for dataset in datasets :
            amt_indice = self._get_amt_indice(dataset)

            eta, eps = self.get_random_variables(int(dataset[EssentialColumns.ID.value][0]))

            dataset = self.parameter_fuction(dataset, self.theta, eta)
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

                f_sliced = f[start_time_index:]

                t = times - start_time
                f_cur = self.pred_function(t, f_sliced, parameters_sliced, self.theta, eta)
                f = f + tc.cat([f_pre, f_cur], 0)

            pred = self.error_function(f, parameters, theta, eta, eps)
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

        #TODO set compartment initial value
    
    def _generate_ode_function(
            self, 
            times : Tensor, 
            parameters : Dict[str, Tensor],
            eta : Dict[str, Tensor], 
            infusion_rate : Tensor, 
            infusion_end_time : Tensor):
        if times.dim() != 1 :
            raise Exception("times' dim should be 1")
        def ode_function(t : Tensor, y : Tensor):
            index = (times < t).sum() - 1
            parameters_sliced = {k: v[index] for k, v in parameters.items()}
            return self.pred_function(t, y, parameters_sliced, self.theta, eta) + infusion_rate * (infusion_end_time > t)
        return ode_function
        

    def forward(self, batch : Dict[str, Tensor]) :

        datasets = self._batch_to_datasets(batch)
        output = []
        for dataset in datasets :
            max_cmt = int(dataset[EssentialColumns.CMT.value].max())

            infusion_rate = tc.zeros(max_cmt+1, device = dataset[EssentialColumns.ID.value].device)
            infusion_end_time = tc.zeros(max_cmt+1, device = dataset[EssentialColumns.ID.value].device)

            eta, eps = self.get_random_variables(int(dataset[EssentialColumns.ID.value][0]))
            
            amt_indice = self._get_amt_indice(dataset)
            dataset = self.parameter_fuction(dataset, self.theta, eta)
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
                    
                result = odeint(self._generate_ode_function(times, parameters, eta, infusion_rate, infusion_end_time), y_init, times, rtol=self.rtol, atol=self.atol)
                y_integrated = result
                y_init = result[-1]

                cmt_mask = tc.nn.functional.one_hot(cmt.to(tc.int64)).to(dataset[EssentialColumns.ID.value].device)  # type: ignore
                y_integrated = y_integrated.masked_select(cmt_mask==1)  # type: ignore

                if amt_indice[i+1]+1 == dataset[EssentialColumns.ID.value].size()[0] :
                    y_pred_arr.append(y_integrated)
                else :
                    y_pred_arr.append(y_integrated[:-1])

            pred = self.error_function(tc.cat(y_pred_arr), parameters, self.theta, eta, eps)

            post_forward_output = self._reshape_dataset(parameters)
            post_forward_output[self.PRED_COLUMN_NAME] = pred
            output.append(post_forward_output)
        return output
