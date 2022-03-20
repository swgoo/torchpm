import abc
from dataclasses import dataclass, field
from turtle import forward
from typing import ClassVar, List, Optional, Dict, Iterable, Union
import torch as tc
from torch import nn
#TODO: relative import 오류수정
from .misc import *
# from misc import *

class Theta(nn.Module):
    """
    scaling for 
    Args:
        init_value: initial value of scala
        lower_boundary: lower boundary of scala
        upper_boundary: upper boundary of scala
    Attributes: .
    """
    def __init__(self, init_value, lower_boundary = 0.0, upper_boundary = 1.0e6, requires_grad = True):
        super().__init__()
        self.scale = True

        self.lb = tc.tensor(lower_boundary)
        self.ub = tc.tensor(upper_boundary)
        
        lb = self.lb
        iv = init_value
        ub = self.ub

        self.alpha = 0.1 - tc.log((iv - lb)/(ub - lb)/(1 - (iv - lb)/(ub - lb)))
        self.parameter = nn.parameter(tc.tensor(0.1), requires_grad = requires_grad)
    
    def descale(self) :
        if self.scale is True:
            with tc.no_grad() :
                self.scaled_parameter_for_save = self.parameter.data.clone()
                self.parameter.data = self.forward()
                self.scale = False
    
    def scale(self) :
        if self.scale is False :
            with tc.no_grad() :
                self.parameter.data = self.scaled_parameter_for_save
                self.scaled_parameter_for_save = None
                self.scale = True
             
    def forward(self) :
        if self.scale :
            return tc.exp(self.parameter - self.alpha)/(tc.exp(self.parameter - self.alpha) + 1)*(self.ub - self.lb) + self.lb
        else :
            return self.parameter

class Eta(nn.Module) :
    def __init__(self) -> None:
        super().__init__()
        self.parameters = nn.ParameterDict()
    
    def forward(self):
        return self.parameters[str(self.id)]

class Eps(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameters : Dict[str, tc.TensorType] = {}
    
    def forward(self):
        return self.parameters[str(self.id)]


class Omega(nn.Module) :
    def __init__(self,
                eta_names : Iterable[Iterable[str]],
                lower_triangular_vectors_init : Iterable[Iterable[tc.Tensor]] , 
                diagonals : Iterable[bool],
                is_scale = True) :
        super().__init__()
        self.diagonals = diagonals
        self.eta_names = eta_names

        #TODO 여러개 받도록 변경
        self.scales = self._set_scale(lower_triangular_vectors_init)

        #TODO 여러개 받도록 변경
        self.lower_triangular_vector_length = lower_triangular_vectors_init.size()[0]
        

    def _set_scale(self, lower_triangular_vector_init) :
        var_mat = lower_triangular_vector_to_covariance_matrix(lower_triangular_vector_init, self.diagonal)
        m1 = tc.linalg.cholesky(var_mat).transpose(-2, -1).conj()
        # m1 = var_mat.cholesky(upper=True)
        v1 = m1.diag()
        m2 = tc.abs(10 * (m1 - v1.diag()) + (v1/tc.exp(tc.tensor(0.1))).diag())
        return m2.t()
    


    def descale(self) :
        pass

    def __call__(self, scaled_matrix) :
        self.scale = self.scale.to(scaled_matrix.device)
        maT = scaled_matrix * self.scale
        diag_part = scaled_matrix.diag().exp() * self.scale.diag()
        maT = tc.tril(maT) - maT.diag().diag() + diag_part.diag()
        return maT @ maT.t()
    
    def forward():
        pass

class Sigma(nn.Module) :
    def __init__(self) -> None:
        super().__init__()
    
    def forward():
        pass