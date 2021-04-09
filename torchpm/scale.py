import abc
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union
import torch as tc

class Scale(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(vector : tc.Tensor) -> Union[tc.Tensor, Iterable[tc.Tensor]]:
        pass

class ScaledVector(Scale):
    """
    scaling for vector
    Args:
        init_value: initial value of vector
        lower_boundary: lower boundary of vector
        upper_boundary: upper boundary of vector
    Attributes: .
    """

    def __init__(self, init_value, lower_boundary = None, upper_boundary = None):
        
        if lower_boundary is None :
            self.lb = tc.zeros_like(init_value).to(init_value.device)
        else:
            self.lb = lower_boundary.to(init_value.device)

        if lower_boundary is None :
            self.ub = (tc.ones_like(init_value)*1e6).to(init_value.device)
        else:
            self.ub = upper_boundary.to(init_value.device)
        
        lb = self.lb
        iv = init_value
        ub = self.ub

        self.alpha = 0.1 - tc.log((iv - lb)/(ub - lb)/(1 - (iv - lb)/(ub - lb)))
    
    def __call__(self, vector) :
        self.alpha = self.alpha.to(vector.device)
        self.ub = self.ub.to(vector.device)
        self.lb = self.lb.to(vector.device)
        return tc.exp(vector - self.alpha)/(tc.exp(vector - self.alpha) + 1)*(self.ub - self.lb) + self.lb
        
class ScaledMatrix(Scale) :
    """
    scaling for matrix
    Args:
        lower_triangular_vector_init: matrix's lower triangular part, for initial value. if matrix is diagonal, it should be diagonal part of matrix.
        diagonal : if matrix is diagonal, it should be setted True.
    Attributes: .
    """
    
    def __init__(self, lower_triangular_vector_init , diagonal : bool):
        self.diagonal = diagonal
        self.scale = self._set_scale(lower_triangular_vector_init)
        

    def _set_scale(self, lower_triangular_vector_init) :
        var_mat = lower_triangular_vector_to_covariance_matrix(lower_triangular_vector_init, self.diagonal)
        m1 = var_mat.cholesky(upper=True)
        v1 = m1.diag()
        m2 = tc.abs(10 * (m1 - v1.diag()) + (v1/tc.exp(tc.tensor(0.1))).diag())
        return m2.t()

    def __call__(self, scaled_matrix) :
        self.scale = self.scale.to(scaled_matrix.device)
        maT = scaled_matrix * self.scale
        diag_part = scaled_matrix.diag().exp() * self.scale.diag()
        maT = tc.tril(maT) - maT.diag().diag() + diag_part.diag()
        return maT @ maT.t()
