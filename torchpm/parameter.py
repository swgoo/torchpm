from re import L
from typing import Dict, Iterable, List, Optional, Tuple, Union

from numpy import diag
from torch import nn
from torch.nn import ParameterDict, ParameterList
from torch import tensor, Tensor
from torch.nn.parameter import Parameter
import torch

from .misc import *

class Theta(Parameter) :
    def __init__(self, data: Tensor, fixed = False, requires_grad: bool = True):
        if data.dim() != 0:
            raise Exception("theta's dim should be 0")
        super().__init__(data, requires_grad)
        
        self.fixed = fixed

class ThetaBoundary(nn.Module):
    @torch.no_grad()
    def __init__(self, *init_values: float):
        super().__init__()

        if len(init_values) > 3 :
            raise Exception('it must be len(init_value) <= 3')

        set_parameter = lambda x : Parameter(tensor(x), requires_grad=False)

        lb = 1.e-6
        iv = tc.tensor(0)
        ub = 1.e6
        if len(init_values) == 1 :
            iv = tc.tensor(init_values[0])
            if lb > init_values[0] :
                lb = init_values[0]
            if ub < init_values[0] :
                ub = init_values[0]
        elif len(init_values) == 2 :
            if init_values[1] < init_values[0]:
                raise Exception('lower value must be lower than upper value.')
            lb = init_values[0]
            iv = (init_values[0] + init_values[1])/2
            ub = init_values[1]
        elif len(init_values) == 3 :
            if init_values[0] < init_values[1] < init_values[2] :
                lb = init_values[0]
                iv = tensor(init_values[1])
                ub = init_values[2]
            else :
                raise Exception('init_values must increase in order.')
        self.lb : Parameter = set_parameter(lb)
        self.ub : Parameter = set_parameter(ub)
        alpha = 0.1 - tc.log((iv - self.lb)/(self.ub - self.lb)/(1 - (iv - self.lb)/(self.ub - self.lb)))
        self.alpha : Parameter = set_parameter(alpha)

    def forward(self, theta: Theta) :
        if isinstance(theta, Tensor):
            return tc.exp(theta - self.alpha)/(tc.exp(theta - self.alpha) + 1)*(self.ub - self.lb) + self.lb
        else :
            raise ValueError("theta should be instance of torch.Tensor")

class ThetaInit:
    def __init__(
            self, 
            *init_values: float,
            fixed = False):
        self.init_values = init_values
        self.fixed = fixed

class Eta(ParameterDict) :
    pass

class Eps(ParameterDict):
    pass

class CovarianceVector(Parameter):
    def __init__(
                self, 
                init_values: List[float], 
                random_variable_names : Tuple[str, ...],
                is_diagonal = True, 
                fixed = False, 
                set_scale = True,
                scale_mode = True,
                requires_grad = True) :
        
        self.random_variable_names = random_variable_names

        dimension = get_dimension_of_lower_triangular_vector(init_values, is_diagonal)
        if dimension != len(random_variable_names) :
            raise Exception("lower_trangular_vector can't be converted to square matrix")

        
        with tc.no_grad() :
            if set_scale :
                data = tensor([0.1]*len(init_values), dtype = torch.float)
            else :
                data = tensor(init_values, dtype = torch.float)
            super().__init__(data, requires_grad = requires_grad)
        self.init_values = init_values
        self.fixed = fixed
        self.is_diagonal = is_diagonal
        self.set_scale = set_scale
        self.scale_mode = scale_mode

class CovarianceVectorList(ParameterList) :
    def __init__(
                self,
                parameters: Optional[Iterable[CovarianceVector]] = None,) :
        super().__init__(parameters)

    @property
    def random_variable_names(self) -> Tuple[str, ...]:
        result : Tuple[str, ...] = ()
        for covariance_vector in self :
            if type(covariance_vector) is CovarianceVector :
                result = result + covariance_vector.random_variable_names
        return result

class CovarianceBlockMatrix(nn.Module) :
    def __init__(self) :
        super().__init__()

    def forward(self, covariance_vector_list : CovarianceVectorList) -> List[Tensor]:
        m : List[Tensor] = []

        for lower_triangular_vector in covariance_vector_list :
            if type(lower_triangular_vector) is CovarianceVector :
                m.append(lower_triangular_vector_to_covariance_matrix(lower_triangular_vector, diag = lower_triangular_vector.is_diagonal))
        return m

class OmegaVectorList(CovarianceVectorList):
    pass

class SigmaVectorList(CovarianceVectorList):
    pass

class CovarianceScaler(nn.Module):
    
    def __init__(self, covariance_vector : CovarianceVector):
        super().__init__()
        self.scale = Parameter(self._set_scale(covariance_vector), requires_grad=False)

    def _get_descaled_matrix(self, scaled_matrix : Tensor):
        x = scaled_matrix * self.scale
        diag_part = scaled_matrix.diag().exp() * self.scale.diag()
        maT = tc.tril(x) - x.diag().diag() + diag_part.diag()
        return maT @ maT.t()
    
    # def _get_scaled_matrix(self, descaled_matrix : Tensor, scale : Tensor) :
    #     maT = tc.linalg.cholesky(descaled_matrix)
    #     # maT = scaled_matrix * scale - (scaled_matrix * scale).diag().diag() + (scaled_matrix.diag().exp() * scale.diag()).diag()
    #     scale_matrix = (maT - maT.diag().diag())/scale + (maT/scale).log().diag().diag()
    #     scale_matrix = scale_matrix.nan_to_num()
    #     return scale_matrix @ scale_matrix.t()

    def _set_scale(self, lower_triangular_vector : CovarianceVector) :
        init_values = tensor(lower_triangular_vector.init_values)
        var_mat = lower_triangular_vector_to_covariance_matrix(init_values, lower_triangular_vector.is_diagonal)
        # m1 = tc.linalg.cholesky(var_mat).transpose(-2, -1).conj()
        m1 = tc.linalg.cholesky(var_mat).transpose(-2, -1)
        v1 = m1.diag()
        m2 = tc.abs(10 * (m1 - v1.diag()) + (v1/tc.exp(tc.tensor(0.1))).diag())
        return m2.t()

    def forward(self, covariance_block_matrix: Tensor):
        # mat = lower_triangular_vector_to_covariance_matrix(covariance_vector, covariance_vector.is_diagonal)
        return self._get_descaled_matrix(covariance_block_matrix)

class CovarianceScalerList(nn.ModuleList):
    def __init__(self, covariance_scaler_list : Optional[List[CovarianceScaler]]):
        super().__init__(covariance_scaler_list)

    def forward(self, covariance_block_matrixes: List[Tensor]) -> List[Tensor]:
        m = []
        for covariance_vector, covariance_scaler in zip(covariance_block_matrixes, self) :
            if type(covariance_scaler) is CovarianceScaler :
                m.append(covariance_scaler(covariance_vector))
        return m

                
