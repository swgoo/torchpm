from typing import Dict, Iterable, List, Optional, Tuple, Union

from numpy import diag
from torch import nn
from torch.nn import ParameterDict
from torch import tensor, Tensor
from torch.nn.parameter import Parameter

from .misc import *

class Theta(Parameter) :
    def __init__(self, *args, fixed : bool= False, scale : bool = True, **kwargs) :
        super().__init__(*args, **kwargs)
        self.fixed : bool = fixed
        self.scale : bool = scale
    
class ThetaScaler(nn.Module):
    def __init__(self, *init_values: float, fixed = False, requires_grad = True):
        super().__init__()

        self.fixed = fixed
        if len(init_values) > 3 :
            raise Exception('it must be len(init_value) <= 3')

        set_parameter = lambda x : Parameter(tensor(x), requires_grad=False)

        lb = set_parameter(1.e-6)
        iv = tc.tensor(0)
        ub = set_parameter(1.e6)
        if len(init_values) == 1 :
            iv = tc.tensor(init_values[0])
            if lb > init_values[0] :
                lb = set_parameter(init_values[0])
            if ub < init_values[0] :
                ub = set_parameter(init_values[0])
        elif len(init_values) == 2 :
            if init_values[1] < init_values[0]:
                raise Exception('lower value must be lower than upper value.')
            lb = set_parameter(init_values[0])
            iv = set_parameter((init_values[0] + init_values[1])/2)
            ub = set_parameter(init_values[1])
        elif len(init_values) == 3 :
            if init_values[0] < init_values[1] < init_values[2] :
                self.lb = set_parameter(init_values[0])
                iv = tensor(init_values[1])
                ub = set_parameter(init_values[2])
            else :
                raise Exception('init_values must increase in order.')
        self.lb : Parameter = lb
        self.ub : Parameter = ub
        self.alpha = 0.1 - tc.log((iv - lb)/(ub - lb)/(1 - (iv - lb)/(ub - lb)))

    def forward(self, theta) :
            theta = tc.exp(theta - self.alpha)/(tc.exp(theta - self.alpha) + 1)*(self.ub - self.lb) + self.lb
            return theta

class ThetaInit:
    """
    Args:
        init_value: initial value of scala
        lower_boundary: lower boundary of scala
        upper_boundary: upper boundary of scala
    Attributes: .
    """
    def __init__(
            self, 
            *init_value: float, 
            fixed = False, 
            requires_grad = True, 
            scale = True):
        if scale :
            self.theta_scaler = ThetaScaler(*init_value, fixed=fixed, requires_grad=requires_grad)
            self.theta = Theta(tensor(0.1))
            self.theta.fixed = fixed
        else :
            if len(init_value) == 1 :
                self.theta = Theta(tensor(init_value[0])) 
                self.theta_scaler = None
            else :
                raise Exception("init_value's length must be 1.")
        
        

class Eta(ParameterDict) :
    pass

class Eps(Dict[str, Tensor]):
    pass

class CovarianceMatrix(nn.Module) :
    def __init__(self,
                lower_triangular_vectors_init : Union[List[List[float]], List[float]] , 
                diagonals : Union[List[bool], bool],
                fixed :Union[List[bool], bool] = False,
                requires_grads : Union[List[bool], bool] = True) :
        super().__init__()
        
        self.lower_triangular_vectors_init = lower_triangular_vectors_init
        self.requires_grads = requires_grads
        
        if isinstance(fixed, bool) :
            self.fixed = [fixed]
        else :
            self.fixed = fixed

        lower_triangular_vectors_init_tensors = []
        if len(lower_triangular_vectors_init) > 0 and isinstance(lower_triangular_vectors_init[0], float) :
            lower_triangular_vectors_init = [lower_triangular_vectors_init]   # type: ignore
        
        for vector in lower_triangular_vectors_init :
            lower_triangular_vectors_init_tensors.append(tc.tensor(vector))

        self.scaled_parameter_for_save : Optional[List[nn.Parameter]] = None # type: ignore
        self.is_scale = True

        self.lower_triangular_vector_lengthes = []
        for init_vector in lower_triangular_vectors_init_tensors :
            l = init_vector.size()[0]
            self.lower_triangular_vector_lengthes.append(l)

        self.parameter_values = nn.ParameterList()
        if type(requires_grads) is bool :
            for length in self.lower_triangular_vector_lengthes:    
                self.parameter_values.append(
                    nn.Parameter( # type: ignore
                        tc.tensor([0.1]*length,  
                                device=lower_triangular_vectors_init_tensors[0].device),
                                requires_grad=requires_grads,))

        elif isinstance(requires_grads, Iterable)  :
            for length, requires_grad in zip(self.lower_triangular_vector_lengthes, requires_grads) :
                self.parameter_values.append(
                    nn.Parameter( # type: ignore
                        tc.tensor([0.1]*length,  
                            device=lower_triangular_vectors_init_tensors[0].device),
                            requires_grad=requires_grad))
        
        self.scales = []
        self.diagonals : List[bool] = []
        if type(diagonals) is bool :
            diagonals_old = diagonals
            for init_vector in lower_triangular_vectors_init_tensors:
                s = self._set_scale(init_vector, diagonals_old)
                self.scales.append(s)
                self.diagonals.append(diagonals_old)
        elif isinstance(diagonals, Iterable)  :
            self.diagonals = diagonals
            for init_vector, diagonal in zip(lower_triangular_vectors_init_tensors, diagonals):
                s = self._set_scale(init_vector, diagonal)
                self.scales.append(s)
        
    def _set_scale(self, lower_triangular_vector_init, diagonal) :
        var_mat = lower_triangular_vector_to_covariance_matrix(lower_triangular_vector_init, diagonal)
        # m1 = tc.linalg.cholesky(var_mat).transpose(-2, -1).conj()
        m1 = tc.linalg.cholesky(var_mat).transpose(-2, -1)
        v1 = m1.diag()
        m2 = tc.abs(10 * (m1 - v1.diag()) + (v1/tc.exp(tc.tensor(0.1))).diag())
        return m2.t()

    def _get_descaled_matrix(self, scaled_matrix, scale) :
        x = scaled_matrix * scale
        diag_part = scaled_matrix.diag().exp() * scale.diag()
        maT = tc.tril(x) - x.diag().diag() + diag_part.diag()
        return maT @ maT.t()
    
    def _get_scaled_matrix(self, descaled_matrix, scale) :
        maT = tc.linalg.cholesky(descaled_matrix)
        # maT = scaled_matrix * scale - (scaled_matrix * scale).diag().diag() + (scaled_matrix.diag().exp() * scale.diag()).diag()
        scale_matrix = (maT - maT.diag().diag())/scale + (maT/scale).log().diag().diag()
        scale_matrix = scale_matrix.nan_to_num()
        return scale_matrix @ scale_matrix.t()

    def descale(self) :
        if self.is_scale is True:
            with tc.no_grad() :
                self.scaled_parameters_for_save = []
                for parameter in self.parameter_values:
                    self.scaled_parameters_for_save.append(parameter.data.clone())
                
                matrixes = []
                for parameter, scale, diagonal in zip(self.parameter_values, self.scales, self.diagonals) :
                    mat = lower_triangular_vector_to_covariance_matrix(parameter, diagonal)
                    matrixes.append(self._get_descaled_matrix(mat, scale.to(mat.device)))
                
                for matrix, para, diagonal in zip(matrixes, self.parameter_values, self.diagonals) :
                    if diagonal :
                        para.data = tc.diag(matrix, 0)
                    else :
                        para.data = matrix_to_lower_triangular_vector(matrix)
            self.is_scale = False

    def scale(self) :
        if self.scale is False and self.scaled_parameter_for_save is not None:
            with tc.no_grad() :
                for parameter, data in zip(self.parameter_values, self.scaled_parameter_for_save):
                    parameter.data = data  
            self.scaled_parameter_for_save : Optional[List[nn.Parameter]] = None # type: ignore
            self.is_scale = True

    def forward(self):
        m = []
        if self.is_scale :
            for tensor, scale, diagonal in zip(self.parameter_values, self.scales, self.diagonals) :
                mat = lower_triangular_vector_to_covariance_matrix(tensor, diagonal)
                m.append(self._get_descaled_matrix(mat, scale.to(mat.device)))
        else :
            for tensor, diagonal in zip(self.parameter_values, self.diagonals) :

                m_block = lower_triangular_vector_to_covariance_matrix(tensor, diagonal)
                m.append(m_block)
        return tc.block_diag(*m)

class Omega(CovarianceMatrix):
    def __init__(self, 
            lower_triangular_vectors_init: Union[List[List[float]], List[float]], 
            diagonals: Union[List[bool], bool], 
            fixed: Union[List[bool], bool] = False, 
            requires_grads: Union[List[bool], bool] = True):
        super().__init__(lower_triangular_vectors_init, diagonals, fixed, requires_grads)
    
class Sigma(CovarianceMatrix) :
    def __init__(self, 
            lower_triangular_vectors_init: Union[List[List[float]], 
            List[float]], diagonals: Union[List[bool], bool], 
            fixed: Union[List[bool], bool] = False, 
            requires_grads: Union[List[bool], bool] = True):
        super().__init__(lower_triangular_vectors_init, diagonals, fixed, requires_grads)
