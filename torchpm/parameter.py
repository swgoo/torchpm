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
    def __init__(
            self, 
            *init_values: float, 
            fixed = False, 
            requires_grad = True, 
            set_boundary = True,
            boundary_mode = True):
        self.init_values = init_values
        self.fixed = fixed
        self.set_boundary = set_boundary
        self.boundary_mode = boundary_mode

        if set_boundary :
            super().__init__(tensor(0.1), requires_grad = requires_grad)
        else :
            if len(init_values) == 1 :
                super().__init__(tensor(init_values[0]), requires_grad = requires_grad) 
            else :
                raise Exception("init_values' length must be 1.")
    
class ThetaBoundary(nn.Module):
    def __init__(self, *init_values: float):
        super().__init__()

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

class Eta(ParameterDict) :
    pass

class Eps(ParameterDict):
    pass

class CovarianceVectorList(nn.Module) :
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
        if self.is_scale is False and self.scaled_parameter_for_save is not None:
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
                parameters: Optional[Iterable[CovarianceVector]] = None, 
                ) :
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

class Omega(CovarianceVectorList):
    pass

class Sigma(CovarianceVectorList):
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
    def __init__(self, covariance_scaler_list : List[CovarianceScaler]):
        super().__init__(covariance_scaler_list)

    def forward(self, covariance_block_matrixes: List[Tensor]) -> List[Tensor]:
        m = []
        for covariance_vector, covariance_scaler in zip(covariance_block_matrixes, self) :
            if type(covariance_scaler) is CovarianceScaler :
                m.append(covariance_scaler(covariance_vector))
        return m

                
