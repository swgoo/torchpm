from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union

import torch as tc

from . import scale
from .misc import *
#TODO model로 통합
@dataclass(repr=False, eq=False)
class DifferentialModule(tc.nn.Module) :
    omega_diagonals : Iterable[bool]
    sigma_diagonals : Iterable[bool]
    omega_scales : Optional[Iterable[scale.Scale]] = None
    sigma_scales : Optional[Iterable[scale.Scale]] = None

    def __post_init__(self):
        super(DifferentialModule, self).__init__()

        self.omega = tc.nn.ParameterList([])
        if self.omega_scales is not None :
            for scale in self.omega_scales :
                tensor = tc.tensor([0.1] * scale.lower_triangular_vector_length)
                self.omega.append(tc.nn.Parameter(tensor))

        self.sigma = tc.nn.ParameterList([])
        if self.sigma_scales is not None :
            for scale in self.sigma_scales :
                tensor = tc.tensor([0.1] * scale.lower_triangular_vector_length)
                self.sigma.append(tc.nn.Parameter(tensor))
        
    def forward(self, y_pred, eta, eps):
        eta_size = eta.size()[-1]
        eps_size = eps.size()[-1]
        
        if eta_size > 0 :
            omega = self.make_covariance_matrix(self.omega, self.omega_diagonals, self.omega_scales)
        else :
            omega = None

        if eps_size > 0 :
            sigma = self.make_covariance_matrix(self.sigma, self.sigma_diagonals, self.sigma_scales)
        else : 
            sigma = None
        

        g = tc.zeros(y_pred.size()[0], eta_size, device = y_pred.device)
        for i_g, y_pred_elem in enumerate(y_pred) :
            if eta_size > 0 :
                g_elem = tc.autograd.grad(y_pred_elem, eta, create_graph=True, retain_graph=True, allow_unused=True)
                g[i_g] = g_elem[0]
        
        h = tc.zeros(y_pred.size()[0], eps_size, device = y_pred.device)
        for i_h, y_pred_elem in enumerate(y_pred) :
            if eps_size > 0 :
                h_elem = tc.autograd.grad(y_pred_elem, eps, create_graph=True, retain_graph=True, allow_unused=True)
                h[i_h] = h_elem[0][i_h]

        return y_pred, g, h, omega, sigma
    
    def make_covariance_matrix(self, flat_tensors, diagonals, scales = None):
        m = []
        if scales is not None :
            for tensor, scale, diagonal in zip(flat_tensors, scales, diagonals) :
                if scale is not None :
                    m.append(scale(lower_triangular_vector_to_covariance_matrix(tensor, diagonal)))
                else :
                    m.append(lower_triangular_vector_to_covariance_matrix(tensor, diagonal))
            return tc.block_diag(*m)
        else :
            for tensor, diagonal in zip(flat_tensors, diagonals) :
                m.append(lower_triangular_vector_to_covariance_matrix(tensor, diagonal))
            return tc.block_diag(*m)
    
    #TODO
    def descale(self) :
        def fn(scaled_matrix_parameters, scales, diagonals) :
            matrixes = []
            for vector, scale, diagonal in zip(scaled_matrix_parameters, scales, diagonals) :
                matrixes.append(scale(lower_triangular_vector_to_covariance_matrix(vector, diagonal)))
            
            for matrix, para, diagonal in zip(matrixes, scaled_matrix_parameters, diagonals) :
                if diagonal :
                    para.data = tc.diag(matrix, 0)
                else :
                    para.data = matrix_to_lower_triangular_vector(matrix)

        with tc.no_grad() :
            if self.omega_scales is not None :
                fn(self.omega, self.omega_scales, self.omega_diagonals)

            if self.sigma_scales is not None :
                fn(self.sigma, self.sigma_scales, self.sigma_diagonals)

        self.omega_scales = None
        self.sigma_scales = None
        
        return self
    
    def get_descaled_parameters(self):
        def fn(scaled_matrix_parameters, scales, diagonals) :
            matrixes = []
            for vector, scale, diagonal in zip(scaled_matrix_parameters, scales, diagonals) :
                matrixes.append(scale(lower_triangular_vector_to_covariance_matrix(vector, diagonal)))
            
            result = []
            for matrix, diagonal in zip(matrixes, diagonals) :
                if diagonal :
                    result.append(tc.diag(matrix, 0))
                else :
                    result.append(matrix_to_lower_triangular_vector(matrix))
            return result

        result = {}
        with tc.no_grad() :
            if self.omega_scales is not None :
                result['omega'] = fn(self.omega, self.omega_scales, self.omega_diagonals)

            if self.sigma_scales is not None :
                result['sigma'] = fn(self.sigma, self.sigma_scales, self.sigma_diagonals)
        
        return result
