from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union

import torch as tc

from . import scale
from .misc import *

@dataclass(repr=False, eq=False)
class DifferentialModule(tc.nn.Module) :
    omega_diagonals : Iterable[bool]
    sigma_diagonals : Iterable[bool]
    omega_scales : Optional[Iterable[scale.Scale]] = None
    sigma_scales : Optional[Iterable[scale.Scale]] = None

    def __post_init__(self):
        super(DifferentialModule, self).__init__()

        self.omega = tc.nn.ParameterList([])
        self.sigma = tc.nn.ParameterList([])
        
    def forward(self, y_pred, eta, eps):
        
        omega = self.make_covariance_matrix(self.omega, self.omega_diagonals, self.omega_scales)

        sigma = self.make_covariance_matrix(self.sigma, self.sigma_diagonals, self.sigma_scales)

        eta_size = eta.size()[-1]
        eps_size = eps.size()[-1]

        g = tc.zeros(y_pred.size()[0], eta_size, device = y_pred.device)
        for i_g, y_pred_elem in enumerate(y_pred) :
            g_elem = tc.autograd.grad(y_pred_elem, eta, create_graph=True, retain_graph=True, allow_unused=True)
            g[i_g] = g_elem[0]
        
        h = tc.zeros(y_pred.size()[0], eps_size, device = y_pred.device)
        for i_h, y_pred_elem in enumerate(y_pred) :
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
