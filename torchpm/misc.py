import torch as tc
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union

def mat_sqrt_inv(mat) :
    ei_values, ei_vectors = tc.linalg.eigh(mat, UPLO='U')
    d = ei_values.abs()
    d2 = d.rsqrt()
    return ei_vectors @ d2.diag() @ ei_vectors.t()

def lower_triangular_vector_to_covariance_matrix(lower_triangular_vector, diag : bool = True) :
    if diag :
        return lower_triangular_vector.diag()
    else :
        lower_triangular_vector_len = len(lower_triangular_vector)
        dim = int(((8*lower_triangular_vector_len+1)**(1/2)-1)//2)
        m = tc.zeros(dim, dim, device=lower_triangular_vector.device)
        tril_indices = tc.tril_indices(row=dim, col=dim, offset=0)
        m[tril_indices[0], tril_indices[1]] = lower_triangular_vector
        return m + tc.tril(m).transpose(0,1) - m.diag().diag()

def matrix_to_lower_triangular_vector(m):
    tril_indices = m.tril().nonzero().t()
    return m[tril_indices[0], tril_indices[1]]

def cwres(y_true, y_pred, g, h, eta, omega, sigma) :    
    if eta.size()[-1] > 0:
        term1 = g @ omega @ g.t()
        term2 =  g @ eta
    else : term1, term2 = (0,0)

    c = term1 + (h @ sigma @ h.t()).diag().diag()
    return mat_sqrt_inv(c) @ (y_true - y_pred + term2)

def covariance_to_correlation(m):
    d = m.diag().sqrt()
    return ((m.t()/d).t())/d

# def mx(m, x) :
#     ei_values, ei_vectors = symeig(m)
#     return ei_vectors @ (ei_values.abs()**x).diag() @ ei_vectors.t()
