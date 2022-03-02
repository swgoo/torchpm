import torch as tc
import abc
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union
from .misc import *

class ObjectiveFunction(metaclass=abc.ABCMeta) :
    @abc.abstractmethod
    def __call__(self, y_true, y_pred, g, h, eta, omega, sigma) :
        pass

class FOCEInterObjectiveFunction(ObjectiveFunction) :
    def __call__(self, y_true, y_pred, g, h, eta, omega, sigma) :

        res = y_true - y_pred
        v = (h @ sigma @ h.t()).diag().diag()
        
        inv_v = v.inverse()
        term1_sign, term1 = v.slogdet()
        term2 = res @ inv_v @ res
        
        eta_size = eta.size()[-1]
        if eta_size > 0 :
            inv_omega = omega.inverse()
            term3 = eta @ inv_omega @ eta
            term4_sign, term4 = omega.slogdet()
            term5_sign, term5 = (inv_omega + g.t() @ inv_v @ g).slogdet()
        else : 
            term3 = 0
            term4 = 0
            term5 = 0

        return tc.squeeze(term1 + term2 + term3 + term4 + term5)

class FOCEObjectiveFunction(ObjectiveFunction) :
    def __call__(self, y_true, y_pred, g, h, eta, omega, sigma) :
        v = (h @ sigma @ h.t()).diag().diag()
        res = y_true - y_pred 
        c = g @ omega @ g.t() + v
        r = mat_sqrt_inv(c) @ (res + g @ eta)
        c_sign, c_det_value = c.slogdet()
        return tc.squeeze(c_det_value + r.t() @ r)
