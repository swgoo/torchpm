import abc
from typing import ClassVar, List, Optional, Dict, Iterable, Union

class PKParameterGenerator(metaclass=abc.ABCMeta) :
    
    @abc.abstractmethod
    def __call__(self, theta, eta) -> Dict[str, tc.Tensor] :
        """
        pk parameter calculation
        returns: 
            typical values of pk parameter
        """
        pass

class PredFunctionGenerator(metaclass=abc.ABCMeta) :
    @abc.abstractmethod
    def __call__(self, t, y, theta, eta, cmt, amt, rate, pk, *cov) :
        """
        predicted value calculation
        returns: 
            vector of predicted values with respect to t
        """
        pass

class ErrorFunctionGenerator(metaclass=abc.ABCMeta) :
    @abc.abstractmethod
    def __call__(self, y_pred, eps, theta, eta, cmt, pk, *cov) :
        """
        error value calculation
        returns: 
            vector of dependent value with respect to y_pred
        """
        pass
