from dataclasses import dataclass
import functools
from typing import Any, Callable, ClassVar, List, Optional
from torch import nn

import torch as tc
from torchpm.models import FOCEInter
from torchpm.parameter import *
from torchpm.predfunction import PredictionFunctionModule
from . import *

@dataclass
class Covariate:
    dependent_parameter_names : List[str]
    dependent_parameter_initial_values : List[List[float]]
    independent_parameter_names : List[str]
    covariate_relationship_function : Callable[..., Dict[str,tc.Tensor]]
    
    def __post_init__(self) :
        pass

#사용자가 만든 Predfunction module을 받아서 covariate_model 클래스를 생성한다.
class CovariateModelDecorator :
    def __init__(self, cls, covariates : List[Covariate]):
        self.covariates = covariates
    
    def __call__(self, cls):
        meta_self = self
        class NewClass(cls):
            def _set_estimated_parameters(self):
                super()._set_estimated_parameters()
                for cov in meta_self.covariates:
                    for ip_name, init_value in zip(cov.dependent_parameter_names, cov.dependent_parameter_initial_values) :
                        setattr(self, ip_name + '_theta', Theta(*init_value))
                        setattr(self, ip_name + '_eta', Eta())

            def _calculate_parameters(self, parameters):
                super()._calculate_parameters(parameters)
                for cov in meta_self.covariates:
                    ip_names = cov.independent_parameter_names
                    ip_dict : Dict[str, tc.Tensor] = {}
                    for name in ip_names :
                        ip_dict[name] = parameters[name]
                    
                    dp_names = cov.dependent_parameter_names
                    dp_dict : Dict[str, tc.Tensor] = {}
                    for name in  dp_names :
                        pop_para_name = name + '_theta'
                        dp_dict[pop_para_name] = getattr(self, pop_para_name)
                        ind_para_name = name + '_eta'
                        dp_dict[ind_para_name] = getattr(self, ind_para_name)
                    
                    function = cov.covariate_relationship_function

                    
                    result_dict = function(**ip_dict, **dp_dict)
                    for name, value in result_dict.items() :
                        parameters[name] = value

        return NewClass

class CovariateSearching:
    def __init__(self, base_model : FOCEInter, covariates : List[Covariate]) -> None:
        self.base_model = base_model
        self.covariates = covariates

    def run(self, learning_rate : float = 1,
                    checkpoint_file_path: Optional[str] = None,
                    tolerance_grad : float= 1e-5,
                    tolerance_change : float = 1e-5,
                    max_iteration : int = 1000) :
        self.base_model.fit_population(checkpoint_file_path=checkpoint_file_path, 
                                        learning_rate=learning_rate,
                                        tolerance_grad=tolerance_grad,
                                        tolerance_change= tolerance_change,
                                        max_iteration=max_iteration)
        base_model_result = self.base_model.descale().evaluate()

