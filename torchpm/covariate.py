from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, OrderedDict
from torch import nn
import torch as tc

from .data import CSVDataset
from .models import FOCEInter
from .parameter import *
from . import predfunction

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
    def __init__(self, covariates : List[Covariate]):
        self.covariates = covariates            
    
    def __call__(self, cls):
        if not issubclass(cls, predfunction.PredictionFunctionModule) :
            raise Exception('Decorated class must be ' + str(predfunction.PredictionFunctionModule))
        meta_self = self
        class CovariateModel(cls):
            
            def _set_estimated_parameters(self):
                super()._set_estimated_parameters()
                self.covariate_relationship_function = []    
                for i, cov in enumerate(meta_self.covariates):
                    function_name = ''
                    for ip_name, init_value in zip(cov.dependent_parameter_names, cov.dependent_parameter_initial_values) :
                        setattr(self, ip_name + '_theta', Theta(*init_value))
                        setattr(self, ip_name + '_eta', Eta())
                    setattr(self, '_covariate_relationship_function_' + str(i), cov.covariate_relationship_function)
                    # self.covariate_relationship_function.append(cov.covariate_relationship_function)

            def _calculate_parameters(self, parameters):
                super()._calculate_parameters(parameters)
                for i, cov in enumerate(meta_self.covariates):

                    para_dict = {}

                    # ip_dict : Dict[str, tc.Tensor] = {}
                    for name in cov.independent_parameter_names :
                        para_dict[name] = parameters[name]
                    
                    # dp_dict : Dict[str, tc.Tensor] = {}
                    for name in  cov.dependent_parameter_names :
                        pop_para_name = name + '_theta'
                        para_dict[pop_para_name] = getattr(self, pop_para_name)
                        ind_para_name = name + '_eta'
                        para_dict[ind_para_name] = getattr(self, ind_para_name)
                    
                    function = getattr(self, '_covariate_relationship_function_' + str(i))
                    
                    result_dict = function(para_dict)
                    for name, value in result_dict.items() :
                        parameters[name] = value

        return CovariateModel

@dataclass
class DeepCovariateSearching:
    dataset : CSVDataset
    BaseModel : predfunction.PredictionFunctionModule
    dependent_parameter_names : List[str]
    dependent_parameter_initial_values : List[List[float]]
    independent_parameter_names : List[str]
    eps_names : List[str]
    omega : Omega
    sigma : Sigma

    def __post_init(self) :
        pass
    
    def _get_covariate_relationship_function(self, dependent_parameter_names, independent_parameter_names) -> Dict[str, nn.Module]:
        idp_para_names_length = len(independent_parameter_names)
        dp_para_names_length = len(dependent_parameter_names)
        class CovariateRelationshipFunction(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Sequential(
                            nn.Linear(idp_para_names_length, dp_para_names_length),
                            nn.Linear(dp_para_names_length, dp_para_names_length))

            def forward(self, para_dict : Dict[str, Any]) -> Dict[str, tc.Tensor] :
                idp_para_tensor = tc.stack([para_dict[name] for name in independent_parameter_names]).t()

                lin_r = self.lin(idp_para_tensor).t()
                para_result = {}
                for i, name in enumerate(dependent_parameter_names):
                    para_result[name] = para_dict[name + '_theta']() * tc.exp(para_dict[name + '_eta']() + lin_r[i])

                return para_result
        return CovariateRelationshipFunction
    
    def _get_model(self, dependent_parameter_names, dependent_parameter_initial_values, independent_parameter_names) :
        cov = Covariate(dependent_parameter_names,
                        dependent_parameter_initial_values,
                        independent_parameter_names,
                        self._get_covariate_relationship_function(dependent_parameter_names,
                                                                independent_parameter_names)())
        cov_model_decorator = CovariateModelDecorator([cov])
        CovModel = cov_model_decorator(self.BaseModel)

        theta_names = [name + '_theta' for name in self.dependent_parameter_names]
        eta_names = [name + '_eta' for name in self.dependent_parameter_names]

        model = FOCEInter(dataset=self.dataset,
                        output_column_names=[],
                        pred_function_module=CovModel, 
                        theta_names= theta_names,
                        eta_names= eta_names, 
                        eps_names= self.eps_names,
                        omega=deepcopy(self.omega),
                        sigma=deepcopy(self.sigma))
        return model.to(self.dataset.device)

    def run(self, learning_rate : float = 1,
                    checkpoint_file_path: Optional[str] = None,
                    tolerance_grad : float= 1e-2,
                    tolerance_change : float = 1e-3,
                    max_iteration : int = 1000) :

        self.independent_parameter_names_candidate = deepcopy(self.independent_parameter_names)

        pre_total_loss = self._fit(learning_rate = learning_rate, 
                                tolerance_grad = tolerance_grad, 
                                tolerance_change= tolerance_change, 
                                max_iteration=max_iteration) 
        loss_history = OrderedDict()
        loss_history['Base'] = [True, float(pre_total_loss)]
        print('================== start searching ============================')
        for name in self.independent_parameter_names :
            self.independent_parameter_names_candidate.remove(name)
            print('============================================',
                '\n searching covariate: ', name,
                '\n=============================================')

            total_loss = self._fit(learning_rate = learning_rate, 
                                tolerance_grad = tolerance_grad, 
                                tolerance_change= tolerance_change, 
                                max_iteration=max_iteration)
            print('=================================================',
                '\n covariate : ', name,
                '\n total : ', total_loss,
                '\n pre total : ', pre_total_loss,
                '\n total-pretotal: ', total_loss - pre_total_loss,
                '\n=================================================')
            #TODO p-value 찾아서 쓰기
            if total_loss - pre_total_loss  < 3.84 :
                loss_history[name] = [False, float(total_loss)]
                print('===============================',
                '\n Removed :', name,
                '\n===================================')
                pre_total_loss = total_loss
            else :
                loss_history[name] = [True, float(total_loss)]
                self.independent_parameter_names_candidate.append(name)
        
        return self.independent_parameter_names_candidate, loss_history

    def _fit(self, learning_rate : float = 1,
                    checkpoint_file_path: Optional[str] = None,
                    tolerance_grad : float= 1e-3,
                    tolerance_change : float = 1e-3,
                    max_iteration : int = 1000) :
        model = self._get_model(self.dependent_parameter_names,
                                self.dependent_parameter_initial_values,
                                self.independent_parameter_names_candidate)
        
        model.fit_population(checkpoint_file_path=checkpoint_file_path,
                            learning_rate = learning_rate, 
                            tolerance_grad = tolerance_grad, 
                            tolerance_change= tolerance_change, 
                            max_iteration=max_iteration)

        result = model.descale().evaluate()

        total_loss = tc.tensor(0., device=self.dataset.device)
        for id, values in result.items() :
            total_loss += values['loss'].detach()
        
        return total_loss