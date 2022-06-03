from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, OrderedDict
import typing
from torch import nn
import torch as tc

from .data import CSVDataset
from .models import FOCEInter, ModelConfig
from .parameter import *
from . import predfunction

@dataclass
class Covariate:
    dependent_parameter_names : List[str]
    independent_parameter_names : List[str]
    covariate_relationship_function : Callable[..., Dict[str,tc.Tensor]]

def _set_estimated_parameters(covariates : List[Covariate]):
    def _set_estimated_parameters(self):
        for i, cov in enumerate(covariates):
            setattr(self, '_covariate_relationship_function_' + str(i), cov.covariate_relationship_function)
    return _set_estimated_parameters

def _calculate_parameters(covariates : List[Covariate]):
    def _calculate_parameters(self, parameters):
        for i, cov in enumerate(covariates):
            para_dict = {}
            for name in cov.independent_parameter_names :
                para_dict[name] = parameters[name]
            
            for name in  cov.dependent_parameter_names :
                para_dict[name] = getattr(self, name)
            
            function : Callable[..., Dict[str, tc.Tensor]] = getattr(self, '_covariate_relationship_function_' + str(i))
            
            result_dict = function(**para_dict)
            for name, value in result_dict.items() :
                parameters[name] = value
    return _calculate_parameters

def _get_covariate_ann_function(independent_parameter_names, dependent_parameter_names) -> nn.Module:  # type: ignore
    idp_para_names_length = len(independent_parameter_names)
    dp_para_names_length = len(dependent_parameter_names)
    class CovariateRelationshipFunction(nn.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Sequential(  # type: ignore
                        nn.Linear(idp_para_names_length, dp_para_names_length),  # type: ignore
                        nn.Sigmoid(),  # type: ignore
                        nn.Linear(dp_para_names_length, dp_para_names_length))  # type: ignore

        def forward(self, **para_dict : Any) -> Dict[str, tc.Tensor] :
            idp_para_tensor = tc.stack([para_dict[name] for name in independent_parameter_names]).t()
            lin_r = self.lin(idp_para_tensor).t()
            para_result = {}
            for i, name in enumerate(dependent_parameter_names):
                para_result[name] = para_dict[name] * tc.exp(lin_r[i])
            return para_result

    return CovariateRelationshipFunction



#사용자가 만든 Predfunction module을 받아서 covariate_function 클래스를 생성한다.
class CovariatePredictionFunctionDecorator :
    def __init__(self, 
            covariates : List[Covariate],
            dependent_parameter_initial_values : List[List[List[float]]],):
        self.covariates : List[Covariate] = covariates
        self.dependent_parameter_initial_values = dependent_parameter_initial_values

    def __call__(self, cls):
        if not issubclass(cls, predfunction.PredictionFunction) :
            raise Exception('Decorated class must be ' + str(predfunction.PredictionFunction))
        meta_self = self
        class CovariateFunction(cls):

            def _set_estimated_parameters(self):
                super()._set_estimated_parameters()
                for i, cov in enumerate(meta_self.covariates):
                    for dp_name, init_value in zip(cov.dependent_parameter_names, meta_self.dependent_parameter_initial_values[i]) :
                        setattr(self, dp_name, Theta(*init_value))
                        setattr(self, dp_name + '_eta', Eta())
                _set_estimated_parameters(meta_self.covariates)(self)
                
            def _calculate_parameters(self, parameters):
                super()._calculate_parameters(parameters)
                for i, cov in enumerate(meta_self.covariates):
                    for dp_name in cov.dependent_parameter_names:
                        theta = getattr(self, dp_name)
                        eta = getattr(self, dp_name + '_eta')
                        parameters[dp_name] = theta().exp(eta())
                _calculate_parameters(meta_self.covariates)(self, parameters)
        
        return CovariateFunction

@dataclass
class DeepCovariateSearching:
    dataset : CSVDataset
    base_function : typing.Type[predfunction.PredictionFunction]
    dependent_parameter_names : List[str]
    dependent_parameter_initial_values : List[List[float]]
    independent_parameter_names : List[str]
    eps_names : List[str]
    omega : Omega
    sigma : Sigma
    
    def _get_covariate_relationship_function(self, dependent_parameter_names, independent_parameter_names) -> nn.Module:  # type: ignore
        idp_para_names_length = len(independent_parameter_names)
        dp_para_names_length = len(dependent_parameter_names)
        class CovariateRelationshipFunction(nn.Module):  # type: ignore
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Sequential(  # type: ignore
                            nn.Linear(idp_para_names_length, dp_para_names_length),  # type: ignore
                            nn.Sigmoid(),  # type: ignore
                            nn.Linear(dp_para_names_length, dp_para_names_length))  # type: ignore

            def forward(self, **para_dict : Any) -> Dict[str, tc.Tensor] :
                idp_para_tensor = tc.stack([para_dict[name] for name in independent_parameter_names]).t()

                lin_r = self.lin(idp_para_tensor).t()
                para_result = {}
                for i, name in enumerate(dependent_parameter_names):
                    para_result[name] = para_dict[name] * tc.exp(para_dict[name + '_eta']() + lin_r[i])

                return para_result
        return CovariateRelationshipFunction
    
    def _get_model(self, 
            dependent_parameter_names : List[str], 
            dependent_parameter_initial_values : List[List[float]], 
            independent_parameter_names : List[str]) :

        cov = Covariate(dependent_parameter_names,
                        independent_parameter_names,
                        _get_covariate_ann_function(
                                dependent_parameter_names,
                                independent_parameter_names)())
        cov_function_decorator = CovariatePredictionFunctionDecorator(
                covariates=[cov],
                dependent_parameter_initial_values = [dependent_parameter_initial_values])
        CovPredFunc = cov_function_decorator(self.base_function)

        theta_names = [name for name in self.dependent_parameter_names]
        eta_names = [name + '_eta' for name in self.dependent_parameter_names]

        model_config = ModelConfig(
                dataset=self.dataset,
                output_column_names=[],
                pred_function=CovPredFunc, 
                theta_names= theta_names,
                eta_names= eta_names, 
                eps_names= self.eps_names,
                omega=deepcopy(self.omega),
                sigma=deepcopy(self.sigma))

        model = FOCEInter(model_config)
        return model.to(self.dataset.device)

    def run(self, learning_rate : float = 1,
            checkpoint_file_path: Optional[str] = None,
            tolerance_grad : float= 1e-3,
            tolerance_change : float = 1e-3,
            max_iteration : int = 1000) :

        self.independent_parameter_names_candidate = deepcopy(self.independent_parameter_names)

        pre_total_loss = self._fit(learning_rate = learning_rate, 
                                tolerance_grad = tolerance_grad, 
                                tolerance_change= tolerance_change, 
                                max_iteration=max_iteration) 
        loss_history = []


        loss_history.append({'loss' : float(pre_total_loss),
                            'removed covariates': [],
                            'loss difference' : 0.})
        print('================== start searching ============================')
        for cov_name in self.independent_parameter_names :
            self.independent_parameter_names_candidate.remove(cov_name)
            print('============================================',
                '\n searching covariate: ', cov_name,
                '\n=============================================')

            total_loss = self._fit(learning_rate = learning_rate, 
                                tolerance_grad = tolerance_grad, 
                                tolerance_change= tolerance_change, 
                                max_iteration=max_iteration)
            print('=================================================',
                '\n covariate : ', cov_name,
                '\n total : ', total_loss,
                '\n pre total : ', pre_total_loss,
                '\n total-pretotal: ', total_loss - pre_total_loss,
                '\n=================================================')
            
            
            #TODO p-value 찾아서 쓰기
            loss_diff = float(total_loss - pre_total_loss)
            removed_covs = deepcopy(loss_history[-1]['removed covariates'])
            record = {'loss' : float(total_loss),
                    'removed covariates': removed_covs,
                    'loss difference' : loss_diff,}
            removed_covs.append(cov_name)
            loss_history.append(record)
            if loss_diff  < 3.84 :
                print('===============================',
                '\n Removed :', cov_name,
                '\n===================================')
                pre_total_loss = total_loss
            else :
                restored_record = deepcopy(loss_history[-2])
                restored_record['loss difference'] = -loss_diff
                loss_history.append(restored_record)
                self.independent_parameter_names_candidate.append(cov_name)
                
            
        return {'selected covariates': self.independent_parameter_names_candidate, 
                'history': loss_history}

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
        with tc.no_grad() :
            total_loss = tc.tensor(0., device=self.dataset.device)
            for id, values in result.items() :
                total_loss += values.loss
        
        return total_loss