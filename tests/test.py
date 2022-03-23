
import logging
from typing import Iterable
import unittest
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torchpm import estimated_parameter, predfunction, models, linearode
from torchpm.data import CSVDataset

'''
class TestTemplate(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
'''

class LinearODETest(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_infusion(self):
        model = linearode.Comp1InfusionModelFunction()
        dose = tc.tensor(320.)
        t = tc.range(0,24,0.05)
        ke = tc.tensor(1.)
        rate = tc.tensor(160.)
        result = model(t, ke, dose, rate)
        print(t)
        print(result)

    def test_gut(self):
        model = linearode.Comp1GutModelFunction()
        dose = tc.tensor(320.)
        t = tc.arange(0., 24., step=0.1)
        ka = tc.tensor(0.1)
        ke = tc.tensor(1.5)
        result = model(t, ka, ke, dose)
        print(t)
        print(result)

class BasementModel(predfunction.PredictionFunctionByTime) :
    def __init__(self, dataset: tc.utils.data.Dataset, column_names: Iterable[str], output_column_names: Iterable[str], *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)

        self.theta_0 = estimated_parameter.Theta(1.5, 0, 10)
        self.theta_1 = estimated_parameter.Theta(32, 0, 100)
        self.theta_2 = estimated_parameter.Theta(0.08, 0, 1)

        self.eta_0 = estimated_parameter.Eta()
        self.eta_1 = estimated_parameter.Eta()
        self.eta_2 = estimated_parameter.Eta()

        self.eps_0 = estimated_parameter.Eps()
        self.eps_1 = estimated_parameter.Eps()

        self.gut_model = linearode.Comp1GutModelFunction()

        self.initialize()
    
    def _calculate_parameters(self, **para):
        k_a = self.theta_0()*tc.exp(self.eta_0())
        #TODO
        v = self.theta_1()*tc.exp(self.eta_1())#*para['BWT']/70
        k_e = self.theta_2()*tc.exp(self.eta_2())
        para['AMT'] = tc.tensor(320., device=self.dataset.device)
        return para | {"k_a": k_a, "v": v, "k_e": k_e}

    def _calculate_preds(self, t, amt, rate, **para) -> tc.Tensor:
        dose = amt
        k_a = para['k_a']
        v = para['v']
        k_e = para['k_e']
        # return (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        ############ Sympy Version Function #############
        comps = self.gut_model(t, k_a, k_e, dose)
        return comps[1]/v
        
    def _calculate_error(self, y_pred, **para) -> tc.Tensor:
        return y_pred +  y_pred * self.eps_0() + self.eps_1(), para

class AmtModel(predfunction.PredictionFunctionByTime) :
    def __init__(self, dataset: tc.utils.data.Dataset, column_names: Iterable[str], output_column_names: Iterable[str], *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)

        self.theta_0 = estimated_parameter.Theta(100, 0, 500)

        self.eta_0 = estimated_parameter.Eta()
        self.eta_1 = estimated_parameter.Eta()
        self.eta_2 = estimated_parameter.Eta()

        self.eps_0 = estimated_parameter.Eps()
        self.eps_1 = estimated_parameter.Eps()

        self.initialize()
    
    def _calculate_parameters(self, **para):
        k_a = 1.4901*tc.exp(self.eta_0())
        v = 32.4667*tc.exp(self.eta_1())
        k_e = 0.0873*tc.exp(self.eta_2())
        para['AMT'] = para['AMT']*self.theta_0()
        return para | {'k_a': k_a, 'v':v, 'k_e':k_e}
    
    def _calculate_preds(self, t, amt, rate, **para) -> tc.Tensor:
        dose = amt
        k_a = para['k_a'] 
        v = para['v']
        k = para['k_e']
        
        return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
    
    def _calculate_error(self, y_pred, **para) -> tc.Tensor:
        return y_pred +  y_pred * self.eps_0() + self.eps_1(), para

class TotalTest(unittest.TestCase) :

    def setUp(self):
        dataset_file_path = './examples/THEO.csv'
        self.column_names =  ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']

        self.device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        self.dataset = CSVDataset(dataset_file_path, self.column_names, self.device)

    def tearDown(self):
        pass
    
    def test_basement_model(self):
        device = self.device
        dataset = self.dataset
        column_names = self.column_names

        pred_function_module = BasementModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = estimated_parameter.Omega([tc.tensor([0.4397,
                                                        0.0575,  0.0198, 
                                                        -0.0069,  0.0116,  0.0205], device = device)], [False], requires_grads=True)
        sigma = estimated_parameter.Sigma( [tc.tensor([0.0177, 0.0762], device = device)], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['0', '1', '2'],
                                eta_names=[['0', '1','2']], 
                                eps_names= [['0','1']], 
                                omega=omega, 
                                sigma=sigma)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-5, tolerance_change= 1e-3)

        eval_values = model.evaluate()
        for k, v in eval_values.items():
            if k == 'parameters': continue
            print(k, v)
        for k, v in eval_values["parameters"].items() :
            print(k, '\n', v)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        assert(eval_values['total_loss'] < 93)

    def test_evaluate(self):
        device = self.device
        dataset = self.dataset
        column_names = self.column_names

        pred_function_module = BasementModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = estimated_parameter.Omega([tc.tensor([0.4397,
                                                        0.0575,  0.0198, 
                                                        -0.0069,  0.0116,  0.0205], device = device)], [False], requires_grads=True)
        sigma = estimated_parameter.Sigma( [tc.tensor([0.0177, 
                                                        0.0762], device = device)], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['0', '1', '2'],
                                eta_names=[['0', '1','2']], 
                                eps_names= [['0','1']], 
                                omega=omega, 
                                sigma=sigma)
        
        model = model.to(device)

        eval_values = model.evaluate()
        for k, v in eval_values.items():
            if k == 'parameters': continue
            print(k, v)
        for k, v in eval_values["parameters"].items() :
            print(k, '\n', v)
        
        # print(model.descale().covariance_step())
        print(model.descale().simulate(dataset, 10))
        

    # TODO AmtModel의 _calculate_parameters에서 AMT 연산할 수 있도록 제공
    def test_pred_amt(self):
        dataset_file_path = './examples/THEO_AMT.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = self.device
        dataset = CSVDataset(dataset_file_path, column_names, device)




        pred_function_module = AmtModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = estimated_parameter.Omega([tc.tensor([0.4397,
                                                        0.0575,  0.0198, 
                                                        -0.0069,  0.0116,  0.0205], device = device)], [False], requires_grads=True)
        sigma = estimated_parameter.Sigma( [tc.tensor([0.0177, 0.0762], device = device)], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['0'],
                                eta_names=[['0', '1','2']], 
                                eps_names= [['0','1']], 
                                omega=omega, 
                                sigma=sigma)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        eval_values = model.evaluate()
        for k, v in eval_values.items():
            if k == 'parameters': continue
            print(k, v)
        for k, v in eval_values["parameters"].items() :
            print(k, '\n', v)

        for p in model.descale().named_parameters():
            print(p)

        assert(eval_values['total_loss'] < 93)

        # print(model.descale().covariance_step())
    
    def test_ODE(self):

        dataset_file_path = './examples/THEO_ODE.csv'

        column_names = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT', 'COV']

        device = tc.device("cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)
        
        class Pred(predfunction.PredictionFunctionByODE) :
            def __init__(self, dataset: tc.utils.data.Dataset, column_names: Iterable[str], output_column_names: Iterable[str], *args, **kwargs):
                super().__init__(dataset, column_names, output_column_names, *args, **kwargs)

                self.theta_0 = estimated_parameter.Theta(1.5, 0, 10)
                self.theta_1 = estimated_parameter.Theta(32, 0, 100)
                self.theta_2 = estimated_parameter.Theta(0.08, 0, 1)

                self.eta_0 = estimated_parameter.Eta()
                self.eta_1 = estimated_parameter.Eta()
                self.eta_2 = estimated_parameter.Eta()

                self.eps_0 = estimated_parameter.Eps()
                self.eps_1 = estimated_parameter.Eps()

                self.initialize()
            
            def _calculate_parameters(self, **covs):
                k_a = self.theta_0()*tc.exp(self.eta_0())
                v = self.theta_1()*tc.exp(self.eta_1())*covs['COV']
                k_e = self.theta_2()*tc.exp(self.eta_2())
                return covs | {'k_a':k_a, 'v':v, 'k_e':k_e}
            
            def _calculate_preds(self, t, y, **para) -> tc.Tensor:
                mat = tc.zeros(2,2, device=y.device)
                mat[0,0] = -para['k_a']
                mat[1,0] = para['k_a']
                mat[1,1] = -para['k_e']
                return mat @ y
            
            def _calculate_error(self, y_pred, **para) -> tc.Tensor:
                y_pred= y_pred/para['v']
                return y_pred +  y_pred * self.eps_0() + self.eps_1(), para


        pred_function_module = Pred(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = estimated_parameter.Omega([tc.tensor([0.4397,
                                                        0.0575,  0.0198, 
                                                        -0.0069,  0.0116,  0.0205], device = device)], [False], requires_grads=True)
        # omega = estimated_parameter.Omega([tc.tensor([0.4397], device = device),
        #                                     tc.tensor([0.0198, 
        #                                                 0.0116,  0.0205], device = device)], [False, False], requires_grads=[True,True])
        sigma = estimated_parameter.Sigma( [tc.tensor([0.0177, 0.0762], device = device)], [True])

        model = models.FOCEInter(pred_function_module, theta_names = ['0', '1'],eta_names=[['0', '1','2']], eps_names= [['0','1']], omega=omega, sigma=sigma)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-1, tolerance_change= 1e-2)

        # for p in model.descale().named_parameters():
        #     print(p)

        # print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for k, v in eval_values["parameters"].items() :
            print(k)
            print(v)
    
if __name__ == '__main__' :
    unittest.main()
