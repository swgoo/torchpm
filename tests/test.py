
import logging
from typing import Any, Iterable
import unittest
import torch as tc
from torchpm import estimated_parameter, predfunction, models, linearode
from torchpm.data import CSVDataset
import matplotlib.pyplot as plt
import numpy as np
from torchpm.estimated_parameter import *

'''
class TestTemplate(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
'''

"""
    Args:.
    Attributes: .
"""
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
'''
'''
class BasementModel(predfunction.PredictionFunctionByTime) :
    def __init__(self, dataset: CSVDataset, column_names: List[str], output_column_names: List[str]):
        super().__init__(dataset, column_names, output_column_names)

        self.theta_0 = Theta(0., 1.5, 10.)
        self.theta_1 = Theta(0., 32., 100.)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()

        self.gut_model = linearode.Comp1GutModelFunction()

        self.initialize()
    
    def _calculate_parameters(self, **p):
        p['k_a'] = self.theta_0()*tc.exp(self.eta_0())
        p['v'] = self.theta_1()*tc.exp(self.eta_1())#*para['BWT']/70
        p['k_e'] = self.theta_2()*tc.exp(self.eta_2())
        p['AMT'] = tc.tensor(320., device=self.dataset.device)
        return p

    def _calculate_preds(self, t, **p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        comps = self.gut_model(t, k_a, k_e, dose)
        return comps[1]/v
        
    def _calculate_error(self, y_pred, **para):
        return y_pred +  y_pred * self.eps_0() + self.eps_1(), para

class AmtModel(predfunction.PredictionFunctionByTime) :
    def __init__(self, dataset, column_names, output_column_names, *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)

        self.theta_0 = Theta(0, 100, 500)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()

        self.initialize()
    
    def _calculate_parameters(self, **para):
        para['k_a'] = 1.4901*tc.exp(self.eta_0())
        para['v'] = 32.4667*tc.exp(self.eta_1())
        para['k_e'] = 0.0873*tc.exp(self.eta_2())
        para['AMT'] = para['AMT']*self.theta_0()
        return para

    def _calculate_preds(self, t, **para):
        dose = para['AMT'][0]
        k_a = para['k_a'] 
        v = para['v']
        k = para['k_e']
        
        return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
    
    def _calculate_error(self, y_pred, **para) :
        return y_pred +  y_pred * self.eps_0() + self.eps_1(), para

class ODEModel(predfunction.PredictionFunctionByODE) :
    def __init__(self, dataset: CSVDataset, column_names: List[str], output_column_names: List[str], *args, **kwargs):
        super().__init__(dataset, column_names, output_column_names, *args, **kwargs)

        self.theta_0 = Theta(0., 1.5, 10)
        self.theta_1 = Theta(0, 32, 100)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()

        self.initialize()
    
    def _calculate_parameters(self, **p):
        p['k_a'] = self.theta_0()*tc.exp(self.eta_0())
        p['v'] = self.theta_1()*tc.exp(self.eta_1())*p['COV']
        p['k_e'] = self.theta_2()*tc.exp(self.eta_2())
        return p
    
    def _calculate_preds(self, t, y, **p) -> tc.Tensor:
        mat = tc.zeros(2,2, device=y.device)
        mat[0,0] = -p['k_a']
        mat[1,0] = p['k_a']
        mat[1,1] = -p['k_e']
        return mat @ y
    
    def _calculate_error(self, y_pred : tc.Tensor, **parameters: tc.Tensor) -> tuple[Any, Dict[str, tc.Tensor]]:
        y = y_pred/parameters['v']
        return y +  y * self.eps_0() + self.eps_1(), parameters
        
class TotalTest(unittest.TestCase) :

    def setUp(self):
        pass
    def tearDown(self):
        pass
    
    def test_basement_model(self):
        dataset_file_path = './examples/THEO.csv'
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_file_path, column_names, device)
        
        pred_function_module = BasementModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e'],)

        omega = Omega([0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205], [False], requires_grads=True)
        sigma = Sigma([0.0177, 0.0762], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['0', '1', '2'],
                                eta_names= [['0', '1','2']], 
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

        tc.manual_seed(42)
        simulation_result = model.simulate(dataset, 300)

        i = 0
        fig = plt.figure()
        for id, time_data in simulation_result['times'].items() :
            i += 1
            ax = fig.add_subplot(12, 1, i)
            print('id', id)
            
            p95 = np.percentile(tc.stack(simulation_result['preds'][id]).to('cpu'), 95, 0)
            p50 = np.percentile(tc.stack(simulation_result['preds'][id]).to('cpu'), 50, 0)
            average = np.average(tc.stack(simulation_result['preds'][id]).to('cpu'), 0)
            p5 = np.percentile(tc.stack(simulation_result['preds'][id]).to('cpu'), 5, 0)
            
            ax.plot(time_data.to('cpu'), p95, color="black")
            ax.plot(time_data.to('cpu'), p50, color="green")
            ax.plot(time_data.to('cpu'), average, color="red")
            ax.plot(time_data.to('cpu'), p5, color="black")
            
            for y_pred in simulation_result['preds'][id] :
                ax.plot(time_data.to('cpu'), y_pred.detach().to('cpu'), marker='.', linestyle='', color='gray')
        plt.show()

    def test_evaluate(self):
        dataset_file_path = './examples/THEO.csv'
        
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_file_path, column_names, device)

        pred_function_module = BasementModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([0.0177, 0.0762], [True])

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
        
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        pred_function_module = AmtModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma( [[0.0177, 0.0762]], [True])

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
        
        pred_function_module = ODEModel(dataset = dataset,
                                    column_names = column_names,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([[0.0177, 0.0762]], [True])

        model = models.FOCEInter(pred_function_module, theta_names = ['0', '1'],eta_names=[['0', '1','2']], eps_names= [['0','1']], omega=omega, sigma=sigma)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-1, tolerance_change= 1e-2)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for k, v in eval_values["parameters"].items() :
            print(k)
            print(v)
    
if __name__ == '__main__' :
    unittest.main()
