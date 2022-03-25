
import logging
from typing import Any, Iterable
import unittest
import torch as tc
from torchpm import estimated_parameter, predfunction, models, linearode
from torchpm.data import CSVDataset
import matplotlib.pyplot as plt
import numpy as np
from torchpm.estimated_parameter import *

if __name__ == '__main__' :
    unittest.main()

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
        print('time-pred')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t.to('cpu'), result[0].detach().to('cpu').numpy())
        plt.show()

    def test_gut(self):
        model = linearode.Comp1GutModelFunction()
        dose = tc.tensor(320.)
        t = tc.arange(0., 24., step=0.1)
        ka = tc.tensor(0.1)
        ke = tc.tensor(1.5)
        result = model(t, ka, ke, dose)
        print(t)
        print(result)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t.to('cpu'), result[0].detach().to('cpu').numpy())
        plt.show()

class BasementModel(predfunction.PredictionFunctionByTime) :
    '''
        pass
    '''
    def _set_estimated_parameters(self):
        self.theta_0 = Theta(0., 1.5, 10.)
        self.theta_1 = Theta(0., 30., 100.)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()

        self.gut_model = linearode.Comp1GutModelFunction()
    
    def _calculate_parameters(self, covariates):
        covariates['k_a'] = self.theta_0()*tc.exp(self.eta_0())
        covariates['v'] = self.theta_1()*tc.exp(self.eta_1())#*para['BWT']/70
        covariates['k_e'] = self.theta_2()*tc.exp(self.eta_2())
        covariates['AMT'] = tc.tensor(320., device=self.dataset.device)

    def _calculate_preds(self, t, p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        comps = self.gut_model(t, k_a, k_e, dose)
        return comps[1]/v
        
    def _calculate_error(self, y_pred, p):
        p['v_v'] = p['v'] 
        return y_pred +  y_pred * self.eps_0() + self.eps_1()

class AmtModel(predfunction.PredictionFunctionByTime) :

    def _set_estimated_parameters(self):

        self.theta_0 = Theta(0, 100, 500)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()
        
    def _calculate_parameters(self, para):
        para['k_a'] = 1.4901*tc.exp(self.eta_0())
        para['v'] = 32.4667*tc.exp(self.eta_1())
        para['k_e'] = 0.0873*tc.exp(self.eta_2())
        para['AMT'] = para['AMT']*self.theta_0()

    def _calculate_preds(self, t, para):
        dose = para['AMT'][0]
        k_a = para['k_a'] 
        v = para['v']
        k = para['k_e']
        
        return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
    
    def _calculate_error(self, y_pred, para) :
        return y_pred +  y_pred * self.eps_0() + self.eps_1()

class ODEModel(predfunction.PredictionFunctionByODE) :
    def _set_estimated_parameters(self):
        self.theta_0 = Theta(0., 1.5, 10)
        self.theta_1 = Theta(0, 30, 100)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()
    
    def _calculate_parameters(self, p):
        p['k_a'] = self.theta_0()*tc.exp(self.eta_0())
        p['v'] = self.theta_1()*tc.exp(self.eta_1())*p['COV']
        p['k_e'] = self.theta_2()*tc.exp(self.eta_2())
    
    def _calculate_preds(self, t, y, p) -> tc.Tensor :
        mat = tc.zeros(2,2, device=y.device)
        mat[0,0] = -p['k_a']
        mat[1,0] = p['k_a']
        mat[1,1] = -p['k_e']
        return mat @ y

    def _calculate_error(self, y_pred: tc.Tensor, parameters: Dict[str, tc.Tensor]) -> tc.Tensor:
        y = y_pred/parameters['v']
        return y +  y * self.eps_0() + self.eps_1()
        
class TotalTest(unittest.TestCase) :

    def setUp(self):
        pass
    def tearDown(self):
        pass
    
    def test_basement_model(self):
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)



        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, device)
        
        pred_function_module = BasementModel(dataset = dataset,
                                    output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e'])

        omega = Omega([0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205], False, requires_grads=False)
        sigma = Sigma([[0.0177], [0.0762]], [True, True], requires_grads=[False, True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['theta_0', 'theta_1', 'theta_2'],
                                eta_names= ['eta_0', 'eta_1','eta_2'], 
                                eps_names= ['eps_0','eps_1'], 
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

        tc.manual_seed(42)
        simulation_result = model.simulate(dataset, 300)

        i = 0
        fig = plt.figure()

        for id, values in simulation_result.items() :
            i += 1
            ax = fig.add_subplot(12, 1, i)
            print('id', id)
            time_data : tc.Tensor = values['time'].to('cpu')
            
            preds : List[tc.Tensor] = values['preds']
            preds_tensor = tc.stack(preds).to('cpu')
            p95 = np.percentile(preds_tensor, 95, 0)
            p50 = np.percentile(preds_tensor, 50, 0)
            average = np.average(preds_tensor, 0)
            p5 = np.percentile(preds_tensor, 5, 0)
            
            ax.plot(time_data, p95, color="black")
            ax.plot(time_data, p50, color="green")
            ax.plot(time_data, average, color="red")
            ax.plot(time_data, p5, color="black")

            for y_pred in values['preds'] :
                ax.plot(time_data, y_pred.detach().to('cpu'), marker='.', linestyle='', color='gray')
        plt.show()
        assert(eval_values['total_loss'] < 93)
    
    def test_pred_amt(self):
        dataset_file_path = './examples/THEO_AMT.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)


        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]
        
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_np, column_names, device)

        pred_function_module = AmtModel(dataset = dataset,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([0.0177, 0.0762], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names=['theta_0'],
                                eta_names=['eta_0', 'eta_1','eta_2'], 
                                eps_names= ['eps_0','eps_1'], 
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


    def test_ODE(self):

        dataset_file_path = './examples/THEO_ODE.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        column_names = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT', 'COV']

        device = tc.device("cpu")
        dataset = CSVDataset(dataset_np, column_names, device)
        
        pred_function_module = ODEModel(dataset = dataset,
                                    output_column_names=column_names+['k_a', 'v', 'k_e'],)

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([[0.0177, 0.0762]], [True])

        model = models.FOCEInter(pred_function_module, 
                                theta_names = ['theta_0', 'theta_1', 'theta_2'],
                                eta_names=['eta_0', 'eta_1','eta_2'], 
                                eps_names= ['eps_0','eps_1'], 
                                omega=omega, 
                                sigma=sigma)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-1, tolerance_change= 1e-2)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for k, v in eval_values["parameters"].items() :
            print(k)
            print(v)