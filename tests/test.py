import unittest
import torch as tc
from torch import nn
from torchpm import covariate, models, ode, predfunc, loss
from torchpm import data
from torchpm.data import CSVDataset, OptimalDesignDataset
from torchpm.parameter import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' :
    unittest.main()
class BasementFunction(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset, output_column_names):
        super().__init__(dataset, output_column_names)
        self.theta_0 = Theta(0., 5., 10.)
        self.theta_1 = Theta(0., 30., 100.)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()
    
    def _calculate_parameters(self, para):
        para['k_a'] = self.theta_0()*tc.exp(self.eta_0())
        para['v'] = self.theta_1()*tc.exp(self.eta_1())
        para['k_e'] = self.theta_2()*tc.exp(self.eta_2())
        para['AMT'] = tc.tensor(320., device=self.dataset.device)

    def _calculate_preds(self, t, p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        return  (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        
    def _calculate_error(self, y_pred, p):
        p['v_v'] = p['v'] 
        return y_pred +  y_pred * self.eps_0() + self.eps_1()

class DatasetTest(unittest.TestCase) :
    def test_csvdataset(self) : 
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, normalization_column_names=['BWT'])

        for data, y_true in dataset:
            bwt = data.t()[column_names.index('BWT')]
            print(bwt)
    
    def test_optimal_design_dataset(self) :
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        equation_config = ode.EquationConfig(
                administrated_compartment_num=0,
                observed_compartment_num=1,
                is_infusion=True)
        dataset = OptimalDesignDataset(
                equation_config=equation_config,
                column_names = column_names,
                dosing_interval= 12,
                sampling_times_after_dosing_time=[8],
                target_trough_concentration=10.,
                repeats=2,
                include_trough_before_dose=True,
                include_last_trough=True,)
        print(dataset.column_names)

        for data, y_true in dataset:
            print(data)

class ShowTimeDVTest(unittest.TestCase):
    def test_show_time_dv(self):
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names)

        for data, y_true in dataset:
            time = data.t()[column_names.index('TIME')]
            
        
            fig = plt.figure()   
            ax = fig.add_subplot(1, 1, 1)             
            ax.plot(time, y_true, color="black")
            plt.show()

class PredFuncTest(unittest.TestCase) : 
    def test_symbolic_pred_func_multi_dose(self) : 
        dataset_file_path = './examples/THEO_multidose.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, device)
        
        output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e']

        basement_function = BasementFunction(dataset, output_column_names)

        for data, y in dataset :

            result = basement_function(data)
            print(result['y_pred'])
        

        # for id, values in simulation_result.items() :
        #     fig = plt.figure()
        #     # i += 1
        #     ax = fig.add_subplot(1, 1, 1)
        #     print('id', id)
        #     time_data : tc.Tensor = values['time'].to('cpu')
            
        #     preds : List[tc.Tensor] = values['preds']
        #     preds_tensor = tc.stack(preds).to('cpu')
        #     p95 = np.percentile(preds_tensor, 95, 0)
        #     p50 = np.percentile(preds_tensor, 50, 0)
        #     average = np.average(preds_tensor, 0)
        #     p5 = np.percentile(preds_tensor, 5, 0)
            
        #     ax.plot(time_data, p95, color="black")
        #     ax.plot(time_data, p50, color="green")
        #     ax.plot(time_data, average, color="red")
        #     ax.plot(time_data, p5, color="black")

        #     for y_pred in values['preds'] :
        #         ax.plot(time_data, y_pred.detach().to('cpu'), marker='.', linestyle='', color='gray')
            
        #     plt.show()


class FisherInformationMatrixTest(unittest.TestCase):
    def test_fisher_information_matrix(self):
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, device)
        
        output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e']

        omega = Omega([0.1, 0.1, 0.1], [True])
        sigma = Sigma([0.1], [True])

        



        print('=================================== A Optimal ===================================')
        model_config = models.ModelConfig(
                pred_function = BasementModelFIM(dataset=dataset,output_column_names=output_column_names,), 
                theta_names=['theta_0', 'theta_1', 'theta_2'],
                eta_names= ['eta_0', 'eta_1','eta_2'], 
                eps_names= ['eps_0'], 
                omega=omega, 
                sigma=sigma,
                optimal_design_creterion=loss.AOptimality())
        model = models.FOCEInter(model_config).to(device)

        model.fit_population_FIM(model.parameters())

        print('=================================== A Optimal, omega, sigma ===================================')

        model = model.descale()
        parameters = [*model.omega.parameter_values, *model.sigma.parameter_values]
        model.fit_population_FIM(parameters)
        


        for p in model.descale().named_parameters():
            print(p)

        print('=================================== Adam ===================================')

        parameters = [*model.omega.parameter_values, *model.sigma.parameter_values]
        # parameters = model.parameters()
        optimizer = tc.optim.Adam(parameters, lr=0.001)

        for i in range(100):
            model.optimization_function_FIM(optimizer)
            optimizer.step()
        

        model = model.descale()

        eval_fim_values, loss_value = model.evaluate_FIM()
        print(loss_value)
        for k, v in eval_fim_values.items():
            print(k)
            print(v)

        # eval_values = model.evaluate()
        # for k, v in eval_values.items():
        #     print(k)
        #     print(v)

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



class BasementModelFIM(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset, output_column_names):
        super().__init__(dataset, output_column_names)
        self.theta_0 = Theta(0.01, 2., 10.)
        self.theta_1 = Theta(0.01, 30., 40.)
        self.theta_2 = Theta(0.01, 0.8, 1.)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        
    
    def _calculate_parameters(self, para):
        para['k_a'] = self.theta_0() * self.eta_0().exp()
        para['v'] = self.theta_1() * self.eta_1().exp()
        para['k_e'] = self.theta_2() * self.eta_2().exp()
        para['AMT'] = tc.tensor(320., device=self.dataset.device)

    def _calculate_preds(self, t, p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        return  (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        
    def _calculate_error(self, y_pred, p):
        p['v_v'] = p['v'] 
        return y_pred +  self.eps_0()


class AnnModel(predfunc.SymbolicPredictionFunction) :
    def __init__(self, dataset, output_column_names):
        super().__init__(dataset, output_column_names)
        self.theta_0 = Theta(0., 1.5, 10.)
        self.theta_1 = Theta(0., 30., 100.)
        self.theta_2 = Theta(0, 0.08, 1)

        self.eta_0 = Eta()
        self.eta_1 = Eta()
        self.eta_2 = Eta()

        self.eps_0 = Eps()
        self.eps_1 = Eps()

        self.lin = nn.Sequential(nn.Linear(1,3),
                                    nn.Linear(3,3))
        
        
    
    def _calculate_parameters(self, para):
        
        lin_r = self.lin(para['BWT'].unsqueeze(-1)/70).t() 
        para['k_a'] = self.theta_0()*tc.exp(self.eta_0()+lin_r[0])
        para['v'] = self.theta_1()*tc.exp(self.eta_1()+lin_r[1])
        para['k_e'] = self.theta_2()*tc.exp(self.eta_2()+lin_r[2])
        para['AMT'] = tc.tensor(320., device=self.dataset.device)

        

    def _calculate_preds(self, t, p):
        dose = p['AMT'][0]
        k_a = p['k_a']
        v = p['v']
        k_e = p['k_e']
        return  (dose / v * k_a) / (k_a - k_e) * (tc.exp(-k_e*t) - tc.exp(-k_a*t))
        
    def _calculate_error(self, y_pred, p):
        p['v_v'] = p['v'] 
        return y_pred +  y_pred * self.eps_0() + self.eps_1()





class AmtModel(predfunc.SymbolicPredictionFunction) :

    def __init__(self, dataset, output_column_names):
        super().__init__(dataset, output_column_names)
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

class ODEModel(predfunc.NumericPredictionFunction) :
    def __init__(self, dataset, output_column_names):
        super().__init__(dataset, output_column_names)
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
        
        output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e']

        omega = Omega([0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205], False, requires_grads=False)
        sigma = Sigma([[0.0177], [0.0762]], [True, True], requires_grads=[False, True])


        model_config = models.ModelConfig(
                theta_names=['theta_0', 'theta_1', 'theta_2'],
                eta_names= ['eta_0', 'eta_1','eta_2'], 
                eps_names= ['eps_0','eps_1'], 
                omega=omega, 
                sigma=sigma)

        model = models.FOCEInter(model_config, pred_function = BasementFunction(dataset, output_column_names))
                                
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-5, tolerance_change= 1e-3)

        eval_values = model.evaluate()
        for k, v in eval_values.items():
            print(k)
            print(v)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        tc.manual_seed(42)
        simulation_result = model.simulate(dataset, 300)

        i = 0
        

        for id, values in simulation_result.items() :
            fig = plt.figure()
            # i += 1
            ax = fig.add_subplot(1, 1, 1)
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
    
    def test_ANN_model(self):
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, device)
        
        output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e']

        omega = Omega([0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205], False, requires_grads=False)
        sigma = Sigma([[0.0177], [0.0762]], [True, True], requires_grads=[False, True])

        model_config = models.ModelConfig(
                pred_function = AnnModel(dataset, output_column_names),
                theta_names=['theta_0', 'theta_1', 'theta_2'],
                eta_names= ['eta_0', 'eta_1','eta_2'], 
                eps_names= ['eps_0','eps_1'], 
                omega=omega, 
                sigma=sigma)

        model = models.FOCEInter(model_config)
                                
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-5, tolerance_change= 1e-3)
    
    def test_pred_amt(self):
        dataset_file_path = './examples/THEO_AMT.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)


        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]
        
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_np, column_names, device)

        output_column_names=column_names+['k_a', 'v', 'k_e']
        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([0.0177, 0.0762], [True])

        model_config = models.ModelConfig(
                pred_function = AmtModel(dataset, output_column_names), 
                theta_names=['theta_0'],
                eta_names=['eta_0', 'eta_1','eta_2'], 
                eps_names= ['eps_0','eps_1'], 
                omega=omega, 
                sigma=sigma)

        model = models.FOCEInter(model_config)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        eval_values = model.evaluate()
        for id, values in eval_values.items():
            print(id)
            for k, v in values.items() :
                print(k)
                print(v)

        for p in model.descale().named_parameters():
            print(p)


    def test_ODE(self):

        dataset_file_path = './examples/THEO_ODE.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        column_names = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT', 'COV']

        device = tc.device("cpu")
        dataset = CSVDataset(dataset_np, column_names, device)
        output_column_names=column_names+['k_a', 'v', 'k_e']

        omega = Omega([[0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205]], [False], requires_grads=True)
        sigma = Sigma([[0.0177, 0.0762]], [True])

        model_config = models.ModelConfig(
                pred_function = ODEModel(dataset, output_column_names), 
                theta_names = ['theta_0', 'theta_1', 'theta_2'],
                eta_names=['eta_0', 'eta_1','eta_2'], 
                eps_names= ['eps_0','eps_1'], 
                omega=omega, 
                sigma=sigma)
        model = models.FOCEInter(model_config)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-1, tolerance_change= 1e-2)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())

        eval_values = model.descale().evaluate()
        for id, values in eval_values.items() :
            print(id)
            for k, v in values.items() :
                print(k)
                print(v)
    
    def test_covariate_model(self) :
        def function(para):
            value = para['v_theta']()*tc.exp(para['v_eta']())*para['BWT']/70
            return {'v': value}
        cov = covariate.Covariate(['v'],[[0,32,50]],['BWT'],function)

        cov_model_decorator = covariate.CovariatePredictionFunctionDecorator([cov])
        CovModel = cov_model_decorator(BasementFunction)
        
        dataset_file_path = './examples/THEO.csv'
        dataset_np = np.loadtxt(dataset_file_path, delimiter=',', dtype=np.float32, skiprows=1)

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        column_names = ['ID', 'AMT', 'TIME', 'DV', 'CMT', "MDV", "RATE", 'BWT']
        dataset = CSVDataset(dataset_np, column_names, device)
        output_column_names=['ID', 'TIME', 'AMT', 'k_a', 'v', 'k_e']

        omega = Omega([0.4397,
                        0.0575,  0.0198, 
                        -0.0069,  0.0116,  0.0205], False, requires_grads=True)
        sigma = Sigma([[0.0177], [0.0762]], [True, True], requires_grads=[True, True])

        model_config = models.ModelConfig(
                pred_function=CovModel(dataset, output_column_names), 
                theta_names=['theta_0', 'v_theta', 'theta_2'],
                eta_names= ['eta_0', 'v_eta','eta_2'], 
                eps_names= ['eps_0','eps_1'], 
                omega=omega, 
                sigma=sigma)

        model = models.FOCEInter(model_config)
                                
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-2, tolerance_change= 1e-3)