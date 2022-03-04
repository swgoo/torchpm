import unittest
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torchpm import funcgen, scale, predfunction, diff, models
from torchpm.data import CSVDataset


class TestTemplate(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_post_init(self):
        assert(0,0)
    
class PytorchTest(unittest.TestCase):
    def test(self):
        t = tc.tensor([[],[],[]])
        print(t)
        # print(t.inverse())
        # print(t.slogdet())

class TotalTest(unittest.TestCase) :
    #TODO Test
    def test_pred_time(self):
        dataset_file_path = './examples/THEO.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()
                #TODO:cov의 평균값을 받아서 scale문제를 해결
                #TODO: Normalization Layer 사용시 주의. 
                #TODO: 모델 개선 scale문제로 작동을 안하는 경우임.
                self.lin = nn.Sequential(nn.Linear(1,3),
                                        nn.SELU(),
                                        nn.Linear(3,3))

            def forward(self, theta, eta, cmt, amt, bwt, tmpcov) :

                cov = tc.stack([bwt/70])
                cov = tc.exp(self.lin(cov.t()).t())

                k_a = theta[0]*tc.exp(eta[0])*tc.exp(cov[0])
                v = theta[1]*tc.exp(eta[1])*tc.exp(cov[1])
                k_e = theta[2]*tc.exp(eta[2])*tc.exp(cov[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                dose = 320
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                
                return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, eta, cmt, pk, bwt, tmpcov) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 32,  0.08], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 3
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 2
        sigma_inits = [tc.tensor([0.0177, 0.0762], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())
        assert(0, 0)
    
    #TODO test
    def pred_time_transformer(self):
        dataset_file_path = './examples/THEO.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()
                #TODO:cov의 평균값을 받아서 scale문제를 해결
                #TODO: Normalization Layer 사용시 주의. 
                #TODO: 모델 개선 scale문제로 작동을 안하는 경우임.
                self.transformer = nn.Transformer(1, 1, 2, 2, 3)
                # self.lin = nn.Sequential(nn.Linear(1,3),
                #                         nn.SELU(),
                #                         nn.Linear(3,3))

            def forward(self, theta, eta, cmt, amt, bwt, tmpcov) :

                cov = tc.stack([bwt/70])
                print(cov)
                print(eta)

                out = self.transformer(cov.t(), eta.t())
                out = out.t()
                # cov = tc.exp(self.lin(cov.t()).t())

                k_a = theta[0]*tc.exp(out[0])
                v = theta[1]*tc.exp(out[1])
                k_e = theta[2]*tc.exp(out[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                dose = 320
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                
                return (dose / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, eta, cmt, pk, bwt, tmpcov) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 32,  0.08], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 3
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 2
        sigma_inits = [tc.tensor([0.0177, 0.0762], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)

        print(model.descale().covariance_step())
        assert(0, 0)


    #TODO TEST
    def pred_amt(self):
        dataset_file_path = './examples/THEO_AMT.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.ParameterGenerator) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, theta, eta, cmt, amt, bwt, tmpcov) :

                k_a = 1.4901 
                v = 32.4667
                k_e = 0.0873

                amt = amt * theta[0] #* bwt
                return {'k_a': k_a, 'v' : v, 'k_e': k_e, 'AMT': amt}
        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :

            def __init__(self):
                super(PredFunction, self).__init__()
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                
                # dose = pk['AMT']
                k_a = pk['k_a'] 
                v = pk['v'] 
                k = pk['k_e']
                # print(dose)
                return (amt / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, eta, cmt, pk, bwt, tmpcov) :
                return y_pred + eps[0] #+  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 1
        theta_init = tc.tensor([200.], device=device)
        theta_lower_boundary  = tc.tensor([0.], device = device)
        theta_upper_boundary  = tc.tensor([500.], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        
        eta_size = 0
        omega_inits = [tc.tensor([0.4397,
                                0.0575,  0.0198, 
                                -0.0069,  0.0116,  0.0205], device = device)]
        omega_diagonals = None # []
        omega_scales = None#[scale.ScaledMatrix(omega_block, omega_diagonal) for omega_block, omega_diagonal in zip(omega_inits, omega_diagonals)]

        eps_size = 1
        sigma_inits = [tc.tensor([0.1], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_block, sigma_diagonal) for sigma_block, sigma_diagonal in zip(sigma_inits, sigma_diagonals)]

        pred_function_module = predfunction.PredictionFunctionByTime(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)
        
        model = model.to(device)
        # model.differential_module.sigma[0].requires_grad = False

        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)
        #TODO
        # print(model.descale().covariance_step())
        assert(0, 0)
    
    #TODO Test
    def ODE(self):

        dataset_file_path = './examples/THEO_ODE.csv'

        column_names = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'MDV', 'CMT', 'COV']

        device = tc.device("cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(nn.Module) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, theta, eta, cmt, amt, cov) :
                k_a = theta[0]*tc.exp(eta[0])
                v = theta[1]*tc.exp(eta[1])
                k_e = theta[2]*tc.exp(eta[2])
                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(nn.Module) :
            def __init__(self) -> None:
                super().__init__()

            def forward(self, t, y, theta, eta, cmt, amt, rate, pk) :
                mat = tc.zeros(2,2, device=y.device)
                mat[0,0] = -pk['k_a']
                mat[1,0] = pk['k_a']
                mat[1,1] = -pk['k_e']
                return mat @ y
        pred_fn = PredFunction()

        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, eta, cmt, pk, COV) :
                v = pk['v']
                return y_pred/v + y_pred/v * eps[0] + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 30,  0.1], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)

        eta_size = 3
        omega_init = [tc.tensor([0.2,
                                0.1, 0.2,
                                0.1, 0.1, 0.2], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_init[0], omega_diagonals[0])]

        eps_size = 2
        sigma_init = [tc.tensor([0.2, 0.1], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_init[0], sigma_diagonals[0])]

        pred_function_module = predfunction.PredictionFunctionByODE(dataset = dataset,
                                                        column_names = column_names,
                                                        theta_size = theta_size,
                                                        eta_size = eta_size,
                                                        eps_size = eps_size,
                                                        parameter = pk_parameter,
                                                        pred_fn  = pred_fn,
                                                        error_fn = error_fn,
                                                        theta_scale = theta_scale)

        differential_module = diff.DifferentialModule(omega_diagonals = omega_diagonals,
                                                sigma_diagonals = sigma_diagonals,
                                                omega_scales = omega_scales,
                                                sigma_scales = sigma_scales)

        model = models.FOCEInter(pred_function_module, differential_module)

        model = model.to(device)
        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        for p in model.descale().named_parameters():
            print(p)
        
        print(model.descale().covariance_step())
        assert(0, 0)


if __name__ == '__main__' :
    unittest.main()
