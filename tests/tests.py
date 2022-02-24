import unittest
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torchpm import funcgen, scale, predfunction, diff, models
from torchpm.data import CSVDataset


class CSVDatasetTest(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_post_init(self):
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        dataset = CSVDataset("./examples/THEO.csv", column_names)

        # for data in dataset :
        #     print(data)
        
        assert(0, 0)

class TotalTest(unittest.TestCase) :
    def test_normal_layer(self) :
        v = tc.tensor([60.,60.,60.,60.,60.])
        r = F.layer_norm(v, [5])
        print(r)

    def test(self):
        dataset_file_path = './examples/THEO.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.PKParameterGenerator) :
            def __call__(self, theta, eta) :
                k_a = theta[0]*tc.exp(eta[0])
                v = theta[1]*tc.exp(eta[1])
                k_e = theta[2]*tc.exp(eta[2])

                # k_a = 1.5*tc.exp(eta[0])
                # v = 32.45*tc.exp(eta[1])
                # k_e = 0.087*tc.exp(eta[2])

                # k_a = 1.5*tc.exp(eta[0])
                # v = 1000*tc.exp(eta[1])
                # k_e = 0.087*tc.exp(eta[2])

                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(tc.nn.Module) :
        

            def __init__(self):
                super(PredFunction, self).__init__()

                #TODO:cov의 평균값을 받아서 scale문제를 해결
                #TODO: Normalization Layer 사용시 주의. 
                #TODO: 모델 개선 scale문제로 작동을 안하는 경우임.
                self.lin = nn.Sequential(nn.Linear(1,3),
                                        nn.SELU(),
                                        nn.Linear(3,3))
                
            def forward(self, t, y, theta, eta, cmt, amt, rate, pk, bwt, tmpcov) :
                
                cov = tc.stack([bwt/70])
                cov = tc.exp(self.lin(cov.t()).t())

                dose = 320
                k_a = pk['k_a']  * cov[0]
                v = pk['v'] * cov[1] 
                k = pk['k_e'] * cov[2]
                
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
                                                        pk_parameter = pk_parameter,
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
