import unittest
import torch as tc
from torchpm import funcgen, scale, predfunction, diff, models
from torchpm.data import CSVCOVDataset, CSVDataset


class CSVCOVDatasetTest(unittest.TestCase) :
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_post_init(self):
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        dataset = CSVCOVDataset("./examples/THEO.csv", "./examples/THEO_COV.csv", column_names)

        # for data in dataset :
        #     print(data)
        
        assert(0, 0)
    




class TotalTest(unittest.TestCase) :
    def test(self):
        dataset_file_path = 'https://raw.githubusercontent.com/yeoun9/torchpm/main/examples/THEO.csv'
        column_names = ['ID', 'AMT', 'TIME',    'DV',   'BWT', 'CMT', "MDV", "tmpcov", "RATE"]

        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        dataset = CSVDataset(dataset_file_path, column_names, device)

        class PKParameter(funcgen.PKParameterGenerator) :
            def __call__(self, theta, eta) :
                k_a = theta[0]*tc.exp(eta[0])
                v = theta[1]*tc.exp(eta[1])
                k_e = theta[2]*tc.exp(eta[2])
                return {'k_a': k_a, 'v' : v, 'k_e': k_e}
        pk_parameter = PKParameter()

        class PredFunction(funcgen.PredFunctionGenerator) :
            def __call__(self, t, y, theta, eta, cmt, amt, rate, pk, bwt, tmpcov) :
                k_a = pk['k_a']
                v = pk['v']
                k = pk['k_e']
                return (320 / v * k_a) / (k_a - k) * (tc.exp(-k*t) - tc.exp(-k_a*t))
        pred_fn = PredFunction()


        class ErrorFunction(funcgen.ErrorFunctionGenerator):
            def __call__(self, y_pred, eps, theta, eta, cmt, pk, bwt, tmpcov) :
                return y_pred +  y_pred * eps[0]  + eps[1]

        error_fn = ErrorFunction()

        theta_size = 3
        theta_init = tc.tensor([ 1.5, 30,  0.1], device=device)
        theta_lower_boundary  = tc.tensor([0.,0.,0.], device = device)
        theta_upper_boundary  = tc.tensor([10,100,10], device = device)
        theta_scale = scale.ScaledVector(theta_init, lower_boundary = theta_lower_boundary, upper_boundary = theta_upper_boundary)
        theta_init = tc.tensor([ 0.1, 0.1,  0.1], device=device)



        eta_size = 3
        omega_init = [tc.tensor([0.2,
                                0.1, 0.2,
                                0.1, 0.1, 0.2], device = device)]
        omega_diagonals = [False]
        omega_scales = [scale.ScaledMatrix(omega_init[0], omega_diagonals[0])]
        omega_inits = [tc.tensor([ 0.1,
                                0.1,  0.1,
                                0.1,  0.1,  0.1], device = device)]

        eps_size = 2
        sigma_init = [tc.tensor([0.2, 0.1], device = device)]
        sigma_diagonals = [True]
        sigma_scales = [scale.ScaledMatrix(sigma_init[0], sigma_diagonals[0])]
        sigma_inits = [tc.tensor([0.1, 0.1], device = device)]



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

        model.pred_function_module.theta = tc.nn.Parameter(theta_init)
        model.differential_module.sigma = tc.nn.ParameterList([tc.nn.Parameter(tensor) for tensor in sigma_inits])
        model.differential_module.omega = tc.nn.ParameterList([tc.nn.Parameter(tensor) for tensor in omega_inits])
        model = model.to(device)

        model.fit_population(learning_rate = 1, tolerance_grad = 1e-3, tolerance_change= 1e-3)

        assert(0, 0)


if __name__ == '__main__' :
    unittest.main()
