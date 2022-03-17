from typing import List, Dict
import torch as tc
import diff
import predfunction

class simulation :
    def __init__(self,
                differential_module: diff.DifferentialModule,
                pred_function_module: predfunction.PredictionFunctionModule) -> None:
        self.differential_module = differential_module
        self.pred_function_module = pred_function_module

    def simulate(self, dataset, repeat) :
        """
        simulationg
        Args:
            dataset: model dataset for simulation
            repeat : simulation times
        """

        omega = self.differential_module.make_covariance_matrix(self.differential_module.omega, self.differential_module.omega_diagonals, self.differential_module.omega_scales)

        sigma = self.differential_module.make_covariance_matrix(self.differential_module.sigma, self.differential_module.sigma_diagonals, self.differential_module.sigma_scales)
 
        mvn_eta = tc.distributions.multivariate_normal.MultivariateNormal(tc.zeros(self.pred_function_module.eta_size, device=dataset.device), omega)
        etas = mvn_eta.rsample(tc.tensor([dataset.len, repeat], device=dataset.device))
 
        mvn_eps = tc.distributions.multivariate_normal.MultivariateNormal(tc.zeros(self.pred_function_module.eps_size, device=dataset.device), sigma)
        epss = mvn_eps.rsample(tc.tensor([dataset.len, repeat, self.pred_function_module.max_record_length], device=dataset.device))

        etas_original : Dict[str, tc.Tensor] = {}
        with tc.no_grad() :
            for id in self.pred_function_module.ids :
                etas_original[str(int(id))] = self.pred_function_module.etas[str(int(id))].clone()

        dataloader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        etas_result : Dict[str, tc.Tensor] = {}
        epss_result : Dict[str, tc.Tensor] = {}
        preds : Dict[str, List[tc.Tensor]] = {}
        times : Dict[str, tc.Tensor] = {}
        parameters : Dict[str, Dict[str, tc.Tensor]] = []
 
        for i, (data, y_true) in enumerate(dataloader):
            
            id = str(int(data[:, self.pred_function_module.column_names.index('ID')][0]))
            
            etas_cur = etas[i,:,:]
            epss_cur = epss[i,:,:]

            time_data = data[:,self.pred_function_module.column_names.index('TIME')].t()

            times[id] = time_data
            etas_result[id] = etas_cur
            epss_result[id] = epss_cur
            preds[id] = []
            parameters[id] = []

            for repeat_iter in range(repeat) :

                with tc.no_grad() :
                    eta_value = etas_cur[repeat_iter]
                    eps_value = epss_cur[repeat_iter]

                    self.pred_function_module.etas.update({str(int(id)): tc.nn.Parameter(eta_value)})

                    self.pred_function_module.epss.update({str(int(id)): tc.nn.Parameter(eps_value[:data.size()[0],:])})

                    y_pred, _, _, _, parameter_value = self.pred_function_module(data)

                    preds[id].append(y_pred)
                    parameters[id].append(parameter_value)

        with tc.no_grad() :
            for id in self.pred_function_module.ids :
                self.pred_function_module.etas.update({str(int(id)): tc.nn.Parameter(etas_original[str(int(id))])})

        return {'times': times, 'preds': preds, 'etas': etas_result, 'epss': epss_result, 'parameters': parameters}