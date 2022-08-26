
from dataclasses import dataclass
import time
from typing import Any, Callable, List, Dict, Optional
import typing
import torch as tc
import torch.distributed as dist
from torch.utils.data import DataLoader

from .parameter import *
from .data import PMDataset
from . import predfunc
from . import loss
from .data import EssentialColumns
from .misc import *

import pytorch_lightning as pl


@dataclass(eq=True, frozen=True)
class ModelConfig :
    omega : OmegaVectorList
    sigma : SigmaVectorList
    objective_function : loss.ObjectiveFunction = loss.FOCEInterObjectiveFunction()
    optimal_design_creterion : loss.DesignOptimalFunction = loss.DOptimality()

@dataclass
class OptimizationResult :
    loss : Optional[tc.Tensor] = None
    cwres_values : Optional[tc.Tensor] = None
    pred : Optional[tc.Tensor] = None
    time : Optional[tc.Tensor] = None
    mdv_mask : Optional[tc.Tensor] = None
    output_columns : Optional[Dict[str, tc.Tensor]] = None

class FOCEInter(pl.LightningModule) :

    def __init__(self,
            model_config : ModelConfig,
            pred_function : predfunc.PredictionFunction,):

        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.pred_function = pred_function

        self.covariate_block_matrix = CovarianceBlockMatrix()
        
        self.omega_vector_list = model_config.omega
        self.omega_scaler_list = CovarianceScalerList(None)
        for omega_vector in self.omega_vector_list :
            if type(omega_vector) is CovarianceVector and omega_vector.set_scale :
                omega_scaler = CovarianceScaler(omega_vector)
                self.omega_scaler_list.append(omega_scaler)
            else :
                self.omega_scaler_list.append(None)   # type: ignore

        self.sigma_vector_list = model_config.sigma
        self.sigma_scaler_list = CovarianceScalerList(None)
        for sigma_vector in self.sigma_vector_list :
            if type(sigma_vector) is CovarianceVector and sigma_vector.set_scale :
                sigma_scaler = CovarianceScaler(sigma_vector)
                self.sigma_scaler_list.append(sigma_scaler)
            else :
                self.sigma_scaler_list.append(None)   # type: ignore
        
        self.objective_function = model_config.objective_function \
            if model_config.objective_function is not None else loss.FOCEInterObjectiveFunction()
        self.design_optimal_function = model_config.optimal_design_creterion \
            if model_config.optimal_design_creterion is not None else loss.DOptimality()
        self.dataloader = None

        self.eta_names = self.omega_vector_list.random_variable_names
        self.eps_names = self.sigma_vector_list.random_variable_names

    @property
    def omega(self):
        return self._get_covariance_matrix(self.omega_vector_list, self.omega_scaler_list)

    @property
    def sigma(self):
        return self._get_covariance_matrix(self.sigma_vector_list, self.sigma_scaler_list)
    
    def _get_covariance_matrix(self, covariance_vector_list : CovarianceVectorList, covariance_scaler_list : CovarianceScalerList):
        covariance = []
        covariance_block_matrix_list = self.covariate_block_matrix(covariance_vector_list)
        for covariance_block_matrix, covariance_scaler, covariance_vector in zip(covariance_block_matrix_list, covariance_scaler_list, covariance_vector_list) :
            if type(covariance_vector) is CovarianceVector and covariance_vector.scale_mode and covariance_scaler is not None :
                covariance_block_matrix = covariance_scaler(covariance_block_matrix)
                covariance.append(covariance_block_matrix)
            else :
                covariance.append(covariance_block_matrix)
        return torch.block_diag(*covariance)


    # TODO
    def get_unfixed_parameter_values(self) -> List[nn.Parameter]:  # type: ignore
        unfixed_parameter_values = []

        for omega_vector in self.omega_vector_list:
            if type(omega_vector) is CovarianceVector and not omega_vector.fixed :
                unfixed_parameter_values.append(omega_vector) 
        
        for sigma_vector in self.sigma_vector_list:
            if type(sigma_vector) is CovarianceVector and not sigma_vector.fixed :
                unfixed_parameter_values.append(sigma_vector) 
        
        pred_function_parameters = list(self.pred_function.parameters())

        for para in pred_function_parameters :
            fixed = getattr(para, 'fixed')
            if fixed :
                continue
            else :
                unfixed_parameter_values.append(para)

        # pred_function_parameters_deleting_indices = []
        # for i, parameter in enumerate(pred_function_parameters) : 
        #     for p in self.pred_function.get_thetas().values() :
        #         if tc.is_tensor(parameter) and (p.parameter_value.data_ptr() == parameter.data_ptr()) and p.fixed :
        #             pred_function_parameters_deleting_indices.append(i)
        # pred_function_parameters_deleting_indices.sort(reverse=True)
        # for i in pred_function_parameters_deleting_indices :
        #     del pred_function_parameters[i]
        
        # unfixed_parameter_values.extend(pred_function_parameters)
        
        return unfixed_parameter_values
        
    def forward(self, dataset) :
        return self.pred_function(dataset)

    def _fit_step(self, batch, batch_idx):
        datasets = self(batch)
        total_loss = 0
        for dataset in datasets :
            id = str(dataset[EssentialColumns.ID.value][0])

            eta_dict = self.pred_function.get_attr_dict(Eta)
            eps_dict = self.pred_function.get_attr_dict(Eps)

            eta = []
            for eta_name in self.eta_names:
                eta.append(eta_dict[eta_name][id])

            eps = []
            for eps_name in self.eps_names:
                eps.append(eps_dict[eps_name][id])

            pred : Tensor = dataset[self.pred_function.PRED_COLUMN_NAME]

            eta_size = len(eta)
            eps_size = len(eps)

            h = tc.zeros(pred.size()[0], eps_size, device = pred.device)
            for i_h, y_pred_elem in enumerate(pred) :
                if eps_size > 0 :
                    for i_eps, cur_eps in enumerate(eps):
                        h_elem = tc.autograd.grad(y_pred_elem, cur_eps, create_graph=True, allow_unused=True, retain_graph=True)
                        h[i_h,i_eps] = h_elem[0][i_h]

            g = tc.zeros(pred.size()[0], eta_size, device = pred.device)
            for i_g, y_pred_elem in enumerate(pred) :
                if eta_size > 0 :
                    for i_eta, cur_eta in enumerate(eta) :
                        g_elem = tc.autograd.grad(y_pred_elem, cur_eta, create_graph=True, allow_unused=True, retain_graph=True)
                        g[i_g, i_eta] = g_elem[0]

            eta = tc.stack(eta)
            eps = tc.stack(eps)

            mdv_mask = dataset[EssentialColumns.MDV] == 0

            pred = pred.masked_select(mdv_mask)
            eta_size = g.size()[-1]
            if eta_size > 0 :
                g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
            eps_size = h.size()[-1]
            if eps_size > 0:
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()

            dv : Tensor = dataset[EssentialColumns.DV.value]
            dv_masked = dv.masked_select(mdv_mask)
            loss_value = self.objective_function(dv_masked, pred, g, h, eta, self.omega, self.sigma)
            total_loss += loss_value
    
    def _fisher_information_step(self, batch, batch_idx):
        theta_dict = self.pred_function.get_theta_parameter_values()
        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega_vector_list.parameter_values :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma_vector_list.parameter_values :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.theta_names]

        fisher_information_matrix_total = tc.zeros(cov_mat_dim, cov_mat_dim, device = dataset.device)
        for data, y_true in dataloader:

            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data, partial_differentiate_by_etas = False, partial_differentiate_by_epss = True)

            y_pred_masked = y_pred.masked_select(mdv_mask)
            
            eps_size = h.size()[-1]
            if eps_size > 0:
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()

            # y_true_masked = y_true.masked_select(mdv_mask)
            # minus_2_loglikelihood = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
            
            gr_theta = []
            for y_elem in y_pred_masked:
                gr_theta_elem = tc.autograd.grad(y_elem, thetas, create_graph=True, allow_unused=True, retain_graph=True)
                gr_theta.append(tc.stack(gr_theta_elem))
            gr_theta = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr_theta]
            gr_theta = tc.stack(gr_theta, dim=0)

            #TODO sigma 개선?
            v = gr_theta @ omega @ gr_theta.t() + (h @ sigma @ h.t()).diag().diag()
            # v = gr_theta @ omega @ gr_theta.t() +  tc.eye(gr_theta.size()[0], device = dataset.device) * (sigma)
            v = v + tc.eye(v.size()[0], device = dataset.device) * 1e-6
            v_inv = v.inverse()

            a_matrix = gr_theta.t() @ v_inv @ gr_theta
            b_matrix = a_matrix * a_matrix
            v_sqaure_inv = v_inv @ v_inv
            c_vector = []
            for i in range(gr_theta.size()[-1]):
                c_vector.append(gr_theta[:,i].t() @ v_sqaure_inv @ gr_theta[:,i])
            c_vector = tc.stack(c_vector, dim=0).unsqueeze(0)

            b_matrix = tc.cat([b_matrix, c_vector])

            d_scalar = tc.trace(v_sqaure_inv).unsqueeze(0).unsqueeze(0)

            b_matrix = tc.cat([b_matrix, tc.cat([c_vector, d_scalar], dim=1).t()], dim=1)

            fisher_information_matrix = tc.block_diag(a_matrix, b_matrix/2)

            fisher_information_matrix_total = fisher_information_matrix_total + fisher_information_matrix
        return self.design_optimal_function(fisher_information_matrix_total)

    
    def train_step(self, batch, batch_idx):

        total_loss = self._fit_step(batch, batch_idx)

        

        return total_loss

    def configure_optimizers(self):
        lr : float = self.hparams.lr  # type: ignore
        tolerance_change : float = self.hparams.tolerance_change  # type: ignore
        tolerance_grad : float = self.hparams.tolerance_grad  # type: ignore
        max_iter : int = self.hparams.max_iter # type: ignore

        optimizer = torch.optim.lbfgs.LBFGS(
                self.get_unfixed_parameter_values(), 
                lr=lr, 
                tolerance_change= tolerance_change,
                max_iter= max_iter,
                tolerance_grad= tolerance_grad,
                line_search_fn='strong_wolfe')
        return [optimizer], []
    
    def count_number_of_parameters(self):
        count = 0
        count += len(self.pred_function.get_thetas())
        count += len(self.pred_function.get_etas())
        count += len(self.pred_function.get_epss())
        for tensor in self.omega_vector_list.parameter_values : count += len(tensor)
        for tensor in self.sigma_vector_list.parameter_values : count += len(tensor)
        return count

    def predict(self) -> Dict[str,OptimizationResult]:

        dataloader = DataLoader(self.pred_function.dataset, batch_size=None, shuffle=False, num_workers=0)

        state = self.state_dict()
        # self.pred_function_module.reset_epss()

        result : Dict[str, OptimizationResult]= {}
        for data, y_true in dataloader:
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data)
            id = str(int(data[EssentialColumns.ID.value][0]))

            result[id] = OptimizationResult()
            result_cur_id = result[id]


            y_pred_masked = y_pred.masked_select(mdv_mask)
            eta_size = g.size()[-1]
            if eta_size >  0 :
                g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
            
            eps_size = h.size()[-1]
            if eps_size > 0 :
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()

            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred_masked, g, h, eta, omega, sigma)

            result_cur_id.loss = loss
            
            result_cur_id.cwres_values = cwres(y_true_masked, y_pred_masked, g, h, eta, omega, sigma)
            result_cur_id.pred = y_pred_masked
            result_cur_id.time = data[EssentialColumns.TIME.value].masked_select(mdv_mask)
            result_cur_id.mdv_mask = mdv_mask

            result_cur_id.output_columns = parameters
            
        self.load_state_dict(state, strict=False)
        return result
    
    def get_AIC(self) :
        total_loss = tc.tensor(0.).to(self.pred_function.dataset.device)
        result = self.evaluate()
        for _, values in result.items() :
                total_loss += values.loss
        k = self.count_number_of_parameters()
        return 2 * k + total_loss
        

    #TODO Property로 변경
    def descale(self) :
        self.pred_function.set_non_boundary_mode()
        self.omega_vector_list.descale()
        self.sigma_vector_list.descale()
        return self
    #TODO property로 변경
    def scale(self) :
        self.pred_function.set_boundary_mode()
        self.omega_vector_list.scale()
        self.sigma_vector_list.scale()
        return self
    
    def parameters_for_individual(self) :
        parameters = []
        for k, p in self.pred_function.get_etas().items() :
            parameters.append(p)
        for k, p in self.pred_function.get_epss().items() :
            parameters.append(p)
        return parameters
   
    def covariance_step(self) :
        dataset = self.pred_function.dataset
        theta_dict = self.pred_function.get_theta_parameter_values()

        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega_vector_list.parameter_values :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma_vector_list.parameter_values :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.theta_names]

        requires_grad_memory = []
        estimated_parameters = [*thetas,
                        *self.omega_vector_list.parameter_values,
                        *self.sigma_vector_list.parameter_values]

        for para in estimated_parameters :
            requires_grad_memory.append(para.requires_grad)
            para.requires_grad = True
 
        r_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        s_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  
 
        for data, y_true in dataloader:
            
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, _ = self(data)

            id = str(int(data[:,self.pred_function._column_names.index(EssentialColumnNames.ID.value)][0]))
            print('id', id)
 
            y_pred = y_pred.masked_select(mdv_mask)

            if eta.size()[-1] > 0 :
                g = g.t().masked_select(mdv_mask).reshape((eta.size()[-1],-1)).t()
            
            if eps.size()[0] > 0 :
                h = h.t().masked_select(mdv_mask).reshape((eps.size()[0],-1)).t()
 
            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)            

            gr = tc.autograd.grad(loss, estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)
            gr = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr]
            gr_cat = tc.concat(gr, dim=0)
            
            with tc.no_grad() :
                s_mat.add_((gr_cat.detach().unsqueeze(1) @ gr_cat.detach().unsqueeze(0))/4)
            
            for i, gr_cur in enumerate(gr_cat) :
                hs = tc.autograd.grad(gr_cur, estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)

                hs = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in hs]
                hs_cat = tc.cat(hs)
                for j, hs_elem in enumerate(hs_cat) :
                    r_mat[i,j] = r_mat[i,j] + hs_elem.detach()/2

        invR = r_mat.inverse()
        cov = invR @ s_mat @ invR
        se = cov.diag().sqrt()
        correl = covariance_to_correlation(cov)
        ei_values, ei_vectors = tc.linalg.eigh(correl)

        ei_values_sorted, _ = ei_values.sort()
        inv_cov = r_mat @ s_mat.inverse() @ r_mat

        for para, grad in zip(estimated_parameters, requires_grad_memory) :
            para.requires_grad = grad
        
        return {'cov': cov, 'se': se, 'cor': correl, 'ei_values': ei_values_sorted , 'inv_cov': inv_cov, 'r_mat': r_mat, 's_mat':s_mat}

    # TODO predict를 여러 eta와 eps로 하면 됨?
    def simulate(self, dataset : PMDataset, repeat : int) :
        """
        simulationg
        Args:
            dataset: model dataset for simulation
            repeat : simulation times
        """
        omega = self.omega_vector_list()
        sigma = self.sigma_vector_list()

        eta_parameter_values = self.pred_function.get_eta_parameter_values()
        eta_size = len(eta_parameter_values)
        mvn_eta = tc.distributions.multivariate_normal.MultivariateNormal(tc.zeros(eta_size, device=dataset.device), omega)
        etas = mvn_eta.rsample(tc.Size((len(dataset), repeat)))
        

        eps_parameter_values = self.pred_function.get_eps_parameter_values()
        eps_size = len(eps_parameter_values)
        mvn_eps = tc.distributions.multivariate_normal.MultivariateNormal(tc.zeros(eps_size, device=dataset.device), sigma)
        epss = mvn_eps.rsample(tc.Size([len(dataset), repeat, self.pred_function._max_record_length]))

        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        result : Dict[str, Dict[str, Union[tc.Tensor, List[tc.Tensor]]]] = {}
        for i, (data, _) in enumerate(dataloader):
            
            id = str(int(data[:, self.pred_function._column_names.index(EssentialColumnNames.ID.value)][0]))
            
            etas_cur = etas[i,:,:]
            epss_cur = epss[i,:,:]

            time_data = data[:,self.pred_function._column_names.index(EssentialColumnNames.TIME.value)].t()

            result[id] = {}
            result_cur_id : Dict[str, Union[tc.Tensor, List[tc.Tensor]]] = result[id]
            result_cur_id['time'] = time_data
            result_cur_id['etas'] = etas_cur
            result_cur_id['epss'] = epss_cur
            result_cur_id['preds'] = []
            for repeat_iter in range(repeat) :

                with tc.no_grad() :
                    eta_value = etas_cur[repeat_iter]
                    eps_value = epss_cur[repeat_iter]

                    for eta_i, name in enumerate(self.eta_names) :
                        eta_parameter_values[name].update({str(int(id)): tc.nn.Parameter(eta_value[eta_i])}) # type: ignore

                    for eps_i, name in enumerate(self.eps_names) :
                        eps_parameter_values[name].update({str(int(id)): tc.nn.Parameter(eps_value[:data.size()[0],eps_i])}) # type: ignore

                    r  = self.pred_function(data)
                    y_pred = r['y_pred']

                    result_cur_id['preds'].append(y_pred)
                    for name, value in r['output_columns'].items() :
                        if name not in result_cur_id.keys() :
                            result_cur_id[name] = []
                        result_cur_id[name].append(value) # type: ignore
        return result