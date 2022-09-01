
from dataclasses import dataclass
import time
from typing import Any, Callable, List, Dict, Optional
import typing
import torch.distributed as dist
from torch.utils.data import DataLoader

from .para import *
from .data import PMDataset
from . import predfunc
from . import loss
from .data import EssentialColumns
from .misc import *

import pytorch_lightning as pl


@dataclass
class ModelConfig :
    dataset : PMDataset
    pred_function : Type[predfunc.PredictionFunction]
    omega : Union[CovarianceVectorInitList,Tuple[CovarianceVectorList,CovarianceScalerList]]
    sigma : Union[CovarianceVectorInitList,Tuple[CovarianceVectorList,CovarianceScalerList]]
    objective_function : loss.NonLinearMixedModelObjectiveFunction = loss.FOCEInterObjectiveFunction()

@dataclass
class OptimizationResult :
    loss : Optional[Tensor] = None
    aic : Optional[Tensor] = None
    cwres_values : Optional[Tensor] = None
    pred : Optional[Tensor] = None
    time : Optional[Tensor] = None
    mdv_mask : Optional[Tensor] = None
    output_columns : Optional[Dict[str, Tensor]] = None

class FOCEInter(pl.LightningModule) :

    def __init__(self,
            model_config : ModelConfig,
            lr = 1.):

        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])

        self.model_config = model_config
        self.pred_function = self.model_config.pred_function(self.model_config.dataset)

        self._scale_mode = True

        if type(model_config.omega) is Tuple[CovarianceVectorList, CovarianceScalerList] :
            self.omega_vector_list, self.omega_scaler_list = model_config.omega
        elif type(model_config.omega) is CovarianceVectorInitList :
            self.omega_vector_list, self.omega_scaler_list = model_config.omega.covariance_list(), model_config.omega.scaler_list()
        
        if type(model_config.sigma) is Tuple[CovarianceVectorList, CovarianceScalerList] :
            self.sigma_vector_list, self.sigma_scaler_list = model_config.sigma
        elif type(model_config.sigma) is CovarianceVectorInitList :
            self.sigma_vector_list, self.sigma_scaler_list = model_config.sigma.covariance_list(), model_config.sigma.scaler_list()
        
        self.objective_function = model_config.objective_function \
            if model_config.objective_function is not None else loss.FOCEInterObjectiveFunction()

        self.eta_names = self.omega_vector_list.random_variable_names()
        self.eps_names = self.sigma_vector_list.random_variable_names()

    @property
    def omega(self):
        return get_covariance(self.omega_vector_list, self.omega_scaler_list, self.scale_mode)

    @property
    def sigma(self):
        return get_covariance(self.sigma_vector_list, self.sigma_scaler_list, self.scale_mode)

    @property
    def scale_mode(self):
        return self._scale_mode
    
    @scale_mode.setter
    @torch.no_grad()
    def scale_mode(self, value : bool):
        def turn_off(vector_list : CovarianceVectorList, scaler_list: CovarianceScalerList):
            new_vector_list = CovarianceVectorList()
            for vector, scaler in zip(vector_list, scaler_list):
                if scaler is not None :
                    new_vector_list.append(scaler(vector))
                else :
                    new_vector_list.append(vector)
            return new_vector_list
        def turn_on(vector_list : CovarianceVectorList, scaler_list: CovarianceScalerList):
            new_vector_list = CovarianceVectorList()
            new_scaler_list = CovarianceScalerList()
            for vector, scaler in zip(vector_list, scaler_list):
                if scaler is not None and type(vector) is CovarianceVector :
                    new_scaler = CovarianceScaler(vector)
                    new_scaler_list.append(new_scaler)
                    vector.data = tensor([0.1]*vector.size()[0])
                    new_vector_list.append(vector)
            return new_vector_list, new_scaler_list
        # turn on
        if not self.scale_mode and value :
            self.pred_function.theta_boundary_mode = value
            self.omega_vector_list, self.omega_scaler_list = turn_on(self.omega_vector_list, self.omega_scaler_list)
            self.sigma_vector_list, self.sigma_scaler_list = turn_on(self.sigma_vector_list, self.sigma_scaler_list)
            self._scale_mode = value
        # turn off
        elif self.scale_mode and not value :
            self.pred_function.theta_boundary_mode = value
            self.omega_vector_list = turn_off(self.omega_vector_list, self.omega_scaler_list)
            self.sigma_vector_list = turn_off(self.sigma_vector_list, self.sigma_scaler_list)
            self._scale_mode = value

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
            if hasattr(para, 'fixed') and not para.fixed:
                unfixed_parameter_values.append(para)
            else :
                unfixed_parameter_values.append(para)
        
        return unfixed_parameter_values
        
    def forward(self, dataset) :
        return self.pred_function(dataset)

    # TODO
    def _differential_pred(self, pred : Tensor, random_variable : List[Tensor]):

        random_variable_length = len(random_variable)

        differential_values = torch.zeros(pred.size()[0], random_variable_length, device = pred.device)

        for i_h, y_pred_elem in enumerate(pred) :
            for i_eps, cur_eps in enumerate(random_variable):
                if random_variable_length > 0 :
                    h_elem = torch.autograd.grad(y_pred_elem, cur_eps, create_graph=True, allow_unused=True, retain_graph=True)
                    differential_values[i_h,i_eps] = h_elem[0][i_h]


    def training_step(self, batch, batch_idx):
        datasets = self(batch)
        total_loss = 0
        for dataset in datasets :
            id = str(int(dataset[EssentialColumns.ID.value][0]))

            eta_dict = self.pred_function.get_attr_dict(EtaDict)
            eps_dict = self.pred_function.get_attr_dict(EpsDict)

            eta = []
            for eta_name in self.eta_names:
                eta.append(eta_dict[eta_name][id])

            eps = []
            for eps_name in self.eps_names:
                eps.append(eps_dict[eps_name][id])

            pred : Tensor = dataset[self.pred_function.PRED_COLUMN_NAME]

            eta_size = len(eta)
            eps_size = len(eps)

            h = torch.zeros(pred.size()[0], eps_size, device = pred.device)
            for i_h, y_pred_elem in enumerate(pred) :
                if eps_size > 0 :
                    for i_eps, cur_eps in enumerate(eps):
                        h_elem = torch.autograd.grad(y_pred_elem, cur_eps, create_graph=True, allow_unused=True, retain_graph=True)
                        h[i_h,i_eps] = h_elem[0][i_h]

            g = torch.zeros(pred.size()[0], eta_size, device = pred.device)
            for i_g, y_pred_elem in enumerate(pred) :
                if eta_size > 0 :
                    for i_eta, cur_eta in enumerate(eta) :
                        g_elem = torch.autograd.grad(y_pred_elem, cur_eta, create_graph=True, allow_unused=True, retain_graph=True)
                        g[i_g, i_eta] = g_elem[0]

            eta = torch.stack(eta)
            eps = torch.stack(eps)

            mdv_mask = dataset[EssentialColumns.MDV.value] == 0

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
            self.log('ofv', total_loss)
        return total_loss

    def configure_optimizers(self):
        # lr : float = self.hparams.lr  # type: ignore
        # tolerance_change : float = self.hparams.tolerance_change  # type: ignore
        # tolerance_grad : float = self.hparams.tolerance_grad  # type: ignore
        # max_iter : int = self.hparams.max_iter # type: ignore

        optimizer = torch.optim.LBFGS(
                self.get_unfixed_parameter_values(), 
                # lr=lr, 
                # tolerance_change= tolerance_change,
                # max_iter= max_iter,
                # tolerance_grad= tolerance_grad,
                line_search_fn='strong_wolfe')
        return [optimizer], []
    
    def count_number_of_parameters(self):
        count = 0
        count += len(self.pred_function.get_attr_dict(Theta).keys())
        count += len(self.eta_names)
        count += len(self.eps_names)
        for tensor in self.omega_vector_list : count += len(tensor)
        for tensor in self.sigma_vector_list : count += len(tensor)
        return count

    def predict(self, batch) -> Dict[str,OptimizationResult]:

        state = self.state_dict()
        # self.pred_function_module.reset_epss()

        result : Dict[str, OptimizationResult]= {}
        output = self(batch)

        for data in output:
            id = str(int(data[EssentialColumns.ID.value][0]))

            result[id] = OptimizationResult()
            result_cur_id = result[id]
            mdv_mask = data[EssentialColumns.MDV] == 0


            y_pred_masked = data[self.pred_function.PRED_COLUMN_NAME].masked_select(mdv_mask)
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
    
    # TODO remove
    # def get_AIC(self) :
    #     total_loss = tensor(0.).to(self.pred_function.dataset.device)
    #     result = self.evaluate()
    #     for _, values in result.items() :
    #             total_loss += values.loss
    #     k = self.count_number_of_parameters()
    #     return 2 * k + total_loss
   
    def covariance_step(self, batch) :
        theta_dict = self.pred_function.get_attr_dict(Theta)
        theta_names = []
        thetas = []
        for k, v in theta_dict.items() :
            theta_names.append(k)
            thetas.append(v)

        cov_mat_dim =  len(theta_names)
        for tensor in self.omega_vector_list :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma_vector_list :
            cov_mat_dim += tensor.size()[0]

        output = self(batch)
        

        requires_grad_memory = []
        estimated_parameters = [*thetas,
                        *self.omega_vector_list,
                        *self.sigma_vector_list]

        for para in estimated_parameters :
            requires_grad_memory.append(para.requires_grad)
            para.requires_grad = True
 
        r_mat = torch.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        s_mat = torch.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  
 
        for data in output:
            pred = data[self.pred_function.PRED_COLUMN_NAME]
            
            g = self._differential_pred(pred, etas)
            h = self._differential_pred(pred, epss)


            id = str(int(data[EssentialColumns.ID.value][0]))
            print('id', id)
 
            y_pred = y_pred.masked_select(mdv_mask)

            if eta.size()[-1] > 0 :
                g = g.t().masked_select(mdv_mask).reshape((eta.size()[-1],-1)).t()
            
            if eps.size()[0] > 0 :
                h = h.t().masked_select(mdv_mask).reshape((eps.size()[0],-1)).t()
 
            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)            

            gr = torch.autograd.grad(loss, estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)
            gr = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr]
            gr_cat = torch.concat(gr, dim=0)
            
            with torch.no_grad() :
                s_mat.add_((gr_cat.detach().unsqueeze(1) @ gr_cat.detach().unsqueeze(0))/4)
            
            for i, gr_cur in enumerate(gr_cat) :
                hs = torch.autograd.grad(gr_cur, estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)

                hs = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in hs]
                hs_cat = torch.cat(hs)
                for j, hs_elem in enumerate(hs_cat) :
                    r_mat[i,j] = r_mat[i,j] + hs_elem.detach()/2

        invR = r_mat.inverse()
        cov = invR @ s_mat @ invR
        se = cov.diag().sqrt()
        correl = covariance_to_correlation(cov)
        ei_values, ei_vectors = torch.linalg.eigh(correl)

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
        omega = self.omega
        sigma = self.sigma

        eta_parameter_values = self.pred_function.get_eta_parameter_values()
        eta_size = len(eta_parameter_values)
        mvn_eta = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(eta_size, device=dataset.device), omega)
        etas = mvn_eta.rsample(torch.Size((len(dataset), repeat)))
        

        eps_parameter_values = self.pred_function.get_eps_parameter_values()
        eps_size = len(eps_parameter_values)
        mvn_eps = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(eps_size, device=dataset.device), sigma)
        epss = mvn_eps.rsample(torch.Size([len(dataset), repeat, self.pred_function._max_record_length]))

        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        result : Dict[str, Dict[str, Union[Tensor, List[Tensor]]]] = {}
        for i, (data, _) in enumerate(dataloader):
            
            id = str(int(data[:, self.pred_function._column_names.index(EssentialColumnNames.ID.value)][0]))
            
            etas_cur = etas[i,:,:]
            epss_cur = epss[i,:,:]

            time_data = data[:,self.pred_function._column_names.index(EssentialColumnNames.TIME.value)].t()

            result[id] = {}
            result_cur_id : Dict[str, Union[Tensor, List[Tensor]]] = result[id]
            result_cur_id['time'] = time_data
            result_cur_id['etas'] = etas_cur
            result_cur_id['epss'] = epss_cur
            result_cur_id['preds'] = []
            for repeat_iter in range(repeat) :

                with torch.no_grad() :
                    eta_value = etas_cur[repeat_iter]
                    eps_value = epss_cur[repeat_iter]

                    for eta_i, name in enumerate(self.eta_names) :
                        eta_parameter_values[name].update({str(int(id)): Parameter(eta_value[eta_i])}) # type: ignore

                    for eps_i, name in enumerate(self.eps_names) :
                        eps_parameter_values[name].update({str(int(id)): Parameter(eps_value[:data.size()[0],eps_i])}) # type: ignore

                    r  = self.pred_function(data)
                    y_pred = r['y_pred']

                    result_cur_id['preds'].append(y_pred)
                    for name, value in r['output_columns'].items() :
                        if name not in result_cur_id.keys() :
                            result_cur_id[name] = []
                        result_cur_id[name].append(value) # type: ignore
        return result

class FOCEInterOptimalDesign(FOCEInter):
    # TODO
    def train_step(self, batch, batch_idx):
        theta_dict = self.pred_function.get_theta_parameter_values()
        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega_vector_list :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma_vector_list :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.pred_function.get_attr_dict(Theta)]

        fisher_information_matrix_total = torch.zeros(cov_mat_dim, cov_mat_dim, device = dataset.device)
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
                gr_theta_elem = torch.autograd.grad(y_elem, thetas, create_graph=True, allow_unused=True, retain_graph=True)
                gr_theta.append(torch.stack(gr_theta_elem))
            gr_theta = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr_theta]
            gr_theta = torch.stack(gr_theta, dim=0)

            #TODO sigma 개선?
            v = gr_theta @ omega @ gr_theta.t() + (h @ sigma @ h.t()).diag().diag()
            # v = gr_theta @ omega @ gr_theta.t() +  torch.eye(gr_theta.size()[0], device = dataset.device) * (sigma)
            v = v + torch.eye(v.size()[0], device = dataset.device) * 1e-6
            v_inv = v.inverse()

            a_matrix = gr_theta.t() @ v_inv @ gr_theta
            b_matrix = a_matrix * a_matrix
            v_sqaure_inv = v_inv @ v_inv
            c_vector = []
            for i in range(gr_theta.size()[-1]):
                c_vector.append(gr_theta[:,i].t() @ v_sqaure_inv @ gr_theta[:,i])
            c_vector = torch.stack(c_vector, dim=0).unsqueeze(0)

            b_matrix = torch.cat([b_matrix, c_vector])

            d_scalar = torch.trace(v_sqaure_inv).unsqueeze(0).unsqueeze(0)

            b_matrix = torch.cat([b_matrix, torch.cat([c_vector, d_scalar], dim=1).t()], dim=1)

            fisher_information_matrix = torch.block_diag(a_matrix, b_matrix/2)

            fisher_information_matrix_total = fisher_information_matrix_total + fisher_information_matrix
        return self.design_optimal_function(fisher_information_matrix_total)