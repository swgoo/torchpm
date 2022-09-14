
from dataclasses import dataclass, field
import time
from typing import Any, Callable, List, Dict, Optional
import typing
import torch.distributed as dist
from torch.utils.data import DataLoader

from .para import *
from .data import PMDataset
from . import predfunc
from . import lossfunc
from .data import EssentialColumns
from .misc import *

import pytorch_lightning as pl


@dataclass
class ModelConfig :
    dataset : PMDataset
    pred_function : Type[predfunc.PredictionFunction]
    omega : Union[CovarianceVectorInitList,Tuple[CovarianceVectorList,CovarianceScalerList]]
    sigma : Union[CovarianceVectorInitList,Tuple[CovarianceVectorList,CovarianceScalerList]]
    objective_function : lossfunc.NonLinearMixedModelObjectiveFunction = lossfunc.FOCEInterObjectiveFunction()


@dataclass
class OptimizationOutputs :
    outputs : Dict[int, Dict[str, Tensor]] = field(default_factory= lambda : {})
    loss : Dict[int, Tensor] = field(default_factory= lambda : {})
    cwres : Dict[int,Tensor] = field(default_factory= lambda : {})
    theta : Dict[str, Theta] = field(default_factory= lambda : {})
    eta_dict : Dict[str, EtaDict] = field(default_factory= lambda : {})
    eps_dict : Dict[str, EpsDict] = field(default_factory= lambda : {})
    omega : CovarianceVectorList = CovarianceVectorList()
    sigma : CovarianceVectorList = CovarianceVectorList()
    cov : Optional[Tensor] = None
    se : Optional[Tensor] = None
    correl : Optional[Tensor] = None
    ei_values : Optional[Tensor] = None
    inv_cov : Optional[Tensor] = None
    r_mat : Optional[Tensor] = None
    s_mat : Optional[Tensor] = None

    @property
    def total_loss(self):
        if self.loss is None :
            raise ValueError("loss attribute is None")

        total_loss = tensor(0.)
        for loss in self.loss.values() :
            total_loss += loss.to(total_loss.device)
        return total_loss
    
    @property
    def num_of_parameters(self):
        num = 0
        for para in [self.theta, self.eta_dict, self.eps_dict,]:
            num += len(para)

        for para in [self.omega, self.sigma]:
            for vector in para : num += len(vector)
        return num
    
    @property
    def num_of_population_parameters(self) :
        num = len(self.theta)
        for para in [self.omega, self.sigma]:
            for vector in para : num += len(vector)
        return num
    
    @property
    def aic(self):
        return self.total_loss + self.num_of_parameters

class FOCEInter(pl.LightningModule) :
   
    def __init__(self,
            model_config : ModelConfig,
            lr = 1.,
            tolerance_change : float = 1e-5,
            tolerance_grad : float = 1e-7,
            max_iter : int = 20,
            random_seed : int = 42):

        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])

        torch.manual_seed(self.hparams.random_seed) # type: ignore

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
            if model_config.objective_function is not None else lossfunc.FOCEInterObjectiveFunction()

        self.eta_names = self.omega_vector_list.random_variable_names()
        self.eps_names = self.sigma_vector_list.random_variable_names()

        self.covariance_step = True
        self.simulation_mode = False

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

    def get_unfixed_parameter_values(self) -> List[Parameter]:
        unfixed_parameter_values = []

        for omega_vector in self.omega_vector_list:
            if type(omega_vector) is CovarianceVector and not omega_vector.fixed :
                unfixed_parameter_values.append(omega_vector) 
        
        for sigma_vector in self.sigma_vector_list:
            if type(sigma_vector) is CovarianceVector and not sigma_vector.fixed :
                unfixed_parameter_values.append(sigma_vector) 
        
        pred_function_parameters = list(self.pred_function.parameters())

        for para in pred_function_parameters :
            if hasattr(para, 'fixed') and not getattr(para, 'fixed'):
                unfixed_parameter_values.append(para)
            elif hasattr(para, 'fixed') and getattr(para, 'fixed') :
                continue
            else :
                unfixed_parameter_values.append(para)
        
        return unfixed_parameter_values
        
    def forward(self, dataset) :
        return self.pred_function(dataset)

    def _common_step(self, batch, batch_idx) -> List[Dict[str, Tensor]]:
        datasets = self(batch)
        outputs = []
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

            pred_masked = pred.masked_select(mdv_mask)
            eta_size = g.size()[-1]
            
            if eta_size > 0 :
                g_masked = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
            else :
                g_masked = tensor([], device = pred.device)

            eps_size = h.size()[-1]
            if eps_size > 0:
                h_masked = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
            else :
                h_masked = tensor([], device=pred.device)

            dv : Tensor = dataset[EssentialColumns.DV.value]
            dv_masked = dv.masked_select(mdv_mask)
            loss_value = self.objective_function(dv_masked, pred_masked, g_masked, h_masked, eta, self.omega, self.sigma)
            dataset['G'] = g
            dataset['H'] = h
            dataset['LOSS'] = loss_value
            outputs.append(dataset)
        return outputs

    def training_step(self, batch, batch_idx):
        datasets = self._common_step(batch=batch, batch_idx=batch_idx)
        total_loss = 0
        for dataset in datasets :
            total_loss += dataset['LOSS']
            self.log('ofv', total_loss, reduce_fx=torch.sum)
        return total_loss

    def configure_optimizers(self):
        lr : float = self.hparams.lr  # type: ignore
        tolerance_change : float = self.hparams.tolerance_change  # type: ignore
        tolerance_grad : float = self.hparams.tolerance_grad  # type: ignore
        max_iter : int = self.hparams.max_iter # type: ignore

        optimizer = torch.optim.LBFGS(
                self.get_unfixed_parameter_values(), 
                lr=lr, 
                tolerance_change= tolerance_change,
                max_iter= max_iter,
                tolerance_grad= tolerance_grad,
                line_search_fn='strong_wolfe')
        return [optimizer], []

    @property
    def num_of_parameters(self):
        num = 0
        theta = self.pred_function.get_attr_dict(Theta)
        eta_dicts = self.pred_function.get_attr_dict(EtaDict)
        eps_dicts = self.pred_function.get_attr_dict(EpsDict)
        for para in [theta, eta_dicts, eps_dicts,]:
            num += len(para)

        for para in [self.omega_vector_list, self.sigma_vector_list]:
            for vector in para : num += len(vector)
        return num

    @property
    def num_of_population_parameters(self):
        theta = self.pred_function.get_attr_dict(Theta)
        num = len(theta)
        for para in [self.omega_vector_list, self.sigma_vector_list]:
            for vector in para : num += len(vector)
        return num


    def predict_step(self, batch : Dict[str, Tensor], batch_id) -> OptimizationOutputs :
        outputs = OptimizationOutputs()
        state_dict = self.state_dict()
        if self.simulation_mode :
            
            mvn_eta = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(len(outputs.eta_dict), device=batch[EssentialColumns.ID.value].device), self.omega)
            simulated_etas = mvn_eta.rsample(torch.Size([self.pred_function.dataset.len])).t()
            eta_names = self.omega_vector_list.random_variable_names()
            eta_dict = self.pred_function.get_attr_dict(EtaDict)
            for simulated_eta, name in zip(simulated_etas, eta_names) :
                for i, id in enumerate(self.pred_function.dataset.ids) :
                    eta_dict[name].update({str(id): simulated_eta[i]})

            mvn_eps = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(len(outputs.eps_dict), device=batch[EssentialColumns.ID.value].device), self.omega)
            simulated_epss = mvn_eps.rsample(torch.Size([self.pred_function.dataset.len, self.pred_function.dataset.max_record_length])).permute([2,0,1])
            eps_names = self.sigma_vector_list.random_variable_names()
            eps_dict = self.pred_function.get_attr_dict(EpsDict)
            for simulated_eps, name in zip(simulated_epss, eps_names):
                for i, id in enumerate(self.pred_function.dataset.ids):
                    eps_dict[name].update({str(id): simulated_eps[i]})

        inputs = self._common_step(batch = batch, batch_idx = batch_id)
        outputs.theta = self.pred_function.get_attr_dict(Theta)
        outputs.eta_dict = self.pred_function.get_attr_dict(EtaDict)
        outputs.eps_dict = self.pred_function.get_attr_dict(EpsDict)
        outputs.omega = self.omega_vector_list
        outputs.sigma = self.sigma_vector_list

        cov_mat_dim = self.num_of_population_parameters

        r_mat = torch.zeros(cov_mat_dim, cov_mat_dim, device=batch[EssentialColumns.ID.value].device)
        s_mat = torch.zeros(cov_mat_dim, cov_mat_dim, device=batch[EssentialColumns.ID.value].device)
        estimated_parameters = [
                *outputs.theta.values(),
                *self.omega_vector_list,
                *self.sigma_vector_list]

        for data in inputs:
            id = int(data[EssentialColumns.ID.value][0])

            mdv_mask = data[EssentialColumns.MDV.value] == 0

            y_true_masked = data[EssentialColumns.DV.value].masked_select(mdv_mask)
            y_pred_masked = data[self.pred_function.PRED_COLUMN_NAME].masked_select(mdv_mask)
            
            outputs.loss[id] = data['LOSS']

            if self.covariance_step :
                gr = torch.autograd.grad(data['LOSS'], estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)
                gr = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr]
                gr_cat = torch.concat(gr, dim=0)
            
                s_mat.add_((gr_cat.detach().unsqueeze(1) @ gr_cat.detach().unsqueeze(0))/4)
            
                for i, gr_cur in enumerate(gr_cat) :
                    hs = torch.autograd.grad(gr_cur, estimated_parameters, create_graph=True, retain_graph=True, allow_unused=True)

                    hs = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in hs]
                    hs_cat = torch.cat(hs)
                    for j, hs_elem in enumerate(hs_cat) :
                        r_mat[i,j] = r_mat[i,j] + hs_elem.detach()/2

            eta_dict = self.pred_function.get_attr_dict(EtaDict)
            eta = []
            for eta_name in self.eta_names:
                eta.append(eta_dict[eta_name][str(id)])
            eta = torch.stack(eta)
            
            outputs.cwres[id] = get_cwres(y_true_masked, y_pred_masked, data['G'], data['H'], eta, self.omega, self.sigma)
            outputs.outputs[id] = data
        
        if self.covariance_step :
            invR = r_mat.inverse()
            cov = invR @ s_mat @ invR
            se = cov.diag().sqrt()
            correl = covariance_to_correlation(cov)
            ei_values, ei_vectors = torch.linalg.eigh(correl)

            ei_values_sorted, _ = ei_values.sort()
            inv_cov = r_mat @ s_mat.inverse() @ r_mat

            outputs.cov = cov
            outputs.se = se
            outputs.correl = correl
            outputs.inv_cov = inv_cov
            outputs.ei_values = ei_values_sorted
            outputs.r_mat = r_mat
            outputs.s_mat = s_mat

        return outputs

class FOCEInterOptimalDesign(FOCEInter):
    # TODO
    def train_step(self, batch, batch_idx):
        cov_mat_dim =  self.num_of_population_parameters
        
        thetas = [value for _, value in self.pred_function.get_attr_dict(Theta).items()]

        fisher_information_matrix_total = torch.zeros(cov_mat_dim, cov_mat_dim, device = batch[EssentialColumns.ID.value].device)

        outputs = self._common_step(batch, batch_idx)


        for data in outputs:

            # y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data, partial_differentiate_by_etas = False, partial_differentiate_by_epss = True)

            mdv_mask = data[EssentialColumns.MDV.value] == 0
            y_pred_masked = data[self.pred_function.PRED_COLUMN_NAME].masked_select(mdv_mask)
            
            h = data['H']
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
            v = gr_theta @ self.omega @ gr_theta.t() + (h @ self.sigma @ h.t()).diag().diag()
            # v = gr_theta @ omega @ gr_theta.t() +  torch.eye(gr_theta.size()[0], device = dataset.device) * (sigma)
            v = v + torch.eye(v.size()[0], device = y_pred_masked.device) * 1e-6
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
        return self.objective_function(fisher_information_matrix_total)