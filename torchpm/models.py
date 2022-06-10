
from dataclasses import dataclass
import time
from typing import Callable, List, Dict, Optional
import typing
import torch as tc
import torch.distributed as dist
from torch.utils.data import DataLoader

from .parameter import *
from .data import CSVDataset
from . import predfunc
from . import loss
from .misc import *


@dataclass
class ModelConfig :
    pred_function : predfunc.PredictionFunction
    theta_names : List[str]
    eta_names : List[str]
    eps_names : List[str]
    omega : Omega
    sigma : Sigma
    objective_function : Optional[loss.ObjectiveFunction] = None
    optimal_design_creterion : Optional[loss.DesignOptimalFunction] = None

@dataclass
class OptimizationResult :
    loss : Optional[tc.Tensor] = None
    cwres_values : Optional[tc.Tensor] = None
    pred : Optional[tc.Tensor] = None
    time : Optional[tc.Tensor] = None
    mdv_mask : Optional[tc.Tensor] = None
    output_columns : Optional[Dict[str, tc.Tensor]] = None

class FOCEInter(tc.nn.Module) :

    def __init__(self,
            model_config : ModelConfig):

        super(FOCEInter, self).__init__()

        self.model_config = model_config
        self.pred_function = model_config.pred_function
        
        self.theta_names = model_config.theta_names
        self.eta_names = model_config.eta_names
        self.eps_names = model_config.eps_names
        self.omega = model_config.omega
        self.sigma = model_config.sigma
        self.objective_function = model_config.objective_function if model_config.objective_function is not None else loss.FOCEInterObjectiveFunction()
        self.design_optimal_function = model_config.optimal_design_creterion if model_config.optimal_design_creterion is not None else loss.DOptimality()
        self.dataloader = None
    
    def get_unfixed_parameter_values(self) -> List[nn.Parameter]:  # type: ignore
        unfixed_parameter_values = []

        omega_len = len(self.omega.parameter_values)
        for i, parameter_value, fixed in zip(range(omega_len), self.omega.parameter_values, self.omega.fixed) :
            if not fixed :
                unfixed_parameter_values.append(parameter_value) 
        
        sigma_len = len(self.sigma.parameter_values)
        for i, parameter_value, fixed in zip(range(sigma_len), self.sigma.parameter_values, self.sigma.fixed) :
            if not fixed :
                unfixed_parameter_values.append(parameter_value)
        
        for k, p in self.pred_function.get_thetas().items() :
            if not p.fixed :
                unfixed_parameter_values.append(p.parameter_value)
            
        for k, p in self.pred_function.get_etas().items() :
            unfixed_parameter_values.extend(list(p.parameter_values.values()))
        
        for k, p in self.pred_function.get_epss().items() :
            unfixed_parameter_values.extend(list(p.parameter_values.values()))
        
        return unfixed_parameter_values
        
    def forward(self, dataset, partial_differentiate_by_etas = True, partial_differentiate_by_epss = True) :
        
        pred_output = self.pred_function(dataset)

        etas = pred_output['etas']
        eta = []
        for eta_name in self.eta_names:
            eta.append(etas[eta_name]())

        epss = pred_output['epss']
        eps = []
        for eps_name in self.eps_names:
            eps.append(epss[eps_name]())

        y_pred, g, h = self._partial_differentiate(pred_output['y_pred'], eta, eps, by_etas = partial_differentiate_by_etas, by_epss = partial_differentiate_by_epss)

        eta = tc.stack(eta)
        eps = tc.stack(eps)

        return y_pred, eta, eps, g, h, self.omega().to(dataset.device), self.sigma().to(dataset.device), pred_output['mdv_mask'], pred_output['output_columns']
    
    def _partial_differentiate(self, y_pred, eta, eps, by_etas, by_epss) :
        eta_size = len(eta)
        eps_size = len(eps)

        if by_epss:
            h = tc.zeros(y_pred.size()[0], eps_size, device = y_pred.device)
            for i_h, y_pred_elem in enumerate(y_pred) :
                if eps_size > 0 :
                    for i_eps, cur_eps in enumerate(eps):
                        h_elem = tc.autograd.grad(y_pred_elem, cur_eps, create_graph=True, allow_unused=True, retain_graph=True)
                        h[i_h,i_eps] = h_elem[0][i_h]
        else : h = None

        if by_etas:
            g = tc.zeros(y_pred.size()[0], eta_size, device = y_pred.device)
            for i_g, y_pred_elem in enumerate(y_pred) :
                if eta_size > 0 :
                    for i_eta, cur_eta in enumerate(eta) :
                        g_elem = tc.autograd.grad(y_pred_elem, cur_eta, create_graph=True, allow_unused=True, retain_graph=True)
                        g[i_g, i_eta] = g_elem[0]
        else : g = None
        
        return y_pred, g, h

    def optimization_function_closure(self, dataset, optimizer, checkpoint_file_path : Optional[str] = None) -> Callable:
        """
        optimization function for L-BFGS 
        Args:
            dataset: model dataset
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()

        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  

        def fit() :
            optimizer.zero_grad()
            total_loss = tc.zeros([], device = dataset.device)
            
            for data, y_true in dataloader:
                y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data)
 
                y_pred = y_pred.masked_select(mdv_mask)
                eta_size = g.size()[-1]
                if eta_size > 0 :
                    g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
                eps_size = h.size()[-1]
                if eps_size > 0:
                    h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
 
                y_true_masked = y_true.masked_select(mdv_mask)
                loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
                loss.backward()
                
                total_loss = total_loss + loss
            
            if checkpoint_file_path is not None :
                tc.save(self.state_dict(), checkpoint_file_path)
        
            print('running_time : ', time.time() - start_time, '\t total_loss:', total_loss)
            return total_loss
        return fit
    
    def optimization_function_closure_FIM(self, dataset, optimizer, checkpoint_file_path : Optional[str] = None) -> Callable:
        """
        optimization function for L-BFGS 
        Args:
            dataset: model dataset
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()

        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  

        def fit() :
            optimizer.zero_grad()
            
            theta_dict = self.pred_function.get_theta_parameter_values()
            cov_mat_dim =  len(theta_dict)
            for tensor in self.omega.parameter_values :
                cov_mat_dim += tensor.size()[0]
            for tensor in self.sigma.parameter_values :
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
            
            
            loss = self.design_optimal_function(fisher_information_matrix_total)
            loss.backward()
            
            if checkpoint_file_path is not None :
                tc.save(self.state_dict(), checkpoint_file_path)
        
            print('running_time : ', time.time() - start_time, '\t total_loss:', loss)
            return loss
        return fit
    
    def optimization_function_FIM(self, optimizer, checkpoint_file_path : Optional[str] = None):
        """
        optimization function for L-BFGS 
        Args:
            dataset: model dataset
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()
        
        if self.dataloader is None :
            self.dataloader = DataLoader(self.pred_function.dataset, batch_size=None, shuffle=False, num_workers=0)  

        optimizer.zero_grad()
        
        theta_dict = self.pred_function.get_theta_parameter_values()
        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega.parameter_values :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma.parameter_values :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.theta_names]

        fisher_information_matrix_total = tc.zeros(cov_mat_dim, cov_mat_dim, device = self.pred_function.dataset.device)
        for data, y_true in self.dataloader:

            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data, partial_differentiate_by_etas = False, partial_differentiate_by_epss = True)

            y_pred_masked = y_pred.masked_select(mdv_mask)

            eps_size = h.size()[-1]
            if eps_size > 0:
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
            
            gr_theta = []
            for y_elem in y_pred_masked:
                gr_theta_elem = tc.autograd.grad(y_elem, thetas, create_graph=True, allow_unused=True, retain_graph=True)
                gr_theta.append(tc.stack(gr_theta_elem))
            gr_theta = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr_theta]
            gr_theta = tc.stack(gr_theta, dim=0)

            #TODO sigma 개선
            v = gr_theta @ omega @ gr_theta.t() + (h @ sigma @ h.t()).diag().diag()
            v = v + tc.eye(v.size()[0], device = v.device) * 1e-6
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
            
            
        loss = self.design_optimal_function(fisher_information_matrix_total)
        loss.backward()
        
        if checkpoint_file_path is not None :
            tc.save(self.state_dict(), checkpoint_file_path)
    
        print('running_time : ', time.time() - start_time, '\t total_loss:', loss)

        return loss
    
    def count_number_of_parameters(self):
        count = 0
        count += len(self.pred_function.get_thetas())
        count += len(self.pred_function.get_etas())
        count += len(self.pred_function.get_epss())
        for tensor in self.omega.parameter_values : count += len(tensor)
        for tensor in self.sigma.parameter_values : count += len(tensor)
        return count

    
    def optimization_function_for_multiprocessing(self, rank, dataset, optimizer, checkpoint_file_path : Optional[str] = None):
        """
        optimization function for L-BFGS multiprocessing
        Args:
            rank : multiprocessing thread number
            dataset: model dataset divided
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()

        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  
        def fit() :
            optimizer.zero_grad()
            total_loss = tc.zeros([], device = self.pred_function.dataset.device)
        
            for data, y_true in dataloader:
                y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data)
 
                y_pred = y_pred.masked_select(mdv_mask)
                eta_size = g.size()[-1]
                g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
                eps_size = h.size()[-1]
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
 
                y_true_masked = y_true.masked_select(mdv_mask)
                loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
                loss.backward()
                
                total_loss.add_(loss)
            
            with tc.no_grad() :
                for param in self.parameters():
                    grad_cur = param.grad
                    if grad_cur is None :
                        grad_cur = tc.zeros_like(param)
                        dist.all_reduce(grad_cur, op=dist.ReduceOp.SUM)
                        param.grad = grad_cur
                    else: 
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                if rank == 0 :
                    print('running_time : ', time.time() - start_time, '\t total_loss:', total_loss)
                if rank == 0 and checkpoint_file_path is not None :
                    tc.save(self.state_dict(), checkpoint_file_path)
            return total_loss
        return fit

    def evaluate_FIM(self) :
        dataloader = DataLoader(self.pred_function.dataset, batch_size=None, shuffle=False, num_workers=0)

        
        theta_dict = self.pred_function.get_theta_parameter_values()
        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega.parameter_values :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma.parameter_values :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.theta_names]
        
        result : Dict[str, Dict[str, Union[tc.Tensor, List[tc.Tensor]]]]= {}

        fisher_information_matrix_total = tc.zeros(cov_mat_dim, cov_mat_dim, device = self.pred_function.dataset.device)
        for data, y_true in dataloader:

            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data)

            id = str(int(data[:,self.pred_function._column_names.index('ID')][0]))
            result[id] = {}
            result_cur_id = result[id]

            y_pred_masked = y_pred.masked_select(mdv_mask)
            # eta_size = g.size()[-1]
            # if eta_size > 0 :
            #     g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
            eps_size = h.size()[-1]
            if eps_size > 0:
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
            
            gr_theta = []
            for y_elem in y_pred_masked:
                gr_theta_elem = tc.autograd.grad(y_elem, thetas, create_graph=True, allow_unused=True, retain_graph=True)
                gr_theta.append(tc.stack(gr_theta_elem))
            gr_theta = [grad.unsqueeze(0) if grad.dim() == 0 else grad for grad in gr_theta]
            gr_theta = tc.stack(gr_theta, dim=0)

            #TODO sigma 개선
            v = gr_theta @ omega @ gr_theta.t() + (h @ sigma @ h.t()).diag().diag()
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
            
            result_cur_id['pred'] = y_pred
            result_cur_id['time'] = data[:,self.pred_function._column_names.index('TIME')]
            result_cur_id['mdv_mask'] = mdv_mask

            for name, value in parameters.items() :
                if name not in result_cur_id.keys() :
                    result_cur_id[name] = []
                result_cur_id[name].append(value)  # type: ignore

        loss = self.design_optimal_function(fisher_information_matrix_total)
        return result, loss

    def evaluate(self) -> Dict[str,OptimizationResult]:

        dataloader = DataLoader(self.pred_function.dataset, batch_size=None, shuffle=False, num_workers=0)

        state = self.state_dict()
        # self.pred_function_module.reset_epss()

        result : Dict[str, OptimizationResult]= {}
        for data, y_true in dataloader:
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, parameters = self(data)
            id = str(int(data[:,self.pred_function._column_names.index('ID')][0]))

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
            result_cur_id.time = data[:,self.pred_function._column_names.index('TIME')].masked_select(mdv_mask)
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
        
    def descale(self) :
        self.pred_function.descale()
        self.omega.descale()
        self.sigma.descale()
        return self
    
    def scale(self) :
        self.pred_function.scale()
        self.omega.scale()
        self.sigma.scale()
        return self
    
    def parameters_for_individual(self) :
        parameters = []
        for k, p in self.pred_function.get_etas().items() :
            parameters.append(p)
        for k, p in self.pred_function.get_epss().items() :
            parameters.append(p)
        return parameters

    def fit_population(self, checkpoint_file_path : Optional[str] = None, learning_rate : float= 1, tolerance_grad = 1e-5, tolerance_change = 1e-7, max_iteration = 9999,):
        max_iter = max_iteration
        parameters = self.get_unfixed_parameter_values()
        self.pred_function.reset_epss()
        optimizer = tc.optim.LBFGS(
                parameters, 
                max_iter = max_iter, 
                lr = learning_rate, 
                tolerance_grad = tolerance_grad, 
                tolerance_change = tolerance_change,
                line_search_fn = 'strong_wolfe')
        opt_fn = self.optimization_function_closure(
                    self.pred_function.dataset, optimizer, checkpoint_file_path = checkpoint_file_path)
        optimizer.step(opt_fn)
        return self
    
    def fit_individual(self, checkpoint_file_path : Optional[str] = None, learning_rate = 1, tolerance_grad = 1e-7, tolerance_change = 1e-9, max_iteration = 9999,):
        max_iter = max_iteration
        parameters = self.parameters_for_individual()
        optimizer = tc.optim.LBFGS(parameters, 
                                   max_iter = max_iter, 
                                   lr = learning_rate, 
                                   tolerance_grad = tolerance_grad, 
                                   tolerance_change = tolerance_change,
                                   line_search_fn = 'strong_wolfe')
        opt_fn = self.optimization_function_closure(self.pred_function.dataset, optimizer, checkpoint_file_path = checkpoint_file_path)
        optimizer.step(opt_fn)

    def fit_population_FIM(self, parameters, checkpoint_file_path : Optional[str] = None, learning_rate : float= 0.6, tolerance_grad = 1e-7, tolerance_change = 1e-9, max_iteration = 9999,):
        max_iter = max_iteration
        self.pred_function.reset_epss()
        optimizer = tc.optim.LBFGS(parameters, 
                                   max_iter = max_iter, 
                                   lr = learning_rate, 
                                   tolerance_grad = tolerance_grad, 
                                   tolerance_change = tolerance_change,
                                   line_search_fn = 'strong_wolfe')
        opt_fn = self.optimization_function_closure_FIM(self.pred_function.dataset, optimizer, checkpoint_file_path = checkpoint_file_path)
        optimizer.step(opt_fn)
    
    def fit_population_FIM_by_adam(self, parameters, checkpoint_file_path : Optional[str] = None, learning_rate : float= 0.05, tolerance_change = 1e-3, max_iteration = 9999,):
        optimizer = tc.optim.Adam(parameters, lr = learning_rate)
        # scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        loss_prev = float("inf")
        loss_best = float("inf")
        for epoch in range(max_iteration) :
            loss = self.optimization_function_FIM(optimizer, checkpoint_file_path = checkpoint_file_path)    
            if (loss - loss_prev).abs() < tolerance_change and loss_best > loss :
                break
            else : loss_prev = loss
            if loss_best > loss :
                loss_best = loss 
            optimizer.step()
            # scheduler.step()           
        return self
   
    def covariance_step(self) :
        dataset = self.pred_function.dataset
        theta_dict = self.pred_function.get_theta_parameter_values()

        cov_mat_dim =  len(theta_dict)
        for tensor in self.omega.parameter_values :
            cov_mat_dim += tensor.size()[0]
        for tensor in self.sigma.parameter_values :
            cov_mat_dim += tensor.size()[0]
        
        thetas = [theta_dict[key] for key in self.theta_names]

        requires_grad_memory = []
        estimated_parameters = [*thetas,
                        *self.omega.parameter_values,
                        *self.sigma.parameter_values]

        for para in estimated_parameters :
            requires_grad_memory.append(para.requires_grad)
            para.requires_grad = True
 
        r_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        s_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)  
 
        for data, y_true in dataloader:
            
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask, _ = self(data)

            id = str(int(data[:,self.pred_function._column_names.index('ID')][0]))
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

    def simulate(self, dataset : CSVDataset, repeat : int) :
        """
        simulationg
        Args:
            dataset: model dataset for simulation
            repeat : simulation times
        """
        omega = self.omega()
        sigma = self.sigma()

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
            
            id = str(int(data[:, self.pred_function._column_names.index('ID')][0]))
            
            etas_cur = etas[i,:,:]
            epss_cur = epss[i,:,:]

            time_data = data[:,self.pred_function._column_names.index('TIME')].t()

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