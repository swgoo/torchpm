import time
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Dict, Iterable, Union
from sympy import false
import torch as tc

from torchdiffeq import odeint

from . import predfunction
from . import diff
from . import loss
from .misc import *

class FOCEInter(tc.nn.Module) :

    def __init__(self,
                 pred_function_module : predfunction.PredictionFunctionModule,
                differential_module : diff.DifferentialModule,
                objective_function : loss.ObjectiveFunction = loss.FOCEInterObjectiveFunction()):
        super(FOCEInter, self).__init__()
        self.pred_function_module = pred_function_module
        self.differential_module = differential_module
        self.objective_function = objective_function
        
    def forward(self, dataset, scaled=True):
        
        y_pred, eta, eps, mdv_mask = self.pred_function_module(dataset)

        y_pred, g, h, omega, sigma = self.differential_module(y_pred, eta, eps)

        return y_pred, eta, eps, g, h, omega, sigma, mdv_mask
 
    def optimization_function(self, dataset, optimizer, checkpoint_file_path : str = None):
        """
        optimization function for L-BFGS 
        Args:
            dataset: model dataset
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()

        dataloader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

        def fit() :
            optimizer.zero_grad()
            total_loss = tc.zeros([], device = dataset.device)
            
            for data, y_true in dataloader:
                y_pred, eta, eps, g, h, omega, sigma, mdv_mask = self(data)
 
                y_pred = y_pred.masked_select(mdv_mask)
                eta_size = g.size()[-1]
                g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
                eps_size = h.size()[-1]
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
 
                y_true_masked = y_true.masked_select(mdv_mask)
                loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
                loss.backward()
                
                # with tc.no_grad() :
                total_loss.add_(loss)
            
            if checkpoint_file_path is not None :
                tc.save(self.state_dict(), checkpoint_file_path)
        
            print('running_time : ', time.time() - start_time, '\t total_loss:', total_loss)
            return total_loss
        return fit
    
    def optimization_function_for_multiprocessing(self, rank, dataset, optimizer, checkpoint_file_path : str = None):
        """
        optimization function for L-BFGS multiprocessing
        Args:
            rank : multiprocessing thread number
            dataset: model dataset divided
            optimizer: L-BFGS optimizer
            checkpoint_file_path : saving for optimized parameters
        """
        start_time = time.time()

        dataloader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        def fit() :
            optimizer.zero_grad()
            # with tc.no_grad() :
            total_loss = tc.zeros([], device = self.pred_function_module.dataset.device)
        
            for data, y_true in dataloader:
                y_pred, eta, eps, g, h, omega, sigma, mdv_mask = self(data)
 
                y_pred = y_pred.masked_select(mdv_mask)
                eta_size = g.size()[-1]
                g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
                eps_size = h.size()[-1]
                h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()
 
                y_true_masked = y_true.masked_select(mdv_mask)
                loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)
                loss.backward()
                
                # with tc.no_grad() :
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

    def evaluate(self):

        dataloader = tc.utils.data.DataLoader(self.pred_function_module.dataset, batch_size=None, shuffle=False, num_workers=0)

        state = self.state_dict()
        
        with tc.no_grad() :
            for k, p in self.pred_function_module.epss.items() :
                p.data = tc.zeros(p.size(), device=p.device)
            total_loss = tc.zeros([], device = self.pred_function_module.dataset.device)

        # datasets = []
        losses : Dict[str, tc.Tensor] = {}
        times : Dict[str, tc.Tensor] = {}
        preds : Dict[str, tc.Tensor] = {} 
        cwress : Dict[str, tc.Tensor] = {}
        mdv_masks : Dict[str, tc.Tensor] = {}
        for data, y_true in dataloader:
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask = self(data)
            id = str(int(data[:,self.pred_function_module.column_names.index('ID')][0]))

            y_pred_masked = y_pred.masked_select(mdv_mask)
            eta_size = g.size()[-1]
            g = g.t().masked_select(mdv_mask).reshape((eta_size,-1)).t()
            eps_size = h.size()[-1]
            h = h.t().masked_select(mdv_mask).reshape((eps_size,-1)).t()

            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred_masked, g, h, eta, omega, sigma)
            
            cwress[id] = cwres(y_true_masked, y_pred_masked, g, h, eta, omega, sigma)
            preds[id] = y_pred
            losses[id] = float(loss)
            times[id] = data[:,self.pred_function_module.column_names.index('TIME')]
            mdv_masks[id] = mdv_mask
            
            with tc.no_grad() :
                total_loss.add_(loss)
            
        self.load_state_dict(state, strict=False)
        
        return {'total_loss': total_loss, 
                'losses': losses, 
                'times': times, 
                'preds': preds, 
                'cwress': cwress,
                'mdv_masks': mdv_masks}
    
    def descale(self) :
        self.pred_function_module.descale()
        self.differential_module.descale()
        return self
    
    #TODO
    def get_descaled_parameters(self):
        pass

    def parameters_for_population(self):
        parameters = []
        for m in self.differential_module.parameters() :
            parameters.append(m)
        
        for m in self.pred_function_module.parameters() :
            parameters.append(m)

        return parameters
    
    def parameters_for_individual(self) :
        parameters = []

        for k, p in self.pred_function_module.etas.items() :
            parameters.append(p)
        
        for k, p in self.pred_function_module.epss.items() :
            parameters.append(p)

        # for m in self.pred_function_module.parameters() :
        #     parameters.append(m)
        
        return parameters

    def fit_population(self, checkpoint_file_path : str = None, learning_rate = 1, tolerance_grad = 1e-2, tolerance_change = 1e-2, max_iteration = 1000,):
        
        max_iter = max_iteration

        parameters = self.parameters_for_population()

        epss = self.pred_function_module.epss
        
        with tc.no_grad() :
            for k, p in self.pred_function_module.epss.items() :
                p.data = tc.zeros(p.size(), device=p.device)
            
        optimizer = tc.optim.LBFGS(parameters, 
                                   max_iter = max_iter, 
                                   lr = learning_rate, 
                                   tolerance_grad = tolerance_grad, 
                                   tolerance_change = tolerance_change)
        
        opt_fn = self.optimization_function(self.pred_function_module.dataset, optimizer, checkpoint_file_path = checkpoint_file_path)

        optimizer.step(opt_fn)

        self.pred_function_module.epss = epss
    
    def fit_individual(self, checkpoint_file_path : str = None, learning_rate = 1, tolerance_grad = 1e-2, tolerance_change = 3e-2, max_iteration = 1000,):

        max_iter = max_iteration
        parameters = self.parameters_for_individual()

        optimizer = tc.optim.LBFGS(parameters, 
                                   max_iter = max_iter, 
                                   lr = learning_rate, 
                                   tolerance_grad = tolerance_grad, 
                                   tolerance_change = tolerance_change)
        opt_fn = self.optimization_function(self.pred_function_module.dataset, optimizer, checkpoint_file_path = checkpoint_file_path)

        optimizer.step(opt_fn)
    
    def covariance_step(self) :

        dataset = self.pred_function_module.dataset
 
        cov_mat_dim =  self.pred_function_module.theta.size()[0]

        for tensor in self.differential_module.omega :
            cov_mat_dim += tensor.size()[0]

        for tensor in self.differential_module.sigma :
            cov_mat_dim += tensor.size()[0]
 
        r_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)
 
        s_mat = tc.zeros(cov_mat_dim, cov_mat_dim, device=dataset.device)

        dataloader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
 
        for data, y_true in dataloader:
            
            y_pred, eta, eps, g, h, omega, sigma, mdv_mask = self(data)

            id = str(int(data[:,self.pred_function_module.column_names.index('ID')][0]))
            print('id', id)
 
            y_pred = y_pred.masked_select(mdv_mask)
            g = g.t().masked_select(mdv_mask).reshape((self.pred_function_module.eta_size,-1)).t()
            h = h.t().masked_select(mdv_mask).reshape((self.pred_function_module.eps_size,-1)).t()
 
            y_true_masked = y_true.masked_select(mdv_mask)
            loss = self.objective_function(y_true_masked, y_pred, g, h, eta, omega, sigma)

            # TODO: descale없이 작동시키기
            # matrix_parameters = self.differential_module.get_descaled_parameters()
            # parameters = [self.pred_function_module.get_descaled_theta(), *matrix_parameters['omega'], *matrix_parameters['sigma']]
            
            parameters = [self.pred_function_module.theta, *self.differential_module.omega, *self.differential_module.sigma]

            gr = tc.autograd.grad(loss, parameters, create_graph=True, retain_graph=True, allow_unused=True)
            gr_cat = tc.cat(gr)
            
            with tc.no_grad() :
                s_mat.add_((gr_cat.detach().unsqueeze(1) @ gr_cat.detach().unsqueeze(0))/4)
            
            for i, gr_cur in enumerate(gr_cat) :
                hs = tc.autograd.grad(gr_cur, parameters, create_graph=True, retain_graph=True, allow_unused=True)

                hs_cat = tc.cat(hs)
                for j, hs_elem in enumerate(hs_cat) :
                    r_mat[i,j] = r_mat[i,j] + hs_elem.detach()/2

        invR = r_mat.inverse()
        
        cov = invR @ s_mat @ invR
        
        se = cov.diag().sqrt()
        
        
        correl = covariance_to_correlation(cov)
        
        # ei_values, ei_vectors = correl.symeig(eigenvectors=False)

        ei_values, ei_vectors = tc.linalg.eigh(correl)

        
        ei_values_sorted, _ = ei_values.sort()
        inv_cov = r_mat @ s_mat.inverse() @ r_mat
        
        return {'cov': cov, 'se': se, 'cor': correl, 'ei_values': ei_values_sorted , 'inv_cov': inv_cov, 'r_mat': r_mat, 's_mat':s_mat}
        # return {'cov': cov, 'se': se, 'cor': correl, 'inv_cov': inv_cov, 'r_mat': r_mat, 's_mat':s_mat}